use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadContext,
    WhisperVadContextParams, WhisperVadParams,
};

pub mod diarize;
pub mod download;
pub mod recorder;
pub mod summarize;

pub const WHISPER_SAMPLE_RATE: usize = 16_000;

/// Which inference engine a speech model runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// ggml Whisper models (`ggml-*.bin`) via whisper.cpp.
    Whisper,
    /// GGUF models (NVIDIA Parakeet, Qwen3-ASR) via transcribe.cpp.
    TranscribeCpp,
}

impl Engine {
    /// Picked from the model file: whisper.cpp's models are `.bin`, the
    /// transcribe.cpp ones are `.gguf`.
    pub fn for_model(model: &Path) -> Engine {
        match model.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("gguf") => Engine::TranscribeCpp,
            _ => Engine::Whisper,
        }
    }
}

/// One decoded segment: start and end in centiseconds, and its text.
type RawSegment = (i64, i64, String);

/// Nudges whisper toward punctuated, capitalized output (it otherwise tends
/// to lock into an unpunctuated style when a file starts mid-sentence).
pub const DEFAULT_PROMPT: &str = "Hello. Okay, let's get started, shall we?";

/// Locate the models/ directory: in the current working directory (running
/// from the repo root), walking up from the executable (running from
/// target/release/ or a .app bundle inside the repo), or the per-user
/// fallback used by released builds.
pub fn find_models_dir() -> Option<PathBuf> {
    let local = PathBuf::from("models");
    if local.is_dir() {
        return Some(local);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(hit) = exe
            .ancestors()
            .skip(1)
            .map(|dir| dir.join("models"))
            .find(|dir| dir.is_dir())
    {
        return Some(hit);
    }
    let fallback = default_models_dir()?;
    fallback.is_dir().then_some(fallback)
}

/// Per-user models directory (`~/.transcribe/models`) — where model
/// downloads go when the app runs outside the repo, e.g. a released
/// bundle dragged to /Applications.
pub fn default_models_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".transcribe").join("models"))
}

/// Markdown file with domain vocabulary (names, jargon) folded into the
/// whisper prompt; lives next to the models/ directory.
pub const VOCABULARY_FILE: &str = "vocabulary.md";

/// Locate the vocabulary file the same way models are found: current
/// working directory first, then upward from the executable. When the
/// file doesn't exist yet, this is where it should be created.
pub fn vocabulary_path() -> PathBuf {
    let local = PathBuf::from(VOCABULARY_FILE);
    if local.exists() {
        return local;
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(hit) = exe
            .ancestors()
            .skip(1)
            .map(|dir| dir.join(VOCABULARY_FILE))
            .find(|path| path.exists())
    {
        return hit;
    }
    match find_models_dir().and_then(|m| m.parent().map(Path::to_path_buf)) {
        Some(dir) => dir.join(VOCABULARY_FILE),
        None => local,
    }
}

/// Build a whisper initial prompt from a markdown vocabulary: every
/// non-heading line becomes a glossary term that biases recognition
/// toward those words.
pub fn prompt_with_vocabulary(vocabulary_md: &str) -> String {
    build_prompt(vocabulary_md, "")
}

/// The terms in a markdown vocabulary: every line that isn't a heading or
/// comment, with list markers and backticks stripped.
pub fn vocabulary_terms(vocabulary_md: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    for line in vocabulary_md.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("<!--") {
            continue;
        }
        let line = line.trim_start_matches(['-', '*', '+']).trim_start();
        let line = line
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .trim_start_matches('.')
            .trim_start();
        let line = line.replace('`', "");
        if !line.is_empty() {
            terms.push(line);
        }
    }
    terms
}

/// Build a whisper initial prompt from the vocabulary glossary plus
/// free-form context about the recording (meeting name, speakers, topics).
/// Ends with [`DEFAULT_PROMPT`] — whisper keeps the tail of an over-long
/// prompt, so the punctuation nudge survives even when a huge glossary
/// gets truncated.
pub fn build_prompt(vocabulary_md: &str, context: &str) -> String {
    let terms = vocabulary_terms(vocabulary_md);

    let mut parts: Vec<String> = Vec::new();
    if !terms.is_empty() {
        parts.push(format!("Glossary: {}.", terms.join(", ")));
    }
    // Collapse the context into one line; whisper treats the prompt as
    // preceding transcript text, not as instructions.
    let context = context.split_whitespace().collect::<Vec<_>>().join(" ");
    if !context.is_empty() {
        let punctuated = context.ends_with(['.', '!', '?']);
        parts.push(format!("Context: {context}{}", if punctuated { "" } else { "." }));
    }
    parts.push(DEFAULT_PROMPT.into());
    parts.join(" ")
}

/// Everything the pipeline needs besides the audio itself.
pub struct Options {
    pub model: PathBuf,
    pub language: String,
    pub beam_size: Option<i32>,
    pub translate: bool,
    pub threads: Option<i32>,
    pub prompt: String,
    pub timestamps: bool,
    pub diarize: Option<DiarizeModels>,
    /// Enrolled speaker names. When these appear in the prompt (via the
    /// context notes), whisper sometimes mimics a script format and
    /// prefixes its own output with "Name:" — such prefixes are stripped
    /// from the transcription text.
    pub known_speakers: Vec<String>,
    /// Text the prompt was built from (context notes and vocabulary).
    /// Names mentioned here are treated like [`Options::known_speakers`]
    /// for prefix stripping, since they're what whisper mimics.
    pub context: String,
    /// Silero VAD model for whisper.cpp (see [`download::VAD_MODEL_FILE`]).
    /// When set, silence and noise are skipped before decoding, which
    /// stops whisper hallucinating text for them; timestamps still refer
    /// to the original audio. None decodes everything.
    pub vad_model: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: PathBuf::from("models/ggml-large-v3-turbo.bin"),
            language: "auto".into(),
            beam_size: None,
            translate: false,
            threads: None,
            prompt: DEFAULT_PROMPT.into(),
            timestamps: false,
            diarize: None,
            known_speakers: Vec::new(),
            context: String::new(),
            vad_model: None,
        }
    }
}

pub struct DiarizeModels {
    pub segmentation_model: PathBuf,
    pub embedding_model: PathBuf,
    pub max_speakers: usize,
    pub threshold: f32,
    /// Enrolled voices; diarized speakers matching one are named in the
    /// transcript instead of being numbered.
    pub profiles: Vec<diarize::SpeakerProfile>,
}

impl Default for DiarizeModels {
    fn default() -> Self {
        Self {
            segmentation_model: PathBuf::from("models/segmentation-3.0.onnx"),
            embedding_model: PathBuf::from("models/wespeaker_en_voxceleb_CAM++.onnx"),
            max_speakers: 8,
            threshold: 0.5,
            profiles: Vec::new(),
        }
    }
}

/// File holding the enrolled speaker profiles, next to vocabulary.md.
pub const SPEAKERS_FILE: &str = "speakers.json";

/// Locate the speaker-profiles file the same way as [`vocabulary_path`].
pub fn speakers_path() -> PathBuf {
    let local = PathBuf::from(SPEAKERS_FILE);
    if local.exists() {
        return local;
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(hit) = exe
            .ancestors()
            .skip(1)
            .map(|dir| dir.join(SPEAKERS_FILE))
            .find(|path| path.exists())
    {
        return hit;
    }
    match find_models_dir().and_then(|m| m.parent().map(Path::to_path_buf)) {
        Some(dir) => dir.join(SPEAKERS_FILE),
        None => local,
    }
}

/// Load enrolled speaker profiles; a missing file is an empty list.
pub fn load_speaker_profiles(path: &Path) -> Result<Vec<diarize::SpeakerProfile>> {
    match std::fs::read_to_string(path) {
        Ok(json) => serde_json::from_str(&json)
            .with_context(|| format!("invalid speaker profiles in {}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e).with_context(|| format!("failed to read {}", path.display())),
    }
}

pub fn save_speaker_profiles(path: &Path, profiles: &[diarize::SpeakerProfile]) -> Result<()> {
    let json = serde_json::to_string_pretty(profiles)?;
    std::fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

/// Pipeline milestones, reported through the callback passed to [`transcribe`].
pub enum Progress {
    Decoded { audio_secs: f64, took_secs: f64 },
    DetectingSpeakers,
    Diarized { segments: usize, took_secs: f64 },
    Transcribing { percent: i32 },
    /// A segment whisper just finalized, streamed as decoding progresses.
    /// Raw text — the finished [`Transcript`] is the authoritative version
    /// (speaker labels, prefix stripping, loop collapsing).
    Segment { text: String },
    Transcribed { took_secs: f64, realtime_factor: f64 },
}

pub struct Transcript {
    pub text: String,
    /// Distinct speakers appearing in the text (None without diarization).
    pub speakers: Option<usize>,
    /// Enrolled names applied to a voice, with the match similarity.
    pub speaker_matches: Vec<(String, f32)>,
    /// Enrolled names that resembled a voice but not confidently enough
    /// to apply — those speakers stay numbered. Usually a sign to
    /// re-enroll that person from audio recorded like the meeting.
    pub weak_matches: Vec<(String, f32)>,
    /// Voice fingerprint per transcript label ("Speaker 1", or an
    /// enrolled name) — lets a label be renamed afterwards and the voice
    /// enrolled under the new name.
    pub speaker_voices: Vec<SpeakerVoice>,
}

/// A speaker as they appear in a finished transcript: the label used in
/// the text plus that voice's embedding from this recording.
pub struct SpeakerVoice {
    pub label: String,
    pub embedding: Vec<f32>,
}

/// Replace a speaker label at the start of transcript lines — also after
/// a `[.. --> ..]` timestamp prefix: "Speaker 1: " becomes "John Doe: ".
pub fn rename_speaker(transcript: &str, old: &str, new: &str) -> String {
    let old_prefix = format!("{old}: ");
    let mut out: Vec<String> = Vec::new();
    for line in transcript.lines() {
        let head_len = if line.starts_with('[') {
            line.find("]  ").map_or(0, |i| i + 3)
        } else {
            0
        };
        let (head, rest) = line.split_at(head_len);
        match rest.strip_prefix(&old_prefix) {
            Some(tail) => out.push(format!("{head}{new}: {tail}")),
            None => out.push(line.to_owned()),
        }
    }
    let mut text = out.join("\n");
    if transcript.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// Run the full pipeline on one audio file: decode, optionally diarize,
/// transcribe with whisper, and format the transcript.
pub fn transcribe<F>(audio: &Path, opts: &Options, progress: F) -> Result<Transcript>
where
    F: Fn(Progress) + Send + Sync + 'static,
{
    let t0 = Instant::now();
    let samples = decode_to_mono_16k(audio)?;
    progress(Progress::Decoded {
        audio_secs: samples.len() as f64 / WHISPER_SAMPLE_RATE as f64,
        took_secs: t0.elapsed().as_secs_f64(),
    });
    transcribe_samples(samples, opts, progress)
}

/// Like [`transcribe`], but on already-decoded 16 kHz mono PCM
/// (e.g. from [`recorder::Recorder`]).
pub fn transcribe_samples<F>(mut samples: Vec<f32>, opts: &Options, progress: F) -> Result<Transcript>
where
    F: Fn(Progress) + Send + Sync + 'static,
{
    // whisper.cpp rejects inputs shorter than ~1s; pad with silence.
    let min_len = WHISPER_SAMPLE_RATE + WHISPER_SAMPLE_RATE / 10;
    if samples.len() < min_len {
        samples.resize(min_len, 0.0);
    }
    let diarization = match &opts.diarize {
        Some(models) => {
            let t = Instant::now();
            progress(Progress::DetectingSpeakers);
            let diarization = diarize::diarize(
                &samples,
                WHISPER_SAMPLE_RATE as u32,
                &diarize::DiarizeOptions {
                    segmentation_model: &models.segmentation_model,
                    embedding_model: &models.embedding_model,
                    max_speakers: models.max_speakers,
                    threshold: models.threshold,
                    profiles: &models.profiles,
                },
            )?;
            progress(Progress::Diarized {
                segments: diarization.segments.len(),
                took_secs: t.elapsed().as_secs_f64(),
            });
            Some(diarization)
        }
        None => None,
    };

    let progress = Arc::new(progress);
    let raw_segments = match Engine::for_model(&opts.model) {
        Engine::Whisper => run_whisper(&samples, opts, &progress)?,
        Engine::TranscribeCpp => run_transcribe_cpp(&samples, opts, &progress)?,
    };

    // First pass: count repeated name-like leading prefixes — the
    // signature of whisper mimicking a "Name: text" format.
    let mut prefix_counts: HashMap<String, usize> = HashMap::new();
    for (_, _, segment_text) in &raw_segments {
        if let Some(prefix) = leading_prefix(segment_text)
            && name_like(prefix)
        {
            *prefix_counts.entry(prefix.to_owned()).or_default() += 1;
        }
    }
    let frequent: std::collections::HashSet<String> = prefix_counts
        .into_iter()
        .filter(|&(_, count)| count >= 3)
        .map(|(prefix, _)| prefix)
        .collect();

    let mut text = String::new();
    // Enrolled speakers keep their name; the rest are numbered so that
    // "Speaker 1" is the first unrecognized voice to appear, and so on.
    let mut speaker_numbers: HashMap<usize, usize> = HashMap::new();
    let mut speakers_seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let context_lower = opts.context.to_lowercase();
    let mut previous_line = String::new();
    let mut repeats = 0usize;
    for &(segment_start, segment_end, ref raw_text) in &raw_segments {
        let segment_text =
            strip_hallucinated_prefix(raw_text, &opts.known_speakers, &context_lower, &frequent);
        if segment_text.is_empty() {
            continue;
        }
        // Whisper occasionally gets stuck emitting the same line over and
        // over (a decoding loop); keep at most two consecutive copies.
        if segment_text == previous_line {
            repeats += 1;
            if repeats > 2 {
                continue;
            }
        } else {
            repeats = 1;
            previous_line = segment_text.to_owned();
        }
        let speaker = diarization.as_ref().map(|diarization| {
            // segment timestamps are centiseconds, diarization uses seconds
            let start = segment_start as f64 / 100.0;
            let end = segment_end as f64 / 100.0;
            match diarize::speaker_for_range(&diarization.segments, start, end) {
                Some(id) => {
                    speakers_seen.insert(id);
                    match diarization.names.get(&id) {
                        Some((name, _)) => format!("{name}: "),
                        None => {
                            let next = speaker_numbers.len() + 1;
                            let number = *speaker_numbers.entry(id).or_insert(next);
                            format!("Speaker {number}: ")
                        }
                    }
                }
                None => "Speaker ?: ".to_string(),
            }
        });
        let speaker = speaker.as_deref().unwrap_or("");
        if opts.timestamps {
            text.push_str(&format!(
                "[{} --> {}]  {speaker}{segment_text}\n",
                format_timestamp(segment_start),
                format_timestamp(segment_end),
            ));
        } else {
            text.push_str(speaker);
            text.push_str(segment_text);
            text.push('\n');
        }
    }
    finish_transcript(text, opts, diarization, speakers_seen, speaker_numbers)
}

/// CPU threads for decoding: what the user asked for, else up to 8 cores.
fn thread_count(opts: &Options) -> i32 {
    opts.threads.unwrap_or_else(|| {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        cores.min(8) as i32
    })
}

/// Decode with whisper.cpp; returns the segments it produced.
fn run_whisper<F>(samples: &[f32], opts: &Options, progress: &Arc<F>) -> Result<Vec<RawSegment>>
where
    F: Fn(Progress) + Send + Sync + 'static,
{
    let audio_secs = samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;
    // Route whisper.cpp/GGML logs away from stderr (no log backend => silenced).
    whisper_rs::install_logging_hooks();

    let mut ctx_params = WhisperContextParameters::default();
    ctx_params.use_gpu(true);
    let ctx = WhisperContext::new_with_params(&opts.model, ctx_params)
        .with_context(|| format!("failed to load model {}", opts.model.display()))?;

    let strategy = match opts.beam_size {
        Some(n) if n > 1 => SamplingStrategy::BeamSearch {
            beam_size: n,
            patience: -1.0,
        },
        _ => SamplingStrategy::Greedy { best_of: 5 },
    };

    let threads = thread_count(opts);
    let mut params = FullParams::new(strategy);
    params.set_n_threads(threads);
    params.set_language(Some(&opts.language));
    params.set_initial_prompt(&opts.prompt);
    params.set_translate(opts.translate);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    if let Some(vad_model) = &opts.vad_model {
        let path = vad_model.to_str().context("VAD model path is not valid UTF-8")?;
        params.set_vad_model_path(Some(path));
        params.enable_vad(true);
    }
    let on_percent = progress.clone();
    params.set_progress_callback_safe(move |percent: i32| {
        on_percent(Progress::Transcribing { percent });
    });
    let on_segment = progress.clone();
    params.set_segment_callback_safe_lossy(move |segment: whisper_rs::SegmentCallbackData| {
        let text = segment.text.trim();
        if !text.is_empty() {
            on_segment(Progress::Segment { text: text.to_owned() });
        }
    });

    let t1 = Instant::now();
    let mut state = ctx.create_state()?;
    state.full(params, samples)?;
    let transcribe_secs = t1.elapsed().as_secs_f64();
    progress(Progress::Transcribed {
        took_secs: transcribe_secs,
        realtime_factor: audio_secs / transcribe_secs,
    });

    let mut raw_segments: Vec<RawSegment> = Vec::new();
    for segment in state.as_iter() {
        let segment_text = segment.to_str_lossy()?;
        raw_segments.push((
            segment.start_timestamp(),
            segment.end_timestamp(),
            segment_text.trim().to_owned(),
        ));
    }
    Ok(raw_segments)
}

/// Decode with transcribe.cpp. These models have no prompt, so
/// vocabulary/context are ignored. Audio is cut at silences found by the
/// VAD model, which bounds memory on long recordings and gives progress
/// updates. Models with word alignment (Parakeet) get chunks of a few
/// minutes and their word timestamps are regrouped into segments at
/// pauses and sentence ends; models without any timestamps (Qwen3-ASR)
/// decode each stretch of speech on its own, timed by the VAD boundaries.
fn run_transcribe_cpp<F>(
    samples: &[f32],
    opts: &Options,
    progress: &Arc<F>,
) -> Result<Vec<RawSegment>>
where
    F: Fn(Progress) + Send + Sync + 'static,
{
    use transcribe_cpp::{Model, RunOptions, SessionOptions, TimestampKind};

    if opts.translate {
        bail!("translation needs a Whisper model; this one only transcribes");
    }
    if opts.beam_size.is_some_and(|n| n > 1) {
        bail!("beam search is a Whisper option; this model decodes greedily");
    }
    static QUIET: std::sync::Once = std::sync::Once::new();
    QUIET.call_once(transcribe_cpp::disable_logging);

    let audio_secs = samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;
    let threads = thread_count(opts);
    let model = Model::load(&opts.model)
        .with_context(|| format!("failed to load model {}", opts.model.display()))?;
    let word_timed = matches!(
        model.capabilities().max_timestamp_kind,
        TimestampKind::Word | TimestampKind::Token
    );
    let mut session =
        model.session_with(&SessionOptions { n_threads: threads, ..SessionOptions::default() })?;
    let run = RunOptions {
        timestamps: if word_timed { TimestampKind::Word } else { TimestampKind::None },
        language: (opts.language != "auto").then(|| opts.language.clone()),
        ..RunOptions::default()
    };
    let to_ms = |sample: usize| (sample as i64) * 1000 / WHISPER_SAMPLE_RATE as i64;
    let secs = |s: f32| (s * WHISPER_SAMPLE_RATE as f32) as usize;

    let spans = speech_spans(samples, opts.vad_model.as_deref(), threads);
    let chunks = if word_timed {
        merge_spans(&spans, secs(ALIGNED_CHUNK_SECS), usize::MAX)
    } else {
        merge_spans(&spans, secs(UTTERANCE_MAX_SECS), secs(UTTERANCE_GAP_SECS))
    };

    let t1 = Instant::now();
    // (start ms, end ms, text): words for aligned models, else one entry
    // per decoded stretch of speech.
    let mut pieces: Vec<(i64, i64, String)> = Vec::new();
    for &(start, end) in &chunks {
        progress(Progress::Transcribing {
            percent: (start * 100 / samples.len().max(1)) as i32,
        });
        let mut chunk = samples[start..end].to_vec();
        if chunk.len() < WHISPER_SAMPLE_RATE {
            chunk.resize(WHISPER_SAMPLE_RATE, 0.0);
        }
        let result = session.run(&chunk, &run)?;
        let offset_ms = to_ms(start);
        if !word_timed || result.words.is_empty() {
            let text = result.text.trim();
            if !text.is_empty() {
                pieces.push((offset_ms, to_ms(end), text.to_owned()));
            }
            continue;
        }
        for word in result.words {
            let text = word.text.trim();
            if !text.is_empty() {
                pieces.push((word.t0_ms + offset_ms, word.t1_ms + offset_ms, text.to_owned()));
            }
        }
    }
    let transcribe_secs = t1.elapsed().as_secs_f64();
    progress(Progress::Transcribed {
        took_secs: transcribe_secs,
        realtime_factor: audio_secs / transcribe_secs,
    });
    if word_timed {
        Ok(group_words(&pieces))
    } else {
        Ok(pieces.into_iter().map(|(s, e, text)| (s / 10, e / 10, text)).collect())
    }
}

/// Longest stretch handed to a word-aligned model (Parakeet) in one call.
/// Its encoder attends over the whole input, so memory grows quadratically.
const ALIGNED_CHUNK_SECS: f32 = 240.0;

/// For models without timestamps, each decoded stretch becomes one
/// transcript segment: stretches of speech separated by less than
/// [`UTTERANCE_GAP_SECS`] of silence are joined, up to
/// [`UTTERANCE_MAX_SECS`], so segments stay short enough for speaker
/// labels and timestamps to be useful.
const UTTERANCE_MAX_SECS: f32 = 20.0;
const UTTERANCE_GAP_SECS: f32 = 0.6;

/// Stretches of speech (sample ranges) found by the VAD model, padded a
/// little so no word is clipped. Without a VAD model (or if it fails),
/// the whole input as one span; [`merge_spans`] then cuts fixed windows.
fn speech_spans(samples: &[f32], vad_model: Option<&Path>, threads: i32) -> Vec<(usize, usize)> {
    let pad = WHISPER_SAMPLE_RATE / 4;
    vad_model
        .and_then(|model| vad_speech_spans(model, samples, threads).ok())
        .map(|spans| {
            spans
                .into_iter()
                .map(|(s, e)| (s.saturating_sub(pad), (e + pad).min(samples.len())))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![(0, samples.len())])
}

/// Speech spans (sample indices) according to whisper.cpp's Silero VAD.
fn vad_speech_spans(model: &Path, samples: &[f32], threads: i32) -> Result<Vec<(usize, usize)>> {
    // whisper.cpp logs every VAD segment at info level; keep stderr clean.
    whisper_rs::install_logging_hooks();
    let path = model.to_str().context("VAD model path is not valid UTF-8")?;
    let mut params = WhisperVadContextParams::default();
    params.set_n_threads(threads);
    let mut vad = WhisperVadContext::new(path, params)
        .with_context(|| format!("failed to load VAD model {}", model.display()))?;
    let segments = vad.segments_from_samples(WhisperVadParams::default(), samples)?;
    Ok(segments
        .map(|segment| {
            // whisper.cpp reports centiseconds
            let to_sample = |cs: f32| ((cs / 100.0) * WHISPER_SAMPLE_RATE as f32) as usize;
            (to_sample(segment.start), to_sample(segment.end).min(samples.len()))
        })
        .filter(|(s, e)| e > s)
        .collect())
}

/// Join consecutive spans into chunks no longer than `max_len` samples
/// (a chunk covers everything from its first span's start to its last
/// span's end, silence included), never joining across a gap wider than
/// `max_gap`. A single span longer than `max_len` is cut into equal
/// pieces.
fn merge_spans(spans: &[(usize, usize)], max_len: usize, max_gap: usize) -> Vec<(usize, usize)> {
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    for &(start, end) in spans {
        current = match current {
            Some((cs, ce))
                if end.saturating_sub(cs) <= max_len && start.saturating_sub(ce) <= max_gap =>
            {
                Some((cs, end))
            }
            Some(chunk) => {
                chunks.push(chunk);
                Some((start, end))
            }
            None => Some((start, end)),
        };
    }
    chunks.extend(current);
    chunks
        .into_iter()
        .flat_map(|(start, end)| {
            let pieces = (end - start).div_ceil(max_len).max(1);
            let piece_len = (end - start).div_ceil(pieces);
            (0..pieces).map(move |i| {
                let s = start + i * piece_len;
                (s, (s + piece_len).min(end))
            })
        })
        .filter(|(s, e)| e > s)
        .collect()
}

/// Regroup timed words (milliseconds) into transcript segments: a new
/// one starts after a pause of over a second, after a sentence end once the segment has
/// some length, or when the segment gets long. Times are returned in
/// centiseconds like whisper's.
fn group_words(words: &[(i64, i64, String)]) -> Vec<RawSegment> {
    const PAUSE_MS: i64 = 1_200;
    const SENTENCE_MIN_MS: i64 = 4_000;
    const MAX_MS: i64 = 20_000;
    let mut segments: Vec<RawSegment> = Vec::new();
    let mut current: Option<(i64, i64, String)> = None;
    for (start, end, word) in words {
        let flush = match &current {
            Some((seg_start, seg_end, text)) => {
                let ended_sentence = text.ends_with(['.', '?', '!']);
                start - seg_end > PAUSE_MS
                    || (ended_sentence && seg_end - seg_start >= SENTENCE_MIN_MS)
                    || end - seg_start > MAX_MS
            }
            None => false,
        };
        if flush && let Some((s, e, text)) = current.take() {
            segments.push((s / 10, e / 10, text));
        }
        match &mut current {
            Some((_, seg_end, text)) => {
                text.push(' ');
                text.push_str(word);
                *seg_end = (*end).max(*seg_end);
            }
            None => current = Some((*start, *end, word.clone())),
        }
    }
    if let Some((s, e, text)) = current {
        segments.push((s / 10, e / 10, text));
    }
    segments
}

/// Attach speaker bookkeeping to the formatted text.
fn finish_transcript(
    text: String,
    opts: &Options,
    diarization: Option<diarize::Diarization>,
    speakers_seen: std::collections::HashSet<usize>,
    speaker_numbers: HashMap<usize, usize>,
) -> Result<Transcript> {
    let mut speaker_matches: Vec<(String, f32)> = diarization
        .as_ref()
        .map(|d| d.names.values().cloned().collect())
        .unwrap_or_default();
    speaker_matches.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut speaker_voices: Vec<SpeakerVoice> = Vec::new();
    if let Some(d) = &diarization {
        let mut ids: Vec<usize> = speakers_seen.iter().copied().collect();
        // Transcript order: numbered speakers by their number, named ones first.
        ids.sort_by_key(|id| speaker_numbers.get(id).copied().unwrap_or(0));
        for id in ids {
            let label = match d.names.get(&id) {
                Some((name, _)) => name.clone(),
                None => match speaker_numbers.get(&id) {
                    Some(number) => format!("Speaker {number}"),
                    None => continue,
                },
            };
            if let Some(embedding) = d.centroids.get(id - 1) {
                speaker_voices.push(SpeakerVoice {
                    label,
                    embedding: embedding.clone(),
                });
            }
        }
    }

    Ok(Transcript {
        text,
        speakers: opts.diarize.as_ref().map(|_| speakers_seen.len()),
        speaker_matches,
        weak_matches: diarization.map(|d| d.weak_matches).unwrap_or_default(),
        speaker_voices,
    })
}

/// A short leading "X:" prefix — at most four words, no sentence
/// punctuation. The shape whisper produces when it mimics a
/// "Name: text" script format from names it saw in its prompt.
fn leading_prefix(text: &str) -> Option<&str> {
    let colon = text.find(':')?;
    if colon == 0 || colon > 48 {
        return None;
    }
    let prefix = text[..colon].trim_end();
    (prefix.len() >= 3
        && !prefix.contains(['.', ',', '!', '?'])
        && prefix.split_whitespace().count() <= 4)
        .then_some(prefix)
}

/// True when the prefix looks like a name (capitalized).
fn name_like(prefix: &str) -> bool {
    prefix.chars().next().is_some_and(char::is_uppercase)
}

/// Remove leading "Name:" prefixes that whisper hallucinates when names
/// appear in its prompt. A prefix is stripped when it matches an enrolled
/// speaker name, appears verbatim in the prompt sources (context notes,
/// vocabulary), or repeats across many segments (`frequent`) — real
/// dialogue never starts dozens of lines with the same name.
fn strip_hallucinated_prefix<'a>(
    text: &'a str,
    names: &[String],
    sources_lower: &str,
    frequent: &std::collections::HashSet<String>,
) -> &'a str {
    let mut t = text;
    while let Some(prefix) = leading_prefix(t) {
        let enrolled = names.iter().any(|n| n.eq_ignore_ascii_case(prefix));
        let from_sources = name_like(prefix)
            && !sources_lower.is_empty()
            && sources_lower.contains(&prefix.to_lowercase());
        if !(enrolled || from_sources || frequent.contains(prefix)) {
            return t;
        }
        let colon = t.find(':').unwrap_or(0);
        t = t[colon + 1..].trim_start();
    }
    t
}

/// Decode any supported audio file to 16 kHz mono f32 PCM (whisper's input format).
pub fn decode_to_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("unrecognized or unsupported container format")?;

    let track = format
        .tracks()
        .iter()
        .find(|t| matches!(t.codec_params, Some(CodecParameters::Audio(_))))
        .context("no audio track found")?;
    let track_id = track.id;
    let Some(CodecParameters::Audio(audio_params)) = &track.codec_params else {
        unreachable!()
    };

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .context("unsupported audio codec")?;

    let mut interleaved: Vec<f32> = Vec::new();
    let mut packet_buf: Vec<f32> = Vec::new();
    let mut rate = 0u32;
    let mut channels = 0usize;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(e) => return Err(e).context("error reading packet"),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                if rate == 0 {
                    rate = decoded.spec().rate();
                    channels = decoded.spec().channels().count();
                }
                decoded.copy_to_vec_interleaved(&mut packet_buf);
                interleaved.extend_from_slice(&packet_buf);
            }
            // A corrupt frame is recoverable: skip it and continue with the next packet.
            Err(SymphoniaError::DecodeError(err)) => {
                eprintln!("warning: skipping corrupt frame: {err}");
            }
            Err(e) => return Err(e).context("fatal decode error"),
        }
    }

    if interleaved.is_empty() || rate == 0 {
        bail!("no audio frames decoded");
    }

    let mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if rate as usize == WHISPER_SAMPLE_RATE {
        Ok(mono)
    } else {
        resample_to_16k(&mono, rate as usize)
    }
}

/// Resample mono PCM from the given rate to 16 kHz.
pub fn resample_to_16k(mono: &[f32], rate: usize) -> Result<Vec<f32>> {
    let frames = mono.len();
    let mut resampler = Fft::<f32>::new(rate, WHISPER_SAMPLE_RATE, 4096, 1, FixedSync::Input)?;
    let input = InterleavedSlice::new(mono, 1, frames)
        .map_err(|e| anyhow::anyhow!("resampler input: {e}"))?;
    let output = resampler.process_all(&input, frames, None)?;
    Ok(output.take_data())
}

/// Format a whisper timestamp (centiseconds) as HH:MM:SS.mmm.
pub fn format_timestamp(centiseconds: i64) -> String {
    let ms = centiseconds * 10;
    let (h, m, s, ms) = (
        ms / 3_600_000,
        (ms / 60_000) % 60,
        (ms / 1_000) % 60,
        ms % 1_000,
    );
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_speech_spans_into_bounded_chunks() {
        // Three spans; the first two fit one chunk, the third starts a new one.
        let any_gap = usize::MAX;
        assert_eq!(
            merge_spans(&[(0, 40), (50, 90), (100, 150)], 100, any_gap),
            vec![(0, 90), (100, 150)]
        );
        // A single overlong span is cut into equal pieces.
        assert_eq!(merge_spans(&[(0, 250)], 100, any_gap), vec![(0, 84), (84, 168), (168, 250)]);
        assert_eq!(merge_spans(&[], 100, any_gap), Vec::<(usize, usize)>::new());
        // A gap wider than max_gap always starts a new chunk.
        assert_eq!(
            merge_spans(&[(0, 40), (45, 60), (80, 90)], 1000, 10),
            vec![(0, 60), (80, 90)]
        );
    }

    #[test]
    fn groups_words_at_pauses_and_sentence_ends() {
        let w = |s: i64, e: i64, t: &str| (s, e, t.to_owned());
        let words = vec![
            w(0, 300, "Hello"),
            w(300, 600, "there."),
            // short sentence: no break yet (under SENTENCE_MIN_MS)
            w(700, 1000, "Still"),
            w(1000, 1300, "going"),
            // long pause -> new segment
            w(3000, 3400, "After"),
            w(3400, 3800, "pause."),
        ];
        let segments = group_words(&words);
        assert_eq!(
            segments,
            vec![(0, 130, "Hello there. Still going".to_owned()), (300, 380, "After pause.".to_owned())]
        );
        assert!(group_words(&[]).is_empty());
    }

    #[test]
    fn empty_vocabulary_keeps_default_prompt() {
        assert_eq!(prompt_with_vocabulary(""), DEFAULT_PROMPT);
        assert_eq!(
            prompt_with_vocabulary("# Only headings\n\n<!-- and comments -->"),
            DEFAULT_PROMPT
        );
    }

    #[test]
    fn vocabulary_terms_become_glossary() {
        let md = "# Vocab\n- Kubernetes\n* `gRPC`\n2. Grafana\nplain term";
        assert_eq!(
            prompt_with_vocabulary(md),
            format!("Glossary: Kubernetes, gRPC, Grafana, plain term. {DEFAULT_PROMPT}")
        );
    }

    #[test]
    fn hallucinated_name_prefixes_are_stripped() {
        let none = std::collections::HashSet::new();
        let names = vec!["Jane Miller".to_string(), "John Carter".to_string()];
        assert_eq!(
            strip_hallucinated_prefix("Jane Miller: Sorry to hear that.", &names, "", &none),
            "Sorry to hear that."
        );
        assert_eq!(
            strip_hallucinated_prefix("john carter: Jane Miller: Yeah.", &names, "", &none),
            "Yeah."
        );
        // Names mid-sentence or other colons stay untouched.
        assert_eq!(
            strip_hallucinated_prefix("Note: Jane Miller is away.", &names, "", &none),
            "Note: Jane Miller is away."
        );
        assert_eq!(
            strip_hallucinated_prefix("Hello there.", &[], "", &none),
            "Hello there."
        );
    }

    #[test]
    fn context_names_are_stripped_without_enrollment() {
        let none = std::collections::HashSet::new();
        // Names can come from the context box or the vocabulary file.
        let sources = "## people\n- jane miller\n- john carter";
        assert_eq!(
            strip_hallucinated_prefix("Jane Miller: Going on and...", &[], sources, &none),
            "Going on and..."
        );
        // Lowercase or punctuated prefixes don't qualify as names.
        assert_eq!(
            strip_hallucinated_prefix("weekly sync: notes", &[], sources, &none),
            "weekly sync: notes"
        );
        assert_eq!(
            strip_hallucinated_prefix("Jane Miller: Yeah.", &[], "", &none),
            "Jane Miller: Yeah."
        );
    }

    #[test]
    fn renaming_swaps_labels_at_line_starts_only() {
        let plain = "Speaker 1: Hello.\nSpeaker 2: I met Speaker 1: no.\n";
        assert_eq!(
            rename_speaker(plain, "Speaker 1", "John Doe"),
            "John Doe: Hello.\nSpeaker 2: I met Speaker 1: no.\n"
        );
        let stamped = "[00:00:01.000 --> 00:00:02.000]  Speaker 1: Hi.\n";
        assert_eq!(
            rename_speaker(stamped, "Speaker 1", "John Doe"),
            "[00:00:01.000 --> 00:00:02.000]  John Doe: Hi.\n"
        );
        // Speaker 10 must not be touched when renaming Speaker 1.
        let ten = "Speaker 10: Hi.\n";
        assert_eq!(rename_speaker(ten, "Speaker 1", "X"), ten);
    }

    #[test]
    fn frequent_prefixes_are_stripped_regardless_of_source() {
        let frequent: std::collections::HashSet<String> =
            [String::from("Jane Miller")].into();
        assert_eq!(
            strip_hallucinated_prefix("Jane Miller: Yeah.", &[], "", &frequent),
            "Yeah."
        );
        assert_eq!(leading_prefix("Jane Miller: Yeah."), Some("Jane Miller"));
        // Long clauses and sentence-punctuated prefixes aren't names.
        assert_eq!(leading_prefix("this is a much longer clause than a name: x"), None);
        assert_eq!(leading_prefix("Yes, well: x"), None);
    }

    #[test]
    fn context_joins_glossary_and_nudge() {
        assert_eq!(
            build_prompt("- Grafana", "Weekly sync\nwith Thomas"),
            format!("Glossary: Grafana. Context: Weekly sync with Thomas. {DEFAULT_PROMPT}")
        );
        assert_eq!(
            build_prompt("", "Kickoff!"),
            format!("Context: Kickoff! {DEFAULT_PROMPT}")
        );
    }
}
