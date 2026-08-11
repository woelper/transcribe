use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use transcribe::{DiarizeModels, Options, Progress};

/// Transcribe an audio file locally with Whisper (Metal-accelerated).
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Input audio file (mp3, mp4/m4a, wav, flac, ogg, ...)
    audio: PathBuf,

    /// Path to a ggml Whisper model (see download-model.sh)
    #[arg(short, long, default_value = "models/ggml-large-v3-turbo.bin")]
    model: PathBuf,

    /// Write the transcript to this file instead of stdout
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Language spoken in the audio ("auto" to detect)
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// Use beam search with this width instead of greedy decoding
    /// (slightly more accurate, slower)
    #[arg(short, long)]
    beam_size: Option<i32>,

    /// Prefix each segment with [HH:MM:SS.mmm --> HH:MM:SS.mmm] timestamps
    #[arg(short, long)]
    timestamps: bool,

    /// Translate the transcript to English
    #[arg(long)]
    translate: bool,

    /// Number of CPU threads (default: min(8, available cores))
    #[arg(long)]
    threads: Option<i32>,

    /// Initial prompt to bias style/vocabulary (e.g. names, jargon).
    /// Default: a punctuation nudge, plus the terms from vocabulary.md
    /// if that file exists (see README)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Detect speakers and prefix each segment with "Speaker N:"
    /// (needs the diarization models, see download-diarization-models.sh)
    #[arg(short, long)]
    diarize: bool,

    /// Path to the pyannote segmentation ONNX model
    #[arg(long, default_value = "models/segmentation-3.0.onnx")]
    segmentation_model: PathBuf,

    /// Path to the wespeaker embedding ONNX model
    #[arg(long, default_value = "models/wespeaker_en_voxceleb_CAM++.onnx")]
    embedding_model: PathBuf,

    /// Maximum number of distinct speakers to detect
    #[arg(long, default_value_t = 8)]
    max_speakers: usize,

    /// Similarity threshold for matching a voice to a known speaker
    /// (lower merges speakers, higher splits them)
    #[arg(long, default_value_t = 0.5)]
    speaker_threshold: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.model.exists() {
        bail!(
            "model not found at {}\nrun ./download-model.sh to fetch it",
            args.model.display()
        );
    }
    if args.diarize {
        for path in [&args.segmentation_model, &args.embedding_model] {
            if !path.exists() {
                bail!(
                    "diarization model not found at {}\nrun ./download-diarization-models.sh to fetch it",
                    path.display()
                );
            }
        }
    }

    let vocabulary = match args.prompt {
        Some(_) => String::new(),
        None => {
            let path = transcribe::vocabulary_path();
            match std::fs::read_to_string(&path) {
                Ok(vocabulary) => {
                    eprintln!("using vocabulary from {}", path.display());
                    vocabulary
                }
                Err(_) => String::new(),
            }
        }
    };
    let prompt = match args.prompt {
        Some(prompt) => prompt,
        None if vocabulary.is_empty() => transcribe::DEFAULT_PROMPT.into(),
        None => transcribe::prompt_with_vocabulary(&vocabulary),
    };

    let speakers_path = transcribe::speakers_path();
    let profiles = transcribe::load_speaker_profiles(&speakers_path)?;
    if args.diarize && !profiles.is_empty() {
        eprintln!(
            "using {} enrolled speaker profile(s) from {}",
            profiles.len(),
            speakers_path.display()
        );
    }

    let opts = Options {
        model: args.model,
        language: args.language,
        beam_size: args.beam_size,
        translate: args.translate,
        threads: args.threads,
        prompt,
        timestamps: args.timestamps,
        known_speakers: profiles.iter().map(|p| p.name.clone()).collect(),
        context: vocabulary,
        diarize: args.diarize.then_some(DiarizeModels {
            segmentation_model: args.segmentation_model,
            embedding_model: args.embedding_model,
            max_speakers: args.max_speakers,
            threshold: args.speaker_threshold,
            profiles,
        }),
    };

    eprintln!("decoding {} ...", args.audio.display());
    let transcript = transcribe::transcribe(&args.audio, &opts, |progress| match progress {
        Progress::Decoded { audio_secs, took_secs } => {
            eprintln!("decoded {audio_secs:.1}s of audio in {took_secs:.1}s");
        }
        Progress::DetectingSpeakers => eprintln!("detecting speakers ..."),
        Progress::Diarized { segments, took_secs } => {
            eprintln!("diarized {segments} speech segment(s) in {took_secs:.1}s");
        }
        Progress::Transcribing { percent } => eprint!("\rtranscribing... {percent}%"),
        Progress::Segment { .. } => {}
        Progress::Transcribed { took_secs, realtime_factor } => {
            eprintln!("\rtranscribed in {took_secs:.1}s ({realtime_factor:.1}x realtime)");
        }
    })?;
    if let Some(speakers) = transcript.speakers {
        eprintln!("{speakers} speaker(s) in the transcript");
    }
    let similarities = |matches: &[(String, f32)]| -> String {
        matches
            .iter()
            .map(|(name, s)| format!("{name} ({s:.2})"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if !transcript.speaker_matches.is_empty() {
        eprintln!("recognized voices: {}", similarities(&transcript.speaker_matches));
    }
    if !transcript.weak_matches.is_empty() {
        eprintln!(
            "too weak to apply: {} — consider re-enrolling from this recording's audio",
            similarities(&transcript.weak_matches)
        );
    }

    match &args.output {
        Some(path) => {
            std::fs::write(path, &transcript.text)
                .with_context(|| format!("failed to write {}", path.display()))?;
            eprintln!("transcript written to {}", path.display());
        }
        None => print!("{}", transcript.text),
    }

    Ok(())
}
