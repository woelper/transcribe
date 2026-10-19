use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

pub struct SpeechModel {
    pub name: &'static str,
    /// Download size in MB (also a fair proxy for RAM use).
    pub mb: u32,
    /// Relative accuracy and speed, 0..=1 across this list — for the
    /// picker's bars, not a benchmark: ordered by Open ASR leaderboard
    /// WER / RTFx where available, else by whisper.cpp's own size ladder.
    pub accuracy: f32,
    pub speed: f32,
    /// Caveat shown in the model picker, e.g. a language restriction.
    pub note: &'static str,
    /// Local filename under models/.
    pub file: &'static str,
    pub url: &'static str,
}

macro_rules! standard_model {
    ($name:literal, $mb:literal, $accuracy:literal, $speed:literal) => {
        SpeechModel {
            name: $name,
            mb: $mb,
            accuracy: $accuracy,
            speed: $speed,
            note: "",
            file: concat!("ggml-", $name, ".bin"),
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-",
                $name,
                ".bin"
            ),
        }
    };
}

/// Speech-to-text models, downloadable from Hugging Face. The `.bin` files
/// are Whisper models run by whisper.cpp: large-v3-turbo is the best
/// multilingual accuracy/speed tradeoff, distil-large-v3.5 is faster and
/// slightly more accurate but English-only, and the small ones trade
/// accuracy for speed and disk space. The `.gguf` entries run on
/// transcribe.cpp and take no vocabulary/context prompt: parakeet-ultra
/// (Moondream's tuning of NVIDIA's Parakeet v3) is more accurate than any
/// Whisper on English and an order of magnitude faster, for 25 European
/// languages; Alibaba's Qwen3-ASR covers 52
/// languages and is by far the most robust to background noise.
pub const SPEECH_MODELS: &[SpeechModel] = &[
    standard_model!("tiny", 75, 0.2, 1.0),
    standard_model!("base", 142, 0.3, 0.95),
    standard_model!("small", 466, 0.45, 0.8),
    standard_model!("medium", 1500, 0.6, 0.45),
    standard_model!("large-v3-turbo", 1600, 0.75, 0.5),
    standard_model!("large-v3", 2900, 0.8, 0.25),
    SpeechModel {
        name: "distil-large-v3.5",
        mb: 1500,
        accuracy: 0.8,
        speed: 0.65,
        note: "English only",
        file: "ggml-distil-large-v3.5.bin",
        url: "https://huggingface.co/distil-whisper/distil-large-v3.5-ggml/resolve/main/ggml-model.bin",
    },
    SpeechModel {
        name: "parakeet-ultra",
        mb: 706,
        accuracy: 0.95,
        speed: 1.0,
        note: "25 EU langs, no prompt",
        file: "parakeet-ultra-Q8_0.gguf",
        url: "https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q8_0.gguf",
    },    SpeechModel {
        name: "qwen3-asr-1.7b",
        mb: 2084,
        accuracy: 0.9,
        speed: 0.55,
        note: "52 languages, no prompt",
        file: "Qwen3-ASR-1.7B-Q8_0.gguf",
        url: "https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q8_0.gguf",
    },
    SpeechModel {
        name: "qwen3-asr-0.6b",
        mb: 811,
        accuracy: 0.75,
        speed: 0.85,
        note: "52 languages, no prompt",
        file: "Qwen3-ASR-0.6B-Q8_0.gguf",
        url: "https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q8_0.gguf",
    },
];

/// "705 MB" / "1.6 GB", for a size in MiB.
fn size_label(mb: u32) -> String {
    if mb >= 1000 {
        format!("{:.1} GB", mb as f32 / 1000.0)
    } else {
        format!("{mb} MB")
    }
}

impl SpeechModel {
    /// "705 MB" / "1.6 GB".
    pub fn size_label(&self) -> String {
        size_label(self.mb)
    }
}

pub fn model_by_name(name: &str) -> Option<&'static SpeechModel> {
    SPEECH_MODELS.iter().find(|m| m.name == name)
}

pub struct DiarizationModel {
    /// Local filename under models/.
    pub file: &'static str,
    pub url: &'static str,
}

pub struct SummaryModel {
    pub name: &'static str,
    /// Download size in MiB (also a fair proxy for RAM use).
    pub mb: u32,
    /// Caveat shown in the model picker, e.g. how much RAM it wants.
    pub note: &'static str,
    /// Local filename under models/.
    pub file: &'static str,
    pub url: &'static str,
}

/// Chat models for the Summarize button and the titles auto mode gives a
/// finished meeting, smallest first. All of them are Qwen3.5 (March 2026,
/// Apache 2.0), which is multilingual over 201 languages and takes a
/// whole meeting transcript in one window. Their optional thinking mode
/// stays off: at these sizes the model spends the entire token budget
/// thinking and never reaches an answer (measured on the 4B — 6144
/// tokens of reasoning, no summary), and any `<think>` block is
/// stripped anyway.
///
/// The ladder spans a factor of thirteen in download and memory on
/// purpose, because the machines that run this app do too. `2b` is the
/// floor for old and small hardware — 1.2 GB down, ~2.3 GB resident,
/// a minute for a 40-minute meeting. It gets the shape of a meeting and
/// the agreed numbers right, but misattributes some action items, so it
/// is a fallback rather than a recommendation.
///
/// Nothing below 2B belongs here: Qwen3.5-0.8B was tried and collapsed
/// into a repetition loop on the same transcript — 91 copies of one
/// action item until it hit the output-token cap.
/// Anything that small would need a presence penalty in the sampler
/// before it could be offered at all.
///
/// `35b-a3b` is a sparse mixture of experts: 35B total but only ~3B
/// active per token, so on a CPU it *generates faster* than the dense 9B
/// while writing a far better summary — it keeps the tradeoffs and the
/// numbers a decision turned on, which every smaller one drops. It is
/// the one to pick when the machine has the RAM; the 4B stays the
/// default only because it is a sixth of the download.
pub const SUMMARY_MODELS: &[SummaryModel] = &[
    SummaryModel {
        name: "qwen3.5-2b",
        mb: 1221,
        note: "small, keeps the shape",
        file: "Qwen3.5-2B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
    },
    SummaryModel {
        name: "qwen3.5-4b",
        mb: 2613,
        note: "fastest, drops detail",
        file: "Qwen3.5-4B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
    },
    SummaryModel {
        name: "qwen3.5-9b",
        mb: 5417,
        note: "needs ~10 GB RAM",
        file: "Qwen3.5-9B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
    },
    SummaryModel {
        name: "qwen3.5-35b-a3b",
        mb: 15831,
        note: "best, needs ~18 GB RAM",
        file: "Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf",
        url: "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf",
    },
];

/// What a fresh install summarizes with: the smallest download that
/// works, not the best summary. See [`SUMMARY_MODELS`].
pub const DEFAULT_SUMMARY_MODEL: &str = "qwen3.5-4b";

impl SummaryModel {
    /// "705 MB" / "1.6 GB".
    pub fn size_label(&self) -> String {
        size_label(self.mb)
    }
}

pub fn summary_model_by_name(name: &str) -> Option<&'static SummaryModel> {
    SUMMARY_MODELS.iter().find(|m| m.name == name)
}

/// whisper.cpp's built-in voice activity detection model (Silero v5.1.2,
/// under 1 MB). With it, whisper only decodes stretches that contain
/// speech, which removes the hallucinated sentences it otherwise invents
/// for silence and background noise in long recordings.
pub const VAD_MODEL_FILE: &str = "ggml-silero-v5.1.2.bin";
pub const VAD_MODEL_URL: &str =
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";

/// Speaker-diarization ONNX models (pyannote segmentation + wespeaker
/// embeddings, ~34 MB total), mirrored by the pyannote-rs project.
pub const DIARIZATION_MODELS: &[DiarizationModel] = &[
    DiarizationModel {
        file: "segmentation-3.0.onnx",
        url: "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx",
    },
    DiarizationModel {
        file: "wespeaker_en_voxceleb_CAM++.onnx",
        url: "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx",
    },
];

/// Stream `url` into `dest`, calling `progress` with
/// (downloaded_bytes, total_bytes_if_known) along the way. Writes through
/// a .part file so an aborted download never leaves a truncated `dest`.
pub fn download(url: &str, dest: &Path, progress: impl Fn(u64, Option<u64>)) -> Result<()> {
    if let Some(dir) = dest.parent()
        && !dir.as_os_str().is_empty()
    {
        fs::create_dir_all(dir)?;
    }

    let response = ureq::get(url)
        .call()
        .with_context(|| format!("request failed: {url}"))?;
    let body = response.into_body();
    let total = body.content_length();
    let mut reader = body.into_reader();

    let part = dest.with_extension("part");
    let mut file =
        File::create(&part).with_context(|| format!("failed to create {}", part.display()))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
    loop {
        let n = reader.read(&mut buf).context("download interrupted")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    file.flush()?;
    drop(file);

    if let Some(total) = total
        && done != total
    {
        let _ = fs::remove_file(&part);
        bail!("incomplete download: got {done} of {total} bytes");
    }
    fs::rename(&part, dest)?;
    Ok(())
}
