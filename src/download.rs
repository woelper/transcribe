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
/// transcribe.cpp and take no vocabulary/context prompt: NVIDIA's Parakeet
/// is more accurate than any Whisper on English and an order of magnitude
/// faster, for 25 European languages; Alibaba's Qwen3-ASR covers 52
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
        name: "parakeet-tdt-0.6b-v3",
        mb: 705,
        accuracy: 0.9,
        speed: 1.0,
        note: "25 EU langs, no prompt",
        file: "parakeet-tdt-0.6b-v3-Q8_0.gguf",
        url: "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf",
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

impl SpeechModel {
    /// "705 MB" / "1.6 GB".
    pub fn size_label(&self) -> String {
        if self.mb >= 1000 {
            format!("{:.1} GB", self.mb as f32 / 1000.0)
        } else {
            format!("{} MB", self.mb)
        }
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

/// Chat model used by the Summarize button: Qwen3.5 4B Instruct (March
/// 2026, Apache 2.0), Q4_K_M-quantized. Small enough to run on CPU at
/// usable speed, strongly multilingual, and with a context window large
/// enough to take a whole meeting transcript. Its optional thinking mode
/// is off by default for this size; any `<think>` block is stripped anyway.
pub const SUMMARY_MODEL_FILE: &str = "Qwen3.5-4B-Q4_K_M.gguf";
pub const SUMMARY_MODEL_URL: &str =
    "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf";
pub const SUMMARY_MODEL_SIZE: &str = "2.6 GB";

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
