use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

pub struct WhisperModel {
    pub name: &'static str,
    pub size: &'static str,
    /// Caveat shown in the model picker, e.g. a language restriction.
    pub note: &'static str,
    /// Local filename under models/.
    pub file: &'static str,
    pub url: &'static str,
}

macro_rules! standard_model {
    ($name:literal, $size:literal) => {
        WhisperModel {
            name: $name,
            size: $size,
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

/// Whisper-architecture models runnable by whisper.cpp, downloadable from
/// Hugging Face. large-v3-turbo is the best multilingual accuracy/speed
/// tradeoff; distil-large-v3.5 is faster and slightly more accurate but
/// English-only; the small models trade accuracy for speed and disk space.
pub const WHISPER_MODELS: &[WhisperModel] = &[
    standard_model!("tiny", "75 MB"),
    standard_model!("base", "142 MB"),
    standard_model!("small", "466 MB"),
    standard_model!("medium", "1.5 GB"),
    standard_model!("large-v3-turbo", "1.6 GB"),
    standard_model!("large-v3", "2.9 GB"),
    WhisperModel {
        name: "distil-large-v3.5",
        size: "1.5 GB",
        note: "English only",
        file: "ggml-distil-large-v3.5.bin",
        url: "https://huggingface.co/distil-whisper/distil-large-v3.5-ggml/resolve/main/ggml-model.bin",
    },
];

pub fn model_by_name(name: &str) -> Option<&'static WhisperModel> {
    WHISPER_MODELS.iter().find(|m| m.name == name)
}

pub struct DiarizationModel {
    /// Local filename under models/.
    pub file: &'static str,
    pub url: &'static str,
}

/// Chat model used by the Summarize button: Llama 3.2 3B Instruct,
/// Q4_K_M-quantized. Small enough to run on CPU at usable speed while
/// still producing solid meeting summaries.
pub const SUMMARY_MODEL_FILE: &str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf";
pub const SUMMARY_MODEL_URL: &str = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf";
pub const SUMMARY_MODEL_SIZE: &str = "2.0 GB";

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
