//! Local transcript summarization with a small llama.cpp chat model.

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context as _, Result, anyhow};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::TokenToStringError;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

/// Model context window. Bounds memory (the KV cache is on the order of
/// 1 GB at this size for a 3B model); transcripts that tokenize past it
/// are trimmed in the middle, keeping the start and the end.
const N_CTX: u32 = 8192;
/// Tokens reserved for the generated summary.
const MAX_OUTPUT_TOKENS: usize = 1024;
/// Prompt tokens are fed to the model in chunks of this size.
const N_BATCH: usize = 2048;

const SYSTEM_PROMPT: &str = "You summarize meeting and speech transcripts. \
    Answer in the language the transcript is written in. Structure the answer \
    as: a short overview paragraph; key points as a bullet list; decisions and \
    action items as bullet lists when the transcript contains any. Be concise \
    and factual — never invent content that is not in the transcript.";

const TRUNCATION_MARKER: &str = "\n[... middle of the transcript omitted ...]\n";

/// The llama backend can only be initialized once per process.
fn backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let backend = LlamaBackend::init().map_err(|e| anyhow!("llama init failed: {e}"))?;
    Ok(BACKEND.get_or_init(|| backend))
}

/// A token's raw text bytes (special tokens render as nothing).
fn token_bytes(model: &LlamaModel, token: LlamaToken) -> Result<Vec<u8>> {
    match model.token_to_piece_bytes(token, 32, false, None) {
        Err(TokenToStringError::InsufficientBufferSpace(i)) => Ok(model.token_to_piece_bytes(
            token,
            usize::try_from(-i).expect("needed buffer size is positive"),
            false,
            None,
        )?),
        result => Ok(result?),
    }
}

/// Largest index `<= at` that sits on a char boundary of `s`.
fn char_floor(s: &str, mut at: usize) -> usize {
    at = at.min(s.len());
    while !s.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// The transcript, cut down to roughly `keep` bytes by dropping the
/// middle — the start (agenda, names) and the end (decisions, action
/// items) carry the most weight in a meeting.
fn trimmed(transcript: &str, keep: usize) -> String {
    if keep >= transcript.len() {
        return transcript.to_owned();
    }
    let head = char_floor(transcript, keep / 2);
    let tail = char_floor(transcript, transcript.len() - (keep - head).min(transcript.len()));
    format!(
        "{}{TRUNCATION_MARKER}{}",
        &transcript[..head],
        &transcript[tail..]
    )
}

/// Summarize `transcript` with the GGUF chat model at `model_path`,
/// streaming the partial summary to `on_progress` as it generates.
/// `context` is the user's free-form notes about the recording.
pub fn summarize(
    model_path: &Path,
    transcript: &str,
    context: &str,
    on_progress: &dyn Fn(&str),
) -> Result<String> {
    anyhow::ensure!(!transcript.trim().is_empty(), "nothing to summarize");
    let backend = backend()?;

    let model_params = LlamaModelParams::default();
    // Offload every layer to Metal; other platforms stay on the CPU.
    #[cfg(target_os = "macos")]
    let model_params = model_params.with_n_gpu_layers(u32::MAX);
    let model = LlamaModel::load_from_file(backend, model_path, &model_params)
        .with_context(|| format!("failed to load model {}", model_path.display()))?;
    let template = model
        .chat_template(None)
        .map_err(|e| anyhow!("the model has no chat template: {e}"))?;

    // Fit the prompt into the context window, trimming the transcript
    // middle if needed. Token counts only shrink roughly linearly with
    // bytes, so re-check after each cut.
    let budget = N_CTX as usize - MAX_OUTPUT_TOKENS - 64;
    let mut keep = transcript.len();
    let tokens = loop {
        let mut user = String::new();
        if !context.trim().is_empty() {
            user.push_str(&format!("Notes about the recording:\n{context}\n\n"));
        }
        user.push_str(&format!(
            "Summarize this transcript:\n\n{}",
            trimmed(transcript, keep)
        ));
        let messages = vec![
            LlamaChatMessage::new("system".into(), SYSTEM_PROMPT.into())?,
            LlamaChatMessage::new("user".into(), user)?,
        ];
        let prompt = model
            .apply_chat_template(&template, &messages, true)
            .map_err(|e| anyhow!("applying the chat template failed: {e}"))?;
        let tokens = model.str_to_token(&prompt, AddBos::Always)?;
        if tokens.len() <= budget {
            break tokens;
        }
        // Cut ~15% below the projected fit so this converges quickly.
        keep = keep * budget * 85 / (tokens.len() * 100);
    };

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(N_CTX))
        .with_n_batch(N_BATCH as u32)
        .with_n_threads(threads)
        .with_n_threads_batch(threads);
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("failed to create llama context")?;

    // Feed the prompt in n_batch-sized chunks; only the last token needs
    // logits computed.
    let mut batch = LlamaBatch::new(N_BATCH, 1);
    let mut pos: i32 = 0;
    for chunk in tokens.chunks(N_BATCH) {
        batch.clear();
        for (i, &token) in chunk.iter().enumerate() {
            let last = pos as usize + i + 1 == tokens.len();
            batch.add(token, pos + i as i32, &[0], last)?;
        }
        ctx.decode(&mut batch).context("prompt decoding failed")?;
        pos += chunk.len() as i32;
    }

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.3),
        LlamaSampler::dist(42),
    ]);
    // Bytes, not a String: a multi-byte char can span two tokens.
    let mut out: Vec<u8> = Vec::new();
    for _ in 0..MAX_OUTPUT_TOKENS {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        out.extend(token_bytes(&model, token)?);
        on_progress(&String::from_utf8_lossy(&out));
        batch.clear();
        batch.add(token, pos, &[0], true)?;
        pos += 1;
        ctx.decode(&mut batch).context("generation failed")?;
    }

    Ok(String::from_utf8_lossy(&out).trim().to_owned())
}
