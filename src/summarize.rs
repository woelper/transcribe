//! Local transcript summarization with a llama.cpp chat model
//! (Qwen3.5, see [`crate::download::SUMMARY_MODELS`]).

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

/// Model context window: room for roughly two hours of meeting transcript
/// (an hour is 10–15k tokens). Qwen3.5's hybrid attention keeps the KV
/// cache small at this size. Transcripts that tokenize past it are trimmed
/// in the middle, keeping the start and the end.
const N_CTX: u32 = 32768;
/// Tokens reserved for the generated summary. 1024 was not enough: the
/// minutes for a 40-minute meeting run to ~1100 tokens on the larger
/// models, and the cap cut them off mid-list.
const MAX_OUTPUT_TOKENS: usize = 2048;
/// Prompt tokens are fed to the model in chunks of this size.
const N_BATCH: usize = 2048;

/// Not "minutes of a meeting": told that, the model refuses outright on
/// a transcript that is not one. A solo dictation about buying a car
/// came back as "No meeting minutes can be generated from this
/// transcript", which is useless — the app transcribes dictation as
/// readily as meetings, so the prompt has to cover both and say so.
///
/// Naming the sections and spelling out what belongs in each is worth
/// more than it looks: asked only to "structure the answer", these
/// models merge decisions into action items, drop tasks that were agreed
/// in passing, and quietly round "nineteen a seat" up to "$19" — a
/// currency the speakers never said.
///
/// The two rules about mishearings pull against each other and both are
/// needed. Told only not to invent, the model copies "Cooper Netis" and
/// "engine X" straight through, and will even write "Kubernetes" in the
/// overview and "Cooper Netis" three times below it. Told only to
/// correct, it starts tidying figures and dates it merely finds
/// implausible. So: facts are fixed, words are not, and the examples
/// show which is which.
///
/// This reliably fixes product and technical terms on every size. It
/// does *not* reliably fix people's names — "Layna" for an enrolled
/// Lena survives on both the 4B and the 35B, in different amounts from
/// run to run, even with the name in the vocabulary. Spelling that out
/// in the prompt was tried and moved the two models in opposite
/// directions, so it was dropped. If it matters, do it in code against
/// [`crate::vocabulary_terms`] rather than by asking.
const SYSTEM_PROMPT: &str = "You write up what was said in a recording, from \
    its transcript. It may be a meeting with several speakers, a talk, an \
    interview, or one person dictating free notes or thinking aloud. Write \
    something \
    useful for whichever it is. Never reply that the transcript is unsuitable \
    and never explain why you cannot summarize it — there is always something \
    to write down.\n\
    \n\
    Rules:\n\
    - Answer in the language the transcript is written in.\n\
    - Numbers, currencies and dates are copied, never interpreted. Write every \
    figure exactly as spoken. The speakers said \"nineteen a seat\", so write \
    \"19 a seat\" — writing \"$19 a seat\" invents a currency nobody used, and \
    is wrong even when the amount is right. If no currency was spoken, use no \
    currency symbol at all. Never add a figure or a deadline the transcript \
    does not contain.\n\
    - Correct misheard words — this applies to words only, never to the \
    figures above. The transcript comes from speech recognition, so \
    a name, product or technical term is often written as whatever it sounded \
    like: \"post grass\" for PostgreSQL, \"engine X\" for nginx, \"Cooper Netis\" \
    for Kubernetes, \"sam all\" for SAML. Work out what was meant and write \
    that — that is transcribing what was said, not inventing it. When a \
    mangled word matches one of the names and terms given below, use that \
    spelling, and use one spelling throughout.\n\
    - Speakers are labelled by name or as \"Speaker N\". Attribute each point to \
    the speaker who made it; one person talking alone needs no attribution.\n\
    - Prefer the concrete over the general: name the thing, the number and the \
    owner.\n\
    \n\
    Use these sections, in this order, and leave out any the transcript has \
    nothing for. One person describing a car gets an Overview and Key points \
    and nothing else; that is a complete answer, not a failure.\n\
    \n\
    ## Overview\n\
    Two to four sentences: what this was and what came out of it.\n\
    \n\
    ## Key points\n\
    One bullet per topic, in the order it came up. Name the thing, the figure \
    and the reasoning. Where people disagreed, give the positions taken and how \
    it was left.\n\
    \n\
    ## Decisions\n\
    One bullet per decision actually settled. Include the number or deadline \
    that was agreed.\n\
    \n\
    ## Open questions\n\
    One bullet per thing raised and left unresolved.\n\
    \n\
    ## Action items\n\
    One bullet per task, written as \"Owner - what, by when\". Use \"unassigned\" \
    when nobody took it. List every task, including ones mentioned only in \
    passing.";

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
/// `context` is the user's free-form notes about the recording;
/// `vocabulary` the names and terms from vocabulary.md, so the summary
/// spells them the way the user does.
pub fn summarize(
    model_path: &Path,
    transcript: &str,
    context: &str,
    vocabulary: &str,
    on_progress: &dyn Fn(&str),
) -> Result<String> {
    let build_user = |transcript: &str| {
        let mut user = String::new();
        if !context.trim().is_empty() {
            user.push_str(&format!("Notes about the recording:\n{context}\n\n"));
        }
        let terms = crate::vocabulary_terms(vocabulary);
        if !terms.is_empty() {
            user.push_str(&format!(
                "Names and terms, spelled as they should appear: {}\n\n",
                terms.join(", ")
            ));
        }
        user.push_str(&format!("Summarize this transcript:\n\n{transcript}"));
        user
    };
    let summary = generate(
        model_path,
        SYSTEM_PROMPT,
        &build_user,
        transcript,
        N_CTX,
        MAX_OUTPUT_TOKENS,
        on_progress,
    )?;
    Ok(strip_unspoken_currency(&summary, transcript))
}

/// A finished meeting's name and one-line gist, from the same model that
/// writes the long summaries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Label {
    pub title: String,
    pub summary: String,
}

const LABEL_SYSTEM_PROMPT: &str = "You label meeting transcripts. The \
    transcript comes from speech recognition, so correct any misheard name or \
    term to what was plainly meant, preferring the spellings listed with the \
    transcript. Reply with exactly two lines and nothing else:\n\
    Title: a specific name for this meeting, at most six words, no quotes\n\
    Summary: one sentence, at most 25 words, saying what it was about\n\
    Name the actual subject rather than the format — \"Q3 budget and \
    hiring freeze\", not \"Team meeting\". Write both in the language the \
    transcript is written in.";

/// Naming a meeting only needs its shape, not every word, so it runs over
/// a much smaller window than a full summary — which keeps it to seconds
/// even on a CPU.
const LABEL_N_CTX: u32 = 8192;
const LABEL_MAX_OUTPUT_TOKENS: usize = 128;

/// Title and one-sentence gist for a finished transcript.
pub fn label(model_path: &Path, transcript: &str, vocabulary: &str) -> Result<Label> {
    let build_user = |transcript: &str| {
        let mut user = String::new();
        let terms = crate::vocabulary_terms(vocabulary);
        if !terms.is_empty() {
            user.push_str(&format!(
                "Names and terms, spelled as they should appear: {}\n\n",
                terms.join(", ")
            ));
        }
        user.push_str(&format!("Label this transcript:\n\n{transcript}"));
        user
    };
    let text = generate(
        model_path,
        LABEL_SYSTEM_PROMPT,
        &build_user,
        transcript,
        LABEL_N_CTX,
        LABEL_MAX_OUTPUT_TOKENS,
        &|_| {},
    )?;
    let label = parse_label(&text);
    if label.title.is_empty() {
        // Nothing usable came back — worth seeing verbatim, since the
        // fallback (filing under the date alone) hides the reason.
        log::warn!("labelling produced no title; the model answered {text:?}");
    }
    Ok(label)
}

/// Pull the two labelled lines out of the model's answer, tolerating the
/// ways a small model strays: markdown bullets, bold keys, quotes, or
/// just the bare title on the first line.
fn parse_label(text: &str) -> Label {
    let mut label = Label::default();
    for line in text.lines() {
        let line = line.trim().trim_start_matches(['-', '*', '#']).trim();
        let line = line.replace("**", "");
        let Some((key, value)) = line.split_once(':') else { continue };
        let value = value.trim().trim_matches(['"', '\'', '*']).trim();
        match key.trim().to_ascii_lowercase().as_str() {
            "title" if label.title.is_empty() => label.title = value.to_owned(),
            "summary" if label.summary.is_empty() => label.summary = value.to_owned(),
            _ => {}
        }
    }
    if label.title.is_empty() {
        // No keys at all: take the first line as the title, the rest as
        // the summary.
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        label.title = lines
            .next()
            .unwrap_or_default()
            .trim_matches(['"', '#', '*', ' '])
            .to_owned();
        if label.summary.is_empty() {
            label.summary = lines.collect::<Vec<_>>().join(" ");
        }
    }
    // A title is a name, not a sentence.
    label.title = label.title.trim_end_matches('.').trim().to_owned();
    label
}

/// Strip a currency symbol the speakers never used.
///
/// Asking the model not to invent one does not work: told to write
/// figures exactly as spoken it still turns "nineteen a seat" into
/// "$19 a seat", and three different wordings of the rule (including
/// putting it first, ahead of the licence to correct misheard words)
/// each left five or six of them in a 40-minute meeting. It is a
/// mechanical problem, so it gets a mechanical fix: a symbol that
/// appears nowhere in the transcript, by sign or by name, is dropped
/// from the summary when it sits directly in front of a digit. The
/// amount is the model's to report; the currency is the speakers'.
fn strip_unspoken_currency(summary: &str, transcript: &str) -> String {
    const CURRENCIES: &[(char, &[&str])] = &[
        ('$', &["$", "dollar", "usd", "buck"]),
        ('€', &["€", "euro", "eur"]),
        ('£', &["£", "pound", "gbp", "sterling"]),
    ];
    let spoken = transcript.to_ascii_lowercase();
    let unspoken: Vec<char> = CURRENCIES
        .iter()
        .filter(|(_, names)| !names.iter().any(|n| spoken.contains(n)))
        .map(|(sign, _)| *sign)
        .collect();
    if unspoken.is_empty() {
        return summary.to_owned();
    }
    let mut out = String::with_capacity(summary.len());
    let mut chars = summary.chars().peekable();
    while let Some(c) = chars.next() {
        // Only "$19" goes; a bare symbol, or "US$" style prose, stays.
        if unspoken.contains(&c) && chars.peek().is_some_and(char::is_ascii_digit) {
            continue;
        }
        out.push(c);
    }
    out
}

/// Run one chat-model generation: load the model, fit the prompt into the
/// context window (trimming the transcript's middle when it doesn't fit),
/// decode, and stream the answer to `on_progress`.
fn generate(
    model_path: &Path,
    system: &str,
    build_user: &dyn Fn(&str) -> String,
    transcript: &str,
    n_ctx: u32,
    max_output: usize,
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

    // A reasoning model would spend the whole token budget thinking
    // before writing a word — for summarizing and titling there is
    // nothing to reason about. Opening the answer with an already-closed
    // thinking block is how these templates are told to skip it.
    let thinks = template
        .to_str()
        .is_ok_and(|template| template.contains("<think>"));
    let skip_thinking = if thinks { "<think>\n\n</think>\n\n" } else { "" };

    // Fit the prompt into the context window, trimming the transcript
    // middle if needed. Token counts only shrink roughly linearly with
    // bytes, so re-check after each cut.
    let budget = n_ctx as usize - max_output - 64;
    let mut keep = transcript.len();
    let tokens = loop {
        let user = build_user(&trimmed(transcript, keep));
        let messages = vec![
            LlamaChatMessage::new("system".into(), system.into())?,
            LlamaChatMessage::new("user".into(), user)?,
        ];
        let prompt = model
            .apply_chat_template(&template, &messages, true)
            .map_err(|e| anyhow!("applying the chat template failed: {e}"))?
            + skip_thinking;
        let tokens = model.str_to_token(&prompt, AddBos::Always)?;
        if tokens.len() <= budget {
            break tokens;
        }
        // Cut ~15% below the projected fit so this converges quickly.
        keep = keep * budget * 85 / (tokens.len() * 100);
    };

    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
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
    for _ in 0..max_output {
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

    Ok(without_thinking(&String::from_utf8_lossy(&out)).trim().to_owned())
}

/// Drop a leading `<think>…</think>` block (Qwen's reasoning mode, off by
/// default but not to be trusted to stay off), or everything up to a stray
/// `</think>` if the opening tag was swallowed by the template.
fn without_thinking(text: &str) -> &str {
    let text = text.trim_start();
    match (text.starts_with("<think>"), text.find("</think>")) {
        (true, Some(end)) => &text[end + "</think>".len()..],
        (true, None) => "",
        (false, Some(end)) if !text[..end].contains('\n') || text.starts_with('\n') => {
            &text[end + "</think>".len()..]
        }
        _ => text,
    }
}

#[cfg(test)]
mod tests {
    use super::{strip_unspoken_currency, without_thinking};

    #[test]
    fn invented_currency_is_dropped_but_spoken_currency_is_kept() {
        // Nobody said dollars, so the model's "$" goes and the figure stays.
        assert_eq!(
            strip_unspoken_currency("A seat is $19 and the tier is $49.", "nineteen a seat"),
            "A seat is 19 and the tier is 49."
        );
        // Said out loud, by name or by sign: left alone.
        assert_eq!(
            strip_unspoken_currency("It is $19.", "nineteen dollars a seat"),
            "It is $19."
        );
        assert_eq!(strip_unspoken_currency("It is €3500.", "3500 euros"), "It is €3500.");
        assert_eq!(strip_unspoken_currency("It is £20.", "twenty pounds"), "It is £20.");
        // One currency spoken does not license another.
        assert_eq!(
            strip_unspoken_currency("€3500, about $4000.", "3500 euros"),
            "€3500, about 4000."
        );
        // Not in front of a digit: prose and code are untouched.
        assert_eq!(strip_unspoken_currency("the $ sign and a$b", "nothing"), "the $ sign and a$b");
        // Nothing to do is the common case and must not rewrite anything.
        assert_eq!(strip_unspoken_currency("plain text", "plain"), "plain text");
    }

    #[test]
    fn strips_reasoning_blocks() {
        assert_eq!(without_thinking("<think>\nhmm\n</think>\n\nSummary."), "\n\nSummary.");
        assert_eq!(without_thinking("Summary only."), "Summary only.");
        assert_eq!(without_thinking("<think>unterminated"), "");
        assert_eq!(without_thinking("\nhmm</think>Summary."), "Summary.");
    }
}
