//! Summarize a transcript from the command line (the GUI has this built in):
//!   cargo run --release --example summarize -- transcript.txt [model.gguf]
//! Without a model argument, uses the summary model from the models/ directory.

use std::io::Write;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let text = std::env::args()
        .nth(1)
        .expect("usage: summarize <transcript.txt> [model.gguf]");
    let model = match std::env::args().nth(2) {
        Some(path) => PathBuf::from(path),
        None => transcribe::find_models_dir()
            .expect("models/ directory not found")
            .join(transcribe::download::SUMMARY_MODEL_FILE),
    };

    let transcript = std::fs::read_to_string(&text)?;
    let vocabulary = std::fs::read_to_string(transcribe::vocabulary_path()).unwrap_or_default();
    let summary = transcribe::summarize::summarize(&model, &transcript, "", &vocabulary, &|partial| {
        eprint!("\r{} chars ...", partial.len());
        let _ = std::io::stderr().flush();
    })?;
    eprintln!();
    println!("{summary}");
    Ok(())
}
