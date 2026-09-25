//! Summarize a transcript from the command line (the GUI has this built in):
//!   cargo run --release --example summarize -- transcript.txt [model.gguf]
//! Without a model argument, uses the summary model from the models/
//! directory. With `--label`, prints the title and one-line summary auto
//! mode names a meeting with instead of the long summary.

use std::io::Write;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    transcribe::applog::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let label_only = args.iter().any(|a| a == "--label");
    let mut positional = args.iter().filter(|a| !a.starts_with("--"));
    let text = positional
        .next()
        .expect("usage: summarize [--label] <transcript.txt> [model.gguf]");
    let model = match positional.next() {
        Some(path) => PathBuf::from(path),
        None => transcribe::find_models_dir()
            .expect("models/ directory not found")
            .join(transcribe::download::SUMMARY_MODEL_FILE),
    };

    let transcript = std::fs::read_to_string(text)?;
    let vocabulary = std::fs::read_to_string(transcribe::vocabulary_path()).unwrap_or_default();
    if label_only {
        let started = std::time::Instant::now();
        let label = transcribe::summarize::label(&model, &transcript, &vocabulary)?;
        eprintln!("labelled in {:.1}s", started.elapsed().as_secs_f64());
        println!("Title:   {}", label.title);
        println!("Summary: {}", label.summary);
        return Ok(());
    }
    let summary = transcribe::summarize::summarize(&model, &transcript, "", &vocabulary, &|partial| {
        eprint!("\r{} chars ...", partial.len());
        let _ = std::io::stderr().flush();
    })?;
    eprintln!();
    println!("{summary}");
    Ok(())
}
