//! Drive auto mode from an audio file instead of a microphone — the way
//! to check the whole unattended path (meeting detection, chunked
//! transcription, naming, filing) without talking to a laptop for an hour:
//!
//!   cargo run --release --example auto-demo -- meeting.wav [models/ggml-tiny.bin]
//!
//! Transcripts land in a `transcripts/` directory next to models/, exactly
//! as they do when the app is listening.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use transcribe::auto;

fn main() -> anyhow::Result<()> {
    // Auto mode reports what it does through the log; without this the
    // interesting failures would be invisible here.
    if let Some(path) = transcribe::applog::init() {
        println!("logging to {}", path.display());
    }
    let audio = std::env::args()
        .nth(1)
        .expect("usage: auto-demo <audio> [model.bin]");
    let models = transcribe::find_models_dir().expect("models/ directory not found");
    let model = std::env::args()
        .nth(2)
        .map_or_else(|| models.join("ggml-tiny.bin"), PathBuf::from);

    // Decoded to 16 kHz, then fed in 100 ms blocks like a capture stream.
    let samples = transcribe::decode_to_mono_16k(std::path::Path::new(&audio))?;
    let rate = transcribe::WHISPER_SAMPLE_RATE as u32;
    println!(
        "feeding {:.0}s of audio through auto mode ({} at {rate} Hz)",
        samples.len() as f64 / rate as f64,
        model.display()
    );

    let dir = transcribe::transcripts::transcripts_dir();
    let settings = auto::Settings {
        cfg: auto::Config::default(),
        model,
        vad_model: Some(models.join(transcribe::download::VAD_MODEL_FILE)),
        language: "auto".into(),
        prompt: transcribe::DEFAULT_PROMPT.into(),
        context: String::new(),
        summary_model: Some(models.join(transcribe::download::SUMMARY_MODEL_FILE)),
        vocabulary: String::new(),
        dir: dir.clone(),
    };

    let status = Arc::new(Mutex::new(auto::Status::default()));
    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let worker = {
        let status = status.clone();
        std::thread::spawn(move || auto::run(&rx, rate, settings, &status))
    };

    let before = transcribe::transcripts::list(&dir).len();
    // Real time would take as long as the meeting; a block every
    // millisecond exercises the same path with the same sample counts.
    // The block size is deliberately not a whole number of gate frames,
    // which is what a capture device delivers.
    for block in samples.chunks(rate as usize / 17) {
        tx.send(block.to_vec())?;
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    drop(tx);
    println!("audio finished, waiting for the worker ...");
    worker.join().expect("the auto worker panicked");

    let status = status.lock().unwrap();
    println!("\nnote: {}", status.note);
    for saved in transcribe::transcripts::list(&dir).into_iter().take(
        transcribe::transcripts::list(&dir).len().saturating_sub(before),
    ) {
        println!("\n--- {} ---", saved.path.display());
        println!("{}", std::fs::read_to_string(&saved.path)?);
    }
    Ok(())
}
