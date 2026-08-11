//! Enroll a speaker from the command line (the GUI has this built in):
//!   cargo run --release --example enroll -- sample.m4a "Jane"
//! Appends/replaces the voice in speakers.json; ~10s of clear speech works best.

use std::path::Path;

fn main() -> anyhow::Result<()> {
    let sample = std::env::args().nth(1).expect("usage: enroll <audio> <name>");
    let name = std::env::args().nth(2).expect("usage: enroll <audio> <name>");

    let models = transcribe::find_models_dir().expect("models/ directory not found");
    let samples = transcribe::decode_to_mono_16k(Path::new(&sample))?;
    let embedding = transcribe::diarize::voice_embedding(
        &samples,
        16_000,
        &models.join("segmentation-3.0.onnx"),
        &models.join("wespeaker_en_voxceleb_CAM++.onnx"),
    )?;

    let path = transcribe::speakers_path();
    let mut profiles = transcribe::load_speaker_profiles(&path)?;
    profiles.retain(|p| p.name != name);
    profiles.push(transcribe::diarize::SpeakerProfile { name: name.clone(), embedding });
    transcribe::save_speaker_profiles(&path, &profiles)?;
    println!("enrolled {name} in {} ({} profile(s) total)", path.display(), profiles.len());
    Ok(())
}
