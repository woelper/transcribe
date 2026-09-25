//! Unattended capture: watch a live input, group what it hears into
//! meetings, transcribe them as they run, and file each finished one away.
//!
//! Two stages decide what is speech. Live, a cheap level gate with an
//! adaptive noise floor tracks whether anyone is talking — that is what
//! starts a meeting, cuts chunks at pauses, and ends a meeting after a
//! long enough silence. Then each chunk handed to the transcriber gets
//! the real neural VAD ([`crate::Options::vad_model`]), which drops the
//! non-speech inside it so whisper doesn't invent sentences for it.
//!
//! Everything here counts in samples at the *device's* rate rather than
//! in wall-clock time: audio arrives in bursts whenever the UI thread
//! gets around to handing it over, and sample counts stay true regardless.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{Options, Progress, transcripts};

/// How the segmenter decides where a meeting begins and ends.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Silence that ends a meeting. A minute by default, so a pause for
    /// reading, thinking, or a slide change doesn't split one in two.
    pub gap_secs: f64,
    /// Speech needed within [`Config::start_window_secs`] to start a
    /// meeting — long enough that a door slam or a cough doesn't.
    pub start_secs: f64,
    /// The window that speech is added up over to start a meeting. It
    /// spans the commas: "Good morning everyone, thanks for joining" is
    /// several short runs, and a meeting has to begin at the first of
    /// them rather than at whichever one happens to cross the threshold.
    pub start_window_secs: f64,
    /// Speech a meeting must contain in total to be worth keeping.
    pub min_meeting_secs: f64,
    /// Audio to hold before cutting a chunk at the next pause.
    pub chunk_secs: f64,
    /// Cut even without a pause once a chunk reaches this — someone can
    /// talk without a real break for a long time.
    pub max_chunk_secs: f64,
    /// Silence that counts as a pause to cut a chunk at.
    pub pause_secs: f64,
    /// Audio kept before speech starts, so the first word survives the
    /// gate's reaction time.
    pub preroll_secs: f64,
    /// Silence kept after the last speech of a meeting.
    pub tail_secs: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gap_secs: 60.0,
            start_secs: 1.2,
            start_window_secs: 6.0,
            min_meeting_secs: 8.0,
            chunk_secs: 45.0,
            max_chunk_secs: 150.0,
            pause_secs: 1.0,
            preroll_secs: 2.0,
            tail_secs: 1.0,
        }
    }
}

/// Level gate: is anyone talking right now?
///
/// The floor tracks background noise — falling fast so a quieter room is
/// noticed at once, rising only while nobody is talking so a monologue
/// can't raise the bar on itself. Speech opens the gate well above the
/// floor and holds it open lower down, so the quiet tail of a sentence
/// doesn't clip.
pub struct SpeechGate {
    frame: usize,
    floor: f32,
    open: bool,
    /// Samples of the last, incomplete frame of the previous block.
    carry: Vec<f32>,
}

/// Root-mean-square amplitude — loudness of one frame.
fn rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f32 = frame.iter().map(|s| s * s).sum();
    (sum / frame.len() as f32).sqrt()
}

impl SpeechGate {
    /// Frames are 20 ms: short enough to place a pause precisely, long
    /// enough that one loud click can't open the gate on its own.
    const FRAME_SECS: f64 = 0.020;
    /// Speech starts this far above the noise floor ...
    const OPEN_OVER_FLOOR: f32 = 3.5;
    /// ... and holds until it drops to here, so trailing syllables stay in.
    const CLOSE_OVER_FLOOR: f32 = 1.8;
    /// Absolute floor under which nothing counts as speech, however quiet
    /// the room is — a silent line has a noise floor near zero, and
    /// everything would be "3.5x the floor".
    const SILENCE: f32 = 0.004;

    pub fn new(rate: u32) -> Self {
        Self {
            frame: ((rate as f64) * Self::FRAME_SECS).round().max(1.0) as usize,
            floor: Self::SILENCE,
            open: false,
            carry: Vec::new(),
        }
    }

    pub fn frame_len(&self) -> usize {
        self.frame
    }

    /// Gate one block, appending a verdict per whole frame it contains.
    /// Leftover samples are held for the next call, so frames never drift.
    pub fn push(&mut self, block: &[f32], out: &mut Vec<Frame>) {
        let mut rest = block;
        while !rest.is_empty() {
            let want = self.frame - self.carry.len();
            let take = want.min(rest.len());
            self.carry.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.carry.len() < self.frame {
                break;
            }
            let level = rms(&self.carry);
            self.carry.clear();
            out.push(self.decide(level));
        }
    }

    fn decide(&mut self, level: f32) -> Frame {
        // Track the floor down always, up only while the room is quiet.
        if level < self.floor {
            self.floor += (level - self.floor) * 0.3;
        } else if !self.open {
            self.floor += (level - self.floor) * 0.01;
        }
        self.floor = self.floor.max(1e-6);
        let open_at = (self.floor * Self::OPEN_OVER_FLOOR).max(Self::SILENCE);
        let close_at = (self.floor * Self::CLOSE_OVER_FLOOR).max(Self::SILENCE * 0.6);
        self.open = if self.open { level > close_at } else { level > open_at };
        Frame {
            voiced: self.open,
            level,
        }
    }
}

/// One gated frame of audio.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub voiced: bool,
    pub level: f32,
}

/// What the segmenter wants done with the audio it has grouped.
#[derive(Debug)]
pub enum Event {
    MeetingStarted,
    /// Audio to transcribe and append to the running meeting, at the
    /// device's sample rate.
    Chunk(Vec<f32>),
    MeetingEnded {
        /// Wall time from the first to the last word.
        secs: f64,
        /// How much of that was actually speech.
        speech_secs: f64,
        /// False when it was too short to be a meeting — a passing
        /// remark, a phone notification — and should be thrown away.
        keep: bool,
    },
}

/// Groups a live input into meetings. Pure: audio in, [`Event`]s out,
/// which makes the whole policy testable without a microphone.
pub struct Segmenter {
    cfg: Config,
    rate: f64,
    gate: SpeechGate,
    /// Audio not yet handed to the transcriber.
    pending: Vec<f32>,
    /// Absolute index of `pending[0]` in the stream.
    pending_start: u64,
    /// How much audio has been handed to the gate, absolute. Ahead of
    /// `analyzed` by the partial frame the gate is still holding, and the
    /// position to read from — feeding those carried samples a second
    /// time would both double-count them and misalign every frame after.
    fed: u64,
    /// How far the gate has looked, in whole frames, absolute. Cuts are
    /// measured against this, so it must never run past what arrived.
    analyzed: u64,
    /// End of the most recent voiced frame, absolute.
    speech_end: u64,
    /// Whether the last frame was voiced — what the UI's talking light
    /// shows.
    voiced_now: bool,
    /// The last [`Config::start_window_secs`] of frames: `(end position,
    /// voiced)`. Deciding over a window rather than one run is what makes
    /// a meeting start at the first word instead of the third.
    window: VecDeque<(u64, bool)>,
    /// Quietest recent frames, to cut a long chunk at the least bad spot.
    quiet: VecDeque<(u64, f32)>,
    meeting: Option<Meeting>,
}

struct Meeting {
    start: u64,
    speech: u64,
}

impl Segmenter {
    pub fn new(rate: u32, cfg: Config) -> Self {
        Self {
            cfg,
            rate: rate as f64,
            gate: SpeechGate::new(rate),
            pending: Vec::new(),
            pending_start: 0,
            fed: 0,
            analyzed: 0,
            speech_end: 0,
            voiced_now: false,
            window: VecDeque::new(),
            quiet: VecDeque::new(),
            meeting: None,
        }
    }

    fn samples(&self, secs: f64) -> u64 {
        (secs * self.rate) as u64
    }

    /// Total samples seen, including what has been handed out already.
    fn total(&self) -> u64 {
        self.pending_start + self.pending.len() as u64
    }

    pub fn in_meeting(&self) -> bool {
        self.meeting.is_some()
    }

    /// Seconds of the meeting so far (0 when none is running).
    pub fn meeting_secs(&self) -> f64 {
        self.meeting
            .as_ref()
            .map_or(0.0, |m| (self.total() - m.start) as f64 / self.rate)
    }

    /// Seconds since anyone last spoke.
    pub fn silence_secs(&self) -> f64 {
        (self.total().saturating_sub(self.speech_end)) as f64 / self.rate
    }

    pub fn voiced(&self) -> bool {
        self.voiced_now
    }

    /// Speech in the start window, and where the first of it began.
    fn window_speech(&self) -> (u64, Option<u64>) {
        let frame = self.gate.frame_len() as u64;
        let speech = self.window.iter().filter(|(_, voiced)| *voiced).count() as u64 * frame;
        let first = self
            .window
            .iter()
            .find(|(_, voiced)| *voiced)
            .map(|(at, _)| at.saturating_sub(frame));
        (speech, first)
    }

    /// Feed the next block of captured audio.
    pub fn push(&mut self, block: &[f32], out: &mut Vec<Event>) {
        self.pending.extend_from_slice(block);
        let mut frames = Vec::new();
        // Gate only what the gate has not already been given.
        let from = (self.fed - self.pending_start) as usize;
        self.gate.push(&self.pending[from..], &mut frames);
        self.fed = self.total();
        let frame_len = self.gate.frame_len() as u64;
        for frame in frames {
            let frame_end = self.analyzed + frame_len;
            self.analyzed = frame_end;
            self.quiet.push_back((frame_end, frame.level));
            // Two seconds of history is enough to find a decent cut.
            while self.quiet.len() > (2.0 / SpeechGate::FRAME_SECS) as usize {
                self.quiet.pop_front();
            }
            self.window.push_back((frame_end, frame.voiced));
            let window = self.samples(self.cfg.start_window_secs);
            while self.window.front().is_some_and(|(at, _)| *at + window <= frame_end) {
                self.window.pop_front();
            }
            self.voiced_now = frame.voiced;
            if frame.voiced {
                self.speech_end = frame_end;
                if let Some(meeting) = &mut self.meeting {
                    meeting.speech += frame_len;
                }
            }
            self.step(out);
        }
        self.trim_idle();
        // pending_start <= analyzed <= fed <= total. Every cut is measured
        // against `analyzed`, so drift here corrupts positions rather than
        // failing outright.
        debug_assert!(self.pending_start <= self.analyzed);
        debug_assert!(self.analyzed <= self.fed);
        debug_assert!(self.fed <= self.total());
    }

    /// Decide what the frames so far add up to.
    fn step(&mut self, out: &mut Vec<Event>) {
        match self.meeting {
            None => {
                // Enough speech in the window: a meeting starts, backdated
                // to just before its first word.
                let (speech, first) = self.window_speech();
                if speech < self.samples(self.cfg.start_secs) {
                    return;
                }
                let Some(first) = first else { return };
                let start = first.saturating_sub(self.samples(self.cfg.preroll_secs));
                let start = start.max(self.pending_start);
                self.drop_before(start);
                self.meeting = Some(Meeting { start, speech });
                out.push(Event::MeetingStarted);
            }
            Some(_) => {
                let silence = self.analyzed.saturating_sub(self.speech_end);
                if silence >= self.samples(self.cfg.gap_secs) {
                    self.end_meeting(out);
                    return;
                }
                let pending = self.analyzed - self.pending_start;
                // Nothing but silence since the last chunk: there is
                // nothing to decode, however much of it piles up.
                if self.speech_end <= self.pending_start {
                    return;
                }
                // Cut at a pause once there is enough to be worth a pass ...
                if pending >= self.samples(self.cfg.chunk_secs)
                    && silence >= self.samples(self.cfg.pause_secs)
                {
                    let cut = self.speech_end + self.samples(self.cfg.tail_secs);
                    self.cut(cut.min(self.analyzed), out);
                } else if pending >= self.samples(self.cfg.max_chunk_secs) {
                    // ... or, with no pause in sight, at the quietest
                    // recent frame, which is the least bad place to split
                    // a sentence.
                    // Only frames still inside `pending` can be cut at; an
                    // older one would clamp away to a no-op cut.
                    let cut = self
                        .quiet
                        .iter()
                        .filter(|(at, _)| *at > self.pending_start)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map_or(self.analyzed, |(at, _)| *at);
                    self.cut(cut, out);
                }
            }
        }
    }

    fn end_meeting(&mut self, out: &mut Vec<Event>) {
        let Some(meeting) = self.meeting.take() else { return };
        let end = (self.speech_end + self.samples(self.cfg.tail_secs)).min(self.analyzed);
        self.cut(end, out);
        let speech_secs = meeting.speech as f64 / self.rate;
        out.push(Event::MeetingEnded {
            secs: (end - meeting.start) as f64 / self.rate,
            speech_secs,
            keep: speech_secs >= self.cfg.min_meeting_secs,
        });
        // Whatever silence followed is only pre-roll for the next meeting.
        self.trim_idle();
    }

    /// Hand `pending` up to `end` to the transcriber. Never past what the
    /// gate has looked at, so `pending_start <= analyzed` always holds.
    fn cut(&mut self, end: u64, out: &mut Vec<Event>) {
        let end = end.clamp(self.pending_start, self.analyzed);
        let take = (end - self.pending_start) as usize;
        if take == 0 {
            return;
        }
        let rest = self.pending.split_off(take);
        let chunk = std::mem::replace(&mut self.pending, rest);
        self.pending_start = end;
        out.push(Event::Chunk(chunk));
    }

    /// Throw away audio before `at` (never anything the gate hasn't seen).
    fn drop_before(&mut self, at: u64) {
        let at = at.clamp(self.pending_start, self.analyzed);
        let drop = (at - self.pending_start) as usize;
        if drop > 0 {
            self.pending.drain(..drop);
            self.pending_start = at;
        }
    }

    /// Between meetings, keep only enough audio to serve as pre-roll.
    fn trim_idle(&mut self) {
        if self.meeting.is_some() {
            return;
        }
        // Keep any speech still inside the start window — it may yet grow
        // into a meeting, and then the audio has to be there — plus the
        // pre-roll ahead of it.
        let (_, first) = self.window_speech();
        let keep_from = first
            .unwrap_or(self.analyzed)
            .saturating_sub(self.samples(self.cfg.preroll_secs));
        self.drop_before(keep_from);
    }

    /// Stopping: finish any meeting in progress rather than losing it.
    pub fn finish(&mut self, out: &mut Vec<Event>) {
        if self.meeting.is_some() {
            // Everything captured counts as analyzed now, even a partial
            // trailing frame — and if the last one was mid-sentence, that
            // sentence counts as speech to the very end.
            self.analyzed = self.total();
            self.fed = self.analyzed;
            if self.voiced_now {
                self.speech_end = self.analyzed;
            }
            self.end_meeting(out);
        }
    }
}

/// What auto mode is doing, for the UI to read each frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase {
    /// Waiting for someone to start talking.
    Listening,
    /// A meeting is running.
    Recording,
    /// Catching up on a chunk of it.
    Transcribing,
    /// Titling and filing a finished meeting.
    Saving,
    /// The worker is gone; nothing is being captured.
    Stopped,
}

/// Shared between the auto worker and the UI thread.
#[derive(Debug)]
pub struct Status {
    pub phase: Phase,
    /// Length of the meeting in progress.
    pub meeting_secs: f64,
    /// How long the room has been quiet.
    pub silence_secs: f64,
    /// Whether the gate hears speech right now.
    pub voiced: bool,
    /// Transcript of the meeting in progress, as far as it is decoded.
    pub transcript: String,
    /// Meetings filed this session, newest last.
    pub saved: Vec<PathBuf>,
    /// One line about what just happened.
    pub note: String,
    /// Set when the worker gave up; auto mode switches itself off.
    pub error: Option<String>,
}

impl Default for Status {
    fn default() -> Self {
        Self {
            phase: Phase::Listening,
            meeting_secs: 0.0,
            silence_secs: 0.0,
            voiced: false,
            transcript: String::new(),
            saved: Vec::new(),
            note: String::new(),
            error: None,
        }
    }
}

/// Everything the worker needs to transcribe and file what it hears.
pub struct Settings {
    pub cfg: Config,
    /// Speech model, VAD model, prompt — as for a manual transcription.
    pub model: PathBuf,
    pub vad_model: Option<PathBuf>,
    pub language: String,
    pub prompt: String,
    pub context: String,
    /// Chat model that titles and summarizes a finished meeting. Without
    /// it, meetings are filed under their date alone.
    pub summary_model: Option<PathBuf>,
    pub vocabulary: String,
    /// Where finished meetings are written.
    pub dir: PathBuf,
}

fn lock<T>(shared: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Run auto mode until `rx` closes, then finish the meeting in progress
/// and return. Owns all the slow work — resampling, transcription,
/// titling — so the UI thread only has to hand over audio.
pub fn run(rx: &Receiver<Vec<f32>>, rate: u32, mut settings: Settings, status: &Arc<Mutex<Status>>) {
    // The VAD model is tiny and keeps whisper from inventing sentences for
    // the quiet parts; fetch it on first use. Offline, carry on without it
    // — the level gate has already cut most of the silence away.
    if let Some(vad) = settings.vad_model.clone().filter(|p| !p.exists()) {
        lock(status).note = "downloading the voice activity model ...".into();
        if let Err(e) = crate::download::download(crate::download::VAD_MODEL_URL, &vad, |_, _| {}) {
            log::error!("auto: fetching the VAD model failed: {e:#}");
            settings.vad_model = None;
        }
        lock(status).note.clear();
    }
    let mut segmenter = Segmenter::new(rate, settings.cfg);
    let mut events: Vec<Event> = Vec::new();
    let mut transcript = String::new();
    let mut started = String::new();
    loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(block) => {
                segmenter.push(&block, &mut events);
                // Take whatever else is queued before doing slow work.
                while let Ok(block) = rx.try_recv() {
                    segmenter.push(&block, &mut events);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                segmenter.finish(&mut events);
                handle(&mut events, &mut transcript, &mut started, rate, &settings, status);
                let mut status = lock(status);
                status.phase = Phase::Stopped;
                status.meeting_secs = 0.0;
                return;
            }
        }
        handle(&mut events, &mut transcript, &mut started, rate, &settings, status);
        let mut shared = lock(status);
        shared.meeting_secs = segmenter.meeting_secs();
        shared.silence_secs = segmenter.silence_secs();
        shared.voiced = segmenter.voiced();
        if shared.phase != Phase::Stopped {
            shared.phase = if segmenter.in_meeting() {
                Phase::Recording
            } else {
                Phase::Listening
            };
        }
    }
}

/// Act on the segmenter's events: transcribe chunks, file finished
/// meetings. Runs on the worker thread, so it may take its time.
fn handle(
    events: &mut Vec<Event>,
    transcript: &mut String,
    started: &mut String,
    rate: u32,
    settings: &Settings,
    status: &Arc<Mutex<Status>>,
) {
    for event in events.drain(..) {
        match event {
            Event::MeetingStarted => {
                transcript.clear();
                *started = now();
                log::info!("auto: meeting started");
                let mut status = lock(status);
                status.transcript.clear();
                status.note = "meeting started".into();
            }
            Event::Chunk(chunk) => {
                lock(status).phase = Phase::Transcribing;
                match transcribe_chunk(&chunk, rate, settings, status) {
                    Ok(text) if text.trim().is_empty() => {}
                    Ok(text) => {
                        if !transcript.is_empty() {
                            transcript.push('\n');
                        }
                        transcript.push_str(text.trim());
                        lock(status).transcript.clone_from(transcript);
                    }
                    Err(e) => {
                        log::error!("auto: transcribing a chunk failed: {e:#}");
                        lock(status).note = format!("error: {e:#}");
                    }
                }
            }
            Event::MeetingEnded { secs, speech_secs, keep } => {
                log::info!(
                    "auto: meeting ended after {secs:.0}s ({speech_secs:.0}s of speech, keep {keep})"
                );
                if !keep || transcript.trim().is_empty() {
                    lock(status).note = format!(
                        "ignored {}s of noise",
                        secs.round() as i64
                    );
                    transcript.clear();
                    lock(status).transcript.clear();
                    continue;
                }
                lock(status).phase = Phase::Saving;
                let meta = describe(transcript, started, secs, settings, status);
                match transcripts::save(&settings.dir, &meta, transcript) {
                    Ok(path) => {
                        log::info!("auto: saved {}", path.display());
                        let mut status = lock(status);
                        status.note = format!("saved \"{}\"", meta.title);
                        status.saved.push(path);
                    }
                    Err(e) => {
                        log::error!("auto: saving failed: {e:#}");
                        lock(status).note = format!("error: saving failed: {e:#}");
                    }
                }
                transcript.clear();
                lock(status).transcript.clear();
            }
        }
    }
}

/// Resample a chunk to 16 kHz and run it through the speech model.
fn transcribe_chunk(
    chunk: &[f32],
    rate: u32,
    settings: &Settings,
    status: &Arc<Mutex<Status>>,
) -> anyhow::Result<String> {
    let samples = if rate as usize == crate::WHISPER_SAMPLE_RATE {
        chunk.to_vec()
    } else {
        crate::resample_to_16k(chunk, rate as usize)?
    };
    let opts = Options {
        model: settings.model.clone(),
        vad_model: settings.vad_model.clone(),
        language: settings.language.clone(),
        prompt: settings.prompt.clone(),
        context: settings.context.clone(),
        // Speaker detection would have to re-cluster every chunk against
        // every other one; a finished transcript can still be diarized
        // by hand.
        diarize: None,
        timestamps: false,
        ..Options::default()
    };
    let secs = samples.len() as f64 / crate::WHISPER_SAMPLE_RATE as f64;
    log::info!("auto: transcribing a {secs:.0}s chunk");
    let live = status.clone();
    let transcript = crate::transcribe_samples(samples, &opts, move |progress| {
        if let Progress::Transcribing { percent } = progress {
            lock(&live).note = format!("transcribing the last {secs:.0}s ({percent}%)");
        }
    })?;
    Ok(transcript.text)
}

/// Title and summarize a finished meeting. Without a summary model — or
/// if it fails — the meeting still gets filed, under its date alone.
fn describe(
    transcript: &str,
    started: &str,
    secs: f64,
    settings: &Settings,
    status: &Arc<Mutex<Status>>,
) -> transcripts::Meta {
    let mut meta = transcripts::Meta {
        date: started.to_owned(),
        duration: crate::format_duration(secs),
        ..transcripts::Meta::default()
    };
    let Some(model) = settings.summary_model.as_ref().filter(|p| p.exists()) else {
        return meta;
    };
    lock(status).note = "naming the meeting ...".into();
    match crate::summarize::label(model, transcript, &settings.vocabulary) {
        Ok(label) => {
            meta.title = label.title;
            meta.summary = label.summary;
        }
        Err(e) => log::error!("auto: labelling failed: {e:#}"),
    }
    meta
}

/// Local date and time, `YYYY-MM-DD HH:MM`.
fn now() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    /// `secs` of a loud-ish tone: what the gate should hear as speech.
    fn speech(secs: f64) -> Vec<f32> {
        let n = (secs * RATE as f64) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                // Amplitude-modulated, like syllables, well above the gate.
                0.3 * (t * 220.0 * std::f32::consts::TAU).sin()
                    * (0.6 + 0.4 * (t * 3.0 * std::f32::consts::TAU).sin())
            })
            .collect()
    }

    /// `secs` of a quiet room — not digital silence, which is unrealistic.
    fn quiet(secs: f64) -> Vec<f32> {
        let n = (secs * RATE as f64) as usize;
        (0..n)
            .map(|i| 0.0005 * ((i as f32 * 0.37).sin() + (i as f32 * 1.7).cos()))
            .collect()
    }

    fn feed(seg: &mut Segmenter, audio: &[f32], out: &mut Vec<Event>) {
        // In blocks, the way a capture stream delivers it.
        for block in audio.chunks(1024) {
            seg.push(block, out);
        }
    }

    fn secs_of(events: &[Event]) -> f64 {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Chunk(c) => Some(c.len() as f64 / RATE as f64),
                _ => None,
            })
            .sum()
    }

    #[test]
    fn the_gate_separates_speech_from_a_quiet_room() {
        let mut gate = SpeechGate::new(RATE);
        let mut frames = Vec::new();
        gate.push(&quiet(2.0), &mut frames);
        assert!(frames.iter().all(|f| !f.voiced), "a quiet room is not speech");
        frames.clear();
        gate.push(&speech(1.0), &mut frames);
        let voiced = frames.iter().filter(|f| f.voiced).count();
        assert!(voiced > frames.len() * 8 / 10, "{voiced} of {} frames voiced", frames.len());
    }

    /// The whole point of the feature: a pause in the middle of a meeting
    /// must not split it, but a long silence has to end it.
    #[test]
    fn a_minute_of_silence_does_not_split_a_meeting() {
        let cfg = Config { gap_secs: 60.0, ..Config::default() };
        let mut seg = Segmenter::new(RATE, cfg);
        let mut events = Vec::new();
        feed(&mut seg, &speech(20.0), &mut events);
        feed(&mut seg, &quiet(55.0), &mut events);
        feed(&mut seg, &speech(20.0), &mut events);
        assert!(
            !events.iter().any(|e| matches!(e, Event::MeetingEnded { .. })),
            "a 55s pause ended the meeting"
        );
        assert_eq!(
            events.iter().filter(|e| matches!(e, Event::MeetingStarted)).count(),
            1,
            "the meeting restarted instead of continuing"
        );
        feed(&mut seg, &quiet(70.0), &mut events);
        let ended = events.iter().find_map(|e| match e {
            Event::MeetingEnded { keep, speech_secs, .. } => Some((*keep, *speech_secs)),
            _ => None,
        });
        let (keep, speech_secs) = ended.expect("70s of silence did not end the meeting");
        assert!(keep, "a 40s meeting should be kept");
        assert!(speech_secs > 30.0, "only {speech_secs:.0}s of speech counted");
    }

    #[test]
    fn a_cough_never_starts_a_meeting() {
        let mut seg = Segmenter::new(RATE, Config::default());
        let mut events = Vec::new();
        feed(&mut seg, &quiet(2.0), &mut events);
        feed(&mut seg, &speech(0.3), &mut events);
        feed(&mut seg, &quiet(5.0), &mut events);
        assert!(events.is_empty(), "{events:?}");
    }

    /// Brief chatter does start a meeting, but is thrown away rather than
    /// filed as one.
    #[test]
    fn a_passing_remark_is_recorded_but_not_kept() {
        let cfg = Config { gap_secs: 3.0, min_meeting_secs: 8.0, ..Config::default() };
        let mut seg = Segmenter::new(RATE, cfg);
        let mut events = Vec::new();
        feed(&mut seg, &speech(2.0), &mut events);
        feed(&mut seg, &quiet(5.0), &mut events);
        let keep = events.iter().find_map(|e| match e {
            Event::MeetingEnded { keep, .. } => Some(*keep),
            _ => None,
        });
        assert_eq!(keep, Some(false));
    }

    /// Long meetings are handed over in pieces as they run, so the text
    /// appears while people are still talking — and memory stays bounded.
    #[test]
    fn a_long_meeting_is_transcribed_in_chunks_as_it_runs() {
        let cfg = Config { chunk_secs: 10.0, pause_secs: 1.0, ..Config::default() };
        let mut seg = Segmenter::new(RATE, cfg);
        let mut events = Vec::new();
        for _ in 0..4 {
            feed(&mut seg, &speech(12.0), &mut events);
            feed(&mut seg, &quiet(2.0), &mut events);
        }
        let chunks = events.iter().filter(|e| matches!(e, Event::Chunk(_))).count();
        assert!(chunks >= 3, "only {chunks} chunk(s) while the meeting ran");
        // Memory stays bounded: only the undecoded tail is held.
        let held = seg.pending.len() as f64 / RATE as f64;
        assert!(held < 15.0, "holding {held:.1}s after chunking");
    }

    /// Without any pause to cut at, a chunk still has to be handed over.
    #[test]
    fn unbroken_speech_is_cut_at_the_limit() {
        let cfg = Config { chunk_secs: 60.0, max_chunk_secs: 20.0, ..Config::default() };
        let mut seg = Segmenter::new(RATE, cfg);
        let mut events = Vec::new();
        feed(&mut seg, &speech(50.0), &mut events);
        let chunks: Vec<f64> = events
            .iter()
            .filter_map(|e| match e {
                Event::Chunk(c) => Some(c.len() as f64 / RATE as f64),
                _ => None,
            })
            .collect();
        assert!(chunks.len() >= 2, "{chunks:?}");
        assert!(chunks.iter().all(|&s| s <= 21.0), "{chunks:?}");
    }

    /// Stopping mid-meeting keeps what was said instead of dropping it.
    #[test]
    fn stopping_finishes_the_meeting_in_progress() {
        let mut seg = Segmenter::new(RATE, Config::default());
        let mut events = Vec::new();
        feed(&mut seg, &speech(20.0), &mut events);
        seg.finish(&mut events);
        assert!(matches!(events.last(), Some(Event::MeetingEnded { keep: true, .. })));
        assert!(secs_of(&events) > 15.0, "only {}s handed over", secs_of(&events));
    }

    /// The first word must survive the gate's reaction time.
    #[test]
    fn a_meeting_keeps_the_audio_before_the_first_word() {
        let cfg = Config { preroll_secs: 2.0, ..Config::default() };
        let mut seg = Segmenter::new(RATE, cfg);
        let mut events = Vec::new();
        feed(&mut seg, &quiet(10.0), &mut events);
        feed(&mut seg, &speech(20.0), &mut events);
        seg.finish(&mut events);
        // 20s of speech plus up to 2s of pre-roll and 1s of tail.
        let handed = secs_of(&events);
        assert!((20.0..=23.5).contains(&handed), "handed over {handed:.1}s");
    }

    /// Nobody opens a meeting with one unbroken sentence: "Good morning
    /// everyone, thanks for joining" is three short bursts, and the
    /// meeting has to start at the first of them. Deciding on a single
    /// unbroken run instead used to cut the greeting off.
    #[test]
    fn a_meeting_starts_at_the_first_word_not_the_longest_run() {
        let cfg = Config { preroll_secs: 2.0, ..Config::default() };
        let mut seg = Segmenter::new(RATE, cfg);
        let mut events = Vec::new();
        feed(&mut seg, &quiet(3.0), &mut events);
        feed(&mut seg, &speech(0.6), &mut events); // "Good morning,"
        feed(&mut seg, &quiet(1.5), &mut events);
        feed(&mut seg, &speech(4.0), &mut events); // "... thanks for joining"
        seg.finish(&mut events);
        // The meeting runs from 2s of pre-roll before the first burst (at
        // 3.0s) to the end of the audio (9.1s): about 8.1s. Starting at
        // the long run instead would hand over only ~6s.
        let handed = secs_of(&events);
        assert!(handed >= 7.5, "only {handed:.1}s handed over — the opening was cut off");
    }

    /// A capture device delivers whatever block size it likes, and it is
    /// rarely a whole number of 20 ms frames — 4608 samples at 48 kHz is
    /// 4.8 of them. The leftover has to be accounted for exactly once:
    /// counting only whole frames while the gate also held the remainder
    /// used to walk the read position off the end of the buffer.
    #[test]
    fn odd_block_sizes_keep_the_counters_in_step() {
        for (rate, block) in [(48_000, 4608), (44_100, 1000), (16_000, 1024), (48_000, 480)] {
            // Cut often, so the read position is exercised right after a
            // chunk hand-off, when almost nothing is left pending.
            let cfg = Config {
                chunk_secs: 5.0,
                pause_secs: 0.5,
                gap_secs: 10.0,
                ..Config::default()
            };
            let mut seg = Segmenter::new(rate, cfg);
            let mut events = Vec::new();
            let push = |seg: &mut Segmenter, audio: &[f32], events: &mut Vec<Event>| {
                for chunk in audio.chunks(block) {
                    seg.push(chunk, events);
                }
            };
            // Long enough to idle-trim many times over, then talk.
            let quiet_at = |secs: f64| vec![0.0005f32; (secs * rate as f64) as usize];
            let speech_at = |secs: f64| {
                (0..(secs * rate as f64) as usize)
                    .map(|i| 0.3 * (i as f32 * 0.05).sin())
                    .collect::<Vec<f32>>()
            };
            push(&mut seg, &quiet_at(30.0), &mut events);
            // Talking with pauses, the way a meeting actually goes.
            for _ in 0..6 {
                push(&mut seg, &speech_at(8.0), &mut events);
                push(&mut seg, &quiet_at(2.0), &mut events);
            }
            push(&mut seg, &quiet_at(30.0), &mut events);
            seg.finish(&mut events);
            assert!(
                seg.analyzed <= seg.pending_start + seg.pending.len() as u64,
                "{rate} Hz / {block}: gated {} samples of {} received",
                seg.analyzed,
                seg.pending_start + seg.pending.len() as u64
            );
            // Frames stayed aligned with the audio, so the meeting was
            // still heard, chunked, and closed.
            let chunks = events.iter().filter(|e| matches!(e, Event::Chunk(_))).count();
            let ended = events
                .iter()
                .any(|e| matches!(e, Event::MeetingEnded { keep: true, .. }));
            assert!(chunks >= 2 && ended, "{rate} Hz / {block}: {chunks} chunk(s), ended {ended}");
        }
    }

    /// Idle listening must not grow without bound.
    #[test]
    fn listening_keeps_only_the_preroll() {
        let mut seg = Segmenter::new(RATE, Config::default());
        let mut events = Vec::new();
        for _ in 0..30 {
            feed(&mut seg, &quiet(10.0), &mut events);
        }
        assert!(events.is_empty());
        let held = seg.pending.len() as f64 / RATE as f64;
        assert!(held <= 2.5, "holding {held:.1}s of idle audio");
    }
}
