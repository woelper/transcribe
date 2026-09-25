use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};

use crate::{WHISPER_SAMPLE_RATE, resample_to_16k_with_progress};

/// Names of all audio input devices. System audio (a YouTube video, a Zoom
/// call) appears here once a loopback driver like BlackHole is installed.
pub fn input_devices() -> Vec<String> {
    cpal::default_host()
        .input_devices()
        .map(|devices| {
            devices
                .filter_map(|d| d.description().ok().map(|d| d.name().to_owned()))
                .collect()
        })
        .unwrap_or_default()
}

/// Records mono audio from an input device until stopped.
pub struct Recorder {
    // Held only to keep the stream alive; dropped on stop.
    _stream: cpal::Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    rate: u32,
    error: Arc<Mutex<Option<String>>>,
    level: Arc<Mutex<f32>>,
    /// Everything ever captured, including what [`Recorder::take`] already
    /// handed out — so the duration stays right in auto mode, which
    /// consumes the buffer as it goes.
    captured: Arc<AtomicU64>,
}

impl Recorder {
    /// Start recording from the named device (None = system default input),
    /// buffering the audio until [`Recorder::stop`].
    pub fn start(device_name: Option<&str>) -> Result<Self> {
        Self::open(device_name, None)
    }

    /// Start recording and pass every captured block straight to `sink`
    /// instead of buffering it. Auto mode listens this way: the audio
    /// reaches its worker from the capture callback itself, so it keeps
    /// flowing while the window is minimized, occluded, or simply busy.
    ///
    /// The stream owns the sender, so dropping the recorder closes the
    /// channel — which is how the worker learns that listening stopped.
    pub fn start_streaming(
        device_name: Option<&str>,
        sink: std::sync::mpsc::Sender<Vec<f32>>,
    ) -> Result<Self> {
        Self::open(device_name, Some(sink))
    }

    fn open(device_name: Option<&str>, sink: Option<std::sync::mpsc::Sender<Vec<f32>>>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => host
                .input_devices()?
                .find(|d| d.description().is_ok_and(|d| d.name() == name))
                .with_context(|| format!("input device '{name}' not found"))?,
            None => host
                .default_input_device()
                .context("no default input device")?,
        };
        let config = device
            .default_input_config()
            .context("device has no input config")?;
        let rate = config.sample_rate();
        let channels = config.channels() as usize;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let error = Arc::new(Mutex::new(None));
        let level = Arc::new(Mutex::new(0.0));
        let captured = Arc::new(AtomicU64::new(0));

        use cpal::SampleFormat as SF;
        let build = |format| {
            let args = StreamArgs {
                channels,
                samples: &samples,
                error: &error,
                level: &level,
                captured: &captured,
                sink: sink.clone(),
            };
            match format {
                SF::F32 => build_stream::<f32>(&device, config.into(), args),
                SF::I16 => build_stream::<i16>(&device, config.into(), args),
                SF::I32 => build_stream::<i32>(&device, config.into(), args),
                SF::U16 => build_stream::<u16>(&device, config.into(), args),
                other => bail!("unsupported sample format {other}"),
            }
        };
        let stream = build(config.sample_format())?;
        stream.play().context("failed to start recording")?;

        Ok(Self {
            _stream: stream,
            samples,
            rate,
            error,
            level,
            captured,
        })
    }

    /// The device's capture rate. Auto mode resamples chunk by chunk, so it
    /// needs to know what it is recording at.
    pub fn sample_rate(&self) -> u32 {
        self.rate
    }

    /// Take everything captured since the last call, leaving the stream
    /// running. Auto mode pulls the input through this instead of letting
    /// hours of audio pile up in memory.
    pub fn take(&self) -> Vec<f32> {
        std::mem::take(&mut *self.samples.lock().unwrap())
    }

    /// Peak amplitude (0..=1) of the most recent capture buffer —
    /// a live signal indicator. Stays at 0.0 when no audio arrives
    /// (muted device, missing microphone permission, silent loopback).
    pub fn level(&self) -> f32 {
        *self.level.lock().unwrap()
    }

    pub fn duration_secs(&self) -> f64 {
        self.captured.load(Ordering::Relaxed) as f64 / self.rate as f64
    }

    /// A stream error, if one occurred (e.g. the device disappeared).
    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }

    /// Stop the stream and hand back the raw capture. Cheap; the costly
    /// resampling happens in [`Recording::into_16k`], which the caller can
    /// run on another thread (the audio stream itself can't leave this one).
    pub fn stop(self) -> Recording {
        drop(self._stream);
        let mono = std::mem::take(&mut *self.samples.lock().unwrap());
        Recording { mono, rate: self.rate }
    }
}

/// A finished capture at the device's sample rate.
pub struct Recording {
    mono: Vec<f32>,
    rate: u32,
}

impl Recording {
    pub fn duration_secs(&self) -> f64 {
        self.mono.len() as f64 / self.rate as f64
    }

    /// Convert to 16 kHz mono (whisper's input format), reporting progress
    /// 0..=1 — a long recording takes seconds to resample.
    pub fn into_16k(self, progress: &mut dyn FnMut(f32)) -> Result<Vec<f32>> {
        if self.mono.is_empty() || self.rate as usize == WHISPER_SAMPLE_RATE {
            progress(1.0);
            return Ok(self.mono);
        }
        resample_to_16k_with_progress(&self.mono, self.rate as usize, progress)
    }
}

/// The shared state a capture stream writes into.
struct StreamArgs<'a> {
    channels: usize,
    samples: &'a Arc<Mutex<Vec<f32>>>,
    error: &'a Arc<Mutex<Option<String>>>,
    level: &'a Arc<Mutex<f32>>,
    captured: &'a Arc<AtomicU64>,
    /// Set by [`Recorder::start_streaming`]: blocks go here instead of
    /// into `samples`.
    sink: Option<std::sync::mpsc::Sender<Vec<f32>>>,
}

fn build_stream<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    args: StreamArgs<'_>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = args.channels;
    let samples = args.samples.clone();
    let error = args.error.clone();
    let level = args.level.clone();
    let captured = args.captured.clone();
    let sink = args.sink;
    /// One interleaved frame mixed down to mono.
    fn mono<T: SizedSample>(frame: &[T], channels: usize) -> f32
    where
        f32: FromSample<T>,
    {
        let sum: f32 = frame.iter().map(|&x| f32::from_sample(x)).sum();
        sum / channels as f32
    }
    let stream = device.build_input_stream::<T, _, _>(
        config,
        move |data, _| {
            let mut peak = 0f32;
            match &sink {
                Some(sink) => {
                    let mut block = Vec::with_capacity(data.len() / channels.max(1));
                    for frame in data.chunks(channels) {
                        let sample = mono(frame, channels);
                        peak = peak.max(sample.abs());
                        block.push(sample);
                    }
                    captured.fetch_add(block.len() as u64, Ordering::Relaxed);
                    // A closed channel means the worker is gone; the
                    // recorder is about to be dropped with it.
                    let _ = sink.send(block);
                }
                None => {
                    let mut samples = samples.lock().unwrap();
                    let before = samples.len();
                    for frame in data.chunks(channels) {
                        let sample = mono(frame, channels);
                        peak = peak.max(sample.abs());
                        samples.push(sample);
                    }
                    let added = (samples.len() - before) as u64;
                    drop(samples);
                    captured.fetch_add(added, Ordering::Relaxed);
                }
            }
            *level.lock().unwrap() = peak;
        },
        move |e| *error.lock().unwrap() = Some(e.to_string()),
        None,
    )?;
    Ok(stream)
}
