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
}

impl Recorder {
    /// Start recording from the named device (None = system default input).
    pub fn start(device_name: Option<&str>) -> Result<Self> {
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

        use cpal::SampleFormat as SF;
        let build = |format| match format {
            SF::F32 => build_stream::<f32>(&device, config.into(), channels, &samples, &error, &level),
            SF::I16 => build_stream::<i16>(&device, config.into(), channels, &samples, &error, &level),
            SF::I32 => build_stream::<i32>(&device, config.into(), channels, &samples, &error, &level),
            SF::U16 => build_stream::<u16>(&device, config.into(), channels, &samples, &error, &level),
            other => bail!("unsupported sample format {other}"),
        };
        let stream = build(config.sample_format())?;
        stream.play().context("failed to start recording")?;

        Ok(Self {
            _stream: stream,
            samples,
            rate,
            error,
            level,
        })
    }

    /// Peak amplitude (0..=1) of the most recent capture buffer —
    /// a live signal indicator. Stays at 0.0 when no audio arrives
    /// (muted device, missing microphone permission, silent loopback).
    pub fn level(&self) -> f32 {
        *self.level.lock().unwrap()
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples.lock().unwrap().len() as f64 / self.rate as f64
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

fn build_stream<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    channels: usize,
    samples: &Arc<Mutex<Vec<f32>>>,
    error: &Arc<Mutex<Option<String>>>,
    level: &Arc<Mutex<f32>>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let samples = samples.clone();
    let error = error.clone();
    let level = level.clone();
    let stream = device.build_input_stream::<T, _, _>(
        config,
        move |data, _| {
            let mut peak = 0f32;
            let mut samples = samples.lock().unwrap();
            for frame in data.chunks(channels) {
                let sum: f32 = frame.iter().map(|&x| f32::from_sample(x)).sum();
                let mono = sum / channels as f32;
                peak = peak.max(mono.abs());
                samples.push(mono);
            }
            drop(samples);
            *level.lock().unwrap() = peak;
        },
        move |e| *error.lock().unwrap() = Some(e.to_string()),
        None,
    )?;
    Ok(stream)
}
