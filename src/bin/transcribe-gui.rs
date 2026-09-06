use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;
use egui_phosphor::fill::{
    ARROWS_CLOCKWISE, BOOK_OPEN, CHECK, COPY, DOWNLOAD_SIMPLE, FLOPPY_DISK, FOLDER_OPEN,
    LIST_BULLETS, MICROPHONE, NOTE_PENCIL, RECORD, STOP, TRASH, USERS, WARNING,
};
use transcribe::diarize::SpeakerProfile;
use transcribe::download::{
    DIARIZATION_MODELS, SUMMARY_MODEL_FILE, SUMMARY_MODEL_SIZE, SUMMARY_MODEL_URL, VAD_MODEL_FILE,
    VAD_MODEL_URL, SPEECH_MODELS, model_by_name,
};
use transcribe::recorder::{self, Recorder};
use transcribe::{DiarizeModels, Engine, Options, Progress, SpeakerVoice, Transcript};

const DEFAULT_MODEL: &str = "large-v3-turbo";

const MIC_PLIST_ERROR: &str = "error: this app bundle was built without microphone access \
    (missing NSMicrophoneUsageDescription) — macOS would kill it when recording starts. \
    Rebuild the bundle with ./bundle.sh instead of `cargo bundle`";

fn main() -> eframe::Result {
    // Same icon as the .app bundle; on macOS eframe applies it to the
    // Dock, so the binary looks right even when run outside the bundle.
    let icon = egui::IconData {
        rgba: include_bytes!("../../assets/icon-256.rgba").to_vec(),
        width: 256,
        height: 256,
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([860.0, 640.0])
            .with_icon(icon),
        ..Default::default()
    };
    eframe::run_native(
        "transcribe",
        options,
        Box::new(|cc| {
            setup_theme(&cc.egui_ctx);
            Ok(Box::new(App::new()))
        }),
    )
}

// The palette, lifted from the design reference: white cards floating on
// a lavender ground, one violet accent, green for anything live (levels,
// toggles), red for recording.
const TEXT: egui::Color32 = egui::Color32::from_rgb(0x26, 0x28, 0x3d);
const GROUND: egui::Color32 = egui::Color32::from_rgb(0xf1, 0xf2, 0xf9);
const CARD: egui::Color32 = egui::Color32::WHITE;
const FIELD: egui::Color32 = egui::Color32::from_rgb(0xf3, 0xf4, 0xfa);
const BUTTON: egui::Color32 = egui::Color32::from_rgb(0xec, 0xee, 0xf6);
const BUTTON_HOVER: egui::Color32 = egui::Color32::from_rgb(0xe1, 0xe4, 0xf1);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x6c, 0x5c, 0xe7);
const ACCENT_SOFT: egui::Color32 = egui::Color32::from_rgb(0xe9, 0xe6, 0xfb);
const GREEN: egui::Color32 = egui::Color32::from_rgb(0x2b, 0xd8, 0x8f);
const RED: egui::Color32 = egui::Color32::from_rgb(0xef, 0x50, 0x6e);
const TRACK_OFF: egui::Color32 = egui::Color32::from_rgb(0xd8, 0xdb, 0xe8);
/// Faint outline on buttons and dropdowns.
const FRAME: egui::Color32 = egui::Color32::from_rgb(0xdd, 0xe0, 0xec);
/// Memory bar in the model picker.
const MEMORY: egui::Color32 = egui::Color32::from_rgb(0xf5, 0xa6, 0x23);
const SHADOW: egui::Color32 = egui::Color32::from_rgba_premultiplied(5, 5, 8, 16);
/// Popups float above cards, so they need a deeper shadow than the cards.
const POPUP_SHADOW: egui::Color32 = egui::Color32::from_rgba_premultiplied(8, 8, 14, 64);

/// White rounded card with a soft drop shadow — the base surface every
/// panel and window sits on.
fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(CARD)
        .corner_radius(egui::CornerRadius::same(16))
        .shadow(egui::Shadow { offset: [0, 2], blur: 8, spread: 0, color: SHADOW })
        .inner_margin(egui::Margin::same(14))
}

/// Corner radius shared by buttons, fields, and progress bars.
const WIDGET_RADIUS: u8 = 12;

/// Progress bar with the same rounding as the buttons (egui's default is
/// a pill).
fn progress_bar(fraction: f32) -> egui::ProgressBar {
    egui::ProgressBar::new(fraction).corner_radius(egui::CornerRadius::same(WIDGET_RADIUS))
}

/// Layout of the model picker's rows: name and size on the left, three
/// rating bars on the right.
const MODEL_ROW_WIDTH: f32 = 366.0;
const MODEL_ROW_HEIGHT: f32 = 32.0;
const MODEL_BARS_LEFT: f32 = 204.0;
const MODEL_BAR_WIDTH: f32 = 42.0;
const MODEL_BAR_GAP: f32 = 12.0;

/// Column titles above the rating bars in the model picker.
fn model_rows_header(ui: &mut egui::Ui) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(MODEL_ROW_WIDTH, 16.0), egui::Sense::hover());
    let font = egui::TextStyle::Small.resolve(ui.style());
    for (i, title) in ["accuracy", "speed", "memory"].into_iter().enumerate() {
        let x = rect.left() + MODEL_BARS_LEFT + i as f32 * (MODEL_BAR_WIDTH + MODEL_BAR_GAP);
        ui.painter().text(
            egui::pos2(x + MODEL_BAR_WIDTH / 2.0, rect.center().y),
            egui::Align2::CENTER_CENTER,
            title,
            font.clone(),
            ui.visuals().weak_text_color(),
        );
    }
}

/// One selectable row of the model picker: installed/download icon and
/// name, size and caveat underneath, and bars for accuracy, speed, and
/// memory (relative to the largest model) so the tradeoff is visible.
fn model_row(
    ui: &mut egui::Ui,
    model: &transcribe::download::SpeechModel,
    selected: bool,
    installed: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(MODEL_ROW_WIDTH, MODEL_ROW_HEIGHT),
        egui::Sense::click(),
    );
    let painter = ui.painter();
    let radius = egui::CornerRadius::same(10);
    if selected {
        painter.rect_filled(rect, radius, ACCENT_SOFT);
    } else if response.hovered() {
        painter.rect_filled(rect, radius, BUTTON_HOVER);
    }

    let icon = if installed { CHECK } else { DOWNLOAD_SIMPLE };
    let body = egui::TextStyle::Body.resolve(ui.style());
    let small = egui::TextStyle::Small.resolve(ui.style());
    let left = rect.left() + 10.0;
    // Text stays in its column; a long caveat is cut rather than
    // running into the bars.
    let text_clip = egui::Rect::from_min_max(
        rect.min,
        egui::pos2(rect.left() + MODEL_BARS_LEFT - 8.0, rect.max.y),
    );
    let text_painter = painter.with_clip_rect(text_clip);
    text_painter.text(
        egui::pos2(left, rect.top() + 2.0),
        egui::Align2::LEFT_TOP,
        format!("{icon} {}", model.name),
        body,
        TEXT,
    );
    let detail = if model.note.is_empty() {
        model.size_label()
    } else {
        format!("{} · {}", model.size_label(), model.note)
    };
    text_painter.text(
        egui::pos2(left, rect.bottom() - 2.0),
        egui::Align2::LEFT_BOTTOM,
        detail,
        small,
        ui.visuals().weak_text_color(),
    );

    let largest = SPEECH_MODELS.iter().map(|m| m.mb).max().unwrap_or(1) as f32;
    let bars = [
        (model.accuracy, ACCENT),
        (model.speed, GREEN),
        (model.mb as f32 / largest, MEMORY),
    ];
    for (i, (fraction, color)) in bars.into_iter().enumerate() {
        let x = rect.left() + MODEL_BARS_LEFT + i as f32 * (MODEL_BAR_WIDTH + MODEL_BAR_GAP);
        let track = egui::Rect::from_min_size(
            egui::pos2(x, rect.center().y - 3.0),
            egui::vec2(MODEL_BAR_WIDTH, 6.0),
        );
        painter.rect_filled(track, 3.0, TRACK_OFF);
        let fill = egui::Rect::from_min_size(
            track.min,
            egui::vec2(MODEL_BAR_WIDTH * fraction.clamp(0.05, 1.0), 6.0),
        );
        painter.rect_filled(fill, 3.0, color);
    }
    response
}

/// Solid filled button with white text, for accented actions.
fn filled_button(text: String, fill: egui::Color32) -> egui::Button<'static> {
    egui::Button::new(
        egui::RichText::new(text)
            .color(egui::Color32::WHITE)
            .family(egui::FontFamily::Name("plex-medium".into())),
    )
    .fill(fill)
    .stroke(egui::Stroke::NONE)
}

/// Violet button for the one action that matters in a view.
fn primary_button(text: String) -> egui::Button<'static> {
    filled_button(text, ACCENT)
}

/// Animated on/off switch, green when on — replaces checkboxes to match
/// the reference. The label toggles too.
fn toggle(ui: &mut egui::Ui, on: &mut bool, label: &str) -> egui::Response {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 7.0;
        let size = egui::vec2(36.0, 20.0);
        let (rect, mut response) = ui.allocate_exact_size(size, egui::Sense::click());
        let text = ui.add(egui::Label::new(label).sense(egui::Sense::click()));
        if response.clicked() || text.clicked() {
            *on = !*on;
            response.mark_changed();
        }
        if ui.is_rect_visible(rect) {
            let t = ui.ctx().animate_bool_responsive(response.id, *on);
            let radius = rect.height() / 2.0;
            ui.painter().rect_filled(rect, radius, TRACK_OFF.lerp_to_gamma(GREEN, t));
            let x = egui::lerp(rect.left() + radius..=rect.right() - radius, t);
            let knob = egui::pos2(x, rect.center().y);
            ui.painter().circle_filled(knob, radius - 3.0, egui::Color32::WHITE);
        }
        response | text
    })
    .inner
}

/// IBM Plex typography over the reference palette, with roomy rounded
/// borderless widgets and soft shadows.
fn setup_theme(ctx: &egui::Context) {
    use egui::{CornerRadius, FontFamily, FontId, Shadow, Stroke, TextStyle, vec2};

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "plex-sans".into(),
        egui::FontData::from_static(include_bytes!("../../assets/fonts/IBMPlexSans-Regular.ttf"))
            .into(),
    );
    fonts.font_data.insert(
        "plex-sans-medium".into(),
        egui::FontData::from_static(include_bytes!("../../assets/fonts/IBMPlexSans-Medium.ttf"))
            .into(),
    );
    fonts.font_data.insert(
        "plex-mono".into(),
        egui::FontData::from_static(include_bytes!("../../assets/fonts/IBMPlexMono-Regular.ttf"))
            .into(),
    );
    fonts
        .families
        .get_mut(&FontFamily::Proportional)
        .unwrap()
        .insert(0, "plex-sans".into());
    fonts
        .families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(0, "plex-mono".into());
    fonts.families.insert(
        FontFamily::Name("plex-medium".into()),
        vec!["plex-sans-medium".into(), "plex-sans".into()],
    );
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Fill);
    // add_to_fonts only extends the built-in families; the medium family
    // (filled buttons) needs the icon font too or "{STOP} Stop" shows "?".
    fonts
        .families
        .get_mut(&FontFamily::Name("plex-medium".into()))
        .unwrap()
        .push("phosphor".into());
    ctx.set_fonts(fonts);

    ctx.set_theme(egui::ThemePreference::Light);
    let mut style = (*ctx.style_of(egui::Theme::Light)).clone();
    style.text_styles = [
        (TextStyle::Heading, FontId::new(18.0, FontFamily::Name("plex-medium".into()))),
        (TextStyle::Body, FontId::new(14.5, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(14.5, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace)),
        (TextStyle::Small, FontId::new(11.5, FontFamily::Proportional)),
    ]
    .into();
    style.spacing.item_spacing = vec2(10.0, 10.0);
    style.spacing.button_padding = vec2(14.0, 7.0);
    style.spacing.interact_size = vec2(40.0, 32.0);

    let mut v = egui::Visuals::light();
    v.override_text_color = Some(TEXT);
    v.panel_fill = GROUND;
    v.window_fill = CARD;
    v.window_stroke = Stroke::NONE;
    v.window_corner_radius = CornerRadius::same(16);
    v.window_shadow = Shadow { offset: [0, 8], blur: 32, spread: 0, color: SHADOW };
    v.popup_shadow = Shadow { offset: [0, 8], blur: 28, spread: 0, color: POPUP_SHADOW };
    v.extreme_bg_color = FIELD; // text-edit backgrounds, progress troughs
    v.faint_bg_color = FIELD;
    v.selection.bg_fill = ACCENT_SOFT;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    for w in [
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::same(WIDGET_RADIUS);
        w.bg_stroke = Stroke::new(1.0, FRAME); // faint outline on every button
        w.expansion = 0.0;
    }
    v.widgets.noninteractive.corner_radius = CornerRadius::same(WIDGET_RADIUS);
    v.widgets.noninteractive.bg_stroke = Stroke::NONE; // no separators/outlines
    v.widgets.inactive.weak_bg_fill = BUTTON;
    v.widgets.inactive.bg_fill = BUTTON;
    v.widgets.hovered.weak_bg_fill = BUTTON_HOVER;
    v.widgets.hovered.bg_fill = BUTTON_HOVER;
    v.widgets.active.weak_bg_fill = ACCENT_SOFT;
    v.widgets.active.bg_fill = ACCENT_SOFT;
    v.widgets.open.weak_bg_fill = BUTTON_HOVER;
    v.widgets.open.bg_fill = BUTTON_HOVER;
    style.visuals = v;
    ctx.set_style_of(egui::Theme::Light, style);
}

/// What gets transcribed: a loaded file or an in-memory recording.
enum Source {
    File(PathBuf),
    Recording { samples: Arc<Vec<f32>>, secs: f64 },
}

impl Source {
    fn label(&self) -> String {
        match self {
            Source::File(path) => path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
            Source::Recording { secs, .. } => format!("recording ({})", format_mmss(*secs)),
        }
    }
}

/// State written by the worker thread and read by the UI each frame.
#[derive(Default)]
struct Job {
    status: String,
    percent: Option<i32>,
    /// When whisper started decoding — basis for the ETA.
    transcribe_started: Option<std::time::Instant>,
    /// Segments streamed while whisper is still decoding; replaced by the
    /// authoritative formatted transcript when the job finishes.
    live: String,
    result: Option<anyhow::Result<Transcript>>,
}

/// State of a model download running in a background thread.
struct Download {
    /// What is being fetched, as shown in the status line, e.g.
    /// "model tiny" or "speaker detection models".
    what: String,
    started: std::time::Instant,
    done: u64,
    total: Option<u64>,
    result: Option<anyhow::Result<()>>,
}

struct App {
    source: Option<Source>,
    devices: Vec<String>,
    device: Option<String>, // None = system default input
    recorder: Option<Recorder>,
    diarize: bool,
    timestamps: bool,
    transcript: String,
    status: String,
    job: Option<Arc<Mutex<Job>>>,
    // Resolved at startup: a bundled .app launches with cwd = "/",
    // so relative model paths would not work there.
    models_dir: Option<PathBuf>,
    vocabulary: String,
    vocabulary_path: PathBuf,
    show_vocabulary: bool,
    /// Free-form notes about the current recording (meeting name, speakers,
    /// topics) — folded into the whisper prompt for this transcription only.
    context: String,
    model: String,
    download: Option<Arc<Mutex<Download>>>,
    show_speakers: bool,
    /// How many people are in the recording (None = automatic, up to 8).
    /// Junk clusters fold into the real speakers when this is set right.
    max_speakers: Option<usize>,
    profiles: Vec<SpeakerProfile>,
    speakers_path: PathBuf,
    enroll_name: String,
    enroll_recorder: Option<Recorder>,
    enroll_from: String,
    enroll_to: String,
    enroll_job: Option<Arc<Mutex<Enroll>>>,
    /// Speakers of the last finished transcript, for post-hoc renaming.
    speaker_voices: Vec<SpeakerVoice>,
    rename_inputs: Vec<String>,
    show_rename: bool,
    enroll_on_rename: bool,
    summary: String,
    show_summary: bool,
    summary_job: Option<Arc<Mutex<Summary>>>,
    /// Summarize was clicked before the summary model was on disk; start
    /// summarizing as soon as its download finishes.
    summarize_pending: bool,
}

/// A file-based enrollment running in a background thread (decoding and
/// VAD over a long recording can take a while).
struct Enroll {
    name: String,
    result: Option<anyhow::Result<Vec<f32>>>,
}

/// A summarization running in a background thread.
struct Summary {
    /// Partial summary text, streamed in as it generates.
    live: String,
    result: Option<anyhow::Result<String>>,
}

impl App {
    fn new() -> Self {
        // Fall back to the per-user directory so a released app that lives
        // outside the repo can still download models into a known place.
        let models_dir = transcribe::find_models_dir().or_else(transcribe::default_models_dir);
        let vocabulary_path = transcribe::vocabulary_path();
        let speakers_path = transcribe::speakers_path();
        let profiles = transcribe::load_speaker_profiles(&speakers_path).unwrap_or_default();
        let status = match &models_dir {
            None => "error: models/ directory not found — run the download \
                scripts and keep the app inside the repo (or put a models/ \
                folder next to it)"
                .into(),
            Some(dir) if !dir.join(model_by_name(DEFAULT_MODEL).unwrap().file).exists() => {
                "no speech model downloaded yet — pick one from the model dropdown".into()
            }
            Some(_) => String::new(),
        };
        Self {
            source: None,
            devices: recorder::input_devices(),
            device: None,
            recorder: None,
            diarize: false,
            timestamps: false,
            transcript: String::new(),
            status,
            job: None,
            models_dir,
            vocabulary: std::fs::read_to_string(&vocabulary_path).unwrap_or_default(),
            vocabulary_path,
            show_vocabulary: false,
            context: String::new(),
            model: DEFAULT_MODEL.into(),
            download: None,
            show_speakers: false,
            max_speakers: None,
            profiles,
            speakers_path,
            enroll_name: String::new(),
            enroll_recorder: None,
            enroll_from: String::new(),
            enroll_to: String::new(),
            enroll_job: None,
            speaker_voices: Vec::new(),
            rename_inputs: Vec::new(),
            show_rename: false,
            enroll_on_rename: true,
            summary: String::new(),
            show_summary: false,
            summary_job: None,
            summarize_pending: false,
        }
    }

    /// Enroll from an existing audio file, optionally restricted to a
    /// time range where only this person speaks.
    fn start_file_enrollment(&mut self, path: PathBuf) {
        let Some(dir) = self.models_dir.clone() else {
            self.status = "error: models/ directory not found".into();
            return;
        };
        let from = match parse_time(&self.enroll_from) {
            Err(input) => {
                self.status = format!("error: can't parse start time \"{input}\" — use mm:ss");
                return;
            }
            Ok(t) => t,
        };
        let to = match parse_time(&self.enroll_to) {
            Err(input) => {
                self.status = format!("error: can't parse end time \"{input}\" — use mm:ss");
                return;
            }
            Ok(t) => t,
        };

        let job = Arc::new(Mutex::new(Enroll {
            name: self.enroll_name.trim().to_owned(),
            result: None,
        }));
        self.enroll_job = Some(job.clone());
        std::thread::spawn(move || {
            let result = (|| {
                let samples = transcribe::decode_to_mono_16k(&path)?;
                let rate = 16_000f64;
                let lo = (from.unwrap_or(0.0) * rate) as usize;
                let hi = to.map_or(samples.len(), |t| (t * rate) as usize).min(samples.len());
                anyhow::ensure!(lo < hi, "the time range contains no audio");
                transcribe::diarize::voice_embedding(
                    &samples[lo..hi],
                    16_000,
                    &dir.join("segmentation-3.0.onnx"),
                    &dir.join("wespeaker_en_voxceleb_CAM++.onnx"),
                )
            })();
            job.lock().unwrap().result = Some(result);
        });
    }

    /// Collect a finished file enrollment; returns whether one is running.
    fn poll_enrollment(&mut self) -> bool {
        let Some(job) = self.enroll_job.clone() else {
            return false;
        };
        let mut job = job.lock().unwrap();
        match job.result.take() {
            Some(Ok(embedding)) => {
                self.profiles.retain(|p| p.name != job.name);
                self.profiles.push(SpeakerProfile {
                    name: job.name.clone(),
                    embedding,
                });
                match transcribe::save_speaker_profiles(&self.speakers_path, &self.profiles) {
                    Ok(()) => {
                        self.status = format!("enrolled {}", job.name);
                        self.enroll_name.clear();
                        self.enroll_from.clear();
                        self.enroll_to.clear();
                    }
                    Err(e) => self.status = format!("error: {e:#}"),
                }
                self.enroll_job = None;
                false
            }
            Some(Err(e)) => {
                self.status = format!("error: enrollment failed: {e:#}");
                self.enroll_job = None;
                false
            }
            None => true,
        }
    }

    fn model_path(&self) -> Option<PathBuf> {
        let model = model_by_name(&self.model)?;
        self.models_dir.as_ref().map(|dir| dir.join(model.file))
    }

    /// Whether the selected model takes the vocabulary/context prompt
    /// (only Whisper does).
    fn prompt_supported(&self) -> bool {
        model_by_name(&self.model)
            .is_none_or(|m| Engine::for_model(Path::new(m.file)) == Engine::Whisper)
    }

    /// Fetch `url` into `path` in the background, shown in the status bar
    /// as `what`. Only one download runs at a time.
    fn start_file_download(&mut self, what: String, url: String, path: PathBuf) {
        if self.download.is_some() {
            return;
        }
        let download = Arc::new(Mutex::new(Download {
            what,
            started: std::time::Instant::now(),
            done: 0,
            total: None,
            result: None,
        }));
        self.download = Some(download.clone());
        std::thread::spawn(move || {
            let progress = {
                let download = download.clone();
                move |done, total| {
                    let mut download = download.lock().unwrap();
                    download.done = done;
                    download.total = total;
                }
            };
            let result = transcribe::download::download(&url, &path, progress);
            download.lock().unwrap().result = Some(result);
        });
    }

    /// Fetch the selected model in the background if it isn't on disk yet.
    fn start_download_if_missing(&mut self) {
        let Some(path) = self.model_path() else { return };
        let Some(model) = model_by_name(&self.model) else { return };
        if path.exists() {
            return;
        }
        self.start_file_download(format!("model {}", self.model), model.url.to_owned(), path);
    }

    /// Whether the speaker-detection (diarization) models are on disk.
    fn diarization_models_ready(&self) -> bool {
        self.models_dir
            .as_ref()
            .is_some_and(|dir| DIARIZATION_MODELS.iter().all(|m| dir.join(m.file).exists()))
    }

    /// Fetch any missing speaker-detection models in the background, so a
    /// released binary works without the repo's download scripts.
    fn start_diarization_download_if_missing(&mut self) {
        let Some(dir) = self.models_dir.clone() else { return };
        if self.download.is_some() {
            return;
        }
        let missing: Vec<(PathBuf, &'static str)> = DIARIZATION_MODELS
            .iter()
            .filter(|m| !dir.join(m.file).exists())
            .map(|m| (dir.join(m.file), m.url))
            .collect();
        if missing.is_empty() {
            return;
        }
        let download = Arc::new(Mutex::new(Download {
            what: "speaker detection models".into(),
            started: std::time::Instant::now(),
            done: 0,
            total: None,
            result: None,
        }));
        self.download = Some(download.clone());
        std::thread::spawn(move || {
            let mut result = Ok(());
            for (path, url) in missing {
                // Cumulative progress across both files; the combined size
                // isn't known up front, so `total` stays None and the status
                // line shows a plain MB counter.
                let base = download.lock().unwrap().done;
                let progress = {
                    let download = download.clone();
                    move |done, _total| download.lock().unwrap().done = base + done
                };
                result = transcribe::download::download(url, &path, progress);
                if result.is_err() {
                    break;
                }
            }
            download.lock().unwrap().result = Some(result);
        });
    }

    fn summary_model_path(&self) -> Option<PathBuf> {
        self.models_dir.as_ref().map(|dir| dir.join(SUMMARY_MODEL_FILE))
    }

    /// Summarize the transcript with the local chat model, fetching the
    /// model first if it isn't on disk yet.
    fn start_summarization(&mut self) {
        let Some(path) = self.summary_model_path() else { return };
        if !path.exists() {
            self.summarize_pending = true;
            self.start_file_download(
                "summary model".into(),
                SUMMARY_MODEL_URL.to_owned(),
                path,
            );
            return;
        }
        let job = Arc::new(Mutex::new(Summary {
            live: String::new(),
            result: None,
        }));
        self.summary_job = Some(job.clone());
        self.summary.clear();
        self.show_summary = true;
        let transcript = self.transcript.clone();
        let context = self.context.clone();
        std::thread::spawn(move || {
            let progress = {
                let job = job.clone();
                move |text: &str| job.lock().unwrap().live = text.to_owned()
            };
            let result = transcribe::summarize::summarize(&path, &transcript, &context, &progress);
            job.lock().unwrap().result = Some(result);
        });
    }

    /// Collect summary updates; returns whether one is running.
    fn poll_summary(&mut self) -> bool {
        let Some(job) = self.summary_job.clone() else {
            return false;
        };
        let mut job = job.lock().unwrap();
        match job.result.take() {
            Some(Ok(summary)) => {
                self.summary = summary;
                self.status = "summary ready".into();
                self.summary_job = None;
                false
            }
            Some(Err(e)) => {
                self.status = format!("error: summarization failed: {e:#}");
                self.summary_job = None;
                false
            }
            None => {
                self.summary = job.live.clone();
                true
            }
        }
    }

    /// Collect download updates; returns the in-flight progress, if any.
    fn poll_download(&mut self) -> Option<(String, Option<f32>, Option<String>)> {
        let download = self.download.clone()?;
        let mut download = download.lock().unwrap();
        match download.result.take() {
            Some(Ok(())) => {
                self.status = format!("{} downloaded", download.what);
                self.download = None;
                // A "speakers" request made while this download was running
                // was skipped (one download at a time) — pick it up now.
                if self.diarize {
                    self.start_diarization_download_if_missing();
                }
                if self.summarize_pending
                    && self.summary_model_path().is_some_and(|p| p.exists())
                {
                    self.summarize_pending = false;
                    self.start_summarization();
                }
                None
            }
            Some(Err(e)) => {
                self.status = format!("error: downloading {} failed: {e:#}", download.what);
                self.download = None;
                self.summarize_pending = false;
                None
            }
            None => {
                let fraction = download.total.map(|t| download.done as f32 / t as f32);
                let remaining =
                    fraction.and_then(|f| eta(download.started, f as f64));
                Some((
                    format!(
                        "downloading {} — {} MB",
                        download.what,
                        download.done >> 20
                    ),
                    fraction,
                    remaining,
                ))
            }
        }
    }

    fn start_transcription(&mut self) {
        let Some(source) = &self.source else { return };
        let Some(models_dir) = self.models_dir.clone() else {
            self.status = "error: models/ directory not found — cannot transcribe".into();
            return;
        };
        let job = Arc::new(Mutex::new(Job {
            status: match source {
                Source::File(_) => "decoding ...".into(),
                Source::Recording { .. } => "transcribing ...".into(),
            },
            ..Job::default()
        }));
        self.job = Some(job.clone());
        self.transcript.clear();
        self.status.clear();
        self.speaker_voices.clear();
        self.rename_inputs.clear();
        self.show_rename = false;

        let Some(model_path) = self.model_path() else {
            return;
        };
        let vad_model = models_dir.join(VAD_MODEL_FILE);
        let mut opts = Options {
            model: model_path,
            vad_model: Some(vad_model.clone()),
            prompt: transcribe::build_prompt(&self.vocabulary, &self.context),
            known_speakers: self.profiles.iter().map(|p| p.name.clone()).collect(),
            context: format!("{}\n{}", self.context, self.vocabulary),
            timestamps: self.timestamps,
            diarize: self.diarize.then(|| DiarizeModels {
                segmentation_model: models_dir.join("segmentation-3.0.onnx"),
                embedding_model: models_dir.join("wespeaker_en_voxceleb_CAM++.onnx"),
                profiles: self.profiles.clone(),
                max_speakers: self.max_speakers.unwrap_or(8),
                ..DiarizeModels::default()
            }),
            ..Options::default()
        };
        enum Input {
            File(PathBuf),
            Samples(Arc<Vec<f32>>),
        }
        let input = match source {
            Source::File(path) => Input::File(path.clone()),
            Source::Recording { samples, .. } => Input::Samples(samples.clone()),
        };
        std::thread::spawn(move || {
            // The VAD model is tiny; fetch it inline on first use. If that
            // fails (offline), transcribe without it rather than giving up.
            if !vad_model.exists() {
                job.lock().unwrap().status = "downloading voice activity model ...".into();
                if transcribe::download::download(VAD_MODEL_URL, &vad_model, |_, _| {}).is_err() {
                    opts.vad_model = None;
                }
            }
            let progress = {
                let job = job.clone();
                move |progress: Progress| {
                    let mut job = job.lock().unwrap();
                    match progress {
                        Progress::Decoded { audio_secs, .. } => {
                            job.status = format!("decoded {audio_secs:.0}s of audio");
                        }
                        Progress::DetectingSpeakers => job.status = "detecting speakers ...".into(),
                        Progress::Diarized { segments, .. } => {
                            job.status = format!("diarized {segments} speech segments");
                        }
                        Progress::Transcribing { percent } => {
                            job.status = "transcribing ...".into();
                            job.percent = Some(percent);
                            job.transcribe_started.get_or_insert_with(std::time::Instant::now);
                        }
                        Progress::Segment { text } => {
                            job.live.push_str(&text);
                            job.live.push('\n');
                        }
                        Progress::Transcribed { .. } => job.percent = Some(100),
                    }
                }
            };
            let result = match input {
                Input::File(path) => transcribe::transcribe(&path, &opts, progress),
                Input::Samples(samples) => {
                    transcribe::transcribe_samples((*samples).clone(), &opts, progress)
                }
            };
            job.lock().unwrap().result = Some(result);
        });
    }

    /// Collect worker updates; returns the current in-flight status, if any.
    fn poll_job(&mut self) -> Option<(String, Option<i32>, Option<String>)> {
        let job = self.job.clone()?;
        let mut job = job.lock().unwrap();
        match job.result.take() {
            Some(Ok(transcript)) => {
                self.transcript = transcript.text;
                self.speaker_voices = transcript.speaker_voices;
                self.rename_inputs = vec![String::new(); self.speaker_voices.len()];
                let mut status = match transcript.speakers {
                    Some(n) => format!("done — {n} speaker(s) detected"),
                    None => "done".into(),
                };
                if !transcript.speaker_matches.is_empty() {
                    let names: Vec<String> = transcript
                        .speaker_matches
                        .iter()
                        .map(|(name, s)| format!("{name} ({s:.2})"))
                        .collect();
                    status.push_str(&format!(", recognized: {}", names.join(", ")));
                }
                if !transcript.weak_matches.is_empty() {
                    let names: Vec<String> = transcript
                        .weak_matches
                        .iter()
                        .map(|(name, s)| format!("{name} ({s:.2})"))
                        .collect();
                    status.push_str(&format!(
                        " — {} matched too weakly and stayed numbered; consider re-enrolling",
                        names.join(", ")
                    ));
                }
                self.status = status;
                self.job = None;
                None
            }
            Some(Err(e)) => {
                self.status = format!("error: {e:#}");
                self.job = None;
                None
            }
            None => {
                // Mirror streamed segments into the transcript view live.
                if self.transcript.len() != job.live.len() {
                    self.transcript = job.live.clone();
                }
                let remaining = match (job.percent, job.transcribe_started) {
                    (Some(p), Some(t0)) => eta(t0, p as f64 / 100.0),
                    _ => None,
                };
                Some((job.status.clone(), job.percent, remaining))
            }
        }
    }

    /// Start or finish recording an enrollment sample. On finish, the
    /// sample becomes a voice fingerprint stored under the entered name,
    /// persisted to speakers.json for all future transcriptions.
    fn toggle_enrollment(&mut self) {
        match self.enroll_recorder.take() {
            None if !mic_usage_declared() => {
                self.status = MIC_PLIST_ERROR.into();
            }
            None => match Recorder::start(self.device.as_deref()) {
                Ok(recorder) => self.enroll_recorder = Some(recorder),
                Err(e) => self.status = format!("error: {e:#}"),
            },
            Some(recorder) => {
                let result = recorder.stop().and_then(|samples| {
                    let Some(dir) = &self.models_dir else {
                        anyhow::bail!("models/ directory not found");
                    };
                    transcribe::diarize::voice_embedding(
                        &samples,
                        16_000,
                        &dir.join("segmentation-3.0.onnx"),
                        &dir.join("wespeaker_en_voxceleb_CAM++.onnx"),
                    )
                });
                match result {
                    Ok(embedding) => {
                        let name = self.enroll_name.trim().to_owned();
                        // Re-enrolling an existing name replaces their voice.
                        self.profiles.retain(|p| p.name != name);
                        self.profiles.push(SpeakerProfile { name, embedding });
                        match transcribe::save_speaker_profiles(&self.speakers_path, &self.profiles)
                        {
                            Ok(()) => {
                                self.status = format!("enrolled {}", self.enroll_name.trim());
                                self.enroll_name.clear();
                            }
                            Err(e) => self.status = format!("error: {e:#}"),
                        }
                    }
                    Err(e) => self.status = format!("error: enrollment failed: {e:#}"),
                }
            }
        }
    }

    fn toggle_recording(&mut self) {
        match self.recorder.take() {
            None if !mic_usage_declared() => {
                self.status = MIC_PLIST_ERROR.into();
            }
            None => match Recorder::start(self.device.as_deref()) {
                Ok(recorder) => {
                    self.recorder = Some(recorder);
                    self.source = None;
                    self.transcript.clear();
                    self.status.clear();
                }
                Err(e) => self.status = format!("error: {e:#}"),
            },
            Some(recorder) => {
                let secs = recorder.duration_secs();
                match recorder.stop() {
                    // Whisper hallucinates phrases like "Thank you." on silence,
                    // so refuse to transcribe a recording with no signal in it.
                    Ok(samples) if is_silent(&samples) => {
                        self.status = "error: the recording contains no audio signal — \
                            check that the input device isn't muted and that this app has \
                            microphone access (System Settings → Privacy & Security → Microphone)"
                            .into();
                    }
                    Ok(samples) => {
                        self.source = Some(Source::Recording {
                            samples: Arc::new(samples),
                            secs,
                        });
                        self.status = format!("recorded {}", format_mmss(secs));
                    }
                    Err(e) => self.status = format!("error: {e:#}"),
                }
            }
        }
    }
}

impl eframe::App for App {
    /// The lavender ground the panel cards float on.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        GROUND.to_normalized_gamma_f32()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Painted here too (not only via clear_color) so off-screen renders
        // such as the snapshot test show the ground instead of transparency.
        ui.painter().rect_filled(ui.ctx().content_rect(), 0.0, GROUND);
        let running = self.poll_job();
        let downloading = self.poll_download();
        let enrolling = self.poll_enrollment();
        let summarizing = self.poll_summary();
        // A dead input device would otherwise record silence forever.
        if let Some(error) = self.recorder.as_ref().and_then(|r| r.error()) {
            self.recorder = None;
            self.status = format!("error: recording failed: {error}");
        }
        if running.is_some()
            || downloading.is_some()
            || enrolling
            || summarizing
            || self.recorder.is_some()
            || self.enroll_recorder.is_some()
        {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        let recording = self.recorder.is_some();
        let busy = running.is_some() || recording;

        // Each panel is a white card floating on the lavender ground.
        let top_frame =
            card().outer_margin(egui::Margin { left: 16, right: 16, top: 16, bottom: 10 });
        egui::Panel::top("controls")
            .frame(top_frame)
            .show_separator_line(false)
            .show(ui, |ui| {
            // One row: record from a device, or open a file; then what's loaded.
            ui.horizontal_wrapped(|ui| {
                let record_button = if recording {
                    filled_button(format!("{STOP} Stop"), RED)
                } else {
                    egui::Button::new(format!("{RECORD} Record"))
                };
                let can_record = running.is_none() && self.enroll_recorder.is_none();
                if ui.add_enabled(can_record, record_button).clicked() {
                    self.toggle_recording();
                }
                ui.label("from");
                let selected = self.device.clone().unwrap_or_else(|| "default input".into());
                egui::ComboBox::from_id_salt("device")
                    .selected_text(selected)
                    .width(220.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.device, None, "default input");
                        for name in &self.devices {
                            ui.selectable_value(&mut self.device, Some(name.clone()), name);
                        }
                    });
                if ui.button(ARROWS_CLOCKWISE).on_hover_text("refresh device list").clicked() {
                    self.devices = recorder::input_devices();
                }
                if ui
                    .add_enabled(!busy, egui::Button::new(format!("{FOLDER_OPEN} Open audio…")))
                    .clicked()
                    && let Some(file) = rfd::FileDialog::new()
                        .add_filter("audio", &["mp3", "m4a", "mp4", "wav", "flac", "ogg"])
                        .pick_file()
                {
                    self.source = Some(Source::File(file));
                    self.transcript.clear();
                    self.status.clear();
                }
                match (&self.source, self.recorder.as_ref()) {
                    (_, Some(recorder)) => {
                        ui.label(
                            egui::RichText::new(format!(
                                "{MICROPHONE} recording {}",
                                format_mmss(recorder.duration_secs())
                            ))
                            .monospace()
                            .color(RED),
                        );
                        // Live input level, so a dead source (muted mic, missing
                        // permission, unrouted loopback) is visible immediately.
                        let level = recorder.level();
                        ui.add(
                            progress_bar((level * 4.0).min(1.0))
                                .desired_width(80.0)
                                .fill(GREEN),
                        )
                        .on_hover_text("input level");
                        if level == 0.0 && recorder.duration_secs() > 2.0 {
                            ui.colored_label(ui.visuals().warn_fg_color, format!("{WARNING} no signal!"));
                        }
                    }
                    (Some(source), _) => {
                        ui.monospace(source.label());
                    }
                    (None, _) => {
                        ui.weak("no audio loaded");
                    }
                };
            });
            let prompt_supported = self.prompt_supported();
            let no_prompt_reason = format!(
                "{} takes no prompt, so vocabulary and context are ignored — \
                 they only apply to Whisper models",
                self.model
            );
            ui.horizontal_wrapped(|ui| {
                let model_enabled =
                    self.models_dir.is_some() && running.is_none() && self.download.is_none();
                let mut picked = false;
                ui.add_enabled_ui(model_enabled, |ui| {
                    egui::ComboBox::from_id_salt("model")
                        .selected_text(&self.model)
                        .width(230.0)
                        // Tall enough for every model without scrolling.
                        .height(600.0)
                        .show_ui(ui, |ui| {
                            ui.spacing_mut().item_spacing.y = 2.0;
                            model_rows_header(ui);
                            for m in SPEECH_MODELS {
                                let installed = self
                                    .models_dir
                                    .as_ref()
                                    .is_some_and(|dir| dir.join(m.file).exists());
                                if model_row(ui, m, self.model == m.name, installed).clicked() {
                                    self.model = m.name.to_owned();
                                    picked = true;
                                }
                            }
                        });
                });
                if picked {
                    self.start_download_if_missing();
                }
                toggle(ui, &mut self.timestamps, "timestamps");
                if toggle(ui, &mut self.diarize, "speakers").changed() && self.diarize {
                    self.start_diarization_download_if_missing();
                }
                if self.diarize {
                    let selected = self
                        .max_speakers
                        .map_or("count: auto".to_owned(), |n| format!("count: {n}"));
                    egui::ComboBox::from_id_salt("max-speakers")
                        .selected_text(selected)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.max_speakers, None, "auto (up to 8)");
                            for n in 2..=8 {
                                ui.selectable_value(&mut self.max_speakers, Some(n), n.to_string());
                            }
                        })
                        .response
                        .on_hover_text(
                            "how many people are in the recording — setting it avoids \
                             phantom extra speakers",
                        );
                }
                if ui
                    .add_enabled(prompt_supported, egui::Button::new(format!("{BOOK_OPEN} Vocabulary…")))
                    .on_hover_text("terms Whisper should spell correctly — product names, people, acronyms")
                    .on_disabled_hover_text(&no_prompt_reason)
                    .clicked()
                {
                    self.show_vocabulary = !self.show_vocabulary;
                }
                if ui
                    .button(format!("{USERS} Speakers…"))
                    .on_hover_text("enroll voices so speakers are named in the transcript")
                    .clicked()
                {
                    self.show_speakers = !self.show_speakers;
                }
                if !self.speaker_voices.is_empty()
                    && ui
                        .button(format!("{NOTE_PENCIL} Rename…"))
                        .on_hover_text("give the detected speakers real names")
                        .clicked()
                {
                    self.show_rename = !self.show_rename;
                }
                let summary_model_missing =
                    self.summary_model_path().is_none_or(|p| !p.exists());
                let can_summarize = !self.transcript.trim().is_empty()
                    && !busy
                    && self.summary_job.is_none()
                    && self.download.is_none()
                    && self.models_dir.is_some();
                let summarize_hover = if summary_model_missing {
                    format!(
                        "summarize the transcript with a local Llama model — \
                         first use downloads it ({SUMMARY_MODEL_SIZE})"
                    )
                } else {
                    "summarize the transcript with the local Llama model".into()
                };
                if ui
                    .add_enabled(
                        can_summarize,
                        egui::Button::new(format!("{LIST_BULLETS} Summarize")),
                    )
                    .on_hover_text(summarize_hover)
                    .clicked()
                {
                    self.start_summarization();
                }
            });
            let context_hint = if prompt_supported {
                "context for this recording — meeting name, speakers, topics, notes \
                 (sent to Whisper along with the vocabulary)"
                    .to_owned()
            } else {
                no_prompt_reason.clone()
            };
            ui.add_enabled(
                prompt_supported,
                egui::TextEdit::multiline(&mut self.context)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY)
                    // Same breathing room as the transcript card.
                    .margin(egui::Margin::symmetric(14, 12))
                    .hint_text(context_hint),
            )
            .on_disabled_hover_text(&no_prompt_reason);
        });

        let bottom_frame = card()
            .inner_margin(egui::Margin::symmetric(14, 10))
            .outer_margin(egui::Margin { left: 16, right: 16, top: 10, bottom: 16 });
        egui::Panel::bottom("status")
            .frame(bottom_frame)
            .show_separator_line(false)
            .show(ui, |ui| {
            // Transcribe sits bottom-right; the status fills the space left of it.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let can_start = self.source.is_some()
                    && !busy
                    && self.download.is_none()
                    && self.model_path().is_some_and(|p| p.exists())
                    && (!self.diarize || self.diarization_models_ready());
                // Violet only when actionable — a filled fill() would
                // otherwise override the disabled look.
                let transcribe = if can_start {
                    primary_button("Transcribe".into())
                } else {
                    egui::Button::new("Transcribe")
                };
                if ui.add_enabled(can_start, transcribe).clicked() {
                    self.start_transcription();
                }
                // Only once there is something to save.
                if !self.transcript.is_empty()
                    && ui
                        .add_enabled(!busy, egui::Button::new(format!("{FLOPPY_DISK} Save transcript…")))
                        .clicked()
                {
                    let name = match &self.source {
                        Some(Source::File(path)) => path
                            .file_stem()
                            .map_or("transcript".into(), |s| s.to_string_lossy().into_owned()),
                        _ => "transcript".into(),
                    };
                    if let Some(path) = rfd::FileDialog::new()
                        .set_file_name(format!("{name}.txt"))
                        .add_filter("text", &["txt"])
                        .save_file()
                    {
                        self.status = match std::fs::write(&path, &self.transcript) {
                            Ok(()) => format!("saved to {}", path.display()),
                            Err(e) => format!("error: failed to save: {e}"),
                        };
                    }
                }
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    match (&running, &downloading) {
                        (Some((status, percent, remaining)), _) => {
                            ui.add(egui::Spinner::new().color(ACCENT));
                            ui.label(status);
                            if let Some(percent) = percent {
                                let mut bar =
                                    progress_bar(*percent as f32 / 100.0).fill(ACCENT);
                                bar = match remaining {
                                    Some(eta) => bar.text(format!("{percent}% — {eta}")),
                                    None => bar.show_percentage(),
                                };
                                ui.add(bar);
                            }
                        }
                        (None, Some((status, fraction, remaining))) => {
                            ui.add(egui::Spinner::new().color(ACCENT));
                            ui.label(status);
                            if let Some(fraction) = fraction {
                                let mut bar = progress_bar(*fraction).fill(ACCENT);
                                bar = match remaining {
                                    Some(eta) => {
                                        bar.text(format!("{:.0}% — {eta}", fraction * 100.0))
                                    }
                                    None => bar.show_percentage(),
                                };
                                ui.add(bar);
                            }
                        }
                        (None, None) => {
                            if !self.status.is_empty() {
                                ui.label(&self.status);
                            } else {
                                ui.weak("open an audio file or record one, then hit Transcribe");
                            }
                        }
                    }
                });
            });
        });

        let central_frame =
            card().outer_margin(egui::Margin { left: 16, right: 16, top: 6, bottom: 6 });
        egui::CentralPanel::default().frame(central_frame).show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(running.is_some())
                .show(ui, |ui| {
                    // Frameless, so the transcript sits directly on the card.
                    ui.add_sized(
                        ui.available_size(),
                        egui::TextEdit::multiline(&mut self.transcript)
                            .font(egui::TextStyle::Monospace)
                            .frame(egui::Frame::NONE)
                            .hint_text("transcript appears here"),
                    );
                });
        });

        let mut show_vocabulary = self.show_vocabulary;
        egui::Window::new("Vocabulary")
            .open(&mut show_vocabulary)
            .default_size([420.0, 320.0])
            .show(ui.ctx(), |ui| {
                ui.label(
                    "Terms Whisper would otherwise misspell — product names, \
                     people, acronyms — one per line. The transcription is \
                     biased toward these spellings.",
                );
                ui.weak(format!("saved to {}", self.vocabulary_path.display()));
                ui.add_space(4.0);
                let edit = egui::ScrollArea::vertical()
                    .auto_shrink(false)
                    .show(ui, |ui| {
                        ui.add_sized(
                            ui.available_size(),
                            egui::TextEdit::multiline(&mut self.vocabulary)
                                .font(egui::TextStyle::Monospace)
                                .hint_text("# Vocabulary\n- Kubernetes\n- Grafana\n- OKR"),
                        )
                    })
                    .inner;
                if edit.changed()
                    && let Err(e) = std::fs::write(&self.vocabulary_path, &self.vocabulary)
                {
                    self.status = format!("error: failed to save vocabulary: {e}");
                }
            });
        self.show_vocabulary = show_vocabulary;

        let mut show_summary = self.show_summary;
        egui::Window::new("Summary")
            .open(&mut show_summary)
            .default_size([480.0, 360.0])
            .show(ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    if summarizing {
                        ui.add(egui::Spinner::new().color(ACCENT));
                        ui.label("summarizing ...");
                    } else if ui
                        .add_enabled(
                            !self.summary.is_empty(),
                            egui::Button::new(format!("{COPY} Copy")),
                        )
                        .clicked()
                    {
                        ui.ctx().copy_text(self.summary.clone());
                    }
                });
                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .auto_shrink(false)
                    .stick_to_bottom(summarizing)
                    .show(ui, |ui| {
                        ui.add_sized(
                            ui.available_size(),
                            egui::TextEdit::multiline(&mut self.summary)
                                .hint_text("the summary appears here"),
                        );
                    });
            });
        self.show_summary = show_summary;

        let mut show_speakers = self.show_speakers;
        egui::Window::new("Speakers")
            .open(&mut show_speakers)
            .default_size([380.0, 260.0])
            .show(ui.ctx(), |ui| {
                ui.label(
                    "Enroll a voice once and that person is named in every \
                     future transcript instead of \"Speaker N\".",
                );
                ui.weak(format!("saved to {}", self.speakers_path.display()));
                ui.add_space(6.0);

                let mut remove: Option<usize> = None;
                for (i, profile) in self.profiles.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{USERS} {}", profile.name));
                        if ui.button(TRASH).on_hover_text("remove this voice").clicked() {
                            remove = Some(i);
                        }
                    });
                }
                if let Some(i) = remove {
                    self.profiles.remove(i);
                    if let Err(e) =
                        transcribe::save_speaker_profiles(&self.speakers_path, &self.profiles)
                    {
                        self.status = format!("error: {e:#}");
                    }
                }
                if self.profiles.is_empty() {
                    ui.weak("no voices enrolled yet");
                }

                ui.add_space(12.0);
                ui.label("Enroll a new voice — enter the name, then record \
                    ~10 seconds of them speaking, or pick a recording where \
                    only they speak (optionally a time range of it):");
                let models_ready = self.diarization_models_ready();
                if !models_ready {
                    ui.horizontal(|ui| {
                        if self.download.is_some() {
                            ui.add(egui::Spinner::new().color(ACCENT));
                            ui.label("downloading ...");
                        } else if ui
                            .button(format!(
                                "{DOWNLOAD_SIMPLE} Download speaker detection models (34 MB)"
                            ))
                            .clicked()
                        {
                            self.start_diarization_download_if_missing();
                        }
                    });
                    ui.add_space(6.0);
                }
                let can_enroll = models_ready
                    && !self.enroll_name.trim().is_empty()
                    && self.recorder.is_none()
                    && !enrolling;
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.enroll_name)
                            .desired_width(140.0)
                            .hint_text("name"),
                    );
                    match &self.enroll_recorder {
                        None => {
                            let button = egui::Button::new(format!("{RECORD} Record sample"));
                            let response = ui.add_enabled(can_enroll, button);
                            if !models_ready {
                                response.on_hover_text(
                                    "speaker detection models missing — download them above",
                                );
                            } else if response.clicked() {
                                self.toggle_enrollment();
                            }
                        }
                        Some(recorder) => {
                            ui.add(
                                progress_bar((recorder.level() * 4.0).min(1.0))
                                    .desired_width(60.0)
                                    .fill(GREEN),
                            );
                            ui.monospace(format_mmss(recorder.duration_secs()));
                            if ui
                                .add(filled_button(format!("{STOP} Stop & enroll"), RED))
                                .clicked()
                            {
                                self.toggle_enrollment();
                            }
                        }
                    }
                });
                ui.horizontal(|ui| {
                    if enrolling {
                        ui.add(egui::Spinner::new().color(ACCENT));
                        ui.label("enrolling from file ...");
                    } else {
                        let button =
                            egui::Button::new(format!("{FOLDER_OPEN} From audio file…"));
                        let can_file =
                            can_enroll && self.enroll_recorder.is_none();
                        if ui.add_enabled(can_file, button).clicked()
                            && let Some(file) = rfd::FileDialog::new()
                                .add_filter("audio", &["mp3", "m4a", "mp4", "wav", "flac", "ogg"])
                                .pick_file()
                        {
                            self.start_file_enrollment(file);
                        }
                        ui.label("from");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.enroll_from)
                                .desired_width(48.0)
                                .hint_text("0:00"),
                        );
                        ui.label("to");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.enroll_to)
                                .desired_width(48.0)
                                .hint_text("end"),
                        );
                    }
                });
            });
        self.show_speakers = show_speakers;

        let mut show_rename = self.show_rename;
        egui::Window::new("Rename speakers")
            .open(&mut show_rename)
            .default_size([360.0, 240.0])
            .show(ui.ctx(), |ui| {
                ui.label("Give the voices from this transcript real names:");
                ui.add_space(4.0);
                for (voice, input) in self.speaker_voices.iter().zip(&mut self.rename_inputs) {
                    ui.horizontal(|ui| {
                        ui.label(format!("{} →", voice.label));
                        ui.add(
                            egui::TextEdit::singleline(input)
                                .desired_width(160.0)
                                .hint_text("new name"),
                        );
                    });
                }
                ui.add_space(4.0);
                toggle(
                    ui,
                    &mut self.enroll_on_rename,
                    "also enroll these voices for future transcripts",
                );
                if ui.add(primary_button(format!("{CHECK} Apply"))).clicked() {
                    let mut renamed = 0;
                    for (voice, input) in
                        self.speaker_voices.iter_mut().zip(&mut self.rename_inputs)
                    {
                        let new_name = input.trim().to_owned();
                        if new_name.is_empty() || new_name == voice.label {
                            continue;
                        }
                        self.transcript =
                            transcribe::rename_speaker(&self.transcript, &voice.label, &new_name);
                        if self.enroll_on_rename {
                            self.profiles.retain(|p| p.name != new_name);
                            self.profiles.push(SpeakerProfile {
                                name: new_name.clone(),
                                embedding: voice.embedding.clone(),
                            });
                        }
                        voice.label = new_name;
                        input.clear();
                        renamed += 1;
                    }
                    if renamed > 0 {
                        self.status = if self.enroll_on_rename {
                            match transcribe::save_speaker_profiles(
                                &self.speakers_path,
                                &self.profiles,
                            ) {
                                Ok(()) => format!(
                                    "renamed and enrolled {renamed} speaker(s) — future \
                                     transcripts will use these names"
                                ),
                                Err(e) => format!("renamed, but enrolling failed: {e:#}"),
                            }
                        } else {
                            format!("renamed {renamed} speaker(s)")
                        };
                    }
                }
            });
        self.show_rename = show_rename;
    }
}

fn format_mmss(secs: f64) -> String {
    let s = secs as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

/// "1 minute 2 seconds remaining" from an estimated remaining duration.
fn format_eta(secs: f64) -> String {
    let s = secs.ceil().max(0.0) as u64;
    let plural = |n: u64| if n == 1 { "" } else { "s" };
    if s >= 3600 {
        let (h, m) = (s / 3600, (s % 3600) / 60);
        format!("{h} hour{} {m} minute{} remaining", plural(h), plural(m))
    } else if s >= 60 {
        let (m, r) = (s / 60, s % 60);
        format!("{m} minute{} {r} second{} remaining", plural(m), plural(r))
    } else {
        format!("{s} second{} remaining", plural(s))
    }
}

/// Remaining time extrapolated from elapsed time and completed fraction.
fn eta(started: std::time::Instant, fraction: f64) -> Option<String> {
    if !(0.01..1.0).contains(&fraction) {
        return None;
    }
    let elapsed = started.elapsed().as_secs_f64();
    // Too early to extrapolate meaningfully.
    if elapsed < 2.0 {
        return None;
    }
    Some(format_eta(elapsed * (1.0 - fraction) / fraction))
}

/// Parse "mm:ss", "h:mm:ss", or plain seconds. Empty input is Ok(None);
/// unparseable input is returned as the error.
fn parse_time(s: &str) -> Result<Option<f64>, &str> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(None);
    }
    let mut total = 0.0;
    for part in s.split(':') {
        match part.parse::<f64>() {
            Ok(v) if v >= 0.0 => total = total * 60.0 + v,
            _ => return Err(s),
        }
    }
    Ok(Some(total))
}

/// macOS kills a bundled app the moment it opens the microphone if its
/// Info.plist lacks NSMicrophoneUsageDescription — which happens when the
/// bundle is built with plain `cargo bundle` instead of ./bundle.sh.
/// Detect that here so recording can refuse with an explanation instead.
fn mic_usage_declared() -> bool {
    let Ok(exe) = std::env::current_exe() else {
        return true;
    };
    // …/Transcribe.app/Contents/MacOS/transcribe-gui → …/Contents/Info.plist
    let Some(contents) = exe.parent().and_then(|dir| dir.parent()) else {
        return true;
    };
    if contents.file_name().and_then(|n| n.to_str()) != Some("Contents") {
        return true; // not running from an .app bundle
    }
    match std::fs::read(contents.join("Info.plist")) {
        // The key name appears as plain ASCII in both XML and binary plists.
        Ok(bytes) => {
            let key = b"NSMicrophoneUsageDescription";
            bytes.windows(key.len()).any(|w| w == key)
        }
        Err(_) => true,
    }
}

/// True when there's no usable signal anywhere in the recording. Even a
/// quiet room leaves a mic noise floor well above this; only a dead input
/// (muted, no permission, unrouted loopback) stays this low throughout.
fn is_silent(samples: &[f32]) -> bool {
    samples.iter().all(|s| s.abs() < 1e-4)
}

#[cfg(test)]
mod tests {
    use eframe::egui;

    use super::parse_time;

    /// Renders the main view off-screen and stores it under
    /// tests/snapshots/ as a regression check, then dresses the same render
    /// up with window chrome and a drop shadow as assets/screenshot.png for
    /// the README. Ignored by default because it needs a GPU (or a
    /// software Vulkan driver such as lavapipe); run with
    /// `cargo test --bin transcribe-gui -- --ignored`, and set
    /// UPDATE_SNAPSHOTS=force to regenerate the baseline image.
    #[test]
    #[ignore = "needs a GPU; run with --ignored to check/update the UI snapshot"]
    fn ui_screenshot() {
        let mut harness = egui_kittest::Harness::builder()
            .with_size(egui::vec2(860.0, 640.0))
            // kittest defaults to dark and overrides the app's preference.
            .with_theme(egui::Theme::Light)
            .build_eframe(|cc| {
                super::setup_theme(&cc.egui_ctx);
                let mut app = super::App::new();
                app.source = Some(super::Source::File("standup-2026-09-05.m4a".into()));
                app.diarize = true;
                app.context = "Weekly standup — Anna, Ben, Chris. Topics: \
                    release 0.3, onboarding, GPU builds."
                    .into();
                app.transcript = "\
                    Anna: Morning everyone, let's keep it short today.\n\
                    Ben: Release 0.3 is tagged, the bundle script now signs the app.\n\
                    Anna: Nice. Anything blocking the GPU builds?\n\
                    Chris: The Metal path is fine, I'm still chasing the Vulkan validation warning.\n\
                    Ben: I can pair on that after lunch.\n\
                    Anna: Perfect. Onboarding doc review moves to Thursday then.\n"
                    .into();
                app.status = "done — 3 speaker(s) detected, recognized: Anna (0.84)".into();
                app
            });
        harness.run();
        harness.snapshot("transcribe-gui");

        let app = harness.render().unwrap();
        let chrome = render_window_chrome(app.width() as f32);
        readme_screenshot(&chrome, &app).save("assets/screenshot.png").unwrap();
    }

    /// A macOS-style title bar (traffic lights, centered title) on the
    /// app's ground color, rendered with the app's own fonts.
    fn render_window_chrome(width: f32) -> image::RgbaImage {
        use egui::{Align2, FontFamily, FontId, pos2, vec2};

        const HEIGHT: f32 = 44.0;
        let mut themed = false;
        let mut harness = egui_kittest::Harness::builder()
            .with_size(vec2(width, HEIGHT))
            .with_theme(egui::Theme::Light)
            .build_ui(move |ui| {
                if !themed {
                    // Fonts apply from the next frame on; run() repeats.
                    super::setup_theme(ui.ctx());
                    themed = true;
                    return;
                }
                let rect = ui.ctx().content_rect();
                let painter = ui.painter();
                painter.rect_filled(rect, 0.0, super::GROUND);
                let lights = [(0xff, 0x5f, 0x57), (0xfe, 0xbc, 0x2e), (0x28, 0xc8, 0x40)];
                for (i, (r, g, b)) in lights.into_iter().enumerate() {
                    let center = pos2(rect.left() + 20.0 + 20.0 * i as f32, rect.center().y);
                    painter.circle_filled(center, 6.0, egui::Color32::from_rgb(r, g, b));
                }
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "Transcribe",
                    FontId::new(13.0, FontFamily::Name("plex-medium".into())),
                    super::TEXT,
                );
            });
        harness.run();
        harness.render().unwrap()
    }

    /// Stacks chrome over the app render, rounds the window corners, and
    /// floats it on a transparent canvas with a soft drop shadow.
    fn readme_screenshot(chrome: &image::RgbaImage, app: &image::RgbaImage) -> image::RgbaImage {
        use image::{Rgba, RgbaImage, imageops};

        const RADIUS: f32 = 12.0;
        const MARGIN: u32 = 48;
        const SHADOW_OFFSET: u32 = 14;
        const SHADOW_BLUR: f32 = 14.0;
        const SHADOW_ALPHA: f32 = 0.4;

        let (w, h) = (app.width(), chrome.height() + app.height());
        let mut window = RgbaImage::new(w, h);
        imageops::overlay(&mut window, chrome, 0, 0);
        imageops::overlay(&mut window, app, 0, chrome.height() as i64);

        // Anti-aliased rounded corners: coverage falls off across the
        // pixel that straddles the corner arc.
        let (fw, fh) = (w as f32, h as f32);
        for (x, y, pixel) in window.enumerate_pixels_mut() {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let cx = px.clamp(RADIUS, fw - RADIUS);
            let cy = py.clamp(RADIUS, fh - RADIUS);
            let distance = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt() - RADIUS;
            let coverage = (0.5 - distance).clamp(0.0, 1.0);
            pixel[3] = (pixel[3] as f32 * coverage).round() as u8;
        }

        let mut shadow = RgbaImage::from_pixel(w + 2 * MARGIN, h + 2 * MARGIN, Rgba([0, 0, 0, 0]));
        for (x, y, pixel) in window.enumerate_pixels() {
            let alpha = (pixel[3] as f32 * SHADOW_ALPHA).round() as u8;
            shadow.put_pixel(x + MARGIN, y + MARGIN + SHADOW_OFFSET, Rgba([0, 0, 0, alpha]));
        }
        let mut canvas = imageops::blur(&shadow, SHADOW_BLUR);
        imageops::overlay(&mut canvas, &window, MARGIN as i64, MARGIN as i64);
        canvas
    }

    /// The model picker opened, so its rows and bars are checked too.
    #[test]
    #[ignore = "needs a GPU; run with --ignored to check/update the UI snapshot"]
    fn ui_screenshot_model_picker() {
        let mut harness = egui_kittest::Harness::builder()
            .with_size(egui::vec2(860.0, 640.0))
            .with_theme(egui::Theme::Light)
            .build_eframe(|cc| {
                super::setup_theme(&cc.egui_ctx);
                let mut app = super::App::new();
                app.source = Some(super::Source::File("standup-2026-09-05.m4a".into()));
                app
            });
        harness.run();
        // Click the model dropdown (second row, left), then move the
        // pointer off the list so no row shows a hover highlight.
        let dropdown = egui::pos2(145.0, 91.0);
        for pressed in [true, false] {
            harness.input_mut().events.push(egui::Event::PointerMoved(dropdown));
            harness.input_mut().events.push(egui::Event::PointerButton {
                pos: dropdown,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            });
            harness.run();
        }
        harness.input_mut().events.push(egui::Event::PointerMoved(egui::pos2(700.0, 400.0)));
        harness.run();
        harness.snapshot("transcribe-gui-model-picker");
    }

    #[test]
    fn parses_enrollment_time_ranges() {
        assert_eq!(parse_time("  "), Ok(None));
        assert_eq!(parse_time("90"), Ok(Some(90.0)));
        assert_eq!(parse_time("1:30"), Ok(Some(90.0)));
        assert_eq!(parse_time("1:02:03"), Ok(Some(3723.0)));
        assert_eq!(parse_time("abc"), Err("abc"));
        assert_eq!(parse_time("1:-2"), Err("1:-2"));
    }
}
