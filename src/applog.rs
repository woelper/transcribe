//! A plain log file next to the app, so a user whose app "just closes" has
//! something to send: one line per event with a timestamp, plus every
//! panic with its location and backtrace.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use log::{Level, LevelFilter, Log, Metadata, Record};

const FILE_NAME: &str = "transcribe.log";
/// Above this size the old log is moved aside at startup.
const ROTATE_BYTES: u64 = 2 << 20;

struct FileLogger {
    file: Mutex<File>,
}

impl Log for FileLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        // Our own modules at info; whisper.cpp, ggml, wgpu, winit and the
        // rest only when something is wrong.
        let own = metadata.target().starts_with("transcribe");
        metadata.level() <= if own { Level::Info } else { Level::Warn }
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!(
            "{} {:5} [{}] {}\n",
            timestamp(),
            record.level(),
            record.target(),
            record.args()
        );
        if let Ok(mut file) = self.file.lock() {
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }
    }

    fn flush(&self) {
        if let Ok(mut file) = self.file.lock() {
            let _ = file.flush();
        }
    }
}

/// Where the log goes: next to the executable, except inside a macOS
/// bundle (which shouldn't be modified after signing) where it is the
/// per-user `~/.transcribe/` directory instead.
pub fn default_path() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .filter(|exe| !exe.to_string_lossy().contains("/Contents/MacOS/"))
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    match exe_dir {
        Some(dir) => dir.join(FILE_NAME),
        None => user_dir().join(FILE_NAME),
    }
}

fn user_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| PathBuf::from(home).join(".transcribe"))
        .unwrap_or_else(std::env::temp_dir)
}

/// Start logging to [`default_path`], falling back to the per-user
/// directory and then the temp directory if that isn't writable. Returns
/// where the log ended up. Calling it twice is harmless.
pub fn init() -> Option<PathBuf> {
    let candidates = [
        default_path(),
        user_dir().join(FILE_NAME),
        std::env::temp_dir().join(FILE_NAME),
    ];
    candidates.into_iter().find(|path| init_at(path).is_ok())
}

/// Start logging to `path` (append; rotated to `.old` when large).
pub fn init_at(path: &Path) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if std::fs::metadata(path).is_ok_and(|m| m.len() > ROTATE_BYTES) {
        let _ = std::fs::rename(path, path.with_extension("log.old"));
    }
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let logger = Box::new(FileLogger {
        file: Mutex::new(file),
    });
    // A second init (tests) just keeps the first logger.
    let _ = log::set_boxed_logger(logger);
    log::set_max_level(LevelFilter::Info);
    Ok(())
}

/// Log every panic (message, location, backtrace) before the default hook
/// runs, so a crash leaves a trace even when there is no console.
pub fn log_panics() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        log::error!(target: "transcribe::panic", "panic: {info}\n{backtrace}");
        log::logger().flush();
        previous(info);
    }));
}

/// `YYYY-MM-DD HH:MM:SS.mmm` in UTC.
fn timestamp() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (h, m, s) = (rem / 3600, (rem / 60) % 60, rem % 60);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(mo <= 2);
    format!(
        "{y:04}-{mo:02}-{d:02} {h:02}:{m:02}:{s:02}.{:03}",
        now.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_timestamped_lines() {
        // Exercises the logger directly: the global one is shared by every
        // test in the process, so which file it writes to is not ours to pick.
        let dir = std::env::temp_dir().join(format!("transcribe-log-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(FILE_NAME);
        let logger = FileLogger {
            file: Mutex::new(File::create(&path).unwrap()),
        };
        let record = |level: Level, target: &str, message: &str| {
            logger.log(
                &Record::builder()
                    .level(level)
                    .target(target)
                    .args(format_args!("{message}"))
                    .build(),
            );
        };
        record(Level::Info, "transcribe::test", "hello");
        record(Level::Info, "wgpu_core", "dropped: not ours");
        record(Level::Warn, "wgpu_core", "kept: a warning");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(" INFO  [transcribe::test] hello"), "{text}");
        assert!(!text.contains("dropped"), "{text}");
        assert!(text.contains("kept: a warning"), "{text}");
        // 2026-09-09 12:34:56.789 …
        let stamp = text.lines().next().unwrap();
        assert_eq!(stamp.as_bytes()[4], b'-');
        assert_eq!(stamp.as_bytes()[10], b' ');
        assert_eq!(stamp.as_bytes()[19], b'.');
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn panics_are_logged() {
        let dir = std::env::temp_dir().join(format!("transcribe-panic-test-{}", std::process::id()));
        let path = dir.join(FILE_NAME);
        // The logger is process-global; whichever test installed it first
        // owns the file, so read back through the logger's own file.
        init_at(&path).unwrap();
        log_panics();
        let caught = std::panic::catch_unwind(|| panic!("boom for the log"));
        assert!(caught.is_err());
        let logged = std::fs::read_dir(std::env::temp_dir())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("transcribe-"))
            .any(|e| {
                std::fs::read_to_string(e.path().join(FILE_NAME))
                    .is_ok_and(|t| t.contains("panic: ") && t.contains("boom for the log"))
            });
        assert!(logged);
        let _ = std::fs::remove_dir_all(dir);
    }
}
