//! What the app remembers between runs.
//!
//! Small and forgiving on purpose: a missing, unreadable, or half-written
//! file just means defaults, never a failure to start. Unknown fields are
//! ignored and known ones are optional, so a settings file written by a
//! newer or older build still loads.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Settings file, next to vocabulary.md and speakers.json.
pub const SETTINGS_FILE: &str = "settings.json";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Speech model last picked in the model dropdown, by name. None on a
    /// first run; a name no longer offered falls back to the default.
    pub model: Option<String>,
}

/// Locate the settings file the same way as [`crate::speakers_path`].
pub fn settings_path() -> PathBuf {
    let local = PathBuf::from(SETTINGS_FILE);
    if local.exists() {
        return local;
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(hit) = exe
            .ancestors()
            .skip(1)
            .map(|dir| dir.join(SETTINGS_FILE))
            .find(|path| path.exists())
    {
        return hit;
    }
    match crate::find_models_dir().and_then(|m| m.parent().map(Path::to_path_buf)) {
        Some(dir) => dir.join(SETTINGS_FILE),
        None => local,
    }
}

/// Read the settings. Anything wrong with the file — missing, corrupt,
/// unreadable — means defaults, since none of it is worth blocking a
/// launch over.
pub fn load(path: &Path) -> Settings {
    match std::fs::read_to_string(path) {
        Ok(json) => serde_json::from_str(&json).unwrap_or_else(|e| {
            log::warn!("ignoring invalid {}: {e}", path.display());
            Settings::default()
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Settings::default(),
        Err(e) => {
            log::warn!("could not read {}: {e}", path.display());
            Settings::default()
        }
    }
}

pub fn save(path: &Path, settings: &Settings) -> Result<()> {
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        std::fs::create_dir_all(dir)?;
    }
    let json = serde_json::to_string_pretty(settings)?;
    std::fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_survive_a_round_trip() {
        let dir = std::env::temp_dir().join("transcribe-settings-test");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join(SETTINGS_FILE);
        assert_eq!(load(&path), Settings::default(), "a missing file is defaults");

        let settings = Settings { model: Some("parakeet-tdt-0.6b-v3".into()) };
        save(&path, &settings).unwrap();
        assert_eq!(load(&path), settings);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A settings file must never keep the app from starting, and one
    /// written by another version must still load what it recognizes.
    #[test]
    fn broken_and_foreign_files_fall_back_to_defaults() {
        let dir = std::env::temp_dir().join("transcribe-settings-broken");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let broken = dir.join("broken.json");
        std::fs::write(&broken, b"{ not json at all").unwrap();
        assert_eq!(load(&broken), Settings::default());

        let foreign = dir.join("foreign.json");
        std::fs::write(&foreign, br#"{"model": "tiny", "future_setting": 42}"#).unwrap();
        assert_eq!(load(&foreign).model.as_deref(), Some("tiny"));

        let empty = dir.join("empty.json");
        std::fs::write(&empty, b"{}").unwrap();
        assert_eq!(load(&empty), Settings::default());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
