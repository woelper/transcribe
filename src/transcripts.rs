//! Where finished meetings are stored, and how they are listed again.
//!
//! Each meeting is one UTF-8 text file: a few `Key: value` header lines
//! (title, date, duration, one-sentence summary), a divider, then the
//! transcript. The header is plain text on purpose — the files stay
//! readable and editable in any editor, and the history list rebuilds
//! itself from the directory alone, with no separate index to corrupt.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Divider between the header and the transcript body.
const DIVIDER: &str = "----------------------------------------------------------------";

/// Where auto mode writes finished meetings: a `transcripts/` directory
/// next to the models directory, so everything the app writes lives in
/// one place (the repo root when run from a checkout, `~/.transcribe`
/// for a released build).
pub fn transcripts_dir() -> PathBuf {
    let base = crate::find_models_dir()
        .or_else(crate::default_models_dir)
        .and_then(|models| models.parent().map(Path::to_path_buf));
    match base {
        Some(dir) => dir.join("transcripts"),
        None => PathBuf::from("transcripts"),
    }
}

/// A meeting's header: everything about it except the transcript itself.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Meta {
    /// Short label for the meeting, from the summary model.
    pub title: String,
    /// When recording started, as `YYYY-MM-DD HH:MM`.
    pub date: String,
    /// Length of the recording, `H:MM:SS` or `MM:SS`.
    pub duration: String,
    /// One sentence about what was discussed; empty when the summary
    /// model wasn't available.
    pub summary: String,
}

/// A stored transcript as the history list sees it.
#[derive(Debug, Clone)]
pub struct Saved {
    pub path: PathBuf,
    pub meta: Meta,
    /// Bytes of the file, for the "how long is this" hint.
    pub bytes: u64,
    /// Modification time, used to sort when a file has no date header.
    pub modified: std::time::SystemTime,
}

/// Characters no mainstream filesystem accepts in a name, plus the ones
/// that would make a shell or a path parser stumble.
fn sanitize(title: &str) -> String {
    let mut out = String::new();
    for c in title.chars() {
        match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => out.push(' '),
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    // Collapse runs of whitespace and trim what Windows strips silently.
    let out = out.split_whitespace().collect::<Vec<_>>().join(" ");
    let out = out.trim_matches(['.', ' ']).to_owned();
    // Leave room for the timestamp prefix and the extension within the
    // 255-byte limit most filesystems have.
    let mut truncated = String::new();
    for c in out.chars() {
        if truncated.len() + c.len_utf8() > 80 {
            break;
        }
        truncated.push(c);
    }
    truncated.trim_end().to_owned()
}

/// `2026-09-17 1432 Sprint planning.txt`, uniquified if it exists.
fn file_name(dir: &Path, meta: &Meta) -> PathBuf {
    // The date header is display format; the file name drops the colon,
    // which Windows forbids.
    let stamp = meta.date.replace(':', "");
    let title = sanitize(&meta.title);
    let stem = match (stamp.is_empty(), title.is_empty()) {
        (true, true) => "transcript".to_owned(),
        (true, false) => title,
        (false, true) => stamp,
        (false, false) => format!("{stamp} {title}"),
    };
    let mut path = dir.join(format!("{stem}.txt"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stem} ({n}).txt"));
        n += 1;
    }
    path
}

/// Render a meeting to its file contents.
pub fn render(meta: &Meta, body: &str) -> String {
    // A header value is one line by definition: the title and summary
    // come from a language model, and a stray newline in either would
    // split the header and swallow the rest of it.
    let line = |value: &str| value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::new();
    if !meta.title.is_empty() {
        out.push_str(&format!("Title: {}\n", line(&meta.title)));
    }
    if !meta.date.is_empty() {
        out.push_str(&format!("Date: {}\n", line(&meta.date)));
    }
    if !meta.duration.is_empty() {
        out.push_str(&format!("Duration: {}\n", line(&meta.duration)));
    }
    if !meta.summary.is_empty() {
        out.push_str(&format!("Summary: {}\n", line(&meta.summary)));
    }
    if !out.is_empty() {
        out.push_str(&format!("\n{DIVIDER}\n\n"));
    }
    out.push_str(body.trim_end());
    out.push('\n');
    out
}

/// Write a finished meeting into `dir`, returning the file it created.
pub fn save(dir: &Path, meta: &Meta, body: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
    let path = file_name(dir, meta);
    std::fs::write(&path, render(meta, body))
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

/// Split stored contents back into header and transcript. Files without a
/// header (a hand-saved transcript dropped into the directory) come back
/// whole, with an empty [`Meta`].
pub fn parse(contents: &str) -> (Meta, &str) {
    let mut meta = Meta::default();
    let mut rest = contents;
    let mut any = false;
    // Header lines run until the first blank line.
    loop {
        let (line, tail) = match rest.split_once('\n') {
            Some((line, tail)) => (line.trim_end_matches('\r'), tail),
            None => (rest, ""),
        };
        let Some((key, value)) = line.split_once(": ") else { break };
        let value = value.trim().to_owned();
        match key {
            "Title" => meta.title = value,
            "Date" => meta.date = value,
            "Duration" => meta.duration = value,
            "Summary" => meta.summary = value,
            _ => break,
        }
        any = true;
        rest = tail;
    }
    if !any {
        return (meta, contents);
    }
    // Then the blank line and the divider, if they are there.
    for _ in 0..3 {
        let (line, tail) = match rest.split_once('\n') {
            Some((line, tail)) => (line.trim_end_matches('\r'), tail),
            None => break,
        };
        if line.trim().is_empty() || line.trim_matches('-').is_empty() {
            rest = tail;
        } else {
            break;
        }
    }
    (meta, rest)
}

/// Read one stored transcript: its header and the transcript text.
pub fn read(path: &Path) -> Result<(Meta, String)> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let (meta, body) = parse(&contents);
    Ok((meta, body.to_owned()))
}

/// Every stored transcript in `dir`, newest first. A directory that isn't
/// there yet is simply empty — auto mode creates it on the first save.
pub fn list(dir: &Path) -> Vec<Saved> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut saved: Vec<Saved> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }
        let Ok(file_meta) = entry.metadata() else { continue };
        let modified = file_meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        // Only the head of the file is needed for the list; the body is
        // read when the user opens one.
        let mut meta = std::fs::read_to_string(&path)
            .map(|contents| parse(&contents).0)
            .unwrap_or_default();
        if meta.title.is_empty() {
            meta.title = path
                .file_stem()
                .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        }
        saved.push(Saved {
            path,
            meta,
            bytes: file_meta.len(),
            modified,
        });
    }
    // Newest first: by the date header when both have one (it sorts
    // lexicographically), else by mtime.
    saved.sort_by(|a, b| {
        b.meta
            .date
            .cmp(&a.meta.date)
            .then_with(|| b.modified.cmp(&a.modified))
    });
    saved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_survives_a_round_trip() {
        let meta = Meta {
            title: "Sprint planning".into(),
            date: "2026-09-17 14:32".into(),
            duration: "33:04".into(),
            summary: "The team agreed to ship on Friday.".into(),
        };
        let body = "Anna: Morning everyone.\nBen: Release is tagged.";
        let stored = render(&meta, body);
        let (parsed, parsed_body) = parse(&stored);
        assert_eq!(parsed, meta);
        assert_eq!(parsed_body.trim_end(), body);
    }

    /// A model-written title is untrusted text: a newline in it must not
    /// be able to break the header apart.
    #[test]
    fn multi_line_header_values_are_flattened() {
        let meta = Meta {
            title: "Sprint\nplanning".into(),
            summary: "First line.\nSecond line.".into(),
            date: "2026-09-17 14:32".into(),
            ..Meta::default()
        };
        let stored = render(&meta, "body text");
        let (parsed, body) = parse(&stored);
        assert_eq!(parsed.title, "Sprint planning");
        assert_eq!(parsed.summary, "First line. Second line.");
        assert_eq!(body.trim_end(), "body text");
    }

    #[test]
    fn a_plain_transcript_parses_as_body_only() {
        let text = "Anna: Morning everyone.\nBen: Release is tagged.\n";
        let (meta, body) = parse(text);
        assert_eq!(meta, Meta::default());
        assert_eq!(body, text);
    }

    /// A transcript line that happens to look like a header key — a
    /// speaker called "Date", say — must not be eaten as one.
    #[test]
    fn unknown_keys_end_the_header() {
        let text = "Title: Standup\n\nDate of birth: not a header\n";
        let (meta, body) = parse(text);
        assert_eq!(meta.title, "Standup");
        assert!(body.contains("Date of birth"));
    }

    #[test]
    fn titles_become_safe_file_names() {
        assert_eq!(sanitize("Q3 planning: budget/scope"), "Q3 planning budget scope");
        assert_eq!(sanitize("Grüße aus München"), "Grüße aus München");
        assert!(sanitize(&"x".repeat(200)).len() <= 80);
        // A model-written title is untrusted text: it must not be able to
        // walk out of the transcripts directory.
        let hostile = sanitize("  ../../etc/passwd  ");
        assert_eq!(hostile, "etc passwd");
        assert!(!hostile.contains(['/', '\\', ':']) && !hostile.starts_with('.'));
    }

    #[test]
    fn file_names_do_not_collide() {
        let dir = std::env::temp_dir().join("transcribe-name-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let meta = Meta {
            title: "Standup".into(),
            date: "2026-09-17 14:32".into(),
            ..Meta::default()
        };
        let first = save(&dir, &meta, "one").unwrap();
        let second = save(&dir, &meta, "two").unwrap();
        assert_ne!(first, second);
        assert_eq!(list(&dir).len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
