#!/usr/bin/env bash
# Install Transcribe for the current user on Linux: binaries into
# ~/.local/bin, plus a launcher (.desktop entry) and icon so it shows up
# in the app menu and the dock/taskbar with its icon. On Wayland the
# icon can't be set by the app itself — compositors look it up from the
# desktop entry that matches the window's app id ("transcribe").
#
# Run from the repo after `cargo build --release`, or from an unpacked
# release tarball (binaries next to this script).
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x target/release/transcribe-gui ]]; then
  BIN_SRC=target/release
  ICON_SRC=assets/icon-256.png
  DESKTOP_SRC=assets/linux/transcribe.desktop
elif [[ -x ./transcribe-gui ]]; then
  BIN_SRC=.
  ICON_SRC=transcribe.png
  DESKTOP_SRC=transcribe.desktop
else
  echo "error: transcribe-gui not found — run 'cargo build --release' first" >&2
  exit 1
fi

BIN_DIR="$HOME/.local/bin"
ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$BIN_DIR" "$ICON_DIR" "$APP_DIR"

install -m 755 "$BIN_SRC/transcribe-gui" "$BIN_DIR/transcribe-gui"
[[ -x "$BIN_SRC/transcribe" ]] && install -m 755 "$BIN_SRC/transcribe" "$BIN_DIR/transcribe"
install -m 644 "$ICON_SRC" "$ICON_DIR/transcribe.png"
# Absolute Exec path, so the launcher works even if ~/.local/bin isn't in PATH.
sed "s|^Exec=.*|Exec=$BIN_DIR/transcribe-gui|" "$DESKTOP_SRC" > "$APP_DIR/transcribe.desktop"
chmod 644 "$APP_DIR/transcribe.desktop"

# Launched from the menu, the app looks for models in ~/.transcribe/models.
# Installing from the repo: point that at the repo's models/ instead of
# downloading everything again.
# The same goes for the vocabulary and enrolled speakers next to models/.
if [[ "$BIN_SRC" == target/release ]]; then
  mkdir -p "$HOME/.transcribe"
  for item in models vocabulary.md speakers.json; do
    if [[ ! -e "$HOME/.transcribe/$item" && -e "$PWD/$item" ]]; then
      ln -s "$PWD/$item" "$HOME/.transcribe/$item"
      echo "linked ~/.transcribe/$item -> $PWD/$item"
    fi
  done
fi

command -v update-desktop-database >/dev/null && update-desktop-database "$APP_DIR" || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

echo "installed:"
echo "  $BIN_DIR/transcribe-gui  (and transcribe)"
echo "  $APP_DIR/transcribe.desktop"
echo "  $ICON_DIR/transcribe.png"
echo "Transcribe should now appear in the app menu; log out and in if the icon doesn't update."
