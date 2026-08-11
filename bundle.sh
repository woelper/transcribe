#!/usr/bin/env bash
# Build Transcribe.app from the GUI binary (requires `cargo install cargo-bundle`).
#
# cargo-bundle alone is not enough:
#  - without --bin it bundles the CLI, which exits immediately when Finder
#    launches it with no arguments
#  - macOS kills a bundled app that opens the microphone unless its
#    Info.plist carries NSMicrophoneUsageDescription, so add it
#  - ad-hoc codesign so the mic permission sticks to a stable identity
set -euo pipefail

cargo bundle --release --bin transcribe-gui

APP="target/release/bundle/osx/Transcribe.app"
PLIST="$APP/Contents/Info.plist"

/usr/libexec/PlistBuddy -c "Delete :NSMicrophoneUsageDescription" "$PLIST" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string Transcribe records audio input devices in order to transcribe them." "$PLIST"

codesign --force --deep -s - "$APP"

echo "bundled $APP"
echo "note: keep the app inside the repo (it looks for the models/ directory upwards from itself)"
