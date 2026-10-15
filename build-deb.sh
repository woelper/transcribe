#!/usr/bin/env bash
# Build a Debian package (.deb) that installs Transcribe system-wide:
# binaries in /usr/bin, a launcher and icon under /usr/share. Once it is
# installed the app no longer needs this checkout's target/ directory or
# a Rust toolchain, so `cargo clean` (or removing ~/.cargo) leaves it
# working. Models, vocabulary, speakers and the log stay per user in
# ~/.transcribe, exactly as for a released build.
#
# Usage: ./build-deb.sh [--no-build] [--install]
#   --no-build   package the binaries already in target/release
#   --install    install the package afterwards (sudo apt-get install)
#
# Needs only dpkg-deb; with dpkg-shlibdeps (package dpkg-dev) present the
# library dependencies are worked out from the binaries themselves.
set -euo pipefail
cd "$(dirname "$0")"
umask 022 # package contents must not inherit a group-writable umask

build=1
install=0
for arg in "$@"; do
  case "$arg" in
    --no-build) build=0 ;;
    --install) install=1 ;;
    *) echo "usage: ./build-deb.sh [--no-build] [--install]" >&2; exit 1 ;;
  esac
done

command -v dpkg-deb >/dev/null || { echo "error: dpkg-deb not found — this needs a Debian-based system" >&2; exit 1; }

if (( build )); then
  cargo build --release --locked
fi
for bin in transcribe-gui transcribe; do
  [[ -x target/release/$bin ]] || { echo "error: target/release/$bin not found — run without --no-build" >&2; exit 1; }
done

version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)
arch=$(dpkg --print-architecture)
out_dir=target/debian
pkg="$out_dir/transcribe_${version}_${arch}"
deb="$pkg.deb"

rm -rf "$pkg" "$deb"
install -Dm755 target/release/transcribe-gui "$pkg/usr/bin/transcribe-gui"
install -Dm755 target/release/transcribe "$pkg/usr/bin/transcribe"
install -Dm644 assets/icon-256.png "$pkg/usr/share/icons/hicolor/256x256/apps/transcribe.png"
install -Dm644 assets/fonts/LICENSE.txt "$pkg/usr/share/doc/transcribe/IBM-Plex-LICENSE.txt"
# Absolute Exec path: a stale per-user copy in ~/.local/bin (from
# install-linux.sh) would otherwise win the PATH lookup.
install -d "$pkg/usr/share/applications"
sed 's|^Exec=.*|Exec=/usr/bin/transcribe-gui|' assets/linux/transcribe.desktop \
  > "$pkg/usr/share/applications/transcribe.desktop"

# Libraries the binaries link against. dpkg-shlibdeps maps them to the
# package names of this distribution (they differ between releases, e.g.
# libasound2 vs libasound2t64); it insists on a debian/control in the
# working directory, hence the scratch directory.
depends=""
if command -v dpkg-shlibdeps >/dev/null; then
  scratch=$(mktemp -d)
  mkdir "$scratch/debian"
  touch "$scratch/debian/control"
  root=$PWD
  depends=$(cd "$scratch" && dpkg-shlibdeps -O \
    "$root/$pkg/usr/bin/transcribe-gui" "$root/$pkg/usr/bin/transcribe" 2>/dev/null \
    | sed -n 's/^shlibs:Depends=//p') || depends=""
  rm -rf "$scratch"
fi
if [[ -z "$depends" ]]; then
  echo "warning: dpkg-shlibdeps unavailable (apt install dpkg-dev), using a generic dependency list" >&2
  depends="libc6, libgcc-s1, libstdc++6, libasound2t64 | libasound2, libopenblas0, libgfortran5"
fi

maintainer="$(git config user.name 2>/dev/null || echo unknown) <$(git config user.email 2>/dev/null || echo unknown@localhost)>"
installed_kb=$(du -sk "$pkg" | cut -f1)

install -d "$pkg/DEBIAN"
# Recommends: what the window, the renderer and the file dialogs load at
# run time (dlopen), which no linker dependency reveals. Any desktop
# already has them. ffmpeg only decodes the codecs symphonia lacks.
cat > "$pkg/DEBIAN/control" <<CONTROL
Package: transcribe
Version: $version
Architecture: $arch
Maintainer: $maintainer
Installed-Size: $installed_kb
Depends: $depends
Recommends: libxkbcommon0, libwayland-client0 | libx11-6, libegl1 | libgl1, xdg-desktop-portal
Suggests: ffmpeg
Section: sound
Priority: optional
Homepage: https://github.com/woelper/transcribe
Description: Local, offline speech-to-text with speaker detection
 Transcribes meetings and recordings on this machine with Whisper,
 Parakeet or Qwen3-ASR, labels speakers, and summarizes transcripts with
 a local language model. Nothing leaves the computer.
 .
 Provides the desktop app (transcribe-gui) and the command line tool
 (transcribe). Models are downloaded into ~/.transcribe/models on first
 use.
CONTROL

dpkg-deb --root-owner-group --build "$pkg" "$deb" >/dev/null
rm -rf "$pkg"
echo "built $deb ($(du -h "$deb" | cut -f1))"

if (( install )); then
  # --reinstall: rebuilding without a version bump must still replace
  # the installed files.
  sudo apt-get install -y --reinstall "./$deb"
else
  echo "install it with:  sudo apt install ./$deb"
fi
echo "remove it with:   sudo apt remove transcribe"

for stale in "$HOME/.local/bin/transcribe-gui" "$HOME/.local/bin/transcribe" \
  "$HOME/.local/share/applications/transcribe.desktop"; do
  if [[ -e "$stale" ]]; then
    echo "note: $stale (from install-linux.sh) takes precedence over the package — remove it to use the system-wide install" >&2
  fi
done
