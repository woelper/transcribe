#!/usr/bin/env bash
# Cut a release: bump the version in Cargo.toml, commit, and create an
# annotated tag whose body is an AI-generated changelog (claude CLI).
# Pushing the tag triggers .github/workflows/release.yml, which builds the
# macOS/Linux/Windows artifacts and reuses the tag body as release notes.
#
# Usage: ./release.sh [major|minor|patch|X.Y.Z]   (default: patch)
set -euo pipefail
cd "$(dirname "$0")"

bump="${1:-patch}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is not clean — commit or stash first" >&2
  exit 1
fi

current=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)
IFS=. read -r major minor patch <<<"$current"
case "$bump" in
  major) version="$((major + 1)).0.0" ;;
  minor) version="$major.$((minor + 1)).0" ;;
  patch) version="$major.$minor.$((patch + 1))" ;;
  *.*.*) version="$bump" ;;
  *) echo "usage: ./release.sh [major|minor|patch|X.Y.Z]" >&2; exit 1 ;;
esac
tag="v$version"

if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
  echo "error: tag $tag already exists" >&2
  exit 1
fi

echo "bumping $current -> $version"
sed -i.bak "s/^version = \"$current\"/version = \"$version\"/" Cargo.toml
rm -f Cargo.toml.bak
# sync the lockfile so the CI build (--locked) doesn't fail
cargo update --workspace --quiet

# --- AI-generated release notes ------------------------------------------
last=$(git describe --tags --abbrev=0 2>/dev/null || true)
if [[ -n "$last" ]]; then
  range="$last..HEAD"
else
  range="HEAD"
fi
commits=$(git log --no-merges --pretty='- %s' "$range")

notes=""
if command -v claude >/dev/null; then
  echo "generating release notes with claude..."
  notes=$(claude -p "Write GitHub release notes for version $version of \
'transcribe', a local offline speech-to-text CLI/GUI. Base them only on \
these commits, grouping related changes and skipping trivial ones. Use \
short markdown bullet points under at most a few bold section headers. \
Output only the release notes, no preamble, no heading with the version.

Commits:
$commits" 2>/dev/null) || notes=""
fi
if [[ -z "$notes" ]]; then
  echo "warning: claude unavailable, falling back to raw commit list" >&2
  notes="$commits"
fi

git commit -am "Release $tag"
git tag -a "$tag" -m "$tag" -m "$notes"

echo
echo "$notes"
echo
echo "created tag $tag — publish the release with:"
echo "  git push origin HEAD --follow-tags"
