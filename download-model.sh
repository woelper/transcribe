#!/usr/bin/env bash
# Download a ggml Whisper model from Hugging Face.
# Usage: ./download-model.sh [model]   (default: large-v3-turbo)
# Other options: tiny, base, small, medium, large-v3, and .en variants.
set -euo pipefail

MODEL="${1:-large-v3-turbo}"
DEST="models/ggml-${MODEL}.bin"

mkdir -p models
echo "downloading ${MODEL} to ${DEST} ..."
curl -L --fail --progress-bar -C - -o "${DEST}" \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
echo "done."
