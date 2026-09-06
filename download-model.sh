#!/usr/bin/env bash
# Download a speech model from Hugging Face.
# Usage: ./download-model.sh [model]   (default: large-v3-turbo)
# Whisper options: tiny, base, small, medium, large-v3, and .en variants.
# "parakeet" fetches NVIDIA Parakeet-TDT-0.6B-v3 (GGUF for transcribe.cpp).
set -euo pipefail

MODEL="${1:-large-v3-turbo}"
mkdir -p models

if [[ "$MODEL" == "parakeet" || "$MODEL" == parakeet-tdt-0.6b-v3* ]]; then
  DEST="models/parakeet-tdt-0.6b-v3-Q8_0.gguf"
  URL="https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf"
else
  DEST="models/ggml-${MODEL}.bin"
  URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
fi

echo "downloading ${MODEL} to ${DEST} ..."
curl -L --fail --progress-bar -C - -o "${DEST}" "${URL}"
echo "done."
