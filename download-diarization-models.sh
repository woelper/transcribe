#!/usr/bin/env bash
# Download the speaker-diarization ONNX models (pyannote segmentation +
# wespeaker embeddings, ~35 MB total) used by the --diarize flag.
set -euo pipefail

BASE="https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0"

mkdir -p models
for MODEL in "segmentation-3.0.onnx" "wespeaker_en_voxceleb_CAM++.onnx"; do
  echo "downloading ${MODEL} ..."
  curl -L --fail --progress-bar -C - -o "models/${MODEL}" "${BASE}/${MODEL}"
done
echo "done."
