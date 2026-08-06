#!/usr/bin/env bash
set -euo pipefail

DEST="hand_landmarker.task"

if [ -f "$DEST" ]; then
  echo "Model already exists at $DEST"
  exit 0
fi

CANDIDATES=(
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task"
  "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
)

download() {
  url="$1"
  echo "Trying: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$DEST" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$DEST" "$url"
  else
    echo "Error: curl or wget is required to download the model."
    return 1
  fi

  if [ -s "$DEST" ]; then
    echo "Downloaded $DEST ($(wc -c < "$DEST") bytes)"
    return 0
  else
    echo "Downloaded file empty or too small; removing."
    rm -f "$DEST"
    return 1
  fi
}

# If user supplied URL, try that first
if [ "${1:-}" != "" ]; then
  if download "$1"; then
    echo "Downloaded from user-provided URL"
  else
    echo "Failed to download from provided URL: $1"
    exit 1
  fi
  exit 0
fi

# Try candidate URLs
for url in "${CANDIDATES[@]}"; do
  if download "$url"; then
    echo "Model acquired from: $url"
    exit 0
  fi
done

cat <<EOF
Failed to download hand_landmarker.task from known locations.
Options:
  1) Provide a direct model URL: ./scripts/get_model.sh https://example.com/hand_landmarker.task
  2) Manually place hand_landmarker.task in the repository root
  3) Host the model in a release or storage bucket and update this script's CANDIDATES array
EOF
exit 2
