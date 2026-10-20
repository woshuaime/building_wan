#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${WAN_PYTHON:-/home/shi/miniconda3/envs/scj/bin/python}"
MODEL_SOURCES="$PROJECT_ROOT/configs/vace_p01_model_sources_v1.json"
DOWNLOAD=0

if [[ $# -gt 1 ]]; then
    printf '%s\n' 'Usage: bash scripts/download_vace_p01.sh [--dry-run|--download]' >&2
    exit 2
fi
case "${1:---dry-run}" in
    --dry-run) ;;
    --download) DOWNLOAD=1 ;;
    *) printf '%s\n' 'Usage: bash scripts/download_vace_p01.sh [--dry-run|--download]' >&2; exit 2 ;;
esac

mapfile -t MODEL_FIELDS < <("$PYTHON_BIN" -B - "$MODEL_SOURCES" <<'PY'
import json, sys
from pathlib import PurePosixPath
source = json.load(open(sys.argv[1], encoding="utf-8"))
assert source["model_id"] == "Wan-AI/Wan2.1-VACE-1.3B"
checkpoint = source["checkpoint"]
target = PurePosixPath(checkpoint["default_local_path"])
assert not target.is_absolute() and ".." not in target.parts
assert target.parts[0] == "models"
official = checkpoint["accepted_files"][0]
assert official["sha256"] == "c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2"
print(source["official_repository"] + "/resolve/" + source["huggingface_revision"] + "/" + checkpoint["filename"])
print(checkpoint["default_local_path"])
print(official["sha256"])
print(official["bytes"])
PY
)
if [[ ${#MODEL_FIELDS[@]} -ne 4 ]]; then
    printf '%s\n' 'Could not read the pinned model source manifest' >&2
    exit 1
fi

MODEL_URL="${MODEL_FIELDS[0]}"
MODEL_PATH="$PROJECT_ROOT/${MODEL_FIELDS[1]}"
MODEL_SHA256="${MODEL_FIELDS[2]}"
MODEL_BYTES="${MODEL_FIELDS[3]}"
MODEL_DIRECTORY="$(dirname -- "$MODEL_PATH")"

printf 'Official URL: %s\nTarget: %s\nBytes: %s\nSHA256: %s\n' \
    "$MODEL_URL" "$MODEL_PATH" "$MODEL_BYTES" "$MODEL_SHA256"
if [[ "$DOWNLOAD" -eq 0 ]]; then
    printf '%s\n' 'Dry-run only: no network access or model-directory writes.'
    exit 0
fi

command -v curl >/dev/null
if [[ -e "$MODEL_DIRECTORY" || -L "$MODEL_DIRECTORY" ]]; then
    printf 'Refusing existing destination: %s\n' "$MODEL_DIRECTORY" >&2
    exit 1
fi
AVAILABLE_BYTES="$(df -PB1 "$PROJECT_ROOT/models" | awk 'NR==2 {print $4}')"
if [[ "$AVAILABLE_BYTES" -lt $((MODEL_BYTES * 2)) ]]; then
    printf '%s\n' 'Insufficient free disk space: reserve at least twice the file size' >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$MODEL_DIRECTORY")"
mkdir -- "$MODEL_DIRECTORY"
cp --no-clobber -- "$MODEL_SOURCES" "$MODEL_DIRECTORY/model_sources.json"
PARTIAL_PATH="$MODEL_PATH.partial"
trap 'printf "Download/check failed; preserving destination and partial file: %s\n" "$MODEL_DIRECTORY" >&2' ERR
curl --fail --location --retry 3 --output "$PARTIAL_PATH" "$MODEL_URL"
ACTUAL_BYTES="$(stat -c '%s' -- "$PARTIAL_PATH")"
[[ "$ACTUAL_BYTES" -eq "$MODEL_BYTES" ]]
printf '%s  %s\n' "$MODEL_SHA256" "$PARTIAL_PATH" | sha256sum -c -
[[ ! -e "$MODEL_PATH" ]]
mv --no-clobber -- "$PARTIAL_PATH" "$MODEL_PATH"
printf '%s  %s\n' "$MODEL_SHA256" "$(basename -- "$MODEL_PATH")" > "$MODEL_PATH.sha256"
printf '%s\n' 'Download and SHA256 verification complete. GPU inference has not started.'
