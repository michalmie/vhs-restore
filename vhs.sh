#!/usr/bin/env bash
# Activate the vhs-env virtual environment and run the pipeline.
# Usage: ./vhs.sh <subcommand> [args...]
#   ./vhs.sh restore capture.mkv output.mkv --test --test-sample
#   ./vhs.sh enhance restored.mkv out.mp4 --warmth 0.3 --cas
#   ./vhs.sh analyze capture.mkv
#   ./vhs.sh trim capture.mkv

VENV="${VHS_VENV:-$HOME/vhs-env}"
SCRIPT="$(cd "$(dirname "$0")" && pwd)/pipeline/restore.py"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "error: venv not found at $VENV" >&2
    echo "  Run setup first:  bash pipeline/setup_ubuntu.sh" >&2
    exit 1
fi

source "$VENV/bin/activate"
exec python "$SCRIPT" "$@"
