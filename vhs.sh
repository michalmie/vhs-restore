#!/usr/bin/env bash
# Activate the vhs-env virtual environment and run the pipeline.
#
# No arguments → launch the interactive TUI (recommended)
#   ./vhs.sh
#
# With subcommand → run CLI directly:
#   ./vhs.sh restore capture.mkv output.mkv --test --test-sample
#   ./vhs.sh test    capture.mkv out.mp4 --test-sample --warmth 0.3 --cas
#   ./vhs.sh analyze capture.mkv
#   ./vhs.sh trim    capture.mkv

VENV="${VHS_VENV:-$HOME/vhs-env}"
DIR="$(cd "$(dirname "$0")" && pwd)/pipeline"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "error: venv not found at $VENV" >&2
    echo "  Run setup first:  bash pipeline/setup_ubuntu.sh" >&2
    exit 1
fi

source "$VENV/bin/activate"

if [ $# -eq 0 ]; then
    exec python "$DIR/tui.py"
else
    exec python "$DIR/restore.py" "$@"
fi
