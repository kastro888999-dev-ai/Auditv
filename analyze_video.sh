#!/usr/bin/env bash
# Convenience wrapper for agents to analyze a video and produce Markdown.
# Usage:
#   ./analyze_video.sh <video_file_or_url> [output.md] [whisper_model] [interval] [extra flags]
#
# Extra flags (from 5th arg on) are passed to the Python CLI, e.g.:
#   --no-llm              Skip Ollama analysis
#   --output-dir DIR      Base dir for the .md and frames (default: cwd for URLs,
#                         next to the video for local files)
#   --frames-dir DIR      Explicit frames dir
#
# Example:
#   ./analyze_video.sh /home/user/video.mp4
#   ./analyze_video.sh /home/user/video.mp4 /home/user/out.md small 5
#   ./analyze_video.sh "https://youtube.com/watch?v=..." out.md tiny 15 --no-llm
#   ./analyze_video.sh "https://youtube.com/watch?v=..." "" tiny 15 --output-dir /home/user/reports

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/venv/bin/python"

VIDEO="${1:?Usage: analyze_video.sh <video_file_or_url> [output.md] [model] [interval] [--no-llm]}"
OUTPUT="${2:-}"
MODEL="${3:-small}"
INTERVAL="${4:-10}"
EXTRA_ARGS=()

if [[ -n "$OUTPUT" ]]; then
    EXTRA_ARGS+=( --output "$OUTPUT" )
fi

# Pass any additional flags (e.g. --no-llm)
for arg in "${@:5}"; do
    EXTRA_ARGS+=( "$arg" )
done

if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] Virtualenv not found. Create it with: python3 -m venv venv"
    echo "        then install deps: ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

exec "$VENV_PY" "$SCRIPT_DIR/video_to_md.py" \
    --video "$VIDEO" \
    --model "$MODEL" \
    --interval "$INTERVAL" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
