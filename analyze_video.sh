#!/usr/bin/env bash
# Convenience wrapper for agents to analyze a video and produce Markdown.
# Usage:
#   ./analyze_video.sh <video_file_or_url> [output.md] [whisper_model] [interval] [extra flags]
#
# URLs are downloaded to the project's `descargas/` folder and default outputs
# go to `informes/`. After the run the user is asked whether to keep or delete
# the generated files (override with --autoclean keep|delete).
#
# Extra flags (from 5th arg on) are passed to the Python CLI, e.g.:
#   --no-llm              Skip Ollama analysis
#   --output-dir DIR      Base dir for the .md and frames
#   --frames-dir DIR      Explicit frames dir
#   --device cpu          Force Whisper to CPU (recommended on weak-cooling
#                         laptops that shut down on heavy GPU load)
#   --autoclean ask|keep|delete
#                         What to do at the end with the downloaded video and
#                         the generated .md/frames: ask, keep or delete
#
# URLs (YouTube, Drive, etc.) are downloaded into the project's `descargas/`
# folder and default outputs go to the project's `informes/` folder.
#   --autoclean ask|keep|delete
#                         What to do with downloaded/generated files at the end
#
# Example:
#   ./analyze_video.sh /home/user/video.mp4
#   ./analyze_video.sh /home/user/video.mp4 /home/user/out.md small 5
#   ./analyze_video.sh "https://youtube.com/watch?v=..." out.md tiny 15 --no-llm
#   ./analyze_video.sh "https://youtube.com/watch?v=..." "" tiny 15 --autoclean keep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/venv/bin/python"

HAS_TRANSCRIPT=false
for arg in "${@:5}"; do
    if [[ "$arg" == "--transcript" ]]; then
        HAS_TRANSCRIPT=true
        break
    fi
done

if [[ "$HAS_TRANSCRIPT" == "true" && -z "${1:-}" ]]; then
    VIDEO=""
else
    VIDEO="${1:?Usage: analyze_video.sh <video_file_or_url> [output.md] [model] [interval] [flags...]}"
fi
OUTPUT="${2:-}"
MODEL="${3:-small}"
INTERVAL="${4:-10}"
EXTRA_ARGS=()

if [[ -n "$OUTPUT" ]]; then
    EXTRA_ARGS+=( --output "$OUTPUT" )
fi

# Pass any additional flags (e.g. --no-llm, --transcript, --cookies-from-browser)
for arg in "${@:5}"; do
    EXTRA_ARGS+=( "$arg" )
done

if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] Virtualenv not found. Create it with: python3 -m venv venv"
    echo "        then install deps: ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

EXEC_ARGS=()
if [[ -n "$VIDEO" ]]; then
    EXEC_ARGS+=( --video "$VIDEO" )
fi
EXEC_ARGS+=( --model "$MODEL" )
EXEC_ARGS+=( --interval "$INTERVAL" )

exec "$VENV_PY" "$SCRIPT_DIR/video_to_md.py" "${EXEC_ARGS[@]}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
