#!/usr/bin/env bash
# Instala AuditV en esta carpeta (modo portable, Linux/macOS).
# Requiere: Python 3.10+ y, opcionalmente, ffmpeg + Ollama en el PATH.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "→ Python: $($PY --version 2>&1)"
"$PY" -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install "gradio>=6"

echo ""
if command -v ffmpeg >/dev/null 2>&1; then echo "→ ffmpeg: OK ($(command -v ffmpeg))"; else echo "→ ffmpeg: NO instalado (instálalo: sudo pacman -S ffmpeg / sudo apt install ffmpeg)"; fi
if command -v ollama >/dev/null 2>&1; then echo "→ ollama: OK  ($(command -v ollama))"; else echo "→ ollama: NO instalado (solo hace falta para el análisis con IA)"; fi
echo ""
echo "Listo. Para arrancar la GUI: ./start_gui.sh   (o doble clic en auditv.desktop)"