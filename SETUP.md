# AuditV — Instalación en otra máquina

Instrucciones para poner en marcha esta herramienta en un computador nuevo
y usarla con cualquier agente (opencode, Claude Code, etc.).

## 1. Requisitos del sistema

| Requisito | Verificación | Nota |
|-----------|--------------|------|
| Python 3.10+ | `python3 --version` | Para el venv y Whisper |
| ffmpeg | `ffmpeg -version` | Extracción de audio y frames |
| Git | `git --version` | Para clonar/copiar el proyecto |
| Ollama | `ollama --version` | Análisis LLM local (http://localhost:11434) |
| ~2-3 GB de disco | — | venv + modelos (más espacio para los videos) |

Instalación de ffmpeg según la distribución:

```bash
# Debian/Ubuntu
sudo apt install ffmpeg
# Fedora
sudo dnf install ffmpeg
# Arch
sudo pacman -S ffmpeg
# macOS
brew install ffmpeg
```

Instalación de Ollama: https://ollama.com/download (Linux, macOS, Windows).

## 2. Obtener el proyecto

```bash
git clone <url_del_repo> AuditV
cd AuditV
```

O copiar la carpeta manualmente (rsync, USB...). **No copies `venv/`**:
hay que crearlo en la máquina nueva.

## 3. Instalar dependencias Python

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Esto instala:

- `openai-whisper` — transcripción local
- `torch` — backend de Whisper
- `yt-dlp` — descarga de videos desde URLs (YouTube, etc.)

El modelo de Whisper se descarga automáticamente en el primer uso
(`~75 MB` para `small`, `~1.5 GB` para `large`).

## 4. Descargar el modelo de Ollama

```bash
ollama pull qwen3.5:4b
```

Verifica que el servidor responde (debe estar corriendo en segundo plano):

```bash
ollama list   # debe mostrar qwen3.5:4b
```

## 5. Probar

```bash
./analyze_video.sh <un_video_corto.mp4> prueba.md tiny 5 --output-dir /tmp/prueba_auditv
```

Si genera el `.md` con resumen, transcripción y frames, todo funciona.

## 6. Uso con un agente

- Copia el contenido de `CLAUDE.md` a las instrucciones del agente
  (o mantenlo en la raíz: la mayoría de los agentes lo leen automáticamente).
- El agente debe ejecutar `./analyze_video.sh ...` desde la raíz del proyecto.
- Según `CLAUDE.md`, el agente debe preguntar siempre al usuario dónde guardar
  el informe y los frames antes de ejecutar (`--output-dir` o `--output` +
  `--frames-dir`).

## Solución de problemas

| Problema | Causa / solución |
|----------|------------------|
| `command not found: yt-dlp` | Ejecuta con el venv del proyecto (`./venv/bin/python video_to_md.py ...`) o activa el venv |
| `Ollama HTTP error` / análisis vacío | Ollama no corre (`ollama serve`) o falta el modelo (`ollama pull qwen3.5:4b`) |
| `venv/bin/python: No such file` | Falta crear el venv (paso 3) |
| Whisper muy lento | Sin GPU compatible: usa modelos `tiny`/`base` |
| No se descarga el video | Actualiza yt-dlp: `./venv/bin/pip install -U yt-dlp` (YouTube cambia con frecuencia) |
| Las secciones del LLM salen vacías | Contexto insuficiente: sube `OLLAMA_NUM_CTX` (default 8192) |
