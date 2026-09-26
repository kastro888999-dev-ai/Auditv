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

En Windows, ffmpeg se instala con `winget install ffmpeg` o desde
<https://ffmpeg.org/download.html> (añadirlo al PATH). AuditV no necesita nada
más: `yt-dlp` y el resto llegan con el venv.

## 2. Obtener el proyecto

```bash
git clone <url_del_repo> AuditV
cd AuditV
```

O copiar la carpeta manualmente (rsync, USB...). **No copies `venv/`**:
hay que crearlo en la máquina nueva.

## 3. Instalar dependencias Python

```bash
# Linux / macOS
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

En Windows, doble clic en `instalar.bat`, que hace lo mismo con
`venv\Scripts\python.exe`. Después se arranca con `iniciar_gui.bat`.

Esto instala:

- `openai-whisper` — transcripción local
- `torch` — backend de Whisper
- `yt-dlp` — descarga de videos desde URLs (YouTube, etc.)
- `gradio` — interfaz
- `psutil` — cifras de CPU/RAM del panel (opcional, pero recomendado)

El modelo de Whisper se descarga automáticamente en el primer uso
(`~75 MB` para `small`, `~1.5 GB` para `large`).

## 4. Descargar un modelo de Ollama

Con uno basta; el programa elige el mejor de los que tengas (ver §7):

```bash
ollama pull qwen3.5:4b        # ~3,4 GB, va bien en CPU
# o, si tu GPU tiene VRAM de sobra:
ollama pull qwen3:8b          # ~5,4 GB
```

Verifica que el servidor responde (debe estar corriendo en segundo plano):

```bash
ollama list   # debe mostrar los modelos que hayas descargado
```

## 5. Probar

```bash
./analyze_video.sh <un_video_corto.mp4> prueba.md auto 5 --output-dir /tmp/prueba_auditv
```

Si genera el `.md` con resumen, transcripción y frames, todo funciona.

## 6. Uso con un agente

- Las instrucciones del agente están en **`AGENTS.md`** (en la raíz), que es el
  nombre que leen Codex, Cursor, Cline, Windsurf y compañía. No hay
  `CLAUDE.md`: si usas Claude Code, apúntale a ese archivo.
- El agente debe ejecutar `./analyze_video.sh ...` desde la raíz del proyecto
  (en Windows, `venv\Scripts\python tools\video_analyzer.py --video ...`).
- Según `AGENTS.md`, el agente debe preguntar siempre al usuario dónde guardar
  el informe y los frames antes de ejecutar (`--output-dir` o `--output` +
  `--frames-dir`).

## 7. Tu equipo se ajusta solo

No hay que configurar nada más: al arrancar, AuditV detecta

- las GPUs NVIDIA (modelo, VRAM, compute capability) y su nivel de potencia,
- si ese PyTorch puede usarlas de verdad (algunas tarjetas antiguas, como la
  GTX 1060 `sm_61`, no son compatibles con los builds actuales de torch: Whisper
  irá a CPU aunque la GPU pueda ejecutar el LLM de Ollama),
- los modelos que tienes en Ollama y cuál es el mejor que cabe,
- qué modelos de Whisper tienes en caché.

y de ahí saca el modelo de Whisper, el device, dónde corre el LLM y los límites
de temperatura. Los log iniciales lo dicen todo:

```
[DEVICE] GPU 0: NVIDIA GeForce RTX 4070 (12 GB, sm_8.9, nivel potente)
[DEVICE] GPU para Whisper: NVIDIA GeForce RTX 4070 (12 GB, sm_89)
[LLM] IA local: GPU potente: el LLM irá a la GPU (si la temperatura lo permite).
[LLM] Modelos en Ollama: gemma3:4b, qwen3.5:4b, qwen3:8b -> usando 'qwen3:8b' (5,2 GB, con GPU).
```

Si prefieres decidir tú, está todo a mano: `--model`, `--llm`, `--device`,
`--ollama-gpu`, `--gpu-temp-warn/abort/resume`. Con `auto` (lo normal) no hace
falta.

## Solución de problemas

| Problema | Causa / solución |
|----------|------------------|
| `command not found: yt-dlp` | Ejecuta con el venv del proyecto (`./venv/bin/python video_to_md.py ...`) o activa el venv |
| `Ollama HTTP error` / análisis vacío | Ollama no corre (`ollama serve`) o no hay ningún modelo (`ollama pull qwen3.5:4b`) |
| `venv/bin/python: No such file` | Falta crear el venv (paso 3) |
| Whisper muy lento | Sin GPU compatible: `auto` ya elige `tiny`/`base` en CPU; se puede forzar con `--model tiny` |
| El equipo se calienta o se apaga | Baja el listón: `--ollama-gpu cpu --device cpu`, o `AUDITV_GPU_TEMP_ABORT=80`. Con `auto` la GPU ya descansa sola y el trabajo sigue en CPU sin perderse |
| No se descarga el video | Actualiza yt-dlp: `./venv/bin/pip install -U yt-dlp` (YouTube cambia con frecuencia) |
| Las secciones del LLM salen vacías | Contexto insuficiente: sube `OLLAMA_NUM_CTX` (por defecto 4096, o 8192 si hay VRAM) |
