# AuditV — Video → Markdown Analyzer

Herramienta que convierte un video en un informe Markdown estructurado:

1. **Transcribe** el audio con **Whisper (local, gratis)**, usando GPU si está disponible (CUDA) o CPU automáticamente.
2. **Extrae frames** representativos cada N segundos (imágenes capturadas).
3. **Analiza el contenido** con un **LLM local vía Ollama** para obtener resumen, ideas principales ordenadas, conceptos clave y conclusiones. Si la transcripción es larga, se **divide en partes de ~6 000 caracteres** y se analiza cada parte, fusionando después ideas/conclusiones de todo el audio (los timestamps se ajustan al segundo real del video).

## Requisitos

- `ffmpeg` instalado (extracción de audio y frames)
- `ollama` corriendo en `http://localhost:11434` (para el análisis de ideas)
- Un modelo local en Ollama (ya tienes `qwen3.5:4b`)

## Instalación

```bash
# 1. Crear el entorno virtual (la primera vez)
python3 -m venv venv

# 2. Instalar dependencias
./venv/bin/pip install -r requirements.txt
```

## Uso

### CLI directo

```bash
./venv/bin/python video_to_md.py --video video.mp4 --output informe.md --model small --interval 10
```

Opciones:

| Flag | Descripción | Default |
|------|-------------|---------|
| `--video / -v` | Ruta al video (obligatorio) | — |
| `--output / -o` | Ruta del `.md` de salida | `<video>_analysis.md` |
| `--output-dir` | Directorio base para el `.md` y los frames | junto al video (URLs: `informes/`) |
| `--frames-dir` | Directorio explícito para los frames | `<output-dir>/frames_<video>` |
| `--model / -m` | Tamaño del modelo Whisper (`base`, `small`, `medium`, `large`) | `small` |
| `--interval / -i` | Segundos entre frames capturados | `10` |
| `--no-frames` | No extraer frames (solo transcripción + análisis). Los archivos de **solo audio** (mp3, m4a, wav, ogg, opus, flac, aac…) se transcriben igual y los frames se omiten automáticamente | `off` |
| `--llm` | Modelo de Ollama para el análisis | `qwen3.5:4b` |
| `--device` | Dispositivo para Whisper: `auto`, `cpu` (seguro en portátiles) o `cuda` | `auto` |
| `--cookies` | Archivo de cookies (Netscape) para descargas con login | — |
| `--cookies-from-browser` | Reutiliza cookies de un navegador (`chrome`, `firefox`, `chromium`, con `:perfil` opcional) para plataformas donde estás logeado | — |
| `--live-from-start` | Si la URL es un directo soportado, descarga desde el inicio de la transmisión | `off` |
| `--autoclean` | Qué hacer con los archivos descargados/generados al terminar: `ask` (preguntar), `keep` o `delete` | `ask` |

### Wrapper para agentes

```bash
./analyze_video.sh video.mp4 [output.md] [model] [interval]
```

### Desde Python (función importable)

```python
from tools.video_analyzer import analyze_video

md = analyze_video("/ruta/video.mp4", whisper_model="small", frame_interval=10)
print(md)  # ruta del markdown generado
```

## Salida (estructura del .md)

```
# 🎬 Análisis: <video>
- Generado: <fecha>
- Modelo Whisper: small (cuda|cpu)
- Modelo LLM: qwen3.5:4b

## 🏷️ Título sugerido
## 📌 Resumen ejecutivo
## 🧠 Ideas principales
   1. Idea 1 `[MM:SS]`
## 🔑 Conceptos clave
## ✅ Conclusiones
## 📝 Transcripción (con timestamps)
## 🖼️ Frames capturados (imágenes con timestamps)
```

Los frames se guardan en una carpeta `frames_<video>/` junto al `.md` (o donde indiques con `--frames-dir` / `--output-dir`). Las referencias a los frames en el `.md` son rutas relativas cuando están junto al informe.

## URLs (YouTube, Google Drive, etc.)

- El video se descarga en la carpeta `descargas/` del proyecto.
- Las salidas por defecto (`.md` + frames) se guardan en `informes/` del proyecto.
- Al terminar se pregunta si quieres **borrar o conservar** los archivos descargados y generados (controla este comportamiento con `--autoclean ask|keep|delete`). En sesiones no interactivas se conservan por defecto.
- Se puede sobreescribir todo con `--output-dir`, `--output` o `--frames-dir`.

### Contenido con login y directos

Para plataformas donde ya estás logeado (algunas aulas virtuales, webinars con
stream HLS, YouTube autenticado) usa cookies de tu navegador:

```bash
./analyze_video.sh "https://plataforma/..." informe.md tiny 10 --cookies-from-browser chrome
```

- `--cookies-from-browser chrome|firefox|chromium[:perfil]` reutiliza tu sesión.
- `--cookies archivo.txt` usa un archivo de cookies en formato Netscape (lo exportas con una extensión como "Get cookies.txt").
- `--live-from-start` intenta descargar un directo desde su inicio (solo sitios que yt-dlp soporta para eso).

**Limitación**: Google Meet, Zoom y Teams son llamadas WebRTC, no streams
descargables; para esas, graba la pantalla (OBS, SimpleScreenRecorder) y
analiza el archivo resultante.

## Resolución de rutas de salida

1. Si se pasa `--frames-dir`, los frames van ahí.
2. Si se pasa `--output-dir`, el `.md` y los frames van a ese directorio.
3. Si solo se pasa `--output`, el `.md` va ahí y los frames junto a él.
4. Si no se pasa nada: junto al video (archivos locales) o en `informes/` del proyecto (URLs).

## Variables de entorno (opcionales)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3.5:4b` | Modelo de Ollama |
| `WHISPER_MODEL` | `small` | Modelo Whisper por defecto |
| `OLLAMA_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `OLLAMA_NUM_CTX` | `4096` | Contexto (tokens) para el análisis con Ollama |
| `OLLAMA_NUM_GPU` | `0` (CPU) | `0` = LLM solo CPU (no calienta la GPU, seguro); `1`/`-1` = usa la GPU si el equipo aguanta |
| `OLLAMA_NUM_THREADS` | (sin usar) | Limitar hilos del LLM |
| `AUDITV_DEVICE` | `auto` | Igual que `--device` (`auto`/`cpu`/`cuda`) |
| `AUDITV_GPU_TEMP_WARN` / `AUDITV_GPU_TEMP_ABORT` | `80` / `92` | Temperatura GPU (°C) de aviso / aborto preventivo |
| `AUDITV_DOWNLOAD_DIR` | `descargas/` en el proyecto | Carpeta donde se descargan los videos de URLs |
| `AUDITV_OUTPUT_DIR` | `informes/` en el proyecto | Carpeta base por defecto para las salidas de URLs |

## Apuntes de reuniones en vivo

`tools/live_meeting.py` transcribe una reunión en tiempo real con Whisper
local y guarda un `.txt` con timestamps. Al finalizar (Ctrl+C), con `--notes`
genera automáticamente el informe de apuntes (`_apuntes.md` con resumen,
ideas, discusiones, conceptos y conclusiones) vía Ollama — el mismo análisis
del modo `--transcript`.

```bash
./venv/bin/python tools/live_meeting.py --list-sources
./venv/bin/python tools/live_meeting.py --source auto --model base --notes -o "informes/reunion.txt"
```

Cómo funciona:
- Captura el audio del **sistema** (`.monitor`, lo que suena por los altavoces:
  ideal para Meet/Teams/Zoom en el navegador) o del **micrófono** (`alsa_input`),
  con ffmpeg sobre PipeWire/PulseAudio.
- Transcribe en trozos de ~4 s con Whisper (`base` en CPU por defecto; evita
  calentar la GPU en reuniones largas) y escribe cada frase con su `[MM:SS]`.
- `Ctrl+C` guarda la transcripción y (si usas `--notes`) genera los apuntes.
- Al final la transcripción se puede reprocesar cuando quieras:
  `./analyze_video.sh "" "" small 15 --transcript reunion.txt --output-dir informes`.

No añade dependencias: usa ffmpeg + openai-whisper, que ya requiere el
proyecto. En hardware muy limitado puedes empezar con `--model tiny`.

## Interfaz gráfica (Gradio)

`tools/auditv_gui.py` es una interfaz web local para hacer todo desde el navegador:

```bash
./venv/bin/pip install "gradio>=6"     # solo la primera vez
./venv/bin/python tools/auditv_gui.py  # abre http://127.0.0.1:7860
```

**Doble clic (KDE):** ya está instalado el acceso `AuditV` en el menú de
aplicaciones (`.desktop` en `~/.local/share/applications/`). Lo ejecutas
desde el arrancador o con doble clic en `auditv.desktop` / `start_gui.sh`
del proyecto; abre la GUI en una ventana de terminal (konsole) y el navegador
se abre solo cuando el servidor está listo.

Tiene tres pestañas:
- **Video / URL**: subir un video local, dar una ruta o pegar una URL; elige
  carpeta de salida, modelo Whisper, device, intervalo de frames, cookies del
  navegador y autoclean. Muestra el progreso en vivo y devuelve la ruta del
  informe.
- **Apuntes**: elige un `.txt` (transcripción) y genera el informe `_apuntes.md`
  (resumen, ideas, discusiones, conceptos, conclusiones).
- **Reunión en vivo**: elige fuente de audio (sistema o micrófono), inicia la
  captura, ves la transcripción en tiempo real, la detienes con un botón y
  generas los apuntes automáticamente.

La interfaz ejecuta los mismos scripts (`video_to_md.py` y
`live_meeting.py`), así que no hay lógica duplicada.

Detalles de la interfaz:
- **Modelo de IA local**: selector desplegable que lista los modelos de tu
  Ollama local (botón `↻ Actualizar modelos` para recargar).
- **Seleccionar Carpeta de Guardado**: botón que abre el selector nativo de
  carpetas del escritorio (KDE/GNOME/Windows) en cada campo de salida.
- La barra inferior de Gradio (Run, API, configuración) se muestra arriba,
  centrada bajo el título.

## Instalar en otro equipo (Linux y Windows)

AuditV funciona como **programa portable**: copia la carpeta del proyecto a
otro equipo y ejecuta el instalador de tu sistema. Solo instala dependencias
dentro de `venv\` de la propia carpeta (no toca el sistema):

| Sistema | Instalar | Arrancar GUI |
|---------|----------|--------------|
| Linux/macOS | `./instalar.sh` | `./start_gui.sh` (o doble clic en `auditv.desktop`) |
| Windows | doble clic en `instalar.bat` | doble clic en `iniciar_gui.bat` |

Requisitos externos en el equipo de destino (independientes de este proyecto):
- Python 3.10+ en el PATH.
- `ffmpeg` (recomendado) para videos/transcripción.
- `Ollama` solo si se quiere el análisis con IA local; sin él, el resto funciona.

### Portable de un solo archivo (opcional)

También se puede empaquetar un ejecutable único por sistema con PyInstaller,
sin necesidad de Python en el destino:

```bash
./venv/bin/pip install pyinstaller
./venv/bin/pyinstaller --onefile --name auditv --collect-all openai_whisper \
  --collect-all whisper --add-data "video_to_md.py:." --add-data "tools:tools" \
  tools/auditv_gui.py
```

Hay que compilar **en cada sistema** (el ejecutable de Linux no corre en
Windows). AVISO: por torch/whisper el archivo pesa varios GB, y sigue
haciendo falta `ffmpeg` y `Ollama` aparte; el arranque tarda más al
desempaquetar. Para un uso normal es mejor la opción portable de carpeta.

> Nota: en Python 3.14 hay que usar Gradio 6+ (Gradio 4 falla al compilar
> Pillow). Opciones de lanzamiento: `--port`, `--no-browser` y `--share`.

## Prevención de apagados por temperatura

En portátiles con refrigeración justa (p. ej. GTX 1060 Mobile) el análisis
LLM de Ollama sobre la GPU puede disparar la temperatura y **apagar el equipo
a mitad de la ejecución**. Para evitarlo la herramienta:

- Monitorea la temperatura de la GPU (`nvidia-smi`): avisa desde 80°C
  (ajustable con `AUDITV_GPU_TEMP_WARN`). Si alcanza 92°C
  (`AUDITV_GPU_TEMP_ABORT`) durante una consulta LLM, **esa consulta y las
  siguientes se reintentan en CPU** en vez de abortar, para que ningún bache
  se quede sin resumen/conclusiones.
- Guarda la transcripción (`_transcripcion.txt`) y un **informe parcial** con
  transcripción + frames ANTES de empezar el análisis con Ollama, para no
  perder progreso si el equipo se apaga.
- Usa 4096 tokens de contexto por defecto (antes 8192).

Si el PC se apagó o está caliente, reejecuta con menos carga de GPU:

```bash
AUDITV_GPU_TEMP_WARN=75 OLLAMA_NUM_GPU=0 OLLAMA_NUM_CTX=2048 \
  ./analyze_video.sh video.mp4 informe.md tiny 15 --autoclean keep --device cpu
```
