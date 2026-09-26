# AuditV — Video → Markdown Analyzer

Herramienta que convierte un video en un informe Markdown estructurado:

1. **Transcribe** el audio con **Whisper (local, gratis)**, usando GPU si está disponible (CUDA) o CPU automáticamente.
2. **Extrae frames** representativos cada N segundos (imágenes capturadas).
3. **Analiza el contenido** con un **LLM local vía Ollama** para obtener resumen, ideas principales ordenadas, conceptos clave y conclusiones. Si la transcripción es larga, se **divide en partes de ~6 000 caracteres** y se analiza cada parte, fusionando después ideas/conclusiones de todo el audio (los timestamps se ajustan al segundo real del video).

## Requisitos

- Python 3.10+ y `ffmpeg` en el PATH (extracción de audio y frames)
- `ollama` corriendo en `http://localhost:11434` (para el análisis de ideas)
- Al menos un modelo descargado en Ollama (`ollama pull qwen3.5:4b` o el que
  prefieras). **No hace falta saber cuál**: con `--llm auto` (por defecto) se
  detectan los tuyos y se elige el mejor que quepa en tu equipo.

### Sistemas operativos

Funciona en **Linux, Windows y macOS**. Lo que cambia según el sistema:

| | Linux | Windows | macOS |
|---|---|---|---|
| GPU para Whisper | CUDA si el torch la soporta | CUDA | **MPS** (chip Apple Silicon), automática |
| GPU del panel | `nvidia-smi` | `nvidia-smi` | la de Apple; sin temperatura ni VRAM |
| Audio de reuniones | PipeWire/PulseAudio | DirectShow | AVFoundation |
| Instalación | `instalar.sh` | `instalar.bat` | `instalar.sh` (con Homebrew para ffmpeg) |
| Arranque de la GUI | `start_gui.sh` | `iniciar_gui.bat` | `start_gui.sh` |

Detalles que importan:

- **Windows**: usa doble clic en `instalar.bat` y luego `iniciar_gui.bat`. No
  hace falta instalar `yt-dlp` a mano (se instala con el resto en el venv).
  Para las reuniones en vivo necesitas un dispositivo de audio que ffmpeg
  pueda abrir; `--list-sources` muestra los nombres exactos.
- **macOS**: `brew install ffmpeg`. En los Mac con chip Apple Silicon Whisper
  corre en la GPU (Metal) sin tocar nada. En la reunión en vivo, el nombre de
  la fuente debe ser el de un **dispositivo de entrada de audio** (no la
  cámara).
- Si `psutil` no está instalado, todo funciona igual: el panel solo pierde las
  cifras de CPU/RAM por proceso.

## Instalación

```bash
# 1. Crear el entorno virtual (la primera vez)
python3 -m venv venv

# 2. Instalar dependencias
./venv/bin/pip install -r requirements.txt
```

En Windows, `instalar.bat` hace los dos pasos (y `iniciar_gui.bat` arranca la
interfaz).

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
| `--model / -m` | Modelo Whisper: `auto` (elige según tu equipo y la duración del audio) o `tiny`/`base`/`small`/`medium`/`large`/`large-v3`/`large-v3-turbo` | `auto` |
| `--interval / -i` | Segundos entre frames capturados | `10` |
| `--no-frames` | No extraer frames (solo transcripción + análisis). Los archivos de **solo audio** (mp3, m4a, wav, ogg, opus, flac, aac…) se transcriben igual y los frames se omiten automáticamente | `off` |
| `--llm` | Modelo de Ollama: `auto` (el mejor instalado que quepa en tu equipo) o el nombre exacto (`qwen3:8b`, `gemma3:4b`, …) | `auto` |
| `--device` | Dispositivo para Whisper: `auto`, `cpu` (seguro en portátiles) o `cuda` | `auto` |
| `--ollama-gpu` | Dónde corre el LLM: `auto` (GPU si la tarjeta es potente y no está caliente, si no CPU), `gpu` (forzada, con la protección térmica) o `cpu` | `auto` |
| `--ollama-gpu-index` | Índice de la GPU para el LLM (`auto` = la que elija Ollama) | `auto` |
| `--gpu-temp-warn` / `--gpu-temp-abort` / `--gpu-temp-resume` | Umbrales de temperatura (°C) de aviso, descanso y vuelta al trabajo | `80` / `92` / `75` |
| `--cookies` | Archivo de cookies (Netscape) para descargas con login | — |
| `--cookies-from-browser` | Reutiliza cookies de un navegador (`chrome`, `firefox`, `chromium`, con `:perfil` opcional) para plataformas donde estás logeado | — |
| `--live-from-start` | Si la URL es un directo soportado, descarga desde el inicio de la transmisión | `off` |
| `--autoclean` | Qué hacer con los archivos descargados/generados al terminar: `ask` (preguntar), `keep` o `delete` | `ask` |

### Wrapper para agentes

```bash
./analyze_video.sh video.mp4 [output.md] [model] [interval]
```

`model` e `interval` valen `auto`/`10` por defecto, así que lo normal es
`./analyze_video.sh video.mp4`.

### Desde Python (función importable)

```python
from tools.video_analyzer import analyze_video

md = analyze_video("/ruta/video.mp4")            # todo se detecta solo
md = analyze_video("/ruta/video.mp4", whisper_model="small", frame_interval=10)
print(md)  # ruta del markdown generado
```

## Detección automática de hardware y modelos

Nada está atado a una máquina concreta: al arrancar, el programa pregunta al
sistema qué hay (`hardware.py`) y decide con esa información.

**Modelos de Ollama** (`--llm auto`, por defecto): lista los que tienes
instalados y elige el más grande que **cabe cómodo** en tu equipo — con ~65 %
de la VRAM si el LLM va a la GPU (ideal ~8B), o ~4,5 GB de RAM si va en CPU
(ideal ~4B). Entre los que empatan, se prefieren las familias que siguen mejor
las instrucciones y el JSON (`qwen`, `gemma3`, `llama3.3`…).

**Modelo de Whisper** (`--model auto`, por defecto): el mayor que quepa en
~75 % de la VRAM si el device acaba siendo `cuda`; en CPU se elige según la
duración del audio (`small` hasta 15 min, `base` hasta 45 min, `tiny` para
horas, donde la diferencia de tiempo es enorme).

**Device** (`--device auto`): la GPU solo se usa si este PyTorch la soporta de
verdad. Una GTX 1060 (sm_61), por ejemplo, cae siempre a CPU con los builds
actuales de torch, aunque la tarjeta pueda ejecutar el LLM de Ollama.

**Potencia de la GPU**: se clasifica en `débil` / `media` / `potente` /
`muy_potente` según VRAM y compute capability. El LLM va a la GPU por defecto a
partir de `media`; con una GPU débil o sin GPU va a CPU para no recalentar el
equipo. Se puede forzar con `--ollama-gpu gpu|cpu` o en la GUI.

## Salida (estructura del .md)

```
# 🎬 Análisis: <video>
- Generado: <fecha>
- Modelo Whisper: small (cuda|cpu)
- Modelo LLM: qwen3.5:4b
- Motor: <descripción del equipo y de dónde corrieron Whisper y el LLM>

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
| `OLLAMA_MODEL` | `auto` | Modelo de Ollama (`auto` = el mejor instalado que quepa) |
| `WHISPER_MODEL` | `auto` | Modelo Whisper (`auto` = el adecuado al equipo y al audio) |
| `OLLAMA_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `OLLAMA_NUM_CTX` | `auto` | Contexto (tokens) para el análisis con Ollama (`auto` = 4096, o 8192 si hay VRAM de sobra) |
| `OLLAMA_NUM_GPU` | `auto` | `auto` = GPU si la tarjeta es potente y está fría; `-1` = forzar GPU; `0` = solo CPU |
| `OLLAMA_GPU_INDEX` | (auto) | Índice de la GPU para el LLM (`main_gpu` de Ollama) |
| `OLLAMA_NUM_THREADS` | (sin usar) | Limitar hilos del LLM |
| `AUDITV_DEVICE` | `auto` | Igual que `--device` (`auto`/`cpu`/`cuda`) |
| `AUDITV_GPU_TEMP_WARN` / `AUDITV_GPU_TEMP_ABORT` / `AUDITV_GPU_TEMP_RESUME` | `80` / `92` / `75` | Temperatura GPU (°C) de aviso, de descanso y de vuelta al trabajo |
| `AUDITV_GPU_TEMP_INTERVAL` | `5` | Segundos entre mediciones de temperatura |
| `AUDITV_TRANSCRIBE_CHUNK_SEC` | `300` | Tamaño (s) de los lotes de audio en GPU (para poder descansar la tarjeta) |
| `AUDITV_STATUS_FILE` | (vacío) | Fichero JSON de estado que escribe el CLI; la GUI lo usa para pintar el panel de GPU |
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
./venv/bin/python tools/live_meeting.py --source auto --notes -o "informes/reunion.txt"
```

Cómo funciona:
- Captura el audio del **sistema** (`.monitor`, lo que suena por los altavoces:
  ideal para Meet/Teams/Zoom en el navegador) o del **micrófono** (`alsa_input`),
  con ffmpeg sobre PipeWire/PulseAudio.
- Transcribe en trozos de ~4 s con Whisper y escribe cada frase con su
  `[MM:SS]`. El modelo y el device se detectan solos (`--model auto`,
  `--device auto`); si la GPU se calienta durante la reunión, la tarjeta
  descansa y el resto se transcribe en CPU sin cortar la captura.
- `Ctrl+C` guarda la transcripción y (si usas `--notes`) genera los apuntes.
- Al final la transcripción se puede reprocesar cuando quieras:
  `./analyze_video.sh "" "" auto 15 --transcript reunion.txt --output-dir informes`.

No añade dependencias: usa ffmpeg + openai-whisper, que ya requiere el
proyecto. En hardware muy limitado puedes forzar `--model tiny --device cpu`.

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

Encima de las pestañas hay un **panel de equipo** que se actualiza cada 2 s:

- **GPU**: nombre de la tarjeta, VRAM libre, temperatura (verde/naranja/rojo
  según los umbrales) y un indicador de estado —`en uso`, `en descanso`, `libre`
  o `no detectada`— para saber si la GPU está trabajando o no.
- **Barra de sistema**: consumo de CPU, RAM y VRAM con barrita de progreso
  (verde < 60 %, naranja hasta 85 %, rojo por encima), más núcleos, RAM libre y
  swap.
- **Por componente**: qué modelo de Whisper y de IA local se han elegido, en qué
  device corren y cuánta RAM y CPU consume cada uno en este momento (`%` de CPU
  por proceso, como en `top`; puede pasar de 100 con varios núcleos), más la
  VRAM que reserva cada uno.

El consumo se mide con `psutil` (procesos) y con la API de Ollama (`/api/ps`,
para la VRAM del modelo). Si `psutil` no está instalado, el panel sigue
funcionando: solo se pierden las cifras de CPU/RAM.

La interfaz ejecuta los mismos scripts (`video_to_md.py` y
`live_meeting.py`), así que no hay lógica duplicada.

Detalles de la interfaz:
- **Todo es `auto` por defecto**: el desplegable de modelos de Ollama se llena
  con los que tienes instalados (`auto` + botón `↻ Actualizar modelos`), el de
  Whisper muestra el recomendado para tu equipo, y «Dónde corre la IA» trae
  `auto`/`gpu`/`cpu` con la recomendación de tu hardware en la descripción.
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

- Mide la temperatura de la GPU (`nvidia-smi`) **cada 5 segundos**
  (`AUDITV_GPU_TEMP_INTERVAL`). Avisa desde 80 °C (`AUDITV_GPU_TEMP_WARN`).
- Al llegar a 92 °C (`AUDITV_GPU_TEMP_ABORT`) la GPU **entra en descanso**: se
  suelta el modelo de Ollama de la VRAM (`keep_alive=0`) y el lote actual
  **termina en CPU sin perder su resultado**. Los lotes siguientes también van
  en CPU, y en cuanto la tarjeta baja de 75 °C (`AUDITV_GPU_TEMP_RESUME`)
  el trabajo **vuelve a la GPU** solo (histéresis, para no oscilar).
- La transcripción en GPU va **por lotes de 5 minutos**
  (`AUDITV_TRANSCRIBE_CHUNK_SEC`): entre lote y lote se comprueba la
  temperatura, así que si se calienta, el lote en curso y los siguientes se
  terminan en CPU. Cada lote se guarda en un checkpoint
  (`_transcripcion.txt`) al terminar.
- Guarda la transcripción (`_transcripcion.txt`) y un **informe parcial** con
  transcripción + frames ANTES de empezar el análisis con Ollama, para no
  perder progreso si el equipo se apaga.
- La GUI lo enseña todo en el panel de equipo: temperatura, si la tarjeta está en
  descanso y por qué, en qué device corre cada cosa y cuánta CPU/RAM/VRAM está
  usando cada componente.
- Contexto de Ollama ajustado al equipo (4096 tokens, 8192 si hay VRAM de
  sobra) en vez de 8192 fijos.

Si el PC se apagó o está caliente, baja el listón explícitamente:

```bash
AUDITV_GPU_TEMP_WARN=75 ./analyze_video.sh video.mp4 informe.md auto 15 \
  --autoclean keep --device cpu --ollama-gpu cpu
```

Y si solo quieres que decida él: no pases nada y usa `auto` (lo normal).
