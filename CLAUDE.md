# AuditV — Herramienta de análisis de video para el agente

Esta carpeta contiene una herramienta que convierte un video en un informe
Markdown estructurado (transcripción + ideas principales + frames).

## Cómo usarla (para el agente)

**Antes de ejecutar**: cuando el usuario pida analizar un video y NO haya
especificado dónde guardar el resultado, el agente DEBE preguntar (en una sola
consulta): dónde guardar el informe y los frames (`--output-dir` o `--output`
+ `--frames-dir`). Solo usa los defaults (junto al video / cwd) si el usuario
confirma explícitamente que no le importa. Nunca dejes que las salidas caigan
en directorios temporales.

El agente puede analizar un video con un único comando:

```bash
cd <carpeta_del_proyecto>  # raíz de AuditV (donde está analyze_video.sh)
./analyze_video.sh <ruta/al/video.mp4> [output.md] [whisper_model] [interval] [flags]

Flags útiles (desde el 5º argumento):
  --no-llm              Análisis Ollama omitido (transcripción + frames)
  --no-frames           No extraer frames (solo transcripción + análisis)
  --output-dir DIR      Base para el .md y los frames
  --frames-dir DIR      Directorio explícito de frames
  --device auto|cpu|cuda
                        Fuerza Whisper a CPU (recomendado en portátiles con
                        refrigeración justa) en vez de auto-detección
  --cookies-from-browser chrome|firefox|chromium[:perfil]
                        Usa cookies del navegador para descargar contenidos
                        de plataformas donde el usuario está logeado
  --cookies ARCHIVO     Archivo de cookies en formato Netscape
  --live-from-start     Descargar directos soportados desde su inicio
  --autoclean ask|keep|delete
                        Qué hacer al terminar (pedir al usuario / conservar /
                        borrar los descargados y generados)
```

Rutas de salida (en orden de prioridad): `--frames-dir` > `--output-dir` >
`--output` (los frames van junto al `.md`) > junto al video (local) o
`informes/` dentro del proyecto (URLs).

**Cuando el input es una URL**: el video se descarga en `descargas/` del
proyecto y las salidas por defecto van a `informes/`. Al terminar la
herramienta pregunta al usuario si quiere borrar o conservar esos archivos
(`--autoclean ask|keep|delete`; en sesiones no interactivas se conservan).

O desde Python:

```bash
cd <carpeta_del_proyecto>
./venv/bin/python tools/video_analyzer.py --video <ruta/al/video.mp4>
```

### Soporte de URLs

La herramienta acepta **URLs de video** (YouTube, Vimeo, Twitter, y cientos de sitios vía `yt-dlp`):

```bash
./analyze_video.sh "https://www.youtube.com/watch?v=xxxxx" informe.md tiny 15
```

También acepta **archivos de solo audio** (mp3, m4a, wav, ogg, opus, flac, aac…):
se transcriben igual y se omiten los frames automáticamente.

Para reuniones grabadas (Google Meet, Teams, Zoom) que estén subidas a YouTube o similar, funciona directamente. Para reuniones en vivo, hay que grabar la pantalla primero y luego analizar el archivo.

### Reuniones en vivo (apuntes en tiempo real)

Si el usuario quiere apuntes de una reunión EN VIVO, usar `tools/live_meeting.py`
(transcribe en tiempo real con Whisper local y, al terminar con Ctrl+C, genera
el informe de ideas/discusiones/conclusiones con `--notes`):

```bash
./venv/bin/python tools/live_meeting.py --list-sources
./venv/bin/python tools/live_meeting.py --source auto --model base --notes \
  -o "informes/reunion.txt"
```

- `--source auto` captura el audio del sistema (monitor) para Meet/Teams/Zoom
  que suenan por altavoz; el micrófono se elige con su nombre (`alsa_input...`).
- `--model base` y `--device cpu` son lo recomendado en portátiles con
  refrigeración justa (tiempo real sin calentar la GPU).
- La transcripción queda en un `.txt` con timestamps; el informe `_apuntes.md`
  es un archivo aparte. Se puede regenerar después con `--transcript`.

### Interfaz gráfica

Existe una GUI local (Gradio) en `tools/auditv_gui.py` para que el usuario
haga todo desde el navegador sin terminal: pestañas de video/URL, apuntes
desde `.txt` y reunión en vivo. Se lanza con
`./venv/bin/python tools/auditv_gui.py` (requiere `pip install "gradio>=6"`,
Gradio 4 no compila Pillow en Python 3.14).

## Qué hace

1. **Transcribe** el audio con Whisper (local, gratis). Usa GPU si está disponible
   y es compatible, si no usa CPU automáticamente.
2. **Captura frames** cada N segundos (imágenes PNG) en `frames_<video>/`.
3. **Analiza el contenido** con un LLM local vía Ollama (`qwen3.5:4b`) para obtener
   resumen, ideas principales ordenadas, conceptos clave y conclusiones.
   Si la transcripción es larga, se **divide en partes de ~6000 caracteres**
   (`AUDITV_ANALYZE_PART_CHARS`) y se analiza cada parte, fusionando después
   ideas/conclusiones (mismo flujo unificado que la pestaña de Apuntes). Los
   timestamps de cada parte se desplazan al segundo real del video.
4. **Escribe** un `.md` con toda la información.

## Salida

El `.md` generado incluye:
- Resumen ejecutivo
- Ideas principales (ordenadas, con timestamps)
- Conceptos clave
- Conclusiones
- Transcripción con timestamps
- Galería de frames con referencia temporal

## Parámetros útiles (opcionales)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| output    | `<video>_analysis.md` | Ruta del informe |
| whisper_model | `small` | `tiny`, `base`, `small`, `medium`, `large` |
| interval  | `10` | Segundos entre frames |

## Nota sobre la GPU y apagados por temperatura

Whisper usa CUDA si hay una GPU compatible con PyTorch >= 2.x (compute
capability sm_75+, es decir RTX 20/30/40 o superior). El script detecta
automáticamente GPUs no compatibles (p. ej. GTX 10xx, sm_61) y trabaja en
CPU. Ollama sí puede usar GPUs antiguas para el análisis LLM. En CPU usa
modelos `tiny`/`base` para videos largos.

En portátiles con refrigeración justa (p. ej. GTX 1060 Mobile) el análisis
LLM de Ollama puede sobrecalentar la GPU y **apagar el equipo a mitad de la
ejecución**. La herramienta incluye protecciones para que no ocurra:

- Monitorea la temperatura de la GPU (nvidia-smi): avisa desde 80°C. Desde
  92°C si se alcanza durante una consulta LLM, **esa misma consulta se
  reintenta en CPU** (sin matar el proceso): así el bache en curso y los
  siguientes se resuelven igualmente (en CPU mientras la tarjeta no baje),
  de modo que NINGÚN bache se queda sin resumen/conclusiones. Solo se usa la
  GPU cuando la temperatura lo permite.
- Escribe la transcripción y un **informe parcial** (`_transcripcion.txt` +
  `.md` con transcripción y frames) ANTES de empezar el análisis con Ollama,
  así nunca se pierde el progreso si el equipo se apaga.
- **Por defecto `OLLAMA_NUM_GPU=0`** (el LLM de Ollama se ejecuta solo en CPU:
  más lento, pero NO calienta la GPU; evita el apagado en portátiles con
  refrigeración justa). Para que Ollama use la GPU si el equipo lo aguanta,
  pon `OLLAMA_NUM_GPU=1`.
- Por defecto el contexto de Ollama es 4096 tokens (antes 8192), para que el
  LLM no se desborde de la VRAM de GPUs pequeñas.

Si el PC se apagó o se nota caliente, el agente debería reejecutar así:

```bash
AUDITV_GPU_TEMP_WARN=75 OLLAMA_NUM_GPU=0 OLLAMA_NUM_CTX=2048 \
  ./analyze_video.sh <video> informe.md tiny 15 --autoclean keep --device cpu
```

Variables útiles (todas opcionales):
- `AUDITV_DEVICE=auto|cpu|cuda` — igual que `--device`.
- `OLLAMA_NUM_GPU` — `0` (valor POR DEFECTO) = LLM solo en CPU (seguro);
  `1` o `-1` = que Ollama use la GPU (más rápido, solo si el equipo aguanta).
- `OLLAMA_NUM_CTX` — recortar el contexto (2048/4096) reduce VRAM y calor.
- `OLLAMA_NUM_THREADS` — limitar hilos del LLM.
- `AUDITV_GPU_TEMP_WARN` / `AUDITV_GPU_TEMP_ABORT` — umbrales de aviso/aborto.
- El informe parcial se conserva aunque falle el paso de Ollama.
