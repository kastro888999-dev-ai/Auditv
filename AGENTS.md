# AuditV — Herramienta de análisis de video para el agente

Esta carpeta contiene una herramienta que convierte un video en un informe
Markdown estructurado (transcripción + ideas principales + frames).

> Este es el único fichero de instrucciones del agente y se llama `AGENTS.md`
> a propósito: es el nombre que leen Codex, Cursor, Cline, Windsurf y casi
> todos los demás. No hay `CLAUDE.md`: quien use Claude Code tiene que apuntar
> aquí (`/memory` o la regla del proyecto). Si añades instrucciones para el
> agente, van en este archivo.

## Funciona en Linux, macOS y Windows

Todo el proyecto es portable: no hay nada atado a un sistema. Cada SO tiene su
instalador y su lanzador, y el código Python es el mismo.

| | Linux | macOS | Windows |
|---|---|---|---|
| Instalar | `./instalar.sh` | `./instalar.sh` en una Terminal (en el Finder hay que abrirlo con «Abrir con → Terminal») | doble clic en `instalar.bat` |
| Lanzar la GUI | `./start_gui.sh` o doble clic en `auditv.desktop` | doble clic en `iniciar_gui.command` | doble clic en `iniciar_gui.bat` |
| Intérprete del venv | `./venv/bin/python` | `./venv/bin/python` | `venv\Scripts\python.exe` |
| Análisis por CLI | `./analyze_video.sh <video>` | `./analyze_video.sh <video>` | `analyze_video.sh` desde Git Bash/WSL, o `venv\Scripts\python tools\video_analyzer.py --video <video>` |

Reglas para el agente (son la causa número uno de fallos en otros sistemas):

- **No escribas código dependiente del SO** en los `.py`. Usa `pathlib.Path`,
  `sys.executable` y `subprocess` con lista de argumentos (nada de cadenas de
  shell con rutas), `os.name == "nt"` para lo que sea exclusivo de Windows y
  `shutil.which()` para localizar `ffmpeg`, `ffprobe`, `ollama`, `nvidia-smi` o
  `yt-dlp`. Nada de rutas como `/usr/bin/...` ni de `pkill`, `killall`,
  `xdg-open`, `konsole` o `apt` dentro del código.
- **Al ejecutar tú los comandos, mira primero el SO** (`uname -s`) y usa el
  intérprete que corresponda: `./venv/bin/python` en Linux/macOS y
  `venv\Scripts\python.exe` en Windows. En Windows, envuelve las llamadas en
  `set PYTHONUTF8=1` / `$env:PYTHONUTF8=1`, o los acentos y los emojis del log
  salen rotos (los lanzadores ya lo hacen por ti).
- **Para parar la GUI usa siempre el puerto**, que funciona igual en los tres:
  - Linux/macOS: `PID=$(ss -lptnH 'sport = :7860' | grep -oP 'pid=\K[0-9]+' | head -1); kill "$PID"`
  - Windows: `for /f "tokens=5" %p in ('netstat -ano ^| findstr :7860') do taskkill /PID %p /F`
- **Nunca uses `pkill -f` para matarla**: el patrón también casa con el propio
  shell que la lanza y te mata a ti en mitad del comando.
- `ffmpeg`/`ffprobe` (frames y detección de formato) y `ollama` (análisis con IA)
  tienen que estar en el PATH en cualquier SO; los scripts avisan si faltan.

## Cómo usarla (para el agente)

**Antes de ejecutar**: cuando el usuario pida analizar un video y NO haya
especificado dónde guardar el resultado, el agente DEBE preguntar (en una sola
consulta): dónde guardar el informe y los frames (`--output-dir` o `--output`
+ `--frames-dir`). Solo usa los defaults (junto al video / cwd) si el usuario
confirma explícitamente que no le importa. Nunca dejes que las salidas caigan
en directorios temporales.

**No subas ficheros temporales al repositorio**: capturas, `.html` de pruebas,
scripts de andamiaje, `.md` de prueba o vídeos de ejemplo se quedan fuera (lo
que ya esté en `.gitignore` no se sube, pero no te fíes: mira
`git status --porcelain` antes de confirmar nada). Para medir la GUI, trabaja
en `/tmp` y borra el andamiaje al terminar.

El agente puede analizar un video con un único comando:

```bash
cd <carpeta_del_proyecto>  # raíz de AuditV (donde está analyze_video.sh)
./analyze_video.sh <ruta/al/video.mp4> [output.md] [whisper_model] [interval] [flags]

  El modelo y el intervalo valen "auto" y 10: lo normal es no pasarlos y
  dejar que la herramienta los detecte (ver "Hardware y modelos: detección
  automática" más abajo).

Flags útiles (desde el 5º argumento):
  --no-llm              Análisis Ollama omitido (transcripción + frames)
  --no-frames           No extraer frames (solo transcripción + análisis)
  --output-dir DIR      Base para el .md y los frames
  --frames-dir DIR      Directorio explícito de frames
  --device auto|cpu|cuda
                        Device de Whisper: auto (default) usa la GPU solo si
                        este PyTorch la soporta; cpu la evita; cuda la fuerza
  --ollama-gpu auto|gpu|cpu
                        Dónde corre el LLM: auto (default; GPU si la tarjeta
                        es potente y no está caliente), gpu o cpu
  --ollama-gpu-index N  Índice de la GPU para el LLM (auto = la que elija Ollama)
  --gpu-temp-warn|abort|resume N
                        Umbrales de temperatura de la GPU (°C)
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
./analyze_video.sh "https://www.youtube.com/watch?v=xxxxx" informe.md auto 15
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
./venv/bin/python tools/live_meeting.py --source auto --notes \
  -o "informes/reunion.txt"
```

- `--source auto` captura el audio del sistema (monitor) para Meet/Teams/Zoom
  que suenan por altavoz; el micrófono se elige con su nombre (`alsa_input...`).
- `--model auto` y `--device auto` (los defaults) eligen lo adecuado al equipo;
  si la GPU se calienta durante la reunión, la tarjeta descansa y la
  transcripción sigue en CPU sin cortar la captura. Para fuerza máxima en
  portátiles: `--model base --device cpu`.
- La transcripción queda en un `.txt` con timestamps; el informe `_apuntes.md`
  es un archivo aparte. Se puede regenerar después con `--transcript`.

### Interfaz gráfica

Existe una GUI local (Gradio) en `tools/auditv_gui.py` para que el usuario
haga todo desde el navegador sin terminal: pestañas de video/URL, apuntes
desde `.txt` y reunión en vivo. Se lanza con
`./venv/bin/python tools/auditv_gui.py` (requiere `pip install "gradio>=6"`,
Gradio 4 no compila Pillow en Python 3.14). Encima de las pestañas hay un
**panel de equipo** refrescado cada 2 s: tarjeta, VRAM, temperatura y estado
(«en uso» / «en descanso» / «libre» / «no detectada»), más una barra de CPU/RAM/
VRAM del sistema y el consumo (RAM, % de CPU, hilos, VRAM) de Whisper y de la
IA local por separado. Ese consumo sale de `psutil` y de `/api/ps` de Ollama;
`psutil` es opcional y sin él el panel solo pierde esas cifras.

#### Cómo está montada la interfaz (para seguir ampliándola)

**Secciones en panel.** Cada grupo de campos va dentro de un
`gr.Column(variant="panel", elem_id="av_<seccion>")`. El `variant="panel"` ya
trae borde, fondo y radio del tema (se ven bien en claro y en oscuro), así que
no hay que escribir CSS para el marco. El primero, `av_origen`, ocupa todo el
ancho de la pestaña:

- `gr.Column` y **no** `gr.Row` para un panel a ancho completo: dentro de una
  fila, Gradio reparte el ancho entre los hijos y el panel queda estrecho.
- Con `variant="panel"` el contenedor sale con `flex-grow: 1` y
  `min-width: min(320px, 100%)`; `_LAYOUT_CSS` solo le ajusta el relleno
  (`padding`, `margin-bottom`, `gap`) y pone `min-width: 0` a los hijos para que
  los campos puedan encogerse en pantallas estrechas.
- Al añadir secciones, replica el patrón: `panel` + `elem_id` + un bloque de CSS
  propio. Los `elem_id` ya usados: `av_origen`, `av_log_video`, `av_log_notes`,
  `av_log_live`.

**Una sola fuente por campo, y el formato lo decide el código.** En la pestaña
"Video / URL" hay un único campo para video **y** audio local
(`Video o audio local (cualquier formato)`) y otro para la URL. `on_run_video`
elige en este orden: **URL → archivo subido → ruta local**, y lo pasa todo a
`--video`. No mires la extensión para decidir nada: `video_to_md.describe_media(ruta)`
lee el formato real con `ffprobe` (contenedor, pistas, resolución, duración,
tamaño) y `video_to_md.fmt_media(info)` lo formatea para la pantalla. Comprobado
con extensiones mentirosas: un `.mp4` que solo tiene audio se trata como audio y
no intenta sacar frames, y un `.m4a` o `.txt` que sí tienen pista de vídeo se
tratan como video.

**Aviso del archivo bajo el campo.** `_describe_path(ruta)` devuelve una línea
Markdown con lo que hay en el archivo y se dispara con `path_in.change` y
`file_in.change`, así que el usuario ve el formato antes de arrancar.

**Primera línea del log.** `_run_cli_streaming(..., preamble=...)` siembra la
línea `[FUENTE] …` en la lista de líneas del log; si no, el primer `yield` la
perdería y el usuario no sabría qué se eligió.

**Panel de equipo.** «Whisper» e «IA local» son *chips* en la misma fila flex que
CPU/RAM/VRAM (separados por una línea vertical), con «Ahora» como línea propia.
Sin ejecución no se muestra ningún modelo: pone `— se elegirá al empezar`.

**CSS: nunca uses las clases `svelte-xxxxx`.** Cambian en cada compilación de
Gradio. Usa solo clases sin hash (`upload-container`, `icon-wrap`, `wrap`,
`column`, `panel`) y `elem_id`. El CSS propio va en `_LAYOUT_CSS` (por sección) y
`_FOOTER_CSS`, concatenados en `head=`.

**No toques el interior del área de subida.** Se deja tal como lo pinta Gradio
(240 px de alto, `--size-60`, y los textos en tres líneas). Se probó a
compactarla y a poner «Coloque el archivo aquí - o - Haga clic para cargar» en
una sola línea y no funciona con CSS: dentro de `.upload-container` el texto no
es un párrafo, son items flex distintos (en un contenedor `display:flex` cada
trozo de texto y el `<span class="or">` se convierten en item), así que
`white-space: nowrap` no tiene nada que arreglar. Juntarlos exigiría rehacer la
maqueta del widget; no compensa.

**Cuidado con las generadoras de Gradio.** Una función de evento que hace
`yield from` es una generadora: un `return` con valor dentro de ella termina la
generación **sin escribir nada**, y el usuario no ve ni el error ni el motivo.
Hay que `yield (mensaje, log)` y luego `return`. (Ya pasó en `on_run_video`.)

## Qué hace

1. **Transcribe** el audio con Whisper (local, gratis). Usa GPU si está disponible
   y es compatible, si no usa CPU automáticamente.
2. **Captura frames** cada N segundos (imágenes PNG) en `frames_<video>/`.
3. **Analiza el contenido** con un LLM local vía Ollama (por defecto, el mejor
   de los que el usuario tenga instalado) para obtener resumen, ideas
   principales ordenadas, conceptos clave y conclusiones.
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
- Cabecera con el modelo de Whisper, el de Ollama, el device de cada uno y una
  línea `Motor:` con la descripción del equipo y por qué se eligió así

## Hardware y modelos: detección automática (nada atado a esta máquina)

Todo vive en `hardware.py` (raíz del proyecto), que **pregunta al sistema** qué
hay y lo convierte en decisiones. La GUI y los wrappers usan ese mismo módulo,
así que no hay nada que rehardcodear.

| Qué decide | Cómo |
|---|---|
| `--model auto` (Whisper) | En GPU, el mayor que quepa en ~75 % de la VRAM; en CPU, según la duración del audio: `small` ≤15 min, `base` ≤45 min, `tiny` para horas. Avisa de los que ya están en caché |
| `--llm auto` (Ollama) | El mayor modelo instalado que **cabe cómodo**: ~65 % de la VRAM si va a GPU (ideal ~8B) o ~4,5 GB de RAM si va a CPU (ideal ~4B). A igualdad, familias que siguen bien el JSON (`qwen`, `gemma3`, `llama3.3`…) |
| `--device auto` | GPU solo si este PyTorch la soporta de verdad (una GTX 1060 `sm_61` no lo es con los builds actuales) |
| `--ollama-gpu auto` | Clasifica la GPU en `débil`/`media`/`potente`/`muy_potente` (VRAM + compute capability). A partir de `media` el LLM va a la GPU; con GPU débil o sin GPU, a CPU para no recalentar el equipo |
| `OLLAMA_NUM_CTX auto` | 4096 en CPU; 8192 si hay ≥10 GB de VRAM |

Los defaults (`--model`, `--llm`, `--device`, `--ollama-gpu`, `OLLAMA_NUM_GPU`,
`OLLAMA_NUM_CTX`) son **`auto`**, así que el agente no necesita saber qué GPU ni
qué modelos tiene el usuario. Al arrancar, el log dice qué ha encontrado:

```
[DEVICE] GPU 0: NVIDIA GeForce GTX 1060 (6 GB, sm_6.1, débil)
[DEVICE] PyTorch NO puede usar la GPU: NVIDIA GeForce GTX 1060 (sm_61) — la GPU es sm_61 y este torch solo soporta [...]
[LLM] IA local: GPU débil: el LLM irá en CPU para no recalentar el equipo.
[LLM] Modelos en Ollama: gemma3:4b, qwen3.5:4b, qwen3:8b -> usando 'qwen3.5:4b' (3.4 GB, en CPU).
```

## Parámetros útiles (opcionales)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| output    | `<video>_analysis.md` | Ruta del informe |
| whisper_model | `auto` | `auto` o `tiny`/`base`/`small`/`medium`/`large` |
| interval  | `10` | Segundos entre frames |

## Nota sobre la GPU y apagados por temperatura

Whisper usa CUDA si hay una GPU compatible con PyTorch >= 2.x (compute
capability sm_75+, es decir RTX 20/30/40 o superior). El script detecta
automáticamente GPUs no compatibles (p. ej. GTX 10xx, sm_61) y trabaja en
CPU. Ollama sí puede usar GPUs antiguas para el análisis LLM. En CPU `auto`
elige `tiny`/`base` para audios largos.

En portátiles con refrigeración justa (p. ej. GTX 1060 Mobile) el análisis
LLM de Ollama puede sobrecalentar la GPU y **apagar el equipo a mitad de la
ejecución**. La herramienta incluye protecciones para que no ocurra:

- Mide la temperatura de la GPU (nvidia-smi) **cada 5 segundos**
  (`AUDITV_GPU_TEMP_INTERVAL`) y avisa desde 80 °C.
- Desde 92 °C la GPU **entra en descanso**: se suelta el modelo de Ollama de
  la VRAM (`keep_alive=0`) y el lote en curso **se termina en CPU sin perder su
  resultado**. Los lotes siguientes también van en CPU, y cuando la tarjeta
  baja de 75 °C (`AUDITV_GPU_TEMP_RESUME`, histéresis) el trabajo **vuelve a la
  GPU** solo. NINGÚN lote se queda sin resumen/conclusiones.
- La transcripción en GPU va **por lotes de 5 minutos**
  (`AUDITV_TRANSCRIBE_CHUNK_SEC`): entre lote y lote se comprueba la
  temperatura, y cada lote se guarda en un checkpoint al terminar. Si la tarjeta
  se calienta, ese lote y los siguientes se terminan en CPU sin perder nada.
- Escribe la transcripción y un **informe parcial** (`_transcripcion.txt` +
  `.md` con transcripción y frames) ANTES de empezar el análisis con Ollama,
  así nunca se pierde el progreso si el equipo se apaga.
- Con `auto` (lo normal) la GPU solo se usa si la tarjeta es potente **y**
  está fresca; con una GPU débil el LLM va directo a CPU sin calentarla.
- El estado (qué device usa cada cosa, temperatura, si la GPU está en descanso
  y por qué) se escribe en un JSON (`AUDITV_STATUS_FILE`) que la GUI lee cada
  2 s para pintar su panel de equipo. El consumo de CPU/RAM se mide aparte con
  `hardware.resource_snapshot()`, que cachea 1,5 s porque la GUI la interroga
  cada 2 s.

Si el PC se apagó o se nota caliente, el agente debería reejecutar así:

```bash
AUDITV_GPU_TEMP_WARN=75 ./analyze_video.sh <video> informe.md auto 15 \
  --autoclean keep --device cpu --ollama-gpu cpu
```

Variables útiles (todas opcionales):
- `AUDITV_DEVICE=auto|cpu|cuda` — igual que `--device`.
- `OLLAMA_NUM_GPU` — `auto` (POR DEFECTO, decide por la potencia de la GPU);
  `-1`/`1` = forzar GPU; `0` = solo CPU.
- `OLLAMA_NUM_CTX` — recortar el contexto (2048/4096) reduce VRAM y calor.
- `OLLAMA_NUM_THREADS` — limitar hilos del LLM.
- `AUDITV_GPU_TEMP_WARN` / `AUDITV_GPU_TEMP_ABORT` / `AUDITV_GPU_TEMP_RESUME` —
  umbrales de aviso, descanso y vuelta al trabajo.
- `AUDITV_GPU_TEMP_INTERVAL` — cada cuántos segundos se mide (5 por defecto).
- `AUDITV_TRANSCRIBE_CHUNK_SEC` — tamaño de los lotes de audio (300 por defecto).
- El informe parcial se conserva aunque falle el paso de Ollama.
