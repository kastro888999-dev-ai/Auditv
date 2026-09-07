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
./analyze_video.sh <ruta/al/video.mp4> [output.md] [whisper_model] [interval] [--no-llm] [--output-dir DIR] [--frames-dir DIR]
```

Rutas de salida (en orden de prioridad): `--frames-dir` > `--output-dir` >
`--output` (los frames van junto al `.md`) > junto al video (local) o cwd (URLs).

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

Para reuniones grabadas (Google Meet, Teams, Zoom) que estén subidas a YouTube o similar, funciona directamente. Para reuniones en vivo, hay que grabar la pantalla primero y luego analizar el archivo.

## Qué hace

1. **Transcribe** el audio con Whisper (local, gratis). Usa GPU si está disponible
   y es compatible, si no usa CPU automáticamente.
2. **Captura frames** cada N segundos (imágenes PNG) en `frames_<video>/`.
3. **Analiza el contenido** con un LLM local vía Ollama (`qwen3.5:4b`) para obtener
   resumen, ideas principales ordenadas, conceptos clave y conclusiones.
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

## Nota sobre la GPU

Whisper usa CUDA si hay una GPU compatible con PyTorch >= 2.x (compute
capability sm_75+, es decir RTX 20/30/40 o superior). El script detecta
automáticamente GPUs no compatibles (p. ej. GTX 10xx, sm_61) y trabaja en
CPU. Ollama sí puede usar GPUs antiguas para el análisis LLM. En CPU usa
modelos `tiny`/`base` para videos largos.
