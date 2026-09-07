# AuditV — Video → Markdown Analyzer

Herramienta que convierte un video en un informe Markdown estructurado:

1. **Transcribe** el audio con **Whisper (local, gratis)**, usando GPU si está disponible (CUDA) o CPU automáticamente.
2. **Extrae frames** representativos cada N segundos (imágenes capturadas).
3. **Analiza el contenido** con un **LLM local vía Ollama** para obtener resumen, ideas principales ordenadas, conceptos clave y conclusiones.

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
| `--output-dir` | Directorio base para el `.md` y los frames | junto al video (URLs: cwd) |
| `--frames-dir` | Directorio explícito para los frames | `<output-dir>/frames_<video>` |
| `--model / -m` | Tamaño del modelo Whisper (`base`, `small`, `medium`, `large`) | `small` |
| `--interval / -i` | Segundos entre frames capturados | `10` |
| `--llm` | Modelo de Ollama para el análisis | `qwen3.5:4b` |

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

## Resolución de rutas de salida

1. Si se pasa `--frames-dir`, los frames van ahí.
2. Si se pasa `--output-dir`, el `.md` y los frames van a ese directorio.
3. Si solo se pasa `--output`, el `.md` va ahí y los frames junto a él.
4. Si no se pasa nada: junto al video (archivos locales) o en el directorio actual (URLs).

## Variables de entorno (opcionales)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3.5:4b` | Modelo de Ollama |
| `WHISPER_MODEL` | `small` | Modelo Whisper por defecto |
| `OLLAMA_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `OLLAMA_NUM_CTX` | `8192` | Contexto (tokens) para el análisis con Ollama |
