#!/usr/bin/env python3
"""Video to Markdown Analyzer.

Transcribe a video using Whisper (auto GPU/CPU), extracts key frames with
timestamps, and uses a local LLM via Ollama to produce an ordered summary of
the main ideas. Outputs a structured Markdown file.

Usage:
    python video_to_md.py --video <video_file> [--output <out.md>] [--model small] [--interval 10]
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEFAULT_FRAME_INTERVAL = 10.0  # seconds between captured frames
# Extensiones de archivos solo-audio (se transcriben igual pero sin frames).
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac",
              ".wma", ".aif", ".aiff", ".m4b", ".m4p", ".ape", ".amr"}
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# 4096 evita que el LLM local se desborde de la VRAM de GPUs pequeñas y evita
# el apagado por sobrecalentamiento en portátiles.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
# Tamaño de parte para el análisis por trozos de transcripciones largas
# (unifica el flujo de video y de apuntes; ~6000 chars = JSON completo en ctx 4096).
ANALYSIS_PART_CHARS = int(os.environ.get("AUDITV_ANALYZE_PART_CHARS", "6000"))
# 0 = Ollama solo CPU (más lento pero NO calienta la GPU; evita el apagado
# por temperatura en portátiles con refrigeración justa). Pon 1/-1 para que
# Ollama use la GPU si tu equipo lo aguanta.
OLLAMA_NUM_GPU = os.environ.get("OLLAMA_NUM_GPU", "0")
OLLAMA_NUM_THREADS = os.environ.get("OLLAMA_NUM_THREADS")
# auto|cpu|cuda: forzar Whisper a CPU evita picos de carga en la GPU.
DEVICE_HINT = os.environ.get("AUDITV_DEVICE", "auto")
# Guardas de temperatura de la GPU (nvidia-smi). Warn por encima de WARN,
# aborto preventivo por encima de ABORT para proteger el hardware.
GPU_TEMP_WARN = int(os.environ.get("AUDITV_GPU_TEMP_WARN", "80"))
GPU_TEMP_ABORT = int(os.environ.get("AUDITV_GPU_TEMP_ABORT", "92"))
GPU_TEMP_CHECK_INTERVAL = 10.0  # segundos entre muestras del monitor

# Default folders inside the project for URL downloads / automated outputs.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_DIR = os.environ.get(
    "AUDITV_DOWNLOAD_DIR", str(PROJECT_DIR / "descargas")
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "AUDITV_OUTPUT_DIR", str(PROJECT_DIR / "informes")
)

# Lazy-import whisper so the module can be imported even when it is missing.
whisper = None


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log(msg: str, level: str = "INFO") -> None:
    print(f"[{level}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Device detection (GPU / CPU)
# ---------------------------------------------------------------------------
def detect_device(hint: str = DEVICE_HINT) -> str:
    """Return 'cuda' if torch can actually run on the GPU, else 'cpu'.

    Hint: 'cpu' forces CPU (safe on laptops with weak cooling), 'cuda' forces
    CUDA, 'auto' keeps the current auto-detection.
    """
    if hint == "cpu":
        log("Device forzado a CPU (AUDITV_DEVICE=cpu / --device cpu).", "DEVICE")
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            arch = torch.cuda.get_arch_list()
            sm = f"sm_{cap[0]}{cap[1]}" if cap else "?"

            # 1) Check supported architecture list (sm_XX or compute_XX).
            supported = any(sm in a for a in arch) or any(
                a.startswith("compute_") for a in arch
            )
            if not supported:
                log(
                    f"GPU {name} ({sm}) not supported by torch {torch.__version__} "
                    f"(supports {arch or '[]'}). Using CPU.",
                    "DEVICE",
                )
                return "cpu"

            # 2) Real functional test: run a tiny CUDA kernel.
            x = torch.ones(1, device="cuda")
            _ = (x + 1).item()
            log(f"GPU detected: {name} ({sm})", "DEVICE")
            if hint == "cuda":
                log("Device forzado a CUDA (--device cuda).", "DEVICE")
            return "cuda"
        else:
            log("No GPU detected, using CPU.", "DEVICE")
            return "cpu"
    except Exception as exc:
        log(f"CUDA unusable, falling back to CPU: {exc}", "DEVICE")
        return "cpu"


# ---------------------------------------------------------------------------
# GPU health monitoring (avoids thermal shutdowns on laptops / weak GPUs)
# ---------------------------------------------------------------------------
def gpu_temperature() -> float:
    """Return current GPU temperature in °C from nvidia-smi, or None if NA."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()
        return float(line[0]) if line else None
    except Exception:
        return None


def check_gpu_health(stage: str) -> None:
    """Warn or abort if the GPU is running too hot (protects the machine)."""
    temp = gpu_temperature()
    if temp is None:
        return
    if temp >= GPU_TEMP_ABORT:
        raise RuntimeError(
            f"GPU a {temp:.0f}°C antes de '{stage}' (límite {GPU_TEMP_ABORT}°C). "
            "Aborto preventivo para proteger el hardware. Deja enfriar el equipo "
            "y vuelve a ejecutar con: --autoclean keep --device cpu "
            "OLLAMA_NUM_GPU=0 OLLAMA_NUM_CTX=2048"
        )
    if temp >= GPU_TEMP_WARN:
        log(
            f"GPU ya está a {temp:.0f}°C antes de '{stage}': riesgo alto de "
            "apagado por temperatura. Reintenta con --device cpu y/o "
            "OLLAMA_NUM_GPU=0 para no exigir a la tarjeta.",
            "WARN",
        )


class _GpuMonitor(threading.Thread):
    """Background thread that logs warnings if the GPU gets too hot.

    On critical temperature it does NOT kill the process: it sets an event and
    cancels any in-flight Ollama request so the pipeline can continue and the
    report is still written (with the analysis that was possible).
    """

    def __init__(self, stage: str):
        super().__init__(daemon=True)
        self._stage = stage
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        global _open_resp
        while not self._stop.wait(GPU_TEMP_CHECK_INTERVAL):
            temp = gpu_temperature()
            if temp is None:
                continue
            if temp >= GPU_TEMP_ABORT:
                log(
                    f"GPU a {temp:.0f}°C durante '{self._stage}': temperatura "
                    f"crítica ({GPU_TEMP_ABORT}°C). Cancelando el análisis LLM "
                    "para proteger el hardware (el informe se generará igualmente).",
                    "ERROR",
                )
                _thermal["event"].set()
                _cancel_open_request()
            elif temp >= GPU_TEMP_WARN:
                log(
                    f"GPU a {temp:.0f}°C durante '{self._stage}' "
                    f"(límite de aviso {GPU_TEMP_WARN}°C). Si sigue subiendo, "
                    "usa OLLAMA_NUM_GPU=0 / --device cpu.",
                    "WARN",
                )


class ThermalAbort(RuntimeError):
    """Raised when the GPU reaches the critical temperature."""


_thermal = {"event": threading.Event()}
_open_resp = {"resp": None}


def _cancel_open_request() -> None:
    """Close the in-flight Ollama response so a blocking read() raises now."""
    resp = _open_resp.get("resp")
    if resp is not None:
        try:
            resp.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------
def extract_audio(video_path: str, output_wav: str) -> str:
    """Extract mono 16kHz WAV audio from video via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"Audio extracted -> {output_wav}")
    return output_wav


def has_video_stream(path: str) -> bool:
    """True if the media file contains a video track (ffprobe).

    Audio-only files (mp3, wav, m4a...) return False so frames are skipped.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return True
    except Exception:
        pass
    # Fallback heurístico por extensión.
    return Path(path).suffix.lower() not in AUDIO_EXTS


# ---------------------------------------------------------------------------
# Transcription with Whisper
# ---------------------------------------------------------------------------
def transcribe_audio(audio_path: str, model_size: str, device: str) -> dict:
    """Transcribe audio with Whisper, return dict with 'text' and 'segments'.

    segments: list of {"start": seconds, "end": seconds, "text": str}
    """
    global whisper
    if whisper is None:
        try:
            import whisper
        except Exception as exc:
            raise RuntimeError(
                "openai-whisper is not installed. Run: "
                "./venv/bin/pip install openai-whisper"
            ) from exc

    log(f"Loading Whisper model '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)
    log("Transcribing...")
    monitor = None
    if device == "cuda":
        check_gpu_health("transcripción")
        monitor = _GpuMonitor("transcripción")
        monitor.start()
    try:
        result = model.transcribe(audio_path, fp16=(device == "cuda"))
    finally:
        if monitor is not None:
            monitor.stop()
            monitor.join(timeout=GPU_TEMP_CHECK_INTERVAL + 1)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": str(seg["text"]).strip(),
        })

    log(f"Transcription complete. {len(segments)} segments.")
    return {"text": result.get("text", "").strip(), "segments": segments}


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(video_path: str, output_dir: str, interval_sec: float) -> list:
    """Extract frames every interval_sec. Returns list of {path, timestamp}."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1/{interval_sec}",
        "-qscale:v", "2",
        os.path.join(output_dir, "frame_%06d.png"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = sorted(Path(output_dir).glob("frame_*.png"))
    result = []
    for i, fpath in enumerate(frames):
        timestamp = i * interval_sec  # frame 0 at t=0, frame 1 at t=interval, ...
        result.append({"path": str(fpath), "timestamp": timestamp})
    log(f"Captured {len(result)} frames.")
    return result


# ---------------------------------------------------------------------------
# Ollama integration (summary + ideas extraction)
# ---------------------------------------------------------------------------
def _ollama_generate(prompt: str, model: str, system: str = "") -> str:
    """Send a prompt to a local Ollama server and return the raw text output."""
    # Si la GPU ya está crítica, ni empezar: se aborta el análisis suavemente
    # y el informe se escribe igualmente.
    temp = gpu_temperature()
    if temp is not None and temp >= GPU_TEMP_ABORT and _thermal["event"].is_set():
        raise ThermalAbort(temp)
    if temp is not None and temp >= GPU_TEMP_ABORT:
        log(
            f"GPU ya está a {temp:.0f}°C (límite {GPU_TEMP_ABORT}°C): se "
            "cancela el análisis LLM para proteger el hardware.",
            "ERROR",
        )
        raise ThermalAbort(temp)
    if _thermal["event"].is_set():
        raise ThermalAbort(GPU_TEMP_ABORT)

    options = {"num_ctx": OLLAMA_NUM_CTX, "num_gpu": int(OLLAMA_NUM_GPU)}
    if OLLAMA_NUM_THREADS is not None:
        options["num_thread"] = int(OLLAMA_NUM_THREADS)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if system:
        payload["system"] = system

    check_gpu_health("análisis LLM (Ollama)")
    monitor = _GpuMonitor("análisis LLM (Ollama)")
    monitor.start()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        _open_resp["resp"] = None
        stop_beat = threading.Event()

        def _heartbeat() -> None:
            # Con Ollama en CPU el análisis tarda; avisa cada 20 s de que sigue.
            started = time.time()
            while not stop_beat.wait(20):
                log(
                    f"El análisis LLM ({model}) sigue trabajando... "
                    f"{int(time.time() - started)} s",
                    "INFO",
                )

        heartbeat = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat.start()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                _open_resp["resp"] = resp
                data = json.loads(resp.read().decode("utf-8"))
                _open_resp["resp"] = None
                if _thermal["event"].is_set():
                    raise ThermalAbort(GPU_TEMP_ABORT)
                return data.get("response", "").strip()
        except ThermalAbort:
            log("Análisis LLM cancelado por temperatura crítica.", "ERROR")
            raise
        except Exception as exc:
            log(f"Ollama request failed: {exc}", "ERROR")
            raise
        finally:
            stop_beat.set()
    finally:
        _open_resp["resp"] = None
        monitor.stop()
        monitor.join(timeout=GPU_TEMP_CHECK_INTERVAL + 1)


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not text:
        return {}
    # 1) Direct parse after stripping code fences.
    t = re.sub(r"^```(?:json)?\s*|```\s*$", "", text.strip(), flags=re.S)
    for cand in (t.strip(), text.strip()):
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else {"raw": cand}
        except Exception:
            continue

    # 2) Find balanced {...} regions; prefer the outermost/longest, handle
    #    trailing commas. (Modelos pequeños a veces cortan o dejan comas sobrantes.)
    candidates = []
    for start in re.finditer(r"\{", t):
        depth = 0
        for i in range(start.start(), len(t)):
            ch = t[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(t[start.start():i + 1])
                    break
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        for c in (cand, re.sub(r",(\s*)([\]}])", r"\1\2", cand)):
            try:
                obj = json.loads(c)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return {"raw": text}


def analyze_with_ollama(
    transcript: str,
    model: str,
    include_timestamps: bool = True,
    lang: str = "es",
) -> dict:
    """Use Ollama to summarize + extract ideas/discussions/conclusions.

    include_timestamps: whether the analysis should ask for per-idea
    timestamps (from a video transcription) or skip them (plain meeting text).
    """
    MAX_TRANSCRIPT_CHARS = 8000
    if not transcript or not transcript.strip():
        log("Transcript vacío: se omite el análisis LLM.", "WARN")
        return {"resumen": "*(Sin transcripción disponible)*", "ideas": [], "discusiones": [], "conceptos": [], "conclusiones": []}
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        log(f"Transcript too long ({len(transcript)} chars), truncating to {MAX_TRANSCRIPT_CHARS} chars for analysis.")
        tail_len = 2500
        head = transcript[:MAX_TRANSCRIPT_CHARS - tail_len]
        tail = transcript[-tail_len:]
        transcript = f"{head}\n[.../...]\n{tail}"

    system = (
        "You are an expert meeting/video analyst. You receive a speech "
        "transcription and must extract its content in a structured way. "
        "Always answer in Spanish using ONLY valid JSON, no extra text."
    )

    ts_field = (
        '    {"titulo": "Idea principal 1", "descripcion": "Detalle de la idea", "timestamp": 12.5},\n'
        '    {"titulo": "Idea principal 2", "descripcion": "Detalle", "timestamp": 30.0}\n'
    )
    ts_rule = (
        "- 'timestamp' en cada idea debe ser el segundo aproximado del video "
        "donde se menciona (usa los tiempos de la transcripción).\n"
    )
    if not include_timestamps:
        ts_field = (
            '    {"titulo": "Idea principal 1", "descripcion": "Detalle de la idea"},\n'
            '    {"titulo": "Idea principal 2", "descripcion": "Detalle"}\n'
        )
        ts_rule = (
            "- NO incluyas 'timestamp' en las ideas: es texto plano de una "
            "reunión, no hay tiempos del video.\n"
        )

    prompt = (
        "Analiza la siguiente transcripción y devuelve un objeto JSON con la "
        "siguiente estructura exacta:\n"
        "{\n"
        '  "titulo": "Título sugerido para la reunión/sesión",\n'
        '  "resumen": "Resumen ejecutivo de 2-3 oraciones",\n'
        "  \"ideas\": [\n"
        + ts_field
        + "  ],\n"
        '  "discusiones": ["punto debatido o tema conversado 1", "punto 2"],\n'
        '  "conceptos": ["concepto 1", "concepto 2"],\n'
        '  "conclusiones": ["conclusión 1", "conclusión 2"]\n'
        "}\n"
        "Reglas:\n"
        "- 'ideas' debe listar las ideas principales ORDENADAS por importancia, "
        "máximo 8.\n"
        + ts_rule
        + "- 'discusiones' son los temas que se debatieron, aclararon o en los "
        "que las personas interactuaron entre sí.\n"
        "- 'conceptos' son términos o conceptos clave mencionados.\n"
        "- Si la transcripción está vacía, devuelve una estructura vacía con "
        "'resumen': 'Sin transcripción disponible'.\n\n"
        f"TRANSCRIPCIÓN:\n{transcript}"
    )

    log(f"Asking Ollama model '{model}' for analysis...")
    result = _extract_json(_ollama_generate(prompt, model, system))
    if "resumen" in result or "ideas" in result or "conclusiones" in result:
        return result

    # Reintento: el modelo no devolvió JSON válido (o vino truncado).
    strict = (
        "Responde ÚNICAMENTE con un bloque de código JSON (```json ... ```) "
        "con la estructura exacta que se pide abajo y nada más; no añadas "
        "texto fuera del bloque ni cortes la respuesta antes de cerrarla.\n\n"
        + prompt
    )
    log("Análisis LLM sin JSON válido; reintentando con formato estricto...", "WARN")
    result2 = _extract_json(_ollama_generate(strict, model, system))
    for key, val in (result2 or {}).items():
        if val:
            result.setdefault(key, val)
    if not result.get("resumen"):
        result = result2 or result
    return result


def merge_analyses(analyses: list) -> dict:
    """Fusiona el análisis de varias partes en un solo dict (sin re-analizar).

    Junta las listas de ideas/discusiones/conceptos/conclusiones eliminando
    duplicados y construye el resumen etiquetando cada parte.
    """
    if not analyses:
        return {}
    result = {"titulo": "", "resumen": "", "ideas": [], "discusiones": [], "conceptos": [], "conclusiones": []}

    resumenes = [a.get("resumen") for a in analyses if a.get("resumen")]
    if resumenes:
        if len(resumenes) == 1:
            result["resumen"] = str(resumenes[0])
        else:
            partes = "\n\n".join(
                f"**Resumen de la parte {i}:** {r}" for i, r in enumerate(resumenes, 1)
            )
            result["resumen"] = (
                "Resumen por partes de la sesión (el análisis cubre el texto completo):\n\n"
                + partes
            )

    def _dedupe(items):
        out, seen = [], set()
        for it in items:
            if isinstance(it, dict):
                key = ((it.get("titulo") or "") + " " + (it.get("descripcion") or "")).strip().lower()
                label = it
            else:
                key, label = str(it).strip().lower(), it
            if key and key not in seen:
                seen.add(key)
                out.append(label)
        return out

    result["ideas"] = _dedupe(it for a in analyses for it in (a.get("ideas") or []))[:8]
    for key in ("discusiones", "conceptos", "conclusiones"):
        result[key] = _dedupe(it for a in analyses for it in (a.get(key) or []))

    for a in analyses:
        if a.get("titulo"):
            result["titulo"] = a["titulo"]
            break
    return result


def _time_at_char(transcript_dict: dict, idx: int) -> float:
    """Segundos aproximados del video para un índice de 'text'."""
    segs = transcript_dict.get("segments") or []
    pos = 0.0
    for seg in segs:
        seg_text = seg.get("text") or ""
        n = len(seg_text) + 1  # +1 por el espacio con que se unen los segmentos
        if idx <= pos + n:
            return float(seg.get("start", 0))
        pos += n
    return float(segs[-1].get("start", 0)) if segs else 0.0


def analyze_transcript_in_parts(transcript_or_text, model: str, include_timestamps: bool = True):
    """Analiza la transcripción completa dividiéndola en partes y fusionando.

    Flujo unificado (video y apuntes): si el texto es largo se parte en
    trozos de ANALYSIS_PART_CHARS, cada trozo se analiza por separado y los
    resultados se fusionan. Para videos, los timestamps de cada parte se
    desplazan al segundo real del video.
    """
    if isinstance(transcript_or_text, dict):
        text = transcript_or_text.get("text") or ""
        time_at = (_time_at_char, transcript_or_text) if include_timestamps else None
    else:
        text = transcript_or_text or ""
        time_at = None
        include_timestamps = False

    if not text.strip():
        return {"resumen": "*(Sin transcripción disponible)*", "ideas": [], "discusiones": [], "conceptos": [], "conclusiones": []}

    parts = split_text(text, ANALYSIS_PART_CHARS)
    if not parts:
        parts = [text]
    if len(parts) == 1:
        return analyze_with_ollama(text, model, include_timestamps)

    log(f"Transcripción larga ({len(text)} chars): análisis en {len(parts)} partes de ~{ANALYSIS_PART_CHARS}.")
    analyses = []
    consumed = 0
    for i, part in enumerate(parts, 1):
        part_analysis = {"resumen": "*(Sin contenido en esta parte)*"}
        if time_at is not None and part:
            time_fn, transcript_dict = time_at
            base_sec = time_fn(transcript_dict, consumed + 1)
            consumed += len(part) + 1
            try:
                part_analysis = analyze_with_ollama(part, model, include_timestamps)
                for idea in part_analysis.get("ideas") or []:
                    if isinstance(idea, dict) and idea.get("timestamp") is not None:
                        idea["timestamp"] = round(float(idea["timestamp"]) + base_sec, 1)
            except Exception as exc:
                log(f"Análisis de la parte {i} falló: {exc}", "WARN")
        else:
            try:
                part_analysis = analyze_with_ollama(part, model, include_timestamps)
            except Exception as exc:
                log(f"Análisis de la parte {i} falló: {exc}", "WARN")
        analyses.append(part_analysis)
    return merge_analyses(analyses)



# ---------------------------------------------------------------------------
# Time formatting helpers
# ---------------------------------------------------------------------------
def fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_transcript_section(segments: list) -> str:
    """Build a markdown block with timestamped transcript segments."""
    lines = []
    if not segments:
        lines.append("*(No se detectó habla en el video.)*")
        return "\n".join(lines)
    for seg in segments:
        stamp = fmt_ts(seg["start"])
        lines.append(f"**[{stamp}]** {seg['text']}")
    return "\n".join(lines)


def build_frames_section(frames: list, frames_dir: str, output_md: str = None) -> str:
    """Build markdown image gallery for frames with timestamps.

    Frame paths are written relative to the .md location when the frames dir
    is inside (or equal to) the .md dir; absolute paths otherwise.
    """
    lines = []
    if not frames:
        lines.append("*(No se extrajeron frames: opción desactivada o el archivo es solo audio.)*")
        return "\n".join(lines)
    md_dir = str(Path(output_md).resolve().parent) if output_md else None
    for f in frames:
        stamp = fmt_ts(f["timestamp"])
        ref = f["path"]
        if md_dir:
            rel = os.path.relpath(str(Path(f["path"]).resolve()), md_dir)
            if not rel.startswith(".."):
                ref = rel
        lines.append(f"### 🖼️ T={stamp}")
        lines.append(f"![Frame T={stamp}]({ref})")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------
def build_markdown(
    video_path: str,
    transcript: dict,
    frames: list,
    analysis: dict,
    whisper_model: str,
    device: str,
    ollama_model: str,
    frames_dir: str,
    output_md: str,
) -> None:
    """Generate the final structured Markdown report."""
    video_name = Path(video_path).name
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"# 🎬 Análisis: {video_name}")
    lines.append("")
    lines.append(f"- **Generado:** {now}")
    lines.append(f"- **Modelo Whisper:** {whisper_model} ({device})")
    lines.append(f"- **Modelo LLM:** {ollama_model}")
    lines.append("")

    if analysis.get("titulo"):
        lines.append(f"## 🏷️ {analysis['titulo']}")
        lines.append("")

    lines.append("## 📌 Resumen ejecutivo")
    lines.append("")
    lines.append(analysis.get("resumen", "*(Sin resumen disponible)*"))
    lines.append("")

    lines.append("## 🧠 Ideas principales")
    lines.append("")
    ideas = analysis.get("ideas", [])
    if ideas:
        for i, idea in enumerate(ideas, 1):
            ts = idea.get("timestamp")
            ts_str = f" `[{fmt_ts(float(ts))}]`" if ts else ""
            lines.append(f"### {i}. {idea.get('titulo', 'Sin título')}{ts_str}")
            lines.append("")
            desc = idea.get("descripcion", "")
            if desc:
                lines.append(desc)
                lines.append("")
    else:
        lines.append("*(No se detectaron ideas principales.)*")
        lines.append("")

    lines.append("## 💬 Discusiones / Puntos debatidos")
    lines.append("")
    discussions = analysis.get("discusiones", [])
    if discussions:
        lines.append("- " + "\n- ".join(str(d) for d in discussions))
    else:
        lines.append("*(Sin discusiones destacadas)*")
    lines.append("")

    lines.append("## 🔑 Conceptos clave")
    lines.append("")
    concepts = analysis.get("conceptos", [])
    if concepts:
        lines.append("- " + "\n- ".join(str(c) for c in concepts))
    else:
        lines.append("*(Sin conceptos destacados)*")
    lines.append("")

    lines.append("## ✅ Conclusiones")
    lines.append("")
    conclusions = analysis.get("conclusiones", [])
    if conclusions:
        lines.append("- " + "\n- ".join(str(c) for c in conclusions))
    else:
        lines.append("*(Sin conclusiones)*")
    lines.append("")

    lines.append("## 📝 Transcripción")
    lines.append("")
    lines.append(build_transcript_section(transcript.get("segments", [])))
    lines.append("")

    lines.append("## 🖼️ Frames capturados")
    lines.append("")
    if frames:
        lines.append(f"*(Frames guardados en `{frames_dir}`)*")
        lines.append("")
    lines.append(build_frames_section(frames, frames_dir, output_md))

    with open(output_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    log(f"Markdown written -> {output_md}")


def save_transcript_checkpoint(transcript: dict, output_md: str) -> str:
    """Save the plain transcript next to the report so progress isn't lost.

    If the machine shuts down mid-run, the transcription is already on disk.
    """
    base = Path(output_md)
    txt_path = base.with_name(base.stem + "_transcripcion.txt")
    text = transcript.get("text", "") or ""
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"Checkpoint transcripción -> {txt_path}")
    return str(txt_path)


# ---------------------------------------------------------------------------
# Transcript mode: structured notes from a plain-text meeting transcript
# ---------------------------------------------------------------------------
def build_transcript_markdown(
    transcript_path: str,
    source_text: str,
    analysis: dict,
    ollama_model: str,
    output_md: str,
) -> None:
    """Generate a standalone notes report (ideas/discusiones/conclusiones)
    from a plain-text transcript, as a separate .md file."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"# 📝 Apuntes: {Path(transcript_path).name}")
    lines.append("")
    lines.append(f"- **Generado:** {now}")
    lines.append(f"- **Modelo LLM:** {ollama_model}")
    lines.append("")

    if analysis.get("titulo"):
        lines.append(f"## 🏷️ {analysis['titulo']}")
        lines.append("")

    lines.append("## 📌 Resumen")
    lines.append("")
    lines.append(analysis.get("resumen", "*(Sin resumen disponible)*"))
    lines.append("")

    lines.append("## 🧠 Ideas principales")
    lines.append("")
    ideas = analysis.get("ideas", [])
    if ideas:
        for i, idea in enumerate(ideas, 1):
            lines.append(f"### {i}. {idea.get('titulo', 'Sin título')}")
            lines.append("")
            desc = idea.get("descripcion", "")
            if desc:
                lines.append(desc)
                lines.append("")
    else:
        lines.append("*(No se detectaron ideas principales.)*")
        lines.append("")

    lines.append("## 💬 Discusiones / Puntos debatidos")
    lines.append("")
    discussions = analysis.get("discusiones", [])
    if discussions:
        lines.append("- " + "\n- ".join(str(d) for d in discussions))
    else:
        lines.append("*(Sin discusiones destacadas)*")
    lines.append("")

    lines.append("## 🔑 Conceptos clave")
    lines.append("")
    concepts = analysis.get("conceptos", [])
    if concepts:
        lines.append("- " + "\n- ".join(str(c) for c in concepts))
    else:
        lines.append("*(Sin conceptos destacados)*")
    lines.append("")

    lines.append("## ✅ Conclusiones")
    lines.append("")
    conclusions = analysis.get("conclusiones", [])
    if conclusions:
        lines.append("- " + "\n- ".join(str(c) for c in conclusions))
    else:
        lines.append("*(Sin conclusiones)*")
    lines.append("")

    lines.append("## 📄 Transcripción completa")
    lines.append("")
    lines.append("```text")
    lines.append(source_text.rstrip("\n"))
    lines.append("```")

    with open(output_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    log(f"Markdown written -> {output_md}")


def split_text(text: str, chunk_chars: int = 6000) -> list:
    """Split a long transcript into parts of ~chunk_chars characters.

    Prefers breaking at paragraph or sentence boundaries so parts read
    naturally. With a 4b LLM and ctx 4096, ~5000-6500 chars per part keeps
    the JSON output well within context (recommended for long meeting texts).
    """
    text = (text or "").strip()
    if not text:
        return []
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text]

    chunks = []
    while len(text) > chunk_chars:
        cut = text.rfind("\n\n", 0, chunk_chars)
        if cut < chunk_chars // 2:
            cut = text.rfind("\n", 0, chunk_chars)
        if cut < chunk_chars // 2:
            cut = text.rfind(". ", 0, chunk_chars)
        if cut < chunk_chars // 2:
            cut = chunk_chars
        cut += 1
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
        if not text:
            break
    if text:
        chunks.append(text)
    return chunks


def transcript_to_md(
    transcript_path: str,
    output_md: str = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    use_llm: bool = True,
    output_dir: str = None,
    split_chars: int = 0,
) -> str:
    """Generate a standalone notes report from a plain-text transcript.

    Creates a separate .md next to the transcript (or in output_dir /
    --output). Reads the text file, extracts ideas/discusiones/conclusiones
    with Ollama and writes the report. No video, audio or frames involved.

    If split_chars > 0 and the text is longer, it is split into parts (one
    .txt + one _apuntes.md per part) so the LLM analyses the whole meeting
    without truncating. Returns a string with the generated file(s).
    """
    transcript_path = str(Path(transcript_path).resolve())
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read().strip()
    if not text:
        raise ValueError(f"Transcript file is empty: {transcript_path}")

    if output_dir is not None:
        base_dir = Path(output_dir).resolve()
    elif output_md is not None:
        base_dir = Path(output_md).resolve().parent
    else:
        base_dir = Path(transcript_path).parent
    os.makedirs(base_dir, exist_ok=True)

    if output_md is None:
        output_md = str(base_dir / f"{Path(transcript_path).stem}_apuntes.md")
    else:
        output_md = str(Path(output_md).resolve())

    parts = split_text(text, split_chars) if split_chars > 0 else [text]
    if len(parts) == 1:
        return _transcript_part_to_md(
            transcript_path, text, ollama_model, use_llm, output_md
        )

    log(f"Texto dividido en {len(parts)} partes de ~{split_chars} caracteres.")
    stem = Path(transcript_path).stem
    outputs = []
    analyses = []
    for i, part in enumerate(parts, 1):
        part_txt = base_dir / f"{stem}_parte{i}.txt"
        part_txt.write_text(part, encoding="utf-8")
        part_md = base_dir / f"{stem}_parte{i}_apuntes.md"
        analysis = _transcript_part_analysis(part, ollama_model, use_llm)
        analyses.append(analysis)
        build_transcript_markdown(
            str(part_txt), part, analysis, ollama_model, str(part_md)
        )
        outputs.append(str(part_md))

    # Informe completo: junta lo ya analizado (sin re-analizar el texto) +
    # transcripción completa. Todo ensamblado a partir de cada parte.
    combined_md = base_dir / f"{stem}_completo_apuntes.md"
    build_combined_transcript_markdown(
        stem, parts, analyses, ollama_model, text, str(combined_md)
    )
    outputs.append(str(combined_md))

    # Los .txt de partes eran temporales: se borran (los .md se conservan).
    for i in range(1, len(parts) + 1):
        try:
            (base_dir / f"{stem}_parte{i}.txt").unlink()
        except OSError:
            pass
    log("Partes temporales .txt eliminadas; se conservan los .md.", "INFO")
    return "\n".join(outputs)


def _transcript_part_analysis(text: str, ollama_model: str, use_llm: bool) -> dict:
    """Analyze one transcript part with Ollama (best-effort)."""
    if use_llm:
        try:
            return analyze_transcript_in_parts(
                text, ollama_model, include_timestamps=False
            )
        except Exception as exc:
            log(f"Ollama analysis skipped: {exc}", "WARN")
            return {"resumen": "*(No se pudo analizar con Ollama)*"}
    log("Skipping Ollama analysis (--no-llm)")
    return {"resumen": "*(Análisis LLM omitido)*"}


def _transcript_part_to_md(
    transcript_path: str,
    text: str,
    ollama_model: str,
    use_llm: bool,
    output_md: str,
) -> str:
    """Analyze one transcript part and write its _apuntes.md."""
    analysis = _transcript_part_analysis(text, ollama_model, use_llm)
    build_transcript_markdown(
        transcript_path=transcript_path,
        source_text=text,
        analysis=analysis,
        ollama_model=ollama_model,
        output_md=output_md,
    )
    return output_md


def build_combined_transcript_markdown(
    stem: str,
    parts: list,
    analyses: list,
    ollama_model: str,
    full_source: str,
    output_md: str,
) -> str:
    """Write a combined report assembling the analyses of all parts.

    No new LLM call: it joins the resumen/ideas/discusiones/conceptos/
    conclusiones already extracted per part and appends the full transcript.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(parts)

    lines = []
    lines.append(f"# 📚 Reunión completa: {stem}")
    lines.append("")
    lines.append(f"- **Generado:** {now}")
    lines.append(f"- **Modelo LLM:** {ollama_model}")
    lines.append(f"- **Partes analizadas:** {n}")
    lines.append("")

    def part_list_section(title: str, key: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for i, a in enumerate(analyses, 1):
            lines.append(f"### Parte {i}")
            lines.append("")
            items = a.get(key) or []
            if items:
                lines.append("- " + "\n- ".join(str(x) for x in items))
            else:
                lines.append("*(Sin contenido en esta parte)*")
            lines.append("")

    lines.append("## 📝 Resúmenes por parte")
    lines.append("")
    for i, a in enumerate(analyses, 1):
        lines.append(f"### Parte {i}")
        lines.append("")
        lines.append(str(a.get("resumen") or "*(Sin resumen)*"))
        lines.append("")

    lines.append("## 🧠 Ideas principales por parte")
    lines.append("")
    for i, a in enumerate(analyses, 1):
        lines.append(f"### Parte {i}")
        lines.append("")
        ideas = a.get("ideas") or []
        if ideas:
            for j, idea in enumerate(ideas, 1):
                titulo = idea.get("titulo", "Sin título") if isinstance(idea, dict) else idea
                desc = idea.get("descripcion", "") if isinstance(idea, dict) else ""
                lines.append(f"{j}. **{titulo}** — {desc}")
        else:
            lines.append("*(Sin ideas en esta parte)*")
        lines.append("")

    part_list_section("💬 Discusiones / Puntos debatidos por parte", "discusiones")
    part_list_section("🔑 Conceptos clave por parte", "conceptos")
    part_list_section("✅ Conclusiones por parte", "conclusiones")

    lines.append("## 🏁 Conclusión final (sin re-analizar)")
    lines.append("")
    todos = []
    for a in analyses:
        todos += (a.get("conclusiones") or []) + (a.get("ideas") or [])
    distinct = []
    for x in todos:
        label = x.get("titulo") if isinstance(x, dict) else str(x)
        if isinstance(x, dict):
            label = f"{label}: {x.get('descripcion','')}"
        if label and label not in distinct:
            distinct.append(str(label))
    if distinct:
        lines.append("**Principales ideas y acuerdos extraídos de todas las partes:**")
        lines.append("")
        lines.append("- " + "\n- ".join(distinct))
        lines.append("")
    else:
        lines.append("*(No se pudieron ensamblar conclusiones: análisis LLM omitido o vacío.)*")
        lines.append("")

    lines.append("## 📄 Transcripción completa")
    lines.append("")
    lines.append("```text")
    lines.append(full_source.rstrip("\n"))
    lines.append("```")

    with open(output_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"Markdown completo escrito -> {output_md}")
    return output_md



# ---------------------------------------------------------------------------
# URL download via yt-dlp
# ---------------------------------------------------------------------------
def is_url(path_or_url: str) -> bool:
    """Check if the input string looks like a URL."""
    return path_or_url.startswith(("http://", "https://", "www."))


def download_video(
    url: str,
    output_dir: str,
    cookies: str = None,
    cookies_from_browser: str = None,
    live_from_start: bool = False,
) -> str:
    """Download a video from a URL using yt-dlp. Returns path to downloaded file.

    cookies: path to a Netscape-format cookie file for logged-in platforms.
    cookies_from_browser: browser to reuse cookies from (e.g. "chrome",
        "firefox", "chromium", optionally ":<profile>"; useful on platforms
        where the user is already logged in).
    live_from_start: if the URL is a live stream supported by yt-dlp, start
        downloading from the beginning of the stream.
    """
    log(f"Downloading video from URL: {url}")
    out_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", out_template,
        "--merge-output-format", "mp4",
        "--no-warnings",
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    if live_from_start:
        cmd += ["--live-from-start"]
    cmd.append(url)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Find the downloaded file (yt-dlp may rename it)
    files = sorted(Path(output_dir).glob("*.*"), key=os.path.getmtime, reverse=True)
    ok_suffixes = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".ts") | AUDIO_EXTS
    for f in files:
        if f.suffix.lower() in ok_suffixes:
            log(f"Downloaded -> {f}")
            return str(f)
    raise RuntimeError(f"Could not find downloaded video in {output_dir}")


def _ask_keep_or_delete(artifacts: list) -> bool:
    """Ask the user interactively whether to delete generated artifacts."""
    print("\nSe generaron los siguientes archivos:")
    for art in artifacts:
        print(f"  - {art}")
    try:
        answer = input("¿Quieres borrarlos? [s/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("s", "si", "y", "yes")


def _remove_if_empty(path: Path) -> None:
    """Remove a directory if it exists and is empty."""
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def maybe_cleanup(
    autoclean: str,
    download_paths: list,
    output_md: str,
    frames_dir: str,
) -> None:
    """Keep or delete artifacts after the run (downloaded video, .md, frames).

    autoclean values: "ask" (interactive prompt), "keep" or "delete".
    In non-interactive sessions "ask" falls back to keeping the files.
    """
    artifacts = list(download_paths) + [output_md]
    if frames_dir and Path(frames_dir).is_dir():
        artifacts.append(frames_dir)

    delete = autoclean == "delete"
    if autoclean == "ask":
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
        if interactive:
            delete = _ask_keep_or_delete(artifacts)
        else:
            delete = False
            log(
                "Sesión no interactiva: se conservan los archivos descargados y "
                "generados. Usa --autoclean keep/delete para controlarlo.",
                "WARN",
            )

    if delete:
        for art in artifacts:
            p = Path(art)
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
            except OSError as exc:
                log(f"No se pudo eliminar {p}: {exc}", "WARN")
        # Limpiar directorios del proyecto que hayan quedado vacíos.
        for art in artifacts:
            _remove_if_empty(Path(art).parent)
        log("Archivos descargados/generados eliminados.", "INFO")
    else:
        log(
            f"Se conservaron los archivos descargados/generados en "
            f"{Path(output_md).parent}.",
            "INFO",
        )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def video_to_md(
    video_path: str,
    output_md: str = None,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    frame_interval: float = DEFAULT_FRAME_INTERVAL,
    extract_frames: bool = True,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    use_llm: bool = True,
    output_dir: str = None,
    frames_dir: str = None,
    autoclean: str = "ask",
    device_hint: str = DEVICE_HINT,
    cookies: str = None,
    cookies_from_browser: str = None,
    live_from_start: bool = False,
) -> str:
    """Full pipeline: transcribe + analyze + capture frames -> markdown.

    video_path can be a local file path or a URL (YouTube, Google Drive, etc.).

    Output location is resolved in this order:
    - output_dir: base dir for both the .md and the frames dir (if given).
    - output_md: the .md goes where specified; frames default to its parent dir.
    - frames_dir: explicit frames dir (overrides any default).
    - Otherwise: local files default next to the video; URLs are downloaded to
      the project's `descargas/` folder and default outputs go to `informes/`.

    autoclean: "ask" (prompt the user at the end), "keep" or "delete" the
    downloaded video and generated files.

    device_hint: "auto" (detect), "cpu" (force CPU; safer on laptops with weak
    cooling) or "cuda".

    cookies / cookies_from_browser / live_from_start: passed to yt-dlp to
    download logged-in content or capture supported live streams.
    """
    download_paths = []
    is_remote = is_url(video_path)

    if is_remote:
        os.makedirs(DEFAULT_DOWNLOAD_DIR, exist_ok=True)
        video_path = download_video(
            video_path,
            DEFAULT_DOWNLOAD_DIR,
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
            live_from_start=live_from_start,
        )
        download_paths.append(video_path)

    video_path = str(Path(video_path).resolve())
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    base_name = Path(video_path).stem

    # Base dir for default outputs.
    if output_dir is not None:
        base_dir = Path(output_dir).resolve()
    elif output_md is not None:
        base_dir = Path(output_md).resolve().parent
    elif is_remote:
        base_dir = Path(DEFAULT_OUTPUT_DIR).resolve()
    else:
        base_dir = Path(video_path).parent

    if output_md is None:
        output_md = str(base_dir / f"{base_name}_analysis.md")
    else:
        output_md = str(Path(output_md).resolve())
    os.makedirs(base_dir, exist_ok=True)

    if frames_dir is None:
        frames_dir = str(base_dir / f"frames_{base_name}")
    else:
        frames_dir = str(Path(frames_dir).resolve())

    # Frames se desactivan explícitamente (--no-frames) o si no hay vídeo
    # (archivos de solo audio: mp3, m4a, wav...).
    skip_frames = (not extract_frames) or (not has_video_stream(video_path))
    if skip_frames:
        if not extract_frames:
            log("Frames omitidos (--no-frames).")
        else:
            log("Archivo sin pista de vídeo (solo audio): frames omitidos.")
        frames_dir = None
    else:
        os.makedirs(frames_dir, exist_ok=True)

    device = detect_device(device_hint)

    with tempfile.TemporaryDirectory(prefix="video_to_md_") as tmpdir:
        # 1. Extract audio
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(video_path, audio_path)

        # 2. Transcribe (+ checkpoint del texto por si el equipo se apaga)
        transcript = transcribe_audio(audio_path, whisper_model, device)
        save_transcript_checkpoint(transcript, output_md)

        # 3. Extract frames
        frames = []
        if not skip_frames:
            frames = extract_frames(video_path, frames_dir, frame_interval)

        # 4. Informe parcial (transcripción + frames) para no perder progreso
        #    si el análisis LLM no termina (corte de luz, apagado, etc.).
        partial = {"resumen": "*(Análisis LLM en curso o pendiente)*"}
        build_markdown(
            video_path=video_path,
            transcript=transcript,
            frames=frames,
            analysis=partial,
            whisper_model=whisper_model,
            device=device,
            ollama_model=ollama_model,
            frames_dir=frames_dir,
            output_md=output_md,
        )
        log("Informe parcial escrito (transcripción + frames).")

        # 5. Analyze with Ollama (best-effort)
        analysis = {}
        if use_llm:
            try:
                analysis = analyze_transcript_in_parts(
                    transcript, ollama_model, include_timestamps=True
                )
            except Exception as exc:
                log(f"Ollama analysis skipped: {exc}", "WARN")
                analysis = {"resumen": "*(No se pudo analizar con Ollama)*"}
        else:
            log("Skipping Ollama analysis (--no-llm)")
            analysis = {"resumen": "*(Análisis LLM omitido)*"}

    # 6. Rebuild the markdown with the LLM analysis.
    build_markdown(
        video_path=video_path,
        transcript=transcript,
        frames=frames,
        analysis=analysis,
        whisper_model=whisper_model,
        device=device,
        ollama_model=ollama_model,
        frames_dir=frames_dir,
        output_md=output_md,
    )

    # 7. Ask whether to keep or delete the downloaded/generated files.
    maybe_cleanup(
        autoclean=autoclean,
        download_paths=download_paths,
        output_md=output_md,
        frames_dir=frames_dir,
    )

    return output_md


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a video: transcribe, extract ideas (via Ollama) "
        "and capture frames, outputting a structured Markdown report."
    )
    parser.add_argument("--video", "-v", required=False,
                        help="Path to the video file or URL (YouTube, etc.)")
    parser.add_argument("--transcript", default=None,
                        help="Ruta a un archivo de texto (transcripción de una "
                             "reunión). Genera un informe de apuntes aparte "
                             "(ideas, discusiones, conclusiones) sin video. "
                             "Excluye --video.")
    parser.add_argument("--output", "-o", default=None, help="Output .md path")
    parser.add_argument("--output-dir", default=None,
                        help="Base output dir for the .md and frames "
                             "(default: next to the video, or cwd for URLs)")
    parser.add_argument("--frames-dir", default=None,
                        help="Explicit frames dir (default: <output-dir>/frames_<video>)")
    parser.add_argument("--model", "-m", default=DEFAULT_WHISPER_MODEL,
                        help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--interval", "-i", type=float, default=DEFAULT_FRAME_INTERVAL,
                        help=f"Frame interval seconds (default: {DEFAULT_FRAME_INTERVAL})")
    parser.add_argument("--no-frames", action="store_true",
                        help="No extraer frames (solo transcripción + análisis)")
    parser.add_argument("--llm", default=DEFAULT_OLLAMA_MODEL,
                         help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--no-llm", action="store_true",
                         help="Skip Ollama analysis (transcription + frames only)")
    parser.add_argument("--split-chars", type=int, default=0,
                         help="Con --transcript: dividir el texto en partes de N "
                              "caracteres (~6000 recomendado) y generar un informe "
                              "por parte. 0 = analizar el texto completo (con truncado)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default=DEVICE_HINT,
                        help="Dispositivo para Whisper: auto (default), cpu "
                             "(más seguro en portátiles con refrigeración justa) "
                             "o cuda")
    parser.add_argument("--autoclean", choices=("ask", "keep", "delete"), default="ask",
                        help="Qué hacer al terminar con los archivos descargados y "
                             "generados: 'ask' (preguntar), 'keep' o 'delete' "
                             "(default: ask)")
    parser.add_argument("--cookies", default=None,
                        help="Ruta a un archivo de cookies (formato Netscape) para "
                             "descargar contenidos con login")
    parser.add_argument("--cookies-from-browser", default=None,
                        help="Reutilizar cookies de un navegador (chrome, firefox, "
                             "chromium, con :perfil opcional). Útil en plataformas "
                             "donde ya estás logeado")
    parser.add_argument("--live-from-start", action="store_true",
                        help="Si la URL es un directo soportado, descargar desde el "
                             "inicio de la transmisión")
    args = parser.parse_args(argv)

    try:
        if args.transcript:
            if args.video:
                raise ValueError("Usa solo uno: --video O --transcript, no ambos.")
            md = transcript_to_md(
                transcript_path=args.transcript,
                output_md=args.output,
                ollama_model=args.llm,
                use_llm=not args.no_llm,
                output_dir=args.output_dir,
                split_chars=args.split_chars,
            )
        elif args.video:
            md = video_to_md(
                video_path=args.video,
                output_md=args.output,
                whisper_model=args.model,
                frame_interval=args.interval,
                extract_frames=not args.no_frames,
                ollama_model=args.llm,
                use_llm=not args.no_llm,
                output_dir=args.output_dir,
                frames_dir=args.frames_dir,
                autoclean=args.autoclean,
                device_hint=args.device,
                cookies=args.cookies,
                cookies_from_browser=args.cookies_from_browser,
                live_from_start=args.live_from_start,
            )
        else:
            parser.error("Especifica --video o --transcript")
            return 1
    except Exception as exc:
        log(f"Error: {exc}", "ERROR")
        return 1

    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

