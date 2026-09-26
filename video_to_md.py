#!/usr/bin/env python3
"""Video to Markdown Analyzer.

Transcribe a video using Whisper (auto GPU/CPU), extracts key frames with
timestamps, and uses a local LLM via Ollama to produce an ordered summary of
the main ideas. Outputs a structured Markdown file.

Usage:
    python video_to_md.py --video <video_file> [--output <out.md>] [--model small] [--interval 10]
"""

import argparse
import contextlib
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
import warnings
import wave
from pathlib import Path, PurePath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hardware  # noqa: E402  (detección agnóstica de GPU/modelos)

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
# "auto" = se elige solo según lo que haya instalado en esta máquina
# (hardware.pick_ollama_model / hardware.recommend_whisper_model).
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "auto")
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "auto")
DEFAULT_FRAME_INTERVAL = 10.0  # seconds between captured frames
# Extensiones de archivos solo-audio (se transcriben igual pero sin frames).
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac",
              ".wma", ".aif", ".aiff", ".m4b", ".m4p", ".ape", ".amr",
              ".w64", ".au", ".ra", ".ac3", ".dts", ".alac"}
# Contenedores de video habituales. La lista NO es restrictiva: si el archivo
# descargado trae otra extensión, se acepta igual (ffmpeg/Whisper la leen).
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".ts", ".flv",
              ".wmv", ".mpg", ".mpeg", ".m2ts", ".mts", ".m2v", ".3gp", ".3g2",
              ".ogv", ".mp4v", ".vob", ".rm", ".rmvb", ".asf", ".divx", ".f4v",
              ".mxf", ".dv", ".gif", ".f4a", ".wtv", ".nsv", ".roq"}
# Acompañantes que yt-dlp deja junto al medio: nunca son el video descargado.
SIDECAR_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".vtt", ".srt", ".ass",
                ".ssa", ".lrc", ".sbv", ".json", ".xml", ".description", ".part",
                ".ytdl", ".temp", ".tmp", ".download"}
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Tamaño de parte para el análisis por trozos de transcripciones largas
# (unifica el flujo de video y de apuntes; ~6000 chars = JSON completo en ctx 4096).
ANALYSIS_PART_CHARS = int(os.environ.get("AUDITV_ANALYZE_PART_CHARS", "6000"))
# auto = Ollama a la GPU si la tarjeta es lo bastante potente (y la temperatura
# lo permite), si no a CPU. "1"/"-1" fuerza GPU, "0" fuerza CPU.
OLLAMA_NUM_GPU = os.environ.get("OLLAMA_NUM_GPU", "auto")
# Índice de la GPU que debe usar el LLM (main_gpu). None = auto/la de Ollama.
OLLAMA_GPU_INDEX = os.environ.get("OLLAMA_GPU_INDEX")
OLLAMA_NUM_THREADS = os.environ.get("OLLAMA_NUM_THREADS")
# auto|cpu|cuda: forzar Whisper a CPU evita picos de carga en la GPU.
DEVICE_HINT = os.environ.get("AUDITV_DEVICE", "auto")
# Guardas de temperatura de la GPU (nvidia-smi). Warn por encima de WARN,
# reposo por encima de ABORT (ese bache y los siguientes van a CPU hasta que
# la tarjeta baje de RESUME) para proteger el hardware.
GPU_TEMP_WARN = int(os.environ.get("AUDITV_GPU_TEMP_WARN", "80"))
GPU_TEMP_ABORT = int(os.environ.get("AUDITV_GPU_TEMP_ABORT", "92"))
GPU_TEMP_RESUME = int(os.environ.get("AUDITV_GPU_TEMP_RESUME", "75"))
GPU_TEMP_CHECK_INTERVAL = float(
    os.environ.get("AUDITV_GPU_TEMP_INTERVAL", "5.0")
)  # segundos entre muestras del monitor
# Segmentos (s) de audio por lote al transcribir en GPU: si la tarjeta se
# calienta, los lotes que falten siguen en CPU sin perder lo ya transcrito.
TRANSCRIBE_CHUNK_SEC = float(os.environ.get("AUDITV_TRANSCRIBE_CHUNK_SEC", "300"))
# Solape entre lotes de audio, para que una palabra no se parta en el corte.
_TRANSCRIBE_OVERLAP_SEC = 1.0
# 4096 evita que el LLM local se desborde de la VRAM de GPUs pequeñas y evita
# el apagado por sobrecalentamiento en portátiles. "auto" lo ajusta al equipo.
OLLAMA_NUM_CTX = os.environ.get("OLLAMA_NUM_CTX", "auto")

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
# Status file (lo lee la GUI para pintar el panel de GPU en tiempo real)
# ---------------------------------------------------------------------------
STATUS_FILE = os.environ.get("AUDITV_STATUS_FILE") or ""
_status = {"data": {}, "lock": threading.Lock()}


def status_reset(**fields) -> None:
    """Vacía el estado (empieza una ejecución) y escribe el fichero."""
    with _status["lock"]:
        _status["data"] = {"running": True, "started": time.time(),
                           "pid": os.getpid()}
        _status["data"].update(fields)
        _write_status()


def status_update(**fields) -> None:
    """Actualiza campos del estado y los escribe (no hace nada si no hay
    AUDITV_STATUS_FILE: así el CLI funciona igual sin la GUI)."""
    if not STATUS_FILE:
        return
    with _status["lock"]:
        _status["data"].update(fields)
        _status["data"]["updated"] = time.time()
        _write_status()


def status_set(running: bool = True, **fields) -> None:
    """Marca la ejecución como terminada (la GUI deja de mostrar actividad)."""
    with _status["lock"]:
        _status["data"].update(fields)
        _status["data"]["running"] = running
        _status["data"]["updated"] = time.time()
        _write_status()


def _write_status() -> None:
    """Volca el estado a disco (escritura atómica para que la GUI no lea
    un JSON a medias)."""
    if not STATUS_FILE:
        return
    tmp = f"{STATUS_FILE}.{os.getpid()}.tmp"
    try:
        payload = dict(_status["data"])
        payload["thresholds"] = {
            "warn": GPU_TEMP_WARN, "abort": GPU_TEMP_ABORT,
            "resume": GPU_TEMP_RESUME,
        }
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
    except Exception as exc:  # nunca romper el análisis por el panel
        log(f"No se pudo escribir el estado ({exc})", "WARN")
        return
    # En Windows, `os.replace` falla con PermissionError si la GUI tiene el
    # JSON abierto en ese instante (no comparte delete). Se reintenta un poco
    # porque, si no, el panel se queda congelado con datos viejos.
    for attempt in range(5):
        try:
            os.replace(tmp, STATUS_FILE)
            return
        except PermissionError:
            if attempt == 4:
                log("No se pudo actualizar el estado (archivo abierto en la "
                    "GUI); se reintentará en la próxima actualización.", "WARN")
                return
            time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            log(f"No se pudo escribir el estado ({exc})", "WARN")
            return
    try:
        os.unlink(tmp)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Device detection (GPU / CPU) y elección de modelos (agnóstica)
# ---------------------------------------------------------------------------
# Reexports de hardware para que la GUI y los wrappers usen el mismo sitio.
gpu_temperature = hardware.gpu_temperature
gpu_infos = hardware.gpu_infos
primary_gpu = hardware.primary_gpu
gpu_tier = hardware.gpu_tier
list_ollama_models = hardware.ollama_models
gpu_summary = hardware.hardware_summary


def gpu_index_list() -> list:
    """['auto', '0', '1', ...] con las GPUs NVIDIA detectadas."""
    gpus = hardware.gpu_infos()
    return ["auto"] + [str(g["index"]) for g in gpus] if gpus else ["auto"]


def detect_device(hint: str = DEVICE_HINT) -> str:
    """Device para Whisper: 'cuda', 'mps' (Mac) o 'cpu'.

    Hint: 'cpu' forces CPU (safe on laptops with weak cooling), 'cuda'/'mps'
    force that backend, 'auto' uses the GPU only if this PyTorch build really
    supports it (una GTX 1060 sm_61, por ejemplo, cae siempre a CPU).
    """
    if hint == "cpu":
        log("Device forzado a CPU (AUDITV_DEVICE=cpu / --device cpu).", "DEVICE")
        status_update(device="cpu", device_reason="forzado por el usuario")
        return "cpu"

    if hint == "mps":
        mps = hardware.torch_mps_info()
        if mps.get("usable"):
            log(f"Device forzado a MPS: {mps.get('name')}.", "DEVICE")
            status_update(device="mps", device_reason=mps.get("name"))
            return "mps"
        log(f"MPS pedido pero no usable ({mps.get('reason')}): CPU.", "DEVICE")
        status_update(device="cpu", device_reason=mps.get("reason"))
        return "cpu"

    info = hardware.torch_cuda_info()
    if not info.get("available"):
        # Sin CUDA: en macOS puede estar la GPU de Apple (MPS).
        mps = hardware.torch_mps_info()
        if mps.get("usable"):
            log(
                f"GPU para Whisper: {mps.get('name')} (Metal, "
                f"{hardware.fmt_vram(mps.get('vram_mb'))} de memoria "
                f"compartida).",
                "DEVICE",
            )
            status_update(device="mps", device_reason=mps.get("name"))
            return "mps"
        log(f"Sin GPU para PyTorch ({info.get('reason') or 'sin CUDA'}): CPU.",
            "DEVICE")
        status_update(device="cpu", device_reason=info.get("reason") or "sin CUDA")
        return "cpu"
    if not info.get("usable"):
        log(
            f"GPU {info.get('name')} (sm_{str(info.get('cap')).replace('.', '')}) "
            f"no la soporta este torch: {info.get('reason')}. Whisper irá en CPU "
            "(el LLM de Ollama sí puede aprovechar la tarjeta).",
            "DEVICE",
        )
        status_update(device="cpu", device_reason=info.get("reason"))
        return "cpu"
    try:
        import torch
        x = torch.ones(1, device="cuda")
        _ = (x + 1).item()  # prueba funcional real
    except Exception as exc:
        log(f"CUDA no utilizable ({exc}): se trabaja en CPU.", "DEVICE")
        status_update(device="cpu", device_reason=str(exc))
        return "cpu"

    log(
        f"GPU para Whisper: {info.get('name')} "
        f"({hardware.fmt_vram(info.get('vram_mb'))}, sm_{str(info.get('cap')).replace('.', '')})",
        "DEVICE",
    )
    status_update(device="cuda", device_reason=info.get("name"))
    return "cuda"


def resolve_whisper_model(requested: str = None, device: str = "cpu",
                          duration_s: float = None) -> str:
    """Convierte 'auto' (o vacío) en un modelo de Whisper concreto.

    No toca nada si ya viene un nombre explícito: así `--model small` sigue
    mandando sobre la recomendación.
    """
    requested = (requested or DEFAULT_WHISPER_MODEL or "auto").strip()
    if requested.lower() not in ("auto", ""):
        return requested
    vram = None
    if str(device).startswith("cuda"):
        vram = hardware.torch_cuda_info().get("vram_mb") or None
    elif str(device).startswith("mps"):
        vram = hardware.torch_mps_info().get("vram_mb") or None
    model = hardware.recommend_whisper_model(device, vram, duration_s)
    cached = hardware.whisper_cached_models()
    extra = f" Ya en caché: {', '.join(cached)}." if cached else ""
    on_gpu = not str(device).startswith("cpu")
    log(
        f"Modelo Whisper 'auto' -> {model} "
        f"({'GPU' if on_gpu else 'CPU'}, "
        f"{int((duration_s or 0) // 60)} min de audio).{extra}",
        "DEVICE",
    )
    status_update(whisper_model=model, whisper_model_auto=True)
    return model


def resolve_ollama_model(requested: str = None, use_gpu: bool = None) -> str:
    """Convierte 'auto' en el mejor modelo que haya instalado en Ollama.

    Así cada usuario acaba usando lo que tiene descargado, sin tener que
    acordarse de la etiqueta. Si Ollama no responde se avisa y se devuelve el
    nombre pedido (para que el error de Ollama sea explícito).
    """
    requested = (requested or DEFAULT_OLLAMA_MODEL or "auto").strip()
    if requested.lower() not in ("auto", ""):
        return requested
    models = hardware.ollama_models()
    if not models:
        log(
            "No se pudo leer la lista de modelos de Ollama "
            f"({OLLAMA_URL}). Arranca 'ollama serve' o fija el modelo con "
            "--llm <nombre> / OLLAMA_MODEL.",
            "WARN",
        )
        status_update(llm_model=None, llm_model_auto=False)
        return requested
    best, info = hardware.pick_ollama_model(models, use_gpu=use_gpu)
    names = ", ".join(m["name"] for m in models)
    log(
        f"Modelos en Ollama: {names} -> usando '{best}' "
        f"({hardware.model_size_gb(info):.1f} GB, "
        f"{'con GPU' if hardware.ollama_budget_gb(use_gpu) > hardware.CPU_BUDGET_GB else 'en CPU'}).",
        "LLM",
    )
    status_update(llm_model=best, llm_model_auto=True,
                  llm_models=[m["name"] for m in models])
    return best


def resolve_num_ctx(device_for_llm: str = "cpu") -> int:
    """Contexto de Ollama: 'auto' lo ajusta a la VRAM disponible."""
    raw = str(OLLAMA_NUM_CTX).strip().lower()
    if raw not in ("auto", ""):
        try:
            return int(float(raw))
        except ValueError:
            pass
    if str(device_for_llm) == "gpu":
        gpu = primary_gpu()
        vram = gpu.get("vram_total_mb") or 0
        return 8192 if vram >= 10240 else 4096
    return 4096


def ollama_use_gpu_default() -> bool:
    """¿Manda Ollama a la GPU? (OLLAMA_NUM_GPU=auto -> según la potencia)."""
    raw = str(OLLAMA_NUM_GPU).strip().lower()
    if raw in ("auto", ""):
        return hardware.ollama_gpu_default()[0]
    try:
        return int(float(raw)) != 0
    except ValueError:
        return hardware.ollama_gpu_default()[0]


def transcript_checkpoint_path(output_md: str) -> str:
    """Ruta del .txt de transcripción que acompaña al informe."""
    base = Path(output_md)
    return str(base.with_name(base.stem + "_transcripcion.txt"))


def _log_environment(device: str) -> None:
    """Log de arranque: qué equipo ha encontrado y cómo va a usarlo."""
    for line in hardware.log_lines():
        log(line, "DEVICE")
    gpu = primary_gpu()
    if gpu:
        use_gpu, reason = hardware.ollama_gpu_default()
        log(f"IA local: {reason}.", "LLM")
        log(
            f"Temperatura GPU: aviso {GPU_TEMP_WARN}°C, descanso {GPU_TEMP_ABORT}°C, "
            f"se retoma a {GPU_TEMP_RESUME}°C (medición cada {GPU_TEMP_CHECK_INTERVAL:g} s).",
            "GPU",
        )
    status_update(gpu_info=hardware.hardware_summary())


# ---------------------------------------------------------------------------
# GPU health monitoring (avoids thermal shutdowns on laptops / weak GPUs)
# ---------------------------------------------------------------------------
class GpuThermalGuard:
    """Decide si la GPU puede trabajar o si debe descansar.

    La GPU entra en 'reposo' al superar GPU_TEMP_ABORT y no vuelve al trabajo
    hasta bajar de GPU_TEMP_RESUME (histéresis: evita subir y bajar cada bache).
    Mientras descansa, cada bache se resuelve en CPU **sin perder su
    resultado**, y en cuanto la tarjeta se enfría el trabajo vuelve a la GPU.
    """

    def __init__(self):
        self.resting = False
        self.last_temp = None
        self.reason = ""
        self._lock = threading.Lock()

    # -- estado -------------------------------------------------------------
    def has_gpu(self) -> bool:
        return bool(hardware.gpu_infos())

    def temp(self) -> float:
        temp = hardware.gpu_temperature()
        if temp is not None:
            self.last_temp = temp
        return temp

    def mark_rest(self, reason: str) -> None:
        with self._lock:
            if not self.resting:
                log(
                    f"GPU en descanso: {reason}. Los lotes siguientes van en CPU "
                    f"hasta que baje de {GPU_TEMP_RESUME}°C (para que el equipo no "
                    "se apague). No se pierde nada: cada lote se guarda igualmente.",
                    "WARN",
                )
            self.resting = True
            self.reason = reason
        status_update(gpu_resting=True, gpu_rest_reason=reason)
        # Que la tarjeta se enfríe de verdad: Ollama suelta el modelo de la GPU.
        threading.Thread(target=unload_ollama_model, daemon=True).start()

    def clear_rest(self) -> None:
        with self._lock:
            if self.resting:
                log(
                    f"GPU ya está a {self.last_temp:.0f}°C (≤ {GPU_TEMP_RESUME}°C): "
                    "vuelve a trabajar en la GPU.",
                    "INFO",
                )
            self.resting = False
            self.reason = ""
        status_update(gpu_resting=False, gpu_rest_reason="")

    def should_use_gpu(self, what: str = "") -> bool:
        """¿Toca usar la GPU ahora mismo? (respeta el descanso en curso)"""
        if not self.has_gpu():
            return False
        temp = self.temp()
        if temp is None:
            return not self.resting
        if self.resting:
            if temp <= GPU_TEMP_RESUME:
                self.clear_rest()
            else:
                return False
        if temp >= GPU_TEMP_ABORT:
            self.mark_rest(
                f"{temp:.0f}°C (límite {GPU_TEMP_ABORT}°C)"
                + (f" antes de '{what}'" if what else "")
            )
            return False
        return True

    def snapshot(self) -> dict:
        return {
            "resting": self.resting,
            "reason": self.reason,
            "temp_c": self.temp(),
            "warn": GPU_TEMP_WARN,
            "abort": GPU_TEMP_ABORT,
            "resume": GPU_TEMP_RESUME,
        }


_thermal = {"event": threading.Event()}
_open_resp = {"resp": None}
GUARD = GpuThermalGuard()


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
            "--ollama-gpu cpu OLLAMA_NUM_CTX=2048"
        )
    if temp >= GPU_TEMP_WARN:
        log(
            f"GPU ya está a {temp:.0f}°C antes de '{stage}': riesgo alto de "
            "apagado por temperatura. Reintenta con --device cpu y/o "
            "--ollama-gpu cpu para no exigir a la tarjeta.",
            "WARN",
        )


class _GpuMonitor(threading.Thread):
    """Hilo que vigila la temperatura cada GPU_TEMP_CHECK_INTERVAL segundos.

    Avisa mientras sube, y al llegar al límite crítico no mata el proceso: pone
    la GPU en descanso y cancela la consulta en vuelo de Ollama, que se
    reintenta en CPU para que ese lote no se pierda. El informe se escribe
    igualmente con lo que se haya podido analizar.
    """

    def __init__(self, stage: str):
        super().__init__(daemon=True)
        self._stage = stage
        self._stop = threading.Event()
        self._warned = False

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(GPU_TEMP_CHECK_INTERVAL):
            temp = hardware.gpu_temperature(max_age=0)
            if temp is None:
                continue
            status_update(gpu_temp=temp)
            if temp >= GPU_TEMP_ABORT:
                log(
                    f"GPU a {temp:.0f}°C durante '{self._stage}': temperatura "
                    f"crítica ({GPU_TEMP_ABORT}°C). La GPU pasa a descansar y el "
                    "análisis LLM se termina en CPU (el informe se genera igual).",
                    "ERROR",
                )
                GUARD.mark_rest(f"{temp:.0f}°C durante '{self._stage}'")
                _thermal["event"].set()
                _cancel_open_request()
            elif temp >= GPU_TEMP_WARN:
                if not self._warned:
                    self._warned = True
                log(
                    f"GPU a {temp:.0f}°C durante '{self._stage}' "
                    f"(aviso {GPU_TEMP_WARN}°C, descanso {GPU_TEMP_ABORT}°C).",
                    "WARN",
                )
            else:
                self._warned = False
                log(
                    f"GPU a {temp:.0f}°C durante '{self._stage}' "
                    f"(límite {GPU_TEMP_ABORT}°C).",
                    "GPU",
                )


class ThermalAbort(RuntimeError):
    """Raised when the GPU reaches the critical temperature."""


def _cancel_open_request() -> None:
    """Close the in-flight Ollama response so a blocking read() raises now."""
    resp = _open_resp.get("resp")
    if resp is not None:
        try:
            resp.close()
        except Exception:
            pass


# Modelo que se está usando ahora mismo (para poder soltarlo de la GPU).
_ollama_state = {"model": None}


def unload_ollama_model(model: str = None) -> None:
    """Pide a Ollama que suelte el modelo de la VRAM (keep_alive=0).

    Así la GPU se enfría de verdad mientras el análisis sigue en CPU. Si Ollama
    no está o falla, no pasa nada: es una optimización, no un requisito.
    """
    model = model or _ollama_state.get("model")
    if not model:
        return
    try:
        payload = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        log(f"Ollama suelta '{model}' de la GPU (keep_alive=0) para enfriarla.", "GPU")
        status_update(gpu_unloaded_model=model)
    except Exception as exc:
        log(f"No se pudo descargar el modelo de la GPU: {exc}", "WARN")


def media_duration(path: str) -> float:
    """Duración en segundos del medio (0 si ffprobe no responde)."""
    try:
        out = subprocess.run(
            [*_tool_cmd("ffprobe"), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------
def _tool_cmd(name: str) -> list:
    """Cómo invocar una herramienta externa (ffmpeg, ffprobe, yt-dlp).

    Se usa `shutil.which` para el PATH, y si no aparece se recurre al
    intérprete del venv: en Windows los ejecutables de pip caen en
    `venv\\Scripts` (o `venv\\bin`), que no suele estar en el PATH cuando la
    GUI se abre con doble clic. Devolver una lista vacía hace que el comando
    falle con un mensaje claro en vez de con un FileNotFoundError.
    """
    found = shutil.which(name)
    if found:
        return [found]
    candidates = [
        Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else "")),
        PROJECT_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin")
        / (name + (".exe" if os.name == "nt" else "")),
    ]
    for cand in candidates:
        if cand.exists():
            return [str(cand)]
    log(f"No se encuentra '{name}'. Instálalo y ponlo en el PATH "
        f"(yt-dlp: pip install -r requirements.txt).", "ERROR")
    return []


def extract_audio(video_path: str, output_wav: str) -> str:
    """Extract mono 16kHz WAV audio from video via ffmpeg."""
    cmd = [
        *_tool_cmd("ffmpeg"), "-y", "-i", video_path,
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
            [*_tool_cmd("ffprobe"), "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30,
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
def _segments_of(result: dict, offset: float = 0.0) -> list:
    """Segmentos de un resultado de Whisper con los tiempos desplazados."""
    out = []
    for seg in result.get("segments") or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.get("start", 0.0)) + offset, 2),
            "end": round(float(seg.get("end", 0.0)) + offset, 2),
            "text": text,
        })
    return out


def _whisper_call(model, device: str, source, language: str = None) -> dict:
    """Una llamada a Whisper con silenciados los warnings genéricos de torch."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Performing inference on CPU when CUDA is available",
        )
        kwargs = {"fp16": device == "cuda", "condition_on_previous_text": False}
        if language:
            kwargs["language"] = language
        return model.transcribe(source, **kwargs)


def _probe_wav_duration(audio_path: str) -> float:
    with contextlib.closing(wave.open(audio_path, "rb")) as wav:
        return wav.getnframes() / float(wav.getframerate() or 1)


def _read_wav_chunks(audio_path: str, chunk_sec: float, overlap_sec: float):
    """Parte un WAV 16 kHz mono en trozos de `chunk_sec` segundos.

    Los trozos se solapan `overlap_sec` para que el modelo tenga contexto del
    corte (una palabra partida no se pierde); el solape se descarta después.
    """
    import numpy as np

    with contextlib.closing(wave.open(audio_path, "rb")) as wav:
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")

    step = max(1, int(rate * chunk_sec))
    overlap = max(0, int(rate * overlap_sec))
    pos = 0
    while pos < samples.size:
        end = min(samples.size, pos + step)
        audio = samples[pos:end].astype("float32") / 32768.0
        if audio.size:
            yield pos / rate, audio
        if end >= samples.size:
            break
        pos = max(pos + 1, end - overlap)


def transcribe_audio(audio_path: str, model_size: str, device: str,
                     checkpoint_path: str = None) -> dict:
    """Transcribe audio with Whisper, return dict with 'text' and 'segments'.

    segments: list of {"start": seconds, "end": seconds, "text": str}

    En GPU se transcribe por lotes (TRANSCRIBE_CHUNK_SEC): antes de cada lote
    se mira la temperatura, así que si la tarjeta se calienta los lotes que
    falten se hacen en CPU **sin perder lo ya transcrito** (además se guarda el
    checkpoint tras cada lote). En cuando la GPU se enfría, se vuelve a ella.
    """
    global whisper
    if whisper is None:
        try:
            import whisper
        except Exception as exc:
            pip = (r"venv\Scripts\pip.exe" if os.name == "nt"
                   else "./venv/bin/pip")
            raise RuntimeError(
                f"openai-whisper is not installed. Run: "
                f"{pip} install openai-whisper"
            ) from exc

    log(f"Loading Whisper model '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)
    if device == "cpu":
        log(
            "Transcribiendo en CPU (la GPU no es compatible con este PyTorch "
            "o elegiste --device cpu). EL warning 'Performing inference on CPU "
            "when CUDA is available' de Whisper es genérico y se ignora.",
            "DEVICE",
        )
    status_update(whisper_device=device, whisper_model=model_size)

    if device != "cpu" and TRANSCRIBE_CHUNK_SEC > 0:
        try:
            duration = _probe_wav_duration(audio_path)
        except Exception as exc:
            log(f"No se pudo calcular la duración del audio: {exc}", "WARN")
            duration = 0.0
        if duration > TRANSCRIBE_CHUNK_SEC:
            return _transcribe_chunked(model, audio_path, device, duration,
                                       checkpoint_path)
    return _transcribe_single(model, audio_path, device)


def _transcribe_single(model, audio_path: str, device: str) -> dict:
    """Pasada única: todo el audio de una vez (CPU o audio corto)."""
    monitor = None
    if device != "cpu":
        check_gpu_health("transcripción")
        monitor = _GpuMonitor("transcripción")
        monitor.start()
    try:
        log("Transcribiendo…", "AUDITV")
        result = _whisper_call(model, device, audio_path)
    finally:
        if monitor is not None:
            monitor.stop()
            monitor.join(timeout=GPU_TEMP_CHECK_INTERVAL + 1)

    segments = _segments_of(result)
    text = str(result.get("text") or "").strip()
    log(f"Transcripción completa: {len(segments)} segmentos.")
    status_update(whisper_segments=len(segments))
    return {"text": text, "segments": segments}


def _transcribe_chunked(model, audio_path: str, device: str, duration: float,
                        checkpoint_path: str = None) -> dict:
    """Transcribe por lotes de TRANSCRIBE_CHUNK_SEC, saltando GPU<->CPU si toca.

    El lote se considera entregado cuando termina (se guarda su checkpoint),
    así que un cambio de dispositivo a mitad no cuesta nada: nunca se pierde un
    lote por el camino.
    """
    n_batches = int(duration // TRANSCRIBE_CHUNK_SEC) + 1
    log(
        f"Audio de {int(duration // 60)} min: transcripción en {n_batches} lotes "
        f"de ~{int(TRANSCRIBE_CHUNK_SEC // 60)} min para poder descansar la GPU.",
        "AUDITV",
    )
    segments = []
    language = None
    current = device
    monitor = _GpuMonitor("transcripción")
    monitor.start()
    t_start = time.time()
    try:
        batches = _read_wav_chunks(audio_path, TRANSCRIBE_CHUNK_SEC,
                                   _TRANSCRIBE_OVERLAP_SEC)
        for i, (offset, audio) in enumerate(batches, 1):
            # Antes de cada lote se decide dónde se hace: la GPU entra en
            # descanso cuando se calienta y vuelve cuando se enfría.
            want = device
            if device != "cpu" and not GUARD.should_use_gpu("transcripción"):
                want = "cpu"
            if want != current:
                _move_whisper(model, want)
                if want == "cpu":
                    log(
                        f"Lote {i}/{n_batches}: la GPU está caliente, este lote va "
                        "en CPU. Lo ya transcrito no se pierde.",
                        "WARN",
                    )
                else:
                    log(
                        f"Lote {i}/{n_batches}: la GPU ya está fría, se retoma en GPU.",
                        "DEVICE",
                    )
                current = want

            result = _whisper_call(model, current, audio, language=language)
            if i == 1 and result.get("language"):
                language = result["language"]
                log(f"Idioma detectado: {language}.", "DEVICE")

            # La cola del lote anterior se repite en este (es el solape): se
            # descarta la vieja transcripción y se queda con la más reciente.
            if segments and offset > 0:
                segments[:] = [s for s in segments if s["start"] < offset]

            new_segs = _segments_of(result, offset)
            segments.extend(new_segs)
            status_update(
                stage="transcripción",
                whisper_batch=f"{i}/{n_batches}",
                whisper_batch_device="GPU" if current != "cpu" else "CPU",
                whisper_segments=len(segments),
            )
            log(
                f"Lote {i}/{n_batches} desde {fmt_ts(offset)} en "
                f"{'GPU' if current != 'cpu' else 'CPU'}: {len(new_segs)} "
                f"segmentos ({len(segments)} en total).",
                "AUDITV",
            )
            if checkpoint_path:
                _checkpoint_transcript(segments, checkpoint_path)
    finally:
        monitor.stop()
        monitor.join(timeout=GPU_TEMP_CHECK_INTERVAL + 1)

    log(
        f"Transcripción completa: {len(segments)} segmentos en "
        f"{int(time.time() - t_start)} s.",
    )
    status_update(whisper_segments=len(segments), whisper_batch=None,
                  whisper_batch_device=None)
    return {"text": _transcript_text(segments), "segments": segments}


def _move_whisper(model, device: str) -> None:
    """Mueve el modelo de Whisper entre GPU y CPU sin recargarlo."""
    try:
        model.to(device)
    except Exception as exc:
        log(f"No se pudo mover Whisper a {device}: {exc}", "WARN")


def _transcript_text(segments: list) -> str:
    """Texto corrido a partir de los segmentos (única fuente de verdad)."""
    return " ".join(s["text"] for s in segments if s.get("text")).strip()


def _checkpoint_transcript(segments: list, path: str) -> None:
    """Guarda lo transcrito hasta ahora (nada se pierde si algo falla)."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_transcript_text(segments) + "\n")
    except Exception as exc:
        log(f"No se pudo guardar el checkpoint de la transcripción: {exc}", "WARN")



# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_video_frames(video_path: str, output_dir: str,
                         interval_sec: float) -> list:
    """Extract frames every interval_sec. Returns list of {path, timestamp}."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        *_tool_cmd("ffmpeg"), "-y", "-i", video_path,
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
def _num_gpu_layers() -> int:
    """Capas del LLM que se mandan a la GPU (OLLAMA_NUM_GPU: auto/1/-1/0)."""
    raw = str(OLLAMA_NUM_GPU).strip().lower()
    if raw in ("auto", ""):
        return -1  # tantas capas como quepan: decide Ollama
    try:
        return int(float(raw))
    except ValueError:
        return -1


def _ollama_generate(prompt: str, model: str, system: str = "", label: str = "") -> str:
    """Send a prompt to a local Ollama server and return the raw text output.

    Cada bache se hace en GPU salvo que la temperatura la desaconseje
    (OLLAMA_NUM_GPU=auto decide además según lo potente que sea la tarjeta):

    * si la GPU está en descanso por calor, el bache va a CPU y **no se pierde**;
    * si se calienta durante la consulta, ESA misma consulta se reintenta en CPU
      (tampoco se pierde el bache) y la GPU queda libre para enfriarse;
    * en cuanto la tarjeta baja de GPU_TEMP_RESUME, los siguientes baches
      vuelven a la GPU solos.
    """
    gpu_wanted = ollama_use_gpu_default()
    use_gpu = gpu_wanted and GUARD.should_use_gpu(label or "análisis")
    if gpu_wanted and not use_gpu:
        log(
            f"'{label or 'análisis'}' se resuelve en CPU: la GPU descansa "
            f"({GUARD.reason or 'temperatura alta'}). El resultado del bache se "
            "guarda igualmente.",
            "WARN",
        )
    _ollama_state["model"] = model
    status_update(llm_device="GPU" if use_gpu else "CPU", llm_gpu_wanted=gpu_wanted)
    try:
        return _ollama_generate_once(prompt, model, system, label, use_gpu)
    except ThermalAbort:
        if not use_gpu:
            raise
        _thermal["event"].clear()
        log(
            f"GPU en temperatura crítica durante '{label or 'análisis'}': se "
            "reintenta ESTA misma consulta en CPU para conservar el resumen/"
            "conclusiones de esta parte (la GPU queda libre para enfriarse).",
            "WARN",
        )
        status_update(llm_device="CPU")
        return _ollama_generate_once(prompt, model, system, label, use_gpu=False)


def _ollama_generate_once(
    prompt: str, model: str, system: str, label: str, use_gpu: bool
) -> str:
    """Envía una consulta a Ollama (con o sin GPU). Puede lanzar ThermalAbort."""
    if _thermal["event"].is_set() and use_gpu:
        raise ThermalAbort(GPU_TEMP_ABORT)

    options = {
        "num_ctx": resolve_num_ctx("gpu" if use_gpu else "cpu"),
        "num_gpu": _num_gpu_layers() if use_gpu else 0,
    }
    if OLLAMA_GPU_INDEX not in (None, "", "auto"):
        try:
            options["main_gpu"] = int(OLLAMA_GPU_INDEX)
        except ValueError:
            pass
    if OLLAMA_NUM_THREADS is not None:
        options["num_thread"] = int(OLLAMA_NUM_THREADS)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": options,
    }
    if system:
        payload["system"] = system

    monitor = None
    if use_gpu:
        monitor = _GpuMonitor("análisis LLM (Ollama)")
        monitor.start()
    started = time.time()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        _open_resp["resp"] = None
        stop_beat = threading.Event()
        gpu_mode = "CPU" if not use_gpu else (
            f"GPU {OLLAMA_GPU_INDEX}"
            if OLLAMA_GPU_INDEX not in (None, "", "auto") else "GPU"
        )
        log(f"LLM [{model}] {label or 'análisis'}: enviando consulta… ({gpu_mode})", "INFO")

        def _heartbeat() -> None:
            # Con Ollama en CPU el análisis tarda; avisa cada 20 s de que sigue.
            phase = "leyendo/analizando la transcripción"
            while not stop_beat.wait(20):
                elapsed = int(time.time() - started)
                if elapsed >= 40:
                    phase = f"generando el {label or 'análisis'}…"
                log(
                    f"LLM [{model}] {label or 'análisis'}: {phase} ({elapsed} s). "
                    f"Esperando su respuesta…",
                    "INFO",
                )
                # Refresca el panel de la GUI (con la temperatura incluida).
                status_update(gpu_temp=GUARD.temp(), stage=label or "análisis LLM")

        heartbeat = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat.start()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                _open_resp["resp"] = resp
                data = json.loads(resp.read().decode("utf-8"))
                _open_resp["resp"] = None
                if _thermal["event"].is_set() and use_gpu:
                    raise ThermalAbort(GPU_TEMP_ABORT)
                return data.get("response", "").strip()
        except ThermalAbort:
            log("Análisis LLM cancelado por temperatura crítica.", "ERROR")
            raise
        except Exception as exc:
            if _thermal["event"].is_set() and use_gpu:
                # El monitor cerró la consulta porque se calentó la GPU: no es
                # un fallo del LLM, es calor. Se relanza como ThermalAbort para
                # que este bache se reintente en CPU y no se pierda.
                log(
                    f"La consulta se interrumpió al calentarse la GPU: {exc}",
                    "WARN",
                )
                raise ThermalAbort(GPU_TEMP_ABORT) from exc
            log(f"LLM [{model}] {label or 'análisis'}: falló la consulta: {exc}", "ERROR")
            raise
        finally:
            stop_beat.set()
    finally:
        _open_resp["resp"] = None
        if monitor is not None:
            monitor.stop()
            monitor.join(timeout=GPU_TEMP_CHECK_INTERVAL + 1)
        log(
            f"LLM [{model}] {label or 'análisis'}: respuesta recibida "
            f"en {int(time.time() - started)} s.",
            "INFO",
        )


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
    result = _extract_json(_ollama_generate(
        prompt, model, system, label="análisis (resumen/ideas/conclusiones)"
    ))
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
    result2 = _extract_json(_ollama_generate(
        strict, model, system, label="reintento (JSON estricto)"
    ))
    for key, val in (result2 or {}).items():
        if val:
            result.setdefault(key, val)
    if not result.get("resumen"):
        result = result2 or result
    return result


def _merge_resumenes_with_llm(analyses: list, model: str) -> str:
    """Una pasada extra del LLM que fusiona los resúmenes de todas las partes
    en un único resumen ejecutivo. Devuelve "" si falla (se conserva el de partes)."""
    system = (
        "You are an expert meeting/video analyst. You receive the partial "
        "summaries of one meeting and must merge them into a single executive "
        "summary. Always answer in Spanish using ONLY valid JSON, no extra text."
    )
    partes = []
    for i, a in enumerate(analyses, 1):
        r = str(a.get("resumen") or "")
        cs = "; ".join(str(c) for c in (a.get("conclusiones") or [])[:3])
        if r or cs:
            partes.append(f"Parte {i}: {r}{(' | Conclusiones: ' + cs) if cs else ''}")
    prompt = (
        "Fusiona los siguientes resúmenes parciales de una misma reunión en "
        "UN único resumen ejecutivo en español, de 2 a 4 oraciones, coherente "
        "y sin mencionar 'parte 1', 'parte 2', etc.\n"
        "Responde ÚNICAMENTE con JSON: {\"resumen\": \"texto\"}\n\n"
        f"RESÚMENES PARCIALES:\n" + "\n".join(partes)
    )
    try:
        result = _extract_json(_ollama_generate(
            prompt, model, system, label="fusión del resumen ejecutivo"
        ))
        return str(result.get("resumen", "")).strip()
    except Exception as exc:
        log(f"Fusión de resúmenes falló: {exc}", "WARN")
        return ""


def merge_analyses(analyses: list, model: str = None) -> dict:
    """Fusiona el análisis de varias partes en un solo dict (sin re-analizar).

    Junta las listas de ideas/discusiones/conceptos/conclusiones eliminando
    duplicados. El resumen: si hay varias partes y se puede, se hace una pasada
    extra del LLM para obtener UN resumen ejecutivo único (con 'model'); si no,
    se etiqueta cada parte.
    """
    if not analyses:
        return {}
    result = {"titulo": "", "resumen": "", "ideas": [], "discusiones": [], "conceptos": [], "conclusiones": []}

    resumenes = [a.get("resumen") for a in analyses if a.get("resumen")]
    if resumenes:
        if len(resumenes) == 1:
            result["resumen"] = str(resumenes[0])
        else:
            merged = _merge_resumenes_with_llm(analyses, model) if model else ""
            if merged:
                result["resumen"] = merged
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


def analyze_transcript_in_parts(
    transcript_or_text,
    model: str,
    include_timestamps: bool = True,
    out_md: str = None,
    video_path: str = None,
    part_chars: int = ANALYSIS_PART_CHARS,
    with_details: bool = False,
    use_llm: bool = True,
):
    """Analiza la transcripción completa dividiéndola en partes y fusionando.

    Flujo unificado (video y apuntes): si el texto es largo se parte en
    trozos de ANALYSIS_PART_CHARS, cada trozo se analiza por separado y los
    resultados se fusionan. Para videos, los timestamps de cada parte se
    desplazan al segundo real del video.

    Si se pasa 'out_md' (y el texto es largo), además guarda junto al informe
    un .txt y un .md por cada parte (igual para video y apuntes).

    Con 'with_details=True' devuelve (analisis, partes, analisis_por_parte)
    para que el llamador pueda ensamblar el informe completo fusionado.
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

    model = resolve_ollama_model(model)
    parts = split_text(text, part_chars)
    if not parts:
        parts = [text]
    if len(parts) == 1:
        status_update(stage="análisis LLM", batch="1/1")
        analysis = analyze_with_ollama(text, model, include_timestamps)
        if with_details:
            return analysis, parts, [analysis]
        return analysis

    log(f"Transcripción larga ({len(text)} chars): análisis en {len(parts)} partes de ~{part_chars}.")
    analyses = []
    consumed = 0
    n_total = len(parts)
    for i, part in enumerate(parts, 1):
        part_analysis = {"resumen": "*(Sin contenido en esta parte)*"}
        status_update(batch=f"{i}/{n_total}", stage="análisis LLM",
                      batch_chars=len(part or ""))
        if not use_llm:
            part_analysis = {
                "resumen": "*(Análisis LLM omitido)*",
                "ideas": [], "discusiones": [], "conceptos": [], "conclusiones": [],
            }
        elif time_at is not None and part:
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
        if out_md is not None:
            _save_part_files(part, part_analysis, out_md, video_path, i, n_total, model)

    merged = merge_analyses(analyses, model)
    status_update(batch=None)
    if with_details:
        return merged, parts, analyses
    return merged


def _save_part_files(
    part_text, analysis, out_md, source_path, part_idx, n_total, ollama_model=None
):
    """Guarda el .txt (bache de ~6000 chars) y su .md en la subcarpeta
    '<informe>_partes' junto a la salida (igual para video y apuntes)."""
    base = Path(out_md)
    parts_dir = base.parent / f"{base.stem}_partes"
    parts_dir.mkdir(parents=True, exist_ok=True)
    txt = parts_dir / f"{base.stem}_parte{part_idx}.txt"
    txt.write_text(_format_running_text(part_text), encoding="utf-8")
    md = parts_dir / f"{base.stem}_parte{part_idx}.md"
    build_part_markdown(
        source_path, part_idx, n_total, part_text, analysis, str(md), ollama_model
    )
    log(
        f"Bache {part_idx}: {txt.name} -> {md.name} (en la carpeta {parts_dir.name}/)",
        "AUDITV",
    )



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
            try:
                rel = os.path.relpath(str(Path(f["path"]).resolve()), md_dir)
            except ValueError:
                # En Windows, relpath lanza si el .md y los frames están en
                # unidades distintas; se deja la ruta absoluta.
                rel = None
            # Siempre con "/" hacia delante: en Markdown la "\" es carácter de
            # escape, así que `..\frames\a.png` no renderiza en ningún visor.
            if rel is not None and not rel.startswith(".."):
                ref = PurePath(rel).as_posix()
        lines.append(f"### 🖼️ T={stamp}")
        lines.append(f"![Frame T={stamp}]({ref})")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------
def _engine_note() -> str:
    """Una línea con el equipo usado (GPU y si está en descanso por calor)."""
    gpu = primary_gpu()
    if not gpu:
        return "sin GPU NVIDIA (todo en CPU)"
    note = f"{gpu['name']} · {hardware.fmt_vram(gpu.get('vram_total_mb'))} · nivel {hardware.TIER_LABEL.get(hardware.gpu_tier(gpu), '?')}"
    if GUARD.resting:
        note += " · en descanso por temperatura"
    return note


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
    llm_device: str = None,
) -> None:
    """Generate the final structured Markdown report."""
    video_name = Path(video_path).name
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"# 🎬 Análisis: {video_name}")
    lines.append("")
    lines.append(f"- **Generado:** {now}")
    lines.append(
        f"- **Modelo Whisper:** {whisper_model} "
        f"({'GPU' if str(device).startswith('cuda') else 'CPU'})"
    )
    llm_dev = f" ({llm_device})" if llm_device else ""
    lines.append(f"- **Modelo LLM:** {ollama_model}{llm_dev}")
    lines.append(f"- **Motor:** {_engine_note()}")
    lines.append("")

    if analysis.get("titulo"):
        lines.append(f"## 🏷️ {analysis['titulo']}")
        lines.append("")

    lines.append("## 📌 Resumen ejecutivo")
    lines.append("")
    lines.append(_format_running_text(analysis.get("resumen", "*(Sin resumen disponible)*")))
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


def build_part_markdown(
    source_path: str,
    part_idx: int,
    part_total: int,
    part_text: str,
    analysis: dict,
    output_md: str,
    ollama_model: str = None,
) -> None:
    """Write a self-contained .md for one analysed part.

    Shared by the video flow (per-part files next to the report) and the
    Apuntes flow (single report or per-part files), so both behave the same:
    resumen, ideas, discusiones, conceptos, conclusiones and the part text.
    With part_total > 1 the titles say "… de esta parte"; otherwise it is a
    standalone notes report.
    """
    name = Path(source_path).name if source_path else "(fuente)"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_part = bool(part_total and part_total > 1)
    lines = []
    if is_part:
        lines.append(f"# 🎬 {name} — Parte {part_idx}")
    else:
        lines.append(f"# 📝 Apuntes: {name}")
    lines.append("")
    lines.append(f"- **Generado:** {now}")
    if ollama_model:
        lines.append(f"- **Modelo LLM:** {ollama_model}")
    lines.append("")

    if analysis.get("titulo"):
        lines.append(f"## 🏷️ {analysis['titulo']}")
        lines.append("")

    sufx = " de esta parte" if is_part else ""
    lines.append(f"## 📌 Resumen{sufx}")
    lines.append("")
    lines.append(_format_running_text(analysis.get("resumen", "*(Sin resumen disponible)*")))
    lines.append("")

    lines.append(f"## 🧠 Ideas principales{sufx}")
    lines.append("")
    ideas = analysis.get("ideas", [])
    if ideas:
        for i, idea in enumerate(ideas, 1):
            ts = idea.get("timestamp") if isinstance(idea, dict) else None
            ts_str = f" `[{fmt_ts(float(ts))}]`" if ts else ""
            titulo = idea.get("titulo", "Sin título") if isinstance(idea, dict) else str(idea)
            lines.append(f"### {i}. {titulo}{ts_str}")
            lines.append("")
            desc = idea.get("descripcion", "") if isinstance(idea, dict) else ""
            if desc:
                lines.append(desc)
                lines.append("")
    else:
        lines.append("*(No se detectaron ideas principales.)*")
        lines.append("")

    lines.append(f"## 💬 Discusiones / Puntos debatidos{sufx}")
    lines.append("")
    discussions = analysis.get("discusiones", [])
    lines.append(("- " + "\n- ".join(str(d) for d in discussions)) if discussions
                 else "*(Sin discusiones destacadas)*")
    lines.append("")

    lines.append(f"## 🔑 Conceptos clave{sufx}")
    lines.append("")
    concepts = analysis.get("conceptos", [])
    lines.append(("- " + "\n- ".join(str(c) for c in concepts)) if concepts
                 else "*(Sin conceptos destacados)*")
    lines.append("")

    lines.append(f"## ✅ Conclusiones{sufx}")
    lines.append("")
    conclusions = analysis.get("conclusiones", [])
    lines.append(("- " + "\n- ".join(str(c) for c in conclusions)) if conclusions
                 else "*(Sin conclusiones)*")
    lines.append("")

    if is_part:
        lines.append("## 📝 Transcripción de esta parte")
        lines.append("")
        lines.append(_format_running_text(part_text))
        lines.append("")
    else:
        lines.append("## 📄 Transcripción completa")
        lines.append("")
        lines.append("```text")
        lines.append(_format_running_text(part_text))
        lines.append("```")

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    tag = f"de la parte {part_idx} " if is_part else ""
    log(f"Markdown {tag}escrito -> {output_md}")


def save_transcript_checkpoint(transcript: dict, output_md: str) -> str:
    """Save the plain transcript next to the report so progress isn't lost.

    If the machine shuts down mid-run, the transcription is already on disk.
    """
    txt_path = transcript_checkpoint_path(output_md)
    text = _format_running_text(transcript.get("text", "") or "")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"Checkpoint transcripción -> {txt_path}")
    return txt_path


# ---------------------------------------------------------------------------
# Transcript mode: structured notes from a plain-text meeting transcript
# ---------------------------------------------------------------------------
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


def _format_running_text(text: str) -> str:
    """Hace legible un texto que llegó "corrido" en una sola línea larga.

    Respeta los párrafos existentes (punto aparte: salto de línea en blanco)
    y pone cada frase (punto seguido: '.', '!' o '?') en su propia línea.
    No rompe números decimales ('11.59'), ni toca textos que ya estén
    multilínea (p. ej. transcripciones con timestamps).
    """
    text = (text or "").strip()
    if not text:
        return ""
    paras = re.split(r"\n[ \t]*\n", text)
    blocks = []
    for para in paras:
        para = re.sub(r"[ \t]+", " ", para).strip()
        if not para:
            continue
        if "\n" in para or len(para) <= 160:
            blocks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        wrapped = []
        for s in sentences:
            if wrapped and re.match(r"^\d", s):
                wrapped[-1] += " " + s
            else:
                wrapped.append(s)
        blocks.append("\n".join(w for w in wrapped if w.strip()))
    return "\n\n".join(blocks)


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

    # Textos largos: se dividen automáticamente en partes de ~ANALYSIS_PART_CHARS
    # aunque no se pase --split-chars, para que cada parte tenga su propio .md
    # y el informe completo fusione todo (resumen ejecutivo único incluido).
    if split_chars <= 0 and len(text) > ANALYSIS_PART_CHARS:
        split_chars = ANALYSIS_PART_CHARS
        log(
            f"Texto largo ({len(text)} chars): división automática en partes "
            f"de ~{split_chars} (cada parte generará su propio .md).",
            "AUDITV",
        )

    parts = split_text(text, split_chars) if split_chars > 0 else [text]
    t_total = time.time()
    log(f"INICIO de apuntes desde: {transcript_path}", "AUDITV")
    _log_environment("cpu")
    status_reset(stage="preparando apuntes", output_md=output_md,
                 device="cpu", kind="apuntes")
    if len(parts) == 1:
        _out = _transcript_part_to_md(
            transcript_path, text, ollama_model, use_llm, output_md
        )
        log(
            f"✅ FIN de apuntes en {int(time.time() - t_total)} s. "
            f"Informe -> {_out}",
            "AUDITV",
        )
        status_set(running=False, stage="terminado")
        return _out

    log(f"Texto dividido en {len(parts)} partes de ~{split_chars} caracteres.")
    stem = Path(transcript_path).stem

    # Mismo motor que el flujo de video: divide, analiza cada parte y guarda
    # el .txt + .md de cada bache (out_md), devolviendo además lo necesario
    # para ensamblar el informe completo fusionado.
    _merged, parts_used, analyses = analyze_transcript_in_parts(
        text,
        ollama_model,
        include_timestamps=False,
        out_md=str(base_dir / stem),
        video_path=transcript_path,
        part_chars=split_chars,
        with_details=True,
        use_llm=use_llm,
    )

    outputs = [
        str(base_dir / f"{stem}_partes" / f"{stem}_parte{i}.md")
        for i in range(1, len(parts_used) + 1)
    ]

    # Informe completo: junta lo ya analizado (sin re-analizar el texto) +
    # transcripción completa. Todo ensamblado a partir de cada parte.
    combined_md = base_dir / f"{stem}_completo_apuntes.md"
    build_combined_transcript_markdown(
        stem, parts_used, analyses, ollama_model, text, str(combined_md)
    )
    outputs.append(str(combined_md))

    log(
        f"Conservados {len(parts_used)} baches (_parteN.txt + _parteN.md) en "
        f"la carpeta '{stem}_partes/'; informe completo -> {combined_md.name}.",
        "INFO",
    )
    log(
        f"✅ FIN de apuntes en {int(time.time() - t_total)} s. "
        f"Informe -> {combined_md}",
        "AUDITV",
    )
    status_set(running=False, stage="terminado")
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
    """Analyze one transcript part and write its .md (standalone report)."""
    analysis = _transcript_part_analysis(text, ollama_model, use_llm)
    build_part_markdown(
        source_path=transcript_path,
        part_idx=1,
        part_total=1,
        part_text=text,
        analysis=analysis,
        output_md=output_md,
        ollama_model=ollama_model,
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

    Joins the resumen/ideas/discusiones/conceptos/conclusiones already
    extracted per part, merges the partial summaries into a single executive
    summary (best-effort LLM pass) and appends the full transcript.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(parts)

    merged_resumen = ""
    if n > 1:
        merged_resumen = _merge_resumenes_with_llm(analyses, ollama_model)

    lines = []
    lines.append(f"# 📚 Reunión completa: {stem}")
    lines.append("")
    lines.append(f"- **Generado:** {now}")
    lines.append(f"- **Modelo LLM:** {ollama_model}")
    lines.append(f"- **Partes analizadas:** {n}")
    lines.append("")

    if merged_resumen:
        lines.append("## 📌 Resumen ejecutivo (fusionado)")
        lines.append("")
        lines.append(_format_running_text(merged_resumen))
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
        lines.append(_format_running_text(str(a.get("resumen") or "*(Sin resumen)*")))
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
    lines.append(_format_running_text(full_source))
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


def _pick_downloaded_file(output_dir: str) -> str:
    """Devuelve la ruta del medio recién descargado en `output_dir`.

    Se acepta cualquier formato: primero se priorizan video/audio conocidos y a
    continuación, si el título de la plataforma trae una extensión rara (p.ej.
    "reunion.mp4.mp4"), cualquier otro archivo que no sea un acompañante
    (miniatura, subtítulos, descarga incompleta .part).
    """
    out = Path(output_dir)
    if not out.is_dir():
        raise RuntimeError(f"Could not find download folder {output_dir}")

    def _mtime(p: Path) -> float:
        try:
            return os.path.getmtime(p)
        except OSError:
            return 0.0

    files = sorted(
        (f for f in out.glob("*") if f.is_file() and not f.name.startswith(".")),
        key=_mtime,
        reverse=True,
    )
    preferred = VIDEO_EXTS | AUDIO_EXTS
    for exts in (preferred, None):
        for f in files:
            suffix = f.suffix.lower()
            if exts is None:
                if suffix in SIDECAR_EXTS:
                    continue
            elif suffix not in exts:
                continue
            if f.stat().st_size > 0:
                log(f"Downloaded -> {f}")
                return str(f)
    raise RuntimeError(f"Could not find downloaded video in {output_dir}")


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
        *_tool_cmd("yt-dlp"),
        "--no-playlist",
        # mp4 si existe; si no, cualquier contenedor soportado + audio.
        "-f", ("bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo*+bestaudio/"
               "bestvideo+bestaudio/best"),
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

    # Find the downloaded file (yt-dlp may rename it).
    return _pick_downloaded_file(output_dir)


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
    t_total = time.time()

    if is_remote:
        os.makedirs(DEFAULT_DOWNLOAD_DIR, exist_ok=True)
        log(f"Descargando video desde URL: {video_path}", "AUDITV")
        video_path = download_video(
            video_path,
            DEFAULT_DOWNLOAD_DIR,
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
            live_from_start=live_from_start,
        )
        download_paths.append(video_path)
        log(
            f"Descarga completada en {int(time.time() - t_total)} s -> {video_path}",
            "AUDITV",
        )

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
    _log_environment(device)
    log(f"INICIO del análisis de: {video_path}", "AUDITV")
    log(f"Informe final -> {output_md}", "AUDITV")
    if frames_dir:
        log(f"Director de frames -> {frames_dir}", "AUDITV")
    status_reset(stage="preparando", output_md=output_md, device=device)

    with tempfile.TemporaryDirectory(prefix="video_to_md_") as tmpdir:
        # 1. Extract audio
        audio_path = os.path.join(tmpdir, "audio.wav")
        log("Paso 1/5: extrayendo audio (ffmpeg)...", "AUDITV")
        _t = time.time()
        extract_audio(video_path, audio_path)
        duration = media_duration(audio_path) or media_duration(video_path)
        log(
            f"Paso 1/5 completado ({int(time.time() - _t)} s, "
            f"{int(duration // 60)} min de audio).",
            "AUDITV",
        )
        status_update(stage="extrayendo audio", duration_s=duration)

        # 2. Modelo de Whisper: 'auto' elige el adecuado a este equipo y duración
        whisper_model = resolve_whisper_model(whisper_model, device, duration)

        # 2. Transcribe (+ checkpoint del texto por si el equipo se apaga)
        log(f"Paso 2/5: transcribiendo audio con Whisper ({whisper_model}/{device})...", "AUDITV")
        _t = time.time()
        transcript = transcribe_audio(
            audio_path, whisper_model, device,
            checkpoint_path=transcript_checkpoint_path(output_md),
        )
        save_transcript_checkpoint(transcript, output_md)
        log(
            f"Paso 2/5 completado ({int(time.time() - _t)} s): "
            f"{len(transcript.get('segments') or [])} segmentos.",
            "AUDITV",
        )

        # 3. Extract frames
        frames = []
        if not skip_frames:
            log(f"Paso 3/5: extrayendo frames (cada {frame_interval}s)...", "AUDITV")
            _t = time.time()
            frames = extract_video_frames(video_path, frames_dir, frame_interval)
            log(f"Paso 3/5 completado ({int(time.time() - _t)} s, {len(frames)} frames).", "AUDITV")
        status_update(stage="frames", frames=len(frames))

        # Modelo de Ollama: 'auto' elige el mejor de los que hay instalados.
        ollama_model = resolve_ollama_model(ollama_model)

        # 4. Informe parcial (transcripción + frames) para no perder progreso
        #    si el análisis LLM no termina (corte de luz, apagado, etc.).
        log("Paso 4/5: escribiendo informe parcial (transcripción + frames)...", "AUDITV")
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
            log(f"Paso 5/5: analizando con LLM ({ollama_model}) por partes...", "AUDITV")
            _t = time.time()
            try:
                analysis = analyze_transcript_in_parts(
                    transcript,
                    ollama_model,
                    include_timestamps=True,
                    out_md=output_md,
                    video_path=video_path,
                )
                log(f"Paso 5/5 completado ({int(time.time() - _t)} s).", "AUDITV")
            except Exception as exc:
                log(f"Ollama analysis skipped: {exc}", "WARN")
                analysis = {"resumen": "*(No se pudo analizar con Ollama)*"}
        else:
            log("Skipping Ollama analysis (--no-llm)")
            analysis = {"resumen": "*(Análisis LLM omitido)*"}

    # 6. Rebuild the markdown with the LLM analysis.
    log("Escribiendo informe final con el análisis LLM...", "AUDITV")
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
        llm_device="GPU" if ollama_use_gpu_default() and not GUARD.resting else "CPU",
    )

    # 7. Ask whether to keep or delete the downloaded/generated files.
    maybe_cleanup(
        autoclean=autoclean,
        download_paths=download_paths,
        output_md=output_md,
        frames_dir=frames_dir,
    )

    log(
        f"✅ FIN del análisis en {int(time.time() - t_total)} s. "
        f"Informe -> {output_md}",
        "AUDITV",
    )
    status_set(running=False, stage="terminado", output=output_md)
    return output_md


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    global OLLAMA_NUM_GPU, OLLAMA_GPU_INDEX
    global GPU_TEMP_WARN, GPU_TEMP_ABORT, GPU_TEMP_RESUME
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
                        help="Whisper: auto (default) elige el modelo adecuado a "
                             "tu GPU/CPU y a la duración del audio; o el tamaño "
                             f"exacto (tiny/base/small/medium/large/large-v3/"
                             f"large-v3-turbo). Default: {DEFAULT_WHISPER_MODEL}")
    parser.add_argument("--interval", "-i", type=float, default=DEFAULT_FRAME_INTERVAL,
                        help=f"Frame interval seconds (default: {DEFAULT_FRAME_INTERVAL})")
    parser.add_argument("--no-frames", action="store_true",
                        help="No extraer frames (solo transcripción + análisis)")
    parser.add_argument("--llm", default=DEFAULT_OLLAMA_MODEL,
                        help="Modelo de Ollama: auto (default) usa el mejor de "
                             "los que tengas instalado; o el nombre exacto "
                             f"(qwen3:8b, gemma3:4b, …). Default: {DEFAULT_OLLAMA_MODEL}")
    parser.add_argument("--ollama-gpu", choices=("auto", "gpu", "cpu"),
                        default=OLLAMA_NUM_GPU if OLLAMA_NUM_GPU in ("auto",) else
                        ("gpu" if ollama_use_gpu_default() else "cpu"),
                        help="Dónde corre el LLM: auto (default) = GPU si la "
                             "tarjeta es potente y no está caliente, si no CPU; "
                             "gpu = forzarla (con la protección por temperatura); "
                             "cpu = solo CPU")
    parser.add_argument("--ollama-gpu-index", default=OLLAMA_GPU_INDEX,
                        help="Índice de la GPU para el LLM (main_gpu de Ollama). "
                             "'auto' = la que elija Ollama")
    parser.add_argument("--gpu-temp-warn", type=int, default=GPU_TEMP_WARN,
                        help=f"Aviso de temperatura de la GPU (default: {GPU_TEMP_WARN}°C)")
    parser.add_argument("--gpu-temp-abort", type=int, default=GPU_TEMP_ABORT,
                        help=f"Por encima de esto la GPU descansa y el lote va a "
                             f"CPU (default: {GPU_TEMP_ABORT}°C)")
    parser.add_argument("--gpu-temp-resume", type=int, default=GPU_TEMP_RESUME,
                        help=f"Temperatura a la que la GPU vuelve al trabajo tras "
                             f"descansar (default: {GPU_TEMP_RESUME}°C)")
    parser.add_argument("--no-llm", action="store_true",
                         help="Skip Ollama analysis (transcription + frames only)")
    parser.add_argument("--split-chars", type=int, default=0,
                         help="Con --transcript: dividir el texto en partes de N "
                              "caracteres (~6000 recomendado) y generar un informe "
                              "por parte. 0 (default) = automático: si el texto es "
                              "largo (> %d chars) se divide igualmente para generar "
                              "los .md por parte y el informe completo fusionado"
                              % ANALYSIS_PART_CHARS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default=DEVICE_HINT,
                        help="Dispositivo para Whisper: auto (default), cpu "
                             "(más seguro en portátiles con refrigeración justa), "
                             "cuda (NVIDIA) o mps (GPU de Apple en macOS)")
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

    # Las opciones de GPU/temperatura mandan sobre el entorno.
    OLLAMA_NUM_GPU = {"auto": "auto", "gpu": "-1", "cpu": "0"}[args.ollama_gpu]
    if args.ollama_gpu_index is not None:
        OLLAMA_GPU_INDEX = args.ollama_gpu_index
    GPU_TEMP_WARN = args.gpu_temp_warn
    GPU_TEMP_ABORT = args.gpu_temp_abort
    GPU_TEMP_RESUME = args.gpu_temp_resume

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

