#!/usr/bin/env python3
"""Detección agnóstica del hardware y de los modelos instalados.

AuditV no debe atarse a una máquina concreta. Cada usuario puede tener otras
GPUs (o ninguna), otros modelos descargados en Ollama y otros modelos de
Whisper en la caché. Este módulo PREGUNTA al sistema qué hay disponible y lo
convierte en decisiones:

* Qué GPU hay, cuánta VRAM tiene y si es potente (o si la que tiene es de las
  débiles que se recalientan, como una GTX 1060 Mobile).
* Qué modelos hay instalados en Ollama y cuál conviene usar (el mejor que
  cabe en la VRAM, o ~4B en CPU), priorizando familias que siguen bien JSON.
* Qué modelo de Whisper usar según el dispositivo, la VRAM y la duración del
  audio, informative de lo que ya está descargado.

Ninguna función falla si no hay GPU/Ollama/torch: devuelven listas vacías o
`None` y quien llama decide con su valor por defecto.
"""

import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# --- Caché corta de nvidia-smi ---------------------------------------------
# La GUI y el monitor térmico preguntan a la vez; sin cache cada pregunta sería
# un proceso nuevo (nvidia-smi tarda ~100 ms).
_GPU_CACHE = {"ts": 0.0, "gpus": []}
_GPU_CACHE_LOCK = threading.Lock()
_GPU_CACHE_TTL = 1.0  # segundos
# Los procesos en la GPU se cachean aparte (otra llamada a nvidia-smi).
_GPU_PROC_CACHE = {"ts": 0.0, "procs": None}
_GPU_PROC_CACHE_LOCK = threading.Lock()

_NVIDIA_FIELDS = (
    "index,name,memory.total,memory.used,temperature.gpu,"
    "utilization.gpu,compute_cap"
)

# VRAM (MiB) y compute capability mínimos para cada nivel. El orden importa:
# se evalúa de más a menos potente.
TIER_TABLE = (
    ("muy_potente", 16384, (8, 0)),
    ("potente", 8192, (7, 5)),
    ("media", 4096, (7, 0)),
)
TIER_FALLBACK = "débil"
TIER_NONE = "sin_gpu"
TIER_ORDER = [TIER_NONE, TIER_FALLBACK, "media", "potente", "muy_potente"]
TIER_LABEL = {
    TIER_NONE: "sin GPU",
    TIER_FALLBACK: "débil",
    "media": "media",
    "potente": "potente",
    "muy_potente": "muy potente",
}

# Familias de LLM ordenadas por lo bien que siguen instrucciones/JSON en
# local. Un número menor = mejor opción (desempate cuando el tamaño empata).
FAMILY_RANK = {
    "qwen35": 0, "qwen3": 1, "qwen25": 2, "qwen2": 3,
    "gemma3": 4, "gemma2": 5, "gemma": 6,
    "llama33": 7, "llama32": 8, "llama31": 9, "llama3": 10, "llama2": 11,
    "mistral": 12, "ministral": 13, "mixtral": 14, "magistral": 15,
    "devstral": 16, "phi4": 17, "phi3": 18,
    "commandr": 19, "granite": 20, "olmo": 21, "olmo2": 22,
    "internlm": 23, "deepseek": 24, "firefunction": 25, "smollm": 26,
}
FAMILY_RANK_DEFAULT = 60

# Tamaño ideal (GB en disco) según dónde se ejecute el LLM.
IDEAL_SIZE_GB_GPU = 8.0
IDEAL_SIZE_GB_CPU = 4.0
# Presupuesto de VRAM/ RAM que se deja libre para el contexto y el sistema.
VRAM_BUDGET_RATIO = 0.65
CPU_BUDGET_GB = 4.5

# Memoria aproximada que necesita cada modelo de Whisper en GPU (GB), de menor
# a mayor. El tamaño del .pt es fp32; se deja margen para los activaciones.
WHISPER_VRAM_GB = (
    ("tiny", 1.0),
    ("base", 1.5),
    ("small", 2.5),
    ("medium", 5.0),
    ("large-v3-turbo", 6.0),
    ("large-v3", 10.0),
)
WHISPER_CHOICES = ("auto", "tiny", "base", "small", "medium", "large",
                   "large-v3", "large-v3-turbo")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _run(cmd, timeout: float = 5.0):
    """subprocess.run que nunca lanza (devuelve None si falla).

    `encoding` explícito porque en Windows el texto se decodifica con la
    codificación de la consola (cp1252/cp850) y un nombre de tarjeta o de
    archivo con acento o emoji lanza UnicodeDecodeError.
    """
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
    except Exception:
        return None


def fmt_vram(mb) -> str:
    """6144 -> '6.0 GB'."""
    if not mb:
        return "?"
    return f"{mb / 1024:.1f} GB".replace(".0 ", " ")


def parse_compute_cap(value):
    """'6.1' -> (6, 1). None si no hay dato."""
    if not value:
        return None
    try:
        major, minor = str(value).split(".")[:2]
        return (int(major), int(minor))
    except Exception:
        return None


def cap_str(cap) -> str:
    return f"{cap[0]}.{cap[1]}" if cap else "?"


# ---------------------------------------------------------------------------
# GPUs (nvidia-smi)
# ---------------------------------------------------------------------------
def gpu_infos(max_age: float = _GPU_CACHE_TTL) -> list:
    """GPUs NVIDIA detectadas por nvidia-smi ([] si no hay o falla).

    Cada elemento: {index, name, vram_total_mb, vram_used_mb, temp_c,
    util_pct, compute_cap}. `max_age=0` fuerza una lectura fresca.
    """
    with _GPU_CACHE_LOCK:
        if _GPU_CACHE["gpus"] and time.time() - _GPU_CACHE["ts"] < max_age:
            return _GPU_CACHE["gpus"]

    gpus = []
    out = _run(["nvidia-smi", f"--query-gpu={_NVIDIA_FIELDS}",
                "--format=csv,noheader,nounits"])
    if out is not None and out.returncode == 0:
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            # Un nombre con comas descuadraría el CSV: se recolocan.
            if len(parts) > 7:
                extra = len(parts) - 7
                parts = [parts[0], ",".join(parts[1:1 + extra])] + parts[1 + extra:]
            if len(parts) < 7:
                continue
            gpus.append({
                "index": int(parts[0]) if parts[0].isdigit() else len(gpus),
                "name": parts[1] or "GPU",
                "vram_total_mb": int(float(parts[2])) if _is_num(parts[2]) else 0,
                "vram_used_mb": int(float(parts[3])) if _is_num(parts[3]) else 0,
                "temp_c": float(parts[4]) if _is_num(parts[4]) else None,
                "util_pct": int(float(parts[5])) if _is_num(parts[5]) else None,
                "compute_cap": parts[6] if parts[6] not in ("", "[N/A]") else None,
            })

    with _GPU_CACHE_LOCK:
        _GPU_CACHE["ts"] = time.time()
        _GPU_CACHE["gpus"] = gpus
    return gpus


def _is_num(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def gpu_temperature(index: int = 0, max_age: float = _GPU_CACHE_TTL):
    """Temperatura actual de la GPU en °C, o None si no se puede leer."""
    gpus = gpu_infos(max_age=max_age)
    if not gpus:
        return None
    for gpu in gpus:
        if gpu["index"] == index:
            return gpu["temp_c"]
    return gpus[0]["temp_c"]


def gpu_free_mb(gpu: dict) -> int:
    if not gpu:
        return 0
    return max(0, (gpu.get("vram_total_mb") or 0) - (gpu.get("vram_used_mb") or 0))


def gpu_compute_processes(max_age: float = _GPU_CACHE_TTL) -> list:
    """Procesos que están usando la GPU ahora mismo.

    [{pid, name, vram_mb}] — lista vacía si nvidia-smi no lo soporta o no hay
    nada en la GPU. Sirve para atribuir la VRAM a Whisper o a la IA local.
    """
    with _GPU_PROC_CACHE_LOCK:
        if (time.time() - _GPU_PROC_CACHE["ts"] < max_age
                and _GPU_PROC_CACHE["procs"] is not None):
            return _GPU_PROC_CACHE["procs"]

    out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"])
    procs = []
    if out is not None and out.returncode == 0:
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            procs.append({
                "pid": int(parts[0]),
                "name": parts[1] or "?",
                "vram_mb": int(float(parts[2])) if _is_num(parts[2]) else 0,
            })

    with _GPU_PROC_CACHE_LOCK:
        _GPU_PROC_CACHE["ts"] = time.time()
        _GPU_PROC_CACHE["procs"] = procs
    return procs


def gpu_vram_of_pid(pid, max_age: float = _GPU_CACHE_TTL):
    """MB de VRAM que usa un PID concreto, o 0 si no está en la GPU."""
    if not pid:
        return 0
    for p in gpu_compute_processes(max_age=max_age):
        if p["pid"] == int(pid):
            return p["vram_mb"]
    return 0


def primary_gpu(gpus: list = None) -> dict:
    """La GPU con más VRAM (empate: la de índice más bajo), o {} si no hay."""
    gpus = gpus if gpus is not None else gpu_infos()
    if not gpus:
        return {}
    return max(gpus, key=lambda g: (g.get("vram_total_mb") or 0, -g["index"]))


def gpu_tier(gpu: dict) -> str:
    """Nivel de potencia: sin_gpu / débil / media / potente / muy_potente.

    Una GTX 1060 (6 GB, sm_61) cae en 'débil': aguanta el LLM en GPU pero
    recalienta el equipo y ni siquiera la soporta este PyTorch.
    """
    if not gpu:
        return TIER_NONE
    vram = gpu.get("vram_total_mb") or 0
    cap = parse_compute_cap(gpu.get("compute_cap"))
    for name, min_vram, min_cap in TIER_TABLE:
        if vram >= min_vram and cap is not None and cap >= min_cap:
            return name
    return TIER_FALLBACK


def tier_rank(tier: str) -> int:
    try:
        return TIER_ORDER.index(tier)
    except ValueError:
        return 0


def describe_gpu(gpu: dict) -> str:
    """'NVIDIA GeForce RTX 3060 (12 GB, sm_86, nivel potente)'."""
    if not gpu:
        return "sin GPU NVIDIA detectable"
    cap = parse_compute_cap(gpu.get("compute_cap"))
    return (
        f"{gpu.get('name', 'GPU')} ({fmt_vram(gpu.get('vram_total_mb'))}, "
        f"sm_{cap_str(cap).replace('.', '')}, "
        f"nivel {TIER_LABEL.get(gpu_tier(gpu), '?')})"
    )


def ollama_gpu_default(tier: str = None):
    """¿Debe Ollama usar la GPU por defecto? -> (bool, motivo).

    Con una GPU decente (media o mejor) sí: es lo que hace que el análisis no
    se arrastre. Con GPU débil o sin GPU, no (más lento, pero no recalienta ni
    swappea). La protección por temperatura sigue mandando en ambos casos.
    """
    tier = tier or gpu_tier(primary_gpu())
    if tier in ("media", "potente", "muy_potente"):
        return True, (f"GPU {TIER_LABEL[tier]}: el LLM irá a la GPU "
                      "(si la temperatura lo permite)")
    return False, (f"GPU {TIER_LABEL[tier]}: el LLM irá en CPU para no "
                   "recalentar el equipo")


# ---------------------------------------------------------------------------
# Ollama: modelos instalados
# ---------------------------------------------------------------------------
def ollama_models(timeout: float = 3.0) -> list:
    """Modelos instalados en el servidor Ollama local ([] si no responde).

    Cada elemento: {name, size_bytes, params, family, quant, capabilities}.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    models = []
    for raw in data.get("models") or []:
        det = raw.get("details") or {}
        name = raw.get("name") or raw.get("model") or ""
        if not name:
            continue
        models.append({
            "name": name,
            "size_bytes": int(raw.get("size") or 0),
            "params": det.get("parameter_size") or "",
            "family": det.get("family") or "",
            "quant": det.get("quantization_level") or "",
            "capabilities": raw.get("capabilities") or [],
        })
    return models


def ollama_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2):
            return True
    except Exception:
        return False


def model_size_gb(model: dict) -> float:
    """Tamaño en GB del archivo del modelo (a partir del tamaño o de '4.7B')."""
    size = (model.get("size_bytes") or 0) / 1e9
    if size > 0.01:
        return size
    params = str(model.get("params") or "")
    m = re.match(r"\s*([\d.]+)\s*([BbMm])", params)
    if not m:
        return 0.0
    value = float(m.group(1))
    return value if m.group(2).lower() == "b" else value / 1000.0


def _family_key(model: dict) -> str:
    family = (model.get("family") or "").lower()
    name = (model.get("name") or "").lower()
    for key in FAMILY_RANK:
        if family == key or family.startswith(key):
            return key
    # Sin metadatos: deducir de la etiqueta ("qwen3.5:4b-instruct").
    tag = name.split(":")[0]
    for key in FAMILY_RANK:
        if tag.startswith(key):
            return key
    return ""


def ollama_budget_gb(use_gpu: bool = None, vram_mb: int = None) -> float:
    """GB de modelo que se pueden permitir sin swapear ni recalentar."""
    if use_gpu is None:
        use_gpu, _ = ollama_gpu_default()
    if use_gpu:
        gpu = primary_gpu()
        vram = vram_mb if vram_mb is not None else (gpu.get("vram_total_mb") or 0)
        if vram:
            return (vram / 1024.0) * VRAM_BUDGET_RATIO
    return CPU_BUDGET_GB


def _model_sort_key(model: dict, budget_gb: float):
    """Ordena de mejor a peor: primero que quepa, luego tamaño, luego familia."""
    size = model_size_gb(model)
    family = FAMILY_RANK.get(_family_key(model), FAMILY_RANK_DEFAULT)
    caps = model.get("capabilities") or []
    cap_penalty = 0 if (not caps or "completion" in caps) else 1
    if size > budget_gb:
        # No cabe: se penaliza por cuánto se pasa (lo que menos se pasa, mejor).
        return (1, round(size - budget_gb, 2), 0, 0.0, family, model["name"])
    ideal = IDEAL_SIZE_GB_GPU if budget_gb > CPU_BUDGET_GB else IDEAL_SIZE_GB_CPU
    # Preferencia monótona: cuanto más grande, mejor, hasta 'ideal' (a partir de
    # ahí se considera que el modelo ya es más que suficiente).
    fit = min(1.0, size / max(ideal, 0.1))
    return (0, 0.0, cap_penalty, -round(fit, 3), family, model["name"])


def pick_ollama_model(models: list = None, use_gpu: bool = None,
                      vram_mb: int = None):
    """Mejor modelo instalado: el mayor que cabe cómodo, con buen JSON.

    En GPU se permite el ~65% de la VRAM y se idealiza ~8B; en CPU el
    presupuesto es de 4.5 GB y se idealizan ~4B (más rápido y no swappea).
    Devuelve (nombre, modelo_dict) o (None, None) si Ollama no responde o no
    hay modelos instalados.
    """
    if models is None:
        models = ollama_models()
    if not models:
        return None, None
    budget = ollama_budget_gb(use_gpu=use_gpu, vram_mb=vram_mb)
    best = sorted(models, key=lambda m: _model_sort_key(m, budget))[0]
    return best["name"], best


def ollama_model_report() -> dict:
    """Resumen para el log y para el panel de la GUI."""
    models = ollama_models()
    auto, best = pick_ollama_model(models)
    return {
        "running": bool(models) or ollama_running(),
        "count": len(models),
        "names": [m["name"] for m in models],
        "auto": auto,
        "best": best,
    }


# ---------------------------------------------------------------------------
# Whisper: modelos disponibles y recomendados
# ---------------------------------------------------------------------------
def whisper_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "whisper"


def whisper_cached_models() -> list:
    """Modelos de Whisper ya descargados (['base', 'small', ...], de menor a mayor)."""
    order = [name for name, _ in WHISPER_VRAM_GB]
    found = set()
    try:
        for path in whisper_cache_dir().glob("*.pt"):
            found.add(path.stem)
    except Exception:
        pass
    return sorted(found, key=lambda n: order.index(n) if n in order else 99)


def recommend_whisper_model(device: str = "cpu", vram_mb: int = None,
                            duration_s: float = None) -> str:
    """Modelo de Whisper adecuado al equipo y a la duración del audio.

    GPU: el mayor que quepa en ~75% de la VRAM. MPS (Mac Apple Silicon): la
    memoria es compartida con la CPU y se estima por los GB de RAM. CPU:
    `small` para audios cortos, `base` para medias y `tiny` para horas (en CPU
    la diferencia de tiempo es enorme).
    """
    if str(device).startswith("cuda"):
        vram = vram_mb
        if vram is None:
            vram = primary_gpu().get("vram_total_mb") or 0
        budget = max(1.0, (vram / 1024.0) * 0.75)
        for name, need in reversed(WHISPER_VRAM_GB):
            if need <= budget:
                return name
        return "tiny"
    if str(device).startswith("mps"):
        # Memoria unificada con la CPU: se es conservador (25% de la RAM) y no
        # se pasa de `medium`, porque en MPS los modelos grandes van tan solos
        # que acabaría paginandose con swap. Para audio largo, mejor en CPU.
        budget = max(1.0, _system_ram_gb() * 0.25)
        for name, need in reversed(WHISPER_VRAM_GB):
            if need <= budget and name in ("tiny", "base", "small", "medium"):
                return name
        return "small"
    mins = (duration_s or 0) / 60.0
    if not duration_s:
        return "small"
    if mins <= 15:
        return "small"
    if mins <= 45:
        return "base"
    return "tiny"


def _system_ram_gb() -> float:
    """RAM del sistema en GB (0 si no se puede saber)."""
    if psutil is not None:
        try:
            return psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass
    try:  # /proc/meminfo en Linux
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def torch_mps_info() -> dict:
    """GPU de Apple (Metal Performance Shaders) vista por PyTorch.

    En macOS no hay nvidia-smi ni CUDA: sin esto, un Mac con chip Apple
    Silicon mandaría Whisper a CPU, que es 5-10x más lento.
    """
    info = {"available": False, "usable": False, "name": None,
            "vram_mb": 0, "reason": ""}
    try:
        import torch
    except Exception as exc:
        info["reason"] = f"torch no instalado ({exc.__class__.__name__})"
        return info
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            info["reason"] = "MPS no disponible en este equipo"
            return info
        info["available"] = True
        info["name"] = "Apple GPU (Metal)"
        info["usable"] = bool(mps.is_built() and mps.is_available())
        # Memoria unificada: se usa la RAM del sistema como aproximación.
        info["vram_mb"] = int(_system_ram_gb() * 1024)
        if not info["usable"]:
            info["reason"] = "MPS compilado pero no disponible"
    except Exception as exc:
        info["reason"] = f"no se pudo consultar MPS ({exc})"
    return info


def torch_gpu_info() -> dict:
    """Qué GPU puede usar PyTorch en este equipo: CUDA si la hay, si no MPS.

    Devuelve el mismo formato que `torch_cuda_info`, con `kind` a modo de
    'cuda' o 'mps', para no tener dos caminos en el código que llama.
    """
    cuda = torch_cuda_info()
    if cuda.get("usable"):
        return dict(cuda, kind="cuda")
    mps = torch_mps_info()
    if mps.get("usable"):
        return dict(mps, kind="mps")
    return dict(cuda if cuda.get("available") else mps, kind="cpu")


def whisper_report(device: str = "cpu", vram_mb: int = None,
                   duration_s: float = None) -> dict:
    """Resumen del 'equipo de transcripción' para el log y la GUI."""
    cached = whisper_cached_models()
    auto = recommend_whisper_model(device, vram_mb, duration_s)
    return {
        "device": device,
        "auto": auto,
        "cached": cached,
        "cached_hit": auto in cached,
    }


# ---------------------------------------------------------------------------
# Resumen general
# ---------------------------------------------------------------------------
def hardware_summary() -> dict:
    """Todo lo relevante del equipo, en un dict serializable."""
    gpus = gpu_infos()
    gpu = primary_gpu(gpus)
    tier = gpu_tier(gpu)
    use_gpu, reason = ollama_gpu_default(tier)
    torch_cuda = torch_cuda_info()
    return {
        "gpus": gpus,
        "gpu": gpu,
        "gpu_count": len(gpus),
        "tier": tier,
        "tier_label": TIER_LABEL.get(tier, tier),
        "ollama_use_gpu": use_gpu,
        "ollama_reason": reason,
        "torch_cuda": torch_cuda,
        "whisper_device": "cuda" if torch_cuda.get("usable") else "cpu",
        "whisper_auto": recommend_whisper_model(
            "cuda" if torch_cuda.get("usable") else "cpu",
            torch_cuda.get("vram_mb"),
        ),
    }


def torch_cuda_info() -> dict:
    """Qué ve PyTorch: usable o no, nombre, compute capability y VRAM."""
    info = {"available": False, "usable": False, "name": None,
            "cap": None, "vram_mb": 0, "reason": ""}
    try:
        import torch
    except Exception as exc:
        info["reason"] = f"torch no instalado ({exc.__class__.__name__})"
        return info
    try:
        info["available"] = bool(torch.cuda.is_available())
    except Exception:
        return info
    if not info["available"]:
        info["reason"] = "sin CUDA en este equipo"
        return info
    try:
        info["name"] = torch.cuda.get_device_name(0)
        info["cap"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        info["vram_mb"] = int(torch.cuda.get_device_properties(0).total_memory
                               // (1024 * 1024))
        arch = torch.cuda.get_arch_list()
        sm = f"sm_{info['cap'].replace('.', '')}"
        info["usable"] = bool(arch) and any(
            sm in a or a.startswith("compute_") for a in arch
        )
        if not info["usable"]:
            info["reason"] = (
                f"la GPU es {sm} y este torch solo soporta {arch or '[]'}"
            )
    except Exception as exc:
        info["reason"] = f"no se pudo consultar la GPU ({exc})"
    return info


def log_lines() -> list:
    """Líneas para el log al arrancar: qué equipo ha encontrado."""
    gpus = gpu_infos()
    lines = []
    if gpus:
        for gpu in gpus:
            lines.append(
                f"GPU {gpu['index']}: {gpu['name']} "
                f"({fmt_vram(gpu['vram_total_mb'])}, sm_{cap_str(parse_compute_cap(gpu.get('compute_cap')))}, "
                f"{TIER_LABEL[gpu_tier(gpu)]})"
            )
    else:
        lines.append("No se detecta GPU NVIDIA (nvidia-smi no disponible).")
    tc = torch_cuda_info()
    if tc.get("available"):
        lines.append(
            f"PyTorch {'' if tc.get('usable') else 'NO puede'} usar la GPU: "
            f"{tc.get('name')} (sm_{str(tc.get('cap')).replace('.', '')})"
            + (f" — {tc['reason']}" if tc.get("reason") else "")
        )
    return lines


# ---------------------------------------------------------------------------
# Consumo de CPU/RAM del equipo y de cada componente (Whisper, IA local)
# ---------------------------------------------------------------------------
# El panel de la GUI se refresca cada 2 s: estas lecturas se cachean un par de
# segundos para no ir a /proc ni a nvidia-smi en cada tick.
_RES_CACHE = {"ts": 0.0, "data": None}
_RES_LOCK = threading.Lock()
_RES_TTL = 1.5  # segundos

try:  # psutil es opcional: sin él el panel funciona igual, sin cifras
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _mb(num_bytes) -> float:
    return round((num_bytes or 0) / (1024 * 1024), 1)


def fmt_mem(mb) -> str:
    """Bytes en MB o GB según el tamaño: 512.0 -> '512 MB', 6144 -> '6.0 GB'.

    Para la RAM del equipo, donde los procesos usan fracciones de GB y siempre
    se muestran en MB (a diferencia de `fmt_vram`, pensada para VRAM).
    """
    if not mb:
        return "—"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB".replace(".0 ", " ")
    return f"{mb:.0f} MB"


def fmt_pct(value) -> str:
    """Porcentaje redondeado, o «—» si no se pudo medir."""
    if value is None:
        return "—"
    return f"{value:.0f} %"


def _find_pids(names) -> list:
    """PIDs cuyo nombre de proceso empieza por alguno de `names`.

    Se busca con psutil (funciona igual en Windows, macOS y Linux) y, si no
    está, se recurre a /proc, que solo existe en Linux.
    """
    if psutil is not None:
        found = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                comm = (proc.info.get("name") or "")
            except Exception:
                continue
            if any(comm == n or comm.startswith(n) for n in names):
                found.append(proc.info["pid"])
        return found

    found = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return found  # sin /proc (Windows/macOS) y sin psutil: no hay datos
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
        except OSError:
            continue
        if any(comm == n or comm.startswith(n) for n in names):
            found.append(int(entry.name))
    return found


def _snapshot_uncached(cli_pid=None) -> dict:
    """Mide CPU/RAM del equipo, del proceso de transcripción y del de Ollama."""
    snap = {
        "cpu_pct": None, "cpu_cores": os.cpu_count(),
        "load1": None,
        "ram_total_mb": None, "ram_used_mb": None, "ram_avail_mb": None,
        "ram_pct": None, "swap_used_mb": None,
        "whisper": None, "ollama": None,
    }
    if psutil is None:
        return snap

    try:
        snap["cpu_pct"] = float(psutil.cpu_percent(interval=None))
        snap["cpu_cores"] = psutil.cpu_count() or snap["cpu_cores"]
    except Exception:
        pass
    try:
        snap["load1"] = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        pass
    try:
        vm = psutil.virtual_memory()
        snap["ram_total_mb"] = _mb(vm.total)
        snap["ram_used_mb"] = _mb(getattr(vm, "used", None) or (vm.total - vm.available))
        snap["ram_avail_mb"] = _mb(vm.available)
        snap["ram_pct"] = float(vm.percent)
        sw = psutil.swap_memory()
        snap["swap_used_mb"] = _mb(getattr(sw, "used", 0))
    except Exception:
        pass

    # Whisper: el proceso del CLI (la GUI escribe su pid en el estado).
    if cli_pid:
        try:
            proc = psutil.Process(int(cli_pid))
            with proc.oneshot():
                mem = proc.memory_info()
                snap["whisper"] = {
                    "pid": int(cli_pid),
                    "rss_mb": _mb(mem.rss),
                    "vms_mb": _mb(mem.vms),
                    "threads": proc.num_threads(),
                    "cpu_pct": _proc_cpu_pct(proc),
                }
        except Exception:
            snap["whisper"] = None

    # IA local: el/los procesos de Ollama.
    pids = _ollama_pids()
    if pids:
        rss = vms = 0.0
        cpu = 0.0
        threads = 0
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                with proc.oneshot():
                    mem = proc.memory_info()
                    rss += mem.rss
                    vms += mem.vms
                    threads += proc.num_threads()
                    cpu += _proc_cpu_pct(proc)
            except Exception:
                continue
        snap["ollama"] = {"pid": pids[0], "pids": pids, "rss_mb": _mb(rss),
                          "vms_mb": _mb(vms), "threads": threads,
                          "cpu_pct": round(cpu, 1)}
    return snap


_PROC_CPU = {"last": {}}


def _proc_cpu_pct(proc) -> float:
    """% de CPU de un proceso respecto a UN núcleo (puede pasar de 100)."""
    pid = proc.pid
    try:
        with proc.oneshot():
            times = proc.cpu_times()
    except Exception:
        return 0.0
    now = time.time()
    cpu_now = (times.user or 0.0) + (times.system or 0.0)
    prev = _PROC_CPU["last"].get(pid)
    _PROC_CPU["last"][pid] = (now, cpu_now)
    if len(_PROC_CPU["last"]) > 64:  # poda de PIDs ya terminados
        _PROC_CPU["last"] = {p: v for p, v in _PROC_CPU["last"].items()
                             if now - v[0] < 60}
    if not prev or now - prev[0] < 0.2:
        return 0.0
    delta_cpu = cpu_now - prev[1]
    delta_t = now - prev[0]
    if delta_t <= 0:
        return 0.0
    return max(0.0, delta_cpu / delta_t * 100.0)


def _ollama_pids() -> list:
    """PIDs de los procesos de Ollama (varios: servidor + runner)."""
    return _find_pids(["ollama"])


def ollama_running_models() -> list:
    """Modelos cargados ahora mismo en Ollama (`/api/ps`).

    [{name, size_vram_mb, vram_pct}] — lista vacía si Ollama no está en marcha.
    """
    out = []
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return out
    for m in (data or {}).get("models") or []:
        size_vram = m.get("size_vram") or 0
        out.append({
            "name": m.get("name") or m.get("model") or "?",
            "size_vram_mb": int(size_vram // (1024 * 1024)) if size_vram else 0,
            "vram_pct": int(m.get("size_vram_percent") or 0),
        })
    return out


def resource_snapshot(cli_pid=None, max_age: float = _RES_TTL) -> dict:
    """Consumo actual de CPU/RAM del equipo y de cada componente.

    `cli_pid` es el proceso del CLI (transcripción con Whisper). Se cachea
    `max_age` segundos porque el panel de la GUI pregunta cada 2 s.
    """
    with _RES_LOCK:
        if (_RES_CACHE["data"] is not None
                and time.time() - _RES_CACHE["ts"] < max_age):
            return _RES_CACHE["data"]
    data = _snapshot_uncached(cli_pid)
    with _RES_LOCK:
        _RES_CACHE["ts"] = time.time()
        _RES_CACHE["data"] = data
    return data
