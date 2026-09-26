#!/usr/bin/env python3
"""AuditV GUI (Gradio).

Interfaz web local para el pipeline completo del proyecto:

  pestaña "Video / URL"  -> analizar un video local o un enlace (descarga),
                            con elección de carpeta de salida, modelo,
                            intervalo, cookies y autoclean.
  pestaña "Apuntes"       -> generar el informe (ideas, discusiones,
                            conclusiones) desde una transcripción .txt.
  pestaña "Reunión en vivo" -> capturar/transcribir una reunión en tiempo
                            real (audio del sistema o micrófono) y generar
                            los apuntes al cerrar.

En todas las pestañas hay un panel de equipo arriba: qué GPU ha encontrado,
su temperatura y si la está usando ahora mismo. Los modelos, el dispositivo y
la GPU del LLM se detectan solos (ver hardware.py), así que nada queda atado
a una máquina concreta.

Ejecutar:
    ./venv/bin/python tools/auditv_gui.py
Se abre en el navegador en http://127.0.0.1:7860
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

_PROJECT = Path(__file__).resolve().parent.parent


def _venv_python() -> str:
    """Intérprete del venv del proyecto, según el sistema.

    En Windows el venv usa `venv\\Scripts\\python.exe`, no `venv/bin/python`.
    Si no existe (alguien lo lanzó con su propio Python), se usa el
    intérprete actual, que es lo que hay.
    """
    if os.name == "nt":
        candidates = [_PROJECT / "venv" / "Scripts" / "python.exe",
                      _PROJECT / "venv" / "bin" / "python"]
    else:
        candidates = [_PROJECT / "venv" / "bin" / "python",
                      _PROJECT / "venv" / "Scripts" / "python.exe"]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


_PY = _venv_python()
_CLI = str(_PROJECT / "video_to_md.py")
_LIVE_SCRIPT = str(_PROJECT / "tools" / "live_meeting.py")

# El módulo `hardware` vive en la raíz del proyecto: se importa para no
# duplicar (ni volver a endurecer) la detección de GPU/modelos.
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
import hardware  # noqa: E402

_DEFAULT_OUTDIR = str(_PROJECT / "informes")
# Fichero de estado que escribe el CLI y lee este panel (ver video_to_md.py).
_STATUS_FILE = str(_PROJECT / ".auditv_status.json")

_LIVE = {"proc": None, "txt": None}

# Procesos CLI actualmente en ejecución desde la GUI (para poder cancelarlos).
_RUNNER = {}

# Mantiene el log pegado al fondo mientras el usuario no lo impida. El
# autoscroll se detiene al primer gesto de SUBIDA (rueda/teclado) de forma
# síncrona, para que nunca pelee contra el scroll manual del usuario; solo se
# reengancha cuando vuelve a llegar abajo del todo.
def _autoscroll_js(n_inputs: int) -> str:
    """JS de autoscroll no invasivo; devuelve intactos los inputs.

    Gradio llama a esta función con los valores de inputs y outputs como
    argumentos y reparte SU RETORNO como keyword args del `fn` de Python. Por
    eso debe aceptarlos todos (`...args`) y devolver SOLO los `n_inputs`
    primeros, en orden; si devuelve un único valor (o todos), Gradio mapea
    mal los parámetros (p. ej. "Parameter `path` is not a valid keyword
    argument").
    """
    return f"""(...args) => {{
  (function () {{
    try {{
      var ids = ['av_log_video', 'av_log_notes', 'av_log_live'];
      var stopped = false;
      var near = function () {{
        var b = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
        return (b - window.innerHeight - window.scrollY) < 40;
      }};
      window.addEventListener('wheel', function (e) {{
        if (e.deltaY < 0) stopped = true;
        else if (e.deltaY > 0 && near()) stopped = false;
      }}, {{passive: true, capture: true}});
      window.addEventListener('keydown', function (e) {{
        if (['ArrowUp', 'PageUp', 'Home'].indexOf(e.key) >= 0) stopped = true;
        else if (['ArrowDown', 'PageDown', 'End', ' '].indexOf(e.key) >= 0 && near()) stopped = false;
      }}, true);
      function bottom() {{
        return Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
      }}
      function elOf(id) {{ return document.querySelector('#' + id + ' textarea'); }}
      function follow() {{
        window.scrollTo(0, bottom());
        ids.forEach(function (id) {{ var el = elOf(id); if (el) el.scrollTop = el.scrollHeight; }});
      }}
      (function tick() {{
        requestAnimationFrame(tick);
        if (!stopped) follow();
      }})();
    }} catch (e) {{}}
  }})();
  return args.slice(0, {n_inputs});
}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _terminate(proc) -> None:
    """Termina el proceso y su grupo (ffmpeg/yt-dlp incluidos).

    `os.killpg`/`os.getpgid` solo existen en POSIX; en Windows se cae a
    `terminate()`/`kill()`, que no llegan a los hijos de ffmpeg.
    """
    if proc is None or proc.poll() is not None:
        return
    pgid = None
    if hasattr(os, "getpgid"):  # POSIX
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pgid = None  # grupo ya no existe: se mata solo el proceso
    if pgid is None:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                proc.kill()
        else:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _request_stop(proc) -> bool:
    """Pide que el proceso termine solo (Ctrl-C / CTRL_BREAK).

    En Windows, `send_signal(SIGINT)` no existe para hijos: se usa
    CTRL_BREAK_EVENT, que solo funciona si el proceso nació en su propio grupo
    (ver `_new_process_kwargs`). Devuelve False si no se pudo pedir.
    """
    if proc is None or proc.poll() is not None:
        return False
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        return True
    except (OSError, ValueError, AttributeError):
        return False


def on_cancel(key: str) -> str:
    """Cancela el proceso en curso del tipo 'key' (video, notes, live)."""
    proc = _RUNNER.get(key)
    alive = proc is not None and proc.poll() is None
    _terminate(proc)
    if _RUNNER.get(key) is proc:
        _RUNNER[key] = None
    return ("⏹ Proceso cancelado: ya no queda en segundo plano." if alive
            else "No hay proceso de ese tipo en ejecución.")


def _gpu_mode(value) -> str:
    """Normaliza el selector de "dónde corre el LLM" a auto | gpu | cpu.

    Acepta también un checkbox booleano (compatibilidad) y, si el valor es
    None, cae en 'auto' (decide el CLI según la potencia de la GPU).
    """
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "gpu" if value else "cpu"
    mode = str(value).strip().lower()
    return mode if mode in ("auto", "gpu", "cpu") else "auto"


def _gpu_index(gpu_idx: str, mode: str) -> str:
    """Índice de GPU a usar ('' = que lo elija Ollama/torch).

    Si el LLM va en CPU el índice no aplica, así que se ignora.
    """
    idx = (gpu_idx or "auto").strip()
    if _gpu_mode(mode) == "cpu":
        return ""
    return idx if idx not in ("", "auto") else ""


def _child_env(base: dict) -> dict:
    """Fuerza UTF-8 en los subprocesos.

    Sin esto, en Windows el log del CLI se decodifica con la codificación de
    la consola (cp1252 en español) y al imprimir '✅' o '°C' el proceso muere
    con UnicodeEncodeError aunque el informe ya esté escrito.
    """
    env = dict(base)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _ollama_env(mode, gpu_idx: str) -> dict:
    """Entorno para el subproceso CLI.

    OLLAMA_NUM_GPU        -> cuántas capas van a la GPU ("auto" = que decida el
                             CLI según la potencia de la tarjeta, "-1" = todas).
    OLLAMA_GPU_INDEX      -> main_gpu (índice) que se envía en la consulta.
    CUDA_VISIBLE_DEVICES  -> índice visible también para Whisper (torch).
    AUDITV_STATUS_FILE    -> dónde escribir el estado que pinta el panel de GPU.
    """
    env = _child_env(os.environ)
    env["OLLAMA_NUM_GPU"] = {"auto": "auto", "gpu": "-1", "cpu": "0"}[_gpu_mode(mode)]
    idx = _gpu_index(gpu_idx, mode)
    if idx:
        env["OLLAMA_GPU_INDEX"] = idx
        env["CUDA_VISIBLE_DEVICES"] = idx
    else:
        env.pop("OLLAMA_GPU_INDEX", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env["AUDITV_STATUS_FILE"] = _STATUS_FILE
    return env


def list_gpu_indices() -> list:
    """['auto', '0', '1', ...] con las GPUs NVIDIA detectadas."""
    gpus = hardware.gpu_infos()
    return ["auto"] + [str(g["index"]) for g in gpus] if gpus else ["auto"]


def _gpu_args(mode, gpu_idx: str) -> list:
    """Argumentos de CLI para decidir dónde corre el LLM de Ollama."""
    args = ["--ollama-gpu", _gpu_mode(mode)]
    idx = _gpu_index(gpu_idx, mode)
    if idx:
        args += ["--ollama-gpu-index", idx]
    return args


# ---------------------------------------------------------------------------
# Estado del proceso (lo escribe el CLI) y panel de GPU
# ---------------------------------------------------------------------------
def read_status() -> dict:
    """Lee el JSON de estado del CLI ([] si aún no ha escrito nada)."""
    try:
        data = json.loads(Path(_STATUS_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _clear_status() -> None:
    """Borra el estado de la ejecución anterior (al empezar algo nuevo)."""
    try:
        Path(_STATUS_FILE).unlink()
    except OSError:
        pass


def _fmt_temp(temp, warn=None, abort=None) -> tuple:
    """(texto, color) con la temperatura ya coloreada según los umbrales."""
    if temp is None:
        return "—", "var(--body-text-color-subdued)"
    if abort is not None and temp >= abort:
        return f"{temp:.0f}°C", "#e03131"
    if warn is not None and temp >= warn:
        return f"{temp:.0f}°C", "#e8590c"
    return f"{temp:.0f}°C", "#2f9e44"


def _load_color(pct) -> str:
    """Verde por debajo del 60 %, naranja hasta el 85 %, rojo por encima."""
    if pct is None:
        return "var(--body-text-color-subdued)"
    if pct >= 85:
        return "#e03131"
    if pct >= 60:
        return "#e8590c"
    return "#2f9e44"


def _bar(pct, color) -> str:
    """Barrita de progreso para un porcentaje (o un guion si no se midió)."""
    if pct is None:
        return ""
    pct = max(0.0, min(100.0, float(pct)))
    return (f"<span style='display:inline-block;width:44px;height:6px;"
            f"border-radius:3px;background:rgba(128,128,128,.22);"
            f"vertical-align:middle;margin-right:4px'>"
            f"<span style='display:block;width:{pct:.0f}%;height:6px;"
            f"border-radius:3px;background:{color}'></span></span>")


def _metric(label, value, color=None, extra="") -> str:
    """Una cifra del panel: etiqueta, valor coloreado y detalle."""
    col = color or "inherit"
    return (f"<span><span style='opacity:.7'>{label}</span> "
            f"<b style='color:{col}'>{value}</b>{extra}</span>")


def _component_row(name, model, device, res, vram_mb=None) -> tuple:
    """(etiqueta, valor) de un componente del panel con su consumo."""
    marks = []
    if res and res.get("rss_mb"):
        marks.append(f"{hardware.fmt_mem(res['rss_mb'])} RAM")
        if res.get("cpu_pct"):
            marks.append(f"{hardware.fmt_pct(res['cpu_pct'])} CPU")
        if res.get("threads"):
            marks.append(f"{res['threads']} hilos")
    if vram_mb:
        marks.append(f"{hardware.fmt_vram(vram_mb)} VRAM")
    where = f" · {device.upper()}" if device else ""
    return (name, f"{model or '—'}{where}" + ("  ·  " + " · ".join(marks) if marks else ""))


def gpu_panel_html() -> str:
    """Panel de equipo: GPU, CPU y RAM, con el consumo de Whisper y de la IA local.

    Se refresca cada 2 s con un Timer. El estado de la ejecución (qué está
    usando la GPU ahora) viene del fichero que escribe el CLI; la temperatura
    y el consumo se leen aquí directo, para que también se vea en reposo.
    """
    st = read_status()
    gpus = hardware.gpu_infos()
    gpu = hardware.primary_gpu(gpus)
    mps = hardware.torch_mps_info() if not gpu else {}
    thresholds = st.get("thresholds") or {}
    warn = thresholds.get("warn")
    abort = thresholds.get("abort")
    resume = thresholds.get("resume")
    running = bool(st.get("running"))
    resting = bool(st.get("gpu_resting"))

    # Consumo medido del equipo y de cada componente.
    res = hardware.resource_snapshot(cli_pid=st.get("pid") if running else None)

    if not gpu and mps.get("usable"):
        # macOS con chip Apple Silicon: la GPU existe pero nvidia-smi no la ve.
        gpu = {"index": 0, "name": mps.get("name") or "Apple GPU",
               "vram_total_mb": mps.get("vram_mb") or 0, "vram_used_mb": 0,
               "temp_c": None, "util_pct": None, "compute_cap": None,
               "shared": True}
        state, state_color = ("en uso", "#2f9e44") if running else (
            "libre", "var(--body-text-color-subdued)")
        detail = (f"{gpu['name']} (Metal, memoria compartida con la CPU"
                  f" ≈ {hardware.fmt_vram(gpu['vram_total_mb'])})")
        temp, temp_color = "—", "var(--body-text-color-subdued)"
        vram_used = 0
        vram_total = gpu["vram_total_mb"]
        vram_pct = None
        # nvidia-smi no mide la memoria de MPS: se muestra la del sistema.
        vram_total = vram_used = 0
    elif gpu:
        temp, temp_color = _fmt_temp(st.get("gpu_temp") or gpu.get("temp_c"),
                                     warn, abort)
        if resting:
            state, state_color = "en descanso (caliente)", "#e8590c"
        elif running and (st.get("llm_device") == "GPU"
                          or st.get("whisper_batch_device") == "GPU"
                          or (st.get("llm_device") is None and gpu.get("util_pct"))):
            state, state_color = "en uso", "#2f9e44"
        elif running:
            state, state_color = "libre (trabajando en CPU)", "var(--body-text-color-subdued)"
        else:
            state, state_color = "libre", "var(--body-text-color-subdued)"
        detail = f"{hardware.describe_gpu(gpu)}"
        util = gpu.get("util_pct")
        if util is not None:
            detail += f" · {util} % de uso"
        free = hardware.gpu_free_mb(gpu)
        detail += f" · {hardware.fmt_vram(free)} libres"
        vram_used = gpu.get("vram_used_mb") or 0
        vram_total = gpu.get("vram_total_mb") or 0
        vram_pct = (vram_used / vram_total * 100) if vram_total else None
    else:
        temp, temp_color = "—", "var(--body-text-color-subdued)"
        state, state_color = "no detectada", "var(--body-text-color-subdued)"
        detail = ("No se detecta GPU (ni NVIDIA ni Apple MPS): todo irá en CPU."
                  if mps else
                  "No hay GPU NVIDIA detectable: todo irá en CPU.")
        vram_used = vram_total = vram_pct = 0

    # Qué equipo de transcripción y de análisis se ha elegido en esta ejecución.
    wdev = st.get("whisper_device") or st.get("device")
    wmodel = st.get("whisper_model")
    ldev = st.get("llm_device")
    lmodel = st.get("llm_model")
    auto_mark = " (auto)" if st.get("whisper_model_auto") else ""
    llm_mark = " (auto)" if st.get("llm_model_auto") else ""
    pending = " (pendiente)"
    if not st:
        # Todavía no ha corrido nada: se enseña lo que se usaría con «auto».
        info = hardware.torch_gpu_info()
        kind = info.get("kind") or "cpu"
        wdev = kind if kind in ("cuda", "mps") else "cpu"
        wmodel = hardware.recommend_whisper_model(wdev, info.get("vram_mb"))
        lmodel, _best = hardware.pick_ollama_model()
        ldev = "GPU" if hardware.ollama_gpu_default()[0] else "CPU"
        auto_mark = llm_mark = pending

    # VRAM de la IA local: la reporta Ollama en /api/ps.
    loaded = hardware.ollama_running_models()
    ollama_vram = 0
    for m in loaded:
        if not lmodel or m["name"].split(":")[0] in lmodel:
            ollama_vram += m["size_vram_mb"]

    # La RAM de Ollama solo se muestra si tiene un modelo cargado: en reposo su
    # proceso está en memoria pero no hace nada, y solo confunde.
    ollama_res = res.get("ollama") if (running and loaded) else None
    rows = [
        _component_row("Whisper", f"{wmodel or '—'}{auto_mark}", wdev,
                       res.get("whisper") if running else None,
                       hardware.gpu_vram_of_pid(st.get("pid")) if running else 0),
        _component_row("IA local", f"{lmodel or '—'}{llm_mark}", ldev,
                       ollama_res, ollama_vram),
    ]
    stage = st.get("stage") or ""
    batch = st.get("batch") or st.get("whisper_batch") or ""
    if running and stage:
        rows.insert(0, ("Ahora", " · ".join(x for x in (stage, batch) if x)))

    table = "".join(
        f"<tr><td style='padding:1px 10px 1px 0;opacity:.7;white-space:nowrap'>{k}</td>"
        f"<td style='padding:1px 0'>{v}</td></tr>"
        for k, v in rows
    )
    limits = (f"aviso {warn}°C · descanso {abort}°C · retoma a {resume}°C"
              if None not in (warn, abort, resume) else
              "protección por temperatura activa")
    if resting and st.get("gpu_rest_reason"):
        limits += f"<br><span style='color:#e8590c'>{st['gpu_rest_reason']}"
        limits += " — los lotes siguen en CPU, sin perder nada.</span>"

    # Barra de equipo: CPU, RAM y VRAM del sistema entero.
    cores = res.get("cpu_cores")
    cpu_pct = res.get("cpu_pct")
    ram_pct = res.get("ram_pct")
    sw = res.get("swap_used_mb")
    sysline = " · ".join(x for x in (
        _metric("CPU", f"{hardware.fmt_pct(cpu_pct)}", _load_color(cpu_pct),
                _bar(cpu_pct, _load_color(cpu_pct))),
        _metric("RAM", f"{hardware.fmt_mem(res.get('ram_used_mb'))} / "
                 f"{hardware.fmt_mem(res.get('ram_total_mb'))}",
                 _load_color(ram_pct), _bar(ram_pct, _load_color(ram_pct))),
        _metric("VRAM", f"{hardware.fmt_vram(vram_used)} / "
                 f"{hardware.fmt_vram(vram_total)}", _load_color(vram_pct),
                _bar(vram_pct, _load_color(vram_pct))),
        _metric("Núcleos", cores or "—"),
        _metric("Libre", hardware.fmt_mem(res.get("ram_avail_mb"))),
        _metric("Swap", hardware.fmt_mem(sw) if sw else "—"),
    ) if x)

    return f"""<div style="border:1px solid rgba(128,128,128,.25);border-radius:8px;
  padding:.5rem .75rem;margin-bottom:.5rem;font-size:.92em;line-height:1.5">
  <div style="display:flex;flex-wrap:wrap;gap:.75rem;align-items:center">
    <span style="font-weight:600">🎮 GPU</span>
    <span>{detail}</span>
    <span style="color:{temp_color};font-weight:600">🌡 {temp}</span>
    <span style="color:{state_color}">● {state}</span>
    <span style="opacity:.6;font-size:.85em">{limits}</span>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:1rem;margin-top:.3rem;
    padding-top:.3rem;border-top:1px solid rgba(128,128,128,.18)">
    {sysline}
  </div>
  <table style="border-collapse:collapse;margin-top:.25rem">{table}</table>
</div>"""


def refresh_gpu_panel() -> str:
    return gpu_panel_html()


def _new_process_kwargs() -> dict:
    """Opciones para crear el CLI en su propio grupo de procesos.

    En POSIX, `start_new_session` lo independiza de la terminal (así se puede
    matar el grupo entero, con ffmpeg/yt-dlp dentro). En Windows no existe esa
    opción: lo equivalente es `CREATE_NEW_PROCESS_GROUP`, y hace falta además
    `CREATE_NO_WINDOW` para que no salte una ventana de consola detrás.
    """
    if os.name == "nt":
        return {"creationflags": (subprocess.CREATE_NEW_PROCESS_GROUP
                                  | getattr(subprocess, "CREATE_NO_WINDOW", 0))}
    return {"start_new_session": True}


def _run_cli_streaming(args, key="video", env=None):
    """Run the CLI and stream its log lines (stderr) to the UI."""
    proc = subprocess.Popen(
        [_PY, _CLI] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env or _child_env(os.environ),
        **_new_process_kwargs(),
    )
    _RUNNER[key] = proc
    lines = []
    try:
        yield ("Iniciando...", "")
        while True:
            line = proc.stderr.readline()
            if line:
                lines.append(line.rstrip())
                yield ("\n".join(lines[-200:]), "")
                continue
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        out = proc.stdout.read().strip()
        proc.wait()
        tail = "\n".join(lines[-200:])
        yield (tail, out or f"Finalizado (código {proc.returncode})")
    finally:
        # Si el evento fue cancelado desde la GUI, el generador se cierra aquí:
        # nos aseguramos de que el proceso (y sus hijos) no queden sueltos.
        if _RUNNER.get(key) is proc:
            _RUNNER[key] = None
        _terminate(proc)


def _read_tail(path: str, max_chars: int = 30000) -> str:
    try:
        data = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return data[-max_chars:]


def list_ollama_models() -> list:
    """Modelos instalados en el Ollama local, con 'auto' al principio.

    'auto' es el valor por defecto: el CLI elige el mejor que quepa en el
    equipo (el mayor en GPU, ~4B en CPU), así que el usuario no necesita saber
    qué tiene instalado.
    """
    models = hardware.ollama_models()
    return ["auto"] + sorted(m["name"] for m in models)


def _refresh_models():
    """Recarga el desplegable de modelos de Ollama (respeta lo elegido)."""
    choices = list_ollama_models()
    return gr.update(choices=choices)


def ollama_dropdown(label="Modelo de IA local (Ollama)"):
    """Desplegable de Ollama: 'auto' + lo instalado, con valor recomendado."""
    models = hardware.ollama_models()
    auto, best = hardware.pick_ollama_model(models)
    info = ""
    if best:
        info = f" (recomendado para este equipo: {auto}, " \
               f"{hardware.model_size_gb(best):.1f} GB)"
    elif not models:
        info = " — Ollama no responde; el análisis LLM fallará"
    return gr.Dropdown(
        list_ollama_models(),
        value="auto",
        label=label + info,
        allow_custom_value=True,
        info="«auto» = el mejor modelo que tengas instalado que quepa en tu equipo.",
    )


def _whisper_dropdown(live: bool = False):
    """Desplegable de Whisper con 'auto' (el modelo adecuado a este equipo)."""
    rep = hardware.whisper_report(device="cpu")
    choices = ["auto", "tiny", "base", "small", "medium"] if live \
        else list(hardware.WHISPER_CHOICES)
    cached = ", ".join(rep["cached"]) or "ninguno"
    return gr.Dropdown(
        choices,
        value="auto",
        label=f"Modelo Whisper (recomendado aquí: {rep['auto']})",
        info=f"«auto» elige según tu GPU, la RAM y la duración del audio. "
             f"Ya descargados: {cached}.",
    )


def _gpu_controls():
    """Selector de dónde corre el LLM, preajustado a la potencia del equipo.

    Devuelve (radio, desplegable de índice de GPU, botón de redetectar).
    """
    use_gpu, reason = hardware.ollama_gpu_default()
    return (
        gr.Radio(
            ["auto", "gpu", "cpu"],
            value="auto",
            label="Dónde corre la IA (Ollama)",
            info=(f"«auto» = recomendado aquí: "
                  f"{'GPU' if use_gpu else 'CPU'}. {reason}"),
        ),
        gr.Dropdown(list_gpu_indices(), value="auto", label="GPU para la IA (índice)"),
        gr.Button("↻ Detectar GPUs"),
    )


def _pick_directory_tk() -> str:
    """Selector de carpetas de Tk (viene con Python en Windows y macOS)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return ""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(
            title="Selecciona la carpeta de salida", initialdir=_DEFAULT_OUTDIR
        )
        root.destroy()
        return chosen or ""
    except Exception:
        return ""


def pick_directory() -> str:
    """Abre el selector de carpetas del sistema y devuelve la ruta.

    Primero se intenta el selector nativo de escritorio (kdialog/zenity, que es
    lo de Linux); en Windows y macOS, donde no existen, se cae al de Tk.
    """
    for picker in (
        ["kdialog", "--getexistingdirectory", _DEFAULT_OUTDIR],
        ["zenity", "--file-selection", "--directory", "--title=Selecciona carpeta"],
    ):
        try:
            out = subprocess.run(
                picker, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
            )
        except Exception:
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return _pick_directory_tk()


# ---------------------------------------------------------------------------
# Tab 1: Video / URL
# ---------------------------------------------------------------------------
def on_run_video(
    file_in, path_in, url_in, outdir, device, model, interval,
    autoclean, llm_m, no_llm, extract_frames_cb, cookies_browser, cookies_file,
    llm_use_gpu=False, llm_gpu_sel="auto",
):
    outdir = (outdir or _DEFAULT_OUTDIR).strip()
    Path(outdir).mkdir(parents=True, exist_ok=True)

    video_arg = None
    if url_in and url_in.strip():
        video_arg = url_in.strip()
    elif file_in:
        src = file_in[0] if isinstance(file_in, list) else file_in
        video_arg = str(src)
    elif path_in and path_in.strip():
        video_arg = path_in.strip()

    if not video_arg:
        return ("Elige un video local (subir o ruta) o pega una URL.", "")

    _clear_status()
    args = [
        "--video", video_arg,
        "--output-dir", outdir,
        "--model", model or "auto",
        "--interval", str(int(interval)),
        "--llm", llm_m or "auto",
        "--autoclean", autoclean or "keep",
        "--device", device or "auto",
    ] + _gpu_args(llm_use_gpu, llm_gpu_sel)
    if no_llm:
        args.append("--no-llm")
    if not extract_frames_cb:
        args.append("--no-frames")
    if cookies_browser and cookies_browser.strip():
        args += ["--cookies-from-browser", cookies_browser.strip()]
    if cookies_file and cookies_file.strip():
        args += ["--cookies", cookies_file.strip()]

    yield from _run_cli_streaming(
        args, key="video", env=_ollama_env(llm_use_gpu, llm_gpu_sel)
    )


# ---------------------------------------------------------------------------
# Tab 2: Apuntes desde transcripción
# ---------------------------------------------------------------------------
def on_transcript(t_file, t_path, t_outdir, t_llm, t_split, t_split_chars,
                  t_llm_use_gpu=False, t_llm_gpu_sel="auto"):
    txt = None
    if t_file:
        src = str(t_file[0] if isinstance(t_file, list) else t_file)
        txt = src
    elif t_path and t_path.strip():
        txt = t_path.strip()

    if not txt:
        yield ("Elige un archivo .txt o escribe su ruta.", "")
        return

    outdir = (t_outdir or _DEFAULT_OUTDIR).strip()
    Path(outdir).mkdir(parents=True, exist_ok=True)

    _clear_status()
    args = [
        "--transcript", txt,
        "--output-dir", outdir,
        "--llm", t_llm or "auto",
    ] + _gpu_args(t_llm_use_gpu, t_llm_gpu_sel)
    if t_split:
        args += ["--split-chars", str(int(t_split_chars or 6000))]
    yield from _run_cli_streaming(
        args, key="notes", env=_ollama_env(t_llm_use_gpu, t_llm_gpu_sel)
    )


# ---------------------------------------------------------------------------
# Tab 3: Reunión en vivo
# ---------------------------------------------------------------------------
def on_live_start(source, model, device, out_txt):
    if _LIVE["proc"] and _LIVE["proc"].poll() is None:
        return "Ya hay una captura en curso. Detenla antes de iniciar otra."
    out_txt = (out_txt or str(_PROJECT / "informes" / "reunion_live.txt")).strip()
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    _LIVE["txt"] = out_txt
    _LIVE["proc"] = subprocess.Popen(
        [_PY, _LIVE_SCRIPT, "--source", source or "auto",
         "--model", model or "auto", "--device", device or "auto",
         "-o", out_txt],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        encoding="utf-8", errors="replace",
        env=_child_env(os.environ),
        **_new_process_kwargs(),
    )
    return f"Captura iniciada desde '{source or 'auto'}'. Escribe la transcripción en {out_txt}"


def on_live_stop():
    proc = _LIVE.get("proc")
    if proc and proc.poll() is None:
        if not _request_stop(proc):
            proc.terminate()  # sin Ctrl-C posible: se corta en seco
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        _LIVE["proc"] = None
    if _LIVE.get("txt") and Path(_LIVE["txt"]).exists():
        return "Captura detenida y transcripción guardada."
    return "No hay captura activa."


def on_live_refresh():
    return _read_tail(_LIVE.get("txt") or "")


def on_live_notes():
    txt = _LIVE.get("txt")
    if not txt or not Path(txt).exists():
        yield ("No hay transcripción todavía. Inicia una captura y detenla al terminar.", "")
        return
    _clear_status()
    use_gpu, _ = hardware.ollama_gpu_default()
    args = ["--transcript", txt, "--output-dir", str(Path(txt).parent),
            "--llm", "auto"] + _gpu_args(use_gpu, "auto")
    yield from _run_cli_streaming(
        args, key="notes", env=_ollama_env(use_gpu, "auto")
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="AuditV — Análisis de video y apuntes") as demo:
        # Panel de equipo: qué GPU hay, su temperatura y si se está usando.
        gpu_panel = gr.HTML(gpu_panel_html())
        with gr.Tab("Video / URL"):
            with gr.Row():
                file_in = gr.File(
                    # Sin lista de extensiones: se acepta cualquier formato
                    # (video, audio o contenedor raro) y ffmpeg decide.
                    label="Video o audio local (opcional)",
                )
                with gr.Column():
                    path_in = gr.Textbox(label="…o ruta del video local")
                    url_in = gr.Textbox(
                        label="…o URL (YouTube/plataforma)",
                        placeholder="https://…",
                    )
            with gr.Row():
                outdir = gr.Textbox(value=_DEFAULT_OUTDIR, label="Carpeta de salida (informe + frames)")
                outdir_btn = gr.Button("📂 Seleccionar Carpeta de Guardado")
                device = gr.Dropdown(["auto", "cpu", "cuda", "mps"], value="auto",
                                label="Device")
                model = _whisper_dropdown()
            with gr.Row():
                interval = gr.Slider(1, 60, value=10, step=1, label="Intervalo frames (s)")
                autoclean = gr.Dropdown(["keep", "ask", "delete"], value="keep", label="Autoclean")
                llm_m = ollama_dropdown()
                llm_refresh = gr.Button("↻ Actualizar modelos")
                no_llm = gr.Checkbox(label="Omitir análisis LLM")
                extract_frames_cb = gr.Checkbox(value=True, label="Extraer frames")
            with gr.Row():
                llm_use_gpu, llm_gpu_sel, llm_gpu_refresh = _gpu_controls()
            with gr.Row():
                cookies_browser = gr.Textbox(label="Cookies del navegador (chrome/firefox…)")
                cookies_file = gr.Textbox(label="Archivo de cookies (ruta)")
            with gr.Row():
                run_btn = gr.Button("▶ Analizar video / URL", variant="primary")
                cancel_btn = gr.Button("⏹ Cancelar", variant="stop")
            log_box = gr.Textbox(label="Progreso / log", lines=16, max_lines=30, interactive=False, elem_id="av_log_video", autoscroll=False)
            status = gr.Textbox(label="Resultado", interactive=False, autoscroll=False)
            run_inputs = [file_in, path_in, url_in, outdir, device, model,
                          interval, autoclean, llm_m, no_llm, extract_frames_cb,
                          cookies_browser, cookies_file, llm_use_gpu, llm_gpu_sel]
            run_event = run_btn.click(
                on_run_video,
                inputs=run_inputs,
                outputs=[log_box, status],
                js=_autoscroll_js(len(run_inputs)),
                scroll_to_output=False,
            )
            cancel_btn.click(
                lambda: on_cancel("video"),
                cancels=[run_event],
                outputs=[status],
            )
            outdir_btn.click(pick_directory, outputs=[outdir])
            llm_refresh.click(_refresh_models, outputs=[llm_m])
            llm_gpu_refresh.click(
                lambda: gr.update(choices=list_gpu_indices()), outputs=[llm_gpu_sel]
            )

        with gr.Tab("Apuntes"):
            with gr.Row():
                t_file = gr.File(file_types=[".txt"], label="Transcripción (.txt)")
                with gr.Column():
                    t_path = gr.Textbox(label="…o ruta del .txt")
                    with gr.Row():
                        t_outdir = gr.Textbox(value=_DEFAULT_OUTDIR, label="Carpeta del informe")
                        t_outdir_btn = gr.Button("📂 Seleccionar Carpeta de Guardado")
                        t_llm = ollama_dropdown()
                        t_llm_refresh = gr.Button("↻ Actualizar modelos")
            with gr.Row():
                t_split = gr.Checkbox(value=True, label="Dividir texto en partes: .md por cada parte + informe fusionado (los textos largos se dividen solos; aquí fijas el tamaño)")
                t_split_chars = gr.Number(value=6000, minimum=2000, maximum=20000,
                                          step=500, label="Caracteres por parte")
            with gr.Row():
                t_llm_use_gpu, t_llm_gpu_sel, t_llm_gpu_refresh = _gpu_controls()
            with gr.Row():
                t_btn = gr.Button("📝 Generar informe (_apuntes.md)", variant="primary")
            t_cancel = gr.Button("⏹ Cancelar", variant="stop")
            t_log = gr.Textbox(label="Log", lines=10, max_lines=20, interactive=False, elem_id="av_log_notes", autoscroll=False)
            t_status = gr.Textbox(label="Informe generado", interactive=False, autoscroll=False)
            t_inputs = [t_file, t_path, t_outdir, t_llm, t_split, t_split_chars,
                        t_llm_use_gpu, t_llm_gpu_sel]
            t_event = t_btn.click(
                on_transcript,
                inputs=t_inputs,
                outputs=[t_log, t_status],
                js=_autoscroll_js(len(t_inputs)),
                scroll_to_output=False,
            )
            t_cancel.click(
                lambda: on_cancel("notes"),
                cancels=[t_event],
                outputs=[t_status],
            )
            t_outdir_btn.click(pick_directory, outputs=[t_outdir])
            t_llm_refresh.click(_refresh_models, outputs=[t_llm])
            t_llm_gpu_refresh.click(
                lambda: gr.update(choices=list_gpu_indices()), outputs=[t_llm_gpu_sel]
            )

        with gr.Tab("Reunión en vivo"):
            with gr.Row():
                live_source = gr.Textbox(
                    value="auto",
                    label="Fuente de audio (auto = sistema/.monitor · alsa_input… = micrófono)",
                )
                live_model = _whisper_dropdown(live=True)
                live_device = gr.Dropdown(["auto", "cpu", "cuda", "mps"], value="auto",
                                label="Device")
            with gr.Row():
                live_out = gr.Textbox(
                    value=str(_PROJECT / "informes" / "reunion_live.txt"),
                    label="Archivo de transcripción destino",
                )
                live_out_btn = gr.Button("📂 Seleccionar Carpeta de Guardado")
            with gr.Row():
                live_start = gr.Button("▶ Iniciar captura")
                live_stop = gr.Button("⏹ Detener y guardar")
            live_area = gr.Textbox(label="Transcripción en vivo", lines=16, interactive=False, autoscroll=False)
            live_status = gr.Textbox(label="Estado", interactive=False, autoscroll=False)
            live_notes_btn = gr.Button("📝 Generar apuntes de esta reunión")
            live_notes_log = gr.Textbox(label="Log apuntes", lines=8, interactive=False, elem_id="av_log_live", autoscroll=False)
            live_notes_status = gr.Textbox(label="Informe de apuntes", interactive=False, autoscroll=False)

            live_start.click(
                on_live_start,
                inputs=[live_source, live_model, live_device, live_out],
                outputs=[live_status],
            )
            live_stop.click(on_live_stop, outputs=[live_status])

            def pick_live_output():
                folder = pick_directory()
                return str(Path(folder) / "reunion_live.txt") if folder else ""

            live_out_btn.click(pick_live_output, outputs=[live_out])

            live_notes_btn.click(
                on_live_notes, outputs=[live_notes_log, live_notes_status],
                js=_autoscroll_js(0), scroll_to_output=False,
            )

            timer = gr.Timer(1)
            timer.tick(on_live_refresh, outputs=[live_area])

        # El panel de GPU se refresca cada 2 s (temperatura + estado del proceso).
        panel_timer = gr.Timer(2.0)
        panel_timer.tick(refresh_gpu_panel, outputs=[gpu_panel])

    return demo


def _open_browser(port: int) -> None:
    """Open the browser a couple of seconds after the server is up."""
    time.sleep(2.5)
    import webbrowser

    webbrowser.open(f"http://127.0.0.1:{port}")


_FOOTER_CSS = """<style>
footer[aria-label="Gradio footer navigation"] {
  position: fixed !important;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  display: flex !important;
  justify-content: center !important;
  width: 100% !important;
  background: var(--body-background-fill, #ffffff) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  padding: 0.35rem 1.5rem !important;
  margin: 0 !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  align-items: center;
}
footer[aria-label="Gradio footer navigation"]::before {
  content: "🎬 AuditV";
  position: absolute;
  left: 1.25rem;
  top: 50%;
  transform: translateY(-50%);
  font-size: 1.02rem;
  font-weight: 600;
  color: var(--body-text-color);
  white-space: nowrap;
}
gradio-app {
  margin-top: 3.1rem !important;
}
</style>"""


def _brand_header() -> gr.Markdown:
    return gr.Markdown(
        """<div style="text-align:center;line-height:1.25;">
        <span style="font-size:1.25rem;font-weight:600;">🎬 AuditV</span>
        </div>"""
    )


def main() -> int:
    import argparse
    import threading

    parser = argparse.ArgumentParser(description="AuditV GUI (Gradio).")
    parser.add_argument("--port", type=int, default=7860, help="Puerto HTTP")
    parser.add_argument("--no-browser", action="store_true",
                        help="No abrir el navegador automáticamente")
    parser.add_argument("--share", action="store_true",
                        help="Crear un enlace público temporal (gradio)")
    args = parser.parse_args()

    demo = build_app()
    try:
        demo.queue()
    except Exception:
        pass
    if not args.no_browser and not args.share:
        threading.Timer(2.5, _open_browser, args=(args.port,)).start()
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=False,
        share=bool(args.share),
        head=_FOOTER_CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())