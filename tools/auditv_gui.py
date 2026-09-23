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

Ejecutar:
    ./venv/bin/python tools/auditv_gui.py
Se abre en el navegador en http://127.0.0.1:7860
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import gradio as gr

_PROJECT = Path(__file__).resolve().parent.parent
_PY = str(_PROJECT / "venv" / "bin" / "python")
_CLI = str(_PROJECT / "video_to_md.py")
_LIVE = str(_PROJECT / "tools" / "live_meeting.py")

_DEFAULT_OUTDIR = str(_PROJECT / "informes")
_OLLAMA_MODELS = ["qwen3.5:4b"]

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
    """Termina el proceso y su grupo (ffmpeg/yt-dlp incluidos)."""
    if proc is None or proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (AttributeError, OSError, ProcessLookupError):
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (AttributeError, OSError, ProcessLookupError):
            proc.kill()
        proc.wait()


def on_cancel(key: str) -> str:
    """Cancela el proceso en curso del tipo 'key' (video, notes, live)."""
    proc = _RUNNER.get(key)
    alive = proc is not None and proc.poll() is None
    _terminate(proc)
    if _RUNNER.get(key) is proc:
        _RUNNER[key] = None
    return ("⏹ Proceso cancelado: ya no queda en segundo plano." if alive
            else "No hay proceso de ese tipo en ejecución.")


def _ollama_env(use_gpu: bool, gpu_idx: str) -> dict:
    """Entorno para el subproceso CLI: activa GPU para el LLM y elige cuál.

    OLLAMA_NUM_GPU -> cuántas capas va a la GPU (1/-1 = GPU, 0 = CPU).
    OLLAMA_GPU_INDEX -> main_gpu (índice) que se envía en la consulta.
    CUDA_VISIBLE_DEVICES -> índice visible también para Whisper (torch).
    """
    env = dict(os.environ)
    if use_gpu:
        env["OLLAMA_NUM_GPU"] = "-1"  # -1 = tantas capas como quepan en la GPU
    else:
        env["OLLAMA_NUM_GPU"] = "0"
    idx = (gpu_idx or "auto").strip()
    if use_gpu and idx not in ("", "auto"):
        env["OLLAMA_GPU_INDEX"] = idx
        env["CUDA_VISIBLE_DEVICES"] = idx
    else:
        env.pop("OLLAMA_GPU_INDEX", None)
        if "CUDA_VISIBLE_DEVICES" in env and not use_gpu:
            env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def list_gpu_indices() -> list:
    """Devuelve ['auto', '0', '1', ...] con las GPUs NVIDIA detectadas."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return ["auto"]
        gpus = [l for l in out.stdout.splitlines()
                if l.strip().lower().startswith("gpu ")]
        return ["auto"] + [str(i) for i in range(len(gpus))]
    except Exception:
        return ["auto"]


def _run_cli_streaming(args, key="video", env=None):
    """Run the CLI and stream its log lines (stderr) to the UI."""
    proc = subprocess.Popen(
        [_PY, _CLI] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=env or os.environ,
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
    """Return the locally available Ollama models (via local API)."""
    import json

    import urllib.request

    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=3
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name") for m in data.get("models", [])]
        return names or ["qwen3.5:4b"]
    except Exception:
        return ["qwen3.5:4b"]


def _refresh_models() -> dict:
    """Rebuild the Ollama model dropdown."""
    choices = sorted(list_ollama_models())
    return gr.update(choices=choices, value=choices[0])


def pick_directory() -> str:
    """Open a native folder picker (kdialog/zenity) and return the path."""
    for picker in (
        ["kdialog", "--getexistingdirectory", _DEFAULT_OUTDIR],
        ["zenity", "--file-selection", "--directory", "--title=Selecciona carpeta"],
    ):
        try:
            out = subprocess.run(
                picker, capture_output=True, text=True, timeout=60
            )
        except Exception:
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return ""


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

    args = [
        "--video", video_arg,
        "--output-dir", outdir,
        "--model", model,
        "--interval", str(int(interval)),
        "--llm", llm_m or "qwen3.5:4b",
        "--autoclean", autoclean or "keep",
        "--device", device or "auto",
    ]
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

    args = [
        "--transcript", txt,
        "--output-dir", outdir,
        "--llm", t_llm or "qwen3.5:4b",
    ]
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
        [_PY, _LIVE, "--source", source or "auto", "--model", model or "base",
         "--device", device or "cpu", "-o", out_txt],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    return f"Captura iniciada desde '{source or 'auto'}'. Escribe la transcripción en {out_txt}"


def on_live_stop():
    proc = _LIVE.get("proc")
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
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
    args = ["--transcript", txt, "--output-dir", str(Path(txt).parent)]
    yield from _run_cli_streaming(args, key="notes")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="AuditV — Análisis de video y apuntes") as demo:
        with gr.Tab("Video / URL"):
            with gr.Row():
                file_in = gr.File(
                    file_types=[
                        ".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".ts",
                        ".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac",
                        ".wma", ".aif", ".aiff", ".m4b",
                    ],
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
                device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                model = gr.Dropdown(["tiny", "base", "small", "medium", "large"], value="small", label="Modelo Whisper")
            with gr.Row():
                interval = gr.Slider(1, 60, value=10, step=1, label="Intervalo frames (s)")
                autoclean = gr.Dropdown(["keep", "ask", "delete"], value="keep", label="Autoclean")
                llm_m = gr.Dropdown(_OLLAMA_MODELS, value=_OLLAMA_MODELS[0], label="Modelo de IA local (Ollama)")
                llm_refresh = gr.Button("↻ Actualizar modelos")
                no_llm = gr.Checkbox(label="Omitir análisis LLM")
                extract_frames_cb = gr.Checkbox(value=True, label="Extraer frames")
            with gr.Row():
                llm_use_gpu = gr.Checkbox(
                    value=False,
                    label="Usar GPU para la IA (Ollama) — más rápido pero calienta más",
                )
                llm_gpu_sel = gr.Dropdown(
                    list_gpu_indices(), value="auto", label="GPU para la IA (índice)"
                )
                llm_gpu_refresh = gr.Button("↻ Detectar GPUs")
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
                        t_llm = gr.Dropdown(_OLLAMA_MODELS, value=_OLLAMA_MODELS[0], label="Modelo de IA local (Ollama)")
                        t_llm_refresh = gr.Button("↻ Actualizar modelos")
            with gr.Row():
                t_split = gr.Checkbox(value=True, label="Dividir texto en partes: .md por cada parte + informe fusionado (los textos largos se dividen solos; aquí fijas el tamaño)")
                t_split_chars = gr.Number(value=6000, minimum=2000, maximum=20000,
                                          step=500, label="Caracteres por parte")
            with gr.Row():
                t_llm_use_gpu = gr.Checkbox(
                    value=False,
                    label="Usar GPU para la IA (Ollama) — más rápido pero calienta más",
                )
                t_llm_gpu_sel = gr.Dropdown(
                    list_gpu_indices(), value="auto", label="GPU para la IA (índice)"
                )
                t_llm_gpu_refresh = gr.Button("↻ Detectar GPUs")
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
                live_model = gr.Dropdown(["tiny", "base", "small"], value="base", label="Modelo Whisper")
                live_device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")
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