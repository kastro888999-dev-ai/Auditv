#!/usr/bin/env python3
"""Live meeting note-taker.

Transcribes a meeting in real time (system audio / microphone) with the local
Whisper model and writes a timestamped transcript. When the meeting ends
(Ctrl+C) it can generate the notes report (resumen, ideas, discusiones,
conclusiones) with Ollama via the same pipeline as --transcript.

Only uses tools already required by the project (ffmpeg + openai-whisper):
no extra dependencies, works even on Python 3.14 / weak GPUs.

Audio sources (PipeWire/PulseAudio):
  - system audio (meeting that plays through the speakers): the source that
    ends in ".monitor" (e.g. alsa_output...stereo.monitor)
  - microphone (meeting in a room or you speaking): alsa_input...

Usage:
    ./venv/bin/python tools/live_meeting.py --list-sources
    ./venv/bin/python tools/live_meeting.py --source auto --notes
    ./venv/bin/python tools/live_meeting.py --source alsa_output...monitor -o "informes/reunion.txt"

El modelo, el dispositivo y la GPU se detectan solos (--model/--device auto):
si el equipo aguanta la GPU se usa y, si se calienta durante la reunión, la
tarjeta descansa y la transcripción continúa en CPU sin perder nada.

Stop the capture with Ctrl+C; the transcript is saved and (with --notes) the
apuntes report is generated automatically.
"""

import argparse
import datetime
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np  # already a dependency of openai-whisper

# Allow importing video_to_md.py living in the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from video_to_md import (  # noqa: E402
    transcript_to_md,
    detect_device,
    resolve_whisper_model,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_WHISPER_MODEL,
    GUARD,
    log,
)

SAMPLE_RATE = 16000
CHUNK_SEC = 4.0          # seconds of audio transcribed per call
OVERLAP_SEC = 0.8        # overlap between chunks to keep words across borders
SILENCE_RMS = 120.0      # int16 RMS below this is treated as silence
MAX_CHARS = 8000         # cap fed to the notes analyzer
# Cada cuánto se mira la temperatura de la GPU durante una reunión larga.
GPU_CHECK_EVERY = 10     # chunks (~40 s)


def list_sources() -> list:
    """Return PipeWire/PulseAudio sink/mic sources via pactl, if available."""
    try:
        out = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        pass
    return []


def pick_source(source: str) -> str:
    """Resolve the audio source name (auto => prefer system monitor)."""
    if source not in ("auto", "default", ""):
        return source
    sources = list_sources()
    for ln in sources:
        if ".monitor" in ln:
            name = ln.split("\t")[1]
            log(f"Fuente de audio de sistema: {name}", "AUDIO")
            return name
    return "default"


def rms_int16(data: np.ndarray) -> float:
    """Root-mean-square of an int16 PCM buffer (silence gate)."""
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data.astype(np.float64)))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Toma apuntes de una reunión en vivo: transcribe con "
                    "Whisper local y genera el informe de ideas/discusiones/"
                    "conclusiones al terminar."
    )
    parser.add_argument("--source", default="auto",
                        help="Fuente de audio: nombre del dispositivo "
                             "(pactl), 'auto' (monitor de sistema si existe) "
                             "o 'default'")
    parser.add_argument("--model", default=DEFAULT_WHISPER_MODEL,
                        help="Modelo Whisper: auto (default, elige según tu "
                             "equipo) o tiny/base/small/medium/large")
    parser.add_argument("--language", default="es", help="Idioma (default: es)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto",
                        help="Device de Whisper: auto (default, usa la GPU solo "
                             "si el equipo y la temperatura lo permiten), cpu "
                             "o cuda")
    parser.add_argument("-o", "--output", default=None,
                        help="Ruta de la transcripción .txt (default: "
                             "'(cwd)/reunion_live_<timestamp>.txt')")
    parser.add_argument("--notes", action="store_true",
                        help="Al terminar (Ctrl+C) generar el informe de "
                             "apuntes (_apuntes.md) con Ollama")
    parser.add_argument("--llm", default=DEFAULT_OLLAMA_MODEL,
                        help=f"Modelo Ollama para los apuntes: auto (default, "
                             f"el mejor instalado que quepa) o el nombre exacto "
                             f"(default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--list-sources", action="store_true",
                        help="Listar fuentes de audio disponibles y salir")
    parser.add_argument("--chunk", type=float, default=CHUNK_SEC,
                        help=f"Segundos de audio por transcribir "
                             f"(default: {CHUNK_SEC})")
    parser.add_argument("--silence", type=float, default=SILENCE_RMS,
                        help=f"Umbral RMS de silencio (default: {SILENCE_RMS})")
    args = parser.parse_args(argv)

    if args.list_sources:
        for ln in list_sources():
            print(ln)
        return 0

    import whisper

    src = pick_source(args.source)
    log(f"Capturando audio desde: {src}")
    log("Pulsa Ctrl+C para finalizar la reunión y guardar.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        txt_path = str(Path(args.output).resolve())
    else:
        txt_path = str(Path.cwd() / f"reunion_live_{timestamp}.txt")
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)

    # 'auto' decide con el mismo criterio que el resto del proyecto: si el
    # equipo aguanta la GPU se usa, y si no, CPU (más lento pero seguro).
    device = detect_device(args.device)
    model_name = resolve_whisper_model(args.model, device, duration_s=None)
    model = whisper.load_model(model_name, device=device)

    ffmpeg_cmd = [
        "ffmpeg", "-loglevel", "error", "-nostdin",
        "-f", "pulse", "-i", src,
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ]

    started = datetime.datetime.now()
    stop = {"flag": False}

    def _on_sigint(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    def ts_now() -> str:
        elapsed = (datetime.datetime.now() - started).total_seconds()
        return _fmt_clock(elapsed)

    def _write_line(fh, line: str) -> None:
        fh.write(line + "\n")
        fh.flush()
        print(line, flush=True)

    last_printed = ""
    samples_per_chunk = int(SAMPLE_RATE * args.chunk)
    samples_overlap = int(SAMPLE_RATE * OVERLAP_SEC)
    buffer = np.zeros(0, dtype=np.int16)

    proc = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, bufsize=1 << 20,
    )

    segment_count = 0
    chunks_done = 0
    try:
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Reunión en vivo — {timestamp}\n")
            fh.write(f"- Fuente: {src}\n")
            fh.write(f"- Modelo Whisper: {model_name} ({device})\n\n")
            fh.flush()

            while not stop["flag"]:
                raw = proc.stdout.read(4096)
                if not raw:
                    break
                buffer = np.concatenate(
                    [buffer, np.frombuffer(raw, dtype=np.int16)]
                )
                if buffer.size < samples_per_chunk:
                    continue

                chunk = buffer[:samples_per_chunk]
                buffer = buffer[samples_per_chunk - samples_overlap:]

                if rms_int16(chunk) < args.silence:
                    continue

                # Una reunión puede durar horas: si la GPU se calienta, se la
                # deja descansar y el resto de la reunión sigue en CPU (la
                # transcripción no se corta ni se pierde nada).
                if device == "cuda" and chunks_done % GPU_CHECK_EVERY == 0:
                    was_resting = GUARD.resting
                    if GUARD.should_use_gpu("transcripción en vivo"):
                        if was_resting:
                            model.to("cuda")
                            device = "cuda"
                            log("GPU recuperada: la transcripción vuelve a la "
                                "GPU.", "GPU")
                    else:
                        model.to("cpu")
                        device = "cpu"
                        log("GPU en descanso: la transcripción sigue en CPU "
                            "(no se pierde nada).", "GPU")

                audio = chunk.astype(np.float32) / 32768.0
                result = model.transcribe(
                    audio,
                    fp16=False,
                    language=args.language,
                    condition_on_previous_text=False,
                )
                text = (result.get("text") or "").strip()
                chunks_done += 1
                if not text or result.get("no_speech_prob", 0) > 0.6:
                    continue

                if last_printed and (
                    text in last_printed or last_printed in text
                ):
                    continue

                stamp = ts_now()
                _write_line(fh, f"[{stamp}] {text}")
                last_printed = text
                segment_count += 1

        proc.stdout.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    log(
        f"Transcripción guardada: {txt_path} ({segment_count} segmentos, "
        f"Whisper {model_name} en {device.upper()})"
    )

    if args.notes:
        log("Generando informe de apuntes con Ollama...")
        md = transcript_to_md(
            transcript_path=txt_path,
            ollama_model=args.llm,
            use_llm=True,
        )
        log(f"Apuntes -> {md}")

    return 0


def _fmt_clock(seconds: float) -> str:
    """Format elapsed seconds as MM:SS or HH:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    raise SystemExit(main())