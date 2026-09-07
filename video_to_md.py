#!/usr/bin/env python3
"""Video to Markdown Analyzer.

Transcribe a video using Whisper (auto GPU/CPU), extracts key frames with
timestamps, and uses a local LLM via Ollama to produce an ordered summary of
the main ideas. Outputs a structured Markdown file.

Usage:
    python video_to_md.py --video <video_file> [--output <out.md>] [--model small] [--interval 10]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import datetime
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEFAULT_FRAME_INTERVAL = 10.0  # seconds between captured frames
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

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
def detect_device() -> str:
    """Return 'cuda' if torch can actually run on the GPU, else 'cpu'.

    Checks not only that CUDA is reported available, but also that the GPU's
    compute capability is supported by the installed PyTorch build and that a
    real CUDA kernel actually executes. Falls back to CPU otherwise.
    """
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
            return "cuda"
        else:
            log("No GPU detected, using CPU.", "DEVICE")
            return "cpu"
    except Exception as exc:
        log(f"CUDA unusable, falling back to CPU: {exc}", "DEVICE")
        return "cpu"


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
    result = model.transcribe(audio_path, fp16=(device == "cuda"))

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
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
    }
    if system:
        payload["system"] = system

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.HTTPError as exc:
        log(f"Ollama HTTP error: {exc.code} - {exc.read().decode()}", "ERROR")
        raise
    except Exception as exc:
        log(f"Ollama request failed: {exc}", "ERROR")
        raise


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    brace = re.search(r"\{.*\}", text, re.S)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass

    return {"raw": text}


def analyze_with_ollama(transcript: str, model: str) -> dict:
    """Use Ollama to summarize + extract ordered main ideas from a transcript."""
    MAX_TRANSCRIPT_CHARS = 8000
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        log(f"Transcript too long ({len(transcript)} chars), truncating to {MAX_TRANSCRIPT_CHARS} chars for analysis.")
        transcript = transcript[:MAX_TRANSCRIPT_CHARS]

    system = (
        "You are an expert video analyst. You receive a speech transcription "
        "from a video and must extract its main ideas in a structured way. "
        "Always answer in Spanish using ONLY valid JSON, no extra text."
    )

    prompt = (
        "Analiza la siguiente transcripción de un video y devuelve un objeto JSON "
        "con la siguiente estructura exacta:\n"
        "{\n"
        '  "titulo": "Título sugerido para el video",\n'
        '  "resumen": "Resumen ejecutivo de 2-3 oraciones",\n'
        '  "ideas": [\n'
        '    {"titulo": "Idea principal 1", "descripcion": "Detalle de la idea", "timestamp": 12.5},\n'
        '    {"titulo": "Idea principal 2", "descripcion": "Detalle", "timestamp": 30.0}\n'
        "  ],\n"
        '  "conceptos": ["concepto 1", "concepto 2"],\n'
        '  "conclusiones": ["conclusión 1", "conclusión 2"]\n'
        "}\n"
        "Reglas:\n"
        "- 'ideas' debe listar las ideas principales ORDENADAS por importancia, "
        "máximo 8.\n"
        "- 'timestamp' en cada idea debe ser el segundo aproximado del video donde "
        "se menciona (usa los tiempos de la transcripción).\n"
        "- 'conceptos' son términos o conceptos clave mencionados.\n"
        "- Si la transcripción está vacía, devuelve una estructura vacía con "
        "'resumen': 'Sin transcripción disponible'.\n\n"
        f"TRANSCRIPCIÓN:\n{transcript}"
    )

    log(f"Asking Ollama model '{model}' for analysis...")
    raw = _ollama_generate(prompt, model, system)
    return _extract_json(raw)



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
        lines.append("*(No se capturaron frames.)*")
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
    lines.append(f"*(Frames guardados en `{frames_dir}`)*")
    lines.append("")
    lines.append(build_frames_section(frames, frames_dir, output_md))

    with open(output_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    log(f"Markdown written -> {output_md}")



# ---------------------------------------------------------------------------
# URL download via yt-dlp
# ---------------------------------------------------------------------------
def is_url(path_or_url: str) -> bool:
    """Check if the input string looks like a URL."""
    return path_or_url.startswith(("http://", "https://", "www."))


def download_video(url: str, output_dir: str) -> str:
    """Download a video from a URL using yt-dlp. Returns path to downloaded file."""
    log(f"Downloading video from URL: {url}")
    out_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", out_template,
        "--merge-output-format", "mp4",
        "--no-warnings",
        url,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Find the downloaded file (yt-dlp may rename it)
    files = sorted(Path(output_dir).glob("*.*"), key=os.path.getmtime, reverse=True)
    for f in files:
        if f.suffix.lower() in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
            log(f"Downloaded -> {f}")
            return str(f)
    raise RuntimeError(f"Could not find downloaded video in {output_dir}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def video_to_md(
    video_path: str,
    output_md: str = None,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    frame_interval: float = DEFAULT_FRAME_INTERVAL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    use_llm: bool = True,
    output_dir: str = None,
    frames_dir: str = None,
) -> str:
    """Full pipeline: transcribe + analyze + capture frames -> markdown.

    video_path can be a local file path or a URL (YouTube, etc.).

    Output location is resolved in this order:
    - output_dir: base dir for both the .md and the frames dir (if given).
    - output_md: the .md goes where specified; frames default to its parent dir.
    - frames_dir: explicit frames dir (overrides any default).
    - Otherwise: local files default next to the video; URLs default to cwd.
    """
    downloaded_tmpdir = None
    if is_url(video_path):
        downloaded_tmpdir = tempfile.TemporaryDirectory(prefix="video_to_md_dl_")
        video_path = download_video(video_path, downloaded_tmpdir.name)

    try:
        video_path = str(Path(video_path).resolve())
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        base_name = Path(video_path).stem
        is_remote = downloaded_tmpdir is not None

        # Base dir for default outputs.
        if output_dir is not None:
            base_dir = Path(output_dir).resolve()
        elif output_md is not None:
            base_dir = Path(output_md).resolve().parent
        elif is_remote:
            base_dir = Path.cwd()
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
        os.makedirs(frames_dir, exist_ok=True)

        device = detect_device()

        with tempfile.TemporaryDirectory(prefix="video_to_md_") as tmpdir:
            # 1. Extract audio
            audio_path = os.path.join(tmpdir, "audio.wav")
            extract_audio(video_path, audio_path)

            # 2. Transcribe
            transcript = transcribe_audio(audio_path, whisper_model, device)

            # 3. Extract frames
            frames = extract_frames(video_path, frames_dir, frame_interval)

            # 4. Analyze with Ollama (best-effort)
            analysis = {}
            if use_llm:
                try:
                    analysis = analyze_with_ollama(transcript["text"], ollama_model)
                except Exception as exc:
                    log(f"Ollama analysis skipped: {exc}", "WARN")
                    analysis = {"resumen": "*(No se pudo analizar con Ollama)*"}
            else:
                log("Skipping Ollama analysis (--no-llm)")
                analysis = {"resumen": "*(Análisis LLM omitido)*"}

        # 5. Build markdown
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

        return output_md
    finally:
        if downloaded_tmpdir is not None:
            downloaded_tmpdir.cleanup()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a video: transcribe, extract ideas (via Ollama) "
        "and capture frames, outputting a structured Markdown report."
    )
    parser.add_argument("--video", "-v", required=True, help="Path to the video file or URL (YouTube, etc.)")
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
    parser.add_argument("--llm", default=DEFAULT_OLLAMA_MODEL,
                         help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--no-llm", action="store_true",
                         help="Skip Ollama analysis (transcription + frames only)")
    args = parser.parse_args(argv)

    try:
        md = video_to_md(
            video_path=args.video,
            output_md=args.output,
            whisper_model=args.model,
            frame_interval=args.interval,
            ollama_model=args.llm,
            use_llm=not args.no_llm,
            output_dir=args.output_dir,
            frames_dir=args.frames_dir,
        )
    except Exception as exc:
        log(f"Error: {exc}", "ERROR")
        return 1

    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

