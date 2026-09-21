"""Agent-friendly wrapper around the video-to-markdown pipeline.

This module exposes a simple, callable interface that an AI agent can use to
analyze a video and get a structured Markdown report plus the images (frames).

It delegates all the heavy lifting to the :mod:`video_to_md` module:

    from tools.video_analyzer import analyze_video

    result = analyze_video("/path/to/video.mp4")
    # result == "/path/to/video_analysis.md"
"""

import os
import sys
from pathlib import Path

# Allow importing video_to_md.py living in the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from video_to_md import (  # noqa: E402
    video_to_md,
    transcript_to_md,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_WHISPER_MODEL,
)


def analyze_video(
    video_path: str,
    output_md: str = None,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    frame_interval: float = 10.0,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    use_llm: bool = True,
    output_dir: str = None,
    frames_dir: str = None,
    autoclean: str = "ask",
    device: str = "auto",
    cookies: str = None,
    cookies_from_browser: str = None,
    live_from_start: bool = False,
) -> str:
    """Analyze a video and return the path to the generated Markdown report.

    Args:
        video_path: Path to the video file or URL (YouTube, etc.).
        output_md: Optional output .md path. Defaults next to the video
            (or in output_dir / `informes/` for URLs).
        whisper_model: Whisper model size (base/small/medium/large).
        frame_interval: Seconds between captured frames.
        ollama_model: Ollama model id used for idea extraction.
        use_llm: If False, skip Ollama analysis (transcription + frames only).
        output_dir: Base dir for the .md and the frames dir. Takes priority
            over output_md's location for default frame placement.
        frames_dir: Explicit frames dir (overrides any default).
        autoclean: "ask" (prompt the user), "keep" or "delete" the downloaded
            video and generated files after the run.
        device: "auto" (detect), "cpu" (force CPU; safer on laptops) or "cuda".
        cookies: Netscape cookie file for logged-in downloads.
        cookies_from_browser: reuse browser cookies (chrome/firefox/...) for
            platforms where the user is already logged in.
        live_from_start: download supported live streams from the start.

    Returns:
        Path (str) to the generated Markdown report.
    """
    return video_to_md(
        video_path=video_path,
        output_md=output_md,
        whisper_model=whisper_model,
        frame_interval=frame_interval,
        ollama_model=ollama_model,
        use_llm=use_llm,
        output_dir=output_dir,
        frames_dir=frames_dir,
        autoclean=autoclean,
        device_hint=device,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
        live_from_start=live_from_start,
    )


def analyze_transcript(
    transcript_path: str,
    output_md: str = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    use_llm: bool = True,
    output_dir: str = None,
) -> str:
    """Generate standalone notes (ideas/discusiones/conclusiones) from a
    plain-text meeting transcript. Returns the path to the .md report.

    The report is written as a separate file next to the transcript (or in
    output_dir / --output). No video/audio/frames involved.
    """
    return transcript_to_md(
        transcript_path=transcript_path,
        output_md=output_md,
        ollama_model=ollama_model,
        use_llm=use_llm,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent wrapper to analyze a video or transcript.")
    parser.add_argument("--video", "-v", required=False, help="Path to the video file")
    parser.add_argument("--transcript", default=None,
                        help="Ruta a un archivo de texto (transcripción de una "
                             "reunión): genera un informe de apuntes aparte "
                             "(ideas, discusiones, conclusiones) sin video")
    parser.add_argument("--output", "-o", default=None, help="Output .md path")
    parser.add_argument("--output-dir", default=None,
                        help="Base output dir for the .md and frames")
    parser.add_argument("--frames-dir", default=None,
                        help="Explicit frames dir")
    parser.add_argument("--model", "-m", default=DEFAULT_WHISPER_MODEL,
                        help=f"Whisper model (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--interval", "-i", type=float, default=10.0,
                        help="Frame capture interval (default: 10.0)")
    parser.add_argument("--llm", default=DEFAULT_OLLAMA_MODEL,
                        help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Ollama analysis (transcription + frames only)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Device para Whisper: auto (default), cpu (seguro en "
                             "portátiles) o cuda")
    parser.add_argument("--cookies", default=None,
                        help="Ruta a un archivo de cookies (Netscape) para "
                             "descargas con login")
    parser.add_argument("--cookies-from-browser", default=None,
                        help="Reutilizar cookies del navegador (chrome, firefox, "
                             "chromium) para plataformas con sesión iniciada")
    parser.add_argument("--live-from-start", action="store_true",
                        help="Descargar directos soportados desde el inicio")
    parser.add_argument("--autoclean", choices=("ask", "keep", "delete"), default="ask",
                        help="Qué hacer con los archivos descargados/generados al terminar")
    args = parser.parse_args()

    if args.transcript:
        if args.video:
            parser.error("Usa solo uno: --video O --transcript, no ambos.")
        md = analyze_transcript(
            transcript_path=args.transcript,
            output_md=args.output,
            ollama_model=args.llm,
            use_llm=not args.no_llm,
            output_dir=args.output_dir,
        )
    elif args.video:
        md = analyze_video(
            video_path=args.video,
            output_md=args.output,
            whisper_model=args.model,
            frame_interval=args.interval,
            ollama_model=args.llm,
            use_llm=not args.no_llm,
            output_dir=args.output_dir,
            frames_dir=args.frames_dir,
            autoclean=args.autoclean,
            device=args.device,
            cookies=args.cookies,
            cookies_from_browser=args.cookies_from_browser,
            live_from_start=args.live_from_start,
        )
    else:
        parser.error("Especifica --video o --transcript")
        sys.exit(1)
    print(md)
