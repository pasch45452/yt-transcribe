import argparse
import os
import sys
import glob
from faster_whisper import WhisperModel

from downloader import download_audio
from writers import write_txt, write_srt, write_vtt


# ---------- FFmpeg auto-detect ----------
def _which_ffmpeg():
    """Return (bin_dir, exe_path) for ffmpeg if found, else (None, None)."""
    from shutil import which
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    found = which(exe)
    if found:
        return os.path.dirname(found), found

    # Windows: common install locations
    if os.name == "nt":
        candidates = []
        # Winget (Gyan)
        candidates += glob.glob(
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-*-full_build\bin")
        )
        # Winget (community FFmpeg.FFmpeg)
        candidates += glob.glob(
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\FFmpeg.FFmpeg_*\ffmpeg-*\bin")
        )
        # Scoop
        candidates += glob.glob(os.path.expandvars(r"%USERPROFILE%\scoop\apps\ffmpeg\current\bin"))
        # Chocolatey
        candidates += [r"C:\ProgramData\chocolatey\bin"]
        # Manual installs
        candidates += [r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin", r"C:\Program Files (x86)\ffmpeg\bin"]

        for d in candidates:
            ff = os.path.join(d, "ffmpeg.exe")
            if os.path.isfile(ff):
                return d, ff

    # macOS
    for d in ["/opt/homebrew/bin", "/usr/local/bin"]:
        if os.path.isfile(os.path.join(d, "ffmpeg")):
            return d, os.path.join(d, "ffmpeg")

    # Linux
    for d in ["/usr/bin", "/usr/local/bin", "/bin", "/snap/bin"]:
        if os.path.isfile(os.path.join(d, "ffmpeg")):
            return d, os.path.join(d, "ffmpeg")

    return None, None


def ensure_ffmpeg() -> bool:
    """Ensure ffmpeg/ffprobe are visible to yt-dlp; returns True if OK, else False."""
    # Respect already-set value
    loc = os.environ.get("FFMPEG_LOCATION")
    if loc:
        if loc not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + loc
        return True

    bin_dir, _ = _which_ffmpeg()
    if bin_dir:
        os.environ["FFMPEG_LOCATION"] = bin_dir
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + bin_dir
        return True

    return False
# ---------- /FFmpeg auto-detect ----------


def transcribe(audio_path: str, model_size: str, device: str, language: str = None):
    # device: "cpu", "cuda", or "auto"
    if device == "auto":
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
    else:
        compute = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute)

    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        beam_size=5,
        word_timestamps=False,
    )

    collected = [(seg.start, seg.end, seg.text) for seg in segments]
    return collected, info.language, info.duration


def _prompt_url_if_needed(cmdline_url: str) -> str:
    if cmdline_url:
        return cmdline_url
    try:
        url = input("Paste YouTube URL: ").strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        print("No URL provided. Exiting.", file=sys.stderr)
        sys.exit(1)
    if not url:
        print("No URL provided. Exiting.", file=sys.stderr)
        sys.exit(1)
    return url


def main():
    parser = argparse.ArgumentParser(description="Download a YouTube video and generate transcripts (TXT/SRT/VTT).")
    parser.add_argument("url", nargs="?", help="YouTube video URL (leave blank to be prompted)")
    parser.add_argument("--model", default="base", help="Whisper model size: tiny, base, small, medium, large-v3")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"], help="Device to run on")
    parser.add_argument("--language", default=None, help="Force language code (e.g., 'en'); defaults to auto-detect")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for transcripts")
    args = parser.parse_args()

    # Prompt for URL if not given
    url = _prompt_url_if_needed(args.url)

    # Ensure folders
    os.makedirs("downloads", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Make ffmpeg usable without user setup
    if not ensure_ffmpeg():
        print(
            "FFmpeg not found. Please install it or set FFMPEG_LOCATION.\n"
            "Windows (PowerShell): winget install FFmpeg.FFmpeg -e\n"
            "macOS: brew install ffmpeg\n"
            "Ubuntu/Debian: sudo apt install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(2)

    print("Downloading audio with yt-dlp...")
    audio_path = download_audio(url, "downloads")
    print(f"Audio saved to: {audio_path}")

    print("Transcribing with faster-whisper...")
    segments, lang, duration = transcribe(audio_path, args.model, args.device, args.language)
    print(f"Detected language: {lang} | Duration: {duration:.1f}s | Segments: {len(segments)}")

    base = os.path.splitext(os.path.basename(audio_path))[0]
    txt_path = os.path.join(args.output_dir, base + ".txt")
    srt_path = os.path.join(args.output_dir, base + ".srt")
    vtt_path = os.path.join(args.output_dir, base + ".vtt")

    write_txt(txt_path, segments)
    write_srt(srt_path, segments)
    write_vtt(vtt_path, segments)

    print(f"Wrote: {txt_path}")
    print(f"Wrote: {srt_path}")
    print(f"Wrote: {vtt_path}")
    print("Done.")


if __name__ == "__main__":
    main()
