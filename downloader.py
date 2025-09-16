import os
from yt_dlp import YoutubeDL


def download_audio(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    ydl_opts = {
        # Prefer native audio-only; m4a if available, else bestaudio
        "format": "bestaudio[ext=m4a]/bestaudio/best",

        # Output & safety
        "outtmpl": os.path.join(out_dir, "%(title).80s.%(ext)s"),
        "restrictfilenames": True,
        "noplaylist": True,  # ignore &list=... URLs

        # Reliability & speed hints
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 5,
        "extractor_args": {"youtube": {"player_client": ["android"]}},

        # See progress (helps diagnose slowness)
        "quiet": False,
        "noprogress": False,
    }

    ffmpeg_loc = os.environ.get("FFMPEG_LOCATION")
    if ffmpeg_loc:
        ydl_opts["ffmpeg_location"] = ffmpeg_loc

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Return whichever extension was actually saved
        base, _ = os.path.splitext(filename)
        for ext in (".m4a", ".opus", ".webm", ".mp3"):
            p = base + ext
            if os.path.exists(p):
                return p
        return filename
