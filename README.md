
# yt-transcribe

A tiny Python CLI that downloads a YouTube video and generates a transcript (TXT, SRT, and VTT) using [faster-whisper]. Works even if YouTube doesn't provide a transcript.

---

## Features
- **Paste-URL workflow:** just run `python app.py` and paste a link when prompted (or pass the URL as an argument).
- **FFmpeg auto-detect:** finds FFmpeg on Windows/macOS/Linux (PATH, `FFMPEG_LOCATION`, winget/Scoop/Chocolatey/Homebrew/common paths).
- Downloads audio via `yt-dlp` (no YouTube API key).
- Transcribes with `faster-whisper` (CPU by default; GPU optional).
- Outputs: `.txt`, `.srt`, `.vtt` with timestamps.
- Optional: auto language detection, configurable model size.

---

## Quickstart

### 1) System requirements
- **Python 3.9+**
- **FFmpeg** (required by `yt-dlp` for audio). The app tries to auto-detect it. If not found, you'll see a friendly message with install instructions.

Common installs:
```powershell
# Windows (PowerShell)
winget install FFmpeg.FFmpeg -e
```
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

> If FFmpeg isn't on PATH, the app also respects `FFMPEG_LOCATION` and wires it up at runtime when possible.

---

### 2) Create & activate a virtualenv
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

---

### 3) Install Python dependencies
```bash
pip install -r requirements.txt
```

---

### 4) Run
**Interactive (prompt):**
```bash
python app.py
# Paste YouTube URL when prompted
```
### Batch mode
Create `urls.txt` (one URL per line, `#` comments allowed), then:

```bash
python batch_transcribe.py --file urls.txt --device cpu --model base
```
---

**Direct (no prompt):**
```bash
python app.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

By default, transcripts are written to `outputs/` and audio is cached in `downloads/`.

---

## Options
```bash
python app.py URL --model base --device cpu --output-dir outputs
```
- `--model` : `tiny` | `base` | `small` | `medium` | `large-v3` (bigger = more accurate, slower)
- `--device`: `cpu` (default), `cuda` (GPU), or `auto` (try GPU, else CPU)
- `--output-dir`: where to save transcripts

---

## Notes
- **Playlists:** The downloader forces single-video mode (`noplaylist=True`) so playlist URLs won't slow things down.
- **Speed:** We avoid re-encoding; yt-dlp downloads best available audio (often `.m4a` or `.webm/.opus`) which is fine for transcription.
- **GPU:** To use CUDA, install CUDA/cuDNN and run with `--device cuda` or `--device auto`.
- **Legal:** Only download/transcribe videos you are allowed to process.

---

## Project structure
```
yt-transcribe/
├─ app.py            # CLI entrypoint (prompts for URL if omitted)
├─ downloader.py     # Fast audio-only downloader (no re-encode)
├─ writers.py        # Helpers to write TXT/SRT/VTT
├─ requirements.txt
└─ README.md
```

---

## Troubleshooting
- **"ffmpeg not found"**: Install FFmpeg (see above). On Windows winget installs often to:
  `C:\Users\<USERNAME>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-*-full_build\bin`  
  The app also respects `FFMPEG_LOCATION` if you want to point it there manually.
- **Slow download**: Using a playlist link can be slower; this app already uses `noplaylist=True`. Try a clean video URL.
- **CUDA errors**: Use `--device cpu` or install proper CUDA/cuDNN runtime.

---

## License
MIT
