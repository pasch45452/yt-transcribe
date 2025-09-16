"""
A simple Tkinter GUI for yt-transcribe with multi-URL support.
- Paste many YouTube URLs (one per line) OR load from a file.
- Click "Transcribe" to process sequentially with live logs.
"""

import glob
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from faster_whisper import WhisperModel
from downloader import download_audio
from writers import write_txt, write_srt, write_vtt


# ---------- FFmpeg auto-detect ----------
def _which_ffmpeg():
    from shutil import which
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    found = which(exe)
    if found:
        return os.path.dirname(found), found

    if os.name == "nt":
        candidates = []

        # Local portable ffmpeg (ship a folder named "ffmpeg/bin" next to the app)
        here = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
        candidates += [os.path.join(here, "ffmpeg", "bin")]

        # Winget (Gyan/Community)
        candidates += glob.glob(
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-*-full_build\bin")
        )
        candidates += glob.glob(
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\FFmpeg.FFmpeg_*\ffmpeg-*\bin")
        )

        # Scoop / Chocolatey / common manual
        candidates += [
            os.path.expandvars(r"%USERPROFILE%\scoop\apps\ffmpeg\current\bin"),
            r"C:\ProgramData\chocolatey\bin",
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin",
        ]
        for d in candidates:
            ff = os.path.join(d, "ffmpeg.exe")
            if os.path.isfile(ff):
                return d, ff

    for d in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/snap/bin"]:
        if os.path.isfile(os.path.join(d, "ffmpeg")):
            return d, os.path.join(d, "ffmpeg")

    return None, None


def ensure_ffmpeg():
    """Expose FFmpeg to yt-dlp via PATH/FFMPEG_LOCATION."""
    if os.environ.get("FFMPEG_LOCATION"):
        loc = os.environ["FFMPEG_LOCATION"]
        if loc not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + loc
        return True

    d, _ = _which_ffmpeg()
    if d:
        os.environ["FFMPEG_LOCATION"] = d
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + d
        return True

    return False
# ---------- /FFmpeg auto-detect ----------


def transcribe(audio_path: str, model_size: str, device: str, language: str | None = None):
    """Transcribe one audio file and return (segments, lang, duration)."""
    # Avoid CUDA DLL issues when packaged; force CPU unless explicitly cuda
    if device != "cuda":
        os.environ["CT2_FORCE_CPU"] = "1"

    if device == "auto":
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception:
            try:
                model = WhisperModel(model_size, device="metal", compute_type="float16")
            except Exception:
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
    elif device == "metal":
        model = WhisperModel(model_size, device="metal", compute_type="float16")
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


class App(tk.Tk):
    """Tk GUI allowing multi-URL batch transcription."""

    def __init__(self):
        super().__init__()
        self.title("YT Transcribe")
        self.geometry("1000x700")  
        self.minsize(760, 560)         # don't let it get too tiny
        self.resizable(True, True)   

        # State
        self.output_dir = tk.StringVar(value=os.path.abspath("outputs"))
        self.model = tk.StringVar(value="base")
        self.device = tk.StringVar(value="cpu")
        self.language = tk.StringVar(value="")  # optional

        self._build_ui()
        # Make grid stretch
        for col in range(3):
            self.grid_columnconfigure(col, weight=1)

        # Rows: URLs box (row=1) and log (row=7) expand vertically
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(7, weight=2)   # log gets a bit more weight
        
        # URLs box
        self.urls_text = tk.Text(self, height=8, width=88, wrap="none")
        self.urls_text.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=10, pady=6)

        # Progress bar
        self.pbar.grid(row=6, column=0, columnspan=3, sticky="we", padx=10, pady=(4, 2))

        # Log box
        self.log = tk.Text(self, height=16, width=88, state="disabled")
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=10, pady=6)

        sizegrip = ttk.Sizegrip(self)
        sizegrip.grid(row=8, column=2, sticky="se", padx=6, pady=4)


        # Ensure folders
        os.makedirs("downloads", exist_ok=True)
        os.makedirs(self.output_dir.get(), exist_ok=True)

    # ---------- UI ----------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # URLs text area (multi-line)
        ttk.Label(self, text="YouTube URLs (one per line):").grid(row=0, column=0, sticky="w", **pad)
        self.urls_text = tk.Text(self, height=8, width=88, wrap="none")
        self.urls_text.grid(row=1, column=0, columnspan=3, sticky="we", **pad)

        ttk.Button(self, text="Load from file…", command=self.load_urls_file).grid(row=2, column=0, sticky="w", **pad)

        # Options
        ttk.Label(self, text="Model:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.model, values=["tiny", "base", "small", "medium", "large-v3"], width=12)\
            .grid(row=3, column=0, sticky="e", padx=140, pady=6)

        ttk.Label(self, text="Device:").grid(row=3, column=1, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.device, values=["cpu", "auto", "cuda", "metal"], width=12)\
            .grid(row=3, column=1, sticky="e", padx=140, pady=6)

        ttk.Label(self, text="Language (optional):").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.language, width=15).grid(row=4, column=0, sticky="e", padx=140, pady=6)

        ttk.Label(self, text="Output folder:").grid(row=4, column=1, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.output_dir, width=45).grid(row=4, column=1, sticky="e", padx=100, pady=6)
        ttk.Button(self, text="Browse…", command=self.choose_dir).grid(row=4, column=2, sticky="w", **pad)

        # Controls
        self.go_btn = ttk.Button(self, text="Transcribe", command=self.start)
        self.go_btn.grid(row=5, column=1, sticky="w", **pad)
        ttk.Button(self, text="Open outputs", command=self.open_outputs).grid(row=5, column=2, sticky="w", **pad)

        # Progress + log
        self.pbar = ttk.Progressbar(self, length=720, mode="determinate")
        self.pbar.grid(row=6, column=0, columnspan=3, sticky="we", padx=10, pady=(4, 2))

        self.log = tk.Text(self, height=16, width=88, state="disabled")
        self.log.grid(row=7, column=0, columnspan=3, **pad)
    # ---------- /UI ----------

    # ---------- Helpers ----------
    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.output_dir.get() or os.getcwd())
        if d:
            self.output_dir.set(d)

    def open_outputs(self):
        path = self.output_dir.get()
        if not os.path.isdir(path):
            messagebox.showerror("Error", "Output directory does not exist.")
            return
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore
        elif sys.platform.startswith("darwin"):
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

    def log_print(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.update_idletasks()

    @staticmethod
    def _parse_urls(text: str) -> list[str]:
        urls = []
        seen = set()
        for line in text.splitlines():
            u = line.strip().strip('"').strip("'")
            if not u or u.startswith("#"):
                continue
            if u not in seen:
                seen.add(u)
                urls.append(u)
        return urls
    # ---------- /Helpers ----------

    # ---------- Actions ----------
    def load_urls_file(self):
        path = filedialog.askopenfilename(
            title="Select a .txt file with one URL per line",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.urls_text.delete("1.0", "end")
            self.urls_text.insert("1.0", content)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")

    def start(self):
        raw = self.urls_text.get("1.0", "end")
        urls = self._parse_urls(raw)
        if not urls:
            messagebox.showwarning("Missing URLs", "Paste one or more YouTube URLs (one per line).")
            return

        self.go_btn.config(state="disabled")
        self.pbar["value"] = 0
        self.pbar["maximum"] = len(urls)
        t = threading.Thread(target=self.run_pipeline_batch, args=(urls,), daemon=True)
        t.start()

    def run_pipeline_batch(self, urls: list[str]):
        try:
            self.log_print("Checking FFmpeg…")
            if not ensure_ffmpeg():
                self.log_print("FFmpeg not found. Install with winget/brew/apt or place ffmpeg/bin next to the app.")
                messagebox.showerror(
                    "FFmpeg missing",
                    "FFmpeg not found.\nInstall with winget/brew/apt or place ffmpeg/bin next to the app.",
                )
                return

            total = len(urls)
            ok = 0
            for idx, url in enumerate(urls, start=1):
                self.log_print(f"\n[{idx}/{total}] Downloading: {url}")
                try:
                    audio = download_audio(url, "downloads")
                    self.log_print(f"[{idx}/{total}] Audio saved to: {audio}")

                    self.log_print(f"[{idx}/{total}] Transcribing (first model download may take a bit)…")
                    segs, lang, dur = transcribe(
                        audio, self.model.get(), self.device.get(), self.language.get() or None
                    )
                    self.log_print(
                        f"[{idx}/{total}] Detected language: {lang} | Duration: {dur:.1f}s | Segments: {len(segs)}"
                    )

                    base = os.path.splitext(os.path.basename(audio))[0]
                    outdir = self.output_dir.get()
                    txt = os.path.join(outdir, base + ".txt")
                    srt = os.path.join(outdir, base + ".srt")
                    vtt = os.path.join(outdir, base + ".vtt")
                    write_txt(txt, segs)
                    write_srt(srt, segs)
                    write_vtt(vtt, segs)

                    self.log_print(f"[{idx}/{total}] Wrote: {txt}")
                    self.log_print(f"[{idx}/{total}] Wrote: {srt}")
                    self.log_print(f"[{idx}/{total}] Wrote: {vtt}")
                    ok += 1
                except Exception as e:
                    self.log_print(f"[{idx}/{total}] ❌ Error: {e}")
                finally:
                    self.pbar["value"] = idx
                    self.update_idletasks()

            self.log_print(f"\nSummary: {ok} ok / {total - ok} failed")
            messagebox.showinfo("Done", f"Transcription finished.\n{ok} ok / {total - ok} failed.")
        finally:
            self.go_btn.config(state="normal")
    # ---------- /Actions ----------


if __name__ == "__main__":
    App().mainloop()
