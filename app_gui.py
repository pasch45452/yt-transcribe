import os
import sys
import glob
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


from downloader import download_audio
from writers import write_txt, write_srt, write_vtt
from faster_whisper import WhisperModel


# ---------- FFmpeg auto-detect (same spirit as app.py) ----------
def _which_ffmpeg():
    from shutil import which
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    found = which(exe)
    if found:
        return os.path.dirname(found), found

    if os.name == "nt":
        candidates = []
        # Local portable ffmpeg (ship a folder named "ffmpeg/bin" next to the exe/py)
        here = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
        candidates += [os.path.join(here, "ffmpeg", "bin")]

        # Winget (Gyan/Community)
        candidates += glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-*-full_build\bin"))
        candidates += glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\FFmpeg.FFmpeg_*\ffmpeg-*\bin"))
        # Scoop / Chocolatey / common manual
        candidates += [os.path.expandvars(r"%USERPROFILE%\scoop\apps\ffmpeg\current\bin"),
                       r"C:\ProgramData\chocolatey\bin",
                       r"C:\ffmpeg\bin",
                       r"C:\Program Files\ffmpeg\bin",
                       r"C:\Program Files (x86)\ffmpeg\bin"]
        for d in candidates:
            ff = os.path.join(d, "ffmpeg.exe")
            if os.path.isfile(ff):
                return d, ff

    for d in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/usr/local/bin", "/bin", "/snap/bin"]:
        if os.path.isfile(os.path.join(d, "ffmpeg")):
            return d, os.path.join(d, "ffmpeg")
    return None, None


def ensure_ffmpeg():
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
    # Avoid CUDA DLL issues in packaged exe by forcing CPU when not explicitly CUDA
    if device != "cuda":
        os.environ["CT2_FORCE_CPU"] = "1"

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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YT Transcribe")
        self.geometry("640x420")
        self.resizable(False, False)

        # State
        self.output_dir = tk.StringVar(value=os.path.abspath("outputs"))
        self.model = tk.StringVar(value="base")
        self.device = tk.StringVar(value="cpu")
        self.language = tk.StringVar(value="")  # optional

        # UI
        pad = {"padx": 10, "pady": 6}

        ttk.Label(self, text="YouTube URL:").grid(row=0, column=0, sticky="w", **pad)
        self.url_entry = ttk.Entry(self, width=70)
        self.url_entry.grid(row=0, column=1, columnspan=2, sticky="we", **pad)

        ttk.Label(self, text="Model:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.model, values=["tiny", "base", "small", "medium", "large-v3"], width=10)\
            .grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(self, text="Device:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.device, values=["cpu", "auto", "cuda"], width=10)\
            .grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(self, text="Language (optional):").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.language, width=12).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(self, text="Output folder:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.output_dir, width=50).grid(row=4, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse…", command=self.choose_dir).grid(row=4, column=2, sticky="w", **pad)

        self.go_btn = ttk.Button(self, text="Transcribe", command=self.start)
        self.go_btn.grid(row=5, column=1, sticky="w", **pad)
        ttk.Button(self, text="Open outputs", command=self.open_outputs).grid(row=5, column=2, sticky="w", **pad)

        self.log = tk.Text(self, height=13, width=78, state="disabled")
        self.log.grid(row=6, column=0, columnspan=3, **pad)

        # Ensure folders
        os.makedirs("downloads", exist_ok=True)
        os.makedirs(self.output_dir.get(), exist_ok=True)

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

    def start(self):
        url = self.url_entry.get().strip().strip('"').strip("'")
        if not url:
            messagebox.showwarning("Missing URL", "Please paste a YouTube URL.")
            return

        self.go_btn.config(state="disabled")
        t = threading.Thread(target=self.run_pipeline, args=(url,), daemon=True)
        t.start()

    def run_pipeline(self, url: str):
        try:
            self.log_print("Checking FFmpeg…")
            if not ensure_ffmpeg():
                self.log_print("FFmpeg not found. Please install it (winget/brew/apt) or place ffmpeg/bin next to the app.")
                messagebox.showerror("FFmpeg missing",
                                     "FFmpeg not found.\nInstall with winget/brew/apt or place ffmpeg/bin folder next to the app.")
                return

            self.log_print("Downloading audio with yt-dlp…")
            audio = download_audio(url, "downloads")
            self.log_print(f"Audio saved to: {audio}")

            self.log_print("Transcribing with faster-whisper… (first run may download a model)")
            segs, lang, dur = transcribe(audio, self.model.get(), self.device.get(), self.language.get() or None)
            self.log_print(f"Detected language: {lang} | Duration: {dur:.1f}s | Segments: {len(segs)}")

            base = os.path.splitext(os.path.basename(audio))[0]
            outdir = self.output_dir.get()
            txt = os.path.join(outdir, base + ".txt")
            srt = os.path.join(outdir, base + ".srt")
            vtt = os.path.join(outdir, base + ".vtt")
            write_txt(txt, segs)
            write_srt(srt, segs)
            write_vtt(vtt, segs)

            self.log_print(f"Wrote: {txt}")
            self.log_print(f"Wrote: {srt}")
            self.log_print(f"Wrote: {vtt}")
            self.log_print("Done ✅")
            messagebox.showinfo("Done", "Transcription finished!")
        except Exception as e:
            self.log_print(f"Error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.go_btn.config(state="normal")


if __name__ == "__main__":
    App().mainloop()
