"""
Batch-run yt-transcribe's app.py with clear, live progress output.

Usage:
  # Use a file with one URL per line (blank lines and # comments allowed)
  python batch_transcribe.py --file urls.txt --device cpu --model base

  # Or pass URLs directly
  python batch_transcribe.py https://youtu.be/AAA https://youtu.be/BBB --device cpu
"""

import argparse
import sys
import subprocess
from pathlib import Path


def read_urls(file_path: str | None, inline_urls: list[str]) -> list[str]:
    urls: list[str] = []
    if file_path:
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    urls.extend(inline_urls)
    # dedupe while keeping order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def main():
    parser = argparse.ArgumentParser(description="Batch-run yt-transcribe (app.py) for many URLs with live progress.")
    parser.add_argument("urls", nargs="*", help="YouTube URLs (optional if --file is provided)")
    parser.add_argument("--file", help="Path to a text file with one YouTube URL per line")
    parser.add_argument("--app", default="app.py", help="Path to app.py (default: app.py)")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use (default: current)")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Device for transcription")
    parser.add_argument("--output-dir", default="outputs", help="Where transcripts are written")
    args = parser.parse_args()

    urls = read_urls(args.file, args.urls)
    if not urls:
        print("No URLs provided. Add them as arguments or via --file path/to/urls.txt", file=sys.stderr)
        sys.exit(1)

    total = len(urls)
    failures = 0

    for idx, url in enumerate(urls, start=1):
        print(f"\n[{idx}/{total}] Starting: {url}")
        cmd = [
            args.python, args.app,
            url, "--model", args.model,
            "--device", args.device,
            "--output-dir", args.output_dir,
        ]
        try:
            # Stream output live with a prefix
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(f"[{idx}/{total}] {line}", end="")
            rc = proc.wait()
            if rc != 0:
                failures += 1
                print(f"[{idx}/{total}] ❌ Exit code {rc}")
            else:
                print(f"[{idx}/{total}] ✅ Done")
        except KeyboardInterrupt:
            print(f"\n[{idx}/{total}] Interrupted by user. Exiting.")
            sys.exit(130)
        except Exception as e:
            failures += 1
            print(f"[{idx}/{total}] ❌ Error: {e}")

    print(f"\n=== Summary: {total - failures} ok / {failures} failed ===")
    sys.exit(0 if failures == 0 else 2)


if __name__ == "__main__":
    main()
