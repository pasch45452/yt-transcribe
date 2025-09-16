from typing import List, Tuple


def format_timestamp(seconds: float, for_srt: bool = True) -> str:
    # converts seconds -> HH:MM:SS,mmm or HH:MM:SS.mmm
    millis = int(round(seconds * 1000))
    h = millis // 3600000
    m = (millis % 3600000) // 60000
    s = (millis % 60000) // 1000
    ms = millis % 1000
    sep = "," if for_srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_txt(path: str, segments: List[Tuple[float, float, str]]):
    with open(path, "w", encoding="utf-8") as f:
        for _, _, text in segments:
            f.write(text.strip() + "\n")


def write_srt(path: str, segments: List[Tuple[float, float, str]]):
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start, True)} --> {format_timestamp(end, True)}\n")
            f.write(text.strip() + "\n\n")


def write_vtt(path: str, segments: List[Tuple[float, float, str]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for (start, end, text) in segments:
            f.write(f"{format_timestamp(start, False)} --> {format_timestamp(end, False)}\n")
            f.write(text.strip() + "\n\n")
