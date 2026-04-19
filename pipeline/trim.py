#!/usr/bin/env python3
"""
Auto-trim trailing blank/black screen from captured VHS video.

Detects the last black frame sequence and cuts there.
Uses -c copy (no re-encode) — fast and lossless.

Usage:
  python trim.py input.mkv                   # auto-detect, saves input_trimmed.mkv
  python trim.py input.mkv output.mkv        # explicit output path
  python trim.py input.mkv --preview         # show detected cut point only, don't write
  python trim.py input.mkv --threshold 0.05  # stricter black detection (default 0.10)
  python trim.py input.mkv --min-duration 1  # min black run to consider (default 0.5s)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def detect_black_segments(
    path: Path,
    min_duration: float = 0.5,
    pixel_threshold: float = 0.10,
) -> list[dict]:
    """Run ffmpeg blackdetect and return list of {start, end, duration} dicts."""
    cmd = [
        "ffmpeg", "-i", str(path),
        "-vf", f"blackdetect=d={min_duration}:pix_th={pixel_threshold}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # blackdetect writes to stderr
    output = result.stderr

    segments = []
    pattern = re.compile(
        r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)"
    )
    for m in pattern.finditer(output):
        segments.append({
            "start":    float(m.group(1)),
            "end":      float(m.group(2)),
            "duration": float(m.group(3)),
        })
    return segments


def get_duration(path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def find_trim_point(segments: list[dict], total_duration: float) -> float | None:
    """
    Return the start time of the last black segment if it runs to (near) the end.
    Returns None if no trailing black found.
    """
    if not segments:
        return None

    last = segments[-1]
    # Consider it "trailing" if the black segment ends within 2 seconds of the file end
    if total_duration - last["end"] <= 2.0:
        return last["start"]

    return None


def trim(input_path: Path, output_path: Path, trim_at: float) -> None:
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-t", str(trim_at),
        "-c", "copy",
        str(output_path),
    ], check=True)


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-trim trailing black screen from VHS capture")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path, nargs="?")
    p.add_argument("--preview", action="store_true", help="Detect only, do not write output")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="Pixel blackness threshold 0.0–1.0 (default 0.10; lower = stricter)")
    p.add_argument("--min-duration", type=float, default=0.5,
                   help="Minimum black run length in seconds (default 0.5)")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.with_name(args.input.stem + "_trimmed" + args.input.suffix)

    print(f"Analysing: {args.input}")
    total = get_duration(args.input)
    print(f"Duration:  {fmt_time(total)}")

    print(f"Detecting black frames (min={args.min_duration}s, threshold={args.threshold})...")
    segments = detect_black_segments(args.input, args.min_duration, args.threshold)

    if not segments:
        print("No black frames detected — video looks clean, nothing to trim.")
        sys.exit(0)

    print(f"Found {len(segments)} black segment(s):")
    for s in segments:
        print(f"  {fmt_time(s['start'])} → {fmt_time(s['end'])}  ({s['duration']:.2f}s)")

    trim_at = find_trim_point(segments, total)

    if trim_at is None:
        print("\nLast black segment is not at the end — no trailing blank screen detected.")
        print("If you still want to trim manually, run:")
        print(f"  ffmpeg -i {args.input} -t <seconds> -c copy {output}")
        sys.exit(0)

    removed = total - trim_at
    print(f"\nTrailing blank detected: {fmt_time(trim_at)} → end  ({removed:.1f}s to remove)")

    if args.preview:
        print("Preview mode — skipping write.")
        print(f"To trim, run:")
        print(f"  ffmpeg -i {args.input} -t {trim_at:.3f} -c copy {output}")
        sys.exit(0)

    print(f"Trimming → {output}")
    trim(args.input, output, trim_at)
    print(f"Done. Saved: {output}")


if __name__ == "__main__":
    main()
