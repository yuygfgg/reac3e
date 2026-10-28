#!/usr/bin/env python3
"""Measure Zimtohrli distance and MOS between a reference and a degraded file.

Requirements:
    ffmpeg in PATH
    python3 -m pip install zimtohrli soundfile numpy

Examples:
    python3 tools/measure_zimtohrli.py reference.wav encoded.ec3
    python3 tools/measure_zimtohrli.py reference.flac encoded.mp3
    cat reference.wav | python3 tools/measure_zimtohrli.py - encoded.ec3
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sys.exit("soundfile is required: python3 -m pip install soundfile")

try:
    import zimtohrli
except ImportError:
    sys.exit("zimtohrli is required: python3 -m pip install zimtohrli")

FS = 48_000


def decode(path: str, destination: Path) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found in PATH")

    command = ["ffmpeg", "-v", "error", "-y"]
    if path == "-":
        command += ["-i", "pipe:0"]
    else:
        command += ["-i", path]
    command += ["-map", "0:a:0", "-ar", str(FS), "-c:a", "pcm_f32le", str(destination)]

    try:
        subprocess.run(
            command,
            check=True,
            stdin=sys.stdin.buffer if path == "-" else subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        message = f"ffmpeg failed to decode {path} (exit {error.returncode})"
        if detail:
            message += f": {detail}"
        raise SystemExit(message)

    audio, sample_rate = sf.read(destination, dtype="float32", always_2d=True)
    if sample_rate != FS:
        raise SystemExit(f"ffmpeg returned {sample_rate} Hz, expected {FS} Hz")
    return audio


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Zimtohrli distance and MOS between two audio files "
            "(any format ffmpeg can decode). Lower distance is better."
        )
    )
    parser.add_argument("reference", help="reference audio file (or '-' for stdin)")
    parser.add_argument("degraded", help="degraded audio file (or '-' for stdin)")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress informational messages"
    )
    args = parser.parse_args()

    if args.reference == "-" and args.degraded == "-":
        parser.error("only one input can be read from stdin")

    api = zimtohrli.Pyohrli()

    with tempfile.TemporaryDirectory(prefix="zimtohrli_") as workdir:
        workdir = Path(workdir)
        reference = decode(args.reference, workdir / "reference.wav")
        degraded = decode(args.degraded, workdir / "degraded.wav")

        if reference.shape[1] != degraded.shape[1]:
            raise SystemExit(
                "channel count mismatch: "
                f"reference has {reference.shape[1]}, degraded has {degraded.shape[1]}"
            )
        channels = reference.shape[1]

        if not args.quiet:
            print(
                f"reference: {args.reference}, {channels} ch, {len(reference)} frames",
                file=sys.stderr,
            )
            print(
                f"degraded:  {args.degraded}, {channels} ch, {len(degraded)} frames",
                file=sys.stderr,
            )
            if len(reference) != len(degraded):
                print(
                    "warning: durations differ by "
                    f"{abs(len(reference) - len(degraded)) / FS:.3f} s",
                    file=sys.stderr,
                )

        distances = []
        mos = []
        for channel in range(channels):
            distance = float(
                api.distance(reference[:, channel], degraded[:, channel])
            )
            score = float(zimtohrli.mos_from_zimtohrli(distance))
            distances.append(distance)
            mos.append(score)

            if not args.quiet:
                print(
                    f"channel {channel}: distance {distance:.4f} mos {score:.4f}",
                    file=sys.stderr,
                )

    print(f"zimtohrli: mean {np.mean(mos):.4f} (min {np.min(mos):.4f})")
    print("per-channel: [" + ", ".join(f"{value:.4f}" for value in mos) + "]")
    print("distance: [" + ", ".join(f"{value:.4f}" for value in distances) + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
