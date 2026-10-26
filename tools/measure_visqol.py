#!/usr/bin/env python3
"""Measure ViSQOL MOS-LQO between a reference and a degraded audio file.

Requirements:
    ffmpeg in PATH
    python3 -m pip install 'visqol-python[accel,fftw]' soundfile numpy

Examples:
    python3 tools/measure_visqol.py reference.wav encoded.ec3
    python3 tools/measure_visqol.py reference.flac encoded.mp3
    cat reference.wav | python3 tools/measure_visqol.py - encoded.ec3
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
    from visqol import VisqolApi
except ImportError:
    sys.exit("visqol is required: python3 -m pip install 'visqol-python[accel,fftw]'")

FS = 48_000


def decode(path: str, destination: Path) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found in PATH")

    command = ["ffmpeg", "-v", "error", "-y"]
    if path == "-":
        command += ["-i", "pipe:0"]
    else:
        command += ["-i", path]
    command += ["-map", "0:a:0", "-ar", str(FS), "-c:a", "pcm_s16le", str(destination)]

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

    audio, sample_rate = sf.read(destination, dtype="int16", always_2d=True)
    if sample_rate != FS:
        raise SystemExit(f"ffmpeg returned {sample_rate} Hz, expected {FS} Hz")
    return audio


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure ViSQOL MOS-LQO between two audio files "
            "(any format ffmpeg can decode)"
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

    with tempfile.TemporaryDirectory(prefix="visqol_") as workdir:
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

        api = VisqolApi()
        api.create(mode="audio")

        mos = []
        vnsim = []
        for channel in range(channels):
            reference_path = workdir / f"reference_ch{channel}.wav"
            degraded_path = workdir / f"degraded_ch{channel}.wav"
            sf.write(reference_path, reference[:, channel], FS, subtype="PCM_16")
            sf.write(degraded_path, degraded[:, channel], FS, subtype="PCM_16")

            result = api.measure(str(reference_path), str(degraded_path))
            mos.append(float(result.moslqo))
            vnsim.append(float(result.vnsim))

            if not args.quiet:
                print(
                    f"channel {channel}: mos {mos[-1]:.4f} vnsim {vnsim[-1]:.4f}",
                    file=sys.stderr,
                )

    print(f"visqol: mean {np.mean(mos):.4f} (min {np.min(mos):.4f})")
    print("per-channel: [" + ", ".join(f"{value:.4f}" for value in mos) + "]")
    print("vnsim: [" + ", ".join(f"{value:.4f}" for value in vnsim) + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
