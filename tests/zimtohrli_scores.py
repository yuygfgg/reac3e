#!/usr/bin/env python3
import sys
import wave

import numpy as np

try:
    import zimtohrli
except ImportError:
    sys.exit("zimtohrli is required: python3 -m pip install zimtohrli")

arguments = sys.argv[1:]
if not arguments or len(arguments) % 2 != 0:
    sys.exit("usage: zimtohrli_scores.py <reference.wav> <degraded.wav> [pair ...]")

pairs = [(arguments[i], arguments[i + 1]) for i in range(0, len(arguments), 2)]


def read_wav(path):
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            sys.exit(f"{path}: expected mono 16-bit PCM")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


api = zimtohrli.Pyohrli()
for reference, degraded in pairs:
    distance = float(api.distance(read_wav(reference), read_wav(degraded)))
    print(f"{zimtohrli.mos_from_zimtohrli(distance):.10f}")
