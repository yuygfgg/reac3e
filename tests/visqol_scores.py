#!/usr/bin/env python3
import sys

try:
    from visqol import VisqolApi
except ImportError:
    sys.exit("visqol is required: python3 -m pip install 'visqol-python[accel,fftw]'")

arguments = sys.argv[1:]
if not arguments or len(arguments) % 2 != 0:
    sys.exit("usage: visqol_scores.py <reference.wav> <degraded.wav> [pair ...]")

pairs = [(arguments[i], arguments[i + 1]) for i in range(0, len(arguments), 2)]
api = VisqolApi()
api.create(mode="audio")
results = api.measure_batch(pairs, max_workers=1, parallel=False)

for result in results:
    if isinstance(result, Exception):
        sys.exit(f"visqol error: {result}")
    print(f"{float(result.moslqo):.10f} {float(result.vnsim):.10f}")
