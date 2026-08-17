"""spec.py — the NeuroGolf contract the search must obey.

Self-contained (only numpy). Everything here is the competition's fixed I/O + scoring
shape that any candidate graph must respect, lifted from Onnx_Advanced/core.md:
 - input tensor is named "input", shape [1,10,30,30] F32, one-hot on the channel axis,
   with the real grid embedded TOP-LEFT in the 30x30 canvas and trailing padding all-zero.
 - output tensor is named "output"; the decoder thresholds (>0.0) then compares.
 - score = memory + params ; points = max(1, 25 - ln(max(1,cost))).
"""
import math
import numpy as np

IN_NAME = "input"
OUT_NAME = "output"
CHANNELS = 10
CANVAS = 30
IN_SHAPE = (1, CHANNELS, CANVAS, CANVAS)

# ONNX TensorProto dtype enums (avoid importing onnx here so spec stays numpy-only)
FLOAT, FLOAT16, BOOL, INT32, INT64, UINT8, INT8 = 1, 10, 9, 6, 7, 2, 3
ITEMSIZE = {FLOAT: 4, FLOAT16: 2, BOOL: 1, INT32: 4, INT64: 8, UINT8: 1, INT8: 1}

# Validator-banned op types (core.md §Validator rejection criteria). "Sequence" substring also banned.
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def encode(grid):
    """Grid (HxW ints 0-9) -> [1,10,30,30] F32 one-hot, embedded top-left (the official encoder)."""
    g = np.asarray(grid)
    out = np.zeros(IN_SHAPE, dtype=np.float32)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            out[0, int(g[r, c]), r, c] = 1.0
    return out


def points(cost):
    return max(1.0, 25.0 - math.log(max(1.0, cost)))
