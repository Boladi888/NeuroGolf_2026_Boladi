#!/usr/bin/env python3
"""Run byte-exact verification on two ONNX files in an isolated subprocess.

Used by spread_scan.py to survive ORT C-level crashes during inference
(e.g. integer-divide-by-zero on certain fp16-converted graphs).

Usage::

    python tools/spread_verify_isolated.py <orig.onnx> <modified.onnx> \
        --inputs <N> --seed <S> --tolerance <float-or-none>

Emits JSON to stdout: ``{"ok": true|false, "reason": "<text>"}``.
Always exits 0 except on unhandled crashes (parent reads non-zero).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import numpy as np
import onnx
import onnxruntime as ort
ort.set_default_logger_severity(3)


def _make_dummy_inputs(model: onnx.ModelProto, n_inputs: int, seed: int = 0) -> list[np.ndarray]:
    in_node = model.graph.input[0]
    shape = []
    for d in in_node.type.tensor_type.shape.dim:
        shape.append(d.dim_value if d.dim_value > 0 else 1)
    if not shape:
        shape = [1, 10, 30, 30]
    rng = np.random.RandomState(seed)
    inputs = []
    canon = np.zeros(shape, dtype=np.float32)
    if len(shape) == 4 and shape[1] >= 1:
        canon[:, 0, :, :] = 1.0
    inputs.append(canon)
    for _ in range(1, n_inputs):
        if len(shape) == 4 and shape[1] >= 1:
            idx = rng.randint(0, shape[1], size=(shape[0], shape[2], shape[3]))
            arr = np.zeros(shape, dtype=np.float32)
            for b in range(shape[0]):
                for r in range(shape[2]):
                    for c in range(shape[3]):
                        arr[b, idx[b, r, c], r, c] = 1.0
            inputs.append(arr)
        else:
            inputs.append(rng.random(shape).astype(np.float32))
    return inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original")
    parser.add_argument("modified")
    parser.add_argument("--inputs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", default="", help="empty/none = byte-exact; else float tol")
    args = parser.parse_args(argv)

    tol_str = (args.tolerance or "").strip().lower()
    tolerance = None if tol_str in ("", "none") else float(tol_str)

    try:
        orig = onnx.load(args.original)
        modified = onnx.load(args.modified)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "reason": f"load failed: {exc}"}))
        return 0

    try:
        # Use ORT_DISABLE_ALL to match the scoring runtime (score_network_v2
        # uses the same setting). Otherwise ORT's optimizer could smooth over
        # divergences that the validator would actually flag.
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_a = ort.InferenceSession(
            orig.SerializeToString(),
            options,
            providers=["CPUExecutionProvider"],
        )
        sess_b = ort.InferenceSession(
            modified.SerializeToString(),
            options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "reason": f"InferenceSession failed: {exc}"}))
        return 0

    inputs = _make_dummy_inputs(orig, args.inputs, seed=args.seed)
    in_name = sess_a.get_inputs()[0].name

    for i, arr in enumerate(inputs):
        try:
            out_a = sess_a.run(None, {in_name: arr})[0]
            out_b = sess_b.run(None, {in_name: arr})[0]
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "reason": f"input #{i} inference failed: {exc}"}))
            return 0
        if out_a.shape != out_b.shape:
            print(json.dumps({
                "ok": False,
                "reason": f"input #{i} shape mismatch {out_a.shape} vs {out_b.shape}",
            }))
            return 0
        if tolerance is None:
            if not np.array_equal(out_a, out_b):
                print(json.dumps({"ok": False, "reason": f"input #{i} byte-exact divergence"}))
                return 0
        else:
            if not np.allclose(out_a, out_b, atol=tolerance, rtol=0):
                max_diff = float(np.max(np.abs(out_a - out_b)))
                print(json.dumps({
                    "ok": False,
                    "reason": f"input #{i} max-diff {max_diff:.4g} > tol {tolerance:.4g}",
                }))
                return 0

    print(json.dumps({"ok": True, "reason": "verified"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
