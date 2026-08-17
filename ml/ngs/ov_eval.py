"""ov_eval.py — evaluate candidate ONNX graphs on the Intel Arc B70 Pro via OpenVINO.

Runs in an OpenVINO env (e.g. Q:/AIz/LLM/OV/venvs/openvino). OpenVINO reads .onnx natively, so this
needs only openvino + numpy — no onnx / onnxruntime — which is why ml.ngs is import-light. The project
.venv builds the candidate graphs (it has onnx); this scores them on the B70 (it has the GPU runtime).

  <ov-env>/python.exe -m ml.ngs.ov_eval <onnx_dir> [GPU|CPU]
      onnx_dir must contain *.onnx + pairs.json  (bench.py --ov produces both, then calls this)
"""
import json
import pathlib
import sys
import time
import numpy as np

from .spec import encode, IN_SHAPE   # spec is numpy-only -> safe in a bare OpenVINO env


def _load_pairs(p):
    d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    return [(np.array(a), np.array(b)) for a, b in d]


def eval_compiled(compiled, pairs):
    """Same official path as score.evaluate, but through an OpenVINO compiled model."""
    ok = 0
    for ig, og in pairs:
        try:
            res = list(compiled([encode(ig)]).values())[0]
            pred = (np.asarray(res) > 0.0).astype(np.float32)
            if pred.shape == IN_SHAPE and np.array_equal(pred, encode(og)):
                ok += 1
        except Exception:
            pass
    return ok


def _resolve_device(core, device):
    """Map a logical name to a real device, picking the B70 dGPU by name (not GPU.0, which is the iGPU)."""
    if device.upper() in ("GPU", "B70", "DGPU"):
        for dv in core.available_devices:
            if dv.startswith("GPU"):
                try:
                    nm = core.get_property(dv, "FULL_DEVICE_NAME")
                except Exception:
                    nm = ""
                if any(k in nm.lower() for k in ("b70", "arc(tm) pro", "arc pro", "battlemage")):
                    return dv, nm
    nm = ""
    if device.startswith("GPU"):
        try:
            nm = core.get_property(device, "FULL_DEVICE_NAME")
        except Exception:
            pass
    return device, nm


def run(onnx_dir, device="GPU"):
    import openvino as ov
    core = ov.Core()
    d = pathlib.Path(onnx_dir)
    pairs = _load_pairs(d / "pairs.json")
    models = sorted(d.glob("*.onnx"))
    device, gpu_name = _resolve_device(core, device)
    print(f"ov_eval: {len(models)} graphs x {len(pairs)} pairs  device={device} {gpu_name}")
    print(f"  ov devices: {core.available_devices}")
    t0 = time.perf_counter()
    solved = errs = 0
    for m in models:
        try:
            compiled = core.compile_model(core.read_model(str(m)), device)
            solved += (eval_compiled(compiled, pairs) == len(pairs))
        except Exception as e:
            errs += 1
            if errs <= 3:
                print(f"  {m.name}: ERR {type(e).__name__}: {str(e)[:90]}")
    dt = time.perf_counter() - t0
    rate = len(models) / dt if dt else 0.0
    print(f"  {len(models)} graphs in {dt:6.2f}s -> {rate:7.2f} graphs/s   (solved {solved}, errors {errs})")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 64
    return run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "GPU")


if __name__ == "__main__":
    sys.exit(main())
