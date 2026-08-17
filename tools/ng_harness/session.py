"""Canonical ONNX Runtime session factory for the test harness.

Every session is created the SAME way all agents should: sanitized (to match the
competition's score_network/run_network path), ORT_DISABLE_ALL graph optimization
(no fusion that could change behavior), and — critically — **single-threaded by
default** so N concurrent agents don't each grab all cores and thrash the CPU.
(The default ORT intra_op/inter_op = 0 = "use all cores"; that oversubscription
was a root cause of the CPU pileup, 2026-05-31.)
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort

ort.set_default_logger_severity(4)

_NG_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DATA_DIR = pathlib.Path(os.environ.get(
    "NEUROGOLF_DATA_DIR", str(_NG_ROOT / "data" / "competitions" / "neurogolf-2026")))
sys.path.insert(0, str(_DATA_DIR / "neurogolf_utils"))
try:
    import neurogolf_utils as _nu  # noqa: E402
    _nu._NEUROGOLF_DIR = str(_DATA_DIR) + "/"
except Exception:  # noqa: BLE001
    _nu = None


def make_session(model, sanitize: bool = True, threads: int = 1):
    """Build an InferenceSession.

    model    : path (str/Path) or a loaded onnx.ModelProto.
    sanitize : run neurogolf_utils.sanitize_model first (faithful to the eval path).
    threads  : intra/inter-op thread cap (1 = no oversubscription; this is the point).
    Returns (session, input_name).
    """
    if isinstance(model, (str, pathlib.Path)):
        m = onnx.load(str(model))
    else:
        m = model
    if sanitize and _nu is not None and hasattr(_nu, "sanitize_model"):
        san = _nu.sanitize_model(m)
        if san:
            m = san
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 4
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = int(threads)
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def run_raw(sess, input_name, x: np.ndarray) -> np.ndarray:
    """Run and return the raw 'output' tensor as float32 (normalized via >0,
    matching neurogolf_utils.run_network's decode)."""
    out = sess.run(["output"], {input_name: x.astype(np.float32)})[0]
    return (np.asarray(out) > 0.0).astype(np.float32)
