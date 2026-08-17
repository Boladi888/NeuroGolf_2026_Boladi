"""Shared wrapper for the NeuroGolf validator.

The official `neurogolf_utils.score_network` has appeared with both
`(sanitized_model, profiler_trace_path)` and
`(filename, sanitized_model, profiler_trace_path)` signatures. This module
normalizes that drift:

- `score_network_v2(filename) -> (macs, memory, params)` — drives the full pipeline
  (sanitize model, profiling InferenceSession, dummy run, end_profiling,
  call score_network, clean up trace file). Returns None on any failure or banned op.
- `score_to_points(memory, params)` — the new scoring formula:
  `max(1, 25 - log(memory + params))`. MACs were removed 2026-05-04.

Used by `build_submission.py` and the analysis tools that previously called
`neurogolf_utils.score_network(filename)` directly.
"""

import inspect
import math
import os
import pathlib
import sys
import time
import types

# Stub notebook deps that neurogolf_utils imports unconditionally at module load.
for _name in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["IPython.display"].display = lambda *a, **kw: None
sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

import numpy as np
import onnx
import onnxruntime

_NG_ROOT = pathlib.Path(__file__).parent.parent.resolve()
_DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(_NG_ROOT / "data" / "competitions" / "neurogolf-2026")))
sys.path.insert(0, str(_DATA_DIR / "neurogolf_utils"))

import neurogolf_utils  # noqa: E402

neurogolf_utils._NEUROGOLF_DIR = f"{_DATA_DIR.as_posix()}/"
_SCORE_NETWORK_PARAM_COUNT = len(inspect.signature(neurogolf_utils.score_network).parameters)
_BANNED_OP_TYPES = {op.upper() for op in getattr(neurogolf_utils, "_EXCLUDED_OP_TYPES", [])}


def _normalize_score_result(result):
    if result is None:
        return None
    if len(result) == 2:
        memory, params = result
        if memory is None or params is None:
            return None
        return 0, memory, params
    if any(value is None for value in result):
        return None
    return result


def score_network_v2(filename):
    """Drive the full new-rules scoring pipeline. Returns (macs, memory, params) or None.

    Steps:
      1. Load the ONNX model fresh and sanitize it with the official helper when available.
      2. Create a profiling InferenceSession with `ORT_DISABLE_ALL` graph optimization.
      3. Run a single dummy inference to populate the profiler trace.
      4. End profiling to capture the JSON trace path.
      5. Call `neurogolf_utils.score_network` using the available signature.
      6. Clean up the trace file regardless of outcome.
    """
    trace_path = None
    try:
        model = onnx.load(filename)
        for node in model.graph.node:
            if node.op_type.upper() in _BANNED_OP_TYPES:
                return None
        if hasattr(neurogolf_utils, "sanitize_model"):
            sanitized = neurogolf_utils.sanitize_model(model)
            if not sanitized:
                return None
        else:
            sanitized = model
            for node in sanitized.graph.node:
                node.name = node.output[0]
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        profile_dir = _NG_ROOT / "submissions"
        profile_dir.mkdir(exist_ok=True)
        profile_prefix = f"_ort_profile_{os.getpid()}_{time.time_ns()}_{pathlib.Path(filename).stem}"
        options.profile_file_prefix = str((profile_dir / profile_prefix).resolve())
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
        input_node = session.get_inputs()[0]
        input_shape = [d if isinstance(d, int) else 1 for d in input_node.shape]
        # Use a valid empty-grid one-hot (ch0=1 everywhere) rather than all
        # zeros. NeuroGolf ONNXes assume one-hot inputs; some (e.g. task394
        # with ArgMax→Mod→Reshape) crash native onnxruntime with all-zeros
        # because ArgMax on a uniform tensor produces a zero-sized dim
        # downstream. ch0=1 matches the encode() convention (background).
        dummy = np.zeros(input_shape, dtype=np.float32)
        if (len(input_shape) == 4
                and input_shape[1] >= 1
                and input_node.name == 'input'):
            # Set channel 0 (background) to 1.0 everywhere — valid one-hot.
            dummy[:, 0, :, :] = 1.0
        session.run(None, {input_node.name: dummy})
        trace_path = session.end_profiling()
    except Exception:  # noqa: BLE001
        return None
    try:
        if _SCORE_NETWORK_PARAM_COUNT == 3:
            result = neurogolf_utils.score_network(filename, sanitized, trace_path)
        else:
            result = neurogolf_utils.score_network(sanitized, trace_path)
        return _normalize_score_result(result)
    except Exception:  # noqa: BLE001
        return None
    finally:
        if trace_path:
            try:
                os.remove(trace_path)
            except (FileNotFoundError, OSError):
                pass


def score_to_points(memory, params):
    """The 2026-05-04 scoring formula: max(1, 25 - log(memory + params)). MACs removed."""
    return max(1.0, 25.0 - math.log(max(1.0, memory + params)))


def cost_v2(macs, memory, params):
    """The new-rules cost objective for the cost-aware dispatcher (memory + params, no MACs)."""
    return memory + params
