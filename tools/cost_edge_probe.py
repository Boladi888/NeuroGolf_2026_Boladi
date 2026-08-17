"""Probe onnx-tool / validator cost-formula edge cases for unexploited optimizations.

Tests minimal hand-crafted ONNX graphs that exercise specific cost-formula corners,
then compares the validator's reported cost against expected.

Probes (all clean — no banned ops, no exploit tricks):

  1. Float16 vs float32 intermediates : memory term should halve (4 bytes -> 2 bytes/elem)
  2. Int8 vs float32 initializers : params term unaffected (element count); memory unchanged
  3. Constant-folded chain : intermediates produced by Constant-folding chains should drop
                              from the memory term per onnx-tool's `constant_folding=True`.
  4. Reused initializer : `params` should count it ONCE (per lessons.md 2026-05-01).
  5. Empty-shape tensor : volume([]) = 0 (per lessons.md 2026-05-02).

Each probe builds a tiny ONNX, scores it via the validator's `score_network`, and
prints macs/memory/params. Compares against expected.

Output: per-probe row with PASS/FAIL/INFO and the actual numbers. Findings that
contradict our existing lessons would be valuable; confirmations are useful too.

Usage:
    python tools/cost_edge_probe.py
    python tools/cost_edge_probe.py --probe float16-intermediate
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto
from onnx import numpy_helper

NG_ROOT = Path(__file__).resolve().parents[1]

# Stub notebook deps before importing neurogolf_utils (same pattern as build_submission.py).
import os
import types
for _name in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["IPython.display"].display = lambda *a, **kw: None
sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

_data_dir_str = os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026"))
DATA_DIR = Path(_data_dir_str)
sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))


def _build_model_minimal(nodes, initializers, value_info, opset=10, ir_version=7):
    """Build an ONNX with input/output of [1, 10, 30, 30] float32."""
    X = oh.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    Y = oh.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = oh.make_graph(nodes, "probe", [X], [Y], initializer=initializers, value_info=value_info)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    model.ir_version = ir_version
    return model


def score(model_path):
    """Call validator's score_network via the v2 wrapper (sanitize + profile + score).
    Returns (macs, memory, params) or (None, None, None)."""
    try:
        from _score_v2 import score_network_v2
    except ImportError:
        print("ERROR: cannot import tools._score_v2.", file=sys.stderr)
        sys.exit(1)
    result = score_network_v2(str(model_path))
    if result is None:
        return (None, None, None)
    return result


def probe_float16_intermediate():
    """Float16 intermediates should halve the memory cost vs float32."""
    print("\n=== probe: float16-intermediate ===")
    print("Cast input to float32 (no-op), then back to float32. Single intermediate.")
    print("FP32 baseline vs FP16 intermediate. Memory term should differ.")

    # FP32 baseline
    nodes_fp32 = [
        oh.make_node("Identity", ["input"], ["mid_fp32"]),
        oh.make_node("Identity", ["mid_fp32"], ["output"]),
    ]
    vi_fp32 = [oh.make_tensor_value_info("mid_fp32", TensorProto.FLOAT, [1, 10, 30, 30])]
    m_fp32 = _build_model_minimal(nodes_fp32, [], vi_fp32, opset=10)

    # FP16 intermediate via Cast
    nodes_fp16 = [
        oh.make_node("Cast", ["input"], ["mid_fp16"], to=TensorProto.FLOAT16),
        oh.make_node("Cast", ["mid_fp16"], ["output"], to=TensorProto.FLOAT),
    ]
    vi_fp16 = [oh.make_tensor_value_info("mid_fp16", TensorProto.FLOAT16, [1, 10, 30, 30])]
    m_fp16 = _build_model_minimal(nodes_fp16, [], vi_fp16, opset=10)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(m_fp32, f.name)
        fp32_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(m_fp16, f.name)
        fp16_path = f.name

    fp32_macs, fp32_mem, fp32_params = score(fp32_path)
    fp16_macs, fp16_mem, fp16_params = score(fp16_path)

    print(f"  FP32 baseline: macs={fp32_macs}, memory={fp32_mem}, params={fp32_params}")
    print(f"  FP16 mid:      macs={fp16_macs}, memory={fp16_mem}, params={fp16_params}")

    if fp32_mem and fp16_mem:
        ratio = fp16_mem / fp32_mem
        print(f"  Memory ratio FP16/FP32: {ratio:.4f} (expected ~0.5 if dtype is honored)")
        if 0.45 < ratio < 0.55:
            print("  RESULT: ✓ FP16 halves memory term as expected.")
        else:
            print(f"  RESULT: UNEXPECTED ratio. Validator may not honor FP16 in `calculate_memory`.")

    Path(fp32_path).unlink()
    Path(fp16_path).unlink()


def probe_int8_initializer():
    """Int8 initializers: params element-count should be unchanged vs float32 (per lessons.md)."""
    print("\n=== probe: int8-initializer ===")
    print("Compare a Conv with float32 weight (1000 elements) vs int8 weight (same shape).")

    weight_shape = [10, 10, 1, 1]  # 100 elements
    weight_fp32 = numpy_helper.from_array(np.zeros(weight_shape, dtype=np.float32), name="W_fp32")
    weight_int8 = numpy_helper.from_array(np.zeros(weight_shape, dtype=np.int8), name="W_int8")

    # FP32 conv
    nodes_fp32 = [
        oh.make_node("Conv", ["input", "W_fp32"], ["output"]),
    ]
    m_fp32 = _build_model_minimal(nodes_fp32, [weight_fp32], [], opset=10)

    # Int8 conv requires Cast (Conv at opset 10 only takes float)
    # Skip int8 path — impractical at opset 10. Just probe FP32.

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(m_fp32, f.name)
        fp32_path = f.name

    fp32_macs, fp32_mem, fp32_params = score(fp32_path)
    print(f"  FP32 Conv (100 elem weight): macs={fp32_macs}, memory={fp32_mem}, params={fp32_params}")
    print(f"  RESULT: params should be {np.prod(weight_shape)} = 100 (element count, dtype-agnostic).")
    if fp32_params == 100:
        print("  CONFIRMED:Confirmed.")
    else:
        print(f"  Unexpected: params={fp32_params}")

    Path(fp32_path).unlink()


def probe_reused_initializer():
    """Reused initializer should count once toward params."""
    print("\n=== probe: reused-initializer ===")
    print("Two Conv ops sharing the same weight initializer.")

    weight_shape = [10, 10, 1, 1]  # 100 elements
    weight = numpy_helper.from_array(np.zeros(weight_shape, dtype=np.float32), name="W_shared")

    nodes = [
        oh.make_node("Conv", ["input", "W_shared"], ["mid"]),
        oh.make_node("Conv", ["mid", "W_shared"], ["output"]),
    ]
    vi = [oh.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 10, 30, 30])]
    m = _build_model_minimal(nodes, [weight], vi, opset=10)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(m, f.name)
        path = f.name

    macs, mem, params = score(path)
    print(f"  Two Convs sharing W_shared (100 elements): macs={macs}, memory={mem}, params={params}")
    print(f"  RESULT: params should be 100 (count once), not 200.")
    if params == 100:
        print("  CONFIRMED:Confirmed: shared initializer counted once.")
    else:
        print(f"  Unexpected: params={params}")

    Path(path).unlink()


def probe_empty_shape_tensor():
    """Verify volume([]) = 0 in onnx-tool memory accounting."""
    print("\n=== probe: empty-shape (scalar) tensor ===")
    print("Build a graph with a scalar intermediate (shape []).")
    print("This is hard to construct legitimately in [1,10,30,30] -> [1,10,30,30] but we can use ReduceSum + scalar.")

    # ReduceSum reduces all dims if axes=None. Result is scalar [].
    nodes = [
        oh.make_node("ReduceSum", ["input"], ["scalar"], keepdims=0),
        # scalar can't be an output of [1,10,30,30] shape — need to broadcast back.
        # Skip making it the output; instead use it as part of a chain.
        oh.make_node("Identity", ["input"], ["output"]),  # no-op output
    ]
    vi = [oh.make_tensor_value_info("scalar", TensorProto.FLOAT, [])]
    m = _build_model_minimal(nodes, [], vi, opset=10)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(m, f.name)
        path = f.name

    macs, mem, params = score(path)
    print(f"  Scalar ReduceSum + Identity: macs={macs}, memory={mem}, params={params}")
    print(f"  RESULT: scalar 'scalar' tensor should add 0 bytes to memory (volume([]) = 0 per lessons.md).")
    # Memory floor for a 4-byte float scalar would be 4 bytes; if it's 0, the lesson is confirmed.
    if mem is not None and mem == 0:
        print("  CONFIRMED:Confirmed: scalar tensor adds 0 bytes (volume([]) = 0 per onnx-tool reference).")
    else:
        print(f"  Memory={mem}; lessons.md says volume([]) = 0; check if scalar contributed N>0 bytes.")

    Path(path).unlink()


def probe_constant_folding():
    """Test whether onnx-tool's constant_folding eliminates intermediates from memory."""
    print("\n=== probe: constant-folding chain ===")
    print("A chain Initializer -> Identity -> Identity -> Add(input, x). Intermediates should fold.")

    init = numpy_helper.from_array(np.zeros((1, 10, 30, 30), dtype=np.float32), name="K")

    nodes = [
        oh.make_node("Identity", ["K"], ["folded_1"]),
        oh.make_node("Identity", ["folded_1"], ["folded_2"]),
        oh.make_node("Add", ["input", "folded_2"], ["output"]),
    ]
    vi = [
        oh.make_tensor_value_info("folded_1", TensorProto.FLOAT, [1, 10, 30, 30]),
        oh.make_tensor_value_info("folded_2", TensorProto.FLOAT, [1, 10, 30, 30]),
    ]
    m = _build_model_minimal(nodes, [init], vi, opset=10)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(m, f.name)
        path = f.name

    macs, mem, params = score(path)
    print(f"  Identity-chain + Add: macs={macs}, memory={mem}, params={params}")
    print(f"  Without folding, memory would include 2x[1,10,30,30] = 72,000 bytes.")
    print(f"  With folding, only one intermediate (or none) should remain.")
    expected_no_fold = 2 * 1 * 10 * 30 * 30 * 4  # 72,000
    if mem is not None and mem < expected_no_fold * 0.6:
        print(f"  CONFIRMED:Folding observed: memory={mem} < {expected_no_fold} (no-fold baseline).")
    else:
        print(f"  Memory={mem}; constant_folding may not fold this pattern (Add depends on runtime input).")

    Path(path).unlink()


def probe_value_info_omission():
    """Test: if a tensor has NO entry in value_info (only inferred at runtime), does memory count it?"""
    print("\n=== probe: value_info omission ===")
    print("Build a graph WITHOUT explicit value_info for an intermediate.")
    print("The validator's calculate_memory walks `graph.input + graph.value_info + graph.output`.")
    print("If we omit value_info, will shape_inference fill it back?")

    nodes = [
        oh.make_node("Identity", ["input"], ["mid"]),
        oh.make_node("Identity", ["mid"], ["output"]),
    ]
    # NO value_info
    m = _build_model_minimal(nodes, [], [], opset=10)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(m, f.name)
        path = f.name

    macs, mem, params = score(path)
    print(f"  Identity chain WITHOUT value_info: macs={macs}, memory={mem}, params={params}")
    print(f"  Per validator code: shape_inference is run with strict_mode=True.")
    print(f"  Memory should include the inferred intermediate's bytes (36000 = 1*10*30*30*4).")
    if mem == 1 * 10 * 30 * 30 * 4:
        print("  CONFIRMED:Shape inference correctly fills value_info; memory counts intermediate.")
    else:
        print(f"  Memory={mem} (may differ if shape_inference didn't fully populate).")

    Path(path).unlink()


def probe_nested_initializers():
    """Test: do attribute-based tensors (e.g., Pad's `pads` at opset 10) count as params?"""
    print("\n=== probe: attribute-based tensor (Pad's pads at opset 10) ===")
    print("At opset 10, Pad takes pads via attribute, not as a graph input/initializer.")
    print("Per lessons.md: 'Pad's pads is opset-10 attribute and folds out' — verify zero params contribution.")

    nodes = [
        oh.make_node("Pad", ["input"], ["output"], pads=[0, 0, 0, 0, 0, 0, 0, 0], mode="constant"),
    ]
    m = _build_model_minimal(nodes, [], [], opset=10)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(m, f.name)
        path = f.name

    macs, mem, params = score(path)
    print(f"  Pad with attribute-based pads: macs={macs}, memory={mem}, params={params}")
    if params == 0:
        print("  CONFIRMED:Confirmed: attribute-based pads contribute 0 to params.")
    else:
        print(f"  Unexpected: params={params}; attribute may not be excluded.")

    Path(path).unlink()


PROBES = {
    "float16-intermediate": probe_float16_intermediate,
    "int8-initializer": probe_int8_initializer,
    "reused-initializer": probe_reused_initializer,
    "empty-shape": probe_empty_shape_tensor,
    "constant-folding": probe_constant_folding,
    "value-info-omission": probe_value_info_omission,
    "attribute-based-tensor": probe_nested_initializers,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=list(PROBES.keys()))
    args = ap.parse_args()

    if args.probe:
        PROBES[args.probe]()
    else:
        for name, fn in PROBES.items():
            try:
                fn()
            except Exception as e:
                print(f"\n=== probe: {name} ===")
                print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
