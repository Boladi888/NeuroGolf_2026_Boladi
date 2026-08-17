"""
NeuroGolf 2026 — Cluster #01 cost probes.

Reproduces every measured cost number in the J.01–J.18 notes.
Pinned deps: numpy==2.4.4, onnx==1.21.0, onnx_tool==1.0.1.

Usage:
    pip install onnx==1.21.0 onnx-tool==1.0.1 numpy --break-system-packages
    python probe_all.py

Each probe builds a minimal ONNX graph, profiles it via onnx-tool with
`constant_folding=True`, and prints the per-node table + total row.

Author: Cluster-1 research agent, May 2026.
"""

import io
import contextlib
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnx_tool


def profile(m: onnx.ModelProto, label: str) -> None:
    onnx.save(m, '_probe.onnx')
    print(f'\n=== {label} ===')
    try:
        mdl = onnx_tool.Model('_probe.onnx', {'constant_folding': True})
        mdl.graph.shape_infer({'input': np.zeros((1, 10, 30, 30), dtype=np.float32)})
        mdl.graph.profile()
    except Exception as e:
        print(f'  PROFILE FAILED: {type(e).__name__}: {e}')
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mdl.graph.print_node_map()
    print(buf.getvalue().rstrip())


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _opset(v=21):
    return [helper.make_opsetid('', v)]


def make_model(nodes, initializers, opset=21):
    inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])
    g = helper.make_graph(nodes, 'g', [inp], [out], initializers)
    m = helper.make_model(g, opset_imports=_opset(opset), ir_version=10)
    return m


# =======================================================================
# Cost-lever baselines (matches COVERED probe table)
# =======================================================================
def conv_baseline(k: int) -> onnx.ModelProto:
    W = numpy_helper.from_array(
        np.random.RandomState(0).randn(10, 10, k, k).astype(np.float32),
        name='W')
    pad = (k - 1) // 2
    n = helper.make_node('Conv', ['input', 'W'], ['output'],
                         kernel_shape=[k, k], pads=[pad, pad, pad, pad])
    return make_model([n], [W])


# =======================================================================
# J.01 — Bit-slicing
# =======================================================================
def j01_bitwise_and_int8() -> onnx.ModelProto:
    """Cast(int8) -> BitwiseAnd -> Cast(float). EXPECTED TO FAIL profile."""
    K = numpy_helper.from_array(np.array([1], dtype=np.int8), name='K')
    return make_model([
        helper.make_node('Cast', ['input'], ['x_i'], to=TensorProto.INT8),
        helper.make_node('BitwiseAnd', ['x_i', 'K'], ['x_a']),
        helper.make_node('Cast', ['x_a'], ['output'], to=TensorProto.FLOAT),
    ], [K])


def j01_bitshift_uint8() -> onnx.ModelProto:
    """Cast(uint8) -> BitShift RIGHT -> Cast(float). EXPECTED TO FAIL profile."""
    K = numpy_helper.from_array(np.array([1], dtype=np.uint8), name='K')
    return make_model([
        helper.make_node('Cast', ['input'], ['x_i'], to=TensorProto.UINT8),
        helper.make_node('BitShift', ['x_i', 'K'], ['x_s'], direction='RIGHT'),
        helper.make_node('Cast', ['x_s'], ['output'], to=TensorProto.FLOAT),
    ], [K])


def j01_arith_one_bit_plane() -> onnx.ModelProto:
    """Mul(input, 0.5) -> Floor -> Mod(2.0, fmod=1).

    Extracts bit-1 of an integer-valued float input. Pure float arithmetic,
    no Cast required. The viable bit-extraction path."""
    half = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name='half')
    two = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name='two')
    return make_model([
        helper.make_node('Mul', ['input', 'half'], ['x1']),
        helper.make_node('Floor', ['x1'], ['x2']),
        helper.make_node('Mod', ['x2', 'two'], ['output'], fmod=1),
    ], [half, two])


def j01_xor_two_halves_via_arith() -> onnx.ModelProto:
    """XOR-equivalent on 0/1 indicator inputs: a + b - 2*a*b.
    For ARC: slice left half, slice right half (both already free Slice),
    combine via this arithmetic. We model just the arithmetic part on full input."""
    two = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name='two')
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name='one')
    # Use input twice (a=b=input) -- diagnostic only
    return make_model([
        helper.make_node('Mul', ['input', 'input'], ['ab']),
        helper.make_node('Mul', ['ab', 'two'], ['ab2']),
        helper.make_node('Add', ['input', 'input'], ['apb']),
        helper.make_node('Sub', ['apb', 'ab2'], ['output']),
    ], [two, one])


# =======================================================================
# J.13 — Gray code (probed via the equivalent two-axis Gather construct)
# =======================================================================
def gray_code_perm(n: int) -> np.ndarray:
    """Length-n int32 permutation via 5-bit BRGC, truncated."""
    bits = (n - 1).bit_length()
    full = 1 << bits
    grays = np.array([i ^ (i >> 1) for i in range(full)], dtype=np.int32)
    # Sort by gray code rank, take first n that map < n
    order = np.argsort(grays).astype(np.int32)
    perm = order[order < n][:n]
    return perm.astype(np.int32)


def j13_two_axis_gray() -> onnx.ModelProto:
    """Apply Gray-code perm to row axis (2) then col axis (3)."""
    pr = numpy_helper.from_array(gray_code_perm(30), name='pr')
    pc = numpy_helper.from_array(gray_code_perm(30), name='pc')
    return make_model([
        helper.make_node('Gather', ['input', 'pr'], ['x1'], axis=2),
        helper.make_node('Gather', ['x1', 'pc'], ['output'], axis=3),
    ], [pr, pc])


# =======================================================================
# J.14 — Morton / Z-order
# =======================================================================
def morton_perm(H: int, W: int) -> np.ndarray:
    bits = max(H, W).bit_length()
    flat_to_morton = np.zeros(H * W, dtype=np.int32)
    for r in range(H):
        for c in range(W):
            m = 0
            for i in range(bits):
                m |= ((r >> i) & 1) << (2 * i)
                m |= ((c >> i) & 1) << (2 * i + 1)
            flat_to_morton[r * W + c] = m
    return np.argsort(flat_to_morton).astype(np.int32)


def j14_morton_oneway(dtype=np.int32) -> onnx.ModelProto:
    """Reshape -> Gather (Morton perm) -> Reshape."""
    perm = morton_perm(30, 30).astype(dtype)
    P = numpy_helper.from_array(perm, name='P')
    s1 = numpy_helper.from_array(np.array([1, 10, 900], dtype=np.int64), name='s1')
    s2 = numpy_helper.from_array(np.array([1, 10, 30, 30], dtype=np.int64), name='s2')
    return make_model([
        helper.make_node('Reshape', ['input', 's1'], ['x1']),
        helper.make_node('Gather', ['x1', 'P'], ['x2'], axis=2),
        helper.make_node('Reshape', ['x2', 's2'], ['output']),
    ], [P, s1, s2])


def j14_morton_roundtrip() -> onnx.ModelProto:
    """Reshape -> Gather (Morton) -> Gather (inverse Morton) -> Reshape.
    Should be identity if perms are inverse — useful for sanity check only."""
    perm = morton_perm(30, 30)
    inv = np.argsort(perm).astype(np.int32)
    P = numpy_helper.from_array(perm, name='P')
    Pinv = numpy_helper.from_array(inv, name='Pinv')
    s1 = numpy_helper.from_array(np.array([1, 10, 900], dtype=np.int64), name='s1')
    s2 = numpy_helper.from_array(np.array([1, 10, 30, 30], dtype=np.int64), name='s2')
    return make_model([
        helper.make_node('Reshape', ['input', 's1'], ['x1']),
        helper.make_node('Gather', ['x1', 'P'], ['x2'], axis=2),
        helper.make_node('Gather', ['x2', 'Pinv'], ['x3'], axis=2),
        helper.make_node('Reshape', ['x3', 's2'], ['output']),
    ], [P, Pinv, s1, s2])


def j14_morton_via_range() -> onnx.ModelProto:
    """Same as oneway but with Range (folds to constant)."""
    zero = numpy_helper.from_array(np.array(0, dtype=np.int64), name='zero')
    n900 = numpy_helper.from_array(np.array(900, dtype=np.int64), name='n900')
    one = numpy_helper.from_array(np.array(1, dtype=np.int64), name='one')
    s1 = numpy_helper.from_array(np.array([1, 10, 900], dtype=np.int64), name='s1')
    s2 = numpy_helper.from_array(np.array([1, 10, 30, 30], dtype=np.int64), name='s2')
    return make_model([
        helper.make_node('Range', ['zero', 'n900', 'one'], ['idx']),
        helper.make_node('Reshape', ['input', 's1'], ['x1']),
        helper.make_node('Gather', ['x1', 'idx'], ['x2'], axis=2),
        helper.make_node('Reshape', ['x2', 's2'], ['output']),
    ], [zero, n900, one, s1, s2])


# =======================================================================
# J.16 — RNS (single residue)
# =======================================================================
def j16_rns_one_residue(modulus=11) -> onnx.ModelProto:
    """Cast(int32) -> Mod(modulus) -> Cast(float)."""
    M = numpy_helper.from_array(np.array([modulus], dtype=np.int32), name='M')
    return make_model([
        helper.make_node('Cast', ['input'], ['xi'], to=TensorProto.INT32),
        helper.make_node('Mod', ['xi', 'M'], ['xm']),
        helper.make_node('Cast', ['xm'], ['output'], to=TensorProto.FLOAT),
    ], [M])


def j16_mod_float_parity() -> onnx.ModelProto:
    """Direct Mod(input, 2.0, fmod=1) -- the only useful single-Mod usage."""
    two = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name='two')
    return make_model([
        helper.make_node('Mod', ['input', 'two'], ['output'], fmod=1),
    ], [two])


# =======================================================================
# Run all
# =======================================================================
if __name__ == '__main__':
    print('Cost-lever baselines')
    profile(conv_baseline(1), 'Conv 1x1 baseline')
    profile(conv_baseline(3), 'Conv 3x3 baseline')
    profile(conv_baseline(5), 'Conv 5x5 baseline')

    print('\n\nJ.01 — Bit-slicing')
    profile(j01_bitwise_and_int8(), 'BitwiseAnd via int8 cast (expect FAIL)')
    profile(j01_bitshift_uint8(), 'BitShift via uint8 cast (expect FAIL)')
    profile(j01_arith_one_bit_plane(), 'Mul+Floor+Mod (one bit plane via float arith)')
    profile(j01_xor_two_halves_via_arith(), 'XOR via float arith (a+b-2ab)')

    print('\n\nJ.13 — Gray code')
    profile(j13_two_axis_gray(), 'Two-axis Gray-code Gather')

    print('\n\nJ.14 — Morton / Z-order')
    profile(j14_morton_oneway(np.int32), 'Morton perm one-way (int32 indices)')
    profile(j14_morton_oneway(np.int64), 'Morton perm one-way (int64 indices, for comparison)')
    profile(j14_morton_roundtrip(), 'Morton roundtrip (Gather + inverse Gather)')
    profile(j14_morton_via_range(), 'Morton via Range (constant-folded)')

    print('\n\nJ.16 — RNS')
    profile(j16_rns_one_residue(11), 'RNS one residue (Cast+Mod(11)+Cast)')
    profile(j16_mod_float_parity(), 'Mod(input, 2.0) float parity (single op)')

    print('\n\nJ.18 — Stochastic computing: NOT PROBED.')
    print('  Reason: variance + onnx-tool MAC accounting both make SC strictly worse.')
    print('  See notes/j18-stochastic-computing.md for the formal argument.')
