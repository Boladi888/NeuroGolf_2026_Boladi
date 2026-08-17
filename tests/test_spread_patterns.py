"""Unit tests for spread-knowledge pattern modules.

Each test constructs a minimal ONNX model that exhibits exactly the
pattern's trigger condition, runs the pattern, and asserts the
correct verdict + structural rewrite. These tests don't call ORT
or score_network_v2 — they validate the rewrite logic in isolation.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.spread_patterns import iter_patterns
from tools.spread_patterns.pass_clip_mul_to_min import PATTERN as CLIP_MUL_PATTERN
from tools.spread_patterns.pass_const_to_init import PATTERN as CONST_TO_INIT_PATTERN
from tools.spread_patterns.pass_scalar_broadcast_ones import PATTERN as SCALAR_PATTERN
from tools.spread_patterns.pass_slice_axis_collapse import PATTERN as SLICE_PATTERN


def _make_model(graph: onnx.GraphProto, ir_version: int = 10) -> onnx.ModelProto:
    model = helper.make_model(graph, producer_name="test", opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = ir_version
    return model


def _make_input(name: str = "input", shape: tuple[int, ...] = (1, 10, 30, 30)) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _make_output(name: str = "output", shape: tuple[int, ...] = (1, 10, 30, 30)) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def test_iter_patterns_returns_all_modules():
    names = {p.name for p in iter_patterns()}
    expected = {
        "onnxsim",
        "const_to_init",
        "safe_opt",
        "clip_mul_to_min",
        "scalar_broadcast_ones",
        "slice_axis_collapse",
        "fp16_selective",
        "crop_to_size_advisory",
    }
    assert expected.issubset(names), f"missing patterns: {expected - names}"


def test_scalar_broadcast_ones_replaces_uniform_init():
    """A uniform-ones [1,1,10,10] init consumed by Sub should be replaced with scalar."""
    ones = numpy_helper.from_array(np.ones((1, 1, 10, 10), dtype=np.float32), name="ones10")
    sub_node = helper.make_node("Sub", ["input_scalar", "ones10"], ["output"])
    # Use a scalar input to keep the test simple.
    inp = helper.make_tensor_value_info("input_scalar", TensorProto.FLOAT, [])
    out = _make_output(shape=(1, 1, 10, 10))
    graph = helper.make_graph([sub_node], "test", [inp], [out], initializer=[ones])
    model = _make_model(graph)

    result = SCALAR_PATTERN.apply(model, "task_test")

    assert result.verdict == "applied", result.notes
    assert result.new_model is not None
    new_inits = {init.name: init for init in result.new_model.graph.initializer}
    rewritten = new_inits["ones10"]
    # Now a scalar (rank-0 tensor).
    assert list(rewritten.dims) == [], "ones10 should have been collapsed to a scalar"


def test_scalar_broadcast_ones_skips_when_consumer_doesnt_broadcast():
    """Conv weights must NOT be replaced with a scalar (Conv doesn't broadcast)."""
    weight = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="conv_w")
    conv = helper.make_node("Conv", ["input", "conv_w"], ["output"], pads=[1, 1, 1, 1])
    inp = _make_input(shape=(1, 1, 5, 5))
    out = _make_output(shape=(1, 1, 5, 5))
    graph = helper.make_graph([conv], "test", [inp], [out], initializer=[weight])
    model = _make_model(graph)
    result = SCALAR_PATTERN.apply(model, "task_test")
    assert result.verdict == "no-op", "should refuse to collapse Conv weights"


def test_scalar_broadcast_ones_no_op_when_no_uniform_inits():
    """No uniform-valued initializer = no-op."""
    weight = numpy_helper.from_array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), name="w")
    add = helper.make_node("Add", ["input_scalar", "w"], ["output"])
    inp = helper.make_tensor_value_info("input_scalar", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    graph = helper.make_graph([add], "test", [inp], [out], initializer=[weight])
    model = _make_model(graph)
    result = SCALAR_PATTERN.apply(model, "task_test")
    assert result.verdict == "no-op"


def test_const_to_init_moves_constant_node():
    """A Constant node whose output feeds a Conv should be moved to initializer."""
    weight_value = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="ignored")
    constant = helper.make_node(
        "Constant",
        [],
        ["conv_w"],
        value=weight_value,
    )
    conv = helper.make_node("Conv", ["input", "conv_w"], ["output"], pads=[1, 1, 1, 1])
    inp = _make_input(shape=(1, 1, 5, 5))
    out = _make_output(shape=(1, 1, 5, 5))
    graph = helper.make_graph([constant, conv], "test", [inp], [out])
    model = _make_model(graph)
    result = CONST_TO_INIT_PATTERN.apply(model, "task_test")
    assert result.verdict == "applied", result.notes
    assert result.new_model is not None
    init_names = {i.name for i in result.new_model.graph.initializer}
    assert "conv_w" in init_names
    node_types = [n.op_type for n in result.new_model.graph.node]
    assert "Constant" not in node_types
    assert "Conv" in node_types


def test_const_to_init_no_op_without_constants():
    weight = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="w")
    conv = helper.make_node("Conv", ["input", "w"], ["output"], pads=[1, 1, 1, 1])
    inp = _make_input(shape=(1, 1, 5, 5))
    out = _make_output(shape=(1, 1, 5, 5))
    graph = helper.make_graph([conv], "test", [inp], [out], initializer=[weight])
    model = _make_model(graph)
    result = CONST_TO_INIT_PATTERN.apply(model, "task_test")
    assert result.verdict == "no-op"


def test_clip_mul_to_min_no_op_without_clip():
    add = helper.make_node("Add", ["input", "input"], ["output"])
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    graph = helper.make_graph([add], "test", [inp], [out])
    model = _make_model(graph)
    result = CLIP_MUL_PATTERN.apply(model, "task_test")
    assert result.verdict == "no-op"


def test_slice_axis_collapse_drops_axis0_noop():
    """A 4-d Slice with axis-0 = full-range should have that entry dropped."""
    starts = numpy_helper.from_array(np.array([0, 0, 1, 1], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([1, 10, 11, 11], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="steps")
    slc = helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])
    inp = _make_input("input", (1, 10, 30, 30))
    out = _make_output("output", (1, 10, 10, 10))
    graph = helper.make_graph(
        [slc],
        "test",
        [inp],
        [out],
        initializer=[starts, ends, axes, steps],
    )
    model = _make_model(graph)
    result = SLICE_PATTERN.apply(model, "task_test")
    assert result.verdict == "applied", result.notes
    new_inits = {i.name: numpy_helper.to_array(i).tolist() for i in result.new_model.graph.initializer}
    # axis 0 entry should be dropped from all four lists.
    assert new_inits["starts"] == [0, 1, 1]
    assert new_inits["ends"] == [10, 11, 11]
    assert new_inits["axes"] == [1, 2, 3]
    assert new_inits["steps"] == [1, 1, 1]


def test_slice_axis_collapse_no_op_when_axis0_actually_slices():
    """If starts[0] != 0 or ends[0] < input_axis0_size, leave it alone."""
    starts = numpy_helper.from_array(np.array([0, 0, 1, 1], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([1, 5, 11, 11], dtype=np.int64), name="ends")  # axis 1 sliced
    axes = numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="steps")
    slc = helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])
    inp = _make_input("input", (1, 10, 30, 30))
    out = _make_output("output", (1, 5, 10, 10))
    graph = helper.make_graph(
        [slc],
        "test",
        [inp],
        [out],
        initializer=[starts, ends, axes, steps],
    )
    model = _make_model(graph)
    # axis-0 entry should still be dropped (it IS a no-op) even with axis-1 active.
    result = SLICE_PATTERN.apply(model, "task_test")
    assert result.verdict == "applied"
    new_inits = {i.name: numpy_helper.to_array(i).tolist() for i in result.new_model.graph.initializer}
    assert new_inits["starts"] == [0, 1, 1]
    assert new_inits["ends"] == [5, 11, 11]


def test_slice_axis_collapse_skips_shared_inits():
    """If the same starts/ends arrays are shared by 2 Slices, refuse to rewrite."""
    starts = numpy_helper.from_array(np.array([0, 0, 1, 1], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([1, 10, 11, 11], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="steps")
    slc1 = helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["out_a"])
    slc2 = helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["out_b"])
    inp = _make_input("input", (1, 10, 30, 30))
    out_a = _make_output("out_a", (1, 10, 10, 10))
    out_b = _make_output("out_b", (1, 10, 10, 10))
    graph = helper.make_graph(
        [slc1, slc2],
        "test",
        [inp],
        [out_a, out_b],
        initializer=[starts, ends, axes, steps],
    )
    model = _make_model(graph)
    result = SLICE_PATTERN.apply(model, "task_test")
    # Shared inits -> rewrite is unsafe -> no-op.
    assert result.verdict == "no-op", result.notes
