"""Unit tests for the 4 spread_patterns added 2026-05-19.

Focus: each pattern's apply() returns the right verdict on a
hand-constructed minimal ONNX exhibiting the targeted structure, and
returns no-op when the structure is absent.
"""
from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.spread_patterns.pass_add_chain_to_sum import PATTERN as P_ADD
from tools.spread_patterns.pass_equal_fold_distance import PATTERN as P_EQ
from tools.spread_patterns.pass_final_cast_f16_to_f32_advisory import PATTERN as P_F16
from tools.spread_patterns.pass_mul_by_all_ones_drop import PATTERN as P_MUL


# ---- helpers ---------------------------------------------------------


def _model(nodes, inits, inputs, outputs, opset=13):
    g = helper.make_graph(nodes, "test", inputs, outputs, inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])


def _f32_input(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _bool_output(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.BOOL, shape)


def _f16_value_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)


def _f32_output(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


# ---- pass_add_chain_to_sum ------------------------------------------


def test_add_chain_to_sum_collapses_three_adds():
    """a+b+c+d via 3 chained Adds → single Sum(a,b,c,d)."""
    nodes = [
        helper.make_node("Add", ["a", "b"], ["t1"]),
        helper.make_node("Add", ["t1", "c"], ["t2"]),
        helper.make_node("Add", ["t2", "d"], ["out"]),
    ]
    m = _model(
        nodes, [],
        [_f32_input(n, [1]) for n in ("a", "b", "c", "d")],
        [_f32_output("out", [1])],
    )
    r = P_ADD.apply(m, "task_test")
    assert r.verdict == "applied"
    sums = [n for n in r.new_model.graph.node if n.op_type == "Sum"]
    adds = [n for n in r.new_model.graph.node if n.op_type == "Add"]
    assert len(sums) == 1
    assert len(adds) == 0
    assert list(sums[0].input) == ["a", "b", "c", "d"]
    assert sums[0].output[0] == "out"
    assert r.extras["intermediates_eliminated"] == 2  # t1, t2


def test_add_chain_to_sum_skips_single_add():
    nodes = [helper.make_node("Add", ["a", "b"], ["out"])]
    m = _model(nodes, [],
               [_f32_input("a", [1]), _f32_input("b", [1])],
               [_f32_output("out", [1])])
    r = P_ADD.apply(m, "task_test")
    assert r.verdict == "no-op"


def test_add_chain_to_sum_preserves_extra_consumer():
    """When an intermediate tensor has multiple consumers, the chain head
    moves PAST that point but the producing Add for that tensor stays
    alive. The collapse only drops INTERMEDIATE outputs that have a
    single consumer. Math: associativity of Add is preserved regardless
    of how many other ops read the operands."""
    nodes = [
        helper.make_node("Add", ["a", "b"], ["t1"]),
        helper.make_node("Add", ["t1", "c"], ["t2"]),
        # t1 also feeds a separate Mul. Add(a,b)→t1 must SURVIVE.
        helper.make_node("Mul", ["t1", "x"], ["side"]),
        helper.make_node("Add", ["t2", "side"], ["out"]),
    ]
    m = _model(
        nodes, [],
        [_f32_input(n, [1]) for n in ("a", "b", "c", "x")],
        [_f32_output("out", [1])],
    )
    r = P_ADD.apply(m, "task_test")
    assert r.verdict == "applied"
    ops_in_order = [(n.op_type, list(n.input), list(n.output)) for n in r.new_model.graph.node]
    # The first Add(a, b) → t1 must survive (Mul still uses t1).
    assert ("Add", ["a", "b"], ["t1"]) in ops_in_order
    # The Mul(t1, x) → side must survive.
    assert ("Mul", ["t1", "x"], ["side"]) in ops_in_order
    # t2's Add and the final Add must collapse to a single Sum(t1, c, side).
    sums = [n for n in r.new_model.graph.node if n.op_type == "Sum"]
    assert len(sums) == 1
    assert list(sums[0].input) == ["t1", "c", "side"]
    assert sums[0].output[0] == "out"
    # Exactly one intermediate was eliminated: t2.
    assert r.extras["intermediates_eliminated"] == 1


# ---- pass_equal_fold_distance_threshold -----------------------------


def test_equal_fold_collapses_sub_mul_less():
    """Sub(x,y) → Mul(t,t) → Less(eps<1) → Equal(x,y)."""
    eps = numpy_helper.from_array(np.array(0.25, dtype=np.float32), name="eps")
    nodes = [
        helper.make_node("Sub", ["x", "y"], ["diff"]),
        helper.make_node("Mul", ["diff", "diff"], ["sq"]),
        helper.make_node("Less", ["sq", "eps"], ["is_equal"]),
    ]
    m = _model(
        nodes, [eps],
        [_f32_input("x", [10]), _f32_input("y", [10])],
        [_bool_output("is_equal", [10])],
    )
    r = P_EQ.apply(m, "task_test")
    assert r.verdict == "applied"
    new_ops = [n.op_type for n in r.new_model.graph.node]
    assert "Equal" in new_ops
    assert "Sub" not in new_ops
    assert "Mul" not in new_ops
    assert "Less" not in new_ops
    equal_nodes = [n for n in r.new_model.graph.node if n.op_type == "Equal"]
    assert equal_nodes[0].output[0] == "is_equal"  # output name preserved


def test_equal_fold_rejects_eps_above_one():
    """Sub→Mul→Less(2.0) is NOT a one-hot construction; reject."""
    eps = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="eps")
    nodes = [
        helper.make_node("Sub", ["x", "y"], ["diff"]),
        helper.make_node("Mul", ["diff", "diff"], ["sq"]),
        helper.make_node("Less", ["sq", "eps"], ["out"]),
    ]
    m = _model(
        nodes, [eps],
        [_f32_input("x", [10]), _f32_input("y", [10])],
        [_bool_output("out", [10])],
    )
    r = P_EQ.apply(m, "task_test")
    assert r.verdict == "no-op"


def test_equal_fold_rejects_mul_with_different_inputs():
    """Mul(a, b) with a != b is not self-squaring; reject."""
    eps = numpy_helper.from_array(np.array(0.25, dtype=np.float32), name="eps")
    nodes = [
        helper.make_node("Sub", ["x", "y"], ["diff"]),
        helper.make_node("Mul", ["diff", "y"], ["prod"]),  # not self-squared
        helper.make_node("Less", ["prod", "eps"], ["out"]),
    ]
    m = _model(
        nodes, [eps],
        [_f32_input("x", [10]), _f32_input("y", [10])],
        [_bool_output("out", [10])],
    )
    r = P_EQ.apply(m, "task_test")
    assert r.verdict == "no-op"


# ---- pass_mul_by_all_ones_drop --------------------------------------


def test_mul_by_all_ones_drops_node():
    ones = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), name="ones")
    nodes = [
        helper.make_node("Mul", ["x", "ones"], ["scaled"]),
        helper.make_node("Add", ["scaled", "x"], ["out"]),
    ]
    m = _model(
        nodes, [ones],
        [_f32_input("x", [1, 4])],
        [_f32_output("out", [1, 4])],
    )
    r = P_MUL.apply(m, "task_test")
    assert r.verdict == "applied"
    ops = [n.op_type for n in r.new_model.graph.node]
    assert "Mul" not in ops, f"Mul should have been dropped, got nodes {ops}"
    # Downstream Add should now consume "x" twice (rewired from "scaled")
    add = next(n for n in r.new_model.graph.node if n.op_type == "Add")
    assert "x" in add.input
    assert "scaled" not in add.input


def test_mul_by_all_ones_when_feeding_output_inserts_identity():
    """If the dropped Mul feeds a graph output, an Identity preserves the output name."""
    ones = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), name="ones")
    nodes = [helper.make_node("Mul", ["x", "ones"], ["out"])]
    m = _model(
        nodes, [ones],
        [_f32_input("x", [1, 4])],
        [_f32_output("out", [1, 4])],
    )
    r = P_MUL.apply(m, "task_test")
    assert r.verdict == "applied"
    ops = [n.op_type for n in r.new_model.graph.node]
    assert "Mul" not in ops
    assert "Identity" in ops  # output name preserved


def test_mul_by_non_ones_init_skipped():
    twos = numpy_helper.from_array(2.0 * np.ones((1, 4), dtype=np.float32), name="twos")
    nodes = [helper.make_node("Mul", ["x", "twos"], ["out"])]
    m = _model(
        nodes, [twos],
        [_f32_input("x", [1, 4])],
        [_f32_output("out", [1, 4])],
    )
    r = P_MUL.apply(m, "task_test")
    assert r.verdict == "no-op"


# ---- pass_final_cast_f16_to_f32_advisory ----------------------------


def test_final_cast_advisory_flags_f16_to_f32_terminal():
    nodes = [
        helper.make_node("Cast", ["t_f16"], ["out"], to=TensorProto.FLOAT),
    ]
    # Input is declared as F16 in value_info so the advisory can detect it.
    f16_in = _f16_value_info("t_f16", [1, 10, 30, 30])
    g = helper.make_graph(
        nodes, "test",
        [f16_in],  # graph input (so dtype is known)
        [_f32_output("out", [1, 10, 30, 30])],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    r = P_F16.apply(m, "task_test")
    assert r.verdict == "advisory"
    assert r.extras["tails_flagged"] == 1
    # Estimated savings: 1*10*30*30 elements * 2 bytes (F32-F16) = 18000
    assert r.extras["estimated_savings_bytes"] == 18000


def test_final_cast_advisory_skips_f32_input():
    """Cast(F32 → F32) is identity, not the target trick. Skip."""
    nodes = [helper.make_node("Cast", ["t"], ["out"], to=TensorProto.FLOAT)]
    g = helper.make_graph(
        nodes, "test",
        [_f32_input("t", [1, 10, 30, 30])],
        [_f32_output("out", [1, 10, 30, 30])],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    r = P_F16.apply(m, "task_test")
    assert r.verdict == "no-op"
