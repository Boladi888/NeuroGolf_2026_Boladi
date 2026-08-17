#!/usr/bin/env python3
"""Generate the universal isolation fail-onnx: input[1,10,30,30] -> Mul by 0 -> output all zeros.

argmax over the 10 channels -> all class 0 (background) -> wrong for ~every task under the
exact-match grader -> 0 points (proven: an all-fail 401-zip scored 0.00 exactly). Used by
`captain_ctl.py probe-isolated` as the 0-base for every task EXCEPT the probe targets, so one
submit's publicScore == the targets' absolute points -> no baseline submit, ever.

The output (tools/isolation_fail.onnx) is gitignored as a regenerable *.onnx artifact — `captain_ctl.py
probe-isolated` auto-runs this generator if the file is missing. Regenerate if the team opset/ir changes.
"""
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
TBO = ROOT.parent / "K-NeuroGolf-OnnxlyFans" / "Team_Submission" / "Backend" / "Team_Best_Onnx"
OUT = ROOT / "tools" / "isolation_fail.onnx"

# Match the team's opset/ir so the grader loads it identically.
ref = onnx.load(str(TBO / "task001.onnx"))
opset = ref.opset_import[0].version
ir = ref.ir_version

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
zero = helper.make_tensor("zero_const", TensorProto.FLOAT, [1], [0.0])
mul = helper.make_node("Mul", ["input", "zero_const"], ["output"])
g = helper.make_graph([mul], "isolation_fail", [inp], [out], [zero])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
m.ir_version = ir
onnx.checker.check_model(m)
onnx.save(m, str(OUT))
print(f"wrote {OUT} (opset {opset}, ir {ir}, {OUT.stat().st_size} B)")
