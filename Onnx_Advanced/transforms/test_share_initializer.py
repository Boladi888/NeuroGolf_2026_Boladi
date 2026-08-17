"""Test for share_initializer.share_initializers. Run:
    .venv\\Scripts\\python.exe Onnx_Advanced\\transforms\\test_share_initializer.py
"""
import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from share_initializer import share_initializers  # noqa: E402


def build():
    """X + A + B + C where A and B are byte-identical (B is the redundant duplicate), C differs."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    A = numpy_helper.from_array(a, name="A")
    B = numpy_helper.from_array(a.copy(), name="B")             # identical bytes to A -> mergeable
    C = numpy_helper.from_array(np.ones((2, 3), np.float32), name="C")  # different -> keep
    nodes = [helper.make_node("Add", ["X", "A"], ["t"]),
             helper.make_node("Add", ["t", "B"], ["u"]),
             helper.make_node("Add", ["u", "C"], ["Y"])]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    g = helper.make_graph(nodes, "g", [X], [Y], [A, B, C])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])


def main() -> int:
    m = build()
    before = len(m.graph.initializer)
    m2, rep = share_initializers(m)

    assert rep["merged"] == 1, rep
    assert rep["removed_names"] == ["B"], rep
    assert len(m2.graph.initializer) == before - 1
    names = {i.name for i in m2.graph.initializer}
    assert "B" not in names and "A" in names and "C" in names, names
    assert m2.graph.node[1].input[1] == "A", "B reference must repoint to A"

    # negative: a graph with no duplicates is untouched
    solo = build()
    del solo.graph.initializer[1]  # drop B
    solo.graph.node[1].input[1] = "A"
    _, rep0 = share_initializers(solo)
    assert rep0["merged"] == 0, rep0

    # ORT parity: shared model computes the same output as the original
    try:
        import onnxruntime as ort
        x = np.random.rand(2, 3).astype(np.float32)
        r1 = ort.InferenceSession(build().SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
        r2 = ort.InferenceSession(m2.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
        assert np.allclose(r1, r2), (r1, r2)
        parity = "ORT parity OK"
    except ImportError:
        parity = "(onnxruntime absent; parity check skipped)"

    print(f"share_initializer test PASS: merged={rep['merged']} removed={rep['removed_names']}; {parity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
