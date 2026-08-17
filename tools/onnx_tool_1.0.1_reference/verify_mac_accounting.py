"""
Empirically verify onnx-tool 1.0.1's MAC/params accounting for the ops that
matter to the J.6/J.7/J.11/J.20/J.36 cluster.

Build a tiny ONNX model with each candidate op, profile it, print the result.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnx_tool
import io

SHAPE = [1, 10, 30, 30]
NUMEL = 1 * 10 * 30 * 30  # 9000

def make_input():
    return helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)

def make_output(name="output", dtype=TensorProto.FLOAT, shape=SHAPE):
    return helper.make_tensor_value_info(name, dtype, shape)

def profile_model(model, label):
    """Save model, load with onnx-tool, print profile."""
    buf = io.BytesIO()
    onnx.save(model, buf)
    buf.seek(0)
    # Reload from bytes
    m = onnx.load_from_string(buf.getvalue())
    # Use onnx-tool's API (mirror of how the validator calls it)
    g = onnx_tool.Model(m).graph
    g.shape_infer()
    g.profile()
    macs = int(g.macs[0]) if isinstance(g.macs, (list, tuple)) else int(g.macs)
    print(f"{label:40s}  macs={macs:>10}  params={int(g.params):>10}  memory={int(g.memory):>10}")
    return macs, int(g.params), int(g.memory)


def model_with_single_node(op_type, inputs, outputs, init=None, attrs=None, output_dtype=TensorProto.FLOAT, output_shape=None):
    if output_shape is None:
        output_shape = SHAPE
    nodes = [helper.make_node(op_type, inputs, outputs, **(attrs or {}))]
    inits = [init] if init is not None else []
    inputs_list = [helper.make_tensor_value_info(n, TensorProto.FLOAT, SHAPE) for n in inputs if n == "input"]
    extra_inputs = [n for n in inputs if n != "input" and not any(i.name == n for i in inits)]
    g = helper.make_graph(
        nodes,
        "test",
        inputs_list,
        [helper.make_tensor_value_info(outputs[0], output_dtype, output_shape)],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m


# Identity baseline
m = model_with_single_node("Identity", ["input"], ["output"])
profile_model(m, "Identity (baseline)")

# Mul against scalar initializer
scalar_init = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="scalar")
m = model_with_single_node("Mul", ["input", "scalar"], ["output"], init=scalar_init)
profile_model(m, "Mul (input * scalar)")

# Add against scalar
m = model_with_single_node("Add", ["input", "scalar"], ["output"], init=scalar_init)
profile_model(m, "Add (input + scalar)")

# Sub
m = model_with_single_node("Sub", ["input", "scalar"], ["output"], init=scalar_init)
profile_model(m, "Sub (input - scalar)")

# Mod (need fmod for float)
m = model_with_single_node("Mod", ["input", "scalar"], ["output"], init=scalar_init, attrs={"fmod": 1})
profile_model(m, "Mod fmod=1 (input % scalar)")

# Cast
m = model_with_single_node("Cast", ["input"], ["output"], attrs={"to": TensorProto.INT8}, output_dtype=TensorProto.INT8)
profile_model(m, "Cast (float32 -> int8)")

# Greater
m = model_with_single_node("Greater", ["input", "scalar"], ["output"], init=scalar_init,
                            output_dtype=TensorProto.BOOL)
profile_model(m, "Greater (input > scalar)")

# Equal
m = model_with_single_node("Equal", ["input", "scalar"], ["output"], init=scalar_init,
                            output_dtype=TensorProto.BOOL)
profile_model(m, "Equal (input == scalar)")

# Where: cond, a, b -- need a bool cond
# We'll Greater first, then Where. Build a 2-node graph.
def model_where_pipeline():
    scalar = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="thresh")
    val_a = numpy_helper.from_array(np.zeros(SHAPE, dtype=np.float32), name="a")
    val_b = numpy_helper.from_array(np.ones(SHAPE, dtype=np.float32), name="b")
    g = helper.make_graph(
        [
            helper.make_node("Greater", ["input", "thresh"], ["mask"]),
            helper.make_node("Where", ["mask", "a", "b"], ["output"]),
        ],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        [scalar, val_a, val_b],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m

profile_model(model_where_pipeline(), "Greater + Where pipeline")

# Where alone -- skip Greater, use a precomputed bool init
def model_where_only():
    bool_init = numpy_helper.from_array(np.ones(SHAPE, dtype=np.bool_), name="cond")
    val_b = numpy_helper.from_array(np.ones(SHAPE, dtype=np.float32), name="b")
    g = helper.make_graph(
        [helper.make_node("Where", ["cond", "input", "b"], ["output"])],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        [bool_init, val_b],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m

profile_model(model_where_only(), "Where only (with bool init)")

# Hardmax
m = model_with_single_node("Hardmax", ["input"], ["output"], attrs={"axis": 1})
profile_model(m, "Hardmax (axis=1)")

# ArgMax (output dtype is int64, shape collapses one dim)
m = model_with_single_node("ArgMax", ["input"], ["output"],
                            attrs={"axis": 1, "keepdims": 0},
                            output_dtype=TensorProto.INT64,
                            output_shape=[1, 30, 30])
profile_model(m, "ArgMax (axis=1, no keepdim)")

# OneHot: indices [1,30,30] int64 -> output [1,30,30,10] float32
def model_onehot():
    depth_init = numpy_helper.from_array(np.array(10, dtype=np.int64), name="depth")
    values_init = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="values")
    g = helper.make_graph(
        [
            helper.make_node("ArgMax", ["input"], ["idx"], axis=1, keepdims=0),
            helper.make_node("OneHot", ["idx", "depth", "values"], ["output"], axis=-1),
        ],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 30, 30, 10])],
        [depth_init, values_init],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m

profile_model(model_onehot(), "ArgMax + OneHot pipeline")

# Gather along channel axis (color-perm)
def model_gather():
    perm = numpy_helper.from_array(np.array([1,0,2,3,4,5,6,7,8,9], dtype=np.int64), name="perm")
    g = helper.make_graph(
        [helper.make_node("Gather", ["input", "perm"], ["output"], axis=1)],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        [perm],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m

profile_model(model_gather(), "Gather (color-perm via axis=1)")

# TopK
def model_topk():
    k_init = numpy_helper.from_array(np.array([3], dtype=np.int64), name="k")
    g = helper.make_graph(
        [helper.make_node("TopK", ["input", "k"], ["values", "indices"], axis=1)],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [
            helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 3, 30, 30]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 3, 30, 30]),
        ],
        [k_init],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m

profile_model(model_topk(), "TopK (k=3, axis=1)")

# ReduceSum on spatial axes
m = model_with_single_node("ReduceSum", ["input", "axes"], ["output"],
                            init=numpy_helper.from_array(np.array([2,3], dtype=np.int64), name="axes"),
                            attrs={"keepdims": 1},
                            output_shape=[1, 10, 1, 1])
profile_model(m, "ReduceSum (spatial axes)")

# Conv 1x1 baseline
def model_conv_1x1():
    w = numpy_helper.from_array(np.zeros((10, 10, 1, 1), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((10,), dtype=np.float32), name="B")
    g = helper.make_graph(
        [helper.make_node("Conv", ["input", "W", "B"], ["output"],
                           kernel_shape=[1,1], pads=[0,0,0,0], strides=[1,1])],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        [w, b],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m

profile_model(model_conv_1x1(), "Conv 1x1 (10->10 channel)")
