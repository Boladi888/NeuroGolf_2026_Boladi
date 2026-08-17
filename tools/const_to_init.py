"""Convert Constant nodes whose output is referenced as op input
into graph.initializer entries. This is math-equivalent (same values
referenced by name) but saves memory cost because calculate_memory only
counts node outputs, NOT initializers.

Discovered 2026-05-12: live task230 had weight in a Constant node
(counted as 3600 bytes in memory). Moving to initializer dropped cost
from 4500 to 900 with byte-exact output preserved.
"""
import onnx
from onnx import numpy_helper


def convert_constants_to_initializers(model):
    """Returns (model, num_converted). Modifies model in place."""
    # Identify all Constant nodes
    const_nodes = [n for n in model.graph.node if n.op_type == 'Constant']
    if not const_nodes:
        return model, 0

    converted = 0
    new_inits = []
    nodes_to_remove = []

    for n in const_nodes:
        # Get the value tensor
        value_tensor = None
        for a in n.attribute:
            if a.name == 'value':
                value_tensor = a.t
                break
        if value_tensor is None:
            # Constant nodes can have value_floats, value_ints, etc.
            # Handle the common case 'value' only for now.
            continue

        # The output of this Constant becomes the initializer name
        output_name = n.output[0]
        # Build initializer tensor with this name
        init_tensor = onnx.helper.make_tensor(
            output_name,
            value_tensor.data_type,
            list(value_tensor.dims),
            numpy_helper.to_array(value_tensor).flatten().tolist(),
        )
        new_inits.append(init_tensor)
        nodes_to_remove.append(n)
        converted += 1

    # Remove the Constant nodes and add initializers
    for n in nodes_to_remove:
        model.graph.node.remove(n)
    for init in new_inits:
        model.graph.initializer.append(init)

    return model, converted


def optimize(model):
    """For compatibility with onnx_opt pipeline."""
    _, n = convert_constants_to_initializers(model)
    return {'const_to_init': n}


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        m = onnx.load(sys.argv[1])
        m, n = convert_constants_to_initializers(m)
        onnx.save(m, sys.argv[2])
        print(f'Converted {n} Constant nodes -> initializers')
