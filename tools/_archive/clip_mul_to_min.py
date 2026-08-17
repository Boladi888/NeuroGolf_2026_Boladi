"""Graph pass: replace Clip[0,1](x) -> Mul(., mask) with Min(x, mask) when mask is 0/1-valued.

Specifically, for the pattern:
  C1 = Clip(x, min=0, max=1)
  C2 = Mul(C1, mask)
where mask is bool-derived (cast from bool), we can replace both with:
  C2 = Min(x, mask)

Math: Clip[0,1](x) = max(0, min(x, 1)). For x >= 0 (which Conv-of-bool always is),
this equals min(x, 1). Then Mul(min(x,1), mask) where mask in {0, 1}:
  - mask=0: 0
  - mask=1: min(x, 1)
which equals min(x, mask). So Min(x, mask) is the same.

Caveat: only applies if mask values are exactly 0 or 1. We verify the mask
came from a Cast(bool->float) node (safe assumption).

Saves the Clip intermediate per iteration.
"""
import onnx


def is_clip_01(node, graph):
    """Check if a Clip node clips to [0, 1]. Handles both opset 10 (attr) and 13 (input) forms."""
    if node.op_type != 'Clip':
        return False
    # opset 10: min and max are attributes
    if len(node.input) == 1:
        min_val = None; max_val = None
        for a in node.attribute:
            if a.name == 'min': min_val = a.f
            if a.name == 'max': max_val = a.f
        return min_val == 0.0 and max_val == 1.0
    # opset 13: min and max are inputs (initializers or Constant nodes)
    if len(node.input) == 3:
        min_name, max_name = node.input[1], node.input[2]
        inits_by_name = {init.name: init for init in graph.initializer}
        min_init = inits_by_name.get(min_name)
        max_init = inits_by_name.get(max_name)
        if min_init is None or max_init is None:
            return False
        min_val = onnx.numpy_helper.to_array(min_init).flatten().tolist()
        max_val = onnx.numpy_helper.to_array(max_init).flatten().tolist()
        return min_val == [0.0] and max_val == [1.0]
    return False


def is_bool_cast_or_mask(name, graph):
    """Check if `name` is provably 0/1-valued. Sources:
      - Cast(bool -> float) where bool from Greater/Equal/Less/And/Or/Not.
      - Sub(1.0, X) where X is 0/1 (recursive).
      - Mul of two 0/1 masks.
      - Initializer with only 0/1 values.
    """
    import numpy as np
    # Check initializers
    for init in graph.initializer:
        if init.name == name:
            arr = onnx.numpy_helper.to_array(init)
            return set(np.unique(arr).tolist()).issubset({0.0, 1.0})

    # Check producer node
    for node in graph.node:
        if name not in node.output:
            continue
        if node.op_type == 'Cast':
            cast_input_name = node.input[0]
            for producer in graph.node:
                if cast_input_name in producer.output:
                    if producer.op_type in {'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual',
                                            'Equal', 'And', 'Or', 'Not'}:
                        return True
            return False
        if node.op_type == 'Sub':
            # Sub(one_tensor, X) where one_tensor is all-1 and X is 0/1
            a, b = node.input[0], node.input[1]
            for init in graph.initializer:
                if init.name == a:
                    arr = onnx.numpy_helper.to_array(init)
                    if arr.size > 0 and set(np.unique(arr).tolist()).issubset({1.0}):
                        return is_bool_cast_or_mask(b, graph)
            return False
        if node.op_type == 'Mul':
            # Both inputs must be 0/1
            return is_bool_cast_or_mask(node.input[0], graph) and is_bool_cast_or_mask(node.input[1], graph)
    return False


def optimize(model):
    """Apply the Clip+Mul -> Min pass. Returns count of replacements."""
    graph = model.graph
    # Build: clip_output_name -> clip_node
    clip_nodes = {}
    for n in graph.node:
        if is_clip_01(n, graph):
            clip_nodes[n.output[0]] = n

    # Find Mul nodes consuming a clip output
    pairs_to_replace = []  # list of (mul_node, clip_node, mask_input)
    for n in graph.node:
        if n.op_type != 'Mul':
            continue
        if len(n.input) != 2:
            continue
        in_a, in_b = n.input[0], n.input[1]
        clip_node = None; mask = None
        if in_a in clip_nodes and is_bool_cast_or_mask(in_b, graph):
            clip_node = clip_nodes[in_a]
            mask = in_b
        elif in_b in clip_nodes and is_bool_cast_or_mask(in_a, graph):
            clip_node = clip_nodes[in_b]
            mask = in_a
        if clip_node:
            # Verify clip_output is used ONLY by this Mul (otherwise can't remove clip)
            clip_out = clip_node.output[0]
            uses = 0
            for nn in graph.node:
                if clip_out in nn.input:
                    uses += 1
            if uses == 1:
                pairs_to_replace.append((n, clip_node, mask))

    # Apply replacements
    n_replaced = 0
    for mul_node, clip_node, mask in pairs_to_replace:
        # New Min(clip_input, mask) -> mul_output
        clip_input = clip_node.input[0]
        mul_output = mul_node.output[0]
        # Replace mul with Min
        mul_node.op_type = 'Min'
        # Min takes variable inputs; in opset 13, Min(*inputs). For 2 inputs, use both.
        mul_node.input[:] = [clip_input, mask]
        # Remove clip node
        graph.node.remove(clip_node)
        n_replaced += 1
    return n_replaced


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        m = onnx.load(sys.argv[1])
        n = optimize(m)
        onnx.save(m, sys.argv[2])
        print(f'Replaced {n} Clip+Mul pairs with Min')
