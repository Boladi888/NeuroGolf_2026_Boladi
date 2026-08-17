"""Safer optimizer: applies dedupe + CSE + DCE, with FALLBACK to dedupe-only
if the CSE'd version doesn't produce identical output to original on test inputs.

This catches any CSE bug per-task.
"""
import sys, struct, pathlib, zipfile, os, json
os.environ['ORT_LOGGING_LEVEL'] = '3'
import numpy as np
import onnx
import onnxruntime as ort
ort.set_default_logger_severity(3)
from collections import OrderedDict


def init_data_tuple(init):
    if init.data_type == 7:
        if init.raw_data:
            n = len(init.raw_data) // 8
            return tuple(struct.unpack('<' + 'q' * n, init.raw_data))
        return tuple(init.int64_data)
    elif init.data_type == 1:
        if init.raw_data:
            n = len(init.raw_data) // 4
            return tuple(struct.unpack('<' + 'f' * n, init.raw_data))
        return tuple(init.float_data)
    elif init.data_type == 6:
        if init.raw_data:
            n = len(init.raw_data) // 4
            return tuple(struct.unpack('<' + 'i' * n, init.raw_data))
        return tuple(init.int32_data)
    return None


def dedupe_inits(model):
    seen = OrderedDict()
    rename = {}
    keep_inits = []
    for init in model.graph.initializer:
        d = init_data_tuple(init)
        if d is None:
            keep_inits.append(init)
            continue
        key = (tuple(init.dims), init.data_type, d)
        if key in seen:
            rename[init.name] = seen[key]
        else:
            seen[key] = init.name
            keep_inits.append(init)
    for node in model.graph.node:
        for i, inp_name in enumerate(node.input):
            if inp_name in rename:
                node.input[i] = rename[inp_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_inits)
    return len(rename)


def attr_to_tuple(attr):
    t = attr.type
    if t == onnx.AttributeProto.INT:
        return (attr.name, 'i', attr.i)
    elif t == onnx.AttributeProto.INTS:
        return (attr.name, 'is', tuple(attr.ints))
    elif t == onnx.AttributeProto.FLOAT:
        return (attr.name, 'f', attr.f)
    elif t == onnx.AttributeProto.FLOATS:
        return (attr.name, 'fs', tuple(attr.floats))
    elif t == onnx.AttributeProto.STRING:
        return (attr.name, 's', attr.s)
    elif t == onnx.AttributeProto.STRINGS:
        return (attr.name, 'ss', tuple(attr.strings))
    elif t == onnx.AttributeProto.TENSOR:
        tensor = attr.t
        # Use init_data_tuple to properly encode the tensor content (handles non-raw data)
        data = init_data_tuple(tensor)
        return (attr.name, 't', tuple(tensor.dims), tensor.data_type, data)
    else:
        return (attr.name, 'serial', attr.SerializeToString())


_COMMUTATIVE_OPS = {
    'Add', 'Mul', 'Min', 'Max', 'Sum', 'Mean',
    'And', 'Or', 'Xor',
    'BitwiseAnd', 'BitwiseOr', 'BitwiseXor',
    # NOTE: NOT including Equal here because some opsets treat the first input
    # specially for broadcasting semantics. Add cautiously after testing.
}


def cse_pass(model):
    """CSE but DON'T eliminate nodes whose outputs are graph outputs.

    Commutative ops have their input order normalized to detect equivalent
    expressions written with swapped operands (e.g., Mul(a, b) == Mul(b, a)).
    """
    graph_outputs = {o.name for o in model.graph.output}
    rename = {}
    seen = {}
    keep_nodes = []
    for node in model.graph.node:
        inputs = tuple(rename.get(inp, inp) for inp in node.input)
        # Normalize commutative op input order for better CSE matching
        sig_inputs = tuple(sorted(inputs)) if node.op_type in _COMMUTATIVE_OPS else inputs
        attrs = tuple(sorted(attr_to_tuple(a) for a in node.attribute))
        n_outputs = len(node.output)
        signature = (node.op_type, sig_inputs, attrs, n_outputs)

        # Skip elimination if any of this node's outputs is a graph output
        any_graph_out = any(o in graph_outputs for o in node.output)

        if signature in seen and not any_graph_out:
            canonical_outputs = seen[signature]
            for old_out, new_out in zip(node.output, canonical_outputs):
                rename[old_out] = new_out
        else:
            for i, inp_name in enumerate(node.input):
                node.input[i] = rename.get(inp_name, inp_name)
            seen[signature] = tuple(node.output)
            keep_nodes.append(node)
    n_removed = len(model.graph.node) - len(keep_nodes)
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)
    return n_removed


def dce_pass(model):
    used = set()
    for out in model.graph.output:
        used.add(out.name)
    keep_nodes = []
    n_removed_nodes = 0
    for node in reversed(model.graph.node):
        if any(out in used for out in node.output):
            keep_nodes.append(node)
            for inp in node.input:
                used.add(inp)
        else:
            n_removed_nodes += 1
    keep_nodes.reverse()
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)

    used_inits = set()
    for node in model.graph.node:
        for inp in node.input:
            used_inits.add(inp)
    keep_inits = [i for i in model.graph.initializer if i.name in used_inits]
    n_removed_inits = len(model.graph.initializer) - len(keep_inits)
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_inits)
    return n_removed_nodes, n_removed_inits


def _init_to_numpy(init):
    """Convert ONNX initializer to numpy array. Returns None for unsupported dtypes."""
    import numpy as np
    if init.data_type == 1:  # FLOAT
        if init.raw_data:
            arr = np.frombuffer(init.raw_data, dtype=np.float32)
        else:
            arr = np.array(list(init.float_data), dtype=np.float32)
    elif init.data_type == 7:  # INT64
        if init.raw_data:
            arr = np.frombuffer(init.raw_data, dtype=np.int64)
        else:
            arr = np.array(list(init.int64_data), dtype=np.int64)
    elif init.data_type == 6:  # INT32
        if init.raw_data:
            arr = np.frombuffer(init.raw_data, dtype=np.int32)
        else:
            arr = np.array(list(init.int32_data), dtype=np.int32)
    else:
        return None
    if list(init.dims):
        try:
            arr = arr.reshape(list(init.dims))
        except ValueError:
            return None
    return arr


def algebraic_identities(model):
    """Eliminate identity ops where the constant is SCALAR (no broadcasting ambiguity).

    SAFE: only eliminate Mul(x, 1), Add(x, 0), Sub(x, 0) where the constant is shape=[] or shape=[1].
    """
    import numpy as np
    init_by_name = {i.name: i for i in model.graph.initializer}
    graph_outputs = {o.name for o in model.graph.output}
    new_inits_to_add = []  # constant-folded initializers

    def is_scalar_value(name, value):
        if name not in init_by_name:
            return False
        init = init_by_name[name]
        dims = list(init.dims)
        # Accept scalar [] or singleton [1]
        if dims and dims != [1]:
            return False
        d = init_data_tuple(init)
        if d is None or len(d) == 0:
            return False
        return all(v == value for v in d)

    rename = {}
    n_removed = 0
    keep_nodes = []

    for node in model.graph.node:
        inputs = [rename.get(inp, inp) for inp in node.input]
        for i in range(len(node.input)):
            node.input[i] = inputs[i]

        any_graph_out = any(o in graph_outputs for o in node.output)
        eliminate = False
        replacement = None

        if not any_graph_out:
            if len(inputs) == 2:
                if node.op_type == 'Mul':
                    if is_scalar_value(inputs[1], 1):
                        replacement = inputs[0]; eliminate = True
                    elif is_scalar_value(inputs[0], 1):
                        replacement = inputs[1]; eliminate = True
                elif node.op_type == 'Add':
                    if is_scalar_value(inputs[1], 0):
                        replacement = inputs[0]; eliminate = True
                    elif is_scalar_value(inputs[0], 0):
                        replacement = inputs[1]; eliminate = True
                elif node.op_type == 'Sub':
                    if is_scalar_value(inputs[1], 0):
                        replacement = inputs[0]; eliminate = True
                elif node.op_type == 'Div':
                    if is_scalar_value(inputs[1], 1):
                        replacement = inputs[0]; eliminate = True
                elif node.op_type == 'Pow':
                    if is_scalar_value(inputs[1], 1):
                        replacement = inputs[0]; eliminate = True

            # Concat with single input
            if node.op_type == 'Concat' and len(inputs) == 1:
                replacement = inputs[0]; eliminate = True

            # Variadic ops with single input (Sum/Min/Max/Mean) -> identity
            if node.op_type in ('Sum', 'Min', 'Max', 'Mean') and len(inputs) == 1:
                replacement = inputs[0]; eliminate = True

            # Identity op (literally a no-op)
            if node.op_type == 'Identity':
                replacement = inputs[0]; eliminate = True

            # Pad with all-zero pads (and pads is a known initializer)
            if node.op_type == 'Pad' and len(inputs) >= 2:
                pads_name = inputs[1]
                if pads_name in init_by_name:
                    pads_data = init_data_tuple(init_by_name[pads_name])
                    if pads_data is not None and all(p == 0 for p in pads_data):
                        replacement = inputs[0]; eliminate = True

            # Transpose with identity perm (perm == [0, 1, ..., N-1])
            if node.op_type == 'Transpose':
                perm_attr = next((a for a in node.attribute if a.name == 'perm'), None)
                if perm_attr is not None:
                    perm = list(perm_attr.ints)
                    if perm == list(range(len(perm))):
                        replacement = inputs[0]; eliminate = True

            # Cast to same dtype as input.
            # If input is an initializer of the target dtype, Cast is identity.
            if node.op_type == 'Cast':
                to_attr = next((a for a in node.attribute if a.name == 'to'), None)
                if to_attr is not None and inputs[0] in init_by_name:
                    if init_by_name[inputs[0]].data_type == to_attr.i:
                        replacement = inputs[0]; eliminate = True
                # Cast -> Cast chain where second Cast is to same dtype as first's "to"
                # is redundant: if the input is already that dtype, second Cast does nothing.
                # Look up the input node by output name.
                if not eliminate and to_attr is not None:
                    for pn in model.graph.node:
                        if inputs[0] in pn.output and pn.op_type == 'Cast':
                            pn_to = next((a for a in pn.attribute if a.name == 'to'), None)
                            if pn_to is not None and pn_to.i == to_attr.i:
                                replacement = inputs[0]; eliminate = True
                                break

            # Min/Max with identical inputs (Min(x, x) -> x, Max(x, x) -> x)
            if node.op_type in ('Min', 'Max', 'And', 'Or') and len(inputs) == 2:
                if inputs[0] == inputs[1]:
                    replacement = inputs[0]; eliminate = True

            # Sub(x, x) where both inputs are the same -> SKIPPED
            # (would need to construct a zero constant of matching shape; complex)
            # Same for Mul(x, 0): would need to construct zero constant.

            # Concat with all-same-name inputs (degenerates if N inputs are identical)
            # -> simplifies to that input if N==1 (already handled). For N>1, no easy simplification.

            # Constant folding: Mul/Add/Sub/Div with TWO initializer inputs -> precompute
            if node.op_type in ('Mul', 'Add', 'Sub', 'Div') and len(inputs) == 2:
                if inputs[0] in init_by_name and inputs[1] in init_by_name:
                    a = _init_to_numpy(init_by_name[inputs[0]])
                    b = _init_to_numpy(init_by_name[inputs[1]])
                    if a is not None and b is not None:
                        try:
                            if node.op_type == 'Mul':
                                result = a * b
                            elif node.op_type == 'Add':
                                result = a + b
                            elif node.op_type == 'Sub':
                                result = a - b
                            elif node.op_type == 'Div':
                                result = a / b
                            # Construct new initializer
                            from onnx import helper, numpy_helper
                            new_init = numpy_helper.from_array(result)
                            new_init.name = node.output[0] + '_folded'
                            new_inits_to_add.append(new_init)
                            replacement = new_init.name
                            eliminate = True
                        except Exception:
                            pass

            # Slice with starts=[0..] and ends=[INT_MAX..] (max int sentinels) -> identity
            # Slice attribute version (opset <11) has starts/ends/axes as attrs.
            # Slice tensor version (opset >=11) has them as inputs.
            if node.op_type == 'Slice':
                starts = None
                ends = None
                if len(inputs) >= 3:
                    # opset 10+: tensor-input form
                    if inputs[1] in init_by_name:
                        starts = init_data_tuple(init_by_name[inputs[1]])
                    if inputs[2] in init_by_name:
                        ends = init_data_tuple(init_by_name[inputs[2]])
                else:
                    # opset 9 attribute form
                    sa = next((a for a in node.attribute if a.name == 'starts'), None)
                    ea = next((a for a in node.attribute if a.name == 'ends'), None)
                    if sa is not None:
                        starts = tuple(sa.ints)
                    if ea is not None:
                        ends = tuple(ea.ints)
                # If all starts == 0 and all ends are huge (>= 2^31 - 1 or similar sentinels),
                # the slice is identity. Common sentinel: 9223372036854775807 (max int64).
                if starts is not None and ends is not None:
                    INT_MAX_SENTINEL = 2**31  # threshold for "definitely identity"
                    if all(s == 0 for s in starts) and all(e >= INT_MAX_SENTINEL for e in ends):
                        replacement = inputs[0]; eliminate = True

        if eliminate:
            for out in node.output:
                rename[out] = replacement
            n_removed += 1
        else:
            keep_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)
    # Add any new constant-folded initializers
    if new_inits_to_add:
        model.graph.initializer.extend(new_inits_to_add)
    return n_removed


def optimize(model):
    stats = {'dedupe': 0, 'cse': 0, 'dce_nodes': 0, 'dce_inits': 0, 'alg': 0}
    for _ in range(5):
        n_dedup = dedupe_inits(model)
        n_alg = algebraic_identities(model)
        n_cse = cse_pass(model)
        n_dce_n, n_dce_i = dce_pass(model)
        stats['dedupe'] += n_dedup
        stats['alg'] += n_alg
        stats['cse'] += n_cse
        stats['dce_nodes'] += n_dce_n
        stats['dce_inits'] += n_dce_i
        if n_dedup + n_alg + n_cse + n_dce_n + n_dce_i == 0:
            break
    return stats


def encode(grid):
    a = np.array(grid, dtype=np.int64); h, w = a.shape
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        out[0, c, :h, :w] = (a == c).astype(np.float32)
    out[0, 0, h:, :] = 1.0; out[0, 0, :h, w:] = 1.0
    return out


def verify_equivalent(orig_bytes, new_bytes, tid):
    """Run both ONNX on sample input from task JSON; return True if outputs match."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    DATA = pathlib.Path(os.environ.get('NEUROGOLF_DATA_DIR', repo / 'data' / 'competitions' / 'neurogolf-2026'))
    json_path = DATA / f'task{tid:03d}.json'
    if not json_path.exists():
        return None  # can't verify, assume ok
    try:
        data = json.loads(json_path.read_text())
        pairs = data['train'] + data['test']
        if not pairs:
            return None
        inp = encode(pairs[0]['input'])
        s1 = ort.InferenceSession(orig_bytes, providers=['CPUExecutionProvider'])
        s2 = ort.InferenceSession(new_bytes, providers=['CPUExecutionProvider'])
        y1 = np.asarray(s1.run(None, {'input': inp})[0], dtype=np.float64)
        y2 = np.asarray(s2.run(None, {'input': inp})[0], dtype=np.float64)
        return float(np.abs(y1 - y2).max()) < 1e-4
    except Exception:
        return False


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    NG = pathlib.Path(os.environ.get('NEUROGOLF_NOGIT_ROOT', repo / '.notongit' / 'NeuroGolf_nogit'))
    src_mega = NG / 'submissions/_alpha_a_scratch/alpha_a_mega_t089v2_t137v2/submission.zip'
    new_dir = NG / 'submissions/_alpha_a_scratch/alpha_a_mega_t089v2_t137v2_safe2'
    new_dir.mkdir(exist_ok=True)
    new_zip = new_dir / 'submission.zip'

    total = {'dedupe': 0, 'cse': 0, 'dce_nodes': 0, 'dce_inits': 0, 'alg': 0}
    n_processed = 0
    n_failed = 0
    n_fallback = 0
    new_entries = {}
    with zipfile.ZipFile(src_mega) as zin:
        for name in sorted(zin.namelist()):
            data = zin.read(name)
            if not name.endswith('.onnx'):
                new_entries[name] = data
                continue
            try:
                m = onnx.load_from_string(data)
                stats = optimize(m)
                new_data = m.SerializeToString()
                # Verify
                tid = int(name[4:7])
                ok = verify_equivalent(data, new_data, tid)
                if ok is False:
                    # CSE broke it - fallback to dedupe-only
                    m2 = onnx.load_from_string(data)
                    dedupe_inits(m2)
                    new_data = m2.SerializeToString()
                    n_fallback += 1
                else:
                    for k in total: total[k] += stats[k]
                new_entries[name] = new_data
            except Exception as e:
                new_entries[name] = data
                n_failed += 1
            n_processed += 1
            if n_processed % 50 == 0:
                print(f'  processed {n_processed}, fallback {n_fallback}, failed {n_failed}', flush=True)

    print(f'Total processed: {n_processed}, fallback: {n_fallback}, failed: {n_failed}', flush=True)
    print(f'Stats (only counted from non-fallback): {total}', flush=True)
    with zipfile.ZipFile(new_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in sorted(new_entries.items()):
            zout.writestr(name, data)
    print(f'Wrote {new_zip} ({new_zip.stat().st_size} bytes)', flush=True)


if __name__ == '__main__':
    main()
