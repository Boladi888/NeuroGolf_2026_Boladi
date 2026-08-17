"""Safe optimizer + constant folding.

In addition to dedup/CSE/DCE/algebraic, also do CONSTANT FOLDING:
- Identify nodes whose all inputs are initializers (or computed-constant)
- Evaluate the node using numpy (for supported ops only)
- Replace the node with a new initializer

SAFE ops for folding (limited set):
- Mul, Add, Sub, Div
- Cast
- Reshape
- Concat

This can collapse chains of constant-only operations.
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


def init_to_array(init):
    """Convert an Initializer (or Constant tensor) to a numpy array."""
    data = init_data_tuple(init)
    if data is None:
        return None
    dtype = {7: np.int64, 1: np.float32, 6: np.int32}.get(init.data_type)
    if dtype is None:
        return None
    arr = np.array(data, dtype=dtype)
    if init.dims:
        arr = arr.reshape(list(init.dims))
    else:
        arr = arr.reshape([])  # scalar
    return arr


def array_to_init(arr, name):
    """Build an Initializer tensor from a numpy array."""
    if arr.dtype == np.int64:
        dtype = 7
    elif arr.dtype == np.float32:
        dtype = 1
    elif arr.dtype == np.int32:
        dtype = 6
    else:
        return None
    t = onnx.TensorProto()
    t.name = name
    t.data_type = dtype
    t.dims.extend(list(arr.shape))
    t.raw_data = arr.tobytes()
    return t


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
        data = init_data_tuple(tensor)
        return (attr.name, 't', tuple(tensor.dims), tensor.data_type, data)
    else:
        return (attr.name, 'serial', attr.SerializeToString())


def cse_pass(model):
    graph_outputs = {o.name for o in model.graph.output}
    rename = {}
    seen = {}
    keep_nodes = []
    for node in model.graph.node:
        inputs = tuple(rename.get(inp, inp) for inp in node.input)
        attrs = tuple(sorted(attr_to_tuple(a) for a in node.attribute))
        n_outputs = len(node.output)
        signature = (node.op_type, inputs, attrs, n_outputs)
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


def algebraic_identities(model):
    init_by_name = {i.name: i for i in model.graph.initializer}
    graph_outputs = {o.name for o in model.graph.output}

    def is_scalar_value(name, value):
        if name not in init_by_name:
            return False
        init = init_by_name[name]
        dims = list(init.dims)
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
        if not any_graph_out and len(inputs) == 2:
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
        if eliminate:
            for out in node.output:
                rename[out] = replacement
            n_removed += 1
        else:
            keep_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)
    return n_removed


def constant_fold(model):
    """Fold Mul/Add/Sub of two initializers into a single new initializer.

    Conservative: only folds binary elementwise ops with both inputs being initializers.
    """
    init_by_name = {i.name: i for i in model.graph.initializer}
    new_inits = []
    rename = {}
    n_folded = 0
    keep_nodes = []
    counter = 0

    SUPPORTED_BINARY = {'Mul': np.multiply, 'Add': np.add, 'Sub': np.subtract}

    for node in model.graph.node:
        inputs = [rename.get(inp, inp) for inp in node.input]
        for i in range(len(node.input)):
            node.input[i] = inputs[i]

        can_fold = False
        if node.op_type in SUPPORTED_BINARY and len(inputs) == 2:
            a_init = init_by_name.get(inputs[0])
            b_init = init_by_name.get(inputs[1])
            if a_init is not None and b_init is not None:
                a = init_to_array(a_init)
                b = init_to_array(b_init)
                if a is not None and b is not None and a.dtype == b.dtype:
                    try:
                        result = SUPPORTED_BINARY[node.op_type](a, b)
                        # Only fold if result is reasonably small (avoid huge memory bloat)
                        if result.nbytes <= max(a.nbytes, b.nbytes) * 2:
                            can_fold = True
                            new_name = f'_fold_{counter}_{node.output[0]}'
                            counter += 1
                            new_init = array_to_init(result, new_name)
                            if new_init is not None:
                                new_inits.append(new_init)
                                init_by_name[new_name] = new_init
                                for out in node.output:
                                    rename[out] = new_name
                                n_folded += 1
                    except Exception:
                        can_fold = False
        if not can_fold:
            keep_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)
    model.graph.initializer.extend(new_inits)
    return n_folded


def optimize(model):
    stats = {'dedupe': 0, 'cse': 0, 'dce_nodes': 0, 'dce_inits': 0, 'alg': 0, 'fold': 0}
    for _ in range(5):
        n_dedup = dedupe_inits(model)
        n_alg = algebraic_identities(model)
        n_fold = constant_fold(model)
        n_cse = cse_pass(model)
        n_dce_n, n_dce_i = dce_pass(model)
        stats['dedupe'] += n_dedup
        stats['alg'] += n_alg
        stats['fold'] += n_fold
        stats['cse'] += n_cse
        stats['dce_nodes'] += n_dce_n
        stats['dce_inits'] += n_dce_i
        if n_dedup + n_alg + n_fold + n_cse + n_dce_n + n_dce_i == 0:
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
    repo = pathlib.Path(__file__).resolve().parents[1]
    DATA = pathlib.Path(os.environ.get('NEUROGOLF_DATA_DIR', repo / 'data' / 'competitions' / 'neurogolf-2026'))
    json_path = DATA / f'task{tid:03d}.json'
    if not json_path.exists(): return None
    try:
        data = json.loads(json_path.read_text())
        pairs = data['train'] + data['test']
        if not pairs: return None
        inp = encode(pairs[0]['input'])
        s1 = ort.InferenceSession(orig_bytes, providers=['CPUExecutionProvider'])
        s2 = ort.InferenceSession(new_bytes, providers=['CPUExecutionProvider'])
        y1 = np.asarray(s1.run(None, {'input': inp})[0], dtype=np.float64)
        y2 = np.asarray(s2.run(None, {'input': inp})[0], dtype=np.float64)
        return float(np.abs(y1 - y2).max()) < 1e-3  # looser tolerance for folded
    except Exception:
        return False


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    NG = pathlib.Path(os.environ.get('NEUROGOLF_NOGIT_ROOT', repo / '.notongit' / 'NeuroGolf_nogit'))
    src_mega = NG / 'submissions/_alpha_a_scratch/alpha_a_mega_t089v2_t137v2/submission.zip'
    new_dir = NG / 'submissions/_alpha_a_scratch/alpha_a_mega_t089v2_t137v2_fold'
    new_dir.mkdir(exist_ok=True)
    new_zip = new_dir / 'submission.zip'

    total = {'dedupe': 0, 'cse': 0, 'dce_nodes': 0, 'dce_inits': 0, 'alg': 0, 'fold': 0}
    n_processed = 0
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
                tid = int(name[4:7])
                ok = verify_equivalent(data, new_data, tid)
                if ok is False:
                    m2 = onnx.load_from_string(data)
                    dedupe_inits(m2)
                    new_data = m2.SerializeToString()
                    n_fallback += 1
                else:
                    for k in total: total[k] += stats[k]
                new_entries[name] = new_data
            except Exception:
                new_entries[name] = data
            n_processed += 1
            if n_processed % 50 == 0:
                print(f'  processed {n_processed}, fallback {n_fallback}', flush=True)

    print(f'Total processed: {n_processed}, fallback: {n_fallback}', flush=True)
    print(f'Stats: {total}', flush=True)
    with zipfile.ZipFile(new_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in sorted(new_entries.items()):
            zout.writestr(name, data)
    print(f'Wrote {new_zip} ({new_zip.stat().st_size} bytes)', flush=True)


if __name__ == '__main__':
    main()
