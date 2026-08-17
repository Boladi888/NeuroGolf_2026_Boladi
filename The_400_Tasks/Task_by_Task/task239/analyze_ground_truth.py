"""Ground-truth measurement for task239 crop work. Run with .venv python."""
import sys, os, zipfile, io, json
sys.path.insert(0, "tools")
import onnx
from _score_v2 import score_network_v2

ROOT = "<REPO_ROOT>"
os.chdir(ROOT)

def score(path_or_bytes, label):
    try:
        if isinstance(path_or_bytes, bytes):
            tmp = f"The_400_Tasks/Task_by_Task/task239/_tmp_{label}.onnx"
            with open(tmp, "wb") as f:
                f.write(path_or_bytes)
            res = score_network_v2(tmp)
        else:
            res = score_network_v2(path_or_bytes)
        if res is None:
            print(f"[{label}] score_network_v2 -> None (REJECTED by strict scorer)")
            return None
        macs, mem, params = res
        print(f"[{label}] macs={macs} memory={mem} params={params} COST={mem+params}")
        return (macs, mem, params)
    except Exception as e:
        print(f"[{label}] ERROR: {type(e).__name__}: {e}")
        return None

print("=" * 70)
print("1. CANONICAL (Path A baseline) — extract from submission.zip")
print("=" * 70)
canon_bytes = None
with zipfile.ZipFile("submissions/submission.zip") as z:
    canon_bytes = z.read("task239.onnx")
print(f"canonical task239.onnx: {len(canon_bytes)} bytes on disk")
score(canon_bytes, "canonical")

# Inspect canonical graph structure
m = onnx.load_from_string(canon_bytes)
g = m.graph
print(f"\nopset: {[(i.domain, i.version) for i in m.opset_import]}")
print(f"nodes: {len(g.node)}  initializers: {len(g.initializer)}")
print(f"graph input: {[(i.name, i.type.tensor_type.elem_type, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input]}")
print(f"graph output: {[(o.name, o.type.tensor_type.elem_type, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output]}")
print("\n--- nodes ---")
for i, n in enumerate(g.node):
    print(f"{i:2d} | {n.op_type:12s} | in={list(n.input)} -> out={list(n.output)}")
print("\n--- initializers ---")
for init in g.initializer:
    from onnx import numpy_helper
    arr = numpy_helper.to_array(init)
    flat = arr.flatten()
    preview = flat[:12].tolist()
    print(f"  {init.name:20s} dtype={init.data_type} shape={list(init.dims)} vals[:12]={preview}")

print("\n" + "=" * 70)
print("2. EXISTING PROTOTYPES")
print("=" * 70)
score("The_400_Tasks/Task_by_Task/task239/prototypes/task239_crop4x4_6A9A4E1D.onnx", "proto_crop4x4")
score("The_400_Tasks/Task_by_Task/task239/prototypes/task239_f16_9085A974.onnx", "proto_f16")

print("\n" + "=" * 70)
print("3. TRUE GRID DIMENSIONS from task239.json")
print("=" * 70)
with open("data/competitions/neurogolf-2026/task239.json") as f:
    data = json.load(f)
print(f"keys: {list(data.keys())}")
for split in data.keys():
    pairs = data[split]
    if not isinstance(pairs, list):
        continue
    max_ih = max_iw = max_oh = max_ow = 0
    for p in pairs:
        ih, iw = len(p["input"]), len(p["input"][0])
        oh, ow = len(p["output"]), len(p["output"][0])
        max_ih = max(max_ih, ih); max_iw = max(max_iw, iw)
        max_oh = max(max_oh, oh); max_ow = max(max_ow, ow)
    print(f"  {split:10s} n={len(pairs):5d}  max_in={max_ih}x{max_iw}  max_out={max_oh}x{max_ow}")

# global max across all splits
all_pairs = []
for split in data.keys():
    if isinstance(data[split], list):
        all_pairs += data[split]
g_ih = max(len(p["input"]) for p in all_pairs)
g_iw = max(len(p["input"][0]) for p in all_pairs)
g_oh = max(len(p["output"]) for p in all_pairs)
g_ow = max(len(p["output"][0]) for p in all_pairs)
print(f"\nGLOBAL max input  = {g_ih} x {g_iw}")
print(f"GLOBAL max output = {g_oh} x {g_ow}")
print(f"total pairs: {len(all_pairs)}")

# Distribution of output sizes
from collections import Counter
osz = Counter((len(p["output"]), len(p["output"][0])) for p in all_pairs)
print(f"\noutput-size distribution (top 15): {osz.most_common(15)}")
isz = Counter((len(p["input"]), len(p["input"][0])) for p in all_pairs)
print(f"input-size distribution (top 15): {isz.most_common(15)}")
