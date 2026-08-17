"""A/B verify task239 crop candidate vs canonical parent + ARC-GEN ground truth.

Gold standard (traps.md hierarchy #1): byte-equal vs the EXACT bytes in
submissions/submission.zip on 1000+ random inputs. We use 20k fresh ARC-GEN
pairs. Also checks both models actually solve (vs ARC-GEN expected output)."""
import sys, os, json, zipfile
sys.path.insert(0, "tools")
sys.path.insert(0, "data/competitions/neurogolf-2026/neurogolf_utils")
import numpy as np
import onnxruntime as ort
import neurogolf_utils as nu

os.chdir("<REPO_ROOT>")

CAND = sys.argv[1] if len(sys.argv) > 1 else "The_400_Tasks/Task_by_Task/task239/prototypes/task239_crop_v1.onnx"
print(f"candidate: {CAND}")

def make_sess(model_bytes_or_path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if isinstance(model_bytes_or_path, bytes):
        return ort.InferenceSession(model_bytes_or_path, so, providers=["CPUExecutionProvider"])
    return ort.InferenceSession(model_bytes_or_path, so, providers=["CPUExecutionProvider"])

with zipfile.ZipFile("submissions/submission.zip") as z:
    canon_bytes = z.read("task239.onnx")
sess_canon = make_sess(canon_bytes)
sess_cand = make_sess(CAND)

def decode_grid(out_decoded, h, w):
    """out_decoded: [1,10,30,30] of {0,1}. Return h x w color grid via argmax-of-active."""
    arr = out_decoded[0]  # [10,30,30]
    g = []
    for r in range(h):
        row = []
        for c in range(w):
            col = arr[:, r, c]
            row.append(int(np.argmax(col)) if col.max() > 0 else 0)
        g.append(row)
    return g

def run_pairs(pairs, label, full_check=True):
    raw_eq = dec_eq = cand_correct = canon_correct = 0
    n = len(pairs)
    first_mismatch = None
    for idx, p in enumerate(pairs):
        x = nu.convert_to_numpy(p)["input"].astype(np.float32)
        rc = sess_canon.run(["output"], {"input": x})[0]
        rd = sess_cand.run(["output"], {"input": x})[0]
        # raw byte-equality (both F16 [1,10,30,30])
        if np.array_equal(rc, rd):
            raw_eq += 1
        decoded_c = (rc > 0.0).astype(float)
        decoded_d = (rd > 0.0).astype(float)
        if np.array_equal(decoded_c, decoded_d):
            dec_eq += 1
        elif first_mismatch is None:
            first_mismatch = (idx, len(p["output"]), len(p["output"][0]))
        if full_check:
            exp = p["output"]
            h, w = len(exp), len(exp[0])
            if decode_grid(decoded_d, h, w) == exp:
                cand_correct += 1
            if decode_grid(decoded_c, h, w) == exp:
                canon_correct += 1
    print(f"[{label}] n={n}")
    print(f"   raw F16 byte-equal cand-vs-canonical : {raw_eq}/{n}")
    print(f"   decoded byte-equal cand-vs-canonical : {dec_eq}/{n}")
    if full_check:
        print(f"   cand correct vs ARC-GEN expected     : {cand_correct}/{n}")
        print(f"   canon correct vs ARC-GEN expected    : {canon_correct}/{n}")
    if first_mismatch:
        print(f"   FIRST DECODED MISMATCH at idx={first_mismatch[0]} (out {first_mismatch[1]}x{first_mismatch[2]})")
    return raw_eq, dec_eq, n

# 1) the 20k fresh ARC-GEN pairs (seed0=100000)
with open("The_400_Tasks/Task_by_Task/task239/_fresh_pairs.json") as f:
    fresh = json.load(f)
run_pairs(fresh, "fresh-arcgen-20k")

# 2) the visible+arc-gen pairs from the task json (sanity, full correctness)
with open("data/competitions/neurogolf-2026/task239.json") as f:
    data = json.load(f)
visible = []
for split in ("train", "test", "arc-gen"):
    if isinstance(data.get(split), list):
        visible += data[split]
run_pairs(visible, "visible+arcgen-json")
