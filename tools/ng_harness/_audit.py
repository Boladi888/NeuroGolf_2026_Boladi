"""Self-audit of ng_harness: verify foundations against the OFFICIAL pipeline,
probe determinism, edge cases, and known-risky spots. Read-only."""
import os, sys, pathlib, tempfile, shutil
import numpy as np
NG = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NG))
sys.path.insert(0, str(NG / "data" / "competitions" / "neurogolf-2026" / "neurogolf_utils"))
import neurogolf_utils as nu
from tools.ng_harness import encode, session, corpus, registry

FAILS = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' :: '+detail) if detail and not cond else ''}")
    if not cond: FAILS.append(name)

print("=== C. encode_grid / encode_batch_index == nu.convert_to_numpy (official) ===")
pairs = registry.official_pairs(324)[:80]
mism_in = mism_out = mism_batch = 0
for p in pairs:
    b = nu.convert_to_numpy(p)
    if b is None: continue
    ein = encode.encode_grid(p["input"]); eout = encode.encode_grid(p["output"])
    if not np.array_equal(ein, b["input"]): mism_in += 1
    if not np.array_equal(eout, b["output"]): mism_out += 1
    ii, ish = encode.grid_to_index(p["input"])
    eb = encode.encode_batch_index(ii[None], np.array([ish]))
    if not np.array_equal(eb, b["input"]): mism_batch += 1
check("encode_grid input == official", mism_in == 0, f"{mism_in} mismatch")
check("encode_grid output == official", mism_out == 0, f"{mism_out} mismatch")
check("encode_batch_index == official", mism_batch == 0, f"{mism_batch} mismatch")

print("=== D. corpus determinism on TRUE regeneration (clear cache, rebuild) ===")
# build task110 in a fresh temp cache dir and compare to a second fresh build
import importlib
orig = corpus.CACHE_DIR
try:
    d1 = pathlib.Path(tempfile.mkdtemp())
    d2 = pathlib.Path(tempfile.mkdtemp())
    corpus.CACHE_DIR = d1; a = corpus.build(110, 60)
    corpus.CACHE_DIR = d2; b = corpus.build(110, 60)
    same = (np.array_equal(a["in_idx"], b["in_idx"]) and np.array_equal(a["out_idx"], b["out_idx"])
            and np.array_equal(a["in_shapes"], b["in_shapes"]))
    check("two independent fresh builds identical", same)
    # incremental vs one-shot: build 30 then extend to 60 == one-shot 60
    d3 = pathlib.Path(tempfile.mkdtemp()); corpus.CACHE_DIR = d3
    corpus.build(110, 30); c = corpus.build(110, 60)
    inc_same = np.array_equal(a["in_idx"], c["in_idx"]) and np.array_equal(a["out_idx"], c["out_idx"])
    check("incremental build == one-shot build", inc_same)
finally:
    corpus.CACHE_DIR = orig
    for d in (d1, d2, d3):
        shutil.rmtree(d, ignore_errors=True)

print("=== E. sanitize on/off output-equivalent (v3, 60 corpus inputs) ===")
V3 = str(NG / "The_400_Tasks/Task_by_Task/task324/prototypes/task324_compress_v3.onnx")
X, _ = corpus.get_tensors(324, 60)
s_on, n_on = session.make_session(V3, sanitize=True)
s_off, n_off = session.make_session(V3, sanitize=False)
diff = sum(not np.array_equal(session.run_raw(s_on, n_on, X[i:i+1]),
                              session.run_raw(s_off, n_off, X[i:i+1])) for i in range(60))
check("sanitize does not change outputs", diff == 0, f"{diff} differ")

print("=== F. decode(>0) vs raw array_equal on a clean one-hot model (v3 self) ===")
# v3 output should be clean 0/1, so >0-decode and raw compare must agree
raw_eq = dec_eq = 0
for i in range(60):
    o1 = s_on.run(["output"], {n_on: X[i:i+1].astype(np.float32)})[0]
    raw_eq += int(np.array_equal((np.asarray(o1) > 0).astype(np.float32),
                                 np.asarray(o1).astype(np.float32)))  # is raw already 0/1?
check("v3 raw output is already 0/1 (decode==raw)", raw_eq == 60, f"{raw_eq}/60 clean")

print("\n=== AUDIT RESULT:", "ALL PASS" if not FAILS else f"FAILURES: {FAILS}", "===")
sys.exit(1 if FAILS else 0)
