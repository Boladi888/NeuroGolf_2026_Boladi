"""Generate fresh ARC-GEN pairs for task239 (uuid 9af7a82c), check output-size
bounds, and stash pairs for byte-equal candidate verification."""
import sys, os, json, random, importlib
sys.path.insert(0, "external/ARC-GEN")
sys.path.insert(0, "external/ARC-GEN/tasks")
os.chdir("<REPO_ROOT>")

import common  # noqa
gen = importlib.import_module("task_9af7a82c")

def make_pairs(n, seed0=0):
    pairs = []
    for i in range(n):
        random.seed(seed0 + i)
        try:
            p = gen.generate()
        except Exception:
            continue
        pairs.append(p)
    return pairs

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    pairs = make_pairs(N, seed0)
    maxH = max(len(p["output"]) for p in pairs)
    maxW = max(len(p["output"][0]) for p in pairs)
    maxIH = max(len(p["input"]) for p in pairs)
    maxIW = max(len(p["input"][0]) for p in pairs)
    from collections import Counter
    osz = Counter((len(p["output"]), len(p["output"][0])) for p in pairs)
    print(f"generated {len(pairs)} pairs (seed0={seed0})")
    print(f"max input  = {maxIH} x {maxIW}")
    print(f"max output = {maxH} x {maxW}  (H=rows, W=cols)")
    print(f"output-size distribution: {sorted(osz.items())}")
    # any output exceeding 12x5?
    over = [(len(p['output']), len(p['output'][0])) for p in pairs
            if len(p['output']) > 12 or len(p['output'][0]) > 5]
    print(f"pairs exceeding 12x5: {len(over)}  examples={over[:5]}")
    # stash
    out = "The_400_Tasks/Task_by_Task/task239/_fresh_pairs.json"
    with open(out, "w") as f:
        json.dump(pairs, f)
    print(f"stashed {len(pairs)} pairs -> {out}")
