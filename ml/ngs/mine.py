"""mine.py — turn the 401 canonical winning ONNXes into (a) an op-frequency map that says
which ops to add to the DSL next, and (b) an imitation corpus (task -> op-sequence).

Runs in the plain .venv (onnx only). Reads submissions/submission.zip (read-only) and the
task data; writes ONLY into ml/ngs/_corpus/.

  python -m ml.ngs.mine            # analyze + write corpus, print op-frequency + DSL coverage
"""
import collections
import io
import json
import pathlib
import zipfile
import onnx

from . import tasks
from .policy import task_features, FEATURE_NAMES
from .ops import base_vocab

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ZIP = REPO_ROOT / "submissions" / "submission.zip"
OUT = pathlib.Path(__file__).resolve().parent / "_corpus"

# ONNX op_types the current DSL can already emit
_DSL_OPTYPES = {"Identity", "Transpose", "Cast", "Gather", "ReduceSum", "Greater"}


def _op_sequence(model):
    return [n.op_type for n in model.graph.node]


def mine():
    if not ZIP.exists():
        print(f"no submission.zip at {ZIP}"); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    freq = collections.Counter()
    per_graph_optypes = []
    rows = []
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".onnx"))
        for nm in names:
            tid = int("".join(ch for ch in pathlib.Path(nm).stem if ch.isdigit()) or -1)
            try:
                model = onnx.load_model_from_string(z.read(nm))
            except Exception:
                continue
            ops = _op_sequence(model)
            freq.update(ops)
            per_graph_optypes.append(set(ops))
            task = tasks.load_task(tid) if tid >= 0 else None
            feats = task_features(tasks.all_pairs(task)).tolist() if task else None
            rows.append({"tid": tid, "ops": ops, "n_nodes": len(ops),
                         "features": feats, "feature_names": FEATURE_NAMES if feats else None})

    (OUT / "imitation.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    json.dump(dict(freq.most_common()), open(OUT / "op_frequency.json", "w"), indent=1)

    total = sum(freq.values())
    print(f"== mined {len(rows)} graphs, {total} nodes, {len(freq)} distinct op types ==")
    print(f"corpus -> {OUT/'imitation.jsonl'}  ({len(rows)} rows)")
    print("\n  op_type            count   in-DSL?")
    for op, c in freq.most_common(25):
        print(f"  {op:18} {c:5}    {'yes' if op in _DSL_OPTYPES else 'ADD'}")

    # graph-level DSL coverage: what fraction of winning graphs use ONLY ops the DSL has
    covered = sum(1 for s in per_graph_optypes if s <= _DSL_OPTYPES)
    print(f"\n  graphs expressible with the CURRENT DSL op-set: {covered}/{len(per_graph_optypes)}")
    missing = collections.Counter()
    for s in per_graph_optypes:
        for op in (s - _DSL_OPTYPES):
            missing[op] += 1
    print("  highest-value ops to add next (appear in the most winning graphs):")
    for op, c in missing.most_common(10):
        print(f"    +{op:18} unlocks up to {c} more graphs")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(mine())
