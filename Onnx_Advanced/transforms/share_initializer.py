"""share_initializer -- merge byte-identical initializers into one shared tensor.

The reusable, tested form of the p43/p44 "two shifted/duplicate tables where one shared table is
enough" win (task260 two shifted power tables; task323 mask reuse). Output-preserving BY
CONSTRUCTION: identical dtype+dims+raw bytes => identical values, so every node input that
referenced a duplicate is repointed to the canonical copy and the redundant initializers (and the
`params` bytes they were charged) are dropped.

Usage (from repo root):
    import sys; sys.path.insert(0, "Onnx_Advanced/transforms")
    import onnx
    from share_initializer import share_initializers
    model, report = share_initializers(onnx.load("cand.onnx"))
    # report == {"merged": <n dropped dups>, "removed_names": [...]}
"""
from __future__ import annotations

import hashlib

import onnx


def _init_key(t: "onnx.TensorProto") -> str:
    """Identity of an initializer = dtype + dims + raw values (raw_data or the typed field)."""
    h = hashlib.sha256()
    h.update(bytes([t.data_type]))
    h.update(",".join(map(str, t.dims)).encode())
    if t.raw_data:
        h.update(t.raw_data)
    else:
        for field in ("float_data", "int32_data", "int64_data", "double_data", "uint64_data"):
            vals = getattr(t, field)
            if vals:
                h.update(field.encode())
                h.update(str(list(vals)).encode())
    return h.hexdigest()


def share_initializers(model: "onnx.ModelProto") -> tuple["onnx.ModelProto", dict]:
    """Merge byte-identical initializers in-place. Returns (model, report)."""
    g = model.graph
    canon: dict[str, str] = {}   # value-key -> canonical initializer name
    remap: dict[str, str] = {}   # dropped duplicate name -> canonical name
    keep = []
    for init in g.initializer:
        key = _init_key(init)
        if key in canon:
            remap[init.name] = canon[key]
        else:
            canon[key] = init.name
            keep.append(init)
    if not remap:
        return model, {"merged": 0, "removed_names": []}
    del g.initializer[:]
    g.initializer.extend(keep)
    # repoint every node input that referenced a dropped duplicate
    for node in g.node:
        for i, name in enumerate(node.input):
            if name in remap:
                node.input[i] = remap[name]
    # some models also declare initializers as graph inputs -- drop the dead ones there too
    dropped = set(remap)
    kept_inputs = [vi for vi in g.input if vi.name not in dropped]
    if len(kept_inputs) != len(g.input):
        del g.input[:]
        g.input.extend(kept_inputs)
    return model, {"merged": len(remap), "removed_names": sorted(remap)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Merge byte-identical initializers in an ONNX model.")
    ap.add_argument("model")
    ap.add_argument("--out", help="Write the shared model here (default: dry-run, report only).")
    a = ap.parse_args()
    m, rep = share_initializers(onnx.load(a.model))
    print(f"merged {rep['merged']} duplicate initializer(s): {rep['removed_names']}")
    if a.out and rep["merged"]:
        onnx.save(m, a.out)
        print(f"wrote {a.out}")
