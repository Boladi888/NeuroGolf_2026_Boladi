"""Structural diff between two ONNX models.

Reports:
  - Nodes added in B (not in A)
  - Nodes removed from A (not in B)
  - Initializers added / removed
  - Op-type count differences
  - Cost comparison (if scorer available)

Usage:
    python tools/agent_helpers/diff_graphs.py before.onnx after.onnx
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

# Make tools.* imports work when invoked from any cwd
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import onnx


def node_key(node):
    """Hashable identity of a node — by op_type + sorted input/output structure."""
    return (node.op_type, tuple(sorted(node.input)), tuple(sorted(node.output)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_a")
    p.add_argument("model_b")
    args = p.parse_args()

    a = onnx.load(args.model_a).graph
    b = onnx.load(args.model_b).graph

    a_ops = Counter(n.op_type for n in a.node)
    b_ops = Counter(n.op_type for n in b.node)

    a_init_names = {init.name for init in a.initializer}
    b_init_names = {init.name for init in b.initializer}

    print(f"=== Diff: {args.model_a} → {args.model_b} ===")
    print(f"Nodes: {len(a.node)} → {len(b.node)} (Δ {len(b.node) - len(a.node):+})")
    print(f"Initializers: {len(a.initializer)} → {len(b.initializer)} (Δ {len(b.initializer) - len(a.initializer):+})")
    print(f"value_info: {len(a.value_info)} → {len(b.value_info)} (Δ {len(b.value_info) - len(a.value_info):+})")

    print(f"\n=== Op-type changes ===")
    all_ops = set(a_ops) | set(b_ops)
    for op in sorted(all_ops):
        an = a_ops.get(op, 0)
        bn = b_ops.get(op, 0)
        if an != bn:
            print(f"  {op:<20} {an:>3} → {bn:>3} (Δ {bn - an:+})")

    print(f"\n=== Initializers added (in B, not in A) ===")
    added_inits = b_init_names - a_init_names
    for name in sorted(added_inits)[:20]:
        init = next(i for i in b.initializer if i.name == name)
        print(f"  + {name:<35} dims={list(init.dims)} dtype={init.data_type}")
    if len(added_inits) > 20:
        print(f"  ... ({len(added_inits) - 20} more)")

    print(f"\n=== Initializers removed (in A, not in B) ===")
    removed_inits = a_init_names - b_init_names
    for name in sorted(removed_inits)[:20]:
        init = next(i for i in a.initializer if i.name == name)
        print(f"  - {name:<35} dims={list(init.dims)} dtype={init.data_type}")
    if len(removed_inits) > 20:
        print(f"  ... ({len(removed_inits) - 20} more)")

    # Try scoring both
    try:
        from tools._score_v2 import score_network_v2
        import math
        for label, path in [("A", args.model_a), ("B", args.model_b)]:
            res = score_network_v2(path)
            if res is None:
                print(f"\n{label}: score_network_v2 returned None")
                continue
            macs, mem, params = res
            cost = mem + params
            points = max(1.0, 25.0 - math.log(max(1.0, cost)))
            print(f"\n{label} ({path}): cost={cost:,} memory={mem:,} params={params:,} points={points:.4f}")
    except Exception as exc:
        print(f"\nScoring failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
