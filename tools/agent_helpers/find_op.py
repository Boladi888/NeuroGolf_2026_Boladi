"""Grep nodes by op_type or by node-name regex.

Usage:
    python tools/agent_helpers/find_op.py model.onnx ReduceSum
    python tools/agent_helpers/find_op.py model.onnx Conv --show-attrs
    python tools/agent_helpers/find_op.py model.onnx --name-regex 'concat.*'
"""
import argparse
import re
import sys
from pathlib import Path

import onnx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_path")
    p.add_argument("op_type", nargs="?", default=None, help="op_type to match (case-sensitive)")
    p.add_argument("--name-regex", default=None, help="regex on node name")
    p.add_argument("--show-attrs", action="store_true", help="dump full attribute list per matched node")
    args = p.parse_args()

    model = onnx.load(args.model_path)
    g = model.graph

    matches = []
    name_re = re.compile(args.name_regex) if args.name_regex else None
    for i, n in enumerate(g.node):
        if args.op_type and n.op_type != args.op_type:
            continue
        if name_re and not name_re.search(n.name or ""):
            continue
        matches.append((i, n))

    print(f"=== {args.model_path}: {len(matches)} matching node(s) ===")
    for i, n in matches:
        name = n.name or "(unnamed)"
        ins = ", ".join(n.input)[:50]
        outs = ", ".join(n.output)[:30]
        print(f"  [{i:>3}] {n.op_type:<15} {name:<25} | in: {ins} | out: {outs}")
        if args.show_attrs:
            for a in n.attribute:
                if a.type == 1:    # FLOAT
                    val = a.f
                elif a.type == 2:  # INT
                    val = a.i
                elif a.type == 3:  # STRING
                    val = a.s.decode("utf-8", errors="replace")
                elif a.type == 6:  # FLOATS
                    val = list(a.floats)
                elif a.type == 7:  # INTS
                    val = list(a.ints)
                elif a.type == 8:  # STRINGS
                    val = [s.decode("utf-8", errors="replace") for s in a.strings]
                else:
                    val = f"<type={a.type}>"
                print(f"         attr: {a.name} = {val}")

    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
