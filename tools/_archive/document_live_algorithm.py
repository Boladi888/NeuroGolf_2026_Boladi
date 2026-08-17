"""Document the live ONNX algorithm for a task into its Task_by_Task/taskNNN/ folder.

Writes Task_by_Task/taskNNN/LIVE_ALGORITHM.md with:
  - Producer / graph name (often describes the algorithm)
  - Node count, opset
  - Top ops by frequency
  - Initializer names (often hint at algorithm structure)
  - First few nodes (entry point of the algorithm)

This is the source-of-truth doc for the next agent: read THIS before
writing a prototype, not the visible test set examples. The visible
test set is a biased subset of the hidden one (lesson 2026-05-11).

Usage:
    python tools/document_live_algorithm.py --tid 22 --tid 260 ...
    python tools/document_live_algorithm.py --hidden-bad  # all hidden-bad
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

os.environ['ORT_LOGGING_LEVEL'] = '4'

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / 'tools'))

import onnx


DEFAULT_LIVE = Path(r"<NOGIT_ROOT>\submissions\alpha_b_t287v4_5653_67\submission.zip")
TASK_BY_TASK = REPO / "The_400_Tasks" / "Task_by_Task"
HIDDEN_BAD = {4, 22, 161, 245, 260, 270, 280, 383, 396}


def document_one(tid: int, live_zip: Path) -> str:
    """Returns the markdown content describing live task{tid}.onnx."""
    name = f"task{tid:03d}.onnx"
    with ZipFile(live_zip) as z:
        blob = z.read(name)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.write(blob)
    tmp.close()
    try:
        m = onnx.load(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    nodes = m.graph.node
    inits = m.graph.initializer
    n_nodes = len(nodes)
    n_inits = len(inits)
    opset = m.opset_import[0].version
    producer = m.producer_name or "(anon)"
    gname = m.graph.name or "(unnamed)"

    op_counter = Counter(n.op_type for n in nodes)
    init_names = [ini.name for ini in inits[:30]]

    # Try to guess algorithm from producer/graph name + ops
    name_hints = []
    s = (producer + " " + gname).lower()
    if "marker" in s:
        name_hints.append("anchor/marker-pattern (silent-fail-trap family)")
    if "diagonal" in s:
        name_hints.append("diagonal-line / cell-coordinate rule")
    if "boundary" in s:
        name_hints.append("boundary-touching condition")
    if "compact" in s or "direction" in s:
        name_hints.append("directional projection / compaction")
    if "rectangle" in s or "lrrc" in s.lower() or any("lrrc_" in n for n in init_names):
        name_hints.append("largest-rectangle scoring")

    lines = []
    lines.append(f"# task{tid:03d} — LIVE ALGORITHM")
    lines.append("")
    lines.append(f"> **Hidden-bad task.** Reading this BEFORE writing a prototype is")
    lines.append(f"> mandatory. The live ONNX scores well on Kaggle's hidden test set —")
    lines.append(f"> any replacement must produce identical outputs on hidden inputs,")
    lines.append(f"> which means matching the live's algorithm exactly. Visible + arc-gen")
    lines.append(f"> verify is necessary but NOT sufficient (see Logs/lessons.md")
    lines.append(f"> 2026-05-11 evening entry).")
    lines.append("")
    lines.append(f"## Provenance")
    lines.append("")
    lines.append(f"- **Source**: live `task{tid:03d}.onnx` in active submission zip")
    lines.append(f"  (`{live_zip.name}`)")
    lines.append(f"- **Producer**: `{producer}`")
    lines.append(f"- **Graph name**: `{gname}`")
    lines.append(f"- **Opset**: {opset}")
    lines.append(f"- **Nodes**: {n_nodes}")
    lines.append(f"- **Initializers**: {n_inits}")
    lines.append("")

    if name_hints:
        lines.append(f"## Algorithm hints (from naming)")
        lines.append("")
        for h in name_hints:
            lines.append(f"- {h}")
        lines.append("")

    lines.append(f"## Op frequency")
    lines.append("")
    lines.append("| Op | Count |")
    lines.append("|---|---:|")
    for op, c in op_counter.most_common(15):
        lines.append(f"| `{op}` | {c} |")
    lines.append("")

    lines.append(f"## Initializer names (first 30)")
    lines.append("")
    lines.append("```")
    for n in init_names:
        lines.append(n)
    lines.append("```")
    lines.append("")

    lines.append(f"## First 12 nodes (algorithm entry point)")
    lines.append("")
    lines.append("```")
    for i, n in enumerate(nodes[:12]):
        attrs = {}
        for a in n.attribute:
            if a.type == onnx.AttributeProto.INTS:
                attrs[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.INT:
                attrs[a.name] = a.i
        attr_s = " " + str(attrs) if attrs else ""
        inputs = ", ".join(n.input)[:60]
        lines.append(f"  {i:>3}: {n.op_type:<14}{attr_s} ({inputs}) -> {n.output[0]}")
    lines.append("```")
    lines.append("")

    lines.append(f"## How to validate a prototype against this")
    lines.append("")
    lines.append("```")
    lines.append(f"python tools/verify_against_live.py \\")
    lines.append(f"  --proto path/to/prototype.onnx \\")
    lines.append(f"  --tid {tid}")
    lines.append("# Optional legacy debug mode only:")
    lines.append(f"# python tools/verify_against_live.py --proto path/to/prototype.onnx --tid {tid} --cropped-decode")
    lines.append("```")
    lines.append("")
    lines.append("**The default output should be `Prototype vs LIVE : N/N` on the official")
    lines.append("full tensor comparison.** Even one diverging pair on visible+arc-gen means")
    lines.append("the algorithm is different and will likely be hidden-bad. The legacy")
    lines.append("cropped-decode mode is for debugging historical results only and must not")
    lines.append("gate submit-captain requeues. And remember: matching on visible is")
    lines.append("NECESSARY but not SUFFICIENT (task004 v3 matched 265/265 visible yet")
    lines.append("was -13.39 LB on hidden). For the strongest signal, the prototype")
    lines.append("should follow the algorithm DERIVED FROM THIS FILE, not from visible")
    lines.append("examples.")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tid", action="append", type=int, default=[],
                   help="Task ID. Repeat for multiple.")
    p.add_argument("--hidden-bad", action="store_true",
                   help="Document all current HIDDEN_BAD_TIDS tasks.")
    p.add_argument("--live-zip", type=Path, default=DEFAULT_LIVE)
    args = p.parse_args(argv)

    tids = list(args.tid)
    if args.hidden_bad:
        tids = sorted(set(tids) | HIDDEN_BAD)
    if not tids:
        print("Specify --tid or --hidden-bad")
        return 1

    for tid in tids:
        content = document_one(tid, args.live_zip)
        out_dir = TASK_BY_TASK / f"task{tid:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "LIVE_ALGORITHM.md"
        out_path.write_text(content, encoding="utf-8")
        print(f"task{tid:03d}: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
