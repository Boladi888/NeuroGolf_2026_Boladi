"""Inspect and lightly repair external ONNX submissions.

Triages a candidate ONNX (or a directory of them) into one of:

  * READY                — passes the validator's static checks as-is.
    No repair needed; can be used directly in a blend.
  * REPAIRABLE-RENAME    — only issue is non-standard tensor names
    (input/output != "input"/"output"). Apply the I/O-rename pass and
    re-check. Recipe sourced from
    `Research_Archive/notes/deep-read-logic-driven-ensembe.md`.
  * BANNED-OP            — graph contains an op in `_EXCLUDED_OP_TYPES`
    (LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS).
    NOT REPAIRED. The tool reports which op and exits.
  * DYNAMIC-SHAPE        — graph input or output has a dim_param /
    missing dim_value, would fail `score_network`'s static-shape gate.
    NOT REPAIRED.
  * IDENTITY-PLACEHOLDER — single-node `Identity(input)->output`.
    Functionally a "give up" submission; technically valid but scores 0
    on every task with a non-trivial rule.
  * BROKEN               — fails to parse / load via `onnx.load`.

Strict policy: the tool NEVER modifies a graph beyond the I/O-rename pass.
We do not strip banned ops, do not collapse dynamic shapes, do not rebuild
graph structure. If a candidate is BANNED-OP / DYNAMIC-SHAPE / BROKEN, the
tool reports WHY but does not produce a "repaired" version. Repair pass is
limited to renaming `graph.input[0].name` and `graph.output[0].name` plus
all internal references.

Usage:
    python tools/onnx_repair.py PATH/model.onnx                    # single triage (table)
    python tools/onnx_repair.py PATH/model.onnx --json             # single triage (JSON)
    python tools/onnx_repair.py PATH/model.onnx --rename-out OUT.onnx  # single + repair
    python tools/onnx_repair.py --bulk DIR --report CSV.csv         # bulk triage report
    python tools/onnx_repair.py --bulk DIR --out OUTDIR             # bulk triage + repair (writes READY + RENAMED to OUTDIR)
"""

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter

import onnx

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent

_BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def _has_dynamic_dims(value_info):
    """Return True if the value_info's tensor shape has any dim_param or non-positive dim_value."""
    if not value_info.type.tensor_type.HasField("shape"):
        return True
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") and dim.dim_param:
            return True
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return True
    return False


def _io_naming_ok(model):
    """Return True if graph.input[0].name == 'input' AND graph.output[0].name == 'output'."""
    if not model.graph.input or not model.graph.output:
        return False
    return model.graph.input[0].name == "input" and model.graph.output[0].name == "output"


def _is_identity_placeholder(model):
    """Single-node graph that is just `Identity(input) -> output`."""
    nodes = list(model.graph.node)
    if len(nodes) != 1:
        return False
    n = nodes[0]
    return (
        n.op_type == "Identity"
        and list(n.input) == ["input"]
        and list(n.output) == ["output"]
    )


def _banned_op_in_graph(model):
    """Return the first banned op_type found, or None."""
    for node in model.graph.node:
        if node.op_type.upper() in _BANNED_OPS:
            return node.op_type
    return None


def triage(onnx_path):
    """Classify an ONNX file into READY / REPAIRABLE-RENAME / BANNED-OP / DYNAMIC-SHAPE / IDENTITY-PLACEHOLDER / BROKEN.

    Returns dict {path, status, reason, ops, file_size}.
    """
    p = pathlib.Path(onnx_path)
    if not p.exists():
        return {"path": str(p), "status": "BROKEN", "reason": "file not found", "ops": {}, "file_size": 0}

    try:
        model = onnx.load(str(p))
    except Exception as e:  # noqa: BLE001
        return {"path": str(p), "status": "BROKEN", "reason": f"onnx.load: {e}", "ops": {}, "file_size": p.stat().st_size}

    file_size = p.stat().st_size
    ops = dict(Counter(node.op_type for node in model.graph.node))

    banned = _banned_op_in_graph(model)
    if banned:
        return {"path": str(p), "status": "BANNED-OP", "reason": f"contains banned op {banned}", "ops": ops, "file_size": file_size}

    for vi in [model.graph.input[0], model.graph.output[0]]:
        if _has_dynamic_dims(vi):
            return {"path": str(p), "status": "DYNAMIC-SHAPE", "reason": f"non-static shape on tensor '{vi.name}'", "ops": ops, "file_size": file_size}

    if _is_identity_placeholder(model):
        return {"path": str(p), "status": "IDENTITY-PLACEHOLDER", "reason": "single Identity(input) -> output node", "ops": ops, "file_size": file_size}

    if _io_naming_ok(model):
        return {"path": str(p), "status": "READY", "reason": "passes static checks; tensor names already correct", "ops": ops, "file_size": file_size}

    return {
        "path": str(p),
        "status": "REPAIRABLE-RENAME",
        "reason": f"input.name={model.graph.input[0].name!r} or output.name={model.graph.output[0].name!r} != 'input'/'output'",
        "ops": ops,
        "file_size": file_size,
    }


def repair_io_naming(onnx_path):
    """Apply the I/O-rename pass per Safe Blend recipe. Returns (ok, fixed_bytes_or_None, reason).

    Pre-condition: triage(onnx_path)['status'] == 'REPAIRABLE-RENAME'.
    Post-condition: returned bytes parse as ONNX with input/output named correctly,
                    OR returns (False, None, ...) if rename couldn't produce a working graph.
    """
    model = onnx.load(str(onnx_path))
    g = model.graph

    rename_map = {}
    if g.input and g.input[0].name != "input":
        rename_map[g.input[0].name] = "input"
    if g.output and g.output[0].name != "output":
        rename_map[g.output[0].name] = "output"

    if not rename_map:
        return False, None, "naming-already-correct (no rename needed)"

    if g.input:
        g.input[0].name = "input"
    if g.output:
        g.output[0].name = "output"
    for node in g.node:
        for i, name in enumerate(node.input):
            if name in rename_map:
                node.input[i] = rename_map[name]
        for i, name in enumerate(node.output):
            if name in rename_map:
                node.output[i] = rename_map[name]
    for vi in g.value_info:
        if vi.name in rename_map:
            vi.name = rename_map[vi.name]
    for init in g.initializer:
        if init.name in rename_map:
            init.name = rename_map[init.name]

    try:
        fixed_bytes = model.SerializeToString()
    except Exception as e:  # noqa: BLE001
        return False, None, f"serialize-fail: {e}"

    try:
        m2 = onnx.load_model_from_string(fixed_bytes)
        if m2.graph.input[0].name != "input" or m2.graph.output[0].name != "output":
            return False, None, "rename-incomplete after serialization"
    except Exception as e:  # noqa: BLE001
        return False, None, f"reload-fail: {e}"

    return True, fixed_bytes, "renamed"


def _bulk(in_dir, out_dir=None, report_csv=None, verbose=True):
    """Triage every task*.onnx in `in_dir`. Optionally repair to `out_dir` and emit `report_csv`."""
    in_dir = pathlib.Path(in_dir)
    files = sorted(in_dir.rglob("task*.onnx"))
    if not files:
        print(f"No task*.onnx under {in_dir}", file=sys.stderr)
        return

    if out_dir:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = Counter()

    for f in files:
        r = triage(f)
        counts[r["status"]] += 1
        repaired = ""
        if out_dir and r["status"] in ("READY", "REPAIRABLE-RENAME"):
            target = out_dir / f.name
            if r["status"] == "READY":
                target.write_bytes(f.read_bytes())
                repaired = "as-is"
            else:
                ok, fixed, reason = repair_io_naming(f)
                if ok:
                    target.write_bytes(fixed)
                    repaired = "renamed"
                else:
                    repaired = f"rename-failed: {reason}"

        if verbose:
            print(f"  {r['status']:<22} {f.name:<20} {r['file_size']:>6}  {r['reason']}")

        rows.append({
            "task_file": f.name,
            "status": r["status"],
            "reason": r["reason"],
            "file_size": r["file_size"],
            "n_ops": sum(r["ops"].values()),
            "ops": ",".join(f"{k}{v}" if v > 1 else k for k, v in sorted(r["ops"].items())),
            "repaired": repaired,
        })

    if verbose:
        print()
        print("Summary:")
        for status, c in counts.most_common():
            print(f"  {status:<22} {c}")

    if report_csv:
        report_csv = pathlib.Path(report_csv)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        with report_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"  Report: {report_csv.relative_to(NG_ROOT).as_posix() if report_csv.is_relative_to(NG_ROOT) else report_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", nargs="?", help="Single ONNX file to triage (or omit with --bulk)")
    parser.add_argument("--rename-out", type=str, default="", help="Write the I/O-renamed ONNX to this path (single-file mode)")
    parser.add_argument("--bulk", type=str, default="", help="Triage every task*.onnx under this directory (recursive)")
    parser.add_argument("--out", type=str, default="", help="Output dir for repaired ONNXs (bulk mode)")
    parser.add_argument("--report", type=str, default="", help="CSV report path (bulk mode)")
    parser.add_argument("--json", action="store_true", help="JSON output (single-file mode)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.bulk:
        _bulk(args.bulk, out_dir=args.out or None, report_csv=args.report or None, verbose=not args.quiet)
        return

    if not args.path:
        parser.print_help()
        sys.exit(2)

    r = triage(args.path)
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"path:    {r['path']}")
        print(f"status:  {r['status']}")
        print(f"reason:  {r['reason']}")
        print(f"size:    {r['file_size']} bytes")
        print(f"ops:     {r['ops']}")

    if args.rename_out and r["status"] == "REPAIRABLE-RENAME":
        ok, fixed, reason = repair_io_naming(args.path)
        if ok:
            pathlib.Path(args.rename_out).write_bytes(fixed)
            print(f"renamed: wrote {args.rename_out} ({len(fixed)} bytes)")
        else:
            print(f"rename failed: {reason}", file=sys.stderr)
            sys.exit(1)
    elif args.rename_out:
        print(f"rename-out skipped: status is {r['status']!r}, not REPAIRABLE-RENAME", file=sys.stderr)


if __name__ == "__main__":
    main()
