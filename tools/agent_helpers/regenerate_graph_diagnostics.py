"""Regenerate the auto-generated `Graph Diagnostic` section in each
The_400_Tasks/Task_by_Task/taskNNN.md.

Each diagnostic is delimited by markers:
  <!-- BEGIN AUTO-GRAPH-DIAG -->
  ...
  <!-- END AUTO-GRAPH-DIAG -->

If the markers exist in the .md, the section is replaced in place.
If not, the section is appended.

Sources:
  - submissions/submission.zip (canonical) — for graph structure + initializers.
  - data/competitions/neurogolf-2026/taskNNN.json — for max active grid.
  - tools/spread_score_isolated.py output — for cost/memory/params.

Run periodically when canonical changes:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python tools/agent_helpers/regenerate_graph_diagnostics.py
"""
from __future__ import annotations

import json
import re
import sys
import subprocess
import zipfile
from pathlib import Path
from collections import Counter

import onnx
from onnx import numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ZIP_PATH = REPO_ROOT / "submissions" / "submission.zip"
TASKS_DIR = REPO_ROOT / "The_400_Tasks" / "Task_by_Task"
DATA_DIR = REPO_ROOT / "data" / "competitions" / "neurogolf-2026"
PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

BEGIN_MARKER = "<!-- BEGIN AUTO-GRAPH-DIAG -->"
END_MARKER = "<!-- END AUTO-GRAPH-DIAG -->"


# ONNX TensorProto dtype enum → numpy itemsize and short name
DTYPE_INFO = {
    1: ("F32", 4),
    2: ("UINT8", 1),
    3: ("INT8", 1),
    4: ("UINT16", 2),
    5: ("INT16", 2),
    6: ("INT32", 4),
    7: ("INT64", 8),
    9: ("BOOL", 1),
    10: ("F16", 2),
    11: ("F64", 8),
}


def dtype_short(dtype_enum):
    info = DTYPE_INFO.get(dtype_enum)
    return info[0] if info else f"dtype{dtype_enum}"


def dtype_itemsize(dtype_enum):
    info = DTYPE_INFO.get(dtype_enum)
    return info[1] if info else 4


def shape_str(dims):
    return "[" + ", ".join(str(d) for d in dims) + "]"


def static_tensor_bytes(value_info):
    """Compute the static byte size of a value_info tensor. Returns 0 if shape has unknowns."""
    tt = value_info.type.tensor_type
    if not tt.shape.dim:
        return 0
    n = 1
    for d in tt.shape.dim:
        if d.dim_value <= 0:
            return 0
        n *= d.dim_value
    return n * dtype_itemsize(tt.elem_type)


def load_task_json(tid):
    """Return the task's JSON data (train + test + arc-gen pairs)."""
    p = DATA_DIR / f"{tid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def max_active_grid(task_json):
    """Compute max (H, W) across all train+test+arc-gen pairs."""
    if not task_json:
        return None
    max_h = max_w = 0
    for split in ("train", "test", "arc-gen"):
        for pair in task_json.get(split, []):
            for key in ("input", "output"):
                grid = pair.get(key) or []
                if grid:
                    max_h = max(max_h, len(grid))
                    max_w = max(max_w, len(grid[0]) if grid[0] else 0)
    return (max_h, max_w) if max_h and max_w else None


def score_onnx(onnx_path):
    """Call tools/spread_score_isolated.py to get cost/memory/params."""
    try:
        r = subprocess.run([str(PY), str(REPO_ROOT / "tools" / "spread_score_isolated.py"),
                            str(onnx_path)],
                           capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout.strip())
    except Exception:
        return None


def analyze_graph(model):
    """Extract structural facts from an ONNX model."""
    g = model.graph

    # Initializer stats
    inits = []
    for init in g.initializer:
        params = 1
        for d in init.dims:
            params *= d
        inits.append({
            "name": init.name,
            "dtype": dtype_short(init.data_type),
            "shape": shape_str(init.dims),
            "params": params,
        })
    inits.sort(key=lambda x: -x["params"])

    # Node stats
    node_list = []
    op_counter = Counter()
    for i, n in enumerate(g.node):
        op_counter[n.op_type] += 1
        node_list.append({
            "idx": i,
            "op": n.op_type,
            "name": n.name or "",
            "inputs": list(n.input),
            "outputs": list(n.output),
        })

    # Intermediates by static byte size (using value_info)
    intermediates = []
    for vi in g.value_info:
        bytes_ = static_tensor_bytes(vi)
        if bytes_ > 0:
            intermediates.append({
                "name": vi.name,
                "dtype": dtype_short(vi.type.tensor_type.elem_type),
                "shape": shape_str([d.dim_value for d in vi.type.tensor_type.shape.dim]),
                "bytes": bytes_,
            })
    intermediates.sort(key=lambda x: -x["bytes"])

    # Opset
    opset = "?"
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            opset = imp.version
            break

    # Output dtype
    out_dtype = dtype_short(g.output[0].type.tensor_type.elem_type) if g.output else "?"

    # Flags
    op_types = set(n.op_type for n in g.node)
    first_node = g.node[0] if g.node else None
    last_node = g.node[-1] if g.node else None
    second_to_last = g.node[-2] if len(g.node) >= 2 else None

    ends_with_cast_to_f32 = (
        last_node and last_node.op_type == "Cast"
        and any(a.name == "to" and a.i == 1 for a in last_node.attribute)
        and last_node.output and last_node.output[0] == "output"
    )

    first_is_slice = first_node and first_node.op_type == "Slice"

    flags = {
        "has_Conv": "Conv" in op_types,
        "has_MatMul": "MatMul" in op_types or "Gemm" in op_types,
        "has_ReduceSum": "ReduceSum" in op_types,
        "has_ReduceMax": "ReduceMax" in op_types,
        "has_Where": "Where" in op_types,
        "has_Gather": "Gather" in op_types,
        "has_OneHot": "OneHot" in op_types,
        "first_node": (first_node.op_type if first_node else "?"),
        "last_node": (last_node.op_type if last_node else "?"),
        "opset": opset,
        "graph_output_dtype": out_dtype,
        "ends_with_Cast_to_F32_output": ends_with_cast_to_f32,
        "first_node_is_Slice": first_is_slice,
        "n_nodes": len(g.node),
        "n_inits": len(g.initializer),
    }

    return {
        "inits": inits,
        "nodes": node_list,
        "intermediates": intermediates,
        "op_counter": op_counter,
        "flags": flags,
    }


def suggest_techniques(flags, max_grid, cost):
    """Suggest techniques based on graph flags."""
    suggestions = []
    if max_grid and max(max_grid) < 30 and flags.get("first_node") in ("Cast", "Conv"):
        suggestions.append(f"**crop_to_active_grid** — max active grid {max_grid[0]}×{max_grid[1]} < 30; see `Onnx_Advanced/exemplars/03_crop_and_f16/task117_crop15_f16.md`")
    if flags.get("ends_with_Cast_to_F32_output"):
        suggestions.append("**F16-output trick** — graph ends with Cast(F16, F32) → output; see `Onnx_Advanced/exemplars/03_crop_and_f16/task384_f16_drop.md`")
    if not flags.get("has_Conv") and not flags.get("has_MatMul"):
        suggestions.append("**input → BOOL whole-graph** candidate — no Conv/MatMul/Gemm; see `Onnx_Advanced/exemplars/01_whole_graph_rewrites/task376_input_to_bool.md`")
    if flags.get("first_node_is_Slice"):
        suggestions.append("**WARNING: pre-crop false positive** — first node is already Slice; verify the model isn't already cropped before applying crop technique")
    if flags.get("opset") == 10:
        suggestions.append("`opset_bump_pad_uint8` candidate if model has Cast(bool, F32) → Pad chain — see `Onnx_Advanced/exemplars/02_bool_first/task319_opset_bump_pad_uint8.md`")
    if flags.get("has_Gather") and not flags.get("has_Conv"):
        suggestions.append("F16 per-channel placement candidate — see `Onnx_Advanced/exemplars/03_crop_and_f16/task055_per_channel.md`")
    if not suggestions:
        suggestions.append("Standard polish via `tools/spread_scan.py --apply` should be the first attempt; this graph doesn't immediately match a known high-leverage pattern.")
    return suggestions


def render_diagnostic(tid, score, structure, task_json):
    """Render the full diagnostic markdown section."""
    flags = structure["flags"]
    max_grid = max_active_grid(task_json)

    lines = [BEGIN_MARKER, ""]
    lines.append(f"## Graph Diagnostic (auto-generated)")
    lines.append("")

    # Header
    if score:
        lines.append(f"**Cost**: {score['cost']:,} | **Memory**: {score['memory']:,} | **Params**: {score['params']:,} | **Points**: {score['points']:.4f}")
    else:
        lines.append("**Cost**: (could not be scored)")
    lines.append(f"**Nodes**: {flags['n_nodes']} | **Initializers**: {flags['n_inits']} | **Opset**: {flags['opset']} | **Graph output dtype**: {flags['graph_output_dtype']}")
    if max_grid:
        lines.append(f"**Max active grid (from `{tid}.json`)**: {max_grid[0]}×{max_grid[1]}")
    lines.append("")

    # Quick-look flags
    lines.append("### Quick-look flags")
    lines.append("")
    flag_yn = lambda v: "✓" if v else "✗"
    lines.append(f"- has_Conv: {flag_yn(flags['has_Conv'])}")
    lines.append(f"- has_MatMul: {flag_yn(flags['has_MatMul'])}")
    lines.append(f"- has_ReduceSum: {flag_yn(flags['has_ReduceSum'])}")
    lines.append(f"- has_ReduceMax: {flag_yn(flags['has_ReduceMax'])}")
    lines.append(f"- has_Where: {flag_yn(flags['has_Where'])}")
    lines.append(f"- has_Gather: {flag_yn(flags['has_Gather'])}")
    lines.append(f"- has_OneHot: {flag_yn(flags['has_OneHot'])}")
    lines.append(f"- first_node: `{flags['first_node']}`")
    lines.append(f"- last_node: `{flags['last_node']}`")
    lines.append(f"- ends_with_Cast(F16→F32)→output: {flag_yn(flags['ends_with_Cast_to_F32_output'])}")
    lines.append(f"- first_node_is_Slice: {flag_yn(flags['first_node_is_Slice'])}")
    lines.append("")

    # Op-type histogram
    lines.append("### Op-type histogram")
    lines.append("")
    lines.append("| op | count |")
    lines.append("|---|---:|")
    for op, n in structure["op_counter"].most_common():
        lines.append(f"| `{op}` | {n} |")
    lines.append("")

    # Initializers (top 10 by params)
    lines.append("### Initializers (top 10 by params)")
    lines.append("")
    lines.append("| name | dtype | shape | params |")
    lines.append("|---|---|---|---:|")
    for init in structure["inits"][:10]:
        name = init["name"][:30]  # truncate long names
        lines.append(f"| `{name}` | {init['dtype']} | {init['shape']} | {init['params']:,} |")
    if len(structure["inits"]) > 10:
        lines.append(f"| ... ({len(structure['inits']) - 10} more) | | | |")
    lines.append("")

    # Top 5 largest intermediates
    if structure["intermediates"]:
        lines.append("### Top 5 largest intermediates (static value_info bytes)")
        lines.append("")
        lines.append("| name | dtype | shape | bytes |")
        lines.append("|---|---|---|---:|")
        for it in structure["intermediates"][:5]:
            name = it["name"][:30]
            lines.append(f"| `{name}` | {it['dtype']} | {it['shape']} | {it['bytes']:,} |")
        lines.append("")
        lines.append("Note: grader charges max(static, runtime-on-real-inputs)/tensor — techniques.md §39 MECHANISM.")
        lines.append("")

    # Technique routing is by_symptom.md's job (audit#2 Phase-1)
    lines.append("*Technique routing: `Onnx_Advanced/by_symptom.md` (+ INDEX verdict list). Flags above feed it.*")
    lines.append("")

    # Node table dropped (audit#2 Phase-1, 2026-06-05): the .onnx is canonical.
    lines.append(f"*Node list ({flags['n_nodes']} nodes) omitted — the .onnx is canonical; run `tools/agent_helpers/inspect_graph.py {tid}` for the full sequence.*")
    lines.append("")

    lines.append(END_MARKER)
    return "\n".join(lines)


def update_task_md(tid, diagnostic_section):
    """Update the task's .md file by replacing or appending the diagnostic section."""
    md_path = TASKS_DIR / f"{tid}.md"
    if not md_path.exists():
        # Create new file with just the diagnostic
        content = f"# {tid}\n\n## Attempts\n\n_(no manual attempts yet)_\n\n{diagnostic_section}\n"
        md_path.write_text(content, encoding="utf-8")
        return "created"

    text = md_path.read_text(encoding="utf-8", errors="replace")

    if BEGIN_MARKER in text and END_MARKER in text:
        # Replace existing section
        pattern = re.compile(re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
        new_text = pattern.sub(diagnostic_section, text)
        md_path.write_text(new_text, encoding="utf-8")
        return "replaced"
    else:
        # Append
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + diagnostic_section + "\n"
        md_path.write_text(text, encoding="utf-8")
        return "appended"


def main(argv=None):
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found", file=sys.stderr)
        return 1

    print(f"Loading {ZIP_PATH}...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        task_names = sorted([n for n in z.namelist() if re.match(r"task\d{3}\.onnx", n)])
    print(f"Found {len(task_names)} task ONNXs in zip.", flush=True)

    # If specific tasks passed on CLI, filter
    if argv and len(argv) > 1:
        filt = set(argv[1:])
        task_names = [t for t in task_names if t.replace(".onnx", "") in filt]
        print(f"Filtered to {len(task_names)} task(s) by CLI args.", flush=True)

    stats = Counter()
    for i, name in enumerate(task_names, 1):
        tid = name.replace(".onnx", "")
        # Extract ONNX to temp
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            onnx_bytes = z.read(name)
        tmp_path = REPO_ROOT / f"_tmp_graph_diag_{tid}.onnx"
        tmp_path.write_bytes(onnx_bytes)

        try:
            model = onnx.load(str(tmp_path))
            score = score_onnx(tmp_path)
            structure = analyze_graph(model)
            task_json = load_task_json(tid)
            diagnostic = render_diagnostic(tid, score, structure, task_json)
            verdict = update_task_md(tid, diagnostic)
            stats[verdict] += 1
        except Exception as e:
            stats["error"] += 1
            print(f"  {tid}: ERROR — {type(e).__name__}: {e}", flush=True)
        finally:
            tmp_path.unlink(missing_ok=True)

        if i % 20 == 0 or i == len(task_names):
            print(f"  [{i}/{len(task_names)}] verdicts so far: {dict(stats)}", flush=True)

    print(f"\nFinal: {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
