"""Add relational / derivable columns to NeuroGolf_Task_Status.xlsx.

These columns help with:
  - **Batch work** — group tasks by family / transformations / op_signature
  - **Cascade detection** — see how many tasks share a source or pattern
  - **Fix discovery** — tasks with similar ONNX structure that admit the
    same cost-rewrite

All values are auto-derived. Re-running this tool overwrites these columns
without touching user-edited columns (last_attempted_at, time_spent_attempting,
notes, family, transformations, etc.).

Columns added (sortable / consistent format):

  ops_count                  int — total nodes in current ONNX (0 if missing)
  op_signature               str — top 3 ops by count, "Op:N|Op:N|Op:N"
                                   alphabetically tied ops sort by name
  unique_ops                 str — sorted "|"-joined unique op_types
                                   (e.g. "Add|Cast|Conv|Slice")
  graph_name                 str — ONNX graph.name (often hints at builder)
  producer                   str — ONNX producer_name (often hints at origin)
  grid_resized               bool — true if any official pair has output shape
                                   different from input shape
  similar_tasks              str — comma-joined taskNNN list of tasks sharing
                                   same family+transformations.
                                   "" if none / family missing.
  similar_task_count         int — count of similar_tasks (for sorting)
  current_src_shared_with    int — count of OTHER tasks with same current_src
                                   (high = cascade opportunity)
  solved_status              str — enum: Solved | NotSolved | HiddenBad
  hidden_bad_history         str — Yes / No (silent-fail-trap list maintained
                                   in this script + Logs/lessons.md)
  live_arch                  str — lookup | lookup_confirmed_zero | trained_nn |
                                   structural, detected from current ONNX
                                   initializer/op signatures
  cherry13b_tested           str — manual field; defaults to not_yet
  structural_attempt         str — manual field; defaults to not_yet
  unlock_potential           str — HIGH iff live_arch=lookup and cherry13b is
                                   not_yet/regresses; else LOW
  structural_kaggle_delta    str — manual field; filled after Kaggle ship
  prototype_path             str — ;-joined in-repo task prototype artifacts
                                   (.py/.onnx), or "" if none

Usage:
    python tools/enrich_task_state.py [--xlsx PATH] [--zip PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import openpyxl
import onnx


REPO = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = REPO / "The_400_Tasks" / "NeuroGolf_Task_Status.xlsx"
DEFAULT_ZIP = REPO / "submissions" / "submission.zip"
DEFAULT_DATA_DIR = REPO / "data" / "competitions" / "neurogolf-2026"
ORGANIC_PROTOTYPES = REPO / "Research_Archive" / "organic_prototypes"

# Tasks confirmed hidden-bad on Kaggle (silent-fail trap). Maintained here +
# in Logs/lessons.md. Add task IDs as you discover more.
HIDDEN_BAD_TIDS = {
    4,     # parallelogram-lean shift; 265/265 visible, organic -13.39 LB (2026-05-11)
    22,    # anchor-5 neighbor color map (this session, -13.54 LB)
    161,   # 4-cell pair lines; 266/266 visible, organic -12.80 LB (2026-05-11)
    245,   # translate shape to marker center; 267/267 visible, organic -12.93 LB (2026-05-11)
    270,   # anchor-cross color partition (PR #64 attempt, -0.54 LB)
    280,   # rect+marker direction extension (PR #66 attempt, -12.29 LB)
    260,   # line + triangle markers + diagonals (PR #66 attempt, -15.38 LB swap)
    396,   # predicted hidden-bad (PR #66, deferred)
    383,   # task383 v2 hidden-bad on Kaggle public LB (-12.31)
}

# Confirmed by true cherry13b Identity/conv-identity diagnostics in the
# library notes. These may still have lookup-shaped LIVE graphs, but they are
# not "unlock-potential" lookup-live targets anymore.
CONFIRMED_CHERRY13B_ZERO_TIDS = {
    18, 133, 157, 182, 255, 324, 366,
}


def task_id_from_filename(name: str) -> int | None:
    """taskNNN.onnx -> NNN."""
    if not (name.startswith("task") and name.endswith(".onnx")):
        return None
    try:
        return int(name[4:7])
    except ValueError:
        return None


def collect_onnx_features(zip_path: Path) -> dict[int, dict]:
    """For each task ONNX in the submission zip, extract:
    - ops_count
    - op_counter (Counter[op_type])
    - graph_name
    - producer
    Returns {tid: {...}}.
    """
    out: dict[int, dict] = {}
    if not zip_path.exists():
        return out
    with ZipFile(zip_path) as z:
        for name in z.namelist():
            tid = task_id_from_filename(name)
            if tid is None:
                continue
            data = z.read(name)
            try:
                m = onnx.load_from_string(data)
            except Exception:
                out[tid] = {"ops_count": 0}
                continue
            counter: Counter = Counter()
            for node in m.graph.node:
                counter[node.op_type] += 1
            init_shapes = [tuple(int(d) for d in init.dims) for init in m.graph.initializer]
            out[tid] = {
                "ops_count": sum(counter.values()),
                "op_counter": counter,
                "init_shapes": init_shapes,
                "live_arch": detect_live_arch(tid, counter, init_shapes),
                "graph_name": m.graph.name or "",
                "producer": m.producer_name or "",
            }
    return out


def format_op_signature(counter: Counter, top_k: int = 3) -> str:
    if not counter:
        return ""
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return "|".join(f"{op}:{n}" for op, n in items)


def format_unique_ops(counter: Counter) -> str:
    return "|".join(sorted(counter.keys())) if counter else ""


def _grid_shape(grid) -> tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def derive_grid_resized(data_dir: Path) -> dict[int, bool]:
    """Return {task_id: any(input_shape != output_shape)} from official JSON."""
    out: dict[int, bool] = {}
    for tid in range(401):
        path = data_dir / f"task{tid:03d}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pairs = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
        out[tid] = any(
            _grid_shape(pair.get("input", [])) != _grid_shape(pair.get("output", []))
            for pair in pairs
        )
    return out


def detect_live_arch(tid: int, counter: Counter, init_shapes: list[tuple[int, ...]]) -> str:
    """Classify the current/LIVE ONNX architecture for cherry13b tracking.

    High-confidence lookup signatures are the same family as
    tools/multi_task_sweeps/cherry13b_scan.py: template tables with
    N≈266 plus a lookup/selector path. Lower-confidence large-template hits
    are left structural so this column marks only actionably clear lookup-live
    targets.
    """
    has_flatten = counter.get("Flatten", 0) > 0
    has_matmul = counter.get("MatMul", 0) > 0
    has_argmax = counter.get("ArgMax", 0) > 0
    has_scatter = counter.get("ScatterElements", 0) > 0 or counter.get("ScatterND", 0) > 0
    has_gather = counter.get("Gather", 0) + counter.get("GatherND", 0) > 0
    has_gathernd = counter.get("GatherND", 0) > 0

    template_first_dim = any(
        len(shape) == 2 and 200 <= shape[0] <= 300 and shape[1] >= 2
        for shape in init_shapes
    )
    has_binary_probes = (
        counter.get("Greater", 0) + counter.get("Less", 0) + counter.get("Equal", 0)
    ) >= 10
    has_packed_rects = counter.get("MatMul", 0) >= 2 and counter.get("BitShift", 0) > 0

    lookup_high_conf = (
        (has_flatten and has_matmul and has_argmax and has_scatter)
        or (has_gather and has_gathernd and template_first_dim)
        or (has_binary_probes and has_packed_rects and template_first_dim)
    )
    if lookup_high_conf:
        return "lookup_confirmed_zero" if tid in CONFIRMED_CHERRY13B_ZERO_TIDS else "lookup"

    # Future-facing detector for explicit trained-weight architectures. Keep
    # this conservative: small Gemm classifier tails can be structural plumbing.
    large_dense_init = any(
        len(shape) == 2 and shape[0] >= 64 and shape[1] >= 64
        for shape in init_shapes
    )
    if (counter.get("ConvTranspose", 0) > 0 and large_dense_init) or (
        counter.get("Gemm", 0) >= 3 and large_dense_init
    ):
        return "trained_nn"

    return "structural"


PROTOTYPE_SEARCH_DIRS = [
    ORGANIC_PROTOTYPES,  # Research_Archive/organic_prototypes/ — primary, in-repo
    # Scratch dirs in .notongit/ — work-in-progress prototypes that haven't
    # been promoted to Research_Archive/organic_prototypes/ yet.
    Path(os.environ.get("NEUROGOLF_NOGIT_ROOT", REPO / ".notongit" / "NeuroGolf_nogit")) / "submissions" / "_alpha_a_scratch",
    Path(os.environ.get("NEUROGOLF_NOGIT_ROOT", REPO / ".notongit" / "NeuroGolf_nogit")) / "submissions" / "_alpha_b_scratch",
]


def find_prototype(tid: int) -> str:
    """Return ;-joined prototype artifact paths for a task.

    Scans:
    - The_400_Tasks/Task_by_Task/taskNNN/prototypes/**/*.{py,onnx}
    - The_400_Tasks/Task_by_Task/taskNNN/candidate_onnx/**/*.onnx
    - Research_Archive/organic_prototypes/taskNNN_*.py
    - selected .notongit scratch dirs

    Paths are REPO-relative when possible, otherwise absolute.
    """
    tag = f"task{tid:03d}"
    matches = []
    task_dir = REPO / "The_400_Tasks" / "Task_by_Task" / tag
    task_artifact_dirs = [
        (task_dir / "prototypes", {".py", ".onnx"}),
        (task_dir / "candidate_onnx", {".onnx"}),
    ]
    for artifact_dir, suffixes in task_artifact_dirs:
        if not artifact_dir.exists():
            continue
        for f in sorted(artifact_dir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in suffixes:
                continue
            try:
                rel = f.relative_to(REPO).as_posix()
            except ValueError:
                rel = f.as_posix()
            matches.append(rel)
    for search_dir in PROTOTYPE_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix != ".py":
                continue
            if not f.name.startswith(f"{tag}_"):
                continue
            # Skip __pycache__ etc (already filtered by is_file + .py)
            # Use REPO-relative if possible, otherwise absolute.
            try:
                rel = f.relative_to(REPO).as_posix()
            except ValueError:
                rel = f.as_posix()
            matches.append(rel)
    return ";".join(matches)


def load_solved_tids() -> set[int]:
    """Read SOLVED_TASKS from tools/inspect_tasks.py."""
    try:
        sys.path.insert(0, str(REPO))
        from tools.inspect_tasks import SOLVED_TASKS  # type: ignore
        return set(SOLVED_TASKS)
    except Exception as e:
        print(f"WARN: could not load SOLVED_TASKS ({e}); marking all as Unknown", file=sys.stderr)
        return set()


def score_current_zip(zip_path: Path) -> dict[int, dict]:
    """Score every task ONNX in the given submission zip and return
    {task_id: {'cost': int, 'macs': int, 'memory': int, 'params': int}}.

    Uses tools._score_v2. Slow (~1s per task ONNX). Only run when
    --refresh-costs is passed.
    """
    import os
    import tempfile
    os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
    out: dict[int, dict] = {}
    if not zip_path.exists():
        return out
    sys.path.insert(0, str(REPO / "tools"))
    import _score_v2 as scv2  # type: ignore
    try:
        import onnxruntime as ort  # type: ignore
        ort.set_default_logger_severity(4)
    except Exception:
        pass
    with ZipFile(zip_path) as z:
        for name in z.namelist():
            tid = task_id_from_filename(name)
            if tid is None:
                continue
            tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            tmp.write(z.read(name))
            tmp.close()
            try:
                r = scv2.score_network_v2(tmp.name)
                if r:
                    macs, memory, params = r
                    cost = (memory or 0) + (params or 0)
                    out[tid] = {
                        "cost": cost,
                        "macs": macs or 0,
                        "memory": memory or 0,
                        "params": params or 0,
                    }
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP,
                   help="Submission zip to extract ONNX features from (default: current best)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                   help="Official task JSON directory for grid_resized derivation")
    p.add_argument("--refresh-costs", action="store_true",
                   help="Also re-score every task ONNX in --zip and overwrite "
                        "current_cost / macs / memory / params columns. "
                        "Slow (~5-10 min for 401 tasks).")
    p.add_argument("--from-json", type=Path, default=None,
                   help="FAST path: apply a precomputed costs JSON "
                        "({tid: {cost,memory,params,macs}}) to the xlsx WITHOUT rescoring, then exit. "
                        "Folds the old _apply_live_costs.py.")
    args = p.parse_args(argv)

    if args.from_json:
        import json as _json
        fresh = {int(k): v for k, v in _json.loads(Path(args.from_json).read_text(encoding="utf-8")).items()}
        wb = openpyxl.load_workbook(args.xlsx)
        ws = wb["Tasks"]
        H = [c.value for c in ws[1]]
        ti = H.index("task_id") + 1; ci = H.index("current_cost") + 1
        mi = H.index("memory") + 1; pi = H.index("params") + 1
        mai = H.index("macs") + 1 if "macs" in H else None
        n = 0
        for r in range(2, ws.max_row + 1):
            tid = ws.cell(r, ti).value
            if tid is None:
                continue
            fc = fresh.get(int(tid))
            if not fc:
                continue
            ws.cell(r, ci).value = fc["cost"]; ws.cell(r, mi).value = fc["memory"]; ws.cell(r, pi).value = fc["params"]
            if mai is not None and "macs" in fc:
                ws.cell(r, mai).value = fc["macs"]
            n += 1
        wb.save(args.xlsx)
        print(f"Applied {n} precomputed costs from {Path(args.from_json).name} (no rescore).")
        return 0

    print(f"Loading xlsx: {args.xlsx}")
    print(f"Loading ONNX features from: {args.zip}")

    wb = openpyxl.load_workbook(args.xlsx)
    ws = wb["Tasks"]
    headers = [c.value for c in ws[1]]

    # Existing columns we'll consult
    fam_idx = headers.index("family") + 1
    trans_idx = headers.index("transformations") + 1
    tid_idx = headers.index("task_id") + 1
    src_idx = headers.index("current_src") + 1
    grid_idx = headers.index("grid_resized") + 1 if "grid_resized" in headers else None
    local_payload_idx = headers.index("local_payload_path") + 1 if "local_payload_path" in headers else None

    # Build the family+transformations cluster map
    cluster_map: dict[tuple, list[int]] = defaultdict(list)
    src_count: Counter = Counter()
    for r in range(2, ws.max_row + 1):
        tid = ws.cell(r, tid_idx).value
        if tid is None:
            continue
        tid = int(tid)
        fam = ws.cell(r, fam_idx).value
        trans = ws.cell(r, trans_idx).value
        src = ws.cell(r, src_idx).value
        if fam and trans:
            cluster_map[(fam, trans)].append(tid)
        if src:
            src_count[src] += 1

    # Load ONNX features
    onnx_features = collect_onnx_features(args.zip)
    print(f"Extracted ONNX features for {len(onnx_features)} tasks")

    grid_resized = derive_grid_resized(args.data)
    print(f"Derived grid_resized for {len(grid_resized)} tasks")

    # Load SOLVED_TASKS
    solved = load_solved_tids()
    print(f"Loaded {len(solved)} SOLVED tasks")

    # New columns to add (in this order, appended to the end)
    NEW_COLS = [
        "ops_count",
        "op_signature",
        "unique_ops",
        "graph_name",
        "producer",
        "similar_tasks",
        "similar_task_count",
        "current_src_shared_with",
        "solved_status",
        "hidden_bad_history",
        "live_arch",
        "cherry13b_tested",
        "structural_attempt",
        "unlock_potential",
        "structural_kaggle_delta",
        "prototype_path",
    ]

    # Find or append each column
    col_idx: dict[str, int] = {}
    for col in NEW_COLS:
        if col in headers:
            col_idx[col] = headers.index(col) + 1
        else:
            ws.cell(1, ws.max_column + 1, value=col)
            col_idx[col] = ws.max_column
            headers.append(col)

    # If --refresh-costs: also rescan zip for fresh cost/macs/memory/params
    fresh_costs: dict[int, dict] = {}
    if args.refresh_costs:
        print("Rescoring all task ONNXes in the submission (slow)...")
        fresh_costs = score_current_zip(args.zip)
        print(f"  Rescored {len(fresh_costs)} task ONNXes")
        cost_idx = headers.index("current_cost") + 1 if "current_cost" in headers else None
        macs_idx = headers.index("macs") + 1 if "macs" in headers else None
        mem_idx = headers.index("memory") + 1 if "memory" in headers else None
        par_idx = headers.index("params") + 1 if "params" in headers else None
    else:
        cost_idx = macs_idx = mem_idx = par_idx = None

    # Fill rows
    for r in range(2, ws.max_row + 1):
        tid = ws.cell(r, tid_idx).value
        if tid is None:
            continue
        tid = int(tid)
        fam = ws.cell(r, fam_idx).value
        trans = ws.cell(r, trans_idx).value
        src = ws.cell(r, src_idx).value

        feats = onnx_features.get(tid, {})
        counter = feats.get("op_counter", Counter())

        # ops_count
        ws.cell(r, col_idx["ops_count"]).value = feats.get("ops_count", 0)
        # op_signature
        ws.cell(r, col_idx["op_signature"]).value = format_op_signature(counter)
        # unique_ops
        ws.cell(r, col_idx["unique_ops"]).value = format_unique_ops(counter)
        # graph_name
        ws.cell(r, col_idx["graph_name"]).value = feats.get("graph_name", "")
        # producer
        ws.cell(r, col_idx["producer"]).value = feats.get("producer", "")

        # similar_tasks
        if fam and trans:
            siblings = sorted(cluster_map.get((fam, trans), []))
            siblings = [t for t in siblings if t != tid]
            ws.cell(r, col_idx["similar_tasks"]).value = ",".join(f"task{t:03d}" for t in siblings) if siblings else ""
            ws.cell(r, col_idx["similar_task_count"]).value = len(siblings)
        else:
            ws.cell(r, col_idx["similar_tasks"]).value = ""
            ws.cell(r, col_idx["similar_task_count"]).value = 0

        # current_src_shared_with
        if src:
            ws.cell(r, col_idx["current_src_shared_with"]).value = max(0, src_count[src] - 1)
        else:
            ws.cell(r, col_idx["current_src_shared_with"]).value = 0

        # grid_resized: objective fact from task JSON; overwrite stale blanks.
        if grid_idx is not None and tid in grid_resized:
            ws.cell(r, grid_idx).value = bool(grid_resized[tid])

        # local_payload_path fallback: every current task is addressable inside
        # the submission zip. Preserve richer source paths when already present.
        if local_payload_idx is not None:
            cell = ws.cell(r, local_payload_idx)
            if cell.value in (None, "") and tid in onnx_features:
                try:
                    rel_zip = args.zip.resolve().relative_to(REPO.resolve()).as_posix()
                except ValueError:
                    rel_zip = args.zip.as_posix()
                cell.value = f"{rel_zip}::task{tid:03d}.onnx"

        # solved_status
        if tid in HIDDEN_BAD_TIDS:
            ws.cell(r, col_idx["solved_status"]).value = "HiddenBad"
        elif tid in solved:
            ws.cell(r, col_idx["solved_status"]).value = "Solved"
        else:
            ws.cell(r, col_idx["solved_status"]).value = "NotSolved"

        # hidden_bad_history
        ws.cell(r, col_idx["hidden_bad_history"]).value = "Yes" if tid in HIDDEN_BAD_TIDS else "No"

        # live_arch
        live_arch = feats.get("live_arch", "structural")
        ws.cell(r, col_idx["live_arch"]).value = live_arch

        # Manual fields: initialize blanks, never overwrite filled values.
        cherry_cell = ws.cell(r, col_idx["cherry13b_tested"])
        if cherry_cell.value in (None, ""):
            cherry_cell.value = "not_yet"
        structural_cell = ws.cell(r, col_idx["structural_attempt"])
        if structural_cell.value in (None, ""):
            structural_cell.value = "not_yet"
        delta_cell = ws.cell(r, col_idx["structural_kaggle_delta"])
        if delta_cell.value is None:
            delta_cell.value = ""

        # unlock_potential
        cherry_status = str(cherry_cell.value or "not_yet").strip().lower()
        unlock = "HIGH" if live_arch == "lookup" and cherry_status in {"not_yet", "regresses", "regressed"} else "LOW"
        ws.cell(r, col_idx["unlock_potential"]).value = unlock

        # prototype_path
        ws.cell(r, col_idx["prototype_path"]).value = find_prototype(tid)

        # --refresh-costs: overwrite current_cost / macs / memory / params
        if fresh_costs and tid in fresh_costs:
            fc = fresh_costs[tid]
            if cost_idx is not None:
                ws.cell(r, cost_idx).value = fc["cost"]
            if macs_idx is not None:
                ws.cell(r, macs_idx).value = fc["macs"]
            if mem_idx is not None:
                ws.cell(r, mem_idx).value = fc["memory"]
            if par_idx is not None:
                ws.cell(r, par_idx).value = fc["params"]

    wb.save(args.xlsx)
    print(f"Wrote enriched: {args.xlsx}")
    print("\nNote: run `python tools/merge_task_status.py` to regenerate the CSV mirror.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
