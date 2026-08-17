"""harvest_alpha_b.py — sustained-iteration probe pipeline for Alpha-B.

Strategy (single iteration):
1. Read `Logs/PROBES_LEDGER_alpha_b.md` to know which (task, source_pool) pairs
   have already been tested and their verdict.
2. Read the current best baseline zip (configurable via --base; defaults to
   `pomegranate12` = 5649.68 = the 2026-05-10 23:48 UTC mega-stack).
3. Pick the next-most-promising untested (task, source_pool) pair:
   - Eligible tasks: Alpha-B-range IDs (per `The_400_Tasks/alpha_sub_lanes.md`)
     that are NOT already in the current baseline and have not been individually
     tested as a "loss" or "no-effect" — but DO test them as part of a different
     source pool if available.
   - Source pools to scan, in priority order:
     a) cherry4_legal_sources_band_<offset>_<size> (high hit rate so far —
        4 of 25 single probes were +12 to +14)
     b) Other competitor-extracted bands in `data/competitor_artifacts/`:
        afr1ste-* / beicicc-* / konbu17-* / jonathan-* / ashok205-*
     c) Older candidate packages (orange5/papaya5/quince5/etc.)
4. For the chosen (task, pool) pair:
   - Build a probe zip = current_best ⊕ swap(task ← pool's taskNNN.onnx)
   - Submit to Kaggle (single submission; respects 2/hour external pacing)
   - Wait for COMPLETE result via `kaggle competitions submissions`
   - Append the result to PROBES_LEDGER and rebuild current_best if positive

Usage:
    python tools/harvest_alpha_b.py                   # one iteration
    python tools/harvest_alpha_b.py --dry-run         # pick + build but don't submit
    python tools/harvest_alpha_b.py --base <zip>      # override baseline zip

Pacing:
    Owner-imposed cadence is 2 Kaggle submissions per hour after 2026-05-11
    00:00 UTC. This script does ONE submission per invocation. Schedule via
    cron / Task Scheduler / harness loop.

Stop condition:
    Owner: stop autonomous harvest when public LB >= 6500 OR when the owner
    instructs otherwise.
"""

# NB: Skeleton only — full implementation needs the score_network_v2 OOM fix
# (see Logs/lessons.md "exhaustive scan crashes silently"). For now the harvest
# uses raw byte size as a proxy for cost when ranking candidates.

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

NG = pathlib.Path(os.environ.get("NEUROGOLF_MAIN_ROOT",
                                 r"<REPO_ROOT>"))
NOTONGIT = NG.parent / ".notongit" / "NeuroGolf_nogit"
# Tracked-file inputs are read from THIS worktree (script's parent.parent),
# so per-worktree edits to alpha_sub_lanes.md / PROBES_LEDGER take effect
# without requiring a commit + pull cycle. Untracked stores (band candidate
# zips in NOTONGIT) and the persistent baseline zip live in NG/submissions/
# which is junctioned and shared across worktrees.
HERE = pathlib.Path(__file__).resolve().parent.parent
ALPHA_B_LANES = HERE / "The_400_Tasks" / "alpha_sub_lanes.md"
LEDGER = HERE / "Logs" / "PROBES_LEDGER_alpha_b.md"
CANDS_CSV = HERE / "The_400_Tasks" / "submission_candidates.csv"
CHERRY_BANDS_PREFIX = "cherry4_legal_sources_band_"


def alpha_b_ids() -> set[int]:
    """Read Alpha-B task ID list from alpha_sub_lanes.md."""
    ids = set()
    in_b = False
    with ALPHA_B_LANES.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("### Alpha-B (200 tasks)"):
                in_b = True
                continue
            if in_b:
                if "```" in line and ids:
                    break
                for tok in line.replace(",", " ").split():
                    if tok.isdigit():
                        ids.add(int(tok))
    return ids


def parse_ledger() -> dict[tuple[int, str], str]:
    """Parse PROBES_LEDGER_alpha_b.md → {(task, source_short): verdict}.

    Reads the "Isolated-probe verdicts by (task, source)" table.
    """
    if not LEDGER.exists():
        return {}
    seen = {}
    with LEDGER.open(encoding="utf-8") as f:
        in_pair_table = False
        for line in f:
            if "Isolated-probe verdicts" in line:
                in_pair_table = True
                continue
            if in_pair_table and line.startswith("|") and "---" not in line and "task" not in line.lower():
                parts = [p.strip() for p in line.strip().strip("|").split("|")]
                if len(parts) >= 6:
                    try:
                        tid = int(parts[0])
                        src = parts[1]
                        verdict = parts[5]
                        seen[(tid, src)] = verdict
                    except ValueError:
                        pass
    return seen


def list_band_zips() -> list[tuple[str, Path]]:
    """List `cherry4_legal_sources_band_*` zips on disk (and a few others)."""
    bands = []
    with CANDS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            role = (row.get("role") or "").strip()
            if role != "reference_or_historical_candidate":
                continue
            est = (row.get("estimated_result") or "").lower()
            if any(s in est for s in ("kaggle", "complete", "error")):
                continue
            p = (row.get("local_payload_path") or "").strip()
            if not p:
                continue
            cand = NOTONGIT / p.replace("\\", "/")
            if cand.exists():
                bands.append((row["candidate"], cand))
    return bands


def pick_next_probe(base_zip: Path, tested: dict[tuple[int, str], str], lanes: set[int]) -> tuple[int, str, Path, bytes] | None:
    """Pick the next-most-promising untested (task, source_pool) probe.

    Strategy: empirically the SAME task tends to score similarly across source
    pools (e.g., task314 was -3.48 from cherry4_band and -3.47 from ugni3).
    So skip tasks where ANY source produced a "loss" verdict — they're
    structurally costly to swap, not source-specific. Only re-test a task with
    a different source pool if prior verdict was "no-effect" or absent.

    Priority:
    1. Alpha-B-range tasks not yet tested AT ALL (highest priority)
    2. Tasks tested with verdict "no-effect" but with NEW source pool not yet tried
    3. Largest base→cand byte savings (proxy for cost reduction)

    Returns (task_id, candidate_name, candidate_zip_path, candidate_onnx_bytes)
    or None if no eligible probe remains.
    """
    bands = list_band_zips()
    with ZipFile(base_zip) as zb:
        base_sizes = {tid: zb.getinfo(f"task{tid:03d}.onnx").file_size
                      for tid in lanes
                      if f"task{tid:03d}.onnx" in zb.namelist()}
        base_bytes = {tid: zb.read(f"task{tid:03d}.onnx") for tid in lanes
                      if f"task{tid:03d}.onnx" in zb.namelist()}

    # Build per-task "best prior verdict" map
    task_verdicts = {}  # tid -> "loss" / "no-effect" / "WIN" / None
    for (tid, src), verdict in tested.items():
        v_lower = verdict.lower()
        prior = task_verdicts.get(tid)
        if "win" in v_lower or "+" in verdict:
            task_verdicts[tid] = "WIN"  # already in baseline; skip
        elif prior != "WIN" and "loss" in v_lower:
            task_verdicts[tid] = "loss"
        elif prior is None:
            task_verdicts[tid] = "no-effect"

    eligible_tids = set()
    for tid in lanes:
        v = task_verdicts.get(tid)
        if v == "WIN" or v == "loss":
            continue  # WIN already in base; loss = waste another submit
        eligible_tids.add(tid)

    candidates = []
    seen_pairs = set()  # avoid duplicate same-bytes entries
    for cand_name, zpath in bands:
        src_short = "cherry4_band" if cand_name.startswith(CHERRY_BANDS_PREFIX) else cand_name.split("_")[0]
        try:
            with ZipFile(zpath) as zb:
                for tid in eligible_tids:
                    if tid not in base_sizes:
                        continue
                    if (tid, src_short) in tested:
                        continue
                    fname = f"task{tid:03d}.onnx"
                    if fname not in zb.namelist():
                        continue
                    sz = zb.getinfo(fname).file_size
                    if sz >= base_sizes[tid]:
                        continue
                    delta_bytes = base_sizes[tid] - sz
                    candidates.append((delta_bytes, tid, cand_name, zpath, fname, src_short))
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(reverse=True)  # biggest byte savings first

    # Walk candidates, skipping any byte-identical-to-base
    for delta_bytes, tid, cand_name, zpath, fname, src_short in candidates:
        with ZipFile(zpath) as zb:
            cand_data = zb.read(fname)
        if cand_data == base_bytes[tid]:
            tested[(tid, src_short)] = "identical"
            continue
        return tid, cand_name, zpath, cand_data

    return None


def build_probe_zip(base_zip: Path, tid: int, cand_data: bytes, label: str) -> Path:
    out_dir = NG / "submissions" / "_alpha_b_probes_b3" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "submission.zip"
    fname = f"task{tid:03d}.onnx"
    with ZipFile(base_zip, "r") as zin, ZipFile(out_zip, "w", ZIP_DEFLATED) as zout:
        for n in zin.namelist():
            if n == fname:
                zout.writestr(n, cand_data)
            else:
                zout.writestr(n, zin.read(n))
    return out_zip


def next_fruit() -> str:
    """Generate next fruit name in the alphabet, suffixed by the current major version (12)."""
    used = set()
    if LEDGER.exists():
        with LEDGER.open(encoding="utf-8") as f:
            for line in f:
                m = re.findall(r"\b([a-z]+\d+)\b", line)
                for x in m:
                    used.add(x)
    fruits = ["apple","banana","cherry","date","elderberry","fig","grape","honeydew","imbe","jackfruit","kiwi","lemon","lime","mango","nectarine","orange","papaya","quince","raspberry","strawberry","tangerine","ugli","vanilla","watermelon","xigua","yuzu","zucchini"]
    for v in (13, 14, 15, 16, 17, 18, 19, 20):
        for fr in fruits:
            name = f"{fr}{v}"
            if name not in used:
                return name
    return "x999"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(NG / "submissions" / "alpha_b_t089v2_5650_99" / "submission.zip"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cherry4-only", action="store_true",
                   help="Restrict source pool to cherry4_legal_sources_band_* (the only pool with consistent +12 wins on Alpha-B; per Logs/PROBES_LEDGER_alpha_b.md non-cherry4 pools have 0 wins / 11 probes).")
    args = p.parse_args(argv)

    base = Path(args.base)
    if not base.exists():
        print(f"FATAL: base zip does not exist: {base}", file=sys.stderr)
        return 1

    lanes = alpha_b_ids()
    tested = parse_ledger()
    print(f"baseline: {base.name} ({base.stat().st_size:,} bytes)")
    print(f"alpha-b lanes: {len(lanes)}, already-tested pairs: {len(tested)}")

    if args.cherry4_only:
        # Pre-filter the bands list to cherry4 only.
        global list_band_zips
        _orig = list_band_zips
        def list_band_zips():
            return [(n, p) for n, p in _orig() if n.startswith(CHERRY_BANDS_PREFIX)]
    pick = pick_next_probe(base, tested, lanes)
    if pick is None:
        print("No eligible probe remains for this base. Pivot needed (try afr1ste/beicicc pools or fresh organic builds).")
        return 0
    tid, cand_name, zpath, cand_data = pick
    fruit = next_fruit()
    label = f"alpha_b_{fruit}_baseline_task{tid:03d}_{cand_name[:25]}"
    out_zip = build_probe_zip(base, tid, cand_data, label)
    sha = hashlib.sha256(out_zip.read_bytes()).hexdigest().upper()[:16]
    sz = out_zip.stat().st_size
    print(f"built {fruit}: task{tid:03d} swap from {cand_name} ({sz:,} bytes, SHA {sha})")

    if args.dry_run:
        print("--dry-run; not submitting")
        return 0

    LIVE = NG / "submissions" / "submission.zip"
    BAK = NG / "submissions" / "_alpha_b_scratch" / f"_pre_harvest_backup_{fruit}.zip"
    BAK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(LIVE, BAK)
    shutil.copy(out_zip, LIVE)

    msg = f"alpha-b {fruit} baseline({base.parent.name}) + task{tid:03d} swap from {cand_name[:50]} (raw bytes saved {len(cand_data):,} vs base)"
    proc = subprocess.run(["kaggle","competitions","submit","-c","neurogolf-2026","-f",str(LIVE),"-m",msg], capture_output=True, text=True, timeout=120)
    last = (proc.stdout + proc.stderr).strip().split("\n")[-1]
    print(f"submit -> {last}")

    shutil.copy(BAK, LIVE)  # restore prior state
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
