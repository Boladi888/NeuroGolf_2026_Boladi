#!/usr/bin/env python3
"""run_audit.py - pre-closure consistency check for one Orchestrator run folder.

The Supervisor runs this before closing a wave (Supervisor README s.Run-finalization check). It catches the
task_claims.csv <-> final_report.csv drift that bit codex_p1_native (task108 'staged' in one, 'reviewing' in the
other) and folder-6, plus positive-delta / staged-row / staging-provenance gaps. Stdlib-only.

Status vocab varies by run, so tasks are classified by KEYWORD (not an exact-match table):
  'drop' -> DROP ; ship|packet|staged|captain|external -> SHIP ; anything else (building/reviewing/blank) -> OPEN.
A terminal task must be DROP or SHIP in BOTH files, and the two classes must agree.

Usage:
  python Agenters/Orchestrator/run_audit.py <run_dir> [--allowlist task011,task022,...]
                                            [--staged-dir <path>] [--repo-root <path>]
Exit 0 = clean (warnings allowed); exit 1 = one or more hard failures -> NOT safe to close.
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

# Paths derive from __file__ so the audit verdict is reproducible from ANY cwd (was the C12 bug: a CWD-relative
# --repo-root made staged_path checks pass from the repo root but FAIL from Agenters/Orchestrator/). run_audit.py
# lives at <repo>/Agenters/Orchestrator/run_audit.py; the team repo is a sibling of <repo>.
REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_STAGED = REPO_ROOT.parent / "K-NeuroGolf-OnnxlyFans" / "Team_Submission" / "Staged_Boladi"

# Terminal-status vocabulary (case-insensitive substring), per the Orchestrator README final_report schema:
# DROP = terminal, did not ship; SHIP = terminal, shipped/owned downstream; else OPEN (not terminal).
# (Bare "captain" was removed — it mis-classified "captain review"/"captain holding" as terminal; the real
#  terminal value is "captain_packet", caught by "packet".)
# "wall"/"floor" = a floor-confirmed no-ship (nothing shippable below team-best); accepted here as a backstop so a
#  legacy ledger that emitted them (e.g. beta_p5_claude_cli) audits clean. Schema-preferred term is "drop"/"no_candidate"
#  — Orchestrators should emit those going forward (see runs/README.md §status vocabulary).
DROP_TERMS = ("drop", "no_candidate", "no-candidate", "rejected", "not_sent", "blocked", "wall", "floor")
SHIP_TERMS = ("ship", "packet", "staged", "external", "shipped", "probe_ok", "probing")


def klass(status):
    s = (status or "").strip().lower()
    if not s:
        return "OPEN"
    if any(k in s for k in DROP_TERMS):
        return "DROP"
    if any(k in s for k in SHIP_TERMS):
        return "SHIP"
    return "OPEN"  # building / reviewing / rework / held / queued / blank -> not terminal


def norm(t):
    t = (t or "").strip()
    if t.lower().startswith("task"):
        d = t[4:].strip()
        return "task" + d.zfill(3) if d.isdigit() else t
    return t


def read_csv(path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_of(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def rel_to_root(root, path):
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def staged_path_hint(root, staged_path, expected_sha, task):
    """Return a correction hint when the same candidate exists in the task-local staging home."""
    clean = staged_path.replace("\\", "/").strip()
    if not clean:
        return ""
    task = norm(task)
    expected_sha = str(expected_sha or "").strip().lower()
    filename = Path(clean).name or f"{task}.onnx"
    parent = Path(clean).parent.name
    task_stage = root / "The_400_Tasks" / "Task_by_Task" / task / "staged_candidates"
    candidates = []
    old_prefixes = ("staged_submits/", "The_400_Tasks/staged_submits/")
    for old in old_prefixes:
        if clean.startswith(old):
            candidates.append(task_stage / clean[len(old):])
    if parent and parent != ".":
        candidates.append(task_stage / parent / filename)
    if task_stage.exists():
        candidates.extend(task_stage.glob(f"{task}_*/{filename}"))

    deduped = []
    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    existing = [candidate for candidate in deduped if candidate.exists()]
    if expected_sha:
        matches = [candidate for candidate in existing if sha256_of(candidate) == expected_sha]
        if len(matches) == 1:
            return f"; SHA-matching candidate exists at {rel_to_root(root, matches[0])}"
        if len(matches) > 1:
            shown = ", ".join(rel_to_root(root, candidate) for candidate in matches[:3])
            more = "" if len(matches) <= 3 else f" (+{len(matches) - 3} more)"
            return f"; multiple SHA-matching candidates found: {shown}{more}"
    elif len(existing) == 1:
        return f"; candidate exists at {rel_to_root(root, existing[0])} (sha not checked; row missing sha256)"
    return ""


def main():
    ap = argparse.ArgumentParser(description="Pre-closure consistency audit for one run folder.")
    ap.add_argument("run_dir")
    ap.add_argument("--allowlist", default="", help="comma-separated task ids; default = union of the two CSVs")
    ap.add_argument("--staged-dir", default=str(TEAM_STAGED))
    ap.add_argument("--repo-root", default=str(REPO_ROOT),
                    help="root that staged_path values resolve against (default: repo root via __file__, cwd-independent)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    root = Path(args.repo_root)
    staged_dir = Path(args.staged_dir)
    fails, warns = [], []

    claims = read_csv(run / "task_claims.csv")
    report = read_csv(run / "final_report.csv")
    if claims is None:
        fails.append(f"missing {run / 'task_claims.csv'}")
    if report is None:
        fails.append(f"missing {run / 'final_report.csv'}")
    if claims is None or report is None:
        return finish(run, fails, warns)

    cl = {norm(r["task"]): r for r in claims if r.get("task")}
    rp = {norm(r["task"]): r for r in report if r.get("task")}

    # 1. allowlist / parity: every owned task lands in BOTH files
    allow = {norm(t) for t in args.allowlist.split(",") if t.strip()} if args.allowlist else (set(cl) | set(rp))
    for t in sorted(allow):
        if t not in cl:
            fails.append(f"{t}: missing from task_claims.csv")
        if t not in rp:
            fails.append(f"{t}: missing from final_report.csv")
    for t in sorted(set(cl) - allow):
        warns.append(f"{t}: in task_claims.csv but not the allowlist")
    for t in sorted(set(rp) - allow):
        warns.append(f"{t}: in final_report.csv but not the allowlist")

    # 2/3/4. per-task: status consistency, positive-delta ownership, staged-row integrity
    for t in sorted(set(cl) & set(rp)):
        cc, rc = klass(cl[t].get("status")), klass(rp[t].get("status"))
        if cc == "OPEN" or rc == "OPEN":
            fails.append(f"{t}: NOT TERMINAL (claims={cl[t].get('status')!r}, report={rp[t].get('status')!r})")
        elif cc != rc:
            fails.append(f"{t}: STATUS DRIFT claims={cl[t].get('status')!r}({cc}) vs report={rp[t].get('status')!r}({rc})")

        try:
            delta = float(rp[t].get("expected_lb_delta") or 0)
        except ValueError:
            delta = 0.0
        cs = (rp[t].get("captain_status") or "").strip().lower()
        if delta > 0 and cs in ("", "held", "queued", "safe_batch_queue"):
            hint = "; use captain_status=external_owner for cross-lane hand-offs" if cs == "held" else ""
            fails.append(f"{t}: positive delta +{delta} with captain_status={cs or 'blank'} "
                         f"(must be owned, not held/queued/blank{hint})")

        if rc == "SHIP":
            sp = (rp[t].get("staged_path") or "").strip()
            # external_owner = a cross-lane handoff (e.g. a Codex run whose candidate ALPHA stages): the source path
            # is a path this lane is forbidden to create, so its absence is a WARN, not a FAIL (the C12 dual-lane gap).
            ext = (cs == "external_owner")
            sink = warns if ext else fails
            tag = "  (external_owner — owning lane stages)" if ext else ""
            if not sp:
                sink.append(f"{t}: SHIP row has empty staged_path{tag}")
            elif not (root / sp).exists():
                hint = staged_path_hint(root, sp, rp[t].get("sha256"), t)
                sink.append(f"{t}: staged_path source missing on disk: {sp}{tag}{hint}")
            if not (rp[t].get("sha256") or "").strip():
                warns.append(f"{t}: SHIP row missing sha256")
            if cs == "staged":  # captain staged -> the pair must exist in Staged_Boladi OR _shipped (already shipped)
                for ext in ("onnx", "json"):
                    if not (staged_dir / f"{t}.{ext}").exists() and not (staged_dir / "_shipped" / f"{t}.{ext}").exists():
                        fails.append(f"{t}: captain_status=staged but {t}.{ext} not in Staged_Boladi or _shipped")

    # 5. issues.md launch-attempt provenance (soft)
    issues = run / "issues.md"
    if issues.exists():
        txt = issues.read_text(encoding="utf-8", errors="ignore").lower()
        if any(k in txt for k in ("claude -p", "external", "launch")) and \
           not any(k in txt for k in ("inactive", "blocked", "recovered", "superseded")):
            warns.append("issues.md mentions a launch attempt without a terminal note "
                         "(inactive/blocked/recovered/superseded)")

    return finish(run, fails, warns)


def finish(run, fails, warns):
    print(f"== run_audit {run} ==")
    for w in warns:
        print(f"WARN  {w}")
    for f in fails:
        print(f"FAIL  {f}")
    if fails:
        print(f"\n{len(fails)} FAILURE(S), {len(warns)} warning(s) -> NOT safe to close")
        return 1
    print(f"clean ({len(warns)} warning(s)) -> safe to close")
    return 0


if __name__ == "__main__":
    sys.exit(main())
