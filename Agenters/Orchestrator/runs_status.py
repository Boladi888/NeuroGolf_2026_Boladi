#!/usr/bin/env python3
"""runs_status.py - liveness + collision signal over ALL Orchestrator run folders.

Companion to run_audit.py (which audits ONE run for closure). The Supervisor runs this during Part-A setup so the
active-run collision check is one command instead of a ~30-folder grep + per-folder mtime forensics (the single
biggest Part-A token sink, per _proposals/supervisor_setup_token_reduction.md P1).

Liveness is inferred (there is no per-run daemon), from two cheap signals:
  CLOSED  = RUN_CLOSED.md present (the Supervisor Done-Condition marker).
  ACTIVE  = no RUN_CLOSED.md AND newest file mtime within --fresh-secs (default 2h) -> something is still writing it.
  STALE   = no RUN_CLOSED.md AND newest mtime older than the window -> probably abandoned; audit it and stamp CLOSED.

Terminal-status vocabulary is imported from run_audit.py so this tool and the closure gate never disagree on what
"terminal" means. Stdlib-only; paths derive from __file__, so it runs from any CWD.

Usage:
  python Agenters/Orchestrator/runs_status.py active                      # what is running right now
  python Agenters/Orchestrator/runs_status.py list                        # full table (default)
  python Agenters/Orchestrator/runs_status.py collisions --tasks task014,task017,task199
  python Agenters/Orchestrator/runs_status.py [list] --fresh-secs 3600    # tighten the "active" window
Exit 0 always for `active`/`list`; `collisions` exits 1 if any requested task is claimed by an ACTIVE run (hard stop).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_audit import klass, norm, read_csv  # noqa: E402  single-source the terminal vocabulary

RUNS_DIR = Path(__file__).resolve().parent / "runs"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRESH_SECS = 2 * 60 * 60  # 2h: a run untouched for >2h with no RUN_CLOSED.md is treated as abandoned


def iter_runs():
    if not RUNS_DIR.is_dir():
        return
    for p in sorted(RUNS_DIR.iterdir()):
        if p.is_dir() and not p.name.startswith((".", "_")):  # skip _template and dotfiles
            yield p


def newest_mtime(run):
    newest = 0.0
    for f in run.rglob("*"):
        try:
            if f.is_file():
                newest = max(newest, f.stat().st_mtime)
        except OSError:
            continue
    return newest


def age_str(secs):
    secs = int(secs)
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= n:
            return f"{secs // n}{unit}"
    return f"{secs}s"


def state_of(run, fresh_secs, now):
    if (run / "RUN_CLOSED.md").exists():
        return "CLOSED"
    nm = newest_mtime(run)
    if nm and (now - nm) <= fresh_secs:
        return "ACTIVE"
    return "STALE"


def run_task_klass(run):
    """task_id -> (claims_klass, report_klass); a missing file contributes nothing for that side."""
    out = {}
    for fname, idx in (("task_claims.csv", 0), ("final_report.csv", 1)):
        rows = read_csv(run / fname)
        if not rows:
            continue
        for r in rows:
            t = norm(r.get("task"))
            if not t:
                continue
            pair = out.setdefault(t, [None, None])
            pair[idx] = klass(r.get("status"))
    return out


def nonterminal_count(tk):
    return sum(1 for kl in tk.values() if None in kl or "OPEN" in kl)


def cmd_list(args, now):
    rows = []
    for run in iter_runs():
        st = state_of(run, args.fresh_secs, now)
        nm = newest_mtime(run)
        tk = run_task_klass(run)
        rows.append((run.name, st, age_str(now - nm) if nm else "-", len(tk), nonterminal_count(tk)))
    print(f"== runs_status (fresh window = {age_str(args.fresh_secs)}) ==")
    print(f"  {'run_id':<32} {'state':<7} {'age':>6} {'tasks':>6} {'nonterm':>8}")
    for name, st, age, ntasks, nonterm in rows:
        flag = "  <- still open, reconcile+close" if (st == "STALE" and nonterm) else (
               "  <- STALE+clean: safe to stamp RUN_CLOSED.md" if st == "STALE" else "")
        print(f"  {name:<32} {st:<7} {age:>6} {ntasks:>6} {nonterm:>8}{flag}")
    active = [r for r in rows if r[1] == "ACTIVE"]
    print(f"\n{len(active)} ACTIVE, {sum(1 for r in rows if r[1]=='STALE')} STALE, "
          f"{sum(1 for r in rows if r[1]=='CLOSED')} CLOSED")
    return 0


def cmd_active(args, now):
    found = False
    for run in iter_runs():
        if state_of(run, args.fresh_secs, now) == "ACTIVE":
            found = True
            print(f"ACTIVE  {run.name}  (touched {age_str(now - newest_mtime(run))} ago)")
    if not found:
        print("no ACTIVE runs (nothing touched within the fresh window)")
    return 0


def cmd_collisions(args, now):
    want = {norm(t) for t in args.tasks.split(",") if t.strip()}
    if not want:
        print("collisions: pass --tasks task014,task017,...")
        return 2
    hits = {t: [] for t in want}
    for run in iter_runs():
        st = state_of(run, args.fresh_secs, now)
        for t, kl in run_task_klass(run).items():
            if t in want:
                hits[t].append((run.name, st, kl))
    hard = 0
    print(f"== collision scan for {len(want)} task(s) ==")
    for t in sorted(want):
        rows = hits[t]
        if not rows:
            print(f"  {t}: clear (no sibling run claims it)")
            continue
        for name, st, kl in rows:
            tag = "TERMINAL" if (kl[0] not in (None, "OPEN") and kl[1] not in (None, "OPEN")) else "OPEN/PARTIAL"
            stop = ""
            if st == "ACTIVE":
                hard += 1
                stop = "  *** HARD STOP: claimed by an ACTIVE run ***"
            elif tag == "OPEN/PARTIAL" and st == "STALE":
                stop = "  (stale-open elsewhere: verify it is abandoned before claiming)"
            else:
                stop = "  (prior art - re-attempt only with a genuinely novel formulation; do not duplicate)"
            print(f"  {t}: {name} [{st}] claims={kl[0]} report={kl[1]} {tag}{stop}")
    if hard:
        print(f"\n{hard} task(s) collide with an ACTIVE run -> NOT safe to launch as-is")
        return 1
    print("\nno ACTIVE-run collisions (any hits above are terminal prior-art or stale-open)")
    return 0


def cmd_pulse(args, now):
    """Compact per-run check-in for the Supervisor's periodic monitoring (proposal 2026-06-19_supervisor_run_pulse_helper):
    state + markers + log growth + open claims + active reviewer leases + unresolved positive deltas, and — only with
    --audit-if-terminal AND when the run is terminal-enough — run_audit. Read-only: never writes RUN_CLOSED.md."""
    rid = (args.target or "").strip()
    if not rid:
        print("pulse: pass a run id, e.g. runs_status.py pulse beta_p8c_claude_cli --audit-if-terminal"); return 2
    run = Path(rid) if Path(rid).is_dir() else (RUNS_DIR / rid)
    if not run.is_dir():
        print(f"pulse: no such run dir: {rid}"); return 2
    st = state_of(run, args.fresh_secs, now); nm = newest_mtime(run)
    print(f"== pulse {run.name} ==  state={st}  touched={age_str(now - nm) if nm else '-'} ago")
    ec = run / "exit_code.txt"
    print(f"  RUN_CLOSED.md={'present' if (run / 'RUN_CLOSED.md').exists() else 'ABSENT'}"
          f"   exit_code={ec.read_text(encoding='utf-8', errors='replace').strip() if ec.exists() else '-'}")
    for nmf in ("orchestrator_events.jsonl", "orchestrator_out.json", "out.json", "orchestrator_err.txt"):
        f = run / nmf
        if f.exists():
            print(f"  {nmf}: {f.stat().st_size} B, touched {age_str(now - f.stat().st_mtime)} ago")
    claims = read_csv(run / "task_claims.csv") or []
    counts = {}; openrows = []
    for r in claims:
        k = klass(r.get("status")); counts[k] = counts.get(k, 0) + 1
        if k == "OPEN": openrows.append(r)
    print(f"  task_claims: {len(claims)} rows  " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for r in openrows:
        print(f"    OPEN {norm(r.get('task'))}: status={r.get('status')!r} {(r.get('notes') or '')[:50]}")
    leases = read_csv(run / "reviewer_leases.csv") or []
    act = [r for r in leases if (r.get("status") or "").strip().lower() not in ("", "ship", "drop", "released")]
    if act:
        print("  reviewer_leases active: " + ", ".join(
            f"{norm(r.get('task'))}->{r.get('reviewer_slot')}({r.get('status')})" for r in act))
    rep = read_csv(run / "final_report.csv") or []
    unresolved = []
    for r in rep:
        try: d = float(r.get("expected_lb_delta") or 0)
        except ValueError: d = 0.0
        cs = (r.get("captain_status") or "").strip().lower()
        if d > 0 and cs in ("", "held", "queued", "safe_batch_queue"):
            unresolved.append((norm(r.get("task")), d, cs or "blank"))
    print(f"  final_report: {len(rep)} rows; unresolved positive deltas: {len(unresolved)}")
    for t, d, cs in unresolved:
        print(f"    !! {t} +{d} captain_status={cs} (must resolve before close)")
    terminal_enough = st in ("CLOSED", "STALE") or (bool(claims) and not openrows)
    print(f"  closure: {'terminal-enough' if terminal_enough else 'ACTIVE with open tasks'}")
    if args.audit_if_terminal:
        if terminal_enough:
            res = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "run_audit.py"), str(run)],
                                 cwd=str(REPO_ROOT), capture_output=True, text=True)
            for ln in (res.stdout or "").splitlines():
                print(f"  | {ln}")
            print(f"  run_audit exit={res.returncode} -> {'SAFE to close' if res.returncode == 0 else 'NOT safe (fix the FAILs)'}")
        else:
            print("  run_audit skipped (not terminal-enough; do not claim closure)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Liveness + collision signal across Orchestrator runs.")
    ap.add_argument("command", nargs="?", default="list", choices=["list", "active", "collisions", "pulse"])
    ap.add_argument("target", nargs="?", default="", help="run id (for `pulse`)")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (for `collisions`)")
    ap.add_argument("--audit-if-terminal", action="store_true", help="(pulse) run run_audit when the run is terminal-enough")
    ap.add_argument("--fresh-secs", type=int, default=DEFAULT_FRESH_SECS,
                    help="a run with no RUN_CLOSED.md but newest mtime within this many seconds counts as ACTIVE")
    args = ap.parse_args()
    now = time.time()
    return {"list": cmd_list, "active": cmd_active, "collisions": cmd_collisions, "pulse": cmd_pulse}[args.command](args, now)


if __name__ == "__main__":
    sys.exit(main())
