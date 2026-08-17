"""longrun.py — resumable, checkpointed, watchable sweep of the ngs solver over a task bucket.

Built for the "get it ready for a long run I can steer" ask. Three properties:

  RESUMABLE   — every finished task is appended to results.jsonl; a restart skips done tasks and
                continues. Kill it (or close the window) any time; re-launch the same command to resume.
  WATCHABLE   — writes status.json every task (state / heartbeat / progress / ETA) and streams a live
                console dashboard to stdout, so a VizTerms visible-console window (or the WebUIz browser
                dashboard, or any file-tailer) shows progress. On exit it drops a done.json sentinel.
  PLUGGABLE   — per task it runs a *cascade of solvers* cheapest-first (DSL beam search now; the
                zero-param Einsum search and the GPU batched-discrete search slot in as `SOLVERS` grow),
                each under a time budget, and keeps the lowest-cost argen-confirmed result.

Follows the repo's background-job rules: monitors watch the ARTIFACT (status.json heartbeat + state),
never a PID; every terminal state (complete / interrupted / crashed) is written, not just the happy path.

  .venv/Scripts/python.exe -m ml.ngs.longrun --tasks hard --name run1 --budget 20
  .venv/Scripts/python.exe -m ml.ngs.longrun --tasks 1-400 --name full --resume
  .venv/Scripts/python.exe -m ml.ngs.longrun --status --name run1        # print status.json and exit
"""
import json
import os
import pathlib
import signal
import sys
import time
import traceback

from . import spec, tasks, score, argen
from .search import beam_search

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "_runs"
TEAM_COSTS = HERE / "_env" / "team_best_costs.json"


def _utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _load_team_costs():
    try:
        return {int(k): v for k, v in json.loads(TEAM_COSTS.read_text()).items()}
    except Exception:
        return {}


# ---- solver cascade ---------------------------------------------------------------------
# Each solver: (name, fn(tid, pairs, infer, verify, budget_s) -> result_dict|None). Result dict:
#   {"cost": int, "points": float, "label": str, "method": name, "argen": (p,t)|None}
# Ordered cheapest/fastest first. beam_search is the substrate; einsum/batched slot in below as
# they land (they explore far larger spaces, which is what makes the "long run" long).

def _solve_beam(tid, pairs, infer, verify, budget_s, cfg):
    best, st = beam_search(pairs, infer_pairs=infer, verify=verify,
                           beam_width=cfg["beam"], max_depth=cfg["depth"],
                           policy=cfg["policy"], top_k=cfg["top_k"], max_evals=cfg["max_evals"])
    if not best:
        rej = st.get("verify", {}).get("rejected")
        return None if not rej else {"rejected_by_gate": rej}
    v = best.get("verify")
    return {"cost": best["cost"], "points": spec.points(best["cost"]), "label": best["label"],
            "method": "beam", "argen": (v[1], v[2]) if v else None}


def _solvers(cfg):
    """The cascade. Extend here: ('einsum', _solve_einsum), ('batched', _solve_batched)."""
    order = [("beam", lambda tid, p, i, v, b: _solve_beam(tid, p, i, v, b, cfg))]
    if cfg.get("einsum"):
        try:
            from .einsum_search import solve_einsum
            order.append(("einsum", lambda tid, p, i, v, b: solve_einsum(tid, p, i, v, b)))
        except Exception:
            pass
    if cfg.get("fit"):
        try:
            from .fit import solve_fit
            order.append(("fit", lambda tid, p, i, v, b: solve_fit(tid, p, i, v, b)))
        except Exception:
            pass
    if cfg.get("batched"):
        try:
            from .batched import solve_batched
            order.append(("batched", lambda tid, p, i, v, b: solve_batched(tid, p, i, v, b)))
        except Exception:
            pass
    return order


def solve_task(tid, cfg):
    """Run the solver cascade on one task; return the best (lowest-cost) argen-confirmed result."""
    task = tasks.load_task(tid)
    if task is None:
        return {"tid": tid, "solved": False, "reason": "no-data"}
    pairs = tasks.all_pairs(task)
    infer = argen.load_grids(tid, cfg["n_infer"]) or None if argen.available(tid) else None
    have_argen = argen.available(tid)

    def verify(model):
        p, t = score.argen_gate(model, tid, n=cfg["n_gate"])
        return (t > 0 and p == t, p, t)

    verify = verify if have_argen else None
    t0 = time.time()
    best, notes = None, []
    for name, fn in _solvers(cfg):
        remaining = cfg["budget_s"] - (time.time() - t0)
        if remaining <= 0:
            break
        try:
            r = fn(tid, pairs, infer, verify, remaining)
        except Exception as e:
            notes.append(f"{name}:err:{type(e).__name__}")
            continue
        if r and "cost" in r and (best is None or r["cost"] < best["cost"]):
            best = r
        elif r and "rejected_by_gate" in r:
            notes.append(f"{name}:gate-rejected {r['rejected_by_gate'][1]}/{r['rejected_by_gate'][2]}")
    res = {"tid": tid, "solved": best is not None, "seconds": round(time.time() - t0, 2),
           "argen_available": have_argen, "notes": notes}
    if best:
        res.update({"cost": best["cost"], "points": round(best["points"], 3),
                    "label": best["label"], "method": best["method"], "argen": best.get("argen")})
    return res


# ---- checkpoint / status ----------------------------------------------------------------
def _run_dir(name):
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _done_set(rundir):
    """Tasks already recorded in results.jsonl (resume skips these)."""
    done, results = set(), []
    f = rundir / "results.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add(int(r["tid"]))
                results.append(r)
            except Exception:
                continue
    return done, results


def _write_status(rundir, status):
    tmp = rundir / "status.tmp.json"
    tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
    tmp.replace(rundir / "status.json")


def _dashboard(status):
    """A compact live dashboard for the visible console (VizTerms window)."""
    s = status
    pct = 100 * s["done"] / max(1, s["total"])
    bar_n = 34
    fill = int(bar_n * pct / 100)
    bar = "#" * fill + "-" * (bar_n - fill)
    eta = s.get("eta_human", "?")
    line1 = f"[{bar}] {pct:5.1f}%  {s['done']}/{s['total']}  ETA {eta}"
    line2 = (f"  solved {s['solved']}  ngs-wins {s['wins']}  "
             f"avg {s.get('per_task_sec', 0):.1f}s/task  now: task{s.get('current_tid', 0):03d}  "
             f"state={s['state']}")
    return line1 + "\n" + line2


def _human(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def longrun(task_ids, name, cfg, resume=True):
    rundir = _run_dir(name)
    team = _load_team_costs()
    done, prior = (_done_set(rundir) if resume else (set(), []))
    todo = [t for t in task_ids if t not in done]
    total = len(task_ids)
    solved = sum(1 for r in prior if r.get("solved"))
    wins = sum(1 for r in prior if _is_win(r, team))
    started = time.time()
    durations = [r["seconds"] for r in prior if "seconds" in r]

    state = {"state": "running", "name": name, "started_utc": _utc(started),
             "total": total, "done": len(done), "solved": solved, "wins": wins,
             "cfg": {k: v for k, v in cfg.items() if k != "policy"}}

    def flush(current=None, extra=None):
        elapsed = time.time() - started
        per = (sum(durations) / len(durations)) if durations else 0.0
        n_done = len(done)
        eta = per * max(0, total - n_done)
        state.update({"heartbeat_utc": _utc(), "done": n_done, "solved": solved, "wins": wins,
                      "current_tid": current or 0, "per_task_sec": round(per, 2),
                      "eta_seconds": round(eta), "eta_human": _human(eta),
                      "elapsed_human": _human(elapsed)})
        if extra:
            state.update(extra)
        _write_status(rundir, state)

    # every terminal state is written (not just the happy path)
    def _terminal(kind, msg=None):
        state["state"] = kind
        if msg:
            state["message"] = msg
        state["heartbeat_utc"] = _utc()
        state["finished_utc"] = _utc()
        _write_status(rundir, state)
        (rundir / "done.json").write_text(json.dumps(
            {"state": kind, "solved": solved, "wins": wins, "done": len(done),
             "total": total, "finished_utc": _utc(), "message": msg}, indent=1), encoding="utf-8")

    def _sig(signum, frame):
        _terminal("interrupted", f"signal {signum}")
        print(f"\n[longrun] interrupted (signal {signum}); {len(done)}/{total} done. "
              f"Re-launch the same command to resume.", flush=True)
        os._exit(130)

    try:
        signal.signal(signal.SIGINT, _sig)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _sig)
    except Exception:
        pass

    print(f"[longrun '{name}'] {len(todo)} to do ({len(done)} already done), {total} total. "
          f"run dir: {rundir}", flush=True)
    flush()
    resfile = open(rundir / "results.jsonl", "a", encoding="utf-8")
    try:
        for tid in todo:
            flush(current=tid)
            r = solve_task(tid, cfg)
            r["team_cost"] = team.get(tid)
            r["ngs_win"] = _is_win(r, team)
            resfile.write(json.dumps(r) + "\n")
            resfile.flush()
            done.add(tid)
            durations.append(r.get("seconds", 0))
            if r.get("solved"):
                solved += 1
            if r["ngs_win"]:
                wins += 1
            flush(current=tid)
            _log_line(r)
            print(_dashboard(state), flush=True)
        _terminal("complete")
        print(f"[longrun '{name}'] COMPLETE. solved {solved}/{total}, ngs-wins {wins}.", flush=True)
        return 0
    except Exception:
        _terminal("crashed", traceback.format_exc().splitlines()[-1])
        print("[longrun] CRASHED:\n" + traceback.format_exc(), flush=True)
        return 1
    finally:
        resfile.close()


def _is_win(r, team):
    if not r.get("solved") or "cost" not in r:
        return False
    tc = team.get(int(r["tid"]))
    return tc is not None and r["cost"] < tc


def _log_line(r):
    if r.get("solved"):
        win = "  ** NGS WIN **" if r.get("ngs_win") else ""
        ag = f" argen {r['argen'][0]}/{r['argen'][1]}" if r.get("argen") else ""
        tc = r.get("team_cost")
        print(f"  task{r['tid']:03d}: SOLVED cost={r['cost']} pts={r['points']:.2f} "
              f"({r['method']}) team={tc}{ag}  {r['label']}{win}", flush=True)
    else:
        note = ("  " + "; ".join(r["notes"])) if r.get("notes") else ""
        print(f"  task{r['tid']:03d}: -{note}", flush=True)


# ---- task-list parsing + CLI ------------------------------------------------------------
def _parse_tasks(spec_str):
    """'1-400' | 'hard' (team cost>100) | 'cheap' (team cost<=100) | 'unsolved' | comma/space list."""
    team = _load_team_costs()
    if spec_str == "hard":
        return sorted(t for t, c in team.items() if c is not None and c > 100)
    if spec_str == "cheap":
        return sorted(t for t, c in team.items() if c is not None and c <= 100)
    if spec_str in ("all", "1-400"):
        return list(range(1, 401))
    out = []
    for part in spec_str.replace(",", " ").split():
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _cfg_from_argv(argv):
    def opt(name, default, cast=str):
        return cast(argv[argv.index(name) + 1]) if name in argv else default
    from .policy import HeuristicPolicy
    return {
        "depth": opt("--depth", 2, int), "beam": opt("--beam", 8, int),
        "top_k": (opt("--top-k", None, int) if "--top-k" in argv else None),
        "max_evals": opt("--max-evals", 5000, int),
        "policy": (HeuristicPolicy().rank if "--no-policy" not in argv else None),
        "n_infer": opt("--n-infer", 48, int), "n_gate": opt("--n-gate", 512, int),
        "budget_s": opt("--budget", 30, float), "einsum": "--einsum" in argv,
        "batched": "--batched" in argv, "fit": "--fit" in argv,
    }


def watch(name, interval=2.0):
    """Tail a run's status.json and render the live dashboard — a standalone 'visual window' that can
    watch a longrun launched in another console. Exits when the run reaches a terminal state."""
    rundir = _run_dir(name)
    p = rundir / "status.json"
    print(f"[watch '{name}'] tailing {p}  (Ctrl+C to stop watching)", flush=True)
    last = None
    while True:
        try:
            s = json.loads(p.read_text())
        except Exception:
            print("  (waiting for status.json ...)", flush=True)
            time.sleep(interval)
            continue
        stamp = s.get("heartbeat_utc")
        if stamp != last:
            print("\x1b[2J\x1b[H", end="")            # clear screen (ANSI) for a stable dashboard
            print(f"NGS longrun '{name}'  (heartbeat {stamp})")
            print(_dashboard(s))
            last = stamp
        if s.get("state") in ("complete", "interrupted", "crashed"):
            print(f"\n[watch] run {s['state']}.", flush=True)
            return 0
        time.sleep(interval)


def main():
    argv = sys.argv[1:]
    name = argv[argv.index("--name") + 1] if "--name" in argv else "run"
    if "--watch" in argv:
        return watch(name)
    if "--status" in argv:
        p = _run_dir(name) / "status.json"
        print(p.read_text() if p.exists() else f"no status at {p}")
        return 0
    task_spec = argv[argv.index("--tasks") + 1] if "--tasks" in argv else "hard"
    task_ids = _parse_tasks(task_spec)
    cfg = _cfg_from_argv(argv)
    resume = "--no-resume" not in argv
    print(f"[longrun] tasks='{task_spec}' ({len(task_ids)} tasks) name='{name}' "
          f"budget={cfg['budget_s']}s/task resume={resume}", flush=True)
    return longrun(task_ids, name, cfg, resume=resume)


if __name__ == "__main__":
    sys.exit(main())
