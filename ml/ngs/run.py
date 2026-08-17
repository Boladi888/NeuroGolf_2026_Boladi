"""run.py — CLI for the search substrate.

  python -m ml.ngs.run selftest              # synthetic identity+transpose must solve (substrate proof)
  python -m ml.ngs.run task <tid>            # search a real task; report best program + cost + points
  python -m ml.ngs.run sweep <a> <b>         # sweep tasks [a,b]; report solved count + costs
Options: --beam N --depth N --max-evals N

Run from repo root with the project venv:
  .venv/Scripts/python.exe -m ml.ngs.run selftest
"""
import sys
from . import spec, tasks, score, argen
from .search import beam_search

N_INFER = 48       # ARC-GEN pairs the data-aware ops read to infer their arguments (robustly)
N_GATE = 512       # ARC-GEN pairs a visible-solve must ALSO pass to count as a trustworthy solve


def _argen_hooks(tid):
    """(infer_pairs, verify) drawn from the ARC-GEN corpus, or (None, None) if unavailable.
    infer_pairs feeds arg-inference; verify is the truth gate that confirms a visible solve."""
    if not argen.available(tid):
        return None, None
    infer = argen.load_grids(tid, N_INFER) or None

    def verify(model):
        p, t = score.argen_gate(model, tid, n=N_GATE)
        return (t > 0 and p == t, p, t)

    return infer, verify


def _gate_tag(best, st):
    """A short ' argen p/t' (or rejection note) for reporting the truth-gate outcome."""
    if best and best.get("verify"):
        p, t = best["verify"][1], best["verify"][2]
        return f"  [argen {p}/{t}]"
    rej = st.get("verify", {}).get("rejected")
    if rej:
        return f"  [visible-solve REJECTED by argen gate {rej[1]}/{rej[2]}]"
    return ""


def _opt(name, default, cast=int):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def _policy():
    """--policy heuristic|none[:K]  -> (policy_callable|None, top_k|None)."""
    kind = _opt("--policy", "none", str)
    top_k = _opt("--top-k", None)
    if kind == "heuristic":
        from .policy import HeuristicPolicy
        return HeuristicPolicy().rank, top_k
    return None, top_k


def _cfg():
    pol, top_k = _policy()
    return dict(beam_width=_opt("--beam", 8), max_depth=_opt("--depth", 3),
                max_evals=_opt("--max-evals", 5000), policy=pol, top_k=top_k)


def selftest():
    rc = 0
    pol, top_k = _policy()
    tag = " [policy=heuristic]" if pol else ""
    print(f"NGS substrate selftest{tag}")
    # (kind, max_cost, n_pairs) — geometric kinds use more pairs so CoordRemap's cell fingerprints
    # are unambiguous, and a looser max_cost (a windowed Gather->Pad carries a small intermediate).
    for kind, max_cost, npair in (("identity", 10, 6), ("transpose", 10, 6), ("colorswap", 60, 6),
                                  ("rot180", 8000, 10), ("mirror2x2", 12000, 10)):
        task = tasks.synthetic(kind, n=npair, seed=0)
        pairs = tasks.all_pairs(task)
        best, st = beam_search(pairs, max_depth=2, beam_width=6, policy=pol, top_k=top_k)
        if best:
            print(f"  [{kind:9}] SOLVED  cost={best['cost']:<6} ({best['cost_kind']})  "
                  f"pts={spec.points(best['cost']):.2f}  prog: {best['label']}  (evals {st['evals']})")
            if best["cost"] > max_cost:
                print(f"           ! cost {best['cost']} > expected <= {max_cost}"); rc = 1
        else:
            print(f"  [{kind:9}] NOT SOLVED  (evals {st['evals']})  <-- substrate broken"); rc = 1
    print("SELFTEST", "PASS" if rc == 0 else "FAIL")
    return rc


def one_task(tid, cfg):
    task = tasks.load_task(tid)
    if task is None:
        print(f"task{tid}: data not found"); return 2
    pairs = tasks.all_pairs(task)
    infer, verify = _argen_hooks(tid)
    best, st = beam_search(pairs, infer_pairs=infer, verify=verify, **cfg)
    if best:
        print(f"task{int(tid):03d}: SOLVED  cost={best['cost']} ({best['cost_kind']})  "
              f"pts={spec.points(best['cost']):.2f}  prog: {best['label']}  "
              f"(evals {st['evals']}, {len(pairs)} pairs){_gate_tag(best, st)}")
        return 0
    print(f"task{int(tid):03d}: no solving program found  "
          f"(evals {st['evals']}, {len(pairs)} pairs){_gate_tag(best, st)}")
    return 1


def sweep(a, b, cfg):
    solved = []
    for tid in range(a, b + 1):
        task = tasks.load_task(tid)
        if task is None:
            continue
        pairs = tasks.all_pairs(task)
        infer, verify = _argen_hooks(tid)
        best, st = beam_search(pairs, infer_pairs=infer, verify=verify, **cfg)
        tag = (f"SOLVED cost={best['cost']} pts={spec.points(best['cost']):.2f}  {best['label']}"
               if best else "-") + _gate_tag(best, st)
        print(f"  task{tid:03d} [{len(pairs):>3} pairs, {st['evals']:>4} evals]  {tag}")
        if best:
            solved.append((tid, best["cost"]))
    print(f"\nSWEEP {a}-{b}: solved {len(solved)} task(s)" +
          (": " + ", ".join(f"t{t}({c})" for t, c in solved) if solved else ""))
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 64
    cmd = sys.argv[1]
    if cmd == "selftest":
        return selftest()
    if cmd == "task" and len(sys.argv) >= 3:
        return one_task(int(sys.argv[2]), _cfg())
    if cmd == "sweep" and len(sys.argv) >= 4:
        return sweep(int(sys.argv[2]), int(sys.argv[3]), _cfg())
    print(__doc__); return 64


if __name__ == "__main__":
    sys.exit(main())
