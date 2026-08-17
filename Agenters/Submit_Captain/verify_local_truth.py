#!/usr/bin/env python3
"""Owner directive 2026-07-13: verify a HELD candidate with an on-box `ng_gate truth --n 8000`
instead of a Kaggle stack-probe submit (flake-free, cap-free; this box is x86 like the grader).

Certifies ONLY when the candidate is at least as correct as the LIVE parent (no hidden regression)
AND strictly cheaper. On pass, stamps the packet probe_ok/probe_kind=local_truth_n8000 so the captain
`stage` path accepts it. Read-only unless --apply.

Usage: python verify_local_truth.py --lane claude --task NNN [--n 8000] [--apply]
"""
import argparse, glob, json, os, subprocess, sys, hashlib, pathlib

HERE = pathlib.Path(__file__).resolve().parent
HOME = HERE.parents[1]                                   # Submit_Captain -> Agenters -> K-NeuroGolf
TEAMBEST = HOME.parent / "K-NeuroGolf-OnnxlyFans" / "Team_Submission" / "Backend" / "Team_Best_Onnx"
NGGATE = HOME / "tools" / "ng_gate.py"

def sha256(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def truth(task_num, onnx, n):
    """Run ng_gate truth --n <n> --json; return (correct, total) or (None, None)."""
    proc = subprocess.run([sys.executable, str(NGGATE), "truth", "--task", str(task_num),
                           "--cand", str(onnx), "--n", str(n), "--json"],
                          capture_output=True, text=True, cwd=str(HOME))
    try:
        d = json.loads(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stdout)
    except Exception:
        # find the json object in stdout
        s = proc.stdout
        i = s.rfind("{")
        d = json.loads(s[i:]) if i >= 0 else {}
    if d.get("error"):
        print(f"   truth error: {d['error']}"); return None, None
    return d.get("correct"), d.get("total")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    tnum = str(int(a.task.replace("task", "")))
    tkey = f"task{int(tnum):03d}"
    inbox = HERE / "Updates_Claude" if a.lane.lower() == "claude" else HERE / "Updates_Codex"
    pkts = glob.glob(str(inbox / f"*{tkey}*.json"))
    if not pkts:
        print(f"no packet for {tkey} in {inbox.name}"); return 2
    pp = pathlib.Path(pkts[0]); d = json.loads(pp.read_text(encoding="utf-8"))
    # LOCAL-TRUTH ONLY CERTIFIES BYTE-EQUAL / INHERITED candidates (identical outputs to LIVE): the only
    # variable is cost, and there is no grader-regression risk. PATH_B / output-CHANGING candidates must go to
    # a real Kaggle probe -- local truth verifies correctness but NOT that the gain materializes on the grader,
    # and the local corpus may not cover the grader's rows (2026-07-12: task131 PATH_B int8pad passed truth
    # 2000/2000 but scored +0.00 on Kaggle -- the quantized cost saving was illusory).
    BYTE_EQUAL_CLASSES = {"path_a_inherited", "path_a_byte_equal", "byte_equal_inherited",
                          "parent_byte_equal", "ab_full_byte_equal"}
    cls = str(d.get("candidate_class") or "").strip().lower()
    if cls not in BYTE_EQUAL_CLASSES:
        print(f"{tkey}: candidate_class={cls or '(missing)'} is NOT byte-equal -- local-truth cannot certify a "
              f"gain-materialization for output-changing candidates; run a Kaggle probe-stack instead")
        return 4
    cand = (d.get("onnx") or "").strip()
    cand_p = cand if os.path.isabs(cand) else str(HOME / cand)
    live_p = TEAMBEST / f"{tkey}.onnx"
    cand_cost = d.get("cost"); live_cost = d.get("live_cost")
    print(f"{tkey}: cand_cost={cand_cost} live_cost={live_cost} class={d.get('candidate_class')}")
    cc, ct = truth(tnum, cand_p, a.n)
    lc, lt = truth(tnum, live_p, a.n)
    print(f"   truth cand: {cc}/{ct}   truth LIVE: {lc}/{lt}")
    if cc is None or lc is None:
        print("   -> truth run failed; NOT certified"); return 3
    ok = (cc >= lc) and (cand_cost is not None and live_cost is not None and cand_cost < live_cost)
    print(f"   -> cand_correct>=live_correct: {cc>=lc} | cheaper: {cand_cost<live_cost if (cand_cost and live_cost) else '?'} => {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("   NOT certified (regression or no cost gain)"); return 1
    if a.apply:
        onnx_sha = sha256(cand_p)
        d["probe_ok"] = True
        d["probe_kind"] = "local_truth_n8000"
        d["candidate_status"] = "confirmed_public"
        d["probe_sha"] = onnx_sha
        d["probe_baseline_source"] = "local_ng_gate_truth"
        d["kaggle_result_reason"] = f"local truth n={a.n}: cand {cc}/{ct} >= LIVE {lc}/{lt}, cost {cand_cost}<{live_cost}"
        d["probe_note"] = ("LOCAL-TRUTH verified via ng_gate truth --n 8000 (no Kaggle submit) per owner "
                           "directive 2026-07-13; on-box x86 grader-equivalent, flake-free")
        pp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f"   CERTIFIED (local_truth_n8000) -> {pp.name}; stage with captain_ctl stage --lane {a.lane} --apply")
    else:
        print("   PASS (dry-run; --apply to certify)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
