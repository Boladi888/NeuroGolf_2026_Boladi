# -*- coding: utf-8 -*-
"""captain_ctl.py — superseded proposal copy.

Live implementation: Agenters/Submit_Captain/captain_ctl.py.

The Submit Captain's mechanical work, pushed out of agent context into a script (see
`token_saving_coordination.md`). The captain RUNS THE OWN-STACK PROBES (the verification gate), reads a
compact digest, and stages only probe-verified candidates — it never reads inbox files, fires swaps, or
computes deltas in its own context.

  probe-batch  --lane <lane> [--apply]          grouped own-stack validation: budget-check, snapshot the
                                                home zip, swap team-best baseline + candidates, fire, spec
  probe-theory --lane <lane> --task <NNN> [--apply]   single own-stack probe (a reviewer RISK_SINGLE
                                                theory, or isolating a short batch)
  probe-collect [--apply]                       poll Kaggle for fired probes, compute deltas, write
                                                probe_delta/probe_ok into packets, RESTORE the home zip
  digest   [--lane all|claude|codex] [--json]   READY (probe-verified) / HELD (needs probe) / CONFLICT
  stage    --lane <lane> [--apply]              READY -> copy onnx + sidecar to Staged_Boladi, relay
                                                lesson -> Librarian inbox, archive packet
  reconcile [--apply]                           audit Staged_Boladi vs manifest (sha, points>=best, dups)
  budget                                        show today's Kaggle submit count vs the daily cap

The captain is the SINGLE own-stack prober (one Kaggle account + one home zip). The home zip is a SCRATCH
probe vehicle: probe-batch snapshots it first, probe-collect restores it, so probes never corrupt each
other's baselines and the durable ship target stays Staged_Boladi (the team folder), not the home zip.

Real-tool wiring (verified 2026-06-10): tools/swap_and_submit.py takes positional taskNNN*.onnx paths +
`--submit -m "<msg>"`, backs up submissions/submission.zip, and the -m message becomes the Kaggle
description. Baseline onnx come from Team_Submission/Backend/Team_Best_Onnx/taskNNN.onnx (401 present).

Conventions from the old nogit protocol: atomic write, lexical naming, write-disjoint dirs,
drain-newest/skip-seen/archive, refuse-without-sidecar, SILENCE IS VALID, BUDGET IS SHARED (don't
auto-retry a probe — the daily Kaggle cap is shared across probes + ships).
"""
import argparse, csv, datetime, hashlib, io, json, math, os, pathlib, shutil, subprocess, sys

# --- config (repoint here on implement) -------------------------------------------------
HERE  = pathlib.Path(__file__).resolve()
HOME  = HERE.parents[2]                                              # _proposals -> Agenters -> K-NeuroGolf
TEAM  = HOME.parent / "K-NeuroGolf-OnnxlyFans" / "Team_Submission"
STAGED    = TEAM / "Staged_Boladi"
MANIFEST  = TEAM / "Backend" / "manifest.csv"
TEAMBEST  = TEAM / "Backend" / "Team_Best_Onnx"                      # baseline onnx for the probe
HOME_ZIP  = HOME / "submissions" / "submission.zip"                  # the own-stack scratch vehicle
LEDGER    = HOME / "Logs" / "submissions_ledger.csv"
CAP_DIR   = HOME / "Agenters" / "Submit_Captain"                     # /Updates_<Lane>/, /_probe/
LIB_DIR   = HOME / "Agenters" / "Librarian"                          # /Updates_<Lane>/
PROBE_DIR = CAP_DIR / "_probe"
DONT_PROBE_FILE = CAP_DIR / "dont_probe.json"                       # [{task, sha256?, reason}] — never probe/stage
SWAP = ["python", "tools/swap_and_submit.py"]                        # own-stack swap+submit (runs in HOME)
COMP = "neurogolf-2026"
DAILY_CAP = 50    # UNRELIABLE: Kaggle 400'd at 34 submits/day (2026-06-10) — real cap is lower and/or rolling-24h. A
#                   count guard is NOT enough and the CLI doesn't expose the limit; treat a submit 400 as the real
#                   ceiling (back off, do NOT retry-spam), and _fire must abort + restore the home zip on a 400.
LANES = ("Claude", "Codex"); CREATOR = "Boladi"; REQUIRED = ("task", "onnx", "cost", "sha256")
BATCH_N = 5; MATCH_TOL = 0.10                                        # |delta - predicted| < tol -> verified

# --- helpers ----------------------------------------------------------------------------
def now_stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def today():     return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(text, encoding="utf-8"); os.replace(tmp, path)
def sha256_of(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def points_of(cost): return round(max(1.0, 25 - math.log(cost)), 4)
def resolve(rel): return pathlib.Path(rel) if os.path.isabs(rel) else (HOME / rel)
def lane_dir(base, lane): return base / f"Updates_{lane}"
def load_manifest():
    m = {}
    if MANIFEST.exists():
        for r in csv.DictReader(MANIFEST.open(encoding="utf-8")):
            try: m[r["task"]] = float(r["cost"])
            except Exception: pass
    return m
def load_dont_probe():
    shas, tasks, reasons = set(), set(), {}
    if DONT_PROBE_FILE.exists():
        try:
            for e in json.loads(DONT_PROBE_FILE.read_text(encoding="utf-8")):
                if not isinstance(e, dict): continue
                r = e.get("reason", "known public-negative")
                if e.get("sha256"): shas.add(e["sha256"]); reasons[e["sha256"]] = r
                elif e.get("task"): tasks.add(e["task"]); reasons[e["task"]] = r
        except Exception: pass
    return shas, tasks, reasons
def dont_probe_reason(pkt, dp):
    shas, tasks, reasons = dp
    if pkt.get("sha256") in shas: return reasons.get(pkt["sha256"])
    if pkt.get("task") in tasks: return reasons.get(pkt["task"])
    return None
def iter_packets(lane):
    for p in sorted(lane_dir(CAP_DIR, lane).glob("*.json")):
        if p.name.endswith(".tmp"): continue
        try: yield p, json.loads(p.read_text(encoding="utf-8"))
        except Exception as e: yield p, {"_error": str(e)}
def run(cmd):
    print("   $", " ".join(cmd[:2]), "...")
    return subprocess.run(cmd, cwd=str(HOME), capture_output=True, text=True, encoding="utf-8", errors="replace")

# --- Kaggle (standard `submissions --csv` cols: fileName,date,description,status,publicScore,privateScore)
def kaggle_rows():
    o = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "--csv"],
                       capture_output=True).stdout.decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(o)))
def score_by(rows, substr):
    for r in rows:
        if substr in (r.get("description") or "") and "complete" in (r.get("status") or "").lower():
            try: return float(r["publicScore"])
            except Exception: return None
    return None
def submits_today(rows): return sum(1 for r in rows if (r.get("date") or "")[:10] == today())
def budget_remaining():
    try: return max(0, DAILY_CAP - submits_today(kaggle_rows()))
    except Exception: return DAILY_CAP                              # can't reach Kaggle -> don't hard-block here

def classify(pkt, man, dp=None):
    if dp is None: dp = load_dont_probe()
    if "_error" in pkt: return "CONFLICT", f"unreadable ({pkt['_error']})"
    miss = [k for k in REQUIRED if k not in pkt]
    if miss: return "HELD", f"missing {miss}"
    onnx = resolve(pkt["onnx"])
    if not onnx.exists(): return "CONFLICT", f"onnx missing: {pkt['onnx']}"
    if sha256_of(onnx) != pkt["sha256"]: return "CONFLICT", "sha mismatch"
    dr = dont_probe_reason(pkt, dp)
    if dr: return "CONFLICT", f"DONT_PROBE: {dr}"
    best = man.get(pkt["task"])
    if best is not None and pkt["cost"] >= best * 0.999: return "CONFLICT", f"team-best {int(best)} <= {pkt['cost']} (no gain)"
    if "probe_ok" not in pkt: return "HELD", "needs probe (run probe-batch)"
    if not pkt["probe_ok"]: return "CONFLICT", f"probe repriced (delta {pkt.get('probe_delta')})"
    return "READY", None

# --- probe (the verification gate) ------------------------------------------------------
def _fire(lane, items, single):
    """items: [(packet_path, pkt, team_best)]; budget-check, snapshot home zip, fire baseline+candidates."""
    needed = 2                                                      # one baseline submit + one candidates submit
    rem = budget_remaining()
    if rem < needed:
        print(f"BUDGET BLOCK: {rem} Kaggle submits left today < {needed} needed (cap {DAILY_CAP}). Not firing."); return
    tasks = [pkt["task"] for _, pkt, _ in items]
    tag = f"probe-{lane}-{'1' if single else 'b'}-{now_stamp()}"
    base = [str(TEAMBEST / f"{t}.onnx") for t in tasks]
    cand = [str(resolve(pkt["onnx"])) for _, pkt, _ in items]
    predicted = round(sum(points_of(pkt["cost"]) - points_of(int(b)) for _, pkt, b in items if b), 3)
    print(f"PROBE {tag}: {tasks}  predicted +{predicted}  (budget {rem} left)")
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HOME_ZIP, PROBE_DIR / f"{tag}.prehome.zip")        # restore point: the scratch vehicle's pre-probe state
    run(SWAP + base + ["--submit", "-m", f"{tag} baseline team-best"])      # S_base
    run(SWAP + cand + ["--submit", "-m", f"{tag} candidates"])              # S_batch
    atomic_write(PROBE_DIR / f"{tag}.json", json.dumps(
        {"tag": tag, "lane": lane, "tasks": tasks, "predicted": predicted,
         "packets": [p.name for p, _, _ in items]}, indent=1))
    print(f"   fired ({tag}); after the poll lands run:  captain_ctl.py probe-collect --apply")

def cmd_probe_batch(args):
    man = load_manifest(); dp = load_dont_probe(); lane = args.lane.capitalize(); pend = []
    for p, pkt in iter_packets(lane):
        if pkt.get("probe") == "theory" or "probe_ok" in pkt: continue
        if [k for k in REQUIRED if k not in pkt]: continue
        if dont_probe_reason(pkt, dp): print(f"DONT_PROBE skip {pkt.get('task')}"); continue
        best = man.get(pkt["task"])
        if best and pkt["cost"] >= best * 0.999: continue
        pend.append((p, pkt, best))
    pend = pend[:BATCH_N]
    if not pend: print("(no un-probed batch candidates)"); return
    print(f"would probe {len(pend)}: {[k[1]['task'] for k in pend]}" + ("" if args.apply else " (dry-run)"))
    if args.apply: _fire(lane, pend, single=False)

def cmd_probe_theory(args):
    man = load_manifest(); lane = args.lane.capitalize()
    hit = [(p, pkt, man.get(pkt["task"])) for p, pkt in iter_packets(lane) if pkt.get("task") == f"task{args.task}"]
    if not hit: print(f"(no packet for task{args.task} in {lane})"); return
    if dont_probe_reason(hit[0][1], load_dont_probe()): print(f"DONT_PROBE: task{args.task} on the guardrail — refusing"); return
    print(f"would single-probe task{args.task}" + ("" if args.apply else " (dry-run)"))
    if args.apply: _fire(lane, hit[:1], single=True)

def cmd_probe_collect(args):
    rows = kaggle_rows()
    for spec_p in sorted(PROBE_DIR.glob("*.json")):
        if spec_p.name.endswith(".tmp"): continue
        spec = json.loads(spec_p.read_text(encoding="utf-8"))
        sb, sc = score_by(rows, f"{spec['tag']} baseline"), score_by(rows, f"{spec['tag']} candidates")
        if sb is None or sc is None: print(f"{spec['tag']}: pending"); continue
        delta = round(sc - sb, 3); ok = abs(delta - spec["predicted"]) < MATCH_TOL
        print(f"{spec['tag']}: delta {delta:+.2f} vs predicted {spec['predicted']:+.2f}  ->  "
              f"{'VERIFIED' if ok else 'SHORT -> isolate per-task (probe-theory)'}")
        if args.apply:
            for pname in spec["packets"]:
                pp = lane_dir(CAP_DIR, spec["lane"]) / pname
                if pp.exists():
                    pkt = json.loads(pp.read_text(encoding="utf-8")); pkt["probe_delta"] = delta; pkt["probe_ok"] = ok
                    atomic_write(pp, json.dumps(pkt, indent=1))
            prehome = PROBE_DIR / f"{spec['tag']}.prehome.zip"      # RESTORE the scratch vehicle
            if prehome.exists(): shutil.copy2(prehome, HOME_ZIP); prehome.unlink()
            (PROBE_DIR / "_done").mkdir(parents=True, exist_ok=True); os.replace(spec_p, PROBE_DIR / "_done" / spec_p.name)
            print(f"   home zip restored from {spec['tag']}.prehome.zip")

# --- drain / stage / reconcile ----------------------------------------------------------
def cmd_digest(args):
    man = load_manifest(); lanes = LANES if args.lane == "all" else (args.lane.capitalize(),)
    out = {"READY": [], "HELD": [], "CONFLICT": []}
    for lane in lanes:
        for p, pkt in iter_packets(lane):
            state, note = classify(pkt, man); best = man.get(pkt.get("task"))
            ev = round(math.log(best / pkt["cost"]), 2) if (state == "READY" and best) else None
            out[state].append({"task": pkt.get("task"), "lane": lane, "cost": pkt.get("cost"), "ev": ev, "note": note})
    out["READY"].sort(key=lambda r: -(r["ev"] or 0))
    if args.json: print(json.dumps(out, indent=1)); return
    if not any(out.values()): print("(inboxes empty — nothing to do)"); return
    print(f"== captain digest (lane: {args.lane}) ==")
    if out["READY"]:
        print("READY (probe-verified — stage these, EV-ranked):")
        for r in out["READY"]: print(f"  {r['task']}  cost {r['cost']}  pts {points_of(r['cost'])}  EV +{r['ev']}  [{r['lane']}]")
    for label in ("HELD", "CONFLICT"):
        if out[label]:
            print(f"{label}:")
            for r in out[label]: print(f"  {r['task']}  [{r['lane']}]  -- {r['note']}")

def cmd_stage(args):
    man = load_manifest(); lane = args.lane.capitalize(); staged = left = 0
    for p, pkt in iter_packets(lane):
        state, note = classify(pkt, man)
        if state != "READY": left += 1; print(f"SKIP {pkt.get('task')} [{state}] {note or ''}"); continue
        task = pkt["task"]; onnx = resolve(pkt["onnx"])
        sidecar = json.dumps({"task": task, "creator": CREATOR, "cost": pkt["cost"],
                              "points": points_of(pkt["cost"]), "sha256": pkt["sha256"], "msg": pkt.get("msg", "")}, indent=1)
        print(f"STAGE {task}  cost {pkt['cost']}  [{lane}]" + ("" if args.apply else "  (dry-run)"))
        if args.apply:
            shutil.copyfile(onnx, STAGED / f"{task}.onnx"); atomic_write(STAGED / f"{task}.json", sidecar)
            if pkt.get("lesson"): atomic_write(lane_dir(LIB_DIR, lane) / f"{now_stamp()}-{task}.lesson.md", str(pkt["lesson"]))
            (lane_dir(CAP_DIR, lane) / "_consumed").mkdir(parents=True, exist_ok=True)
            os.replace(p, lane_dir(CAP_DIR, lane) / "_consumed" / p.name)
        staged += 1
    print(f"-- {'staged' if args.apply else 'would stage'} {staged}, left {left} --" + ("" if args.apply or not staged else "  (--apply to commit)"))

def cmd_reconcile(args):
    man = load_manifest(); issues = 0; n = 0
    for js in sorted(STAGED.glob("*.json")):
        n += 1; sc = json.loads(js.read_text(encoding="utf-8")); onnx = STAGED / f"{sc['task']}.onnx"
        if not onnx.exists(): print(f"MISSING onnx {sc['task']}"); issues += 1; continue
        if sha256_of(onnx) != sc.get("sha256"): print(f"SHA DRIFT {sc['task']}"); issues += 1
        best = man.get(sc["task"])
        if best is not None and sc["cost"] >= best * 1.001: print(f"REGRESSION {sc['task']}: {sc['cost']} >= best {int(best)}"); issues += 1
    print(f"-- reconcile: {issues} issue(s) across {n} staged --")

def cmd_budget(args):
    try: rows = kaggle_rows(); print(f"today: {submits_today(rows)} / {DAILY_CAP} submits used, {budget_remaining()} left")
    except Exception as e: print(f"(could not reach Kaggle: {e})")

def main():
    ap = argparse.ArgumentParser(description="Submit Captain probe/drain/stage/reconcile (superseded proposal copy)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("probe-batch"); pb.add_argument("--lane", required=True, choices=["claude", "codex"]); pb.add_argument("--apply", action="store_true"); pb.set_defaults(fn=cmd_probe_batch)
    pt = sub.add_parser("probe-theory"); pt.add_argument("--lane", required=True, choices=["claude", "codex"]); pt.add_argument("--task", required=True); pt.add_argument("--apply", action="store_true"); pt.set_defaults(fn=cmd_probe_theory)
    pc = sub.add_parser("probe-collect"); pc.add_argument("--apply", action="store_true"); pc.set_defaults(fn=cmd_probe_collect)
    dg = sub.add_parser("digest"); dg.add_argument("--lane", default="all", choices=["all", "claude", "codex"]); dg.add_argument("--json", action="store_true"); dg.set_defaults(fn=cmd_digest)
    st = sub.add_parser("stage"); st.add_argument("--lane", required=True, choices=["claude", "codex"]); st.add_argument("--apply", action="store_true"); st.set_defaults(fn=cmd_stage)
    rc = sub.add_parser("reconcile"); rc.add_argument("--apply", action="store_true"); rc.set_defaults(fn=cmd_reconcile)
    bd = sub.add_parser("budget"); bd.set_defaults(fn=cmd_budget)
    args = ap.parse_args(); args.fn(args)

if __name__ == "__main__":
    sys.exit(main())
