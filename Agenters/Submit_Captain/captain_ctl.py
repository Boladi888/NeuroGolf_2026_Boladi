# -*- coding: utf-8 -*-
"""captain_ctl.py - Submit Captain control CLI.

The Submit Captain's mechanical work, pushed out of agent context into a script (see
`token_saving_coordination.md`). The captain RUNS THE OWN-STACK PROBES (the verification gate) by driving
this CLI: the agent reads compact digests and decisions, while this script reads inbox files, fires swaps,
computes deltas, restores the home zip, and stages only probe-verified candidates.

  probe-batch  --lane <lane> [--apply]          grouped own-stack validation: snapshot the home zip,
                                                swap team-best baseline + candidates, fire, spec
  probe-theory --lane <lane> --task <NNN> [--apply]   single own-stack probe (a reviewer RISK_SINGLE
                                                theory, or isolating a short batch)
  probe-isolated --lane <l> [--task NNN] [--apply]    0-base probe: candidates vs the fail-onnx -> publicScore =
                                                absolute points (NO baseline submit); verifies grade + gain vs team-best
  probe-collect [--apply]                       poll Kaggle for fired probes, compute deltas, write
                                                probe_delta/probe_ok into packets, RESTORE the home zip
  digest   [--lane all|claude|codex] [--json]   READY (probe/local-gold verified) / HELD (needs probe) / CONFLICT
  stage    --lane <lane> [--apply]              READY -> copy onnx + sidecar to Staged_Boladi, relay
                                                lesson -> Librarian inbox, archive packet
  reconcile [--apply]                           audit Staged_Boladi vs manifest (sha, points>=best, dups)
  reconcile-live                                audit manifest vs Team_Best_Onnx vs submission.zip (merge-gap) + un-banked staged gains
  reject   --lane <lane> --task <NNN> --reason <r> [--apply]   dispose a dead/stale packet -> _consumed/*.rejected_<r>.json
  usage                                         show today's observed Kaggle submit count; diagnostic only

The captain is the SINGLE own-stack prober (one Kaggle account + one home zip). The home zip is a SCRATCH
probe vehicle: probe-batch snapshots it first, probe-collect restores it, so probes never corrupt each
other's baselines and the durable ship target stays Staged_Boladi (the team folder), not the home zip.
After every restore, the script asserts the restored home-zip SHA equals the pre-probe snapshot SHA; a mismatch blocks
collection/staging and writes a `_probe/*.home_zip_block.json` record.

Real-tool wiring (verified 2026-06-10): tools/swap_and_submit.py takes positional taskNNN*.onnx paths +
`--submit -m "<msg>"`, backs up submissions/submission.zip, and the -m message becomes the Kaggle
description. Baseline onnx come from Team_Submission/Backend/Team_Best_Onnx/taskNNN.onnx (401 present).

Conventions from the old nogit protocol: atomic write, lexical naming, write-disjoint dirs,
drain-newest/skip-seen/archive, refuse-without-sidecar, SILENCE IS VALID, RATE LIMITS ARE SHARED (don't
auto-retry after a Kaggle 400; restore the home zip and retry later).
"""
import argparse, csv, datetime, hashlib, io, json, math, os, pathlib, shutil, subprocess, sys, zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- config -----------------------------------------------------------------------------
HERE  = pathlib.Path(__file__).resolve()
HOME  = HERE.parents[2]                                              # Submit_Captain -> Agenters -> K-NeuroGolf
TEAM  = HOME.parent / "K-NeuroGolf-OnnxlyFans" / "Team_Submission"
STAGED    = TEAM / "Staged_Boladi"
MANIFEST  = TEAM / "Backend" / "manifest.csv"
TEAMBEST  = TEAM / "Backend" / "Team_Best_Onnx"                      # baseline onnx for the probe
HOME_ZIP  = HOME / "submissions" / "submission.zip"                  # the own-stack scratch vehicle
SUBMISSION = TEAM / "Backend" / "submission.zip"                     # the team's live merged submission (reconcile-live target)
LEDGER    = HOME / "Logs" / "submissions_ledger.csv"
CAP_DIR   = HOME / "Agenters" / "Submit_Captain"                     # /Updates_<Lane>/, /_probe/
LIB_DIR   = HOME / "Agenters" / "Librarian"                          # /Updates_<Lane>/
RUNS_DIR  = HOME / "Agenters" / "Orchestrator" / "runs"              # Orchestrator run ledgers the captain stamps (writeback)
PROBE_DIR = CAP_DIR / "_probe"
DONT_PROBE_FILE = CAP_DIR / "dont_probe.json"                       # [{task, sha256?, reason}] — never probe/stage
RESCUE_QUEUE = HOME / "Handoff" / "private" / "review_rescue_queue.jsonl"  # strict-gain packets that lost their reviewer verdict (skill runner-workflow.md)
SWAP = ["python", "tools/swap_and_submit.py"]                        # own-stack swap+submit (runs in HOME)
FAIL_ONNX = HOME / "tools" / "isolation_fail.onnx"                   # 0-base for isolated probes (regen: tools/build_isolation_fail.py)
FAIL_GEN  = HOME / "tools" / "build_isolation_fail.py"
COMP = "neurogolf-2026"
ADVISORY_RATE_REFERENCE = 50  # Diagnostic reference only. Kaggle 400s are authoritative and trigger abort+restore.
LANES = ("Claude", "Codex"); CREATOR = "Boladi"; REQUIRED = ("task", "source_lane", "onnx", "cost", "sha256")
BATCH_N = 5; MATCH_TOL = 0.10                                        # |delta - predicted| < tol -> verified
CONFIRMED_STATUS = "confirmed_public"
STUCK_CS = ("", "held", "queued", "safe_batch_queue")               # captain_status values run_audit flags non-terminal (writeback targets)

# --- helpers ----------------------------------------------------------------------------
def now_stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def packet_age_str(p):
    try:
        secs = datetime.datetime.now(datetime.timezone.utc).timestamp() - p.stat().st_mtime
        return f"{secs/86400:.1f}d" if secs >= 86400 else f"{secs/3600:.1f}h"
    except Exception: return "?"
def today():     return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(text, encoding="utf-8"); os.replace(tmp, path)
def sha256_of(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def norm_sha(s): return str(s or "").strip().lower()
def points_of(cost): return round(max(1.0, 25 - math.log(max(1, cost))), 6)  # clamp: a 0/negative cost (e.g. a placeholder manifest entry like task067=0) must not crash log
def candidate_status_from_kaggle(expected, actual):
    if expected is None or actual is None:
        return "unknown", None
    deviation = actual - expected
    tol = max(MATCH_TOL, abs(expected) * 0.25)
    if abs(deviation) <= tol:
        return "confirmed_public", "OK"
    if expected > 0 and actual <= max(MATCH_TOL, expected * 0.25):
        return "hidden_failed", f"expected-if-correct {expected:+.3f}, actual {actual:+.3f}"
    return "suspicious", f"expected-if-correct {expected:+.3f}, actual {actual:+.3f}"
def resolve(rel): return pathlib.Path(rel) if os.path.isabs(rel) else (HOME / rel)
def lane_dir(base, lane): return base / f"Updates_{lane}"
def norm_task(task):
    s = str(task or "").strip()
    if s.lower().startswith("task"):
        s = s[4:]
    try:
        return f"task{int(s):03d}"
    except Exception:
        return str(task or "").strip()
def load_manifest():
    m = {}
    if MANIFEST.exists():
        for r in csv.DictReader(MANIFEST.open(encoding="utf-8")):
            try: m[norm_task(r["task"])] = float(r["cost"])
            except Exception: pass
    return m
def load_manifest_full():
    """{task: (cost, onnx_sha256)} for reconcile-live — the manifest<->Team_Best_Onnx<->submission audit."""
    m = {}
    if MANIFEST.exists():
        for r in csv.DictReader(MANIFEST.open(encoding="utf-8")):
            try: m[norm_task(r["task"])] = (float(r["cost"]), norm_sha(r.get("onnx_sha256")))
            except Exception: pass
    return m
def load_dont_probe():
    # DISABLED 2026-06-20 (owner directive): NO task is ever permanently blocked from probing/improvement.
    # Every task stays improvable with a fresh formulation — a stale guard wrongly blocked task177's +0.554
    # byte-equal win for ~9 days. Hidden-bad FORMULATIONS now live in Onnx_Advanced/traps.md (T3 §PUBLIC-NEG),
    # not as a task-level blocklist. Always returns empty so dont_probe_reason() never blocks; dont_probe.json is ignored.
    return set(), set(), {}
def dont_probe_reason(pkt, dp):
    shas, tasks, reasons = dp
    h = norm_sha(pkt.get("sha256"))
    if h in shas: return reasons.get(h)
    task = norm_task(pkt.get("task"))
    if task in tasks: return reasons.get(task)
    return None
def rescue_row_key(row):
    task = norm_task(row.get("task"))
    cand = str(row.get("candidate_path") or "")
    if cand:
        try:
            cand = os.path.normcase(str(resolve(cand).resolve()))
        except Exception:
            cand = os.path.normcase(str(resolve(cand)))
    return task, cand

def load_rescue_rows(open_only=False):
    """Strict-gain worker packets that lost their reviewer verdict (timeout / unparseable / gate-wait). One JSONL row
    per rescue; a later row for the same canonical (task, candidate_path) supersedes, so a 'resolved' marker clears it.

    Candidate paths are canonicalized so an absolute-path resolution clears a row
    originally written with a relative path, and vice versa.
    """
    rows = []
    if RESCUE_QUEUE.exists():
        for line in RESCUE_QUEUE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try: rows.append(json.loads(line))
            except Exception: continue
    latest = {}
    for r in rows:
        latest[rescue_row_key(r)] = r
    rows = list(latest.values())
    if open_only:
        rows = [r for r in rows if str(r.get("status", "open")).lower() == "open"]
    return rows
def compact_rescue_rows(apply=False):
    """Rewrite the append-only rescue queue to one latest row per canonical candidate.

    This keeps the useful current state while removing stale raw `open` rows that
    older watchers/tools would otherwise keep rendering forever.
    """
    raw = []
    parse_errors = 0
    if RESCUE_QUEUE.exists():
        for line in RESCUE_QUEUE.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except Exception:
                parse_errors += 1
    latest = {}
    for r in raw:
        latest[rescue_row_key(r)] = r
    compacted = list(latest.values())
    open_before = sum(1 for r in raw if str(r.get("status", "open")).lower() == "open")
    open_after = sum(1 for r in compacted if str(r.get("status", "open")).lower() == "open")
    backup = None
    if apply and RESCUE_QUEUE.exists():
        backup = RESCUE_QUEUE.with_name(f"{RESCUE_QUEUE.name}.{now_stamp()}.bak")
        shutil.copy2(RESCUE_QUEUE, backup)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in compacted)
        atomic_write(RESCUE_QUEUE, text)
    return {
        "path": str(RESCUE_QUEUE),
        "applied": bool(apply),
        "backup": str(backup) if backup else None,
        "raw_rows": len(raw),
        "compacted_rows": len(compacted),
        "removed_rows": len(raw) - len(compacted),
        "parse_errors": parse_errors,
        "open_before_raw": open_before,
        "open_after_compact": open_after,
    }
def append_rescue_row(row):
    RESCUE_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with RESCUE_QUEUE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
# Packet key aliases — normalize the older template's field names so a mis-keyed packet is accepted, not silently HELD
# (the F2 fix: alpha_p45b emitted candidate_path/static_cost/captain_lane + no source_lane; the captain rejected all 3).
PKT_ALIASES = {"onnx": ("candidate_path", "onnx_path", "path"),
               "cost": ("static_cost", "cost_static"),
               "lane": ("captain_lane", "queue_lane")}
def normalize_packet(pkt):
    if not isinstance(pkt, dict): return pkt
    for canon, alts in PKT_ALIASES.items():
        if pkt.get(canon) in (None, ""):
            for a in alts:
                if pkt.get(a) not in (None, ""):
                    pkt[canon] = pkt[a]; break
    if not pkt.get("source_lane") and pkt.get("_inbox_lane"):     # a packet's inbox folder IS its source lane
        pkt["source_lane"] = str(pkt["_inbox_lane"]).capitalize()
    return pkt
def iter_packets(lane):
    for p in sorted(lane_dir(CAP_DIR, lane).glob("*.json")):
        if p.name.endswith(".tmp"): continue
        try:
            pkt = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(pkt, dict):
                pkt["_inbox_lane"] = lane
                normalize_packet(pkt)
            yield p, pkt
        except Exception as e: yield p, {"_error": str(e)}
def run(cmd):
    print("   $", " ".join(cmd[:2]), "...")
    return subprocess.run(cmd, cwd=str(HOME), capture_output=True, text=True, encoding="utf-8", errors="replace")

def tail(text, n=1200):
    text = (text or "").strip()
    return text[-n:] if len(text) > n else text

def record_failed_probe(tag, lane, tasks, predicted, stage, proc, prehome):
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(PROBE_DIR / f"{tag}.failed.json", json.dumps({
        "tag": tag,
        "lane": lane,
        "tasks": tasks,
        "predicted": predicted,
        "failed_stage": stage,
        "returncode": proc.returncode,
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
        "restore_source": str(prehome),
        "failed_utc": now_stamp(),
    }, indent=1))

def record_home_zip_block(tag, prehome, expected, got, reason):
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(PROBE_DIR / f"{tag}.home_zip_block.json", json.dumps({
        "tag": tag,
        "home_zip": str(HOME_ZIP),
        "restore_source": str(prehome),
        "expected_sha256": expected,
        "actual_sha256": got,
        "reason": reason,
        "blocked_utc": now_stamp(),
    }, indent=1))

def restore_home_zip(prehome, tag=None):
    tag = tag or prehome.stem
    if not prehome.exists():
        print(f"HOME ZIP RESTORE BLOCK: missing snapshot {prehome}")
        record_home_zip_block(tag, prehome, None, sha256_of(HOME_ZIP) if HOME_ZIP.exists() else None, "missing pre-probe snapshot")
        return False
    expected = sha256_of(prehome)
    shutil.copy2(prehome, HOME_ZIP)
    got = sha256_of(HOME_ZIP)
    if got != expected:
        print(f"HOME ZIP RESTORE BLOCK: restored SHA mismatch for {tag}; do not stage or fire another probe")
        record_home_zip_block(tag, prehome, expected, got, "restored home zip does not match snapshot")
        return False
    print(f"   restored home zip from {prehome.name} (sha verified)")
    return True

def write_swap_snapshot_meta(tag, kind, items):
    """Metadata consumed by tools/swap_and_submit.py for immutable probe submit snapshots."""
    tasks = {}
    expected = 0.0
    for _, pkt, best in items:
        task = norm_task(pkt["task"])
        if kind == "baseline":
            cost = best
            onnx = TEAMBEST / f"{task}.onnx"
            status = "confirmed_baseline"
        else:
            cost = pkt["cost"]
            onnx = resolve(pkt["onnx"])
            status = "probe_candidate_unverified"
        pts = points_of(cost) if cost else None
        if pts is not None:
            expected += pts
        tasks[task] = {
            "task": task,
            "source_path": str(onnx),
            "onnx_sha256": sha256_of(onnx) if onnx.exists() else None,
            "cost": cost,
            "points": pts,
            "expected_score_if_correct": pts,
            "candidate_status": status,
        }
    meta = {
        "tag": f"{tag}-{kind}",
        "score_kind": "local_points_if_correct",
        "expected_score_if_correct": round(expected, 6),
        "tasks": tasks,
    }
    path = PROBE_DIR / f"{tag}.{kind}.snapshot_meta.json"
    atomic_write(path, json.dumps(meta, indent=1))
    return path

def write_direct_probe_snapshot(tag, zip_path, items, expected_score_if_correct, candidate_message):
    """Snapshot for direct Kaggle probe submits that do not go through swap_and_submit.py."""
    snap_dir = PROBE_DIR / "_snapshots" / tag
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_zip = snap_dir / "submission.zip"
    shutil.copy2(zip_path, snap_zip)
    rows = []
    for _, pkt, _ in items:
        task = norm_task(pkt["task"])
        onnx = resolve(pkt["onnx"])
        rows.append({
            "task": task,
            "source_path": str(onnx),
            "onnx_sha256": sha256_of(onnx),
            "cost": pkt["cost"],
            "points_if_correct": points_of(pkt["cost"]),
            "expected_score_if_correct": points_of(pkt["cost"]),
            "candidate_status": "probe_candidate_unverified",
        })
    manifest = {
        "schema": "neurogolf_probe_submit_snapshot_v1",
        "tag": tag,
        "timestamp_utc": now_stamp(),
        "kaggle_message": candidate_message,
        "zip_path": "submission.zip",
        "zip_sha256": sha256_of(snap_zip),
        "selected_tasks": [r["task"] for r in rows],
        "tasks": rows,
        "score_kind": "local_points_if_correct",
        "expected_score_if_correct": expected_score_if_correct,
        "confirmed_kaggle_public_score": None,
        "candidate_status": "submitted_unverified",
    }
    atomic_write(snap_dir / "snapshot_manifest.json", json.dumps(manifest, indent=1))
    return snap_dir

def mark_probe_snapshot(spec, candidate_status, reason, actual_score):
    snap = spec.get("snapshot_manifest")
    if not snap:
        return
    path = pathlib.Path(snap)
    if not path.exists():
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    manifest["candidate_status"] = candidate_status
    manifest["kaggle_result_reason"] = reason
    manifest["confirmed_kaggle_public_score"] = actual_score
    for row in manifest.get("tasks", []):
        row["candidate_status"] = candidate_status
        row["kaggle_result_reason"] = reason
    atomic_write(path, json.dumps(manifest, indent=1))


def packet_probe_shas(items):
    return {p.name: sha256_of(resolve(pkt["onnx"])) for p, pkt, _ in items}

def pending_probe_specs():
    return sorted(
        p for p in PROBE_DIR.glob("*.json")
        if not p.name.endswith(".tmp") and not p.name.endswith(".failed.json") and not p.name.endswith(".home_zip_block.json")
        and not p.name.endswith(".snapshot_meta.json")   # swap snapshot metadata (tag.{baseline,candidate}.snapshot_meta.json), NOT a collectable probe spec
        and not p.name.startswith("_")                   # bookkeeping (e.g. _baseline_cache.json), NOT a collectable probe spec
    )

# --- Kaggle (standard `submissions --csv` cols: fileName,date,description,status,publicScore,privateScore)
def kaggle_rows():
    # --page-size 200 (CLI default is 20/50): the COW cached-baseline row must stay findable after a
    # heavy probe night, or every collect reports "pending" forever (2026-07-08 baseline-paged-out stall)
    o = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "--csv", "--page-size", "200"],
                       capture_output=True).stdout.decode("utf-8", "replace")
    lines = o.splitlines()                                           # kaggle v2.0.2 prints a version-warning on STDOUT before
    for i, ln in enumerate(lines):                                   # the CSV header; skip any preamble so DictReader sees the
        if ln.startswith("fileName,"): o = "\n".join(lines[i:]); break   # real header (no-op when the warning is absent)
    return list(csv.DictReader(io.StringIO(o)))
def score_by(rows, substr):
    for r in rows:
        if substr in (r.get("description") or "") and "complete" in (r.get("status") or "").lower():
            try: return float(r["publicScore"])
            except Exception: return None
    return None
def submits_today(rows): return sum(1 for r in rows if (r.get("date") or "")[:10] == today())
def observed_submit_count():
    try: return submits_today(kaggle_rows())
    except Exception: return None                                  # can't reach Kaggle -> don't hard-block here

def _is_false(value):
    if isinstance(value, bool):
        return value is False
    if value is None:
        return False
    return str(value).strip().lower() in {"false", "no", "0", "none", "not_required", "not-required"}


def _safety_payload(pkt):
    raw = pkt.get("safety") or pkt.get("ng_safety") or pkt.get("safety_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


def _flatten_values(obj):
    if isinstance(obj, dict):
        for key, val in obj.items():
            yield str(key)
            yield from _flatten_values(val)
    elif isinstance(obj, list):
        for val in obj:
            yield from _flatten_values(val)
    elif obj is not None:
        yield str(obj)


def _local_gold_reason(pkt, onnx_sha):
    """Return (ok, reason) for the no-Kaggle path.

    This is intentionally narrow: only explicit Path-A byte-equal/inherited candidates with clean
    ng_safety evidence bypass the stack probe. Path-B, fitted, x86-required, or incomplete packets
    keep using the existing probe-stack gate.
    """
    if str(pkt.get("status") or pkt.get("safety_status") or "").strip().upper() != "LOCAL_GOLD":
        return False, "status is not LOCAL_GOLD"

    cls = str(pkt.get("candidate_class") or "").strip().lower()
    allowed = {"path_a_inherited", "path_a_byte_equal", "byte_equal_inherited", "parent_byte_equal", "ab_full_byte_equal"}
    if cls not in allowed:
        return False, f"candidate_class {cls or '(missing)'} is not a no-probe Path-A class"

    if not _is_false(pkt.get("kaggle_probe_required")):
        return False, "kaggle_probe_required is not false"
    if not _is_false(pkt.get("x86_required")):
        return False, "x86_required is not false"

    cand_sha = norm_sha(pkt.get("candidate_sha256") or pkt.get("sha256"))
    if cand_sha != onnx_sha:
        return False, "candidate_sha256/sha256 does not match ONNX bytes"
    if not norm_sha(pkt.get("parent_sha256")):
        return False, "parent_sha256 is missing"
    if not pkt.get("parent_source"):
        return False, "parent_source is missing"
    if not pkt.get("proof_commands_run") or not pkt.get("proof_results"):
        return False, "proof_commands_run/proof_results missing"

    safety = _safety_payload(pkt)
    if not safety:
        return False, "safety/ng_safety JSON missing or unparseable"
    if str(safety.get("verdict") or "").upper() != "PASS":
        return False, f"ng_safety verdict is {safety.get('verdict')!r}, not PASS"
    model = safety.get("model") if isinstance(safety.get("model"), dict) else {}
    if norm_sha(model.get("sha256")) != onnx_sha:
        return False, "ng_safety model sha does not match ONNX bytes"
    summary = safety.get("summary") if isinstance(safety.get("summary"), dict) else {}
    if int(summary.get("n_failures") or 0) != 0 or int(summary.get("n_warnings") or 0) != 0:
        return False, "ng_safety has failures or warnings"

    proof_blob = " ".join(_flatten_values(pkt.get("proof_results"))) + " " + " ".join(_flatten_values(pkt.get("gates")))
    proof_blob = proof_blob.lower()
    positive_byte_equal = any(
        token in proof_blob
        for token in ("byte_equal=true", "byte_equal: true", "byte_equal true",
                      "byte-equal=true", "byte-equal: true", "byte-equal true")
    )
    if "ab" not in proof_blob or not positive_byte_equal:
        return False, "affirmative AB byte_equal=True proof is missing"
    return True, "LOCAL_GOLD Path-A byte-equal safety packet"


def classify(pkt, man, dp=None):
    if dp is None: dp = load_dont_probe()
    if "_error" in pkt: return "CONFLICT", f"unreadable ({pkt['_error']})"
    miss = [k for k in REQUIRED if k not in pkt]
    if miss: return "HELD", f"missing {miss}"
    inbox_lane = str(pkt.get("_inbox_lane") or "").capitalize()
    source_lane = str(pkt.get("source_lane") or "").capitalize()
    if source_lane not in LANES: return "CONFLICT", f"bad source_lane {pkt.get('source_lane')!r}"
    if inbox_lane and source_lane != inbox_lane: return "CONFLICT", f"source_lane {source_lane} != inbox {inbox_lane}"
    task = norm_task(pkt.get("task"))
    onnx = resolve(pkt["onnx"])
    if not onnx.exists(): return "CONFLICT", f"onnx missing: {pkt['onnx']}"
    onnx_sha = sha256_of(onnx)
    if onnx_sha != norm_sha(pkt["sha256"]): return "CONFLICT", "sha mismatch"
    dr = dont_probe_reason(pkt, dp)
    if dr: return "CONFLICT", f"DONT_PROBE: {dr}"
    best = man.get(task)
    if best is not None and pkt["cost"] >= best * 0.999: return "CONFLICT", f"team-best {int(best)} <= {pkt['cost']} (no gain)"
    if "probe_ok" not in pkt:
        local_ok, local_note = _local_gold_reason(pkt, onnx_sha)
        if local_ok:
            pkt.setdefault("candidate_status", "local_gold")
            pkt.setdefault("probe_sha", onnx_sha)
            pkt.setdefault("probe_kind", "local_gold")
            pkt.setdefault("kaggle_result_reason", local_note)
            return "READY", local_note
        return "HELD", f"needs STACK probe (run probe-stack); no-probe local gold unavailable: {local_note}"
    if not pkt["probe_ok"]:
        status = pkt.get("candidate_status") or "probe_mismatch"
        return "CONFLICT", f"{status} (delta {pkt.get('probe_delta')}; {pkt.get('kaggle_result_reason', '')})"
    probe_sha = norm_sha(pkt.get("probe_sha"))
    if not probe_sha:
        return "HELD", "stale probe (missing probe_sha; re-probe these bytes)"
    if probe_sha != onnx_sha:
        return "HELD", f"stale probe (probe_sha {probe_sha[:12]} != onnx sha {onnx_sha[:12]}; re-probe these bytes)"
    # STACK-ONLY GATE (2026-07-03): a 0-base ISOLATED probe measures ABSOLUTE points vs a zero floor and gives
    # false CONFIRMs -- candidates that hidden-fail the real overlay at -17..-36 pts. Only a per-task real-stack
    # overlay (probe-stack / probe-theory -> _fire single) may certify staging. Legacy packets lacking probe_kind
    # are treated as un-stacked and HELD. Ref: _proposals/isolated_probe_false_confirm_stack_only_gate_2026-07-03.md
    probe_kind = str(pkt.get("probe_kind") or "").strip().lower()
    # STACK certifies via the real Kaggle overlay. LOCAL_TRUTH_N8000 certifies via an on-box
    # `ng_gate truth --n 8000` run (owner directive 2026-07-13): this box is x86 like the grader, so a
    # candidate that produces CORRECT hidden outputs locally cannot hidden-fail its independent stack slot --
    # and it's flake-free + cap-free (validated: task005's local truth stayed correct while its Kaggle probe
    # flaked -18.82). Unlike a 0-base ISOLATED probe (which false-confirms), truth tests real correctness.
    if probe_kind not in ("stack", "local_truth_n8000", "local_truth_n2000"):
        return "HELD", f"isolated/absent probe (probe_kind={probe_kind or 'none'}) cannot stage -- run a per-task STACK overlay or local-truth verify: probe-stack --lane {source_lane.lower()} --task {task}"
    status = str(pkt.get("candidate_status") or "").strip().lower()
    if status != CONFIRMED_STATUS:
        return "CONFLICT", f"status {status or 'blank'} not {CONFIRMED_STATUS}"
    return "READY", None


def ort_load_check(onnx):
    """Stage-time smoke test: verify the ONNX LOADS in onnxruntime and its input/output types are
    supported. LOAD-ONLY by design (fixed 2026-06-30, owner-authorized). The previous version also ran the
    session on an ALL-ZERO synthetic feed, but all-zeros is an INVALID one-hot for these ARC color-channel
    graphs (input [1,10,30,30] float): data-dependent Gather/ScatterND go out of bounds, so the run hard-
    FAILED (often aborting the process) on candidates that are correct on real inputs -- it rejected 100%
    of probe-confirmed candidates, including graphs byte-equal to the live team-best. There is NO single
    synthetic input valid for every data-dependent graph (a non-zero one-hot that fixes one graph breaks
    another), and real-input runtime is already gated upstream by ng_gate (worker) + the Kaggle isolated
    probe (captain) -- cmd_stage only reaches here for probe_ok candidates. So the gate now validates
    loadability + supported input types and does NOT execute the graph (which also makes it abort-safe)."""
    try:
        import onnxruntime as ort
    except Exception as e:
        return False, f"ORT load gate unavailable: {e}"
    supported = {
        "tensor(float)", "tensor(float16)", "tensor(double)", "tensor(bool)",
        "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)",
        "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)",
    }
    try:
        sess = ort.InferenceSession(pathlib.Path(onnx).read_bytes(), providers=["CPUExecutionProvider"])
        for meta in sess.get_inputs():
            if meta.type not in supported:
                return False, f"unsupported ORT input type {meta.type!r} for {meta.name}"
        if not sess.get_outputs():
            return False, "ONNX has no outputs"
        return True, None
    except Exception as e:
        return False, f"ORT load failed: {e}"

# --- probe (the verification gate) ------------------------------------------------------
def _fire(lane, items, single):
    """items: [(packet_path, pkt, team_best)]; snapshot home zip, fire baseline+candidates."""
    # Spread dedup: a probe+fallback spread is multiple packets for ONE task. Keep the LAST per task — that
    # variant is the one swap_and_submit lands in the home zip (last write wins), so tasks/base/cand/predicted
    # must all describe it. Without this the predicted double-counts the spread and the batch reads falsely "short".
    if len(items) > 1:
        _dd = {}
        for _it in items:
            _dd[norm_task(_it[1]["task"])] = _it
        items = list(_dd.values())
    pending = pending_probe_specs()
    if pending:
        names = ", ".join(p.name for p in pending[:5])
        more = "" if len(pending) <= 5 else f" (+{len(pending) - 5} more)"
        print(f"PROBE BLOCK: uncollected probe spec(s): {names}{more}. Run probe-collect before firing another probe.")
        return
    observed = observed_submit_count()
    tasks = [norm_task(pkt["task"]) for _, pkt, _ in items]
    tag = f"probe-{lane}-{'1' if single else 'b'}-{now_stamp()}"
    base = [str(TEAMBEST / f"{t}.onnx") for t in tasks]
    cand = [str(resolve(pkt["onnx"])) for _, pkt, _ in items]
    probe_shas = packet_probe_shas(items)
    predicted = round(sum(points_of(pkt["cost"]) - points_of(b) for _, pkt, b in items if b), 3)
    usage = "unknown" if observed is None else str(observed)
    print(f"PROBE {tag}: {tasks}  expected delta IF correct +{predicted}  "
          f"(confirmed Kaggle/public delta unknown; observed submits today {usage}; diagnostic only)")
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    prehome = PROBE_DIR / f"{tag}.prehome.zip"
    shutil.copy2(HOME_ZIP, prehome)                                  # restore point: the scratch vehicle's pre-probe state
    base_msg = f"{tag} baseline team-best"
    cand_msg = f"{tag} candidates"
    base_meta = write_swap_snapshot_meta(tag, "baseline", items)
    cand_meta = write_swap_snapshot_meta(tag, "candidate", items)
    base_proc = run(SWAP + base + ["--submit", "-m", base_msg,
                                   "--snapshot-tag", f"{tag}-baseline",
                                   "--snapshot-meta", str(base_meta)])      # S_base
    if base_proc.returncode != 0:
        print(f"BASELINE SUBMIT FAILED rc={base_proc.returncode}; not firing candidate submit.")
        print(tail(base_proc.stdout))
        print(tail(base_proc.stderr))
        restore_home_zip(prehome, tag)
        record_failed_probe(tag, lane, tasks, predicted, "baseline", base_proc, prehome)
        return
    cand_proc = run(SWAP + cand + ["--submit", "-m", cand_msg,
                                   "--snapshot-tag", f"{tag}-candidate",
                                   "--snapshot-meta", str(cand_meta)])      # S_batch
    if cand_proc.returncode != 0:
        print(f"CANDIDATE SUBMIT FAILED rc={cand_proc.returncode}; probe is incomplete.")
        print(tail(cand_proc.stdout))
        print(tail(cand_proc.stderr))
        restore_home_zip(prehome, tag)
        record_failed_probe(tag, lane, tasks, predicted, "candidate", cand_proc, prehome)
        return
    atomic_write(PROBE_DIR / f"{tag}.json", json.dumps(
        {"tag": tag, "lane": lane, "tasks": tasks, "single": single, "predicted": predicted,
         "expected_delta_if_correct": predicted,
         "score_kind": "local_delta_if_correct",
         "baseline_message": base_msg, "candidate_message": cand_msg,
         "restore_source": str(prehome), "packet_probe_shas": probe_shas,
         "packets": [p.name for p, _, _ in items]}, indent=1))
    print(f"   fired ({tag}); collect from a main loop or explicit second turn: captain_ctl.py probe-collect --apply")

# --- COW cached-baseline stack probe (2026-07-04) ------------------------------------------
# The stale scratch home zip drifted from the live leaderboard stack in 291/401 slots, so the old
# per-probe "team-best baseline submit" in _fire could not be a single cached number. This path drops
# the home zip from the baseline entirely: the baseline is the LIVE, READ-ONLY leaderboard stack
# (Backend/submission.zip == the team-best assembly, reached read-only per the Team_Best_Onnx junction
# convention). It is submitted ONCE and its public score cached by the stack sha; each candidate is a
# COPY-ON-WRITE scratch zip (exactly ONE onnx swapped) submitted separately. So a per-task stack probe
# costs ONE submit (was two), the live stack is NEVER mutated (read-only, no swap_and_submit, no home
# restore), and probe_kind stays "stack" (batch-size-1 -> no task hidden behind a batch total, so
# classify() still lets it stage). Delta == the old stale-home delta under the additivity the gate
# already assumes; any violation still trips candidate_status_from_kaggle -> suspicious/hidden_failed.
BASELINE_ZIP   = SUBMISSION                                          # read-only live leaderboard stack (== team-best assembly)
BASELINE_CACHE = PROBE_DIR / "_baseline_cache.json"
LEADERBOARD_STATE = SUBMISSION.parent / "current_leaderboard_state.json"  # owner-maintained confirmed team-best score

def confirmed_baseline_score(base_sha):
    """Owner directive 2026-07-11: the probe baseline SCORE must come from an already-scored, CONFIRMED
    team-best (a stable value), NOT a fresh submit whose per-run Kaggle score can hit the ~-18.85 grader-flake
    tier (that inflated task169's stack delta to +19.04 against a real +0.17 gain). Source: the owner-maintained
    current_leaderboard_state.json beside submission.zip; trusted ONLY when its submission_sha256 matches the
    live stack we probe (else it is stale -> None, and ensure_cached_baseline falls back to a one-time submit)."""
    try:
        d = json.loads(LEADERBOARD_STATE.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    sha = str(d.get("submission_sha256") or "")
    if base_sha and sha[:16] == base_sha[:16]:
        try: return float(d.get("current_public_score"))
        except Exception: return None
    return None

def _load_baseline_cache():
    try: return json.loads(BASELINE_CACHE.read_text(encoding="utf-8"))
    except Exception: return {}
def _baseline_msg(base_sha): return f"neurogolf cached-baseline live-stack {base_sha[:16]}"

def ensure_cached_baseline(apply):
    """Return (message, base_sha, newly_fired). Owner directive 2026-07-11: prefer the CONFIRMED team-best
    score (confirmed_baseline_score) as the baseline -- a stable, already-scored value immune to the per-submit
    grader flake -- cached under the stack-sha with a 'score' field that collect reads directly. A fresh baseline
    submit is only the FALLBACK, used when no confirmed score is available for the current stack. Reads
    BASELINE_ZIP; never writes it."""
    base_sha = sha256_of(BASELINE_ZIP)
    msg = _baseline_msg(base_sha)
    cache = _load_baseline_cache()
    if cache.get(base_sha, {}).get("score") is not None: return msg, base_sha, False
    conf = confirmed_baseline_score(base_sha)
    if conf is not None:
        cache[base_sha] = {"message": msg, "score": conf, "source": "confirmed_team_best",
                           "submitted_utc": now_stamp(), "zip": str(BASELINE_ZIP)}
        if apply: atomic_write(BASELINE_CACHE, json.dumps(cache, indent=1))
        print(f"   baseline from CONFIRMED team-best {conf:.2f} (no submit; stack sha {base_sha[:12]})")
        return msg, base_sha, False
    if base_sha in cache: return msg, base_sha, False
    if not apply:         return msg, base_sha, True
    proc = run(["kaggle", "competitions", "submit", "-c", COMP, "-f", str(BASELINE_ZIP), "-m", msg])
    if proc.returncode != 0:
        print(f"BASELINE SUBMIT FAILED rc={proc.returncode}; {tail(proc.stdout)} {tail(proc.stderr)}")
        return None, base_sha, True
    cache[base_sha] = {"message": msg, "submitted_utc": now_stamp(), "zip": str(BASELINE_ZIP)}
    atomic_write(BASELINE_CACHE, json.dumps(cache, indent=1))
    print(f"   baseline submitted ONCE (cached by live-stack sha {base_sha[:12]}); reused by all candidate probes")
    return msg, base_sha, True

def _build_cow_candidate_zip(tag, task, cand_bytes):
    """Copy-on-write: read the read-only live stack, write a scratch submission.zip with exactly ONE onnx swapped.
    Never writes through the junction / into the live Backend folder."""
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    cow_dir = PROBE_DIR / tag; cow_dir.mkdir(exist_ok=True)
    cow_zip = cow_dir / "submission.zip"                            # MUST be named submission.zip for Kaggle
    want = norm_task(task); swapped = False
    with zipfile.ZipFile(BASELINE_ZIP) as zin, zipfile.ZipFile(cow_zip, "w", zipfile.ZIP_DEFLATED) as zout:
        for nm in zin.namelist():
            if nm.endswith(".onnx") and norm_task(pathlib.Path(nm).stem) == want:
                zout.writestr(nm, cand_bytes); swapped = True
            else:
                zout.writestr(nm, zin.read(nm))
    if not swapped:
        shutil.rmtree(cow_dir, ignore_errors=True)
        raise RuntimeError(f"{want} not present in live stack {BASELINE_ZIP.name}")
    return cow_zip

def _fire_stack_cow(lane, items):
    """Per-task STACK gate via copy-on-write vs the read-only live leaderboard stack + a cached baseline.
    ONE candidate submit per task; baseline once. Certifies per-task (probe_kind=stack, single) at collect."""
    if len(items) > 1:                                              # single per-task gate: dedup, keep last variant
        _dd = {}
        for _it in items: _dd[norm_task(_it[1]["task"])] = _it
        items = [list(_dd.values())[-1]]
    pending = pending_probe_specs()
    if pending:
        print(f"PROBE BLOCK: uncollected probe spec(s): {', '.join(p.name for p in pending[:5])}. Run probe-collect first.")
        return
    p, pkt, best = items[0]; task = norm_task(pkt["task"])
    if best is None: print(f"(no team-best for {task}; nothing to beat)"); return
    base_msg, base_sha, _ = ensure_cached_baseline(apply=True)
    if base_msg is None: print("   baseline unavailable; not firing candidate"); return
    predicted = round(points_of(pkt["cost"]) - points_of(best), 3)
    tag = f"probe-{lane}-1-{now_stamp()}"; cand_msg = f"{tag} candidates"
    try:
        cow_zip = _build_cow_candidate_zip(tag, task, resolve(pkt["onnx"]).read_bytes())
    except Exception as e:
        print(f"COW BUILD FAILED for {task}: {e}"); return
    observed = observed_submit_count(); usage = "unknown" if observed is None else str(observed)
    print(f"PROBE {tag}: [{task}]  expected delta IF correct +{predicted}  "
          f"(cached baseline reused; observed submits today {usage}; ONE candidate submit)")
    proc = run(["kaggle", "competitions", "submit", "-c", COMP, "-f", str(cow_zip), "-m", cand_msg])
    if proc.returncode != 0:
        print(f"CANDIDATE SUBMIT FAILED rc={proc.returncode}; {tail(proc.stdout)} {tail(proc.stderr)}")
        shutil.rmtree(PROBE_DIR / tag, ignore_errors=True)
        record_failed_probe(tag, lane, [task], predicted, "candidate", proc, None); return
    atomic_write(PROBE_DIR / f"{tag}.json", json.dumps(
        {"tag": tag, "lane": lane, "tasks": [task], "single": True, "cow": True, "probe_style": "stack_cow",
         "predicted": predicted, "expected_delta_if_correct": predicted, "score_kind": "local_delta_if_correct",
         "baseline_message": base_msg, "candidate_message": cand_msg, "baseline_sha": base_sha,
         "baseline_score": _load_baseline_cache().get(base_sha, {}).get("score"),  # confirmed team-best (flake-immune); None -> collect reads the submitted baseline row
         "cow_zip": str(cow_zip), "packet_probe_shas": packet_probe_shas(items),
         "packets": [p.name for p, _, _ in items]}, indent=1))
    print(f"   fired ({tag}); collect with: captain_ctl.py probe-collect --apply")

def stack_cow_dryrun(lane, items):
    """No-submit validation: report baseline cache status + build (then discard) the COW candidate zip so the
    swap path is exercised without touching Kaggle or the live stack."""
    if not items: print("(no target)"); return
    p, pkt, best = items[-1]; task = norm_task(pkt["task"])
    base_msg, base_sha, need = ensure_cached_baseline(apply=False)
    predicted = round(points_of(pkt["cost"]) - points_of(best), 3) if best else None
    print(f"   DRY-RUN stack-cow {task}: baseline {'MISS -> 1 submit' if need else 'CACHED -> 0 submit'} "
          f"(live-stack sha {base_sha[:12]}); candidate 1 submit; expected delta +{predicted}")
    try:
        z = _build_cow_candidate_zip(f"dryrun-{now_stamp()}", task, resolve(pkt["onnx"]).read_bytes())
        n = len(zipfile.ZipFile(z).namelist()); shutil.rmtree(z.parent, ignore_errors=True)
        print(f"   DRY-RUN COW zip OK ({n} entries, 1 swapped, live stack untouched)")
    except Exception as e:
        print(f"   DRY-RUN COW build FAILED: {e}")

def cmd_probe_batch(args):
    man = load_manifest(); dp = load_dont_probe(); lane = args.lane.capitalize(); pend = []
    for p, pkt in iter_packets(lane):
        if pkt.get("probe") == "theory" or "probe_ok" in pkt: continue
        if [k for k in REQUIRED if k not in pkt]: continue
        if dont_probe_reason(pkt, dp): print(f"DONT_PROBE skip {pkt.get('task')}"); continue
        best = man.get(norm_task(pkt["task"]))
        if best and pkt["cost"] >= best * 0.999: continue
        pend.append((p, pkt, best))
    # Dedup spread packets by task BEFORE the BATCH_N cap so a probe+fallback spread takes ONE slot, not two
    # (keep the last variant per task — matches the _fire dedup + the swap's last-write-wins).
    _seen = {}
    for _it in pend:
        _seen[norm_task(_it[1]["task"])] = _it
    pend = list(_seen.values())[:BATCH_N]
    if not pend: print("(no un-probed batch candidates)"); return
    print(f"would probe {len(pend)}: {[k[1]['task'] for k in pend]}" + ("" if args.apply else " (dry-run)"))
    if args.apply: _fire(lane, pend, single=False)

def cmd_probe_theory(args):
    man = load_manifest(); lane = args.lane.capitalize()
    want = norm_task(args.task)
    hit = [(p, pkt, man.get(norm_task(pkt["task"]))) for p, pkt in iter_packets(lane) if norm_task(pkt.get("task")) == want]
    if not hit: print(f"(no packet for {want} in {lane})"); return
    if dont_probe_reason(hit[0][1], load_dont_probe()): print(f"DONT_PROBE: {want} on the guardrail — refusing"); return
    print(f"would single-probe {want} (COW overlay)" + ("" if args.apply else " (dry-run)"))
    if args.apply: _fire_stack_cow(lane, [hit[-1]])               # cached-baseline COW; last variant (spread-dedup)
    else:          stack_cow_dryrun(lane, [hit[-1]])

def cmd_probe_stack(args):
    """STACK overlay probe = THE staging gate (2026-07-03). Swap ONE candidate into the real home-zip stack,
    submit baseline+candidate, read the per-task MARGINAL delta vs the live team-best. Batch size 1 so no task's
    failure can be masked by a batch total. Only this probe (probe_kind==stack, single) certifies staging;
    probe-isolated is a 0-base smoke test that CANNOT stage. With --task, probes that task; without, probes the
    first un-probed candidate (PROBE BLOCK serializes to one in flight -- collect between calls)."""
    man = load_manifest(); dp = load_dont_probe(); lane = args.lane.capitalize()
    if args.task:
        want = norm_task(args.task)
        items = [(p, pkt, man.get(norm_task(pkt["task"]))) for p, pkt in iter_packets(lane)
                 if norm_task(pkt.get("task")) == want]
        if not items: print(f"(no packet for {want} in {lane})"); return
    else:
        cand = []
        for p, pkt in iter_packets(lane):
            if "probe_ok" in pkt or [k for k in REQUIRED if k not in pkt]: continue
            best = man.get(norm_task(pkt["task"]))
            if best and pkt["cost"] >= best * 0.999: continue
            cand.append((p, pkt, best))
        _seen = {}
        for _it in cand: _seen[norm_task(_it[1]["task"])] = _it
        cand = list(_seen.values())
        if not cand: print("(no un-probed stack candidates)"); return
        print(f"({len(cand)} un-probed; stack-probing one per call)")
        items = cand[:1]
    target = items[-1]                                             # last variant (spread-dedup + swap last-write-wins)
    if dont_probe_reason(target[1], dp): print(f"DONT_PROBE: {norm_task(target[1]['task'])} -- refusing"); return
    print(f"would STACK-probe (per-task COW overlay, batch size 1) {norm_task(target[1]['task'])}" + ("" if args.apply else " (dry-run)"))
    if args.apply: _fire_stack_cow(lane, [target])
    else:          stack_cow_dryrun(lane, [target])

def _fire_isolated(lane, items):
    """ISOLATED probe: build a 0-base zip (every entry = the fail-onnx) with the candidates swapped in, and submit
    it ONCE. publicScore == the candidates' ABSOLUTE points, so NO baseline submit is ever needed (the fail-base is
    known to score 0). probe-collect then verifies grade-correctness (score == predicted points) AND the bankable
    gain vs the manifest team-best, with no second submit. Does not touch the home zip (builds a separate zip)."""
    if len(items) > 1:                                              # spread dedup: last variant per task wins
        _dd = {}
        for _it in items: _dd[norm_task(_it[1]["task"])] = _it
        items = list(_dd.values())
    pending = pending_probe_specs()
    if pending:
        print(f"PROBE BLOCK: uncollected probe spec(s): {', '.join(p.name for p in pending[:5])}. Run probe-collect first.")
        return
    if not FAIL_ONNX.exists():                                      # auto-generate the 0-base if missing
        print(f"   generating {FAIL_ONNX.name} ...")
        gen = run(["python", str(FAIL_GEN)])
        if gen.returncode != 0 or not FAIL_ONNX.exists():
            print(f"FAIL-ONNX GEN FAILED rc={gen.returncode}; {tail(gen.stderr)}"); return
    fail_bytes = FAIL_ONNX.read_bytes()
    tasks = [norm_task(pkt["task"]) for _, pkt, _ in items]
    cand = {}
    probe_shas = {}
    for p, pkt, _ in items:
        task = norm_task(pkt["task"])
        data = resolve(pkt["onnx"]).read_bytes()
        cand[task] = data
        probe_shas[p.name] = hashlib.sha256(data).hexdigest()
    predicted = round(sum(points_of(pkt["cost"]) for _, pkt, _ in items), 3)        # absolute points the candidates should score
    team_best_pts = round(sum(points_of(b) for _, _, b in items if b), 3)           # known team-best points (NO submit)
    tag = f"iso-{lane}-{now_stamp()}"; cand_msg = f"{tag} isolated"
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    iso_dir = PROBE_DIR / tag; iso_dir.mkdir(exist_ok=True)
    iso_zip = iso_dir / "submission.zip"                            # MUST be named submission.zip for Kaggle
    with zipfile.ZipFile(HOME_ZIP) as zin:
        names = zin.namelist()
        with zipfile.ZipFile(iso_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for nm in names:
                if nm.endswith(".onnx"):
                    z.writestr(nm, cand.get(norm_task(pathlib.Path(nm).stem), fail_bytes))
                else:
                    z.writestr(nm, zin.read(nm))                     # pass any non-onnx entry through unchanged
    print(f"ISOLATED PROBE {tag}: {tasks}  local points IF correct {predicted} (team-best {team_best_pts}; "
          f"gain if Kaggle confirms +{round(predicted - team_best_pts, 3)}; ONE submit, no baseline)")
    snapshot_dir = write_direct_probe_snapshot(tag, iso_zip, items, predicted, cand_msg)
    print(f"   snapshot saved: {snapshot_dir / 'snapshot_manifest.json'}")
    proc = run(["kaggle", "competitions", "submit", "-c", COMP, "-f", str(iso_zip), "-m", cand_msg])
    if proc.returncode != 0:
        print(f"ISOLATED SUBMIT FAILED rc={proc.returncode}; {tail(proc.stdout)} {tail(proc.stderr)}")
        record_failed_probe(tag, lane, tasks, predicted, "isolated", proc, None); return
    atomic_write(PROBE_DIR / f"{tag}.json", json.dumps(
        {"tag": tag, "lane": lane, "tasks": tasks, "isolated": True, "predicted": predicted,
         "expected_score_if_correct": predicted,
         "score_kind": "local_points_if_correct",
         "team_best_points": team_best_pts, "candidate_message": cand_msg,
         "snapshot_manifest": str(snapshot_dir / "snapshot_manifest.json"),
         "snapshot_zip": str(snapshot_dir / "submission.zip"),
         "packet_probe_shas": probe_shas,
         "packets": [p.name for p, _, _ in items]}, indent=1))
    print(f"   fired ({tag}); collect with: captain_ctl.py probe-collect --apply")

def cmd_probe_isolated(args):
    man = load_manifest(); dp = load_dont_probe(); lane = args.lane.capitalize()
    if args.task:
        want = norm_task(args.task)
        items = [(p, pkt, man.get(norm_task(pkt["task"]))) for p, pkt in iter_packets(lane)
                 if norm_task(pkt.get("task")) == want]
        if not items: print(f"(no packet for {want} in {lane})"); return
        if dont_probe_reason(items[0][1], dp): print(f"DONT_PROBE: {want} — refusing"); return
    else:
        items = []
        for p, pkt in iter_packets(lane):
            if "probe_ok" in pkt or [k for k in REQUIRED if k not in pkt]: continue
            if dont_probe_reason(pkt, dp): print(f"DONT_PROBE skip {pkt.get('task')}"); continue
            best = man.get(norm_task(pkt["task"]))
            if best and pkt["cost"] >= best * 0.999: continue
            items.append((p, pkt, best))
        _seen = {}
        for _it in items: _seen[norm_task(_it[1]["task"])] = _it
        items = list(_seen.values())[:BATCH_N]
        if not items: print("(no un-probed isolated candidates)"); return
    print(f"would isolated-probe {[norm_task(i[1]['task']) for i in items]}" + ("" if args.apply else " (dry-run)"))
    if args.apply: _fire_isolated(lane, items)

def cmd_probe_collect(args):
    rows = kaggle_rows()
    for spec_p in pending_probe_specs():
        spec = json.loads(spec_p.read_text(encoding="utf-8"))
        if spec.get("cow"):                                         # COW cached-baseline stack probe (2026-07-04): live stack never mutated -> no home restore
            sb = spec.get("baseline_score")                         # owner 2026-07-11: confirmed team-best baseline (flake-immune)
            if sb is None: sb = score_by(rows, spec["baseline_message"])   # fallback: the once-submitted baseline row
            sc = score_by(rows, spec["candidate_message"])
            if sb is None or sc is None: print(f"{spec['tag']}: pending"); continue
            delta = round(sc - sb, 3)
            candidate_status, reason = candidate_status_from_kaggle(spec["predicted"], delta)
            ok = candidate_status == "confirmed_public"
            print(f"{spec['tag']}: Kaggle public delta {delta:+.2f} vs local-if-correct {spec['predicted']:+.2f} "
                  f"(baseline {sb:.2f}) -> {'CONFIRMED' if ok else candidate_status.upper() + ' -> reject/re-probe'}")
            if args.apply:
                for pname in spec["packets"]:
                    pp = lane_dir(CAP_DIR, spec["lane"]) / pname
                    if pp.exists():
                        pkt = json.loads(pp.read_text(encoding="utf-8"))
                        pkt["probe_delta"] = delta; pkt["probe_ok"] = ok
                        pkt["candidate_status"] = candidate_status; pkt["kaggle_result_reason"] = reason
                        pkt["probe_kind"] = "stack"                 # COW single = per-task real-stack overlay -> stageable (classify requires =="stack")
                        pkt["probe_style"] = "stack_cow"; pkt["probe_baseline"] = sb
                        probe_sha = (spec.get("packet_probe_shas") or {}).get(pname)
                        if probe_sha:
                            pkt["probe_sha"] = norm_sha(probe_sha); pkt["probe_checked_utc"] = now_stamp()
                        atomic_write(pp, json.dumps(pkt, indent=1))
                (PROBE_DIR / "_done").mkdir(parents=True, exist_ok=True)
                os.replace(spec_p, PROBE_DIR / "_done" / spec_p.name)
                shutil.rmtree(PROBE_DIR / spec["tag"], ignore_errors=True)   # drop the COW scratch zip; live stack untouched
                print(f"   collected {spec['tag']} (COW; live stack untouched, no home-zip restore needed)")
            continue
        if spec.get("isolated"):                                    # ISOLATED probe: score IS the absolute points (base=0, no baseline submit)
            score = score_by(rows, spec["candidate_message"])
            if score is None: print(f"{spec['tag']}: pending"); continue
            candidate_status, reason = candidate_status_from_kaggle(spec["predicted"], score)
            ok = candidate_status == "confirmed_public"
            if ok: candidate_status = "isolated_smoke_ok"   # a 0-base isolated pass is a SMOKE test, NOT stage-ready; classify requires a probe-stack overlay
            gain = round(score - spec.get("team_best_points", 0), 3)
            print(f"{spec['tag']}: Kaggle public score {score:.2f} vs local-if-correct {spec['predicted']:.2f}  ->  "
                  f"{'ISOLATED-SMOKE-OK (needs probe-stack)' if ok else candidate_status.upper()}; gain vs team-best +{gain}")
            if args.apply:
                mark_probe_snapshot(spec, candidate_status, reason, score)
                for pname in spec["packets"]:
                    pp = lane_dir(CAP_DIR, spec["lane"]) / pname
                    if pp.exists():
                        pkt = json.loads(pp.read_text(encoding="utf-8"))
                        pkt["probe_ok"] = ok; pkt["probe_delta"] = gain; pkt["probe_isolated_score"] = score
                        pkt["candidate_status"] = candidate_status
                        pkt["probe_kind"] = "isolated"      # smoke test only (absolute vs 0-base); classify requires probe_kind==stack to stage
                        pkt["kaggle_result_reason"] = reason
                        probe_sha = (spec.get("packet_probe_shas") or {}).get(pname)
                        if probe_sha:
                            pkt["probe_sha"] = norm_sha(probe_sha)
                            pkt["probe_checked_utc"] = now_stamp()
                        atomic_write(pp, json.dumps(pkt, indent=1))
                (PROBE_DIR / "_done").mkdir(parents=True, exist_ok=True)
                os.replace(spec_p, PROBE_DIR / "_done" / spec_p.name)
                shutil.rmtree(PROBE_DIR / spec["tag"], ignore_errors=True)   # drop the iso submission.zip dir
            continue
        sb, sc = score_by(rows, f"{spec['tag']} baseline"), score_by(rows, f"{spec['tag']} candidates")
        if sb is None or sc is None: print(f"{spec['tag']}: pending"); continue
        delta = round(sc - sb, 3)
        candidate_status, reason = candidate_status_from_kaggle(spec["predicted"], delta)
        ok = candidate_status == "confirmed_public"
        print(f"{spec['tag']}: Kaggle public delta {delta:+.2f} vs local-if-correct {spec['predicted']:+.2f}  ->  "
              f"{'CONFIRMED' if ok else candidate_status.upper() + ' -> isolate per-task'}")
        if args.apply:
            prehome = PROBE_DIR / f"{spec['tag']}.prehome.zip"      # RESTORE the scratch vehicle before marking packets
            if not restore_home_zip(prehome, spec["tag"]):
                print(f"   BLOCKED: {spec['tag']} left pending; fix home zip before packet updates/staging")
                continue
            for pname in spec["packets"]:
                pp = lane_dir(CAP_DIR, spec["lane"]) / pname
                if pp.exists():
                    pkt = json.loads(pp.read_text(encoding="utf-8")); pkt["probe_delta"] = delta; pkt["probe_ok"] = ok
                    pkt["candidate_status"] = candidate_status; pkt["kaggle_result_reason"] = reason
                    pkt["probe_kind"] = "stack" if spec.get("single") else "stack_batch"   # only a batch-size-1 overlay certifies per-task (a batch total can mask a bad task)
                    probe_sha = (spec.get("packet_probe_shas") or {}).get(pname)
                    if probe_sha:
                        pkt["probe_sha"] = norm_sha(probe_sha)
                        pkt["probe_checked_utc"] = now_stamp()
                    atomic_write(pp, json.dumps(pkt, indent=1))
            (PROBE_DIR / "_done").mkdir(parents=True, exist_ok=True); os.replace(spec_p, PROBE_DIR / "_done" / spec_p.name)
            prehome.unlink()                                         # archive the spec FIRST; a replace failure must not delete the restore snapshot
            for _m in PROBE_DIR.glob(f"{spec['tag']}.*.snapshot_meta.json"): _m.unlink()   # drop swap snapshot-meta so they don't linger as phantom 'pending' specs
            print(f"   home zip integrity verified for {spec['tag']}")

def cmd_probe_status(args):
    """Read-only: is the single shared probe resource free, and if not, who owns the pending probe? (proposal
    2026-06-19_submit_captain_probe_status_command). Lets a non-Captain (Supervisor/Orchestrator) check before
    firing an allowed probe, without the side effects of attempting one."""
    pending = pending_probe_specs()
    failed  = sorted(PROBE_DIR.glob("*.failed.json"))
    blocks  = sorted(PROBE_DIR.glob("*.home_zip_block.json"))
    def load(p):
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return {"_unreadable": p.name}
    if args.json:
        print(json.dumps({"clear": not (pending or failed or blocks),
                          "pending": [load(p) for p in pending], "failed": [load(p) for p in failed],
                          "home_zip_blocks": [load(p) for p in blocks]}, indent=1))
        return 0
    if not (pending or failed or blocks):
        print("probe-status: CLEAR -- no pending/failed/blocked probes; safe to fire a new probe"); return 0
    if blocks:
        print(f"HOME-ZIP BLOCK ({len(blocks)}): staging + new probes are BLOCKED until resolved:")
        for p in blocks: print(f"  {p.name}  -> fix the home zip before probing/staging")
    if pending:
        print(f"PENDING ({len(pending)}): a probe is IN FLIGHT -- do NOT fire another; collect it first:")
        for p in pending:
            s = load(p)
            print(f"  {s.get('tag', p.stem)}  lane={s.get('lane','?')}  tasks={s.get('tasks')}  "
                  f"isolated={s.get('isolated', False)}  packets={s.get('packets')}")
        print("  next: python Agenters/Submit_Captain/captain_ctl.py probe-collect --apply")
    if failed:
        print(f"FAILED ({len(failed)}): inspect before retrying (home zip should already be restored):")
        for p in failed:
            s = load(p); print(f"  {p.name}  failed_stage={s.get('failed_stage')}  rc={s.get('returncode')}")
    return 0

# --- ledger writeback (close the L1 drift: the captain ships/rejects but never stamped the run ledger) ----------
def writeback_captain_status(task, sha, new_status, allow_task_fallback=True):
    """Best-effort: stamp captain_status in the originating Orchestrator run's final_report.csv so the wave can pass
    run_audit. Fixes the L1 drift where the captain ships/rejects but never writes back, leaving 'positive-delta +
    held' rows that block closure though the work shipped. The packet carries no run-id, so match by (task, sha256)
    case-insensitively; when a row has no sha, fall back to task-only but ONLY for rows run_audit treats as
    non-terminal (STUCK_CS), so an already-resolved row is never clobbered (reject disables the fallback entirely).
    Touches ONLY the captain_status cell - status/staged_path/claims.csv stay the Orchestrator's. Optimistic-
    concurrency guarded (skips a ledger that changes mid-read) and NEVER raises: a writeback failure must not block
    staging. Returns ['<run_id>:<task>', ...] for the caller to print."""
    updated = []
    task = norm_task(task); sha = norm_sha(sha)
    try:
        report_paths = sorted(RUNS_DIR.glob("*/final_report.csv"))
    except Exception:
        return updated
    for fr in report_paths:
        try:
            mtime_before = fr.stat().st_mtime
            with fr.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f); fieldnames = reader.fieldnames; rows = list(reader)
            if not fieldnames or "captain_status" not in fieldnames or "task" not in fieldnames: continue
            if any(None in r for r in rows): continue           # a row has extra columns -> don't risk a lossy rewrite
            local = []
            for r in rows:
                if norm_task(r.get("task")) != task: continue
                row_sha = norm_sha(r.get("sha256")); cs = (r.get("captain_status") or "").strip().lower()
                if row_sha and sha:        is_match = (row_sha == sha)
                elif allow_task_fallback:  is_match = cs in STUCK_CS
                else:                      is_match = False
                if is_match and cs != new_status:
                    r["captain_status"] = new_status; local.append(f"{fr.parent.name}:{task}")
            if not local: continue
            if fr.stat().st_mtime != mtime_before:               # the Orchestrator wrote the ledger while we read it
                print(f"   ledger writeback: {fr.parent.name} busy (changed mid-read); skipped - re-run audit after run close")
                continue
            sio = io.StringIO()
            w = csv.DictWriter(sio, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
            atomic_write(fr, sio.getvalue())
            updated.extend(local)
        except Exception:
            continue                                             # one unreadable ledger must never break staging
    return updated

# --- drain / stage / reconcile ----------------------------------------------------------
def cmd_digest(args):
    man = load_manifest(); lanes = LANES if args.lane == "all" else (args.lane.capitalize(),)
    out = {"READY": [], "HELD": [], "CONFLICT": []}
    for lane in lanes:
        for p, pkt in iter_packets(lane):
            task = norm_task(pkt.get("task"))
            state, note = classify(pkt, man); best = man.get(task)
            ev = round(math.log(best / pkt["cost"]), 2) if (state == "READY" and best) else None
            out[state].append({"task": task, "lane": lane, "cost": pkt.get("cost"), "ev": ev, "note": note, "age": packet_age_str(p)})
    out["READY"].sort(key=lambda r: -(r["ev"] or 0))
    rescue = load_rescue_rows(open_only=True)
    if args.lane != "all": rescue = [r for r in rescue if str(r.get("lane", "")).lower() == args.lane.lower()]
    if args.json: print(json.dumps({**out, "RESCUE": rescue}, indent=1)); return
    if not any(out.values()) and not rescue: print("(inboxes empty -- nothing to do)"); return
    for failed in sorted(PROBE_DIR.glob("*.failed.json")):
        print(f"FAILED_PROBE: {failed.name} -- inspect before retrying; home zip should already be restored")
    print(f"== captain digest (lane: {args.lane}) ==")
    if out["READY"]:
        print("READY (Kaggle/public probe confirmed or LOCAL_GOLD safety-verified - stage these, EV-ranked):")
        for r in out["READY"]:
            print(f"  {r['task']}  cost {r['cost']}  local_points_if_correct {points_of(r['cost'])}  EV +{r['ev']}  [{r['lane']}]")
    for label in ("HELD", "CONFLICT"):
        if out[label]:
            print(f"{label}:")
            for r in out[label]: print(f"  {r['task']}  [{r['lane']}]  (age {r['age']})  -- {r['note']}")
    if rescue:
        print("RESCUE (strict-gain worker packets that lost their reviewer verdict -- RE-VERIFY only, do NOT re-run the worker):")
        for r in rescue:
            miss = ",".join(r.get("checks_missing") or []) or "?"
            print(f"  {norm_task(r.get('task'))}  [{r.get('lane','?')}]  cand {r.get('candidate_cost')} vs live {r.get('live_cost')} (delta {r.get('cost_delta')})  missing: {miss}  -> {r.get('candidate_path','?')}")

def cmd_stage(args):
    man = load_manifest(); lane = args.lane.capitalize(); staged = left = 0
    for p, pkt in iter_packets(lane):
        state, note = classify(pkt, man)
        if state != "READY": left += 1; print(f"SKIP {pkt.get('task')} [{state}] {note or ''}"); continue
        task = norm_task(pkt["task"]); onnx = resolve(pkt["onnx"])
        ok, load_note = ort_load_check(onnx)
        if not ok:
            left += 1; print(f"SKIP {task} [CONFLICT] {load_note}"); continue
        status = str(pkt.get("candidate_status") or pkt.get("status") or "local_gold").strip().lower()
        sidecar = json.dumps({"task": task, "creator": CREATOR, "cost": pkt["cost"],
                              "points": points_of(pkt["cost"]), "points_if_correct": points_of(pkt["cost"]),
                              "expected_score_kind": "local_points_if_correct",
                              "candidate_status": status,
                              "probe_sha": norm_sha(pkt.get("probe_sha") or pkt.get("candidate_sha256") or pkt.get("sha256")),
                              "probe_delta": pkt.get("probe_delta"),
                              "kaggle_result_reason": pkt.get("kaggle_result_reason", ""),
                              "sha256": norm_sha(pkt["sha256"]), "msg": pkt.get("msg", "")}, indent=1)
        print(f"STAGE {task}  cost {pkt['cost']}  [{lane}]" + ("" if args.apply else "  (dry-run)"))
        if args.apply:
            shutil.copyfile(onnx, STAGED / f"{task}.onnx"); atomic_write(STAGED / f"{task}.json", sidecar)
            if pkt.get("lesson"): atomic_write(lane_dir(LIB_DIR, lane) / f"{now_stamp()}-{task}.lesson.md", str(pkt["lesson"]))
            (lane_dir(CAP_DIR, lane) / "_consumed").mkdir(parents=True, exist_ok=True)
            os.replace(p, lane_dir(CAP_DIR, lane) / "_consumed" / p.name)
            for u in writeback_captain_status(task, pkt["sha256"], "staged"):
                print(f"   ledger writeback: {u} captain_status->staged")
        staged += 1
    print(f"-- {'staged' if args.apply else 'would stage'} {staged}, left {left} --" + ("" if args.apply or not staged else "  (--apply to commit)"))

def cmd_reconcile(args):
    man = load_manifest(); issues = 0; n = 0
    for js in sorted(STAGED.glob("*.json")):
        n += 1; sc = json.loads(js.read_text(encoding="utf-8")); onnx = STAGED / f"{sc['task']}.onnx"
        if not onnx.exists(): print(f"MISSING onnx {sc['task']}"); issues += 1; continue
        if sha256_of(onnx) != norm_sha(sc.get("sha256")): print(f"SHA DRIFT {sc['task']}"); issues += 1
        if str(sc.get("candidate_status") or "").lower() in {"hidden_failed", "suspicious"}:
            print(f"BAD_CANDIDATE_STATUS {sc['task']}: {sc.get('candidate_status')} {sc.get('kaggle_result_reason', '')}")
            issues += 1
            if args.apply:
                q = STAGED / "_hidden_failed"; q.mkdir(parents=True, exist_ok=True)
                moved = []
                for src in (onnx, js):
                    if not src.exists():
                        continue
                    dest = q / src.name
                    if dest.exists():
                        dest = q / f"{src.stem}.{now_stamp()}{src.suffix}"
                    shutil.move(str(src), str(dest)); moved.append(dest.name)
                print(f"   quarantined {sc['task']} -> _hidden_failed/ ({', '.join(moved)})")
                continue
        best = man.get(sc["task"])
        if best is not None and sc["cost"] >= best * 1.001: print(f"REGRESSION {sc['task']}: {sc['cost']} >= best {int(best)}"); issues += 1
    print(f"-- reconcile: {issues} issue(s) across {n} staged --")

def cmd_reconcile_live(args):
    """Audit the manifest against the ACTUAL merged submission (Finding 2 / snapshot-vs-live merge gap): read-only.
    manifest <-> Team_Best_Onnx file <-> submission.zip entry (sha), plus un-banked staged gains. Exit 1 on drift."""
    man = load_manifest_full(); issues = unbanked = 0
    zip_shas = {}
    if SUBMISSION.exists():
        with zipfile.ZipFile(SUBMISSION) as z:
            for nm in z.namelist():
                if nm.endswith(".onnx"): zip_shas[norm_task(pathlib.Path(nm).stem)] = hashlib.sha256(z.read(nm)).hexdigest()
        print(f"== reconcile-live: manifest {len(man)} tasks, submission.zip {len(zip_shas)} onnx ==")
    else:
        print(f"== reconcile-live: manifest {len(man)} tasks; NO submission.zip at {SUBMISSION} (file<->zip check skipped) ==")
    for task, (cost, msha) in sorted(man.items()):
        f = TEAMBEST / f"{task}.onnx"
        if not f.exists(): print(f"MANIFEST-NO-FILE {task}: manifest row has no Team_Best_Onnx/{task}.onnx"); issues += 1; continue
        fsha = sha256_of(f)
        if msha and fsha != msha: print(f"MANIFEST<>FILE {task}: manifest {msha[:12]} != file {fsha[:12]}"); issues += 1
        if zip_shas:
            zsha = zip_shas.get(task)
            if zsha is None: print(f"FILE-NOT-IN-ZIP {task}: in Team_Best_Onnx but absent from submission.zip"); issues += 1
            elif zsha != fsha: print(f"FILE<>ZIP {task}: file {fsha[:12]} != submitted {zsha[:12]} (submission stale vs files)"); issues += 1
    for js in sorted(STAGED.glob("*.json")):
        try: sc = json.loads(js.read_text(encoding="utf-8"))
        except Exception: continue
        t = norm_task(sc.get("task", "")); best = man.get(t, (None, None))[0]
        if best is not None and sc.get("cost", best) < best * 0.999:
            gain = points_of(sc["cost"]) - points_of(int(best))
            print(f"UN-BANKED {t}: staged {sc['cost']} < manifest-best {int(best)} (+{gain:.3f} awaiting merge)"); unbanked += 1
    print(f"-- reconcile-live: {issues} drift issue(s), {unbanked} un-banked staged gain(s) across {len(man)} manifest tasks --")
    return 1 if issues else 0

def known_hidden_failed_shas():
    """Full sha256 -> location for every candidate the overlay has quarantined: Staged_Boladi/_hidden_failed/**
    plus _shipped/ entries whose sidecar is hidden_failed/suspicious or whose filename marks a prev hidden failure.
    Keeps reconcile-banked (a cost-only audit) from presenting a KNOWN failure as recoverable gold."""
    out = {}
    hf = STAGED / "_hidden_failed"
    if hf.exists():
        for f in hf.rglob("*.onnx"):
            try: out.setdefault(sha256_of(f), f"_hidden_failed/{f.relative_to(hf).as_posix()}")
            except Exception: pass
    sh = STAGED / "_shipped"
    if sh.exists():
        for f in sh.glob("*.onnx"):
            marked = "prev_hidden_failed" in f.name.lower()
            if not marked:
                sc = sh / f"{f.stem}.json"
                if sc.exists():
                    try: marked = str(json.loads(sc.read_text(encoding="utf-8")).get("candidate_status", "")).lower() in ("hidden_failed", "suspicious")
                    except Exception: marked = False
            if marked:
                try: out.setdefault(sha256_of(f), f"_shipped/{f.name}")
                except Exception: pass
    return out

def cmd_reconcile_banked(args):
    """Detect staged gold that was CONSUMED from Staged_Boladi but never banked into the manifest.
    reconcile-live structurally misses this: it only audits CURRENTLY-staged files, so once submit.py
    clears the staged file without merging, the lost gain is invisible. Here we scan the staged-consumed
    archive (Updates_<Lane>/_consumed/*.json, excluding rejects) and flag any whose staged cost is still
    STRICTLY below the live manifest best -> that improvement never banked. Read-only; exit 1 if any leak.
    (Motivating case: task243 staged 4718 vs manifest 6985, +0.392, silently dropped by a submit.py run.)"""
    man = load_manifest()
    best_per_task = {}; checked = 0                                 # dedup: keep the single best (lowest-cost) leak per task
    for lane in ("Claude", "Codex"):
        cons = lane_dir(CAP_DIR, lane) / "_consumed"
        if not cons.exists():
            continue
        for p in sorted(cons.glob("*.json")):
            if ".rejected_" in p.name:                              # rejects were intentionally NOT staged
                continue
            try:
                pkt = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "cost" not in pkt or "task" not in pkt:
                continue
            task = norm_task(pkt["task"]); cost = pkt["cost"]; best = man.get(task)
            checked += 1
            if best is not None and cost < best * 0.999:            # our staged cost beats the live manifest -> not banked
                gain = points_of(cost) - points_of(int(best))
                cur = best_per_task.get(task)
                if cur is None or cost < cur[3]:
                    best_per_task[task] = (gain, task, lane, cost, int(best), norm_sha(pkt.get("sha256")), p.name)
    known = known_hidden_failed_shas()
    rows = sorted(best_per_task.values(), reverse=True)
    recoverable = [r for r in rows if r[5] not in known]
    suppressed  = [r for r in rows if r[5] in known]
    total = sum(r[0] for r in recoverable)
    for gain, task, lane, cost, best, sha, name in recoverable:
        print(f"STAGED-NOT-BANKED {task} [{lane}]: staged {cost} < manifest {best} (+{gain:.3f}) sha={sha[:12]}  {name}")
    if suppressed:
        print(f"-- SUPPRESSED {len(suppressed)} QUARANTINED (already in _hidden_failed/ or _shipped prev-hidden-failed; NOT recoverable gold -- revive only via a fresh probe-stack overlay): --")
        for gain, task, lane, cost, best, sha, name in suppressed:
            print(f"   [QUARANTINED] {task} [{lane}] staged {cost} sha={sha[:12]}  seen:{known[sha]}")
    print(f"-- reconcile-banked: {len(recoverable)} recoverable task(s) ~+{total:.2f} pts staged-but-not-banked "
          f"({len(suppressed)} suppressed as known hidden-failed) across {checked} archived packet(s). "
          f"Each recoverable still needs a per-task STACK overlay (probe-stack) before re-merge -- this audit is cost-only. --")
    return 1 if recoverable else 0

def cmd_reject(args):
    """Dispose a dead/stale packet -> _consumed/<name>.rejected_<reason>.json. The disposal verb for CONFLICTs that
    detection alone leaves rotting in the inbox (the HELD-is-not-terminal defect in its CONFLICT form)."""
    lane = args.lane.capitalize(); want = norm_task(args.task)
    reason = "".join(c if (c.isalnum() or c == "-") else "-" for c in args.reason.lower())
    while "--" in reason: reason = reason.replace("--", "-")
    reason = reason.strip("-") or "rejected"
    hits = [(p, pkt) for p, pkt in iter_packets(lane) if norm_task(pkt.get("task")) == want]
    if not hits: print(f"(no packet for {want} in {lane})"); return 0
    consumed = lane_dir(CAP_DIR, lane) / "_consumed"; consumed.mkdir(parents=True, exist_ok=True)
    for p, pkt in hits:
        dest = consumed / f"{p.stem}.rejected_{reason}.json"
        print(f"REJECT {want} [{lane}] -> {dest.name}  ({args.reason})" + ("" if args.apply else "  (dry-run)"))
        if args.apply:
            os.replace(p, dest)
            for u in writeback_captain_status(want, pkt.get("sha256"), "rejected", allow_task_fallback=False):
                print(f"   ledger writeback: {u} captain_status->rejected")
    return 0

def cmd_rescue_add(args):
    """Log a strict-gain candidate whose reviewer verdict was lost to a timeout / unparseable output / gate wait.
    A rescue row is VERIFICATION-ONLY follow-up for the captain -- NOT a fresh worker assignment (skill runner-workflow.md)."""
    task = norm_task(args.task)
    live = float(args.live_cost) if args.live_cost is not None else None
    cand = float(args.candidate_cost) if args.candidate_cost is not None else None
    delta = round(cand - live, 6) if (live is not None and cand is not None) else None
    cand_path = args.candidate or ""
    cand_sha = ""
    if cand_path and resolve(cand_path).exists():
        try: cand_sha = sha256_of(resolve(cand_path))
        except Exception: cand_sha = ""
    row = {"task": task, "lane": (args.lane or "").lower(), "candidate_path": cand_path, "candidate_sha": cand_sha,
           "manifest_path": args.manifest or str(MANIFEST), "live_cost": live, "candidate_cost": cand, "cost_delta": delta,
           "checks_done": [c.strip() for c in (args.checks_done or "").split(",") if c.strip()],
           "checks_missing": [c.strip() for c in (args.checks_missing or "").split(",") if c.strip()],
           "worker_final_path": args.worker_final or "", "reviewer_timeout_path": args.reviewer_timeout or "",
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
           "status": "open", "reason": args.reason or "reviewer verdict lost (timeout/invalid/gate-wait)"}
    append_rescue_row(row)
    print(f"RESCUE+ {task} [{row['lane'] or '?'}] cand {cand} vs live {live} (delta {delta}); missing {row['checks_missing'] or '?'} -> {RESCUE_QUEUE}")
    return 0
def cmd_rescue_list(args):
    rows = load_rescue_rows(open_only=not args.all)
    if args.json: print(json.dumps(rows, indent=1)); return 0
    if not rows: print("(no open rescue rows)"); return 0
    for r in rows:
        miss = ",".join(r.get("checks_missing") or []) or "?"
        print(f"{norm_task(r.get('task'))} [{r.get('lane','?')}] {r.get('status','open')}  cand {r.get('candidate_cost')} vs live {r.get('live_cost')} (delta {r.get('cost_delta')})  missing:{miss}  -> {r.get('candidate_path','?')}")
    return 0
def cmd_rescue_resolve(args):
    task = norm_task(args.task)
    hits = [r for r in load_rescue_rows(open_only=True)
            if norm_task(r.get("task")) == task and (not args.candidate or str(r.get("candidate_path", "")) == args.candidate)]
    if not hits: print(f"(no open rescue row for {task})"); return 0
    for r in hits:
        r2 = dict(r); r2["status"] = "resolved"
        r2["resolved_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        r2["resolution"] = args.reason or "handled"
        append_rescue_row(r2); print(f"RESCUE~ resolved {task} ({r2['resolution']})")
    return 0
def cmd_rescue_compact(args):
    summary = compact_rescue_rows(apply=args.apply)
    print(json.dumps(summary, indent=2))
    if not args.apply:
        print("dry-run only; re-run with --apply to rewrite the queue with a backup")
    return 0
def cmd_usage(args):
    try:
        rows = kaggle_rows()
        print(f"today: {submits_today(rows)} observed submits; reference {ADVISORY_RATE_REFERENCE}; diagnostic only; Kaggle 400 is authoritative")
    except Exception as e: print(f"(could not reach Kaggle: {e})")

def main():
    ap = argparse.ArgumentParser(description="Submit Captain probe/drain/stage/reconcile")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("probe-batch"); pb.add_argument("--lane", required=True, choices=["claude", "codex"]); pb.add_argument("--apply", action="store_true"); pb.set_defaults(fn=cmd_probe_batch)
    pt = sub.add_parser("probe-theory"); pt.add_argument("--lane", required=True, choices=["claude", "codex"]); pt.add_argument("--task", required=True); pt.add_argument("--apply", action="store_true"); pt.set_defaults(fn=cmd_probe_theory)
    psk = sub.add_parser("probe-stack"); psk.add_argument("--lane", required=True, choices=["claude", "codex"]); psk.add_argument("--task", default=None); psk.add_argument("--apply", action="store_true"); psk.set_defaults(fn=cmd_probe_stack)
    pi = sub.add_parser("probe-isolated"); pi.add_argument("--lane", required=True, choices=["claude", "codex"]); pi.add_argument("--task", default=None); pi.add_argument("--apply", action="store_true"); pi.set_defaults(fn=cmd_probe_isolated)
    pc = sub.add_parser("probe-collect"); pc.add_argument("--apply", action="store_true"); pc.set_defaults(fn=cmd_probe_collect)
    pst = sub.add_parser("probe-status"); pst.add_argument("--json", action="store_true"); pst.set_defaults(fn=cmd_probe_status)
    dg = sub.add_parser("digest"); dg.add_argument("--lane", default="all", choices=["all", "claude", "codex"]); dg.add_argument("--json", action="store_true"); dg.set_defaults(fn=cmd_digest)
    st = sub.add_parser("stage"); st.add_argument("--lane", required=True, choices=["claude", "codex"]); st.add_argument("--apply", action="store_true"); st.set_defaults(fn=cmd_stage)
    rc = sub.add_parser("reconcile"); rc.add_argument("--apply", action="store_true"); rc.set_defaults(fn=cmd_reconcile)
    rl = sub.add_parser("reconcile-live"); rl.set_defaults(fn=cmd_reconcile_live)
    rb = sub.add_parser("reconcile-banked"); rb.set_defaults(fn=cmd_reconcile_banked)
    rj = sub.add_parser("reject"); rj.add_argument("--lane", required=True, choices=["claude", "codex"]); rj.add_argument("--task", required=True); rj.add_argument("--reason", required=True); rj.add_argument("--apply", action="store_true"); rj.set_defaults(fn=cmd_reject)
    ra = sub.add_parser("rescue-add"); ra.add_argument("--task", required=True); ra.add_argument("--lane", default=""); ra.add_argument("--candidate", default=""); ra.add_argument("--live-cost", type=float, default=None); ra.add_argument("--candidate-cost", type=float, default=None); ra.add_argument("--checks-done", default=""); ra.add_argument("--checks-missing", default=""); ra.add_argument("--worker-final", default=""); ra.add_argument("--reviewer-timeout", default=""); ra.add_argument("--manifest", default=""); ra.add_argument("--reason", default=""); ra.set_defaults(fn=cmd_rescue_add)
    rls = sub.add_parser("rescue-list"); rls.add_argument("--all", action="store_true"); rls.add_argument("--json", action="store_true"); rls.set_defaults(fn=cmd_rescue_list)
    rrs = sub.add_parser("rescue-resolve"); rrs.add_argument("--task", required=True); rrs.add_argument("--candidate", default=""); rrs.add_argument("--reason", default=""); rrs.set_defaults(fn=cmd_rescue_resolve)
    rcp = sub.add_parser("rescue-compact"); rcp.add_argument("--apply", action="store_true"); rcp.set_defaults(fn=cmd_rescue_compact)
    us = sub.add_parser("usage"); us.set_defaults(fn=cmd_usage)
    args = ap.parse_args(); return args.fn(args)

if __name__ == "__main__":
    sys.exit(main())
