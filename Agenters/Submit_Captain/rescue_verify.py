#!/usr/bin/env python3
"""Plan and locally re-verify review-rescue candidates.

This is intentionally Captain-local: it reads the rescue queue, checks every
candidate against the live team-best bar, resolves stale queue rows, and, when
requested, runs the local verification ladder before writing a Captain packet.

It does not run Kaggle, stage to Staged_Boladi, or touch submission.zip.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
CAPTAIN_CTL = REPO / "Agenters" / "Submit_Captain" / "captain_ctl.py"


def load_captain_ctl():
    spec = importlib.util.spec_from_file_location("captain_ctl", CAPTAIN_CTL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CAPTAIN_CTL}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cap = load_captain_ctl()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )


def task_num(task: str) -> int:
    return int(cap.norm_task(task)[4:])


def rel_or_abs(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def current_bar(task: str) -> dict[str, Any]:
    cp = run([sys.executable, "tools/ng_current_bar.py", "--task", str(task_num(task)), "--json"], timeout=90)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or cp.stdout.strip())
    return json.loads(cp.stdout)


def parse_profile(stdout: str) -> dict[str, int]:
    m = re.search(r"static=(None|\d+)\s+profile=(\d+)\s+\(memory\s+(\d+)\s+\+\s+params\s+(\d+)", stdout)
    if not m:
        raise ValueError(f"could not parse profile output: {stdout.strip()!r}")
    return {
        "static": None if m.group(1) == "None" else int(m.group(1)),
        "profile": int(m.group(2)),
        "memory": int(m.group(3)),
        "params": int(m.group(4)),
    }


def points(cost: float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, cost)))


def append_resolved(row: dict[str, Any], reason: str) -> None:
    row2 = dict(row)
    row2["status"] = "resolved"
    row2["resolved_utc"] = utc_now()
    row2["resolution"] = reason
    cap.append_rescue_row(row2)


def packet_candidate_key(path_text: str | Path | None) -> str:
    text = str(path_text or "")
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = REPO / path
    try:
        return os.path.normcase(str(path.resolve()))
    except Exception:
        return os.path.normcase(str(path))


def existing_packet_for(item: dict[str, Any]) -> Path | None:
    task = cap.norm_task(item.get("task"))
    cand_sha = str(item.get("candidate_sha") or "").lower()
    cand_key = packet_candidate_key(item.get("candidate_path"))
    inboxes = [
        REPO / "Agenters" / "Submit_Captain" / "Updates_Codex",
        REPO / "Agenters" / "Submit_Captain" / "Updates_Claude",
    ]
    for inbox in inboxes:
        if not inbox.exists():
            continue
        for path in sorted(inbox.glob(f"*{task}*.json")):
            try:
                pkt = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if cap.norm_task(pkt.get("task")) != task:
                continue
            hashes = {
                str(pkt.get("sha256") or "").lower(),
                str(pkt.get("candidate_sha256") or "").lower(),
                str(pkt.get("candidate_sha") or "").lower(),
            }
            if cand_sha and cand_sha in hashes:
                return path
            packet_paths = [
                pkt.get("onnx"),
                pkt.get("candidate_path"),
                pkt.get("onnx_path"),
                pkt.get("path"),
            ]
            if cand_key and any(packet_candidate_key(p) == cand_key for p in packet_paths if p):
                return path
    return None


def write_pid_state(args: argparse.Namespace, state: str, exit_code: int | None = None) -> None:
    if not args.pid_file:
        return
    payload: dict[str, Any] = {
        "pid": os.getpid(),
        "lane": args.pid_lane,
        "batch": args.batch,
        "state": state,
        "running": state == "running",
        "updated_utc": utc_now(),
        "tasks": [cap.norm_task(t) for t in args.task],
        "report": args.report,
        "plan": args.plan_json,
        "command": [Path(sys.executable).name, *sys.argv],
    }
    if state == "running":
        payload["started_utc"] = utc_now()
    else:
        payload["stopped_utc"] = utc_now()
        payload["exit_code"] = exit_code
    path = Path(args.pid_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def packet_path(lane: str, task: str) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lane_name = "Claude" if str(lane).lower() == "claude" else "Codex"
    out_dir = REPO / "Agenters" / "Submit_Captain" / f"Updates_{lane_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stamp}-{task}-rescue.json"


def gate(mode: str, task: str, cand: Path, args: argparse.Namespace, full: bool = False, n: int | None = None) -> dict[str, Any]:
    cmd = [sys.executable, "tools/ng_gate.py", mode, "--task", str(task_num(task)), "--cand", str(cand), "--json"]
    if full:
        cmd.append("--full")
    elif n:
        cmd += ["--n", str(n)]
    if not args.watch_queue:
        cmd.append("--no-watch-queue")
    cp = run(cmd, timeout=args.gate_timeout)
    if cp.returncode != 0:
        return {"ok": False, "returncode": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr}
    try:
        # ng_gate may print a queue-watcher notice before JSON when the visible
        # watcher is enabled; parse from the first JSON object.
        data = json.loads(cp.stdout[cp.stdout.index("{"):])
    except Exception:
        return {"ok": False, "returncode": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr, "parse_error": True}
    data["ok"] = True
    return data


def gate_pass(mode: str, result: dict[str, Any]) -> bool:
    if not result.get("ok"):
        return False
    if mode == "visible":
        return bool(result.get("pass"))
    if mode == "truth":
        return result.get("correct") == result.get("total") and int(result.get("total") or 0) > 0
    if mode == "ab":
        return result.get("divergences") == 0 and bool(result.get("byte_equal"))
    return False


def plan_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = cap.load_rescue_rows(open_only=True)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw:
        task = cap.norm_task(row.get("task"))
        cand = row.get("candidate_path") or ""
        resolved = str(cap.resolve(cand).resolve()) if cand else ""
        latest[(task, resolved)] = row

    items: list[dict[str, Any]] = []
    for row in latest.values():
        task = cap.norm_task(row.get("task"))
        cand = cap.resolve(row.get("candidate_path") or "")
        reasons: list[str] = []
        bar: dict[str, Any] | None = None
        live_cost: float | None = None
        live_sha: str | None = None
        cand_sha = ""
        try:
            bar = current_bar(task)
            live_cost = float(bar["live_cost"])
            live_sha = str(bar.get("live_sha256") or "").lower()
        except Exception as exc:
            reasons.append(f"bar_error:{exc}")
        try:
            candidate_cost = float(row.get("candidate_cost"))
        except Exception:
            candidate_cost = None
            reasons.append("no_candidate_cost")
        if not cand.exists():
            reasons.append("missing_candidate")
        else:
            cand_sha = cap.sha256_of(cand)
        if candidate_cost is not None and live_cost is not None and candidate_cost >= live_cost:
            reasons.append("stale_no_strict_gain")
        if cand_sha and live_sha and cand_sha.lower() == live_sha:
            reasons.append("already_live_sha")
        items.append({
            "row": row,
            "task": task,
            "candidate": cand,
            "candidate_path": str(cand),
            "candidate_cost": candidate_cost,
            "candidate_sha": cand_sha,
            "bar": bar,
            "live_cost": live_cost,
            "live_sha": live_sha,
            "reasons": reasons,
        })

    best_by_task: dict[str, dict[str, Any]] = {}
    for item in items:
        if item["reasons"]:
            continue
        task = item["task"]
        current = best_by_task.get(task)
        if current is None or (item["candidate_cost"], item["candidate_path"]) < (
            current["candidate_cost"],
            current["candidate_path"],
        ):
            best_by_task[task] = item

    selected = list(best_by_task.values())
    for item in items:
        if item["reasons"]:
            continue
        if best_by_task.get(item["task"]) is not item:
            item["reasons"].append("superseded_by_lower_cost_same_task")

    skipped = [item for item in items if item["reasons"]]
    return selected, skipped


def write_plan(path: Path, selected: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    payload = {
        "created_utc": utc_now(),
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "selected": [summarize_item(i) for i in sorted(selected, key=lambda x: (x["live_cost"] - x["candidate_cost"]) * -1)],
        "skipped": [summarize_item(i) | {"reasons": i["reasons"]} for i in skipped],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_item(item: dict[str, Any]) -> dict[str, Any]:
    cand = item.get("candidate_cost")
    live = item.get("live_cost")
    delta = None if cand is None or live is None else cand - live
    return {
        "task": item["task"],
        "candidate_cost": cand,
        "live_cost": live,
        "cost_delta": delta,
        "candidate_path": rel_or_abs(Path(item["candidate_path"])),
        "candidate_sha": item.get("candidate_sha") or "",
        "live_sha": item.get("live_sha") or "",
    }


def verify_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    task = item["task"]
    cand = Path(item["candidate_path"])
    lane = item["row"].get("lane") or "codex"
    bar = current_bar(task)
    live_cost = float(bar["live_cost"])
    live_sha = str(bar.get("live_sha256") or "").lower()
    cand_sha = cap.sha256_of(cand)
    result: dict[str, Any] = {
        "task": task,
        "candidate_path": rel_or_abs(cand),
        "candidate_sha": cand_sha,
        "live_cost": live_cost,
        "live_sha": live_sha,
        "status": "started",
        "started_utc": utc_now(),
        "commands": {},
    }
    if cand_sha.lower() == live_sha:
        result["status"] = "stale_already_live_sha"
        result["finished_utc"] = utc_now()
        append_resolved(item["row"], "stale_already_live_sha")
        return result

    profile_cp = run(
        [
            sys.executable,
            "tools/agent_helpers/profile_real_cost.py",
            "--task",
            str(task_num(task)),
            "--cand",
            str(cand),
            "--arcgen",
            str(args.arcgen),
        ],
        timeout=args.profile_timeout,
    )
    result["commands"]["profile_real_cost"] = {
        "returncode": profile_cp.returncode,
        "stdout": profile_cp.stdout,
        "stderr": profile_cp.stderr,
    }
    if profile_cp.returncode != 0:
        result["status"] = "profile_failed"
        result["finished_utc"] = utc_now()
        return result
    profile = parse_profile(profile_cp.stdout)
    result["profile"] = profile
    if profile["profile"] >= live_cost:
        result["status"] = "stale_no_strict_gain_after_profile"
        result["finished_utc"] = utc_now()
        append_resolved(item["row"], f"stale_no_strict_gain_after_profile profile={profile['profile']} live={live_cost}")
        return result

    visible = gate("visible", task, cand, args)
    result["commands"]["visible"] = visible
    if not gate_pass("visible", visible):
        result["status"] = "visible_failed"
        result["finished_utc"] = utc_now()
        append_resolved(item["row"], "visible_failed_during_rescue_reverify")
        return result

    if args.capped_n:
        truth_n = gate("truth", task, cand, args, n=args.capped_n)
        result["commands"][f"truth_n{args.capped_n}"] = truth_n
        if not gate_pass("truth", truth_n):
            result["status"] = "truth_capped_failed"
            result["finished_utc"] = utc_now()
            append_resolved(item["row"], f"truth_n{args.capped_n}_failed")
            return result
        ab_n = gate("ab", task, cand, args, n=args.capped_n)
        result["commands"][f"ab_n{args.capped_n}"] = ab_n
        if not gate_pass("ab", ab_n):
            result["status"] = "ab_capped_failed"
            result["finished_utc"] = utc_now()
            append_resolved(item["row"], f"ab_n{args.capped_n}_failed")
            return result

    heavy_n = int(args.heavy_n)
    truth_heavy = gate("truth", task, cand, args, n=heavy_n)
    result["commands"][f"truth_n{heavy_n}"] = truth_heavy
    if not gate_pass("truth", truth_heavy):
        result["status"] = f"truth_n{heavy_n}_failed"
        result["finished_utc"] = utc_now()
        append_resolved(item["row"], f"truth_n{heavy_n}_failed")
        return result

    ab_heavy = gate("ab", task, cand, args, n=heavy_n)
    result["commands"][f"ab_n{heavy_n}"] = ab_heavy
    if not gate_pass("ab", ab_heavy):
        result["status"] = f"ab_n{heavy_n}_failed"
        result["finished_utc"] = utc_now()
        append_resolved(item["row"], f"ab_n{heavy_n}_failed")
        return result

    safety_cp = run(
        [sys.executable, "tools/ng_safety.py", "model", "--task", str(task_num(task)), "--cand", str(cand), "--json"],
        timeout=args.safety_timeout,
    )
    result["commands"]["safety"] = {
        "returncode": safety_cp.returncode,
        "stdout": safety_cp.stdout,
        "stderr": safety_cp.stderr,
    }
    safety: dict[str, Any] = {}
    if safety_cp.returncode == 0:
        try:
            safety = json.loads(safety_cp.stdout)
        except Exception:
            safety = {"parse_error": True, "raw": safety_cp.stdout}
    result["safety"] = safety

    diff_text = ""
    if live_cost - profile["profile"] <= 10:
        live_path = REPO / "Team_Best_Onnx" / f"{task}.onnx"
        diff_cp = run(
            [sys.executable, "tools/ng_graph_cost_diff.py", "--task", str(task_num(task)), "--live", str(live_path), "--candidate", str(cand)],
            timeout=args.diff_timeout,
        )
        diff_text = diff_cp.stdout if diff_cp.returncode == 0 else (diff_cp.stderr or diff_cp.stdout)
        result["commands"]["graph_cost_diff"] = {
            "returncode": diff_cp.returncode,
            "stdout": diff_cp.stdout,
            "stderr": diff_cp.stderr,
        }

    pkt = {
        "task": task,
        "source_lane": "Claude" if str(lane).lower() == "claude" else "Codex",
        "lane": "SAFE_BATCH_QUEUE",
        "onnx": str(cand),
        "cost": profile["profile"],
        "profile_cost": profile["profile"],
        "sha256": cand_sha,
        "reviewer_verdict": "SHIP",
        "live_cost": live_cost,
        "candidate_profile_cost": profile["profile"],
        "cost_delta": profile["profile"] - live_cost,
        "points_delta": points(profile["profile"]) - points(live_cost),
        "gates": {
            "visible": f"{visible.get('correct')}/{visible.get('total')} PASS",
            "profile_real_cost": profile,
            f"truth_n{heavy_n}": f"{truth_heavy.get('correct')}/{truth_heavy.get('total')}",
            f"ab_n{heavy_n}": f"{ab_heavy.get('match')}/{ab_heavy.get('total')} byte_equal={ab_heavy.get('byte_equal')}",
        },
        "candidate_class": f"PATH_B_TRUTH_N{heavy_n}_BYTE_EQUAL",
        "parent_source": "Team_Submission/Backend/manifest.csv",
        "parent_sha256": live_sha,
        "candidate_sha256": cand_sha,
        "proof_commands_run": [
            "tools/ng_current_bar.py",
            "tools/agent_helpers/profile_real_cost.py",
            "tools/ng_gate.py visible",
            f"tools/ng_gate.py truth --n {heavy_n}",
            f"tools/ng_gate.py ab --n {heavy_n}",
            "tools/ng_safety.py model",
        ],
        "proof_results": result["commands"],
        "safety": safety,
        "x86_required": False,
        "kaggle_probe_required": True,
        "status": "PROBE_READY",
        "msg": (
            f"Rescue reverify: strict gain {profile['profile']} vs live {int(live_cost)} "
            f"(delta {profile['profile'] - live_cost:+.0f}); truth --n {heavy_n} and A/B --n {heavy_n} passed."
        ),
    }
    if diff_text:
        pkt["graph_cost_diff"] = diff_text

    if args.write_packets:
        out = packet_path(str(lane), task)
        out.write_text(json.dumps(pkt, indent=2), encoding="utf-8")
        result["packet_path"] = rel_or_abs(out)

    append_resolved(item["row"], f"verified_n{heavy_n}_gates packet={result.get('packet_path', '')}")
    result["status"] = "verified_packeted" if args.write_packets else "verified_no_packet"
    result["finished_utc"] = utc_now()
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply-stale-resolutions", action="store_true", help="Append resolved rows for stale/superseded candidates.")
    ap.add_argument("--verify", action="store_true", help="Run local verification gates for selected candidates.")
    ap.add_argument("--write-packets", action="store_true", help="Write Submit Captain packets for verified candidates.")
    ap.add_argument("--limit", type=int, default=0, help="Maximum selected candidates to verify.")
    ap.add_argument("--task", action="append", default=[], help="Only plan/verify these task ids.")
    ap.add_argument("--report", default="", help="Write JSONL verification report here.")
    ap.add_argument("--plan-json", default="", help="Write plan JSON here.")
    ap.add_argument("--watch-queue", action="store_true", help="Allow ng_gate to open/use the visible queue watcher.")
    ap.add_argument("--capped-n", type=int, default=0, help="Optional capped truth/ab before full gates.")
    ap.add_argument("--heavy-n", type=int, default=8000, help="Heavy verification truth/ab cap.")
    ap.add_argument("--arcgen", type=int, default=266)
    ap.add_argument("--profile-timeout", type=int, default=900)
    ap.add_argument("--safety-timeout", type=int, default=300)
    ap.add_argument("--diff-timeout", type=int, default=300)
    ap.add_argument("--gate-timeout", type=int, default=14400)
    ap.add_argument("--pid-file", default="", help="Write managed rescue verifier pid/status JSON here.")
    ap.add_argument("--pid-lane", default="", help="Lane name to record in the managed pid file.")
    ap.add_argument("--batch", default="", help="Batch id to record in the managed pid file.")
    ap.add_argument(
        "--no-resolve-existing-packets",
        dest="resolve_existing_packets",
        action="store_false",
        help="Do not auto-resolve queue rows whose candidate already has a Captain packet.",
    )
    ap.add_argument(
        "--no-compact-queue",
        dest="compact_queue",
        action="store_false",
        help="Do not compact the rescue queue after verification/resolution work.",
    )
    ap.set_defaults(resolve_existing_packets=True)
    ap.set_defaults(compact_queue=True)
    args = ap.parse_args(argv)

    write_pid_state(args, "running")
    exit_code = 1
    try:
        exit_code = run_main(args)
        return exit_code
    finally:
        if args.compact_queue:
            try:
                summary = cap.compact_rescue_rows(apply=True)
                print(json.dumps({"rescue_compact": summary}, indent=2), flush=True)
            except Exception as exc:
                print(f"WARNING rescue_compact_failed: {exc}", file=sys.stderr, flush=True)
        write_pid_state(args, "stopped", exit_code)


def run_main(args: argparse.Namespace) -> int:

    selected, skipped = plan_rows()
    if args.task:
        wanted = {cap.norm_task(t) for t in args.task}
        selected = [i for i in selected if i["task"] in wanted]
        skipped = [i for i in skipped if i["task"] in wanted]

    if args.resolve_existing_packets:
        kept: list[dict[str, Any]] = []
        for item in selected:
            pkt = existing_packet_for(item)
            if pkt:
                item["reasons"].append("already_packeted")
                append_resolved(item["row"], f"already_packeted packet={rel_or_abs(pkt)}")
                skipped.append(item)
            else:
                kept.append(item)
        selected = kept

    if args.plan_json:
        write_plan(Path(args.plan_json), selected, skipped)

    print(json.dumps({
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "selected": [summarize_item(i) for i in selected],
        "skip_reasons": sorted({r for item in skipped for r in item["reasons"]}),
    }, indent=2), flush=True)

    if args.apply_stale_resolutions:
        for item in skipped:
            append_resolved(item["row"], ",".join(item["reasons"]))
        print(f"resolved_stale_rows={len(skipped)}", flush=True)

    if not args.verify:
        return 0

    to_run = sorted(selected, key=lambda item: (item["candidate_cost"] - item["live_cost"], item["task"]))
    if args.limit:
        to_run = to_run[: args.limit]
    report_path = Path(args.report) if args.report else None
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    for item in to_run:
        summary = summarize_item(item)
        print(f"VERIFY {summary['task']} cand={summary['candidate_cost']} live={summary['live_cost']} delta={summary['cost_delta']}", flush=True)
        try:
            result = verify_one(item, args)
        except subprocess.TimeoutExpired as exc:
            result = summary | {"status": "timeout", "cmd": exc.cmd, "timeout": exc.timeout, "finished_utc": utc_now()}
        except Exception as exc:
            result = summary | {"status": "error", "error": str(exc), "finished_utc": utc_now()}
        print(json.dumps({"task": result.get("task"), "status": result.get("status"), "packet_path": result.get("packet_path")}, indent=2), flush=True)
        if report_path:
            with report_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
