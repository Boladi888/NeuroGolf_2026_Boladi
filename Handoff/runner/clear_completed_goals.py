#!/usr/bin/env python3
"""Clear completed HandOff board goals the owner-approved way (delete-to-unassigned).

Implements the HandOffz skill `references/board-cleanup-and-archival.md` "Python Automation
Target": for the K-NeuroGolf live board, a *completed* batch goal is cleared by DELETing the
goal so its terminal tasks fall into the unassigned/home area and drop off the active board.

SAFETY (this touches the LIVE board, so the defaults are conservative):
- Dry-run by default. Deleting requires BOTH --apply AND --delete-to-unassigned.
- Refuses any goal that still has a non-terminal task (todo / in_progress / pending_verification)
  unless --require-terminal is turned off (which is intentionally NOT offered — non-terminal is a
  hard block).
- Never deletes a protected goal: `Bo to 34t7h`, anything that looks like an external partner /
  advice goal, plus any --protect you pass.
- After each delete, verifies GET /nodes/<goal_id> returns 410 resource_deleted.
- Writes a cleanup event to the operations log and prints before/after counts.

The decision core (`plan_clear`) is a pure function over already-fetched goals+tasks, so it can be
unit-tested offline without any board call.

Usage:
  python Handoff/runner/clear_completed_goals.py --goal <goal-id-or-title> [--goal ...] --dry-run
  python Handoff/runner/clear_completed_goals.py --goal <goal-id-or-title> --delete-to-unassigned --apply
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]                 # Handoff/runner/clear_completed_goals.py -> repo root
DEFAULT_CONFIG = REPO / "Handoff" / "config" / "runner_claude.json"
OPS_LOG = REPO / "Handoff" / "logs" / "operations.jsonl"
API_BASE = "https://handoff.socnet.lol/api/v1"

TERMINAL_STATUSES = {"verified", "rejected"}               # board-canonical terminal statuses
NONTERMINAL_STATUSES = {"todo", "in_progress", "pending_verification"}
# Never delete these (case-insensitive title/external-key match). `Bo to 34t7h` is the owner's pinned goal.
PROTECTED_TITLES = {"bo to 34t7h"}
PROTECTED_PATTERNS = (re.compile(r"\b(external|partner|advice|advisor)\b", re.I),)
_UUID_RE = re.compile(r"[0-9a-fA-F-]{36}")


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def is_uuid(val: Any) -> bool:
    return isinstance(val, str) and bool(_UUID_RE.fullmatch(val))


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


class BoardClient:
    """Minimal HandOff REST client (mirrors prepare_handoff_batch.BoardClient; own copy so this tool
    does not depend on cwd or that module importing cleanly)."""

    def __init__(self, api_base: str = API_BASE, agent_id: str | None = None) -> None:
        self.api_base = api_base.rstrip("/")
        env = load_env_file(REPO / ".env.handoff")
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 NeuroGolf-GoalCleanup/1.0",
        }
        if env.get("HANDOFF_BOOTSTRAP_TOKEN"):
            self.headers["Authorization"] = f"Bearer {env['HANDOFF_BOOTSTRAP_TOKEN']}"
        if agent_id:
            agent_env = load_env_file(REPO / "Handoff" / "private" / f"{agent_id}.env")
            key = agent_env.get("HANDOFF_AGENT_KEY") or agent_env.get("AGENT_KEY")
            if key:
                self.headers["X-Agent-Key"] = key
                self.headers["X-Agent-Id"] = agent_env.get("HANDOFF_AGENT_ID") or agent_id

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(f"{self.api_base}{path}", data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(text) if text else {})
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            try:
                return exc.code, json.loads(text)
            except Exception:
                return exc.code, {"raw": text[:500]}


def _iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


def extract_goals(project_data: Any) -> list[dict]:
    """Best-effort: a goal is a dict with a uuid id + a title that lives under a 'goals' collection."""
    goals: dict[str, dict] = {}
    for node in _iter_dicts(project_data):
        vals = node.get("goals") if isinstance(node, dict) else None
        if isinstance(vals, list):
            for g in vals:
                if isinstance(g, dict) and is_uuid(g.get("id") or g.get("goal_id")):
                    gid = g.get("id") or g.get("goal_id")
                    goals[gid] = {"id": gid, "title": g.get("title", ""),
                                  "external_key": g.get("external_key", ""), "status": g.get("status", "")}
    return list(goals.values())


def extract_tasks(tasks_data: Any) -> list[dict]:
    """Normalize the /tasks response to [{id,title,status,goal_id,parent_id}]."""
    raw = tasks_data if isinstance(tasks_data, list) else None
    if raw is None and isinstance(tasks_data, dict):
        for key in ("tasks", "items", "data", "results"):
            if isinstance(tasks_data.get(key), list):
                raw = tasks_data[key]
                break
    out = []
    for t in raw or []:
        if not isinstance(t, dict):
            continue
        tid = t.get("id") or t.get("task_id")
        if not is_uuid(tid):
            continue
        out.append({"id": tid, "title": t.get("title", ""), "status": str(t.get("status", "")).lower(),
                    "goal_id": t.get("goal_id"), "parent_id": t.get("parent_id"),
                    "external_key": t.get("external_key", "")})
    return out


def is_protected(goal: dict, extra_protect: set[str]) -> str | None:
    title = str(goal.get("title", ""))
    key = str(goal.get("external_key", ""))
    low = title.lower()
    if low in PROTECTED_TITLES or title in extra_protect or goal.get("id") in extra_protect or key in extra_protect:
        return "protected (pinned/named)"
    for pat in PROTECTED_PATTERNS:
        if pat.search(title) or pat.search(key):
            return "protected (looks like an external/advice goal)"
    return None


def resolve_targets(goals: list[dict], wanted: list[str]) -> tuple[list[dict], list[str]]:
    """Map each --goal (uuid, title, or external_key) to a goal dict. Returns (matched, unresolved)."""
    by_id = {g["id"]: g for g in goals}
    matched, unresolved = [], []
    for w in wanted:
        if w in by_id:
            matched.append(by_id[w]); continue
        hits = [g for g in goals if g.get("title") == w or g.get("external_key") == w
                or str(g.get("title", "")).lower() == w.lower()]
        if len(hits) == 1:
            matched.append(hits[0])
        elif len(hits) > 1:
            unresolved.append(f"{w} (ambiguous: {len(hits)} goals match)")
        else:
            unresolved.append(f"{w} (no matching goal)")
    return matched, unresolved


def tasks_of_goal(goal_id: str, tasks: list[dict]) -> list[dict]:
    return [t for t in tasks if t.get("goal_id") == goal_id or t.get("parent_id") == goal_id]


def plan_clear(goals: list[dict], tasks: list[dict], targets: list[dict], extra_protect: set[str]) -> list[dict]:
    """PURE decision core. For each target goal decide clearable vs blocked with a reason + task counts.
    No I/O — unit-testable offline."""
    plan = []
    for g in targets:
        gtasks = tasks_of_goal(g["id"], tasks)
        counts: dict[str, int] = {}
        for t in gtasks:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        nonterminal = [t for t in gtasks if t["status"] in NONTERMINAL_STATUSES
                       or (t["status"] not in TERMINAL_STATUSES and t["status"] != "")]
        prot = is_protected(g, extra_protect)
        if prot:
            decision, reason = "BLOCKED", prot
        elif not gtasks:
            decision, reason = "BLOCKED", "no tasks found under this goal (refusing to delete an unexpected/empty goal)"
        elif nonterminal:
            decision, reason = "BLOCKED", f"{len(nonterminal)} non-terminal task(s): " + ",".join(sorted({t['status'] for t in nonterminal}))
        else:
            decision, reason = "CLEARABLE", f"all {len(gtasks)} task(s) terminal ({dict(counts)})"
        plan.append({"goal_id": g["id"], "title": g.get("title", ""), "decision": decision,
                     "reason": reason, "task_counts": counts, "n_tasks": len(gtasks)})
    return plan


def unassigned_counts(tasks: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        if not t.get("goal_id") and not t.get("parent_id"):
            out[t["status"]] = out.get(t["status"], 0) + 1
    return out


def log_ops(kind: str, summary: str, data: dict) -> None:
    try:
        OPS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with OPS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts_utc": now_iso(), "kind": kind, "summary": summary, "data": data}) + "\n")
    except Exception:
        pass


def default_project_id() -> str | None:
    for cfg in (DEFAULT_CONFIG, REPO / "Handoff" / "config" / "runner.json"):
        if cfg.exists():
            try:
                p = json.loads(cfg.read_text(encoding="utf-8")).get("project", {})
                pid = p.get("project_id") or p.get("request_id")
                if pid:
                    return pid
            except Exception:
                continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Clear completed HandOff goals (delete-to-unassigned).")
    ap.add_argument("--project", help="Project/request id (default: from Handoff/config/runner_claude.json).")
    ap.add_argument("--goal", action="append", default=[], help="Goal id, title, or external_key to clear. Repeatable.")
    ap.add_argument("--protect", action="append", default=[], help="Extra goal id/title/external_key to never delete. Repeatable.")
    ap.add_argument("--delete-to-unassigned", action="store_true", help="Required (with --apply) to actually DELETE the goal.")
    ap.add_argument("--apply", action="store_true", help="Perform the delete. Without it, dry-run only.")
    ap.add_argument("--api-base", default=API_BASE)
    ap.add_argument("--agent-id", help="Requester/orchestrator agent env for the delete (X-Agent-Key).")
    ap.add_argument("--json", action="store_true", help="Emit the plan as JSON.")
    args = ap.parse_args()

    project_id = args.project or default_project_id()
    if not project_id:
        print("ERROR: no --project and could not read project_id from a base config", file=sys.stderr)
        return 2
    if not args.goal:
        print("ERROR: pass at least one --goal", file=sys.stderr)
        return 2

    client = BoardClient(args.api_base, args.agent_id)
    pq = urllib.parse.quote(str(project_id), safe="")
    st_p, project_data = client.request("GET", f"/requests/{pq}")
    st_t, tasks_data = client.request("GET", f"/requests/{pq}/tasks")
    if st_p >= 400 or st_t >= 400:
        print(f"ERROR: board read failed (project HTTP {st_p}, tasks HTTP {st_t}). "
              f"Check .env.handoff creds / --agent-id.", file=sys.stderr)
        return 1

    goals = extract_goals(project_data)
    tasks = extract_tasks(tasks_data)
    targets, unresolved = resolve_targets(goals, args.goal)
    extra_protect = set(args.protect)
    plan = plan_clear(goals, tasks, targets, extra_protect)

    if args.json:
        print(json.dumps({"project_id": project_id, "live_goals": len(goals), "total_tasks": len(tasks),
                          "unresolved": unresolved, "plan": plan,
                          "unassigned_before": unassigned_counts(tasks)}, indent=2))
    else:
        print(f"== clear_completed_goals (project {project_id}) ==")
        print(f"live goals: {len(goals)} | tasks: {len(tasks)} | unassigned/home now: {unassigned_counts(tasks)}")
        for u in unresolved:
            print(f"  UNRESOLVED --goal {u}")
        for row in plan:
            print(f"  [{row['decision']}] {row['title'] or row['goal_id']}  -- {row['reason']}")

    clearable = [row for row in plan if row["decision"] == "CLEARABLE"]
    if not args.apply or not args.delete_to_unassigned:
        print(f"-- DRY-RUN: {len(clearable)} goal(s) would be cleared. "
              f"Re-run with --delete-to-unassigned --apply to delete. --" if clearable
              else "-- DRY-RUN: nothing clearable. --")
        return 0

    deleted, failed = [], []
    for row in clearable:
        gid = row["goal_id"]; gq = urllib.parse.quote(gid, safe="")
        st_del, _ = client.request("DELETE", f"/requests/{pq}/goals/{gq}")
        st_chk, _ = client.request("GET", f"/nodes/{gq}")
        ok = st_del < 400 and st_chk == 410
        (deleted if ok else failed).append({"goal_id": gid, "title": row["title"],
                                            "delete_http": st_del, "verify_410": st_chk})
        print(f"  {'DELETED' if ok else 'FAILED'} {row['title'] or gid}  (delete HTTP {st_del}, verify {st_chk})")
    log_ops("board_goals_cleared", f"cleared {len(deleted)} goal(s), {len(failed)} failed",
            {"project_id": project_id, "deleted": deleted, "failed": failed})
    # re-read to report the resulting unassigned/home counts
    _, tasks_after = client.request("GET", f"/requests/{pq}/tasks")
    print(f"-- APPLIED: {len(deleted)} deleted, {len(failed)} failed. "
          f"unassigned/home after: {unassigned_counts(extract_tasks(tasks_after))} --")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
