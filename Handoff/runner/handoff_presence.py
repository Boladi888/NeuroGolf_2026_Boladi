from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HANDOFF_ROOT = REPO / "Handoff"
LOG_PATH = HANDOFF_ROOT / "logs" / "presence.jsonl"
MCP_URL = "https://handoff.socnet.lol/mcp"
AGENTS = [
    "boladi_a",
    "bo_ng_orchestrator",
    "bo_ng_worker1",
    "bo_ng_cdx_worker2",
    "bo_ng_cdx_worker3",
    "bo_ng_reviewer1",
    "bo_ng_cdx_reviewer2",
    "bo_ng_submit_captain",
    "bo_ng_cld_orchestrator",
    "bo_ng_cld_worker1",
    "bo_ng_cld_worker2",
    "bo_ng_cld_worker3",
    "bo_ng_cld_reviewer1",
    "bo_ng_cld_reviewer2",
    "bo_ng_cld_submit_captain",
]


class HandoffMcp:
    def __init__(self) -> None:
        self.session_id: str | None = None
        self.counter = 0
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "Mozilla/5.0 NeuroGolf-HandoffPresence/1.0",
        }

    def _post(self, payload: dict, timeout: int = 30) -> tuple[dict, dict[str, str]]:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = dict(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(MCP_URL, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_headers = dict(response.headers.items())
            body = response.read().decode("utf-8", "replace")
        match = re.search(r"data:\s*(\{.*\})", body, re.S)
        if match:
            body = match.group(1)
        return json.loads(body), response_headers

    def connect(self) -> None:
        self.counter += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self.counter,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "neurogolf-presence", "version": "1.0"},
            },
        }
        _, headers = self._post(payload)
        self.session_id = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")
        try:
            self._post(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
                timeout=15,
            )
        except Exception:
            pass

    def call_tool(self, name: str, arguments: dict, max_reconnects: int = 2) -> dict:
        # Bounded reconnect loop. The previous version recursed on every "no
        # valid session" reply, so a persistent auth/session failure recursed
        # forever (RecursionError / hang). Cap the reconnect attempts and then
        # surface the error to the caller.
        obj: dict = {}
        for attempt in range(max_reconnects + 1):
            if not self.session_id:
                self.connect()
            self.counter += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self.counter,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
            obj, _ = self._post(payload)
            error = obj.get("error") if isinstance(obj, dict) else None
            stale = (
                isinstance(error, dict)
                and str(error.get("message", "")).lower().startswith("no valid session")
            )
            if stale and attempt < max_reconnects:
                self.session_id = None
                continue
            return obj
        return obj


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(kind: str, **data: object) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts_utc": utc_now(), "kind": kind, **data}
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":")) + "\n")


def heartbeat_once(client: HandoffMcp, agents: list[str]) -> list[dict]:
    results = []
    for agent_id in agents:
        try:
            result = client.call_tool("agent_heartbeat", {"agent_id": agent_id})
            error = result.get("error")
            is_error = bool(error) or bool(result.get("result", {}).get("isError"))
            results.append(
                {
                    "agent_id": agent_id,
                    "ok": not is_error,
                    "error": error.get("message") if isinstance(error, dict) else None,
                }
            )
        except Exception as exc:
            client.session_id = None
            results.append({"agent_id": agent_id, "ok": False, "error": str(exc)})
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--agents", nargs="*", default=AGENTS)
    args = parser.parse_args()

    client = HandoffMcp()
    while True:
        results = heartbeat_once(client, args.agents)
        ok_count = sum(1 for item in results if item["ok"])
        log("heartbeat", ok_count=ok_count, total=len(results), results=results)
        print(json.dumps({"ts_utc": utc_now(), "ok_count": ok_count, "total": len(results)}), flush=True)
        if args.once:
            return 0 if ok_count == len(results) else 1
        time.sleep(max(15, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())


