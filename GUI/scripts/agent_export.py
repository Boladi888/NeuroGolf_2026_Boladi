#!/usr/bin/env python3
"""Headless NeuroGolf Lab export helper for agents.

This intentionally uses the same /api/export (or /api/validate) path as the
GUI. It does not write ONNX directly. Validation runs against the official
task JSON (train + test + arc-gen) with the exact competition semantics, and
passing exports are scored with the project's official scorer and saved under
The_400_Tasks/Task_by_Task/<task>/prototypes/.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_DIRS = [ROOT / "client" / "dist" / "tasks", ROOT / "client" / "public" / "tasks"]


def load_task(task_id: str) -> dict[str, Any]:
    for task_dir in TASK_DIRS:
        path = task_dir / f"{task_id}.json"
        if path.exists():
            return json.loads(path.read_text())
    raise SystemExit(f"could not find {task_id}.json in {[str(path) for path in TASK_DIRS]}")


def color_remap_graph(task_id: str, mapping: dict[str, int]) -> dict[str, Any]:
    nodes = [
        {"id": "input_1", "type": "op", "position": {"x": 70, "y": 80}, "data": {"label": "input_1", "opType": "Input", "shape": "1,1,30,30"}},
    ]
    edges = []
    current = "input_1"
    x_pos = 250
    for index, (source_color, target_color) in enumerate(mapping.items(), start=1):
        match_id = f"const_match_{source_color}_{index}"
        value_id = f"const_value_{target_color}_{index}"
        equal_id = f"equal_{index}"
        where_id = f"where_{index}"
        nodes.extend(
            [
                {"id": match_id, "type": "op", "position": {"x": x_pos, "y": 40}, "data": {"label": match_id, "opType": "Constant", "shape": "1,1,30,30", "value": str(source_color)}},
                {"id": value_id, "type": "op", "position": {"x": x_pos, "y": 170}, "data": {"label": value_id, "opType": "Constant", "shape": "1,1,30,30", "value": str(target_color)}},
                {"id": equal_id, "type": "op", "position": {"x": x_pos + 180, "y": 70}, "data": {"label": equal_id, "opType": "Equal", "shape": "1,1,30,30"}},
                {"id": where_id, "type": "op", "position": {"x": x_pos + 360, "y": 95}, "data": {"label": where_id, "opType": "Where", "shape": "1,1,30,30"}},
            ]
        )
        edges.extend(
            [
                {"id": f"eq{index}a", "source": "input_1", "target": equal_id, "targetHandle": "a"},
                {"id": f"eq{index}b", "source": match_id, "target": equal_id, "targetHandle": "b"},
                {"id": f"wh{index}c", "source": equal_id, "target": where_id, "targetHandle": "condition"},
                {"id": f"wh{index}t", "source": value_id, "target": where_id, "targetHandle": "true"},
                {"id": f"wh{index}f", "source": current, "target": where_id, "targetHandle": "false"},
            ]
        )
        current = where_id
        x_pos += 220
    nodes.append({"id": "output_1", "type": "op", "position": {"x": x_pos + 200, "y": 95}, "data": {"label": "output_1", "opType": "Output", "shape": "1,1,30,30"}})
    edges.append({"id": "out", "source": current, "target": "output_1", "targetHandle": "input"})
    return {"projectName": f"agent-{task_id}", "taskId": task_id, "nodes": nodes, "edges": edges}


def post_json(host: str, endpoint: str, payload: dict[str, Any]) -> tuple[int, str]:
    url = host.rstrip("/") + endpoint
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def summarize(body: str) -> None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return
    validation = data.get("validation") or {}
    for set_name, stats in (validation.get("sets") or {}).items():
        print(f"  {set_name}: {stats['pass']} pass, {stats['fail']} fail, {stats['skipped']} skipped")
    score = data.get("score") or {}
    if score.get("scored"):
        print(f"  score: memory={score['memory']} params={score['params']} cost={score['cost']} points={score['points']:.3f}")
    elif score.get("reason"):
        print(f"  score: unavailable ({score['reason']})")
    baseline = data.get("baseline") or {}
    if baseline.get("available") and baseline.get("currentPoints") is not None:
        delta = data.get("deltaPoints")
        delta_text = f" (delta {delta:+.3f})" if isinstance(delta, (int, float)) else ""
        print(f"  baseline: cost={baseline.get('currentCost')} points={baseline['currentPoints']:.3f}{delta_text}")
    if data.get("artifact"):
        print(f"  artifact: {data['artifact']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a NeuroGolf graph through the live backend export gate.")
    parser.add_argument("--host", default="http://127.0.0.1:8082", help="NeuroGolf Lab backend URL")
    parser.add_argument("--task", required=True, help="task id, for example task276")
    parser.add_argument("--graph", help="path to graph JSON payload or nodes/edges JSON")
    parser.add_argument("--color-remap", help='quick graph builder mapping JSON, for example \'{"6":2}\'')
    parser.add_argument("--representation", choices=["scalar", "onehot"], help="graph representation (default: payload value or scalar)")
    parser.add_argument("--bridge-mask", choices=["input", "none"], help="scalar bridge padding mask (default input)")
    parser.add_argument("--sets", help='comma list of example sets to validate, e.g. "train" or "train,test,arc-gen"')
    parser.add_argument("--validate-only", action="store_true", help="validate + score without saving an artifact")
    args = parser.parse_args()

    task_id = args.task.lower()
    if args.color_remap:
        payload = color_remap_graph(task_id, json.loads(args.color_remap))
    elif args.graph:
        payload = json.loads(Path(args.graph).read_text())
        payload.setdefault("taskId", task_id)
        payload.setdefault("projectName", f"agent-{task_id}")
    else:
        raise SystemExit("provide --graph or --color-remap")

    if args.representation:
        payload["representation"] = args.representation
    if args.bridge_mask:
        payload["bridgeMask"] = args.bridge_mask
    if args.sets:
        payload["exampleSets"] = [part.strip() for part in args.sets.split(",") if part.strip()]

    endpoint = "/api/validate" if args.validate_only else "/api/export"
    status, body = post_json(args.host, endpoint, payload)
    print(status)
    print(body)
    summarize(body)
    if status >= 400:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
