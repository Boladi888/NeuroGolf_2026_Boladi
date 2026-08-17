"""Bridge between NeuroGolf Lab and the surrounding K-NeuroGolf project.

Everything here treats the parent project as read-only EXCEPT for one
additive write path: exported artifacts are saved as new, timestamped files
under ``The_400_Tasks/Task_by_Task/<task_id>/prototypes/``. Existing files are
never modified or overwritten, so concurrent agent sessions are not disturbed.

Resolution order for the project root:

1. ``NEUROGOLF_PROJECT_ROOT`` environment variable, when set.
2. Walking up from this file looking for ``The_400_Tasks`` + ``tools``.

If no project root is found the GUI still works standalone; exports then land
in ``GUI/exports/`` and scoring/baseline lookups report ``available: false``.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

GUI_ROOT = Path(__file__).resolve().parent
CACHE_DIR = GUI_ROOT / ".cache"
FALLBACK_EXPORT_DIR = GUI_ROOT / "exports"

_TASK_STATE_FIELDS = (
    "status",
    "best_known_status",
    "current_cost",
    "current_points",
    "current_src",
    "memory",
    "params",
    "solved_status",
    "verdict",
    "next_action",
    "family",
    "similar_tasks",
    "recipe_hint",
    "unlock_potential",
)

_lock = threading.Lock()
_task_state_cache: dict[str, Any] = {"mtime": None, "rows": {}}
_score_module: Any = None
_score_import_error: str | None = None


def find_project_root() -> Path | None:
    env_root = os.getenv("NEUROGOLF_PROJECT_ROOT", "").strip()
    if env_root:
        path = Path(env_root).expanduser().resolve()
        if (path / "The_400_Tasks").is_dir():
            return path
    for candidate in (GUI_ROOT, *GUI_ROOT.parents):
        if (candidate / "The_400_Tasks").is_dir() and (candidate / "tools").is_dir():
            return candidate
    return None


PROJECT_ROOT = find_project_root()


def _relative_to_root(path: Path) -> str:
    if PROJECT_ROOT is not None:
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            pass
    return str(path)


# ---------------------------------------------------------------------------
# Official scorer (tools/_score_v2.py)
# ---------------------------------------------------------------------------

def _load_score_module() -> Any:
    global _score_module, _score_import_error
    with _lock:
        if _score_module is not None or _score_import_error is not None:
            return _score_module
        if PROJECT_ROOT is None:
            _score_import_error = "project root not found"
            return None
        try:
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from tools import _score_v2  # noqa: PLC0415

            _score_module = _score_v2
        except Exception as exc:  # noqa: BLE001
            _score_import_error = f"scorer import failed: {exc}"
        return _score_module


def scorer_available() -> dict[str, Any]:
    module = _load_score_module()
    return {"available": module is not None, "reason": _score_import_error}


def score_file(path: Path) -> dict[str, Any]:
    """Run the official scoring pipeline on an ONNX file.

    Returns ``{"scored": bool, ...}``. A ``scored: false`` result with a
    ``reason`` mirrors what the Kaggle validator does: the model is dropped.
    """
    module = _load_score_module()
    if module is None:
        return {"scored": False, "reason": _score_import_error or "scorer unavailable"}
    file_size = path.stat().st_size
    result = module.score_network_v2(str(path))
    if result is None:
        return {
            "scored": False,
            "fileSize": file_size,
            "reason": "official scorer returned None (validator would drop this model)",
        }
    macs, memory, params = result
    points = module.score_to_points(memory, params)
    return {
        "scored": True,
        "fileSize": file_size,
        "memory": int(memory),
        "params": int(params),
        "cost": int(memory) + int(params),
        "points": round(points, 6),
    }


# ---------------------------------------------------------------------------
# task_state.csv baseline (read-only)
# ---------------------------------------------------------------------------

def _task_state_rows() -> dict[str, dict[str, str]]:
    if PROJECT_ROOT is None:
        return {}
    csv_path = PROJECT_ROOT / "The_400_Tasks" / "task_state.csv"
    if not csv_path.is_file():
        return {}
    mtime = csv_path.stat().st_mtime_ns
    with _lock:
        if _task_state_cache["mtime"] == mtime:
            return _task_state_cache["rows"]
    rows: dict[str, dict[str, str]] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                task = (row.get("task") or "").strip().lower()
                if task:
                    rows[task] = {field: (row.get(field) or "").strip() for field in _TASK_STATE_FIELDS}
    except Exception:  # noqa: BLE001 - a half-written CSV mid-agent-update is not our problem
        return _task_state_cache["rows"]
    with _lock:
        _task_state_cache["mtime"] = mtime
        _task_state_cache["rows"] = rows
    return rows


def task_baseline(task_id: str) -> dict[str, Any]:
    row = _task_state_rows().get(task_id.strip().lower())
    if not row:
        return {"available": False}
    def _num(value: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return {
        "available": True,
        "currentCost": _num(row["current_cost"]),
        "currentPoints": _num(row["current_points"]),
        "currentSrc": row["current_src"],
        "memory": _num(row["memory"]),
        "params": _num(row["params"]),
        "status": row["status"],
        "solvedStatus": row["solved_status"],
        "verdict": row["verdict"],
        "nextAction": row["next_action"],
        "family": row.get("family", ""),
    }


def campaign_queue(limit: int = 60) -> list[dict[str, str]]:
    """NotSolved tasks ordered for the Doors game: geometric-friendly families
    first (DOORS_DESIGN.md §8), then by family name for stability."""
    family_order = {"Spatial_and_Geometric": 0, "Object_Based": 1, "Pattern_Recognition": 2, "Color_and_Logical": 3}
    rows = []
    for task, row in _task_state_rows().items():
        if row.get("solved_status") != "NotSolved":
            continue
        rows.append({
            "taskId": task,
            "family": row.get("family", ""),
            "unlockPotential": row.get("unlock_potential", ""),
        })
    rows.sort(key=lambda item: (family_order.get(item["family"], 9), item["taskId"]))
    return rows[:limit]


def task_rule_doc(task_id: str) -> dict[str, Any]:
    """The project's best-known rule for a task: the ``## Rule`` section of
    ``The_400_Tasks/Task_by_Task/taskNNN.md`` (read-only)."""
    if PROJECT_ROOT is None:
        return {"rule": None, "known": False, "source": None}
    md_path = PROJECT_ROOT / "The_400_Tasks" / "Task_by_Task" / f"{task_id}.md"
    if not md_path.is_file():
        return {"rule": None, "known": False, "source": None}
    text = md_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^## Rule\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    rule = match.group(1).strip() if match else None
    known = bool(rule) and not rule.startswith("_unknown")
    return {"rule": rule or None, "known": known, "source": _relative_to_root(md_path)}


def list_recipes(task_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Most recent door recipes / rule notes saved for a task (newest first)."""
    out_dir = artifact_dir(task_id)
    if not out_dir.is_dir():
        return []
    recipes = []
    for path in sorted(out_dir.glob("*.recipe.json"), reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        recipes.append({
            "file": path.name,
            "exportRoute": data.get("exportRoute", ""),
            "ruleHypothesis": data.get("ruleHypothesis", ""),
            "path": data.get("path") or [],
            "win": data.get("win"),
        })
    return recipes


# ---------------------------------------------------------------------------
# Artifact export (additive writes only)
# ---------------------------------------------------------------------------

def artifact_dir(task_id: str) -> Path:
    if PROJECT_ROOT is not None:
        return PROJECT_ROOT / "The_400_Tasks" / "Task_by_Task" / task_id / "prototypes"
    return FALLBACK_EXPORT_DIR / task_id


def save_recipe(task_id: str, recipe: dict[str, Any]) -> dict[str, str]:
    """Write a neurogolf-doorpath recipe JSON next to the prototypes. Additive
    only — timestamped name, never overwrites."""
    out_dir = artifact_dir(task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{task_id}_doorpath_{stamp}.recipe.json"
    suffix = 2
    while path.exists():
        path = out_dir / f"{task_id}_doorpath_{stamp}_{suffix}.recipe.json"
        suffix += 1
    path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    return {"recipe": _relative_to_root(path), "absolute": str(path)}


def save_artifact(model_bytes: bytes, task_id: str, sidecar: dict[str, Any]) -> dict[str, str]:
    """Write ``<task>_gui_<utc>.onnx`` plus a ``.json`` sidecar. Never overwrites."""
    out_dir = artifact_dir(task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{task_id}_gui_{stamp}"
    onnx_path = out_dir / f"{base}.onnx"
    suffix = 2
    while onnx_path.exists():
        onnx_path = out_dir / f"{base}_{suffix}.onnx"
        suffix += 1
    onnx_path.write_bytes(model_bytes)
    sidecar_path = onnx_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return {
        "onnx": _relative_to_root(onnx_path),
        "sidecar": _relative_to_root(sidecar_path),
        "absoluteOnnx": str(onnx_path),
    }


# ---------------------------------------------------------------------------
# Current best submission (submissions/submission.zip, read-only)
# ---------------------------------------------------------------------------

def _submission_zip() -> Path | None:
    if PROJECT_ROOT is None:
        return None
    path = PROJECT_ROOT / "submissions" / "submission.zip"
    return path if path.is_file() else None


def best_zip_path() -> Path | None:
    return _submission_zip()


def best_onnx_bytes(task_id: str) -> bytes | None:
    zip_path = _submission_zip()
    if zip_path is None:
        return None
    try:
        with zipfile.ZipFile(zip_path) as archive:
            return archive.read(f"{task_id}.onnx")
    except (KeyError, zipfile.BadZipFile, OSError):
        return None


def best_manifest() -> dict[str, Any] | None:
    """Manifest of the canonical best submission, cached by zip mtime+size."""
    zip_path = _submission_zip()
    if zip_path is None:
        return None
    stat = zip_path.stat()
    cache_key = f"{stat.st_mtime_ns}:{stat.st_size}"
    cache_file = CACHE_DIR / "best_manifest.json"
    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("cacheKey") == cache_key:
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    import onnx  # noqa: PLC0415 - heavy import, only when rebuilding the cache

    tasks: list[dict[str, Any]] = []
    op_types: set[str] = set()
    op_inputs: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            name = info.filename
            if not name.endswith(".onnx"):
                continue
            task_id = name[: -len(".onnx")]
            entry: dict[str, Any] = {
                "taskId": task_id,
                "path": f"/api/best-onnx/{task_id}",
                "bytes": info.file_size,
            }
            try:
                model = onnx.load_from_string(archive.read(name))
                ops = sorted({node.op_type for node in model.graph.node})
                entry["opTypes"] = ops
                entry["nodeCount"] = len(model.graph.node)
                op_types.update(ops)
                for node in model.graph.node:
                    visible = len([item for item in node.input if item])
                    op_inputs[node.op_type] = max(op_inputs.get(node.op_type, 0), visible)
            except Exception:  # noqa: BLE001 - keep the entry, just without op info
                entry["opTypes"] = []
            tasks.append(entry)

    manifest = {
        "cacheKey": cache_key,
        "zipName": zip_path.name,
        "zipPath": "/api/best-zip",
        "zipSize": stat.st_size,
        "fileCount": len(tasks),
        "source": _relative_to_root(zip_path),
        "opTypes": sorted(op_types),
        "opInputs": op_inputs,
        "tasks": tasks,
    }
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(manifest), encoding="utf-8")
    except OSError:
        pass
    return manifest


# ---------------------------------------------------------------------------
# Project status summary
# ---------------------------------------------------------------------------

def project_status() -> dict[str, Any]:
    zip_path = _submission_zip()
    return {
        "projectRoot": str(PROJECT_ROOT) if PROJECT_ROOT else None,
        "scorer": scorer_available(),
        "taskState": PROJECT_ROOT is not None
        and (PROJECT_ROOT / "The_400_Tasks" / "task_state.csv").is_file(),
        "bestZip": _relative_to_root(zip_path) if zip_path else None,
        "artifactRoot": _relative_to_root(artifact_dir("taskNNN").parent),
    }
