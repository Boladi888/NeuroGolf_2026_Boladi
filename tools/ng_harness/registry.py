"""Resolve everything about a task from its number — no hardcoding in agent scripts.

  task_uuid(N)        -> ARC-GEN UUID (sorted-UUID order; via arc_gen_enrich)
  generator_pairs(N)  -> fresh ARC-GEN {input,output} pairs (deterministic, seeded)
  official_pairs(N)   -> the visible+arc-gen pairs from data/taskNNN.json
  parent_onnx(N)      -> reference onnx path for A/B: models/taskNNN.onnx, else
                         submission.zip::taskNNN.onnx (extracted), else newest staged
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import zipfile

NG_ROOT = pathlib.Path(os.environ.get("NEUROGOLF_MAIN_ROOT", str(pathlib.Path(__file__).resolve().parents[2])))
DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026")))
MODELS_DIR = NG_ROOT / "models"
SUBMISSION_ZIP = NG_ROOT / "submissions" / "submission.zip"
TASK_BY_TASK_ROOT = NG_ROOT / "The_400_Tasks" / "Task_by_Task"

sys.path.insert(0, str(NG_ROOT / "tools"))
import arc_gen_enrich as _age  # noqa: E402

_ARC_GEN_DIR = None


def arc_gen_dir():
    global _ARC_GEN_DIR
    if _ARC_GEN_DIR is None:
        _ARC_GEN_DIR = _age.find_arc_gen_dir()
    return _ARC_GEN_DIR


def task_uuid(task_num: int):
    d = arc_gen_dir()
    return _age.task_num_to_uuid(task_num, d) if d else None


def generator_pairs(task_num: int, n: int, seed_offset: int = 2026):
    """Deterministic fresh ARC-GEN pairs. Returns (pairs, err)."""
    d = arc_gen_dir()
    if d is None:
        return None, "ARC-GEN dir not found (set ARC_GEN_DIR or vendor external/ARC-GEN)"
    return _age.generate_pairs(task_num, n, d, seed_offset=seed_offset)


def generator(task_num: int):
    """Return (generator_callable, err) for a task — for robust per-seed generation
    (the caller seeds `random` then calls it, skipping seeds that raise)."""
    d = arc_gen_dir()
    if d is None:
        return None, "ARC-GEN dir not found"
    sys.path.insert(0, str(d))
    try:
        import task_list as arc_task_list  # noqa: E402
    except ImportError as e:
        return None, f"could not import ARC-GEN task_list: {e}"
    uuid = task_uuid(task_num)
    if uuid is None:
        return None, f"task_num {task_num} out of range 1..400"
    info = arc_task_list.task_list().get(uuid)
    if info is None:
        return None, f"ARC-GEN has no generator for {uuid}"
    return info[0], None


def official_pairs(task_num: int):
    """visible (train+test) + arc-gen pairs shipped in data/taskNNN.json."""
    p = DATA_DIR / f"task{task_num:03d}.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])


def parent_onnx(task_num: int):
    """Resolve the reference onnx for A/B. Returns (path, source) or (None, reason).
    Order: models/taskNNN.onnx -> submission.zip::taskNNN.onnx -> newest staged."""
    name = f"task{task_num:03d}.onnx"
    # 1. models/ registry
    m = MODELS_DIR / name
    if m.exists():
        return str(m), "models/"
    # 2. live submission.zip (extract to a stable per-(task,pid) path, overwritten
    #    each call -> no per-call leak, and never stale since re-extracted).
    if SUBMISSION_ZIP.exists():
        try:
            with zipfile.ZipFile(SUBMISSION_ZIP) as z:
                if name in z.namelist():
                    d = pathlib.Path(tempfile.gettempdir()) / "ng_parents"
                    d.mkdir(parents=True, exist_ok=True)
                    out = d / f"task{task_num:03d}_{os.getpid()}.onnx"
                    out.write_bytes(z.read(name))
                    return str(out), "submission.zip"
        except Exception:  # noqa: BLE001
            pass
    # 3. newest task-local staged candidate for this task
    staged_dir = TASK_BY_TASK_ROOT / f"task{task_num:03d}" / "staged_candidates"
    if staged_dir.exists():
        cands = sorted(staged_dir.glob(f"task{task_num:03d}_*/{name}"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return str(cands[0]), f"staged:{cands[0].parent.name}"
    return None, f"no parent onnx for task{task_num:03d} (checked models/, submission.zip, staged)"
