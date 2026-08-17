#!/usr/bin/env python3
"""Strict ONNX peephole rewriter for sanctioned byte-equal cost cuts.

This is intentionally a small, gated v1:

* generate conservative ONNX-native candidates
* reject non-improving static candidates before expensive gates
* require ng_gate ab --full before staging
* require gate-5 real-input profile and compare credited profile costs
* stage only to task-local staged_candidates/; never submit

The first pass set avoids task-structural synthesis. Equal-tail rebuilds,
groups halving, no-Mul decode, and separable-Einsum finals need stronger
task-specific guards and are left for a later version.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.spread_patterns._base import SpreadPattern, SpreadResult  # noqa: E402
from tools.spread_patterns.pass_add_chain_to_sum import PATTERN as ADD_CHAIN_TO_SUM  # noqa: E402
from tools.spread_patterns.pass_const_to_init import PATTERN as CONST_TO_INIT  # noqa: E402
from tools.spread_patterns.pass_drop_cast_before_pad_bool import PATTERN as DROP_CAST_PAD_BOOL  # noqa: E402
from tools.spread_patterns.pass_opset_bump_pad_uint8 import PATTERN as OPSET_BUMP_PAD_UINT8  # noqa: E402

SUBMISSION_ZIP = REPO_ROOT / "submissions" / "submission.zip"
TASK_STATE_CSV = REPO_ROOT / "The_400_Tasks" / "task_state.csv"
TASK_BY_TASK_ROOT = REPO_ROOT / "The_400_Tasks" / "Task_by_Task"
REPORT_PATH = REPO_ROOT / "Logs" / "research" / "R3_peephole_sweep.md"
SCRATCH_ROOT = REPO_ROOT / "Logs" / "scratch" / "peephole_rewrite"
SCORE_SCRIPT = REPO_ROOT / "tools" / "spread_score_isolated.py"
NG_GATE = REPO_ROOT / "tools" / "ng_gate.py"
PROFILE_SCRIPT = REPO_ROOT / "tools" / "agent_helpers" / "profile_real_cost.py"

DEFAULT_PATTERNS = (
    "const_to_init",
    "zero_pad_into_conv",
    "add_chain_to_sum",
    "drop_cast_before_pad_bool",
    "opset_bump_pad_uint8",
)

PROFILE_RE = re.compile(
    r"static=(?P<static>None|\d+)\s+profile=(?P<profile>\d+)\s+"
    r"\(memory (?P<memory>\d+) \+ params (?P<params>\d+); "
    r"(?P<n_inputs>\d+) real inputs\)"
)


@dataclass
class Candidate:
    model: onnx.ModelProto
    rewrites: list[dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _points(cost: int | None) -> float | None:
    if cost is None:
        return None
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _task_id(task: int) -> str:
    return f"task{task:03d}"


def _ai_onnx_opset(model: onnx.ModelProto) -> int | None:
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return int(imp.version)
    return None


def _copy_model(model: onnx.ModelProto) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    return out


def _attr_ints(node: onnx.NodeProto, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name:
            return [int(x) for x in attr.ints]
    return None


def _attr_string(node: onnx.NodeProto, name: str, default: str = "") -> str:
    for attr in node.attribute:
        if attr.name == name:
            return attr.s.decode("utf-8") if attr.s else ""
    return default


def _set_attr_ints(node: onnx.NodeProto, name: str, values: list[int]) -> onnx.NodeProto:
    new_node = onnx.NodeProto()
    new_node.CopyFrom(node)
    kept = [a for a in new_node.attribute if a.name != name]
    new_node.ClearField("attribute")
    new_node.attribute.extend(kept)
    new_node.attribute.append(helper.make_attribute(name, values))
    return new_node


def _build_producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            if output:
                producers[output] = node
    return producers


def _build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def _initializer_map(graph: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {init.name: init for init in graph.initializer}


def _is_graph_output(graph: onnx.GraphProto, name: str) -> bool:
    return any(output.name == name for output in graph.output)


def _tensor_to_array(init: onnx.TensorProto) -> np.ndarray | None:
    try:
        return numpy_helper.to_array(init)
    except Exception:
        return None


def _pad_pads_and_zero_value(node: onnx.NodeProto, graph: onnx.GraphProto) -> list[int] | None:
    """Return pads if this is constant-zero Pad; otherwise None."""
    mode = _attr_string(node, "mode", "constant")
    if mode != "constant":
        return None

    pads = _attr_ints(node, "pads")
    if pads is None:
        if len(node.input) < 2 or not node.input[1]:
            return None
        init = _initializer_map(graph).get(node.input[1])
        if init is None:
            return None
        arr = _tensor_to_array(init)
        if arr is None:
            return None
        pads = [int(x) for x in arr.flatten().tolist()]

    # Attribute-form Pad uses "value"; input-form uses input[2].
    for attr in node.attribute:
        if attr.name == "value" and abs(float(attr.f)) > 1e-9:
            return None
    if len(node.input) >= 3 and node.input[2]:
        init = _initializer_map(graph).get(node.input[2])
        if init is None:
            return None
        arr = _tensor_to_array(init)
        if arr is None:
            return None
        if arr.size and any(abs(float(v)) > 1e-9 for v in arr.flatten()):
            return None

    return pads


def _rewrite_zero_pad_into_conv(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    """Fuse positive constant-zero NCHW Pad into a following Conv's pads attr."""
    graph = model.graph
    consumers = _build_consumer_map(graph)
    replacements: dict[int, onnx.NodeProto] = {}
    drop_ids: set[int] = set()
    pad_outputs_to_drop: set[str] = set()
    matches: list[dict[str, Any]] = []

    for pad in graph.node:
        if pad.op_type != "Pad" or not pad.output:
            continue
        pad_out = pad.output[0]
        if _is_graph_output(graph, pad_out):
            continue
        pad_consumers = consumers.get(pad_out, [])
        if len(pad_consumers) != 1:
            continue
        conv = pad_consumers[0]
        if conv.op_type != "Conv" or len(conv.input) < 2 or conv.input[0] != pad_out:
            continue
        auto_pad = _attr_string(conv, "auto_pad", "NOTSET") or "NOTSET"
        if auto_pad not in ("NOTSET", ""):
            continue

        pads = _pad_pads_and_zero_value(pad, graph)
        if pads is None or len(pads) != 8:
            continue
        # Only NCHW spatial padding can be fused into Conv. Padding batch or
        # channel dimensions changes a different axis and is skipped.
        n_b, c_b, h_b, w_b, n_e, c_e, h_e, w_e = pads
        if any(x != 0 for x in (n_b, c_b, n_e, c_e)):
            continue
        if any(x < 0 for x in pads):
            continue
        conv_pads = _attr_ints(conv, "pads") or [0, 0, 0, 0]
        if len(conv_pads) != 4 or any(x < 0 for x in conv_pads):
            continue

        new_conv = onnx.NodeProto()
        new_conv.CopyFrom(conv)
        new_conv.input[0] = pad.input[0]
        new_pads = [
            conv_pads[0] + h_b,
            conv_pads[1] + w_b,
            conv_pads[2] + h_e,
            conv_pads[3] + w_e,
        ]
        new_conv = _set_attr_ints(new_conv, "pads", new_pads)

        drop_ids.add(id(pad))
        replacements[id(conv)] = new_conv
        pad_outputs_to_drop.add(pad_out)
        matches.append({
            "pad": pad.name or pad_out,
            "conv": conv.name or (conv.output[0] if conv.output else ""),
            "pad_pads": pads,
            "old_conv_pads": conv_pads,
            "new_conv_pads": new_pads,
        })

    if not matches:
        return SpreadResult(
            pattern_name="zero_pad_into_conv",
            verdict="no-op",
            notes="no single-use constant-zero NCHW Pad -> Conv chain detected",
        )

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if id(node) in drop_ids:
            continue
        new_nodes.append(replacements.get(id(node), node))

    new_value_info = [vi for vi in graph.value_info if vi.name not in pad_outputs_to_drop]
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=list(graph.initializer),
        value_info=new_value_info,
    )
    new_model = helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
    )
    return SpreadResult(
        pattern_name="zero_pad_into_conv",
        verdict="applied",
        new_model=new_model,
        notes=f"fused {len(matches)} constant-zero Pad -> Conv chain(s)",
        extras={"matches": matches, "chains_fused": len(matches)},
    )


def _rewrite_drop_cast_before_pad_bool_guarded(model: onnx.ModelProto, task_id: str) -> SpreadResult:
    opset = _ai_onnx_opset(model)
    if opset is None or opset < 13:
        return SpreadResult(
            pattern_name="drop_cast_before_pad_bool",
            verdict="no-op",
            notes=f"opset {opset} < 13; bool Pad is not allowed by the v1 guard",
        )
    return DROP_CAST_PAD_BOOL.apply(model, task_id)


ZERO_PAD_INTO_CONV = SpreadPattern(
    name="zero_pad_into_conv",
    category="math-equivalent",
    description="fuse single-use constant-zero NCHW Pad into following Conv.pads",
    apply=_rewrite_zero_pad_into_conv,
    safe_by_construction=True,
)

DROP_CAST_PAD_BOOL_GUARDED = SpreadPattern(
    name="drop_cast_before_pad_bool",
    category="math-equivalent",
    description="drop Cast(bool->F32) before Pad only for opset 13+ bool-Pad graphs",
    apply=_rewrite_drop_cast_before_pad_bool_guarded,
    safe_by_construction=True,
)

PATTERN_REGISTRY: dict[str, SpreadPattern] = {
    "const_to_init": CONST_TO_INIT,
    "zero_pad_into_conv": ZERO_PAD_INTO_CONV,
    "add_chain_to_sum": ADD_CHAIN_TO_SUM,
    "drop_cast_before_pad_bool": DROP_CAST_PAD_BOOL_GUARDED,
    "opset_bump_pad_uint8": OPSET_BUMP_PAD_UINT8,
}


def _prune_unused_initializers(model: onnx.ModelProto) -> int:
    graph = model.graph
    used = {inp for node in graph.node for inp in node.input if inp}
    used.update(o.name for o in graph.output)
    graph_inputs = {i.name for i in graph.input}
    keep = [init for init in graph.initializer if init.name in used or init.name in graph_inputs]
    removed = len(graph.initializer) - len(keep)
    if removed:
        del graph.initializer[:]
        graph.initializer.extend(keep)
    return removed


def _cleanup_model(model: onnx.ModelProto) -> onnx.ModelProto:
    _prune_unused_initializers(model)
    model.graph.ClearField("value_info")
    return model


def _apply_patterns(model: onnx.ModelProto, task: int, patterns: list[SpreadPattern]) -> Candidate:
    current = _copy_model(model)
    rewrites: list[dict[str, Any]] = []
    tid = _task_id(task)
    for pattern in patterns:
        try:
            result = pattern.apply(current, tid)
        except Exception as exc:
            rewrites.append({
                "name": pattern.name,
                "verdict": "error",
                "notes": f"{type(exc).__name__}: {exc}",
            })
            continue
        if result.verdict != "applied" or result.new_model is None:
            continue
        current = result.new_model
        rewrites.append({
            "name": pattern.name,
            "verdict": result.verdict,
            "notes": result.notes,
            "extras": result.extras,
        })
    if rewrites:
        current = _cleanup_model(current)
    return Candidate(model=current, rewrites=rewrites)


def _load_workable_tasks() -> list[int]:
    tasks: list[int] = []
    with TASK_STATE_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("boladi_workable") == "Y":
                raw = row.get("task_id") or ""
                if raw:
                    tasks.append(int(raw))
    return sorted(set(tasks))


def _parse_task_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part.replace("task", "")))
    return sorted(set(out))


def _extract_from_zip(task: int, out_dir: Path, zip_path: Path = SUBMISSION_ZIP) -> Path:
    name = f"{_task_id(task)}.onnx"
    out = out_dir / name
    with zipfile.ZipFile(zip_path) as z:
        out.write_bytes(z.read(name))
    return out


def _resolve_source(task: int, source: str, work_dir: Path) -> tuple[Path, str]:
    tid = _task_id(task)
    if source == "live":
        return _extract_from_zip(task, work_dir), "submission.zip"
    if source == "root-first":
        root_path = REPO_ROOT / f"{tid}.onnx"
        if root_path.exists():
            out = work_dir / root_path.name
            shutil.copy2(root_path, out)
            return out, "repo-root"
        return _extract_from_zip(task, work_dir), "submission.zip"
    path = Path(source)
    if path.is_dir():
        src = path / f"{tid}.onnx"
    else:
        src = path
    if not src.exists():
        raise FileNotFoundError(f"source ONNX not found for {tid}: {src}")
    out = work_dir / f"{tid}.onnx"
    shutil.copy2(src, out)
    return out, str(src)


def _save_model(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _run_json(cmd: list[str], timeout: float | None = None) -> tuple[int, dict[str, Any] | None, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    payload = None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if lines:
        for line in reversed(lines):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return proc.returncode, payload, proc.stdout, proc.stderr


def _score_path(path: Path, timeout: float | None = None) -> dict[str, Any]:
    cmd = [sys.executable, str(SCORE_SCRIPT), str(path)]
    rc, payload, stdout, stderr = _run_json(cmd, timeout=timeout)
    if rc != 0:
        return {"error": f"score subprocess exit={rc}: {stderr.strip()[:300]}"}
    if payload is None:
        return {"error": f"score subprocess emitted no JSON: {stdout.strip()[:300]}"}
    if "error" in payload:
        return {"error": payload["error"]}
    return payload


def _run_ab_full(task: int, cand: Path, ref: Path, out_path: Path, timeout: float | None) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(NG_GATE),
        "ab",
        "--task",
        str(task),
        "--cand",
        str(cand),
        "--ref",
        str(ref),
        "--full",
        "--json",
        "--out",
        str(out_path),
    ]
    rc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    else:
        payload = None
    if payload is None:
        return {
            "error": f"ng_gate exit={rc.returncode}; no JSON out; stdout={rc.stdout[-300:]} stderr={rc.stderr[-300:]}",
            "byte_equal": False,
        }
    payload["returncode"] = rc.returncode
    if rc.returncode != 0 and payload.get("byte_equal") is not True:
        payload.setdefault("error", f"ng_gate exit={rc.returncode}")
    return payload


def _profile_path(
    task: int,
    path: Path,
    *,
    arcgen: int,
    max_n: int,
    timeout: float | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--task",
        str(task),
        "--cand",
        str(path),
    ]
    if arcgen:
        cmd.extend(["--arcgen", str(arcgen)])
    if max_n:
        cmd.extend(["--max-n", str(max_n)])
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return {"error": text.strip()[:500], "raw": text}
    m = PROFILE_RE.search(text)
    if not m:
        return {"error": f"could not parse profile output: {text.strip()[:500]}", "raw": text}
    static_raw = m.group("static")
    return {
        "static_cost": None if static_raw == "None" else int(static_raw),
        "profile_cost": int(m.group("profile")),
        "profile_memory": int(m.group("memory")),
        "params": int(m.group("params")),
        "n_inputs": int(m.group("n_inputs")),
        "raw": text.strip(),
    }


def _profile_gap(static_cost: int | None, profile_cost: int | None) -> dict[str, Any]:
    if not static_cost or profile_cost is None:
        return {"abs": None, "pct": None}
    gap = profile_cost - static_cost
    return {"abs": gap, "pct": gap / static_cost}


def _stage_candidate(
    task: int,
    cand_path: Path,
    record: dict[str, Any],
    *,
    stage_root: Path | None = None,
) -> Path:
    cost = record["new_profile"]["profile_cost"]
    task_name = _task_id(task)
    if stage_root is None:
        stage_root = TASK_BY_TASK_ROOT / task_name / "staged_candidates"
    stage_dir = stage_root / f"{task_name}_peephole_cost{cost}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_onnx = stage_dir / f"{task_name}.onnx"
    shutil.copy2(cand_path, staged_onnx)

    new_score = record.get("new_score") or {}
    old_profile = record.get("old_profile") or {}
    new_profile = record.get("new_profile") or {}
    ab = record.get("ab_full") or {}
    rewrites = [r["name"] for r in record.get("rewrites", [])]
    manifest = {
        "task": task_name,
        "candidate": f"{task_name}_peephole_cost{cost}",
        "role": "sanctioned ONNX peephole byte-equal lower-cost rewrite",
        "cost": cost,
        "profile_cost": cost,
        "memory": new_profile.get("profile_memory"),
        "params": new_profile.get("params"),
        "score_v2_cost": new_score.get("cost"),
        "sha256": _sha256(staged_onnx),
        "onnx": str(staged_onnx.relative_to(REPO_ROOT)).replace("\\", "/"),
        "builder": "tools/peephole_rewrite.py",
        "builder_args": record.get("builder_args", ""),
        "rewrites": rewrites,
        "rewrite_details": record.get("rewrites", []),
        "source": {
            "kind": record.get("source_kind"),
            "sha256": record.get("old_sha256"),
            "score_v2": record.get("old_score"),
            "profile_real_cost": old_profile,
        },
        "gates": {
            "ab_full": (
                f"{ab.get('match')}/{ab.get('total')} byte_equal vs source; "
                f"divergences={ab.get('divergences')}"
            ),
            "score_v2": (
                f"static={new_score.get('cost')} "
                f"(memory {new_score.get('memory')} + params {new_score.get('params')})"
            ),
            "profile_real_cost": (
                f"{new_profile.get('profile_cost')} "
                f"({new_profile.get('n_inputs')} real inputs; "
                f"static/profile gap {record.get('profile_gap_label')})"
            ),
        },
        "notes": (
            "Auto-staged by peephole_rewrite.py after full AB byte-equality and "
            "real-input profile cost reduction. Captain review required; tool never submits."
        ),
    }
    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stage_dir


def _empty_record(task: int, source_kind: str, old_path: Path) -> dict[str, Any]:
    return {
        "task": _task_id(task),
        "task_num": task,
        "source_kind": source_kind,
        "source_path": str(old_path),
        "old_sha256": _sha256(old_path) if old_path.exists() else None,
        "rewrites": [],
        "old_score": None,
        "new_score": None,
        "old_profile": None,
        "new_profile": None,
        "ab_full": None,
        "byte_equal": "N",
        "profile_gap": None,
        "profile_gap_label": "not_run",
        "staged": "N",
        "stage_dir": "",
        "notes": "",
    }


def process_task(
    task: int,
    *,
    source: str,
    patterns: list[SpreadPattern],
    check: bool,
    gate_in_check: bool,
    profile_arcgen: int,
    profile_max_n: int,
    timeout: float | None,
    builder_args: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"peephole_{task:03d}_") as td_raw:
        work_dir = Path(td_raw)
        old_path, source_kind = _resolve_source(task, source, work_dir)
        record = _empty_record(task, source_kind, old_path)
        record["builder_args"] = builder_args

        old_score = _score_path(old_path, timeout=timeout)
        record["old_score"] = old_score
        if "error" in old_score:
            record["notes"] = f"old score failed: {old_score['error']}"
            return record

        base_model = onnx.load(str(old_path))
        candidate = _apply_patterns(base_model, task, patterns)
        if not candidate.rewrites:
            record["notes"] = "no pattern matched"
            return record
        record["rewrites"] = candidate.rewrites

        cand_path = work_dir / f"{_task_id(task)}_peephole.onnx"
        _save_model(candidate.model, cand_path)
        record["new_sha256"] = _sha256(cand_path)
        new_score = _score_path(cand_path, timeout=timeout)
        record["new_score"] = new_score
        if "error" in new_score:
            record["notes"] = f"candidate score failed: {new_score['error']}"
            return record

        old_static = int(old_score["cost"])
        new_static = int(new_score["cost"])
        record["static_delta"] = new_static - old_static
        if new_static >= old_static:
            record["notes"] = f"static cost not lower ({old_static} -> {new_static})"
            return record

        should_gate = (not check) or gate_in_check
        if not should_gate:
            record["byte_equal"] = "not_run"
            record["profile_gap_label"] = "not_run"
            record["notes"] = f"static candidate only ({old_static} -> {new_static}); gates not run in --check"
            return record

        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        ab_out = SCRATCH_ROOT / f"{_task_id(task)}_ab_full_{int(time.time())}_{os.getpid()}.json"
        ab = _run_ab_full(task, cand_path, old_path, ab_out, timeout=timeout)
        record["ab_full"] = ab
        if not ab.get("byte_equal"):
            record["byte_equal"] = "N"
            record["notes"] = f"ab_full failed: {ab.get('error') or ab.get('first_divergence')}"
            return record
        record["byte_equal"] = "Y"

        old_profile = _profile_path(
            task,
            old_path,
            arcgen=profile_arcgen,
            max_n=profile_max_n,
            timeout=timeout,
        )
        record["old_profile"] = old_profile
        if "error" in old_profile:
            record["notes"] = f"old profile failed: {old_profile['error']}"
            return record

        new_profile = _profile_path(
            task,
            cand_path,
            arcgen=profile_arcgen,
            max_n=profile_max_n,
            timeout=timeout,
        )
        record["new_profile"] = new_profile
        if "error" in new_profile:
            record["notes"] = f"candidate profile failed: {new_profile['error']}"
            return record

        gap = _profile_gap(new_profile.get("static_cost"), new_profile.get("profile_cost"))
        record["profile_gap"] = gap
        if gap["abs"] is None:
            record["profile_gap_label"] = "unknown"
        else:
            record["profile_gap_label"] = f"{gap['abs']} ({gap['pct'] * 100:.1f}%)"

        old_cost = int(old_profile["profile_cost"])
        new_cost = int(new_profile["profile_cost"])
        record["profile_delta"] = new_cost - old_cost
        if new_cost >= old_cost:
            record["notes"] = f"profile cost not lower ({old_cost} -> {new_cost}); rejected"
            return record

        if not check:
            stage_dir = _stage_candidate(task, cand_path, record)
            record["staged"] = "Y"
            record["stage_dir"] = str(stage_dir.relative_to(REPO_ROOT)).replace("\\", "/")
            record["notes"] = f"staged profile cut {old_cost} -> {new_cost}"
        else:
            record["staged"] = "N"
            record["notes"] = f"gated profile cut {old_cost} -> {new_cost}; dry-run not staged"
        return record


def write_report(records: list[dict[str, Any]], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = [r for r in records if r.get("staged") == "Y"]
    gated_cuts = [
        r for r in records
        if r.get("byte_equal") == "Y"
        and (r.get("profile_delta") is not None)
        and int(r["profile_delta"]) < 0
    ]
    static_candidates = [
        r for r in records
        if r.get("static_delta") is not None and int(r["static_delta"]) < 0
    ]
    saved = sum(-int(r["profile_delta"]) for r in staged if r.get("profile_delta") is not None)
    point_gain = 0.0
    for r in staged:
        old_cost = (r.get("old_profile") or {}).get("profile_cost")
        new_cost = (r.get("new_profile") or {}).get("profile_cost")
        if old_cost is not None and new_cost is not None:
            point_gain += (_points(int(new_cost)) or 0.0) - (_points(int(old_cost)) or 0.0)

    by_rewrite: dict[str, dict[str, Any]] = {}
    for r in staged:
        delta = -int(r.get("profile_delta") or 0)
        for rw in r.get("rewrites", []):
            name = rw.get("name", "?")
            slot = by_rewrite.setdefault(name, {"tasks": 0, "profile_saved_on_tasks": 0})
            slot["tasks"] += 1
            slot["profile_saved_on_tasks"] += delta

    lines = [
        "# R3 Peephole Sweep",
        "",
        f"Generated: `{_utc_now()}`",
        "",
        "## Verdict",
        "",
        f"- Tasks scanned: **{len(records)}**",
        f"- Static candidates found: **{len(static_candidates)}**",
        f"- Full-AB/profile cuts: **{len(gated_cuts)}**",
        f"- Staged for captain review: **{len(staged)}**",
        f"- Total staged credited cost reduction: **{saved:,}**",
        f"- Total staged point gain estimate: **{point_gain:.4f}**",
        "",
        "The v1 pass set is deliberately conservative: `const_to_init`, "
        "`zero_pad_into_conv`, `add_chain_to_sum`, guarded `drop_cast_before_pad_bool`, "
        "and `opset_bump_pad_uint8`. Equal-tail re-synthesis, groups halving, "
        "no-Mul decode, and separable Einsum are left unstaged for v2 because they "
        "need task-structural guards beyond local ONNX shape/type checks.",
        "",
        "## Rewrite Payoff",
        "",
        "| Rewrite | Staged tasks containing rewrite | Credited saved on those tasks |",
        "|---|---:|---:|",
    ]
    if by_rewrite:
        for name, stats in sorted(by_rewrite.items()):
            lines.append(f"| `{name}` | {stats['tasks']} | {stats['profile_saved_on_tasks']:,} |")
    else:
        lines.append("| - | 0 | 0 |")

    lines += [
        "",
        "## Task Table",
        "",
        "| task | rewrite(s) applied | old_cost | new_cost | byte_equal | profile_gap | staged | notes |",
        "|---|---|---:|---:|---|---|---|---|",
    ]

    for r in sorted(records, key=lambda x: x.get("task_num", 0)):
        rewrites = ",".join(rw.get("name", "?") for rw in r.get("rewrites", [])) or "-"
        old_cost = (r.get("old_profile") or {}).get("profile_cost")
        if old_cost is None:
            old_cost = (r.get("old_score") or {}).get("cost")
        new_cost = (r.get("new_profile") or {}).get("profile_cost")
        if new_cost is None:
            new_cost = (r.get("new_score") or {}).get("cost")
        old_s = "" if old_cost is None else str(old_cost)
        new_s = "" if new_cost is None else str(new_cost)
        notes = str(r.get("notes") or "").replace("|", "/")
        if r.get("stage_dir"):
            notes += f" ({r['stage_dir']})"
        lines.append(
            f"| {r.get('task')} | `{rewrites}` | {old_s} | {new_s} | "
            f"{r.get('byte_equal', 'N')} | {r.get('profile_gap_label', 'not_run')} | "
            f"{r.get('staged', 'N')} | {notes} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=int, help="single task number")
    group.add_argument("--tasks", default="", help="task list/ranges, e.g. 1,6,30-35")
    group.add_argument("--sweep", action="store_true", help="sweep all boladi_workable=Y tasks")
    parser.add_argument(
        "--check",
        action="store_true",
        help="dry-run: report static potential cuts without staging (use --gate-in-check to run gates too)",
    )
    parser.add_argument("--gate-in-check", action="store_true", help="run full AB/profile gates in --check mode, but do not stage")
    parser.add_argument("--source", default="live", help="'live' submission.zip, 'root-first', a directory, or one ONNX path")
    parser.add_argument("--patterns", default=",".join(DEFAULT_PATTERNS), help="comma-separated pattern allowlist")
    parser.add_argument("--limit", type=int, default=0, help="limit number of tasks after selection")
    parser.add_argument("--profile-arcgen", type=int, default=100000, help="number of stored arc-gen examples to include in gate-5 profile")
    parser.add_argument("--profile-max-n", type=int, default=0, help="cap real-input profile sample count; 0 means no cap")
    parser.add_argument("--timeout", type=float, default=0.0, help="per subprocess timeout seconds; 0 means no timeout")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.patterns.split(",") if name.strip()]
    unknown = [name for name in names if name not in PATTERN_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown pattern(s): {unknown}; known={sorted(PATTERN_REGISTRY)}")
    patterns = [PATTERN_REGISTRY[name] for name in names]

    if args.task is not None:
        tasks = [args.task]
    elif args.tasks:
        tasks = _parse_task_list(args.tasks)
    else:
        tasks = _load_workable_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    timeout = args.timeout or None
    builder_args = " ".join(sys.argv[1:])
    records: list[dict[str, Any]] = []
    print(
        f"[peephole] tasks={len(tasks)} check={args.check} gate_in_check={args.gate_in_check} "
        f"source={args.source} patterns={names}",
        flush=True,
    )
    for i, task in enumerate(tasks, 1):
        t0 = time.time()
        try:
            rec = process_task(
                task,
                source=args.source,
                patterns=patterns,
                check=args.check,
                gate_in_check=args.gate_in_check,
                profile_arcgen=args.profile_arcgen,
                profile_max_n=args.profile_max_n,
                timeout=timeout,
                builder_args=builder_args,
            )
        except Exception as exc:
            rec = {
                "task": _task_id(task),
                "task_num": task,
                "rewrites": [],
                "byte_equal": "N",
                "staged": "N",
                "profile_gap_label": "not_run",
                "notes": f"process_task raised {type(exc).__name__}: {exc}",
            }
        records.append(rec)
        elapsed = time.time() - t0
        rewrites = ",".join(r.get("name", "?") for r in rec.get("rewrites", [])) or "-"
        status = rec.get("notes", "")
        print(
            f"  [{i:>3}/{len(tasks)}] {_task_id(task)} rewrites={rewrites} "
            f"byte_equal={rec.get('byte_equal')} staged={rec.get('staged')} ({elapsed:.1f}s) {status}",
            flush=True,
        )

    write_report(records, args.report)
    print(f"[peephole] wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
