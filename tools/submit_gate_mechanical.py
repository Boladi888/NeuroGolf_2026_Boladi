"""Mechanical verification layer for submit_gate.

Replaces the worker-bug-detection work the reviewer LLM used to do. Each check
is a pure function returning a list of failure strings (empty list = passed).
The orchestrator ``run_mechanical_checks`` runs them in order and aggregates
into a ``VerifyResult`` that ``submit_gate.classify_handoff`` consumes.

The slow checks (score_v2 reproduce, verify_against_live) accept injectable
callables so tests can stub them. The cheap structural checks (SHA, zip
entries, ONNX shape declaration) run for real even in unit tests — they're
deterministic and fast.

Failure messages are short, worker-readable strings — they're echoed verbatim
into the retry-card directive.
"""
from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import onnx

try:
    from tools.auto_triage_blocked import Handoff
except ImportError:  # auto_triage_blocked is private orchestration, withheld from the shared build
    class Handoff:  # used only as a type annotation here; this stand-in keeps the module importable
        """Stand-in for the private orchestration Handoff (shared build)."""
from tools.submit_gate import VerifyResult

# Type alias for the score function: takes ONNX file path → (macs, memory, params) or None
ScoreFn = Callable[[str], tuple[int, int, int] | None]
# Verify function: takes task_id + candidate ONNX path → (pass_count, total_count, max_abs_diff)
VerifyFn = Callable[[str, str], tuple[int, int, float]]


@dataclass
class MechanicalCheckInput:
    """Bundles all artifacts the mechanical layer needs.

    ``candidate_zip_path``: the worker's submission .zip.
    ``current_best_zip_path``: the live submissions/submission.zip we're
        diffing against (None when no current-best is staged yet — only the
        first-ever submission of a brand-new pipeline).
    ``task_id``: which task this handoff is for (used to locate the changed
        ONNX in both zips).
    """

    handoff: Handoff
    candidate_zip_path: Path
    current_best_zip_path: Path | None
    task_id: str


# ============================================================================
# Individual checks — each returns list[str] (empty = passed)
# ============================================================================


def check_zip_exists(zip_path: Path) -> list[str]:
    if not zip_path.exists():
        return [f"candidate zip not found at {zip_path}"]
    if not zip_path.is_file():
        return [f"candidate zip path is not a file: {zip_path}"]
    return []


def check_zip_sha(zip_path: Path, expected_sha: str) -> list[str]:
    """SHA-256 of the candidate zip matches the handoff claim.

    Failure here means either the worker reported the wrong SHA (sloppy) or
    the file changed on disk after the handoff (race / corruption). Either
    way, do NOT submit a SHA-mismatch package — Kaggle would attribute the
    score to this hash and we'd lose audit-trail integrity.
    """
    if not expected_sha:
        return ["handoff missing package_sha256 — cannot verify package integrity"]
    if not zip_path.exists():
        return []  # covered by check_zip_exists; don't double-report
    actual = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if actual.lower() != expected_sha.lower():
        return [
            f"package_sha256 mismatch: handoff claims {expected_sha.lower()[:16]}…, "
            f"zip on disk is {actual.lower()[:16]}…"
        ]
    return []


def check_package_structure(
    candidate_zip_path: Path,
    current_best_zip_path: Path | None,
    expected_changed: list[str],
) -> list[str]:
    """Single-task delta audit: the candidate zip must change exactly the
    entries declared in handoff.changed_package_entries.

    Without this, the task275 trap recurs: a worker hands off a "full 401-
    entry" zip claiming only ``task275.onnx`` changed, but some other
    entries silently differ vs current-best. Submitting that ships unknown
    changes to Kaggle and the SHA-attributed score is meaningless.

    When ``current_best_zip_path`` is None (no current-best staged), this
    check is skipped — the candidate IS the new current-best by definition.
    """
    if not expected_changed:
        return ["handoff missing changed_package_entries — cannot audit package"]
    try:
        with zipfile.ZipFile(candidate_zip_path) as candidate_zip:
            candidate_entries = {name: candidate_zip.read(name) for name in candidate_zip.namelist()}
    except (zipfile.BadZipFile, OSError) as exc:
        return [f"candidate zip unreadable: {exc}"]

    expected_set = set(expected_changed)
    candidate_names = set(candidate_entries.keys())

    # The declared changes must actually be in the zip.
    missing_declared = expected_set - candidate_names
    if missing_declared:
        return [f"changed_package_entries declared but missing from zip: {sorted(missing_declared)}"]

    # If no current-best to diff against, skip the cross-check.
    if current_best_zip_path is None or not current_best_zip_path.exists():
        return []

    try:
        with zipfile.ZipFile(current_best_zip_path) as base_zip:
            base_entries = {name: base_zip.read(name) for name in base_zip.namelist()}
    except (zipfile.BadZipFile, OSError) as exc:
        return [f"current-best zip unreadable: {exc}"]

    # Find what ACTUALLY differs between the two zips.
    actually_changed: set[str] = set()
    for name, payload in candidate_entries.items():
        base_payload = base_entries.get(name)
        if base_payload is None or base_payload != payload:
            actually_changed.add(name)
    for name in base_entries:
        if name not in candidate_entries:
            actually_changed.add(name)  # entry removed from candidate

    undeclared = actually_changed - expected_set
    if undeclared:
        return [
            f"zip differs vs current-best on entries not declared in changed_package_entries: "
            f"{sorted(undeclared)} (declared: {sorted(expected_set)})"
        ]
    return []


def check_raw_shape_parity(
    candidate_zip_path: Path,
    current_best_zip_path: Path | None,
    task_id: str,
) -> list[str]:
    """The task127 lesson encoded as a static check.

    Compare the declared output tensor shape of the candidate ONNX against
    the current-best ONNX for the same task. If they differ, refuse to submit
    — task127's -16.69 LB regression came from exactly this (no-pad candidate
    output [1,10,7,11] vs live [1,10,30,30]; visible+arc-gen decoded parity
    passed because the verifier cropped, but hidden eval used raw output).

    This is a static check on the ONNX header — no inference needed, so it
    runs in milliseconds. The deeper "compare actual outputs over 267 pairs"
    is verify_against_live's job.
    """
    task_entry = f"{task_id}.onnx"
    if current_best_zip_path is None or not current_best_zip_path.exists():
        return []

    try:
        candidate_shape = _read_onnx_output_shape(candidate_zip_path, task_entry)
    except _ShapeReadError as exc:
        return [f"candidate {task_entry} output shape unreadable: {exc}"]

    try:
        live_shape = _read_onnx_output_shape(current_best_zip_path, task_entry)
    except _ShapeReadError as exc:
        # Live missing the entry is a major anomaly but not a worker bug;
        # warn but don't block (could be a brand-new task addition).
        return [f"current-best {task_entry} output shape unreadable: {exc} — proceeding without parity check"]

    if candidate_shape != live_shape:
        return [
            f"raw output shape divergence vs live for {task_entry}: "
            f"candidate {candidate_shape} vs live {live_shape} "
            f"(task127 lesson — hidden eval uses raw output, not the decode-cropped view)"
        ]
    return []


def check_score_reproduces(
    candidate_zip_path: Path,
    task_id: str,
    expected_cost: int | None,
    *,
    score_fn: ScoreFn,
    tolerance: int = 0,
) -> list[str]:
    """Re-run score_network_v2 on the candidate ONNX and compare to handoff.

    ``expected_cost`` is the handoff's claimed candidate cost (memory + params).
    A mismatch means either the worker computed cost differently (different
    rules version) or referenced a different ONNX in their handoff. Submitting
    a cost-mismatched candidate breaks the lineage tracking in
    candidate_registry.json.

    ``tolerance`` allows small drift from non-deterministic memory profiling.
    Default 0 (exact match) — if we see false positives in production we can
    relax.
    """
    if expected_cost is None:
        return []  # nothing to compare to; not a failure
    task_entry = f"{task_id}.onnx"
    try:
        onnx_path = _extract_onnx_to_temp(candidate_zip_path, task_entry)
    except _ShapeReadError as exc:
        return [f"score_v2 reproduce skipped: {exc}"]
    try:
        result = score_fn(str(onnx_path))
    finally:
        try:
            onnx_path.unlink()
        except OSError:
            pass
    if result is None:
        return [f"score_v2 returned None for {task_entry} — likely an invalid ONNX"]
    macs, memory, params = result
    reproduced_cost = memory + params
    if abs(reproduced_cost - expected_cost) > tolerance:
        return [
            f"score_v2 reproduce disagrees with handoff for {task_entry}: "
            f"handoff cost={expected_cost} but score_v2 reports memory+params={reproduced_cost} "
            f"(macs={macs}, memory={memory}, params={params})"
        ]
    return []


def check_verify_against_live(
    candidate_zip_path: Path,
    current_best_zip_path: Path | None,
    task_id: str,
    *,
    verify_fn: VerifyFn | None,
    require_full_pass: bool = True,
) -> list[str]:
    """Run the equivalence test suite (visible + arc-gen pairs) against live.

    ``verify_fn`` should return ``(pass_count, total_count, max_abs_diff)``.
    When ``verify_fn`` is None (testing without the heavy ORT session), skip
    this check rather than failing — the caller decides whether to require it.

    ``require_full_pass=True`` (default) means anything less than total/total
    is a failure. Set to False to accept e.g. 265/267 if the small mismatch is
    a known noise issue.
    """
    if verify_fn is None:
        return []
    task_entry = f"{task_id}.onnx"
    try:
        candidate_path = _extract_onnx_to_temp(candidate_zip_path, task_entry)
    except _ShapeReadError as exc:
        return [f"verify_against_live skipped: {exc}"]
    try:
        passed, total, max_abs_diff = verify_fn(task_id, str(candidate_path))
    finally:
        try:
            candidate_path.unlink()
        except OSError:
            pass
    if total == 0:
        return [f"verify_against_live had 0 pairs for {task_id} — test data missing?"]
    if require_full_pass and passed < total:
        return [
            f"verify_against_live {passed}/{total} for {task_id} "
            f"(max_abs_diff={max_abs_diff:.6f}) — candidate diverges from live on "
            f"{total - passed} pair(s)"
        ]
    return []


# ============================================================================
# Top-level orchestrator
# ============================================================================


def run_mechanical_checks(
    inp: MechanicalCheckInput,
    *,
    score_fn: ScoreFn | None = None,
    verify_fn: VerifyFn | None = None,
    require_score_match: bool = True,
    require_full_verify: bool = True,
    score_tolerance: int = 0,
) -> VerifyResult:
    """Run all mechanical checks in dependency order, aggregate failures.

    Each check produces an independent failure string; we keep going past
    early failures so the worker sees ALL issues in one round-trip rather
    than discovering them one-at-a-time across 3 retries.

    Args:
        inp: bundled artifact paths + handoff.
        score_fn: callable for score_network_v2 (None to skip score check).
        verify_fn: callable for verify_against_live (None to skip verify check).
        require_score_match: when True, score_v2 disagreement is a fatal check.
        require_full_verify: when True, anything less than total/total fails.
        score_tolerance: integer cost-delta allowed in score reproduction.
    """
    failures: list[str] = []

    # Cheap checks first — if the zip is missing, the deeper checks would
    # crash. But we still run the SHA check below so the worker sees both
    # "missing" and "SHA wrong" in one pass when relevant.
    failures.extend(check_zip_exists(inp.candidate_zip_path))

    failures.extend(check_zip_sha(inp.candidate_zip_path, inp.handoff.package_sha))

    failures.extend(check_package_structure(
        inp.candidate_zip_path,
        inp.current_best_zip_path,
        inp.handoff.changed_entries,
    ))

    failures.extend(check_raw_shape_parity(
        inp.candidate_zip_path,
        inp.current_best_zip_path,
        inp.task_id,
    ))

    # Slow checks last so quick failures fail fast in CI.
    if require_score_match and score_fn is not None:
        expected_cost: int | None = None
        scores_block = inp.handoff.parsed.get("scores") if isinstance(inp.handoff.parsed, dict) else None
        if isinstance(scores_block, dict):
            raw = scores_block.get("candidate_cost")
            if isinstance(raw, (int, float)):
                expected_cost = int(raw)
        if expected_cost is None:
            # Fall back: try candidate.cost (alternative shape)
            candidate_block = inp.handoff.parsed.get("candidate") if isinstance(inp.handoff.parsed, dict) else None
            if isinstance(candidate_block, dict):
                raw = candidate_block.get("cost")
                if isinstance(raw, (int, float)):
                    expected_cost = int(raw)
        failures.extend(check_score_reproduces(
            inp.candidate_zip_path,
            inp.task_id,
            expected_cost,
            score_fn=score_fn,
            tolerance=score_tolerance,
        ))

    if verify_fn is not None:
        failures.extend(check_verify_against_live(
            inp.candidate_zip_path,
            inp.current_best_zip_path,
            inp.task_id,
            verify_fn=verify_fn,
            require_full_pass=require_full_verify,
        ))

    return VerifyResult(ok=not failures, failures=failures)


# ============================================================================
# Internal helpers
# ============================================================================


class _ShapeReadError(RuntimeError):
    """Raised when we can't get an output shape from an ONNX entry in a zip."""


def _read_onnx_output_shape(zip_path: Path, entry: str) -> tuple[int, ...]:
    """Return the declared output[0] shape from the ONNX inside ``zip_path``.

    Reads the entry's bytes, parses as ONNX, returns the first output's
    dim values. Dynamic dims (no dim_value set) come back as 0 — submit_gate
    treats matching 0s as matching dims (we're comparing declared shapes,
    not runtime shapes).
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if entry not in zf.namelist():
                raise _ShapeReadError(f"entry {entry} not in zip {zip_path.name}")
            payload = zf.read(entry)
    except zipfile.BadZipFile as exc:
        raise _ShapeReadError(f"bad zip {zip_path}: {exc}") from exc
    try:
        model = onnx.load_from_string(payload)
    except Exception as exc:  # noqa: BLE001
        raise _ShapeReadError(f"ONNX parse failed for {entry}: {exc}") from exc
    if not model.graph.output:
        raise _ShapeReadError(f"ONNX {entry} has no declared output")
    shape = tuple(
        dim.dim_value if dim.HasField("dim_value") else 0
        for dim in model.graph.output[0].type.tensor_type.shape.dim
    )
    return shape


def _extract_onnx_to_temp(zip_path: Path, entry: str) -> Path:
    """Extract one entry from a zip to a temp .onnx file. Caller must delete."""
    import tempfile
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if entry not in zf.namelist():
                raise _ShapeReadError(f"entry {entry} not in zip {zip_path.name}")
            payload = zf.read(entry)
    except zipfile.BadZipFile as exc:
        raise _ShapeReadError(f"bad zip {zip_path}: {exc}") from exc
    fd, name = tempfile.mkstemp(suffix=".onnx", prefix="submit_gate_")
    try:
        import os
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)
