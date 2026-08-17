#!/usr/bin/env python3
"""Pre-compute the recommended next submit stack from the picker cache.

The submit-captain agent currently spends ~11 minutes deciding which
candidate(s) to stack and submit. The decision is mostly mechanical:

  1. Sort ready candidates by expected EV.
  2. Greedily pick the top candidate whose ``changed_task_ids`` don't
     overlap with anyone already in the proposed stack.
  3. Stop at ``--max-stack-size`` (default 3) or when no more compatible
     candidates remain.
  4. Skip anything flagged in the picker cache as ``hidden_bad_task_hit``
     or ``failed_sha_hit``.
  5. Sanity-check that each candidate's staged package_path exists on
     disk and has plausible content.

This script writes the recommendation to
``.hermes/board/next_submit_recommendation.json``. Submit-captain reads
it as a HINT (not a directive) — the agent still owns the final call
and runs ``kaggle competitions submit`` itself. Reading a cached
recommendation should turn the 11-minute thinking phase into ~30 seconds
of "load JSON, verify, submit."

Input contract: ``submit_picker_cache.json`` produced by
``tools/neurogolf_submit_picker_cache.py`` (Tier 3). If the cache is
absent or stale, this script falls back to building it inline so the
output is always self-consistent.

Output schema: see ``build_recommendation()``.

Usage:
    python tools/neurogolf_stack_picker.py            # dry-run
    python tools/neurogolf_stack_picker.py --apply    # write file
    python tools/neurogolf_stack_picker.py --apply --json   # cron
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = REPO_ROOT / ".hermes" / "board"
CACHE_PATH = BOARD_DIR / "submit_picker_cache.json"
OUTPUT_PATH = BOARD_DIR / "next_submit_recommendation.json"

SCHEMA_VERSION = "neurogolf.next_submit_recommendation.v1"
DEFAULT_MAX_STACK_SIZE = 3
DEFAULT_MIN_STAND_ALONE_EV = 0.05  # below this, don't bother solo-submitting

# Expected entry count in the canonical submission.zip. The competition's
# submission ships task000.onnx .. task400.onnx (401 files including the
# task400 bonus). Verified empirically against the live canonical zip.
EXPECTED_ZIP_ENTRY_COUNT = 401


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_cache(cache_path: Path = CACHE_PATH) -> dict[str, Any]:
    """Read the picker cache. Fall back to building inline if absent so
    the recommendation is always self-consistent with current state."""
    doc = _read_json(cache_path)
    if isinstance(doc, dict) and doc.get("schema") == "neurogolf.submit_picker_cache.v1":
        return doc
    # Fall back to rebuilding inline so the recommendation isn't blocked
    # by a missing cache.
    from tools.neurogolf_submit_picker_cache import (  # local import to avoid cycle
        STATE_JSON,
        SUBMIT_QUEUE_JSON,
        CURRENT_BEST_MANIFEST,
        TASK_OUTCOME_JSON,
        build_cache,
    )
    return build_cache(
        queue_doc=_read_json(SUBMIT_QUEUE_JSON),
        state_doc=_read_json(STATE_JSON),
        manifest_doc=_read_json(CURRENT_BEST_MANIFEST),
        task_outcome_doc=_read_json(TASK_OUTCOME_JSON),
    )


def validate_package_path(package_path: str, changed_task_ids: list[str]) -> dict[str, Any]:
    """Light sanity check on a staged submit package. We don't byte-diff
    against the canonical baseline (that's submit-captain's job during
    actual submission) — we just confirm the file exists, is a valid zip
    with the expected entry count, and includes the changed task entries.
    """
    result: dict[str, Any] = {
        "ok": False,
        "exists": False,
        "is_zip": False,
        "entry_count": 0,
        "missing_changed_entries": [],
        "extra_entries": [],
        "messages": [],
    }
    if not package_path:
        result["messages"].append("no package_path")
        return result
    path = Path(package_path)
    if not path.exists():
        result["messages"].append(f"package_path does not exist: {package_path}")
        return result
    result["exists"] = True
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except zipfile.BadZipFile:
        result["messages"].append(f"package_path is not a valid zip: {package_path}")
        return result
    result["is_zip"] = True
    result["entry_count"] = len(names)
    if len(names) != EXPECTED_ZIP_ENTRY_COUNT:
        result["messages"].append(
            f"unexpected entry count: {len(names)} (expected {EXPECTED_ZIP_ENTRY_COUNT})"
        )
    expected_entries = {f"{tid}.onnx" for tid in changed_task_ids}
    have = set(names)
    missing = sorted(expected_entries - have)
    if missing:
        result["missing_changed_entries"] = missing
        result["messages"].append(f"missing changed entries: {missing}")
    result["ok"] = (
        result["exists"]
        and result["is_zip"]
        and result["entry_count"] == EXPECTED_ZIP_ENTRY_COUNT
        and not missing
    )
    return result


def pick_stack(
    safe_candidates: list[dict[str, Any]],
    *,
    max_stack_size: int = DEFAULT_MAX_STACK_SIZE,
    require_single_lineage: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy stack-compatibility pick. Returns (selected_stack, skipped_overflow).

    Two candidates can be stacked iff their ``changed_task_ids`` are
    disjoint sets. We sort safe_candidates by EV descending, then greedily
    accept candidates until we hit the stack-size cap or run out of
    compatible options. Skipped candidates are returned so the
    recommendation can explain "why not this one?"

    With ``require_single_lineage=True`` (default), the stack also requires
    every selected candidate to share the same ``lineage_base_candidate_id``.
    Multi-lineage stacks force submit-captain to rebase candidates onto the
    current canonical, and the observed result (2026-05-18) was a
    catastrophic -26.45 LB submit on the picker's own +4.28 EV recommendation
    when the 3 candidates had different lineage_bases. Submit-captain can
    still build cross-lineage stacks manually; the picker just won't recommend
    them by default.

    Algorithm with single-lineage on: group safe candidates by lineage_base,
    pick the highest-EV non-empty group, then run the disjoint-task pick
    within that group. Tie-break by combined EV.
    """
    skipped: list[dict[str, Any]] = []
    # Defensive re-sort
    ordered = sorted(safe_candidates, key=lambda c: (-(c.get("ev") or 0.0), c.get("candidate_id") or ""))

    if require_single_lineage:
        # Group by lineage_base_candidate_id; empty string is its own group
        # (legacy entries with no lineage info).
        groups: dict[str, list[dict[str, Any]]] = {}
        for c in ordered:
            key = c.get("lineage_base_candidate_id") or ""
            groups.setdefault(key, []).append(c)
        # For each group, do the disjoint-task pick and compute combined EV.
        best_picks: list[dict[str, Any]] = []
        best_ev: float = -1.0
        best_lineage: str = ""
        for lineage, group_cands in groups.items():
            picks, _ = _disjoint_task_pick(group_cands, max_stack_size=max_stack_size)
            group_ev = sum(c.get("ev") or 0 for c in picks)
            if group_ev > best_ev:
                best_ev = group_ev
                best_picks = picks
                best_lineage = lineage
        # Candidates outside the winning lineage become "skipped due to
        # cross-lineage exclusion" so the rationale can mention them.
        chosen_ids = {c.get("candidate_id") for c in best_picks}
        for c in ordered:
            if c.get("candidate_id") in chosen_ids:
                continue
            cand_lineage = c.get("lineage_base_candidate_id") or ""
            if cand_lineage != best_lineage:
                skipped.append({
                    **c,
                    "skip_reason": (
                        f"different lineage_base ({cand_lineage!r}) than the recommended "
                        f"stack ({best_lineage!r}); cross-lineage stacks need --allow-multi-lineage"
                    ),
                })
            else:
                # Same lineage but lost the disjoint-task race — fall through
                # to the standard overlap message
                skipped.append({
                    **c,
                    "skip_reason": "changed_task_ids overlap with already-selected candidate in the same lineage",
                })
        return best_picks, skipped

    # Multi-lineage allowed — run the legacy single-pass pick.
    return _disjoint_task_pick(ordered, max_stack_size=max_stack_size)


def _disjoint_task_pick(
    ordered_candidates: list[dict[str, Any]],
    *,
    max_stack_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The original disjoint-changed-task-ids greedy pick. Caller is
    responsible for pre-sorting by EV."""
    used_tasks: set[str] = set()
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cand in ordered_candidates:
        changed = set(cand.get("changed_task_ids") or [])
        if not changed:
            skipped.append({**cand, "skip_reason": "empty changed_task_ids"})
            continue
        overlap = changed & used_tasks
        if overlap:
            skipped.append({
                **cand,
                "skip_reason": f"changed_task_ids overlap with already-selected: {sorted(overlap)}",
            })
            continue
        selected.append(cand)
        used_tasks |= changed
        if len(selected) >= max_stack_size:
            break
    return selected, skipped


def build_recommendation(
    cache: dict[str, Any],
    *,
    max_stack_size: int = DEFAULT_MAX_STACK_SIZE,
    validate_packages: bool = True,
    require_single_lineage: bool = True,
    now_iso: str | None = None,
) -> dict[str, Any]:
    ready = cache.get("ready_candidates_by_ev") or []
    safe = [c for c in ready if c.get("safe_to_submit")]
    blocked = [c for c in ready if not c.get("safe_to_submit")]
    selected, skipped_overlap = pick_stack(
        safe, max_stack_size=max_stack_size,
        require_single_lineage=require_single_lineage,
    )
    combined_gain = round(sum(c.get("ev") or 0 for c in selected), 6)
    stack_ids = [c.get("candidate_id") for c in selected]
    stack_changed_tasks = sorted({
        t for c in selected for t in (c.get("changed_task_ids") or [])
    })
    package_validations: dict[str, dict[str, Any]] = {}
    if validate_packages:
        for cand in selected:
            cid = cand.get("candidate_id") or ""
            package_validations[cid] = validate_package_path(
                cand.get("package_path") or "",
                cand.get("changed_task_ids") or [],
            )
    all_packages_ok = (not validate_packages) or all(
        v.get("ok") for v in package_validations.values()
    )

    # Lineage compatibility: every candidate in the stack should be rebased
    # onto the same lineage_base. If they're not, the stack still works
    # (submit-captain rebases) but we flag it so the agent knows extra
    # validation is needed.
    lineage_bases = sorted({
        c.get("lineage_base_candidate_id") or "" for c in selected
    })
    lineage_consistent = len(lineage_bases) <= 1
    rationale_parts: list[str] = []
    if not selected:
        rationale_parts.append("No safe candidates available for submission.")
    else:
        rationale_parts.append(
            f"Picked {len(selected)}-candidate stack with combined expected gain "
            f"+{combined_gain} (top: +{selected[0].get('ev')})."
        )
        if not lineage_consistent:
            rationale_parts.append(
                f"WARNING: candidates have different lineage_base values "
                f"({lineage_bases}). Submit-captain must rebase onto current canonical."
            )
        if not all_packages_ok:
            bad = [cid for cid, v in package_validations.items() if not v.get("ok")]
            rationale_parts.append(f"Package validation FAILED for: {bad}. Inspect before submit.")
    if blocked:
        rationale_parts.append(
            f"{len(blocked)} ready candidate(s) excluded as hidden-bad or "
            f"already-failed-SHA — see picker cache for details."
        )
    if skipped_overlap:
        rationale_parts.append(
            f"{len(skipped_overlap)} ready candidate(s) skipped this round due to "
            f"stack-compatibility overlap with selected stack."
        )

    return {
        "schema": SCHEMA_VERSION,
        "computed_at": now_iso or _utc_now(),
        "based_on_cache_at": cache.get("computed_at"),
        "current_best": cache.get("current_best", {}),
        "recommendation": {
            "stack": stack_ids,
            "stack_size": len(stack_ids),
            "expected_combined_gain": combined_gain,
            "changed_task_ids": stack_changed_tasks,
            "lineage_bases": lineage_bases,
            "lineage_consistent": lineage_consistent,
            "package_validation_pass": all_packages_ok,
            "rationale": " ".join(rationale_parts) or "(no rationale generated)",
        },
        "selected_candidates": selected,
        "package_validations": package_validations,
        "blocked_for_hidden_bad_or_failed_sha": blocked,
        "skipped_due_to_stack_overlap": skipped_overlap,
        "max_stack_size": max_stack_size,
    }


def write_recommendation(rec: dict[str, Any], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the recommendation file (default: dry-run).")
    parser.add_argument("--json", action="store_true", help="Emit a compact JSON summary.")
    parser.add_argument("--max-stack-size", type=int, default=DEFAULT_MAX_STACK_SIZE)
    parser.add_argument("--no-validate-packages", action="store_true",
                        help="Skip on-disk package validation (useful in tests).")
    parser.add_argument("--allow-multi-lineage", action="store_true",
                        help=(
                            "Allow candidates with different lineage_base values in the same stack. "
                            "Default: refuse — multi-lineage stacks require submit-captain to rebase "
                            "and have caused observed -26.45 LB regressions (2026-05-18 task384+"
                            "task049+task011 had 3 different lineage_bases and the rebase failed)."
                        ))
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    cache = load_cache(args.cache_path)
    rec = build_recommendation(
        cache,
        max_stack_size=args.max_stack_size,
        validate_packages=not args.no_validate_packages,
        require_single_lineage=not args.allow_multi_lineage,
    )

    if args.apply:
        write_recommendation(rec, args.output)

    if args.json:
        # Compact one-line summary for cron --no-agent.
        summary = {
            "schema": SCHEMA_VERSION,
            "applied": args.apply,
            "computed_at": rec["computed_at"],
            "stack_size": rec["recommendation"]["stack_size"],
            "expected_combined_gain": rec["recommendation"]["expected_combined_gain"],
            "stack": rec["recommendation"]["stack"],
            "lineage_consistent": rec["recommendation"]["lineage_consistent"],
            "package_validation_pass": rec["recommendation"]["package_validation_pass"],
        }
        json.dump(summary, sys.stdout)
        sys.stdout.write("\n")
    else:
        print(f"stack: {rec['recommendation']['stack']}")
        print(f"expected_combined_gain: +{rec['recommendation']['expected_combined_gain']}")
        print(f"lineage_consistent: {rec['recommendation']['lineage_consistent']}")
        print(f"package_validation_pass: {rec['recommendation']['package_validation_pass']}")
        print(f"rationale: {rec['recommendation']['rationale']}")
        print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
