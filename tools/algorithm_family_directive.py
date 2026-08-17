"""Build the 'try a different algorithm family' directive for retry cards.

Workers self-report their `algorithm_family` in the handoff JSON (a short
string identifier like ``where-select``, ``reduce-broadcast``, ``conv-stamp``).
``tools/submit_gate.py`` accumulates these in ``task_outcome_registry.json``
per-pass and, when it issues a retry, calls into this module to produce the
human-readable directive embedded in the new worker card body.

Design choices:
- **Self-report, not classify.** No LLM classification of the worker's ONNX —
  cheaper, simpler, and the worker has the strongest model of what they tried.
- **Normalize aggressively.** ``"Where-Select"``, ``"where_select"``, and
  ``"  where-select  "`` all collapse to ``"where-select"`` so the
  "don't repeat" set behaves intuitively.
- **Empty/missing maps to ``"unspecified"``.** The blacklist grows monotonically
  even when workers forget to label, which means the retry directive can
  legitimately say "you tried unspecified — pick a real family this time".
"""
from __future__ import annotations

import re
from typing import Iterable

# Lowercase, dashes, alphanumerics. Anything else gets collapsed to a dash.
_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")
_MAX_FAMILY_LEN = 40
_UNSPECIFIED = "unspecified"


def normalize_family(raw: str | None) -> str:
    """Normalize a worker-reported family string to the canonical form.

    Examples (input -> output):
        ``"Where-Select"`` -> ``"where-select"``
        ``"where_select"`` -> ``"where-select"``
        ``"  where  select  "`` -> ``"where-select"``
        ``""`` -> ``"unspecified"``
        ``None`` -> ``"unspecified"``
        ``"a-very-long-name-that-keeps-going-and-going-and-going..."`` ->
            truncated to 40 chars

    The canonical form is what gets stored in ``task_outcome_registry``'s
    ``attempted_families`` list and compared against on retry. Always call
    this before reading the registry or before storing a new family.
    """
    if raw is None:
        return _UNSPECIFIED
    text = str(raw).strip().lower()
    if not text:
        return _UNSPECIFIED
    # Collapse any run of non-alphanumerics to a single dash, then strip
    # leading/trailing dashes that result.
    canonical = _NORMALIZE_PATTERN.sub("-", text).strip("-")
    if not canonical:
        return _UNSPECIFIED
    if len(canonical) > _MAX_FAMILY_LEN:
        canonical = canonical[:_MAX_FAMILY_LEN].rstrip("-")
        if not canonical:
            return _UNSPECIFIED
    return canonical


def build_directive(
    families_tried: Iterable[str],
    *,
    retry_attempt_index: int,
    reason: str = "",
) -> str:
    """Produce the directive text that goes into the retry card body.

    Args:
        families_tried: Already-normalized family strings from
            ``task_outcome_registry[task].attempted_families``.
        retry_attempt_index: Which attempt this is for the new worker
            (so they can write it back into their handoff for audit).
        reason: Optional one-line reason from submit_gate (e.g. "mechanical
            fail: SHA mismatch" or "predicted gain 0.10 < 0.20"). Surfaced
            verbatim so the worker can see WHY they're being asked to retry.

    Returns:
        Multi-line markdown that the kanban card body can embed directly.
    """
    families_list = list(families_tried) or []
    blacklist_str = ", ".join(f"`{f}`" for f in families_list) if families_list else "(none yet)"
    lines = [
        "## Retry directive — try a different algorithm family",
        "",
        f"Attempt index: **{retry_attempt_index}** of 3 this pass.",
    ]
    if reason:
        lines.extend([
            "",
            f"Why this is a retry: {reason}",
        ])
    lines.extend([
        "",
        f"Families already tried this pass: {blacklist_str}",
        "",
        "Try a **materially different** approach. Examples of distinct families:",
        "",
        "- `where-select` (Where + boolean selectors over channel dims)",
        "- `reduce-broadcast` (ReduceSum/Max + broadcast against a template)",
        "- `conv-stamp` (1x1 / 3x3 Conv with a stamp-pattern kernel)",
        "- `matmul-route` (MatMul-based column/row selection)",
        "- `slice-pad` (Slice the active grid, do work small, Pad back to 30x30)",
        "- `gather-route` (Gather/ScatterND for explicit indexing)",
        "- `cast-collapse` (dtype trick: drop a Cast that the runtime ignores)",
        "",
        "Different reduction pattern, different convolution structure, different",
        "output-channel routing — your call. Set the new `algorithm_family` in your",
        "handoff JSON to a short kebab-case identifier (max 40 chars, lowercase",
        "alphanumeric + dashes) so this loop can track what you tried.",
        "",
        "If you genuinely cannot find a materially-different approach on this task,",
        "hand off with `candidate_id: null` and `algorithm_family: \"unspecified\"`",
        "and a one-line `risk_notes` entry explaining why. submit_gate will count it",
        "as a consumed attempt and the task will defer to the next pass after 3.",
    ])
    return "\n".join(lines)


def families_already_tried(
    families: Iterable[str] | None,
) -> list[str]:
    """Return the normalized, de-duplicated family list for a task this pass.

    Convenience for the autoforeman / submit_gate when reading back the
    registry — strips duplicates that may have crept in from manual edits
    and ensures every entry is in canonical form before being shown to a
    worker.
    """
    seen: list[str] = []
    for raw in (families or ()):
        canonical = normalize_family(raw)
        if canonical not in seen:
            seen.append(canonical)
    return seen
