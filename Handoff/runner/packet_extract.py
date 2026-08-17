"""Robust reviewer-verdict extractor for the NeuroGolf HandOff runner.

Integrated into Handoff/runner/ (Phase 1, 2026-06-30). Staging source: Boladi_Skillz/Instructor1/neurogolf_integration/packet_extract.py.

Tiers:
  1-2. strict, LLM-FREE: scan brace-balanced {...} blocks (fixes the RUN-16 truncation where the old
       `r"```json\\s*(\\{.*?\\})\\s*```"` regex stopped at the first '}' inside a nested captain_packet),
       json.loads each, and validate against ReviewVerdict. Deterministic, no cost, no provider.
  3.   OPTIONAL Instructor coercion, gated behind env (off by default): if the reviewer emitted broken/partial
       text, ask a cheap/local model to coerce it into a valid ReviewVerdict. Only fires on the malformed tail.

Drop-in: `extract_review_json(text)` keeps the runner's existing dict-returning signature, so the swap is
`from packet_extract import extract_review_json` in handoff_runner.py / handoff_runner_claude.py.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterator, Optional

from packet_models import CaptainPacket, ReviewVerdict

_FENCED = re.compile(r"```json\s*(\{.*\})\s*```", re.S)


def validate_packet(packet: Optional[dict]) -> Optional[CaptainPacket]:
    """Strict-validate a captain_packet dict (typed version of write_captain_packet's required check).
    Returns the model on success, None if missing/invalid — so the runner can route a SHIP-with-bad-packet
    to needs_packet_fix exactly as today."""
    if not isinstance(packet, dict):
        return None
    try:
        return CaptainPacket.model_validate(packet)
    except Exception:
        return None


def _balanced_objects(text: str) -> Iterator[str]:
    """Yield every top-level brace-balanced {...} substring (string/escape aware, nesting aware)."""
    depth = 0
    start: Optional[int] = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : i + 1]
                    start = None


def _candidates(text: str) -> list[str]:
    out: list[str] = []
    for m in _FENCED.finditer(text):
        out.extend(_balanced_objects(m.group(1)))  # balanced INSIDE the fence (handles nested packet)
    out.extend(_balanced_objects(text))             # and anywhere in the raw text
    return out


def extract_verdict(text: str, *, allow_llm: Optional[bool] = None) -> Optional[ReviewVerdict]:
    """Return a validated ReviewVerdict, or None. Tries the last verdict-bearing block first (the reviewer's
    final answer usually comes last)."""
    if not text:
        return None
    seen: set[str] = set()
    for block in reversed(_candidates(text)):
        if block in seen:
            continue
        seen.add(block)
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            try:
                return ReviewVerdict.model_validate(obj)
            except Exception:
                continue  # well-formed JSON but wrong shape — keep looking
    # tier 3: optional Instructor coercion
    if allow_llm is None:
        allow_llm = os.environ.get("INSTRUCTOR_EXTRACT") == "1"
    if allow_llm:
        return _llm_coerce(text)
    return None


def extract_review_json(text: str, *, allow_llm: Optional[bool] = None) -> Optional[dict]:
    """Drop-in replacement for the runner's old function: returns a plain dict (or None)."""
    v = extract_verdict(text, allow_llm=allow_llm)
    return v.model_dump(exclude_none=True) if v is not None else None


def _llm_coerce(text: str) -> Optional[ReviewVerdict]:
    """Phase-2 fallback. Requires `pip install instructor` + a provider in INSTRUCTOR_MODEL
    (e.g. 'groq/llama-3.1-8b-instant', 'openai/gpt-4o-mini', 'ollama/llama3'). Returns None if unavailable —
    so a half-configured env degrades to Phase-1 instead of crashing the runner."""
    model = os.environ.get("INSTRUCTOR_MODEL")
    if not model:
        return None
    try:
        import instructor
    except Exception:
        return None
    try:
        client = instructor.from_provider(model)
        return client.chat.completions.create(
            response_model=ReviewVerdict,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the reviewer's verdict as structured data. Copy values verbatim from the "
                        "text; do not invent fields. captain_packet is null unless verdict is SHIP."
                    ),
                },
                {"role": "user", "content": text[:20000]},
            ],
            max_retries=2,
        )
    except Exception:
        return None


if __name__ == "__main__":  # tiny manual check
    sample = '```json\n{"verdict":"SHIP","task":"task279","captain_packet":{"task":"task279",' \
             '"source_lane":"Codex","lane":"SAFE_BATCH_QUEUE","onnx":"x.onnx","cost":7903,"sha256":"abc",' \
             '"gates":{"ab_full":"byte_equal=True"}}}\n```'
    print(extract_review_json(sample, allow_llm=False))
