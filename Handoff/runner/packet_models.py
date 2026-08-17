"""Pydantic models for the NeuroGolf reviewer verdict + Submit-Captain packet.

Integrated into Handoff/runner/ (Phase 1, 2026-06-30). Staging source: Boladi_Skillz/Instructor1/neurogolf_integration/packet_models.py.
Schema source: Agenters/Submit_Captain/PACKET_SCHEMA.md + Handoff/runner/prompts/reviewer.md.
Requires pydantic v2 (which Instructor also requires). No LLM here — pure schema/validation.

The constraints intentionally mirror the runner's "truthy required" check
(`if any(not packet.get(k) for k in required)`): empty strings and cost==0 are INVALID, exactly as
write_captain_packet() / captain_ctl.classify() would reject them — so a packet that validates here is one
the downstream gate will accept on shape.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Gates(BaseModel):
    """Self-reported gate results (free-form strings as the worker prints them)."""
    model_config = ConfigDict(extra="allow")
    visible: Optional[str] = None
    truth_full: Optional[str] = None
    ab_full: Optional[str] = None
    profile_real_cost: Optional[str] = None


class CaptainPacket(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate future / probe-written keys

    # --- required (captain_ctl.classify + PACKET_SCHEMA); non-empty / cost>0 to match the runner gate ---
    task: str = Field(min_length=1)
    source_lane: str = Field(min_length=1)   # "Codex" | "Claude" — captain_ctl does the authoritative case/lane check
    lane: str = Field(min_length=1)          # SAFE_BATCH_QUEUE | RISK_SINGLE_PROBE | SAFE_FALLBACK_FOR_SPREAD
    onnx: str = Field(min_length=1)
    cost: int = Field(gt=0)                  # 0 is a reviewer-template placeholder and is rejected by the gate
    sha256: str = Field(min_length=1)

    # --- strongly recommended ---
    profile_cost: Optional[int] = None
    reviewer_verdict: Optional[str] = "SHIP"
    gates: Optional[Gates] = None
    msg: Optional[str] = None
    lesson: Optional[str] = None
    source_manifest: Optional[str] = None
    task_note: Optional[str] = None

    # --- unambiguous delta fields (preferred; see the ONNX win-rate action reports) ---
    # `expected_delta` was overloaded across lanes (cost bytes on some packets,
    # scoreboard points on others, sentinels on others) so Captain/alpha had to
    # infer units from prose. Prefer the explicit fields below; the runner also
    # derives `cost_delta` from live_cost/candidate_profile_cost when it can.
    live_cost: Optional[int] = None               # live team-best cost the reviewer re-read
    candidate_profile_cost: Optional[int] = None  # candidate's profiled cost
    cost_delta: Optional[int] = None              # candidate_profile_cost - live_cost (NEGATIVE = bytes saved; same sign as expected_delta & the staged-manifest cost_delta)
    points_delta: Optional[float] = None          # expected scoreboard points change (POSITIVE = more points; 0.0 if unknown)
    live_sha256: Optional[str] = None
    candidate_sha256: Optional[str] = None
    batch_id: Optional[str] = None                # stamped by the runner from workflow.batch_id
    runner_id: Optional[str] = None               # stamped by the runner from workflow.runner_id
    expected_delta: Optional[float] = None        # DEPRECATED: ambiguous units; use cost_delta / points_delta

    # --- probe-written downstream (NOT produced by the reviewer; optional) ---
    probe_ok: Optional[bool] = None
    probe_delta: Optional[float] = None
    probe_sha: Optional[str] = None
    candidate_status: Optional[str] = None


class ReviewVerdict(BaseModel):
    """The reviewer's top-level output. captain_packet is null unless verdict == SHIP — but we keep it
    Optional even for SHIP so the runner can handle 'SHIP but packet missing/invalid' as needs_packet_fix
    (rather than the model hard-rejecting it)."""
    model_config = ConfigDict(extra="allow")

    verdict: str = Field(min_length=1)        # SHIP | KEEP_WORKING | DROP (normalized by the runner)
    task: Optional[str] = None
    rework_prompt: str = ""
    attack_log: str = ""
    # kept RAW (not the strict CaptainPacket) on purpose: a SHIP with a malformed packet must still surface
    # verdict=SHIP so the runner routes it to needs_packet_fix. Validate the packet separately with
    # packet_extract.validate_packet() before staging.
    captain_packet: Optional[dict] = None

    @field_validator("rework_prompt", "attack_log", mode="before")
    @classmethod
    def _none_to_empty(cls, v: object) -> object:
        # A reviewer shipping a SHIP commonly emits `null` for the now-unused string fields (the prompt
        # primes `captain_packet: null`, and models generalize that to rework_prompt/attack_log). These are
        # bare `str`, so an explicit null would RAISE here — regressing a good SHIP to review_parse_failed,
        # which the old json.loads path accepted. Coerce None -> "" (an ABSENT key already uses the ""
        # default; this only catches the explicit-null case) so Phase 1 stays a true superset of the old
        # parser for valid SHIPs.
        return "" if v is None else v
