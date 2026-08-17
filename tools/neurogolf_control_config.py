#!/usr/bin/env python3
"""Shared NeuroGolf control-plane runtime configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from tools.neurogolf_paths import board_dir, resolve_canonical_root
except ModuleNotFoundError:  # pragma: no cover
    from neurogolf_paths import board_dir, resolve_canonical_root  # type: ignore


REPO_ROOT = resolve_canonical_root(Path(__file__).resolve())
BOARD_DIR = board_dir(REPO_ROOT)
CONTROL_CONFIG = BOARD_DIR / "control_config.json"

BASE_SPECIALIST_PROFILES = (
    "neurogolf-blocked-55",
    "neurogolf-board-analyst",
    "neurogolf-orchestrator",
    "neurogolf-submit-captain",
    "neurogolf-sync-manager",
)
DEFAULT_REVIEWER_PROFILES = (
    "neurogolf-reviewer",
    "neurogolf-reviewer-2",
)
DEFAULT_REVIEWER_MAX_ACTIVE = 1
DEFAULT_REVIEW_PACKET_LIMIT = 4


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def load_control_config(path: Path = CONTROL_CONFIG) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            doc = data
    reviewers = _as_str_list(doc.get("reviewer_profiles")) or list(DEFAULT_REVIEWER_PROFILES)
    try:
        max_active = int(doc.get("reviewer_max_active_per_profile") or DEFAULT_REVIEWER_MAX_ACTIVE)
    except (TypeError, ValueError):
        max_active = DEFAULT_REVIEWER_MAX_ACTIVE
    try:
        packet_limit = int(doc.get("review_packet_limit") or DEFAULT_REVIEW_PACKET_LIMIT)
    except (TypeError, ValueError):
        packet_limit = DEFAULT_REVIEW_PACKET_LIMIT
    return {
        "schema": "neurogolf.control_config.v1",
        **doc,
        "reviewer_profiles": reviewers,
        "reviewer_max_active_per_profile": max(1, max_active),
        "review_packet_limit": max(1, packet_limit),
    }


def reviewer_profiles(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    config = config or load_control_config()
    reviewers = _as_str_list(config.get("reviewer_profiles")) or list(DEFAULT_REVIEWER_PROFILES)
    return tuple(reviewers)


def primary_reviewer_profile(config: dict[str, Any] | None = None) -> str:
    profiles = reviewer_profiles(config)
    return profiles[0] if profiles else DEFAULT_REVIEWER_PROFILES[0]


def reviewer_max_active_per_profile(config: dict[str, Any] | None = None) -> int:
    config = config or load_control_config()
    try:
        return max(1, int(config.get("reviewer_max_active_per_profile") or DEFAULT_REVIEWER_MAX_ACTIVE))
    except (TypeError, ValueError):
        return DEFAULT_REVIEWER_MAX_ACTIVE


def review_packet_limit(config: dict[str, Any] | None = None) -> int:
    config = config or load_control_config()
    try:
        return max(1, int(config.get("review_packet_limit") or DEFAULT_REVIEW_PACKET_LIMIT))
    except (TypeError, ValueError):
        return DEFAULT_REVIEW_PACKET_LIMIT


def specialist_profiles(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(BASE_SPECIALIST_PROFILES + reviewer_profiles(config))


def is_reviewer_profile(assignee: str, config: dict[str, Any] | None = None) -> bool:
    return assignee in set(reviewer_profiles(config))
