"""Tests for tools/algorithm_family_directive.py.

Pure-function module — every branch is exercised here so submit_gate.py can
assume canonical-form strings throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.algorithm_family_directive import (
    build_directive,
    families_already_tried,
    normalize_family,
)


# ----- normalize_family -----------------------------------------------------


class TestNormalizeFamily:
    def test_already_canonical_passthrough(self):
        assert normalize_family("where-select") == "where-select"

    def test_case_normalized(self):
        assert normalize_family("Where-Select") == "where-select"
        assert normalize_family("WHERE-SELECT") == "where-select"

    def test_underscore_collapsed_to_dash(self):
        """Workers might naturally write snake_case; treat it as the same family
        as kebab-case so the blacklist intuitively de-duplicates."""
        assert normalize_family("where_select") == "where-select"

    def test_whitespace_collapsed_to_dash(self):
        assert normalize_family("where select") == "where-select"
        assert normalize_family("where  select") == "where-select"

    def test_leading_trailing_whitespace_stripped(self):
        assert normalize_family("  where-select  ") == "where-select"

    def test_empty_string_maps_to_unspecified(self):
        assert normalize_family("") == "unspecified"

    def test_whitespace_only_maps_to_unspecified(self):
        assert normalize_family("   ") == "unspecified"

    def test_none_maps_to_unspecified(self):
        assert normalize_family(None) == "unspecified"

    def test_non_string_coerced(self):
        """Workers writing JSON might accidentally send a number or bool —
        coerce via str() rather than crashing."""
        assert normalize_family(42) == "42"
        assert normalize_family(True) == "true"

    def test_punctuation_collapsed(self):
        assert normalize_family("where.select") == "where-select"
        assert normalize_family("where/select") == "where-select"
        assert normalize_family("where+select") == "where-select"

    def test_alphanumeric_preserved(self):
        assert normalize_family("fp16-conv-stamp") == "fp16-conv-stamp"
        assert normalize_family("v2-rowcol") == "v2-rowcol"

    def test_truncated_at_40_chars(self):
        result = normalize_family("a" * 60)
        assert len(result) <= 40
        assert result == "a" * 40

    def test_truncation_strips_trailing_dashes(self):
        """If truncating lands exactly on a dash, strip it so the result still
        looks like a clean identifier."""
        # 39 'a's, then a dash at position 40, then more
        raw = ("a" * 39) + "-" + ("b" * 30)
        result = normalize_family(raw)
        assert not result.endswith("-")
        assert len(result) <= 40

    def test_only_punctuation_maps_to_unspecified(self):
        """A string that's nothing but dashes/dots after normalization."""
        assert normalize_family("---") == "unspecified"
        assert normalize_family("...") == "unspecified"

    def test_leading_trailing_dashes_stripped_after_collapse(self):
        assert normalize_family("-where-select-") == "where-select"

    def test_idempotent(self):
        """normalize(normalize(x)) == normalize(x) for any input."""
        for raw in ["Where_Select", "  WHERE.SELECT  ", "where-select", "", None]:
            once = normalize_family(raw)
            twice = normalize_family(once)
            assert twice == once


# ----- families_already_tried ----------------------------------------------


class TestFamiliesAlreadyTried:
    def test_empty_input(self):
        assert families_already_tried([]) == []
        assert families_already_tried(None) == []

    def test_dedupes_after_normalization(self):
        """The user's task275 case: workers write the same family with
        different cases / underscores — should collapse to one entry."""
        result = families_already_tried(["where-select", "Where-Select", "where_select"])
        assert result == ["where-select"]

    def test_preserves_order_of_first_occurrence(self):
        result = families_already_tried(["a", "b", "a", "c"])
        assert result == ["a", "b", "c"]

    def test_normalizes_each_entry(self):
        result = families_already_tried(["Where-Select", "Reduce_Broadcast"])
        assert result == ["where-select", "reduce-broadcast"]

    def test_unspecified_counted_once(self):
        result = families_already_tried(["", "  ", None, "unspecified"])
        assert result == ["unspecified"]


# ----- build_directive ------------------------------------------------------


class TestBuildDirective:
    def test_includes_attempt_index(self):
        text = build_directive([], retry_attempt_index=2)
        assert "Attempt index: **2** of 3" in text

    def test_includes_reason_when_provided(self):
        text = build_directive(
            ["where-select"],
            retry_attempt_index=2,
            reason="predicted gain 0.10 < 0.20",
        )
        assert "predicted gain 0.10 < 0.20" in text

    def test_omits_reason_block_when_blank(self):
        text = build_directive([], retry_attempt_index=1, reason="")
        assert "Why this is a retry:" not in text

    def test_renders_empty_blacklist_as_none_yet(self):
        """First retry — no families tried yet."""
        text = build_directive([], retry_attempt_index=1)
        assert "(none yet)" in text

    def test_lists_tried_families_inline(self):
        text = build_directive(["where-select", "reduce-broadcast"], retry_attempt_index=2)
        assert "`where-select`" in text
        assert "`reduce-broadcast`" in text

    def test_mentions_example_families(self):
        """The directive should give the worker concrete starting points,
        not just 'try something different'."""
        text = build_directive([], retry_attempt_index=1)
        # Examples we documented should be present
        for example in ["where-select", "reduce-broadcast", "conv-stamp", "matmul-route", "slice-pad"]:
            assert example in text

    def test_explains_unspecified_handoff_path(self):
        """If the worker genuinely can't find a new family, they should know
        the no-candidate handoff path is allowed."""
        text = build_directive([], retry_attempt_index=3)
        assert "candidate_id: null" in text
        assert "unspecified" in text

    def test_mentions_task_will_defer_after_3(self):
        text = build_directive([], retry_attempt_index=3)
        assert "defer to the next pass after 3" in text or "3" in text
