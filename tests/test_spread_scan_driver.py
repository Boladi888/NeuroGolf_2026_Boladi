"""Tests for the spread_scan driver + annotation + report generators.

These tests exercise the pure-logic parts (report rendering, .md
annotation, JSON shape) without invoking ORT or score_network_v2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.spread_annotate_notes import (
    SPREAD_BEGIN,
    SPREAD_END,
    _apply_block,
    _render_spread_block,
    annotate,
)
from tools.spread_master_report import build_report


def _sample_record(task_id: str = "task000") -> dict:
    return {
        "task_id": task_id,
        "scanned_at": "2026-05-18T18:00:00Z",
        "original": {"macs": 0, "memory": 9524, "params": 157, "cost": 9681, "points": 15.82208},
        "patterns": [
            {
                "name": "clip_mul_to_min",
                "category": "math-equivalent",
                "verdict": "no-op",
                "cost_delta": None,
                "notes": "no Clip[0,1] + Mul(mask) pairs detected",
                "extras": {},
            },
            {
                "name": "scalar_broadcast_ones",
                "category": "math-equivalent",
                "verdict": "applied",
                "cost_delta": -99,
                "notes": "replaced 1 uniform init(s) with scalars; ~99 fewer param elements",
                "extras": {},
            },
            {
                "name": "fp16_selective",
                "category": "selective",
                "verdict": "regression",
                "cost_delta": 67238,
                "notes": "fp16-converted; cost INCREASED by 67238 bytes",
                "extras": {},
            },
            {
                "name": "crop_to_size_advisory",
                "category": "advisory",
                "verdict": "advisory",
                "cost_delta": -32000,
                "notes": "max active grid is 10x10 but graph has 1 [...,30,30] intermediate",
                "extras": {"max_h": 10, "max_w": 10},
            },
        ],
    }


def test_render_spread_block_contains_required_sections():
    block = _render_spread_block(_sample_record(), "20260518")
    assert SPREAD_BEGIN in block
    assert SPREAD_END in block
    assert "Current canonical cost" in block
    # The savings panel.
    assert "scalar_broadcast_ones" in block
    assert "-99" in block
    # Advisory panel.
    assert "crop_to_size_advisory" in block
    # Regression warning.
    assert "Do NOT apply" in block
    assert "+67238" in block or "67238" in block


def test_apply_block_appends_when_no_existing_marker(tmp_path: Path):
    notes = tmp_path / "task000.md"
    notes.write_text("# task000\n\n## Rule\n\nDrop shadow.\n", encoding="utf-8")
    block = _render_spread_block(_sample_record(), "20260518")
    new = _apply_block(notes, block)
    assert "## Rule" in new  # original preserved
    assert SPREAD_BEGIN in new
    assert SPREAD_END in new


def test_apply_block_replaces_existing_in_place(tmp_path: Path):
    notes = tmp_path / "task000.md"
    initial = (
        "# task000\n\n## Rule\n\nDrop shadow.\n\n"
        f"{SPREAD_BEGIN}\nold content goes here\n{SPREAD_END}\n\n## After\n\nOther stuff.\n"
    )
    notes.write_text(initial, encoding="utf-8")
    block = _render_spread_block(_sample_record(), "20260518")
    new = _apply_block(notes, block)
    assert "old content goes here" not in new
    assert "scalar_broadcast_ones" in new
    assert "## After" in new  # post-block content preserved
    # Only ONE block in the file.
    assert new.count(SPREAD_BEGIN) == 1
    assert new.count(SPREAD_END) == 1


def test_annotate_writes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    notes_dir = tmp_path / "Task_by_Task"
    notes_dir.mkdir()
    (notes_dir / "task000.md").write_text("# task000\n\n## Rule\n\nDrop shadow.\n", encoding="utf-8")
    # We don't create task001.md — annotate should create it.

    scan_json = tmp_path / "spread_scan_20260518.json"
    scan_data = {
        "tasks": [_sample_record("task000"), _sample_record("task001")],
    }
    scan_json.write_text(json.dumps(scan_data), encoding="utf-8")

    monkeypatch.setattr("tools.spread_annotate_notes.TASK_NOTES_DIR", notes_dir)
    counts = annotate(scan_json)
    assert counts["updated"] == 1
    assert counts["created"] == 1
    t000 = (notes_dir / "task000.md").read_text(encoding="utf-8")
    t001 = (notes_dir / "task001.md").read_text(encoding="utf-8")
    assert SPREAD_BEGIN in t000
    assert SPREAD_BEGIN in t001
    assert "## Rule" in t000  # original content preserved


def test_master_report_renders_headline_table_and_savings(tmp_path: Path):
    scan_json = tmp_path / "spread_scan_20260518.json"
    scan_data = {
        "tasks": [
            _sample_record("task000"),
            _sample_record("task001"),
        ],
    }
    scan_json.write_text(json.dumps(scan_data), encoding="utf-8")
    md = build_report(scan_json)
    assert "Headline" in md
    assert "Pattern verdict breakdown" in md
    assert "scalar_broadcast_ones" in md
    # 2 tasks, each saves 99 bytes -> 198 total upper bound.
    assert "198" in md
    # Both tasks should appear in top-N.
    assert "task000" in md
    assert "task001" in md
    # When no polish-sweep CSV is given, headline labels output as upper bound.
    assert "UPPER BOUND" in md or "upper-bound" in md.lower() or "OVERSTATES" in md


def test_master_report_uses_polish_csv_when_provided(tmp_path: Path):
    """Annotator + master report should prefer sequential-pipeline truth
    over the per-pattern upper bound when a polish_sweep CSV is given."""
    scan_json = tmp_path / "spread_scan_20260518.json"
    scan_data = {
        "tasks": [_sample_record("task000"), _sample_record("task001")],
    }
    scan_json.write_text(json.dumps(scan_data), encoding="utf-8")
    polish_csv = tmp_path / "polish_sweep_20260518.csv"
    polish_csv.write_text(
        "task_id,orig_cost,polished_cost,delta,applied_passes,verify_failed_passes\n"
        "task000,9681,9580,-101,onnxsim+scalar_broadcast_ones,\n"
        "task001,9681,9681,0,,\n",
        encoding="utf-8",
    )
    md = build_report(scan_json, polish_csv=polish_csv)
    # The true sequential delta on task000 should be reflected.
    assert "9,580" in md or "9580" in md
    # Headline label changes when polish CSV is available.
    assert "sequential polish pipeline" in md
    # Real recoverable bytes should be 101 (only task000 saved anything).
    assert "**101**" in md or "101" in md
    # Per-pattern "upper-bound" still appears for context.
    assert "upper-bound" in md.lower() or "OVERSTATES" in md


def test_annotator_uses_polish_csv_when_provided(tmp_path: Path, monkeypatch):
    notes_dir = tmp_path / "Task_by_Task"
    notes_dir.mkdir()
    (notes_dir / "task000.md").write_text("# task000\n", encoding="utf-8")
    scan_json = tmp_path / "spread_scan_20260518.json"
    scan_json.write_text(
        json.dumps({"tasks": [_sample_record("task000")]}),
        encoding="utf-8",
    )
    polish_csv = tmp_path / "polish_sweep_20260518.csv"
    polish_csv.write_text(
        "task_id,orig_cost,polished_cost,delta,applied_passes,verify_failed_passes\n"
        "task000,9681,9580,-101,onnxsim+scalar_broadcast_ones,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("tools.spread_annotate_notes.TASK_NOTES_DIR", notes_dir)
    annotate(scan_json, polish_csv=polish_csv)
    text = (notes_dir / "task000.md").read_text(encoding="utf-8")
    # Sequential-pipeline truth must be quoted explicitly.
    assert "sequential pipeline" in text.lower() or "polish_candidate" in text
    assert "101 bytes" in text
    # The upper-bound caveat should be present alongside the per-pattern table.
    assert "overlap" in text.lower() or "overstate" in text.lower()
