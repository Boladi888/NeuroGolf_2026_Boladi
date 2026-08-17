#!/usr/bin/env python3
"""Polish a single candidate ONNX with every math-equivalent pass.

Usage::

    .venv/Scripts/python.exe tools/polish_candidate.py path/to/candidate.onnx
    .venv/Scripts/python.exe tools/polish_candidate.py candidate.onnx -o polished.onnx
    .venv/Scripts/python.exe tools/polish_candidate.py candidate.onnx --json

Workflow (per `Logs/lessons.md` 2026-05-12 "standard optimization pipeline"):

  1. onnxsim.simplify    (deeper algebraic + constant folding)
  2. onnx_opt_safe       (dedupe + CSE + DCE + algebraic identities)
  3. onnx_opt_extended   (18 elimination passes)
  4. const_to_init       (Constant nodes → graph.initializer)
  5. clip_mul_to_min     (Clip[0,1] + Mul(0/1 mask) → Min)
  6. scalar_broadcast_ones (uniform float init → scalar)
  7. slice_axis_collapse   (drop axis-0 no-op from Slice inits)

Each step is rolled back if it fails byte-exact verification on 3 dummy
inputs (matches ``spread_scan`` policy). The final polished ONNX is
written next to the input with suffix ``_polished.onnx`` unless
``--output`` is given.

This tool is intended to be called by the submit-captain / worker review
gate BEFORE shipping a candidate to Kaggle. The math-equivalent passes
preserve raw tensor outputs exactly (the Kaggle hidden scorer grades raw
tensor values per the 2026-05-12 DIAG lessons), so they're guaranteed
safe — but cumulative savings are non-trivial.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Use the same pattern set + verifier as spread_scan.
from tools.spread_patterns import iter_patterns  # noqa: E402
from tools.spread_scan import (  # noqa: E402
    DEFAULT_VERIFY_INPUTS,
    VERIFY_TOLERANCE,
    _score_model,
    _verify_byte_exact,
)

# Patterns to apply, in canonical order. Excludes advisory + selective.
DEFAULT_PIPELINE = (
    "onnxsim",
    "safe_opt",
    "const_to_init",
    "clip_mul_to_min",
    "scalar_broadcast_ones",
    "slice_axis_collapse",
)


def polish(
    model: onnx.ModelProto,
    *,
    work_dir: Path,
    pipeline: tuple[str, ...] = DEFAULT_PIPELINE,
    n_verify_inputs: int = DEFAULT_VERIFY_INPUTS,
    quiet: bool = False,
) -> dict[str, Any]:
    """Apply every pass in ``pipeline`` to ``model``, verifying each step.
    Returns ``{"final_model", "steps": [...]}`` with per-step info.
    """
    patterns_by_name = {p.name: p for p in iter_patterns()}
    current = onnx.ModelProto()
    current.CopyFrom(model)
    orig_score = _score_model(current, work_dir=work_dir)
    steps: list[dict[str, Any]] = []
    for name in pipeline:
        pat = patterns_by_name.get(name)
        if pat is None:
            steps.append({"pass": name, "verdict": "missing", "delta": None})
            continue
        try:
            result = pat.apply(current, "candidate")
        except Exception as exc:  # noqa: BLE001
            steps.append({"pass": name, "verdict": "error", "notes": f"{type(exc).__name__}: {exc}"})
            continue
        if result.verdict != "applied" or result.new_model is None:
            steps.append({"pass": name, "verdict": result.verdict, "notes": result.notes})
            continue
        ok, why = _verify_byte_exact(
            current,
            result.new_model,
            work_dir=work_dir,
            n_inputs=n_verify_inputs,
        )
        if not ok:
            steps.append({"pass": name, "verdict": "verify-failed", "notes": why})
            continue
        new_score = _score_model(result.new_model, work_dir=work_dir)
        if new_score is None:
            steps.append({"pass": name, "verdict": "score-failed", "notes": "scorer crashed on result"})
            continue
        prev_cost = (
            steps[-1]["cost"] if (steps and "cost" in steps[-1])
            else (orig_score or {}).get("cost", 0) or 0
        )
        delta = int(new_score["cost"]) - int(prev_cost)
        if delta >= 0:
            steps.append({
                "pass": name,
                "verdict": "no-savings",
                "delta": delta,
                "cost": int(new_score["cost"]),
                "notes": result.notes,
            })
            continue
        # Adopt the new model.
        current = result.new_model
        steps.append({
            "pass": name,
            "verdict": "applied",
            "delta": delta,
            "cost": int(new_score["cost"]),
            "notes": result.notes,
        })
        if not quiet:
            print(f"  {name}: {delta:+d} bytes -> {new_score['cost']}")
    final_score = _score_model(current, work_dir=work_dir)
    return {
        "final_model": current,
        "original_score": orig_score,
        "final_score": final_score,
        "steps": steps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the candidate ONNX")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: <input stem>_polished.onnx)")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    parser.add_argument("--verify-inputs", type=int, default=DEFAULT_VERIFY_INPUTS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    model = onnx.load(str(args.input))
    work_dir = Path(tempfile.mkdtemp(prefix="polish_"))
    out = polish(model, work_dir=work_dir, n_verify_inputs=args.verify_inputs, quiet=args.quiet)
    final = out["final_model"]
    out_path = args.output or args.input.with_name(args.input.stem + "_polished.onnx")
    onnx.save(final, str(out_path))
    summary = {
        "input": str(args.input),
        "output": str(out_path),
        "original_cost": (out["original_score"] or {}).get("cost"),
        "final_cost": (out["final_score"] or {}).get("cost"),
        "steps": out["steps"],
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        orig = summary["original_cost"]
        final_cost = summary["final_cost"]
        if orig is not None and final_cost is not None:
            saved = int(orig) - int(final_cost)
            print(f"original cost: {orig}")
            print(f"polished cost: {final_cost}  ({'-' if saved >= 0 else '+'}{abs(saved)} bytes)")
        applied_steps = [s for s in summary["steps"] if s["verdict"] == "applied"]
        if applied_steps:
            print("applied:")
            for s in applied_steps:
                print(f"  - {s['pass']}: {s['delta']:+d} bytes")
        skipped_steps = [s for s in summary["steps"] if s["verdict"] in ("no-op", "no-savings")]
        if skipped_steps:
            print(f"no-op: {', '.join(s['pass'] for s in skipped_steps)}")
        failed_steps = [s for s in summary["steps"] if s["verdict"] in ("verify-failed", "error", "score-failed")]
        if failed_steps:
            print("failed (kept original):")
            for s in failed_steps:
                print(f"  - {s['pass']}: {s['verdict']} - {s.get('notes', '')[:80]}")
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
