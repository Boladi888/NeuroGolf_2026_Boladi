#!/usr/bin/env python3
"""Sho-style safety checks for NeuroGolf ONNX models and submission bundles.

Exit codes:
  0 PASS
  1 WARN (for example, x86-required runtime integer Div/Mod)
  2 FAIL
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from ng_harness import safety


def _task(raw: str | None) -> int | None:
    if not raw:
        return None
    s = raw.strip().lower()
    if s.startswith("task"):
        s = s[4:]
    n = int(s)
    if not 1 <= n <= 400:
        raise argparse.ArgumentTypeError("task must be in 1..400")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NeuroGolf safety verifier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_model = sub.add_parser("model", help="scan one ONNX model")
    p_model.add_argument("--task", type=_task, default=None, help="task number, e.g. 355 or task355")
    p_model.add_argument("--cand", type=pathlib.Path, required=True, help="candidate ONNX path")
    p_model.add_argument("--dynamic-official", action="store_true", help="probe flagged integer denominators on official boards")
    p_model.add_argument("--ignore-input-denominator", action="store_true", help="do not flag direct denominator named input")
    p_model.add_argument("--no-load", action="store_true", help="skip ORT load-after-sanitize check")
    p_model.add_argument("--json", action="store_true", help="emit pretty JSON")
    p_model.add_argument("--compact-json", action="store_true", help="emit compact JSON")
    p_model.add_argument("--out", type=pathlib.Path, default=None, help="write JSON result to path")

    p_bundle = sub.add_parser("bundle", help="scan a submission zip")
    p_bundle.add_argument("--zip", type=pathlib.Path, required=True, help="submission zip path")
    p_bundle.add_argument("--dynamic-official", action="store_true", help="probe flagged integer denominators on official boards")
    p_bundle.add_argument("--ignore-input-denominator", action="store_true", help="do not flag direct denominator named input")
    p_bundle.add_argument("--no-load", action="store_true", help="skip ORT load-after-sanitize check")
    p_bundle.add_argument("--json", action="store_true", help="emit pretty JSON")
    p_bundle.add_argument("--compact-json", action="store_true", help="emit compact JSON")
    p_bundle.add_argument("--out", type=pathlib.Path, default=None, help="write JSON result to path")

    args = parser.parse_args(argv)
    if args.cmd == "model":
        result = safety.scan_model_path(
            args.cand,
            task=args.task,
            load=not args.no_load,
            dynamic_official=args.dynamic_official,
            include_input_denominator=not args.ignore_input_denominator,
        )
    else:
        result = safety.scan_bundle(
            args.zip,
            load=not args.no_load,
            dynamic_official=args.dynamic_official,
            include_input_denominator=not args.ignore_input_denominator,
        )

    text = safety.dumps(result, compact=args.compact_json)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.json or args.compact_json or not args.out:
        print(text)
    else:
        print(f"[ng_safety] {result['verdict']} wrote {args.out}")
    return safety.exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
