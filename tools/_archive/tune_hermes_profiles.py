#!/usr/bin/env python3
"""Apply NeuroGolf-tuned settings to all Hermes profile config.yaml files.

Idempotent — re-running is safe. Each change rewrites a single line via regex.
Backups (``config.yaml.bak_pre_neurogolf_tune``) are written once per profile.

Profiles tuned (15):
    neurogolf-worker-1..8, neurogolf-reviewer, neurogolf-reviewer-2,
    neurogolf-submit-captain, neurogolf-sync-manager, neurogolf-board-analyst,
    neurogolf-orchestrator, neurogolf-blocked-55

The 11 knobs, with source-verified semantics:

  agent.max_turns: 150 -> 250
      Total LLM turns per task. Bumping gives hard ARC tasks room to iterate.

  agent.reasoning_effort: medium -> high
      o3/gpt-5.5 reasoning-token budget per turn. The biggest dial for
      candidate quality on ARC.

  code_execution.max_tool_calls: 50 -> 200
      Cap on code_execution invocations per task. Workers running 100+
      Python iterations hit this otherwise.

  code_execution.timeout: 300 -> 600
      Per-code_execution call timeout. Some ONNX builds + onnx-tool checks
      take 4-8 min.

  kanban.failure_limit: 2 -> 3
      Crashes before dispatcher gives up. Adds one retry for transient
      issues like API rate-limits.

  terminal.timeout: 180 -> 600
      Per-shell-command timeout. Long verify_against_live runs need
      multi-minute headroom.

  terminal.lifetime_seconds: 300 -> 1800
      Idle timeout for persistent shells. Source: tools/terminal_tool.py
      :1257 — shell cleanup if no commands in window. Deep-thinking phases
      with no shell interaction would lose env state otherwise.

  tool_output.max_lines: 2000 -> 5000
      Tool output line truncation. Verify-suite outputs (266+ pairs)
      hit this cap and lose visibility into late failures.

  compression.hygiene_hard_message_limit: 400 -> 600
      Source: gateway/run.py:7459 — death-spiral safety valve. 400 is
      tight given max_turns=250 with 1-3 messages per turn.

  updates.pre_update_backup: false -> true
      Make `hermes update` write a backup before overwriting cli.py.
      Would have made the v0.14 regression trivially reversible.

  memory.nudge_interval: 10 -> 20
      Memory-update nudge frequency. 250-turn workers get 12 nudges
      instead of 25 — less token overhead with no real downside.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROFILES_ROOT = Path.home() / "AppData" / "Local" / "hermes" / "profiles"
PROFILES = (
    "neurogolf-worker-1", "neurogolf-worker-2", "neurogolf-worker-3", "neurogolf-worker-4",
    "neurogolf-worker-5", "neurogolf-worker-6", "neurogolf-worker-7", "neurogolf-worker-8",
    "neurogolf-reviewer", "neurogolf-reviewer-2", "neurogolf-submit-captain",
    "neurogolf-sync-manager", "neurogolf-board-analyst", "neurogolf-orchestrator",
    "neurogolf-blocked-55",
)

# Each entry: (yaml_key_path_for_humans, regex, replacement). Single-line YAML
# scalar updates — preserves indentation. The regex matches the key + scalar
# value within the existing indentation; replacement keeps indentation, key,
# and surrounding whitespace.
EDITS = (
    ("agent.max_turns",                            r"(\s*max_turns:\s*)\d+",                 r"\g<1>250"),
    ("agent.reasoning_effort",                     r"(\s*reasoning_effort:\s*)\w+",          r"\g<1>high"),
    ("code_execution.max_tool_calls",              r"(\s*max_tool_calls:\s*)\d+",            r"\g<1>200"),
    ("code_execution.timeout",                     r"(^code_execution:[\s\S]*?\n\s*timeout:\s*)\d+",  r"\g<1>600"),
    ("kanban.failure_limit",                       r"(\s*failure_limit:\s*)\d+",             r"\g<1>3"),
    ("terminal.timeout",                           r"(^terminal:[\s\S]*?\n\s*timeout:\s*)\d+",       r"\g<1>600"),
    ("terminal.lifetime_seconds",                  r"(\s*lifetime_seconds:\s*)\d+",          r"\g<1>1800"),
    ("tool_output.max_lines",                      r"(\s*max_lines:\s*)\d+",                 r"\g<1>5000"),
    ("compression.hygiene_hard_message_limit",     r"(\s*hygiene_hard_message_limit:\s*)\d+", r"\g<1>600"),
    ("updates.pre_update_backup",                  r"(\s*pre_update_backup:\s*)\w+",         r"\g<1>true"),
    ("memory.nudge_interval",                      r"(\s*nudge_interval:\s*)\d+",            r"\g<1>20"),
)


def apply_one(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    out = text
    for label, pattern, repl in EDITS:
        pat = re.compile(pattern, re.MULTILINE)
        new = pat.sub(repl, out, count=1)
        if new != out:
            changes.append(label)
            out = new
    return out, changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes. Default is dry-run.")
    parser.add_argument("--only", nargs="*", help="Restrict to these profile names.")
    args = parser.parse_args(argv)

    targets = args.only or list(PROFILES)
    summary: list[tuple[str, list[str], str]] = []
    for prof in targets:
        cfg = PROFILES_ROOT / prof / "config.yaml"
        if not cfg.exists():
            summary.append((prof, [], "MISSING"))
            continue
        text = cfg.read_text(encoding="utf-8")
        new_text, changes = apply_one(text)
        if not changes:
            summary.append((prof, [], "no change"))
            continue
        if not args.apply:
            summary.append((prof, changes, "DRY-RUN"))
            continue
        bak = cfg.with_suffix(cfg.suffix + ".bak_pre_neurogolf_tune")
        if not bak.exists():
            shutil.copy2(cfg, bak)
        cfg.write_text(new_text, encoding="utf-8")
        summary.append((prof, changes, "applied"))

    print(f"{'profile':32} {'status':12} changes")
    print("-" * 90)
    for prof, changes, status in summary:
        print(f"{prof:32} {status:12} {', '.join(changes) if changes else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
