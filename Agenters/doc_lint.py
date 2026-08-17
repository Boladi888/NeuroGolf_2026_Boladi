#!/usr/bin/env python3
"""Lightweight lint for Agenters role docs.

This mirrors the Onnx_Advanced discipline at a smaller scale: keep role docs
bounded, keep canonical doctrine in Agenters/README.md, and keep regenerable run
transcripts out of git.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

LINE_BUDGETS = {
    "README.md": 140,
    "Supervisor/README.md": 240,
    "Orchestrator/README.md": 430,
    "Orchestrator/TROUBLESHOOTING.md": 90,
    "Orchestrator/WORKER_BRIEF.md": 80,
    "Submit_Captain/README.md": 410,
    "Submit_Captain/PACKET_SCHEMA.md": 90,
    "Tough_Reviewer/README.md": 240,
    "Prompter/README.md": 210,  # raised from 180 for the owner's 2026-06 expansion (trim back if it slims down)
    "Janitor/README.md": 80,
    "Librarian/README.md": 80,
}

SPAWN_PROMPT_BUDGET = 80
REQUIRED_GITIGNORE_LINES = {
    "Agenters/Orchestrator/runs/**/*.jsonl",
    "Agenters/Orchestrator/runs/**/*.err",
}

CANONICAL_PHRASES = {
    "home-cost trap": 3,
    "task_state.csv current_cost": 3,
    "+0.05": 4,
    "+0.20": 4,
}

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def live_markdown_files() -> list[Path]:
    skip_parts = {"_archive", "_proposals", "runs", "_kanban_plans"}
    out = []
    for path in ROOT.rglob("*.md"):
        if skip_parts.intersection(path.relative_to(ROOT).parts):
            continue
        out.append(path)
    return sorted(out)


def resolve_doc_link(source: Path, target: str) -> Path | None:
    target = target.strip()
    if not target or target.startswith(("#", "http:", "https:", "mailto:")):
        return None
    target = target.split("#", 1)[0]
    if not target.endswith(".md"):
        return None
    target = target.replace("\\", "/")
    first = target.split("/", 1)[0]
    base = REPO if first in {"Agenters", "Onnx_Advanced", "The_400_Tasks", "Logs", "tools"} else source.parent
    return (base / target).resolve()


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8", errors="replace")
    for line in sorted(REQUIRED_GITIGNORE_LINES):
        if line not in gitignore:
            errors.append(f".gitignore missing required Agenters ignore: {line}")

    for rel_path, budget in LINE_BUDGETS.items():
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"missing expected doc: Agenters/{rel_path}")
            continue
        n = line_count(path)
        if n > budget:
            errors.append(f"doc too long: Agenters/{rel_path} has {n} lines > budget {budget}")

    for path in ROOT.rglob("SPAWN_PROMPT.md"):
        if any(part in {"_archive", "_proposals", "runs", "_kanban_plans"} for part in path.relative_to(ROOT).parts):
            continue
        n = line_count(path)
        if n > SPAWN_PROMPT_BUDGET:
            errors.append(f"spawn prompt too long: {rel(path)} has {n} lines > budget {SPAWN_PROMPT_BUDGET}")

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
    for heading in ("## ⛔ Shared source of truth", "## ⛔ Shared doctrine"):
        if heading not in root_readme:
            errors.append(f"Agenters/README.md missing canonical heading: {heading}")

    live_text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in live_markdown_files())
    for phrase, max_count in CANONICAL_PHRASES.items():
        count = live_text.lower().count(phrase.lower())
        if count > max_count:
            errors.append(f"doctrine phrase drift: {phrase!r} appears {count} times > limit {max_count}")
        elif count == max_count:
            warnings.append(f"doctrine phrase at limit: {phrase!r} appears {count}/{max_count} times")

    for path in live_markdown_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in LINK_RE.finditer(text):
            resolved = resolve_doc_link(path, m.group(1))
            if resolved and not resolved.exists():
                errors.append(f"dead markdown link in {rel(path)} -> {m.group(1)}")

    for msg in warnings:
        print(f"WARN: {msg}")
    if errors:
        for msg in errors:
            print(f"ERROR: {msg}")
        return 1
    print("Agenters doc lint OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
