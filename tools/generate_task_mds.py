"""Generate per-task .md files in The_400_Tasks/Task_by_Task/.

One .md per task (taskNNN.md for NNN in 0..400).

Design:
- For AI agents only. Technical, no fancy formatting.
- Does NOT duplicate task_state info (that lives in NeuroGolf_Task_Status.xlsx /
  task_state.csv). Each .md auto-links to those.
- AUTO-LINKS to Research_Archive/notes/recipe-taskNNN-*.md and
  Research_Archive/organic_prototypes/taskNNN_*.py / taskNNN_*.py / taskNNN_numpy_reference.py
  if they exist.
- Auto-scrapes task_cards/taskNNN.md sections (Attempts, Don't-do log, etc.)
  into the new file structure (only the unique-to-this-task content; not the
  duplicated task_state info).
- Idempotent: re-running preserves manual edits between the auto-generated
  fences (`<!-- AUTO:BEGIN -->` ... `<!-- AUTO:END -->`).

Sections in each .md:
  1. Rule (1-3 sentences on transformation)
  2. Algorithm sketch (numpy pseudocode if known)
  3. Attempts (one row per attempt)
  4. Cost analysis (why current cost; how to reduce)
  5. Don't-do log
  6. Cross-references (auto-generated)

Usage:
    python tools/generate_task_mds.py
"""
from __future__ import annotations

import argparse
import csv as _csv
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "The_400_Tasks" / "Task_by_Task"
OLD_CARDS_DIR = REPO / "The_400_Tasks" / "task_cards"
RESEARCH_NOTES = REPO / "Research_Archive" / "notes"
ORGANIC_PROTOTYPES = REPO / "Research_Archive" / "organic_prototypes"


AUTO_BEGIN = "<!-- AUTO:BEGIN -->"
AUTO_END = "<!-- AUTO:END -->"


_TASK_STATE_ROWS = None
def _status_line(tid):
    global _TASK_STATE_ROWS
    try:
        if _TASK_STATE_ROWS is None:
            with open(REPO / "The_400_Tasks" / "task_state.csv", encoding="utf-8", newline="") as handle:
                _TASK_STATE_ROWS = list(_csv.DictReader(handle))
        for row in _TASK_STATE_ROWS:
            if row.get("task") == f"task{tid:03d}" or row.get("task_id") == str(tid):
                c = row.get("current_cost")
                return f"**STATUS:** current cost **{int(float(c)):,}** (task_state.csv)." if c else "**STATUS:** current cost unknown — run show_task.py."
        return "**STATUS:** current cost unknown — run show_task.py."
    except Exception:
        return "**STATUS:** current cost unavailable."

def find_research_links(tid: int) -> list[tuple[str, Path]]:
    """Find any Research_Archive/notes/*.md or Research_Archive/organic_prototypes/*.py for this task."""
    out: list[tuple[str, Path]] = []
    tag = f"task{tid:03d}"
    tag_alt = f"task-{tid:03d}"
    tag_short = f"task{tid}"  # less common but possible
    # Research notes
    if RESEARCH_NOTES.exists():
        for f in sorted(RESEARCH_NOTES.iterdir()):
            if not f.is_file():
                continue
            n = f.name.lower()
            if tag in n or tag_alt in n:
                out.append(("Research_Archive/notes", f))
    # Organic prototypes
    if ORGANIC_PROTOTYPES.exists():
        for f in sorted(ORGANIC_PROTOTYPES.iterdir()):
            if not f.is_file():
                continue
            n = f.name.lower()
            if n.startswith(f"{tag}_") or n.startswith(f"{tag_alt}_"):
                out.append(("Research_Archive/organic_prototypes", f))
    return out


def parse_existing_card(path: Path) -> dict[str, str]:
    """Pull the unique-content sections from an existing task_card .md.

    Returns dict of {section: body} for sections we care about:
      rule, algorithm, attempts, blockers, dont_do, pointers (raw)
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    # Section keys to look for (case-insensitive H2)
    key_map = {
        "transformation rule": "rule",
        "algorithm": "algorithm",
        "attempts": "attempts",
        "open questions / blockers": "blockers",
        "open questions": "blockers",
        "blockers": "blockers",
        "don't-do log": "dont_do",
        "dont-do log": "dont_do",
        "pointers": "pointers",
        "cross-references": "cross_refs",
    }
    # Split on H2
    parts = re.split(r"^## ", text, flags=re.MULTILINE)
    for part in parts[1:]:
        first_line, _, body = part.partition("\n")
        key = first_line.strip().lower()
        mapped = key_map.get(key)
        if mapped is None:
            # try to match approximately
            for k, v in key_map.items():
                if k in key:
                    mapped = v
                    break
        if mapped is None:
            continue
        sections[mapped] = body.strip()
    return sections


def existing_user_content(out_path: Path) -> str | None:
    """If the .md already exists, extract everything outside the AUTO block.

    Returns the user-edited content (pre + post), or None if no AUTO markers found.
    """
    if not out_path.exists():
        return None
    text = out_path.read_text(encoding="utf-8")
    if AUTO_BEGIN not in text or AUTO_END not in text:
        return None
    pre, _, rest = text.partition(AUTO_BEGIN)
    _, _, post = rest.partition(AUTO_END)
    return pre + AUTO_BEGIN + "<!-- generated; manual edits go OUTSIDE this block -->\n{auto}\n" + AUTO_END + post


def build_md(tid: int, prior_card_sections: dict[str, str]) -> str:
    """Build the .md content for task tid."""
    tag = f"task{tid:03d}"
    links = find_research_links(tid)

    # Build auto-generated section (cross-references + links)
    auto_lines = [
        f"<!-- generated by tools/generate_task_mds.py; manual edits OUTSIDE this block -->",
        "",
        _status_line(tid),
        "(one row: `python tools/agent_helpers/show_task.py --task N` — NEVER Read task_state.csv whole)",
        "",
        "**Linked research / prototypes** (auto-detected by filename):",
    ]
    if links:
        for kind, p in links:
            rel = str(p.relative_to(REPO)).replace("\\", "/")
            auto_lines.append(f"- `{rel}`")
    else:
        auto_lines.append("- _none found_")
    auto_block = "\n".join(auto_lines)

    # Build the body sections — prefer prior-card content when present
    rule = prior_card_sections.get("rule", "_unknown — derive and document here. If a recipe exists in `Research_Archive/notes/`, summarize it._").strip()
    algorithm = prior_card_sections.get("algorithm", "").strip()
    attempts = prior_card_sections.get("attempts", "").strip()
    blockers = prior_card_sections.get("blockers", "").strip()
    dont_do = prior_card_sections.get("dont_do", "").strip()
    pointers = prior_card_sections.get("pointers", "").strip()

    parts = [f"# {tag}", ""]
    parts.append(AUTO_BEGIN)
    parts.append(auto_block)
    parts.append(AUTO_END)
    parts.append("")
    parts.append("## Rule")
    parts.append("")
    parts.append(rule if rule else "_unknown — derive and document here. If a recipe exists in `Research_Archive/notes/`, summarize it._")
    parts.append("")
    parts.append("## Algorithm sketch")
    parts.append("")
    if algorithm:
        parts.append(algorithm)
    else:
        parts.append("_If known, paste numpy pseudocode here or point at `Research_Archive/notes/recipe-taskNNN-*.md`._")
    parts.append("")
    parts.append("## Attempts")
    parts.append("")
    if attempts:
        parts.append(attempts)
    else:
        parts.append("_One row per attempt. Format: `YYYY-MM-DD | approach | result | cost | session log`._")
    parts.append("")
    parts.append("## Cost analysis")
    parts.append("")
    parts.append("_Why is the current cost what it is? What ops dominate? Are there cheaper expressions of the same rule? Link a session log if you've measured._")
    parts.append("")
    parts.append("## Don't-do log")
    parts.append("")
    if dont_do:
        parts.append(dont_do)
    else:
        parts.append("_Record ruled-out hypotheses here so future agents don't retry them._")
    parts.append("")
    if pointers:
        parts.append("## Pointers (legacy task_card)")
        parts.append("")
        parts.append(pointers)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=400)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    preserved_manual = 0
    for tid in range(args.start, args.end + 1):
        tag = f"task{tid:03d}"
        out_path = args.out_dir / f"{tag}.md"
        # Pull from existing task_card if present
        prior_path = OLD_CARDS_DIR / f"{tag}.md"
        prior_sections = parse_existing_card(prior_path) if prior_path.exists() else {}

        # If the destination .md already exists with AUTO markers and user content,
        # only regenerate the AUTO section; keep user content
        if out_path.exists():
            text = out_path.read_text(encoding="utf-8")
            if AUTO_BEGIN in text and AUTO_END in text:
                # Regenerate only the AUTO block
                new_md = build_md(tid, prior_sections)
                # Extract new auto block
                m = re.search(rf"{re.escape(AUTO_BEGIN)}.*?{re.escape(AUTO_END)}", new_md, re.DOTALL)
                if m:
                    new_auto = m.group(0)
                    text = re.sub(rf"{re.escape(AUTO_BEGIN)}.*?{re.escape(AUTO_END)}", new_auto, text, count=1, flags=re.DOTALL)
                    out_path.write_text(text, encoding="utf-8")
                    preserved_manual += 1
                    continue

        # Otherwise write fresh
        out_path.write_text(build_md(tid, prior_sections), encoding="utf-8")
        written += 1

    print(f"Generated {written} new files, regenerated AUTO block in {preserved_manual} existing files.")
    print(f"Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
