"""Mine git history, session logs, lessons, and Research_Archive/notes/ for every
mention of each task, then inject a History block into each
Task_by_Task/taskNNN.md.

Scans (per task ID NNN):

  1. **Git log** — all commit subjects touching files matching `*taskNNN*`,
     plus pickaxe-search commits whose diff contains the string `taskNNN`.
  2. **Session logs** — Logs/sessions/*.md mentioning the task. Extracts
     the surrounding section header + 1-2 surrounding lines for context.
  3. **Lessons** — Logs/lessons.md entries that mention the task.
  4. **Research notes** — Research_Archive/notes/*.md mentioning the task that
     aren't already in the auto-link list (which only matches by filename).
  5. **Prototype docstring** — first paragraph of
     `Research_Archive/organic_prototypes/taskNNN_*.py` if present.

Output is injected between `<!-- HIST:BEGIN -->` and `<!-- HIST:END -->`
markers in the .md (added if missing). Idempotent — re-running refreshes
the block in place; manual edits outside it are preserved.

Usage:
    python tools/mine_task_history.py             # all 401 tasks
    python tools/mine_task_history.py --tid 110   # one task
    python tools/mine_task_history.py --tid 110,287,351
"""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TASK_BY_TASK = REPO / "The_400_Tasks" / "Task_by_Task"
SESSIONS = REPO / "Logs" / "sessions"
LESSONS = REPO / "Logs" / "lessons.md"
RESEARCH_NOTES = REPO / "Research_Archive" / "notes"
ORGANIC_PROTOTYPES = REPO / "Research_Archive" / "organic_prototypes"

HIST_BEGIN = "<!-- HIST:BEGIN -->"
HIST_END = "<!-- HIST:END -->"


def task_tag(tid: int) -> str:
    return f"task{tid:03d}"


def task_aliases(tid: int) -> list[str]:
    """All string forms a task ID might appear in."""
    return [
        f"task{tid:03d}",
        f"task{tid:03d}_",
        f"task{tid}",  # rarer
        f"`{tid:03d}`",
        f"task-{tid:03d}",
    ]


def git_history(tid: int) -> list[tuple[str, str]]:
    """Returns [(sha7, subject)] for commits touching this task.

    Combines:
      - Path-based: any commit modifying a file with taskNNN in its path.
      - Message-based: any commit whose subject contains taskNNN.
      - Pickaxe (-S): commits whose diff adds/removes the string taskNNN
        (limited to keep noise down).
    """
    tag = task_tag(tid)
    out: dict[str, str] = {}
    try:
        # Path-based
        r = subprocess.run(
            ["git", "log", "--all", "--pretty=format:%h\t%s", "--", f"*{tag}*"],
            capture_output=True, text=True, cwd=REPO, timeout=30,
        )
        for line in r.stdout.splitlines():
            if "\t" in line:
                h, s = line.split("\t", 1)
                out[h] = s
        # Subject grep
        r = subprocess.run(
            ["git", "log", "--all", "--pretty=format:%h\t%s", "--grep", tag],
            capture_output=True, text=True, cwd=REPO, timeout=30,
        )
        for line in r.stdout.splitlines():
            if "\t" in line:
                h, s = line.split("\t", 1)
                out[h] = s
    except Exception:
        pass
    INFRA = ("captain ", "Phase-", "audit#", "docs sweep", "cleanup:", "VERIFY ", "double-check")
    out = {h: s2 for h, s2 in out.items() if not any(k in s2 for k in INFRA)}  # audit#2 Phase-1b: infra commits appeared in all 401 files
    # Sort newest first (git log returns reverse-chrono; preserve order)
    return list(out.items())


def session_log_mentions(tid: int, max_per_file: int = 1) -> list[tuple[str, str]]:
    """Returns [(session_log_filename, snippet)] for sessions mentioning the task."""
    out: list[tuple[str, str]] = []
    if not SESSIONS.exists():
        return out
    tag = task_tag(tid)
    for f in sorted(SESSIONS.iterdir()):
        if not f.is_file() or not f.name.endswith(".md"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if tag not in text:
            continue
        # Find first N occurrences; emit a short snippet
        lines = text.split("\n")
        emitted = 0
        for i, line in enumerate(lines):
            if tag in line:
                # Trim long lines
                snippet = line.strip()
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                out.append((f.name, snippet))
                emitted += 1
                if emitted >= max_per_file:
                    break
    return out


def lessons_mentions(tid: int) -> list[str]:
    """Returns snippets from Logs/lessons.md mentioning the task."""
    if not LESSONS.exists():
        return []
    try:
        text = LESSONS.read_text(encoding="utf-8")
    except Exception:
        return []
    tag = task_tag(tid)
    if tag not in text:
        return []
    out: list[str] = []
    lines = text.split("\n")
    for line in lines:
        if tag in line:
            s = line.strip()
            if len(s) > 200:
                s = s[:197] + "..."
            out.append(s)
    return out[:5]  # cap to 5


def research_note_mentions(tid: int) -> list[tuple[str, str]]:
    """Notes that mention the task but whose filename doesn't already include taskNNN
    (those are already auto-linked by generate_task_mds.py).
    """
    out: list[tuple[str, str]] = []
    if not RESEARCH_NOTES.exists():
        return out
    tag = task_tag(tid)
    for f in sorted(RESEARCH_NOTES.iterdir()):
        if not f.is_file() or not f.name.endswith(".md"):
            continue
        # Skip files whose name already contains the task tag — those are
        # auto-linked via generate_task_mds.py
        if tag in f.stem or f"task-{tid:03d}" in f.stem or f"task_{tid:03d}" in f.stem:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if tag not in text:
            continue
        # Find first mention with context
        idx = text.find(tag)
        # Find the line
        start = text.rfind("\n", 0, idx) + 1
        end = text.find("\n", idx)
        if end == -1:
            end = len(text)
        line = text[start:end].strip()
        if len(line) > 160:
            line = line[:157] + "..."
        out.append((f.name, line))
    return out


def prototype_docstring(tid: int) -> tuple[str, str] | None:
    """First paragraph of any taskNNN_*.py docstring in Research_Archive/organic_prototypes/."""
    if not ORGANIC_PROTOTYPES.exists():
        return None
    tag = task_tag(tid)
    for f in sorted(ORGANIC_PROTOTYPES.iterdir()):
        if not f.name.startswith(f"{tag}_"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        # Extract docstring (between triple quotes at top)
        m = re.search(r'^"""(.*?)"""', text, flags=re.DOTALL | re.MULTILINE)
        if not m:
            continue
        doc = m.group(1).strip()
        # First paragraph only
        para = doc.split("\n\n", 1)[0].strip()
        if len(para) > 600:
            para = para[:597] + "..."
        return f.name, para
    return None


def build_history_block(tid: int) -> str:
    """Return the markdown content for the HIST:BEGIN..HIST:END block."""
    out = ["<!-- HIST:BEGIN -->",
           "<!-- mined by tools/mine_task_history.py; rerun to refresh -->",
           ""]

    proto = prototype_docstring(tid)
    if proto:
        fname, para = proto
        out.append("**Prototype docstring** (`Research_Archive/organic_prototypes/" + fname + "`):")
        out.append("")
        for line in para.split("\n"):
            out.append("> " + line)
        out.append("")

    notes = research_note_mentions(tid)
    if notes:
        out.append("**Research_Archive/notes/ mentions** (files whose filename does NOT already match):")
        for fname, snip in notes[:8]:
            out.append(f"- `Research_Archive/notes/{fname}` — {snip}")
        out.append("")

    commits = git_history(tid)
    if commits:
        out.append("**Git history** (commits touching files or subjects mentioning this task):")
        for sha, subj in commits[:15]:
            out.append(f"- `{sha}` — {subj}")
        if len(commits) > 15:
            out.append(f"- _...and {len(commits) - 15} more_")
        out.append("")

    sess = session_log_mentions(tid)
    if sess:
        out.append("**Session log mentions:**")
        for fname, snip in sess[:10]:
            out.append(f"- `Logs/sessions/{fname}` — {snip}")
        if len(sess) > 10:
            out.append(f"- _...and {len(sess) - 10} more session logs_")
        out.append("")

    lessons = lessons_mentions(tid)
    if lessons:
        out.append("**Logs/lessons.md mentions:**")
        for snip in lessons:
            out.append(f"- {snip}")
        out.append("")

    if len(out) == 3:  # only the headers, nothing found
        out.append("_No history found in git / sessions / lessons / Research_Archive/notes/ beyond auto-linked filenames._")
        out.append("")

    out.append(HIST_END)
    return "\n".join(out)


def update_md(tid: int) -> str:
    """Update Task_by_Task/taskNNN.md with a fresh HIST block. Returns the
    chosen action: 'created', 'inserted', 'refreshed'."""
    path = TASK_BY_TASK / f"task{tid:03d}.md"
    if not path.exists():
        # No .md to update; skip
        return "skipped"
    text = path.read_text(encoding="utf-8")
    block = build_history_block(tid)
    if HIST_BEGIN in text and HIST_END in text:
        # Pass block via a lambda so re.sub doesn't interpret \q etc as escapes
        # in the replacement string. Windows paths in commit subjects (e.g.
        # `<USER_HOME>\...`) would otherwise raise bad-escape errors.
        new_text = re.sub(
            rf"{re.escape(HIST_BEGIN)}.*?{re.escape(HIST_END)}",
            lambda m: block, text, count=1, flags=re.DOTALL,
        )
        action = "refreshed"
    else:
        # Insert just after the AUTO:END marker if present, else after the H1.
        AUTO_END = "<!-- AUTO:END -->"
        if AUTO_END in text:
            insert_at = text.index(AUTO_END) + len(AUTO_END)
            new_text = text[:insert_at] + "\n\n" + block + "\n" + text[insert_at:]
        else:
            # Insert after first H1 line
            m = re.search(r"^# .*?\n", text)
            if m:
                insert_at = m.end()
                new_text = text[:insert_at] + "\n" + block + "\n" + text[insert_at:]
            else:
                new_text = block + "\n\n" + text
        action = "inserted"
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
    return action


def parse_tid_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("task"):
            part = part[4:]
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tid", type=str, default="",
                   help="Comma-separated task IDs (e.g. '110,287'); empty means all 0-400")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=400)
    args = p.parse_args(argv)

    if args.tid:
        tids = parse_tid_list(args.tid)
    else:
        tids = list(range(args.start, args.end + 1))

    counts: dict[str, int] = defaultdict(int)
    for tid in tids:
        action = update_md(tid)
        counts[action] += 1
    print(f"Mined {len(tids)} tasks: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
