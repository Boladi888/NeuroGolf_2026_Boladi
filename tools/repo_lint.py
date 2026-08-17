#!/usr/bin/env python3
"""repo_lint.py — standing invariants from the 2026-06 audit cycles.

On-demand CLI (no hooks, no CI, never edits files). Catches the finding classes
that recurred across audits #1-#4: dead tool refs, retired-system refs (hermes/
lanes), hardcoded paths, duplicate library anchors, missing token-bomb banners,
oversize task journals (whale engine), prompt-contract drift, staging hygiene.

Usage:
    python tools/repo_lint.py                 # all checks, compact table
    python tools/repo_lint.py --only task-md-size --task 233
    python tools/repo_lint.py --only prompts
    python tools/repo_lint.py --list          # list check names
    python tools/repo_lint.py --quiet         # errors only

Exit code: 1 if any ERROR-severity finding, else 0 (warnings don't fail).
Writes Logs/lint_last.md with the full finding list.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- allowlists
ALLOWLISTS = {
    # Files allowed to mention .hermes / hermes kanban (stubs & archived-in-place docs)
    "hermes_refs": {
        "CURRENT_STATE.md",                     # pointer stub documents the coupling
        ".ai_instructions/playbooks/WorkerAgent_Setup.md",  # archived-in-place transport doc
    },
    # Lines (or their preceding line) mentioning retired things are fine if tombstoned
    "lane_tombstone_markers": ("archived", "retired", "tombstone", "obe", "never committed",
                               "since deleted", "deleted", "no longer exists", "_archive"),
    # tools/*.py allowed to hardcode absolute paths
    "hardcoded_path_files": {
        "neurogolf_paths.py",                   # THE canonical-root anchor
        "repo_lint.py",                         # this linter's own check patterns
        # pre-existing legacy tools with hardcoded DEFAULT_* paths (work in this repo;
        # not from active dev — acknowledged 2026-06-08 housekeeping, relativize if ever edited):
        "adversarial_test.py", "catalog_organic_prototypes.py", "onnx_opt_safe.py",
        "onnx_opt_with_fold.py", "scan_scratch_onnxes.py", "verify_against_live.py",
    },
    # a line is fine if it points outside the repo by convention
    "hardcoded_path_line_ok": (".notongit",),
    # generated docs: (path, staleness-days, how-to-refresh)
    "generated_docs": [
        ("The_400_Tasks/CASCADE_ANALYSIS.md", 21, "python tools/cascade_report.py"),
    ],
}

# Active doc surfaces (instruction-bearing; historical corpora excluded on purpose:
# Logs/, Research_Archive/, exemplars/, _archive/, Already_Ran/, Backlog/, Task_by_Task/)
def active_doc_surfaces() -> list[Path]:
    out: list[Path] = []
    for rel in ("README.md", "AGENT_TASK_START.md", "WHERE_DATA_LIVES.md",
                "CLAUDE.md", "CURRENT_STATE.md", "tools/README.md"):
        p = REPO / rel
        if p.exists():
            out.append(p)
    for pat in (".ai_instructions/*.md", ".ai_instructions/playbooks/*.md",
                "Onnx_Advanced/*.md", "The_400_Tasks/*.md",
                "The_400_Tasks/Task Prompts/1/*.md", "The_400_Tasks/Task Prompts/2/*.md"):
        for p in sorted(REPO.glob(pat)):
            if "_archive" in p.parts:
                continue
            if p.name == "03_PROJECT_HISTORY.md":   # explicitly a history archive
                continue
            out.append(p)
    return out


def _tombstoned(lines: list[str], idx: int) -> bool:
    """True if this line or the preceding one carries a tombstone marker."""
    window = lines[max(0, idx - 1): idx + 1]
    low = " ".join(window).lower()
    return any(m in low for m in ALLOWLISTS["lane_tombstone_markers"])


class Finding:
    def __init__(self, check: str, severity: str, where: str, msg: str):
        self.check, self.severity, self.where, self.msg = check, severity, where, msg

    def row(self) -> str:
        return f"| {self.severity} | {self.check} | {self.where} | {self.msg} |"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


# ------------------------------------------------------------------- checks
MARKER_RE = re.compile(
    r"<!--\s*(AUTO|HIST|GDIAG|SPREAD):BEGIN\s*-->.*?<!--\s*\1:END\s*-->",
    re.DOTALL,
)


def manual_size_bytes(md_path: Path) -> int:
    """Size of a taskNNN.md once generated marker blocks are removed."""
    txt = _read(md_path)
    return len(MARKER_RE.sub("", txt).encode("utf-8"))


def check_task_md_size(task_filter: str | None) -> list[Finding]:
    out: list[Finding] = []
    tdir = REPO / "The_400_Tasks" / "Task_by_Task"
    files = sorted(tdir.glob("task[0-9][0-9][0-9].md"))
    if task_filter:
        files = [f for f in files if f.stem == f"task{int(task_filter):03d}"]
    for f in files:
        kb = manual_size_bytes(f) / 1024
        if kb >= 20:
            out.append(Finding("task-md-size", "ERROR", _rel(f),
                               f"manual content {kb:.0f} KB >= 20 KB - move history to "
                               f"Task_by_Task/_archive/{f.stem}_history.md (WORKER_PROTOCOL staging rule)"))
        elif kb >= 15:
            out.append(Finding("task-md-size", "WARN", _rel(f),
                               f"manual content {kb:.0f} KB >= 15 KB - approaching the 20 KB tripwire"))
    return out


TOOL_REF_RE = re.compile(r"tools/(?:agent_helpers/)?[A-Za-z0-9_]+\.py")


def check_dead_tool_refs(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    tools_root = REPO / "tools"
    for doc in active_doc_surfaces():
        lines = _read(doc).splitlines()
        for idx, line in enumerate(lines):
            for ref in TOOL_REF_RE.findall(line):
                p = REPO / ref
                if p.exists():
                    continue
                name = Path(ref).name
                in_archive = (tools_root / "_archive" / name).exists()
                where = f"{_rel(doc)}:{idx + 1}"
                if _tombstoned(lines, idx):
                    continue
                if in_archive:
                    out.append(Finding("dead-tool-refs", "WARN", where,
                                       f"`{ref}` is archived (tools/_archive/) but the line doesn't say so"))
                else:
                    out.append(Finding("dead-tool-refs", "ERROR", where,
                                       f"`{ref}` does not exist anywhere"))
    return out


def check_hermes_refs(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    for doc in active_doc_surfaces():
        rel = _rel(doc)
        if rel in ALLOWLISTS["hermes_refs"]:
            continue
        lines = _read(doc).splitlines()
        for idx, line in enumerate(lines):
            low = line.lower()
            if ".hermes" in low or "hermes kanban" in low:
                if "_hermes_archive" in low or _tombstoned(lines, idx):
                    continue  # tombstone/pointer to the archive is fine
                out.append(Finding("hermes-refs", "ERROR", f"{rel}:{idx + 1}",
                                   "live reference to retired hermes system (use task-local staged_candidates/ + Task Prompts)"))
    return out


LANE_RE = re.compile(r"07_LANES|alpha_sub_lanes|NewSession_Alpha|NewSession_Beta|split_alpha_lanes")


def check_lane_refs(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    for doc in active_doc_surfaces():
        lines = _read(doc).splitlines()
        for idx, line in enumerate(lines):
            if LANE_RE.search(line):
                if _tombstoned(lines, idx):
                    continue
                out.append(Finding("lane-refs", "ERROR", f"{_rel(doc)}:{idx + 1}",
                                   "live reference to retired lane system"))
    return out


def check_hardcoded_paths(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    for py in sorted((REPO / "tools").glob("*.py")) + sorted((REPO / "tools" / "agent_helpers").glob("*.py")):
        if py.name in ALLOWLISTS["hardcoded_path_files"]:
            continue
        txt = _read(py)
        for lineno, line in enumerate(txt.splitlines(), 1):
            if ".claude" in line and "worktrees" in line:
                out.append(Finding("hardcoded-paths", "ERROR", f"{_rel(py)}:{lineno}",
                                   "dead-worktree path (use Path(__file__).resolve().parents[1])"))
            elif 'Path(r"C:' in line or "Path(r'C:" in line:
                if any(ok in line for ok in ALLOWLISTS["hardcoded_path_line_ok"]):
                    continue
                out.append(Finding("hardcoded-paths", "WARN", f"{_rel(py)}:{lineno}",
                                   "absolute repo path hardcoded (works here, breaks elsewhere; "
                                   "prefer Path(__file__).parents[1] or tools/neurogolf_paths.py)"))
    return out


def check_dup_anchors(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    from collections import Counter
    tech = REPO / "Onnx_Advanced" / "techniques.md"
    if tech.exists():
        # Catch BOTH styles: clean "# 94. Title" (period) AND worker-drafted
        # "### #94 — Title" (em/en-dash or hyphen). The old period-only regex let
        # parallel-worker "### #NN —" collisions through invisibly (2026-06-08 fix).
        nums = re.findall(r"^#{1,4}\s+#?(\d+)\s*[.—–-]\s", _read(tech), re.M)
        for num, count in Counter(nums).items():
            if count > 1:
                out.append(Finding("dup-anchors", "ERROR", _rel(tech),
                                   f"heading number #{num} appears {count}x - renumber with breadcrumb"))
    traps = REPO / "Onnx_Advanced" / "traps.md"
    if traps.exists():
        anchors = re.findall(r"^#+\s*(§[\w-]+)", _read(traps), re.M)
        for a, count in Counter(anchors).items():
            if count > 1:
                out.append(Finding("dup-anchors", "ERROR", _rel(traps),
                                   f"anchor {a} appears {count}x"))
    return out


def check_banners(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    bs = REPO / "tools" / "build_submission.py"
    if bs.exists():
        head = bs.open(encoding="utf-8", errors="replace").read(2500)
        if "DO NOT Read this file whole" not in head:
            out.append(Finding("banners", "ERROR", "tools/build_submission.py:1",
                               "missing the 240k-token DO-NOT-Read banner"))
    schema = REPO / "The_400_Tasks" / "SCHEMA.md"
    if schema.exists():
        first = _read(schema).splitlines()[0] if _read(schema).splitlines() else ""
        if "NEVER Read it whole" not in first:
            out.append(Finding("banners", "ERROR", "The_400_Tasks/SCHEMA.md:1",
                               "missing the task_state.csv token-bomb banner"))
    return out


def check_tracked_secrets(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    try:
        res = subprocess.run(["git", "ls-files", ".env*", "*.token", "*.pem", "credentials.json", "secrets.json"],
                             cwd=REPO, capture_output=True, text=True, timeout=30)
        for f in res.stdout.split():
            if f.endswith(".env.example"):
                continue
            out.append(Finding("tracked-secrets", "ERROR", f, "secret-pattern file is git-TRACKED"))
    except Exception:
        pass  # git unavailable - skip silently
    return out


def check_generated_staleness(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    now = datetime.datetime.now()
    for rel, max_days, how in ALLOWLISTS["generated_docs"]:
        p = REPO / rel
        if not p.exists():
            continue
        age = (now - datetime.datetime.fromtimestamp(p.stat().st_mtime)).days
        if age > max_days:
            out.append(Finding("generated-staleness", "WARN", rel,
                               f"{age} days old (budget {max_days}) - refresh: {how}"))
    return out


# (removed _team_split() 2026-06-08 — it read the now-stale OnnxlyFans_TeamTasks.csv;
#  the prompt check uses _assigned_split() = the authoritative task_state.team_split instead.)


def _boladi_workable() -> dict[str, str]:
    """Per-task 'Y'/'N' from task_state.boladi_workable (2026-06-07: boladi is team-best
    => prompting it counts for the team regardless of the original assignment bucket)."""
    csv_p = REPO / "The_400_Tasks" / "task_state.csv"
    if not csv_p.exists():
        return {}
    import csv as _csv
    out = {}
    with open(csv_p, encoding="utf-8", newline="") as f:
        for r in _csv.DictReader(f):
            if "boladi_workable" in r:
                out[r["task"]] = r.get("boladi_workable", "")
    return out


def _assigned_split() -> dict[str, str]:
    """Per-task assignment bucket from task_state.team_split — the AUTHORITATIVE current
    split (updated per the owner's xlsx), not the stale OnnxlyFans_TeamTasks.csv. An
    explicitly boladi-assigned task is always OK to prompt even if aditya currently leads
    it (the owner wants boladi to beat aditya there)."""
    csv_p = REPO / "The_400_Tasks" / "task_state.csv"
    if not csv_p.exists():
        return {}
    import csv as _csv
    out = {}
    with open(csv_p, encoding="utf-8", newline="") as f:
        for r in _csv.DictReader(f):
            if "team_split" in r:
                out[r["task"]] = (r.get("team_split", "") or "").lower()
    return out


def check_prompts(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    split = _assigned_split()  # authoritative task_state.team_split, not the stale TeamTasks.csv
    workable = _boladi_workable()
    for folder in ("1", "2"):
        for p in sorted((REPO / "The_400_Tasks" / "Task Prompts" / folder).glob("*.md")):
            rel = _rel(p)
            txt = _read(p)
            if "staged_candidates/" not in txt:
                out.append(Finding("prompts", "ERROR", rel, "missing staged_candidates/ staging instruction"))
            if "WORKER_PROTOCOL" not in txt:
                out.append(Finding("prompts", "WARN", rel, "doesn't cite WORKER_PROTOCOL.md"))
            if ".hermes" in txt:
                out.append(Finding("prompts", "ERROR", rel, "references retired .hermes path"))
            m = re.search(r"Task(\d{3})", p.name)
            if m:
                t = f"task{m.group(1)}"
                # Eligibility rule (2026-06-07): a prompt is OK if boladi is the team-best
                # scorer for that task (boladi_workable == Y) OR it's boladi-assigned.
                # Only flag tasks where aditya is genuinely better (workable == N) AND not
                # boladi-assigned — prompting those is wasted (can't beat aditya's lower cost).
                w = workable.get(t)
                assigned_boladi = split.get(t) == "boladi"
                if w == "N" and not assigned_boladi:
                    out.append(Finding("prompts", "ERROR", rel,
                                       f"{t}: aditya is team-best (boladi_workable=N) and not boladi-assigned "
                                       f"- prompting wastes effort (can't beat aditya's lower cost)"))
                elif w is None and split.get(t) == "aditya":
                    out.append(Finding("prompts", "WARN", rel,
                                       f"{t} is ADITYA's and no workable flag - verify boladi is team-best before prompting"))
    return out


def _manifest_costs(m: dict) -> tuple[float | None, float | None]:
    """Resolve (score_v2_cost, profile_cost) across the manifest layouts workers emit.

    Layout A (task246-style): top-level "cost": <number>, "profile_cost": <number>.
    Layout B (task321-style): nested "cost": {"score_v2_cost": N, "profile_cost": N, ...}.
    Layout C (task346-style): "candidate_cost": {"cost": N, ...}, top-level "profile_cost": N.
    """
    cost = m.get("cost")
    cand = m.get("candidate_cost")
    if isinstance(cost, dict):
        static = cost.get("score_v2_cost") or cost.get("score_v2") or cost.get("total") or cost.get("cost")
        profile = cost.get("profile_cost") or m.get("profile_cost")
    elif isinstance(cand, dict):
        static = cand.get("cost") or cand.get("score_v2_cost") or cand.get("total")
        profile = m.get("profile_cost")
    else:
        static = cost if isinstance(cost, (int, float)) else m.get("score_v2") or m.get("score_v2_cost")
        profile = m.get("profile_cost")
    return (static if isinstance(static, (int, float)) else None,
            profile if isinstance(profile, (int, float)) else None)


def check_staging(_: str | None) -> list[Finding]:
    out: list[Finding] = []
    task_root = REPO / "The_400_Tasks" / "Task_by_Task"
    for sdir in sorted(task_root.glob("task*/staged_candidates")):
        if not sdir.is_dir():
            continue
        for d in sorted(p for p in sdir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            mf = d / "manifest.json"
            if not mf.exists():
                out.append(Finding("staging", "WARN", _rel(d), "staged dir has no manifest.json"))
            else:
                try:
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    static, profile = _manifest_costs(m)
                    if static is None:
                        out.append(Finding("staging", "WARN", _rel(d),
                                           "manifest has no resolvable score_v2 cost (top-level number or cost.score_v2_cost)"))
                    if profile is None:
                        out.append(Finding("staging", "WARN", _rel(d),
                                           "manifest missing profile_cost (WORKER_PROTOCOL gate 5 - real-input profile)"))
                    elif static and abs(profile - static) / static > 0.05:
                        out.append(Finding("staging", "WARN", _rel(d),
                                           f"profile {profile:.0f} vs static {static:.0f} (>5% gap) - credit-probe class, not exact"))
                except (json.JSONDecodeError, OSError) as e:
                    out.append(Finding("staging", "WARN", _rel(d), f"manifest unreadable: {e}"))
            zips = list(d.glob("*.zip"))
            if zips:
                out.append(Finding("staging", "WARN", _rel(d),
                                   f"contains {len(zips)} zip(s) - stage ONNX+manifest only (delta convention)"))
    return out


CHECKS = {
    "task-md-size": check_task_md_size,
    "dead-tool-refs": check_dead_tool_refs,
    "hermes-refs": check_hermes_refs,
    "lane-refs": check_lane_refs,
    "hardcoded-paths": check_hardcoded_paths,
    "dup-anchors": check_dup_anchors,
    "banners": check_banners,
    "tracked-secrets": check_tracked_secrets,
    "generated-staleness": check_generated_staleness,
    "prompts": check_prompts,
    "staging": check_staging,
}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="comma-separated check names")
    ap.add_argument("--task", help="restrict task-md-size to one task number")
    ap.add_argument("--quiet", action="store_true", help="print errors only")
    ap.add_argument("--list", action="store_true", help="list checks and exit")
    args = ap.parse_args()

    if args.list:
        for name in CHECKS:
            print(name)
        return 0

    selected = list(CHECKS) if not args.only else [c.strip() for c in args.only.split(",")]
    unknown = [c for c in selected if c not in CHECKS]
    if unknown:
        print(f"unknown checks: {unknown}; --list to see names")
        return 2

    findings: list[Finding] = []
    for name in selected:
        findings.extend(CHECKS[name](args.task))

    errors = [f for f in findings if f.severity == "ERROR"]
    warns = [f for f in findings if f.severity == "WARN"]

    shown = errors if args.quiet else findings
    if shown:
        print("| sev | check | where | finding |")
        print("|---|---|---|---|")
        for f in shown:
            print(f.row())
    print(f"\nrepo_lint: {len(errors)} error(s), {len(warns)} warning(s) "
          f"across {len(selected)} check(s)")

    # durable report
    report = REPO / "Logs" / "lint_last.md"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# lint_last.md - generated {stamp} by tools/repo_lint.py",
             f"\nChecks run: {', '.join(selected)}",
             f"\nResult: **{len(errors)} errors / {len(warns)} warnings**\n"]
    if findings:
        lines += ["| sev | check | where | finding |", "|---|---|---|---|"]
        lines += [f.row() for f in findings]
    else:
        lines.append("\nClean.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
