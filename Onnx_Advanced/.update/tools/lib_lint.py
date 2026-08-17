#!/usr/bin/env python3
"""lib_lint.py - standing invariants for the Onnx_Advanced library.

Every check here encodes a contract that a past audit found broken after it had
been fixed once by hand. Run before committing library changes (and after any
sorter integration pass):

    python Onnx_Advanced/.update/tools/lib_lint.py [--only check1,check2]

Exit 0 = no errors (warnings allowed). Exit 1 = errors. Stdlib only, Windows-safe.

Checks:
  dup-sections        duplicate top-level `# N.` numbers in techniques.md            [ERROR]
  catalog-stale       techniques_catalog.md source-sha != techniques.md sha          [ERROR]
  catalog-coverage    every techniques.md section has a catalog row and vice versa   [ERROR]
  index-two-way       root *.md all routed in INDEX.md; INDEX rows point at real files[ERROR/WARN]
  size-budgets        INDEX budget table vs actual sizes; first-read sum <= 75 KB    [ERROR>110%, WARN>100%]
  routing-coverage    catalog sections (>= #48) with no by_symptom.md mention        [INFO]
  stub-expiry         tiny pointer files at root with no purpose                     [WARN]
  done-archival       >15 KB non-audit files in Research/Done (unarchived bodies?)   [WARN]
  changelog-cap       CHANGELOG.md over 60 KB (rotate to _archive)                   [WARN]
  integrated-clean    .update/_integrated exists / non-empty                         [WARN]
  required-reading    the phrase outside WORKER_PROTOCOL.md / INDEX.md               [WARN]
  update-inbox        pending .update/DONE_*.md drops awaiting the sorter            [INFO]
  library-db-fresh    generated SQLite lookup DB exists and matches sources           [ERROR]
  dead-md-refs         worker/process-facing backticked .md routes resolve            [WARN]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../Onnx_Advanced
REPO = ROOT.parent
AGENTERS = REPO / "Agenters"
SKIP_DIRS = {"_archive", "_integrated", ".git"}  # consumed-archive dirs are not worker-facing — don't dead-ref-scan

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"
FINDINGS: list[tuple[str, str, str]] = []


def finding(sev: str, check: str, msg: str) -> None:
    FINDINGS.append((sev, check, msg))


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def live_md_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*.md"):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        out.append(p)
    return out


def agenters_md_files() -> list[Path]:
    if not AGENTERS.exists():
        return []
    out = []
    for p in AGENTERS.rglob("*.md"):
        if any(part in SKIP_DIRS for part in p.relative_to(AGENTERS).parts):
            continue
        out.append(p)
    return out


def display_path(p: Path) -> str:
    try:
        return str(p.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


SEC_RE = re.compile(r"^#\s?(\d+[a-z]?)\.\s", re.MULTILINE)


def techniques_sections() -> list[str]:
    return SEC_RE.findall(read(ROOT / "techniques.md"))


def check_dup_sections(_only: bool) -> None:
    secs = techniques_sections()
    seen: dict[str, int] = {}
    for s in secs:
        seen[s] = seen.get(s, 0) + 1
    dups = sorted(k for k, v in seen.items() if v > 1)
    if dups:
        finding(ERROR, "dup-sections", f"duplicate section numbers in techniques.md: {', '.join('#' + d for d in dups)}")


def check_catalog_stale(_only: bool) -> None:
    cat = ROOT / "techniques_catalog.md"
    if not cat.exists():
        finding(ERROR, "catalog-stale", "techniques_catalog.md missing — run build_techniques_catalog.py")
        return
    m = re.search(r"source-sha256:([0-9a-f]+)", read(cat))
    actual = hashlib.sha256((ROOT / "techniques.md").read_bytes()).hexdigest()[:16]
    if not m or m.group(1) != actual:
        finding(ERROR, "catalog-stale",
                f"catalog sha {(m.group(1) if m else 'missing')} != techniques.md sha {actual} — regenerate: "
                "python Onnx_Advanced/.update/tools/build_techniques_catalog.py")


def check_catalog_coverage(_only: bool) -> None:
    cat = ROOT / "techniques_catalog.md"
    if not cat.exists():
        return
    cat_rows = set(re.findall(r"^\| #(\d+[a-z]?) \|", read(cat), re.MULTILINE))
    secs = set(techniques_sections())
    missing = sorted(secs - cat_rows)
    extra = sorted(cat_rows - secs)
    if missing:
        finding(ERROR, "catalog-coverage", f"sections missing from catalog: {', '.join('#' + s for s in missing[:12])}")
    if extra:
        finding(ERROR, "catalog-coverage", f"catalog rows with no section: {', '.join('#' + s for s in extra[:12])}")


def check_index_two_way(_only: bool) -> None:
    index_text = read(ROOT / "INDEX.md")
    exclude = {"INDEX.md", "README.md"}
    for p in sorted(ROOT.glob("*.md")):
        if p.name in exclude:
            continue
        if p.name not in index_text:
            finding(WARN, "index-two-way", f"root file not routed in INDEX.md: {p.name}")
    for tok in set(re.findall(r"`([A-Za-z0-9_./\\-]+\.md)`", index_text)):
        if any(ph in tok for ph in ("NNN", "*", "<")):
            continue
        cand = ROOT / tok
        if not cand.exists():
            finding(ERROR, "index-two-way", f"INDEX.md routes a missing file: {tok}")


def check_size_budgets(_only: bool) -> None:
    index_text = read(ROOT / "INDEX.md")
    m = re.search(r"## Size budgets.*?\n(\|.*?)(?:\n\n|\n\*)", index_text, re.DOTALL)
    if not m:
        finding(WARN, "size-budgets", "no '## Size budgets' table found in INDEX.md")
        return
    for fname, kb in re.findall(r"^\| ([A-Za-z0-9_.]+\.md) \| (\d+) \|", m.group(1), re.MULTILINE):
        p = ROOT / fname
        if not p.exists():
            finding(ERROR, "size-budgets", f"budget row for missing file: {fname}")
            continue
        actual = p.stat().st_size / 1024
        budget = int(kb)
        if actual > budget * 1.10:
            finding(ERROR, "size-budgets", f"{fname} = {actual:.1f} KB > 110% of budget {budget} KB — evict equivalent KB (charter rule 9)")
        elif actual > budget:
            finding(WARN, "size-budgets", f"{fname} = {actual:.1f} KB over budget {budget} KB")
    first_read = sum((ROOT / f).stat().st_size for f in
                     ("WORKER_PROTOCOL.md", "INDEX.md", "core.md", "traps.md") if (ROOT / f).exists()) / 1024
    if first_read > 75:
        finding(ERROR, "size-budgets", f"first-read set = {first_read:.1f} KB > 75 KB cap")


def check_routing_coverage(_only: bool) -> None:
    cat = ROOT / "techniques_catalog.md"
    if not cat.exists():
        return
    by_symptom = read(ROOT / "by_symptom.md") + read(ROOT / "WORKER_PROTOCOL.md")
    unrouted = []
    for num in re.findall(r"^\| #(\d+[a-z]?) \|", read(cat), re.MULTILINE):
        base = re.match(r"\d+", num).group(0)
        if int(base) < 48:
            continue
        if f"#{num}" not in by_symptom and f"#{base}" not in by_symptom:
            unrouted.append(num)
    if unrouted:
        finding(INFO, "routing-coverage",
                f"{len(unrouted)} sections >= #48 with no by_symptom/protocol mention: "
                f"{', '.join('#' + u for u in unrouted[:15])}{' …' if len(unrouted) > 15 else ''}")


def check_stub_expiry(_only: bool) -> None:
    for p in sorted(ROOT.glob("*.md")):
        if p.stat().st_size < 800 and re.search(r"moved to|pointer|redirect|merged into", read(p), re.IGNORECASE):
            if p.name != "README.md":
                finding(WARN, "stub-expiry", f"root stub with no remaining role? {p.name} ({p.stat().st_size} B) — archive if rg-clean")


def check_done_archival(_only: bool) -> None:
    done = ROOT / "Research" / "Done"
    if not done.exists():
        return
    for p in sorted(done.glob("*.md")):
        if p.name.startswith(("AUDIT_", "VERIFY_", "COMPLETED_")):
            continue
        if p.stat().st_size > 15 * 1024:
            finding(WARN, "done-archival",
                    f"Research/Done/{p.name} = {p.stat().st_size / 1024:.0f} KB — extracted deliverable body? "
                    "rule 7: archive to _archive/Research_<name>.md after extraction")


def check_changelog_cap(_only: bool) -> None:
    p = ROOT / "CHANGELOG.md"
    if p.exists() and p.stat().st_size > 60 * 1024:
        finding(WARN, "changelog-cap", f"CHANGELOG.md = {p.stat().st_size / 1024:.0f} KB > 60 — rotate older entries to _archive/")


def check_integrated_clean(_only: bool) -> None:
    integ = ROOT / ".update" / "_integrated"
    if integ.exists():
        n = len(list(integ.iterdir()))
        finding(WARN, "integrated-clean",
                f".update/_integrated exists ({n} files) — sorter protocol: archive consumed drops to _archive/ and remove the folder")


def check_required_reading(_only: bool) -> None:
    allowed = {"WORKER_PROTOCOL.md", "INDEX.md"}
    for p in live_md_files():
        if p.name in allowed or p.name.startswith(("AUDIT_", "VERIFY_")):
            continue  # audit docs quote the rule; they don't conscript
        if re.search(r"(?<!not )(?<!NOT )required reading", read(p), re.IGNORECASE):
            finding(WARN, "required-reading", f"'required reading' phrase in {p.relative_to(ROOT)} — only the protocol/INDEX may conscript readers")


def check_update_inbox(_only: bool) -> None:
    drops = sorted((ROOT / ".update").glob("DONE_*.md"))
    if drops:
        finding(INFO, "update-inbox", f"{len(drops)} pending .update drop(s) awaiting the sorter: {', '.join(d.name for d in drops[:5])}")


def check_library_db_fresh(_only: bool) -> None:
    script = ROOT / "library_db" / "build_library_db.py"
    if not script.exists():
        finding(ERROR, "library-db-fresh", "missing library_db/build_library_db.py")
        return
    proc = subprocess.run(
        [sys.executable, str(script), "--check", "--quiet"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        detail = (proc.stdout or proc.stderr or "").strip()
        msg = "generated SQLite library is missing/stale — run: python Onnx_Advanced/library_db/build_library_db.py"
        if detail:
            msg += f" ({detail})"
        finding(ERROR, "library-db-fresh", msg)

MD_TOKEN_RE = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./\\ -]*\.md)`")
DEAD_REF_EXEMPT = {"CHANGELOG.md", "traps_appendix.md"}  # historical/verbatim records
DEAD_REF_TOKEN_ALLOW = {
    "deep-research-report.md",  # the generic landing-filename in the filing protocol
    "DONE_2026-05-20T214530Z_task110_periodicity.md",  # illustrative example in .update/README
}


def check_dead_md_refs(_only: bool) -> None:
    """A worker-facing doc must never route to a file that doesn't open (2026-06-10
    preservation audit: with _archive rg-hidden, a dead path behaves like data loss)."""
    repo = REPO
    check_files = live_md_files() + agenters_md_files()
    # bare-name fallback: a `name.md` mention resolves if the name exists anywhere live
    live_names = {p.name for p in check_files}
    for p in check_files:
        if p.name in DEAD_REF_EXEMPT or p.name.startswith(("AUDIT_", "VERIFY_")):
            continue
        if "cards" in p.parts:  # retired card bodies are verbatim historical prompts
            continue
        try:
            rel_parts = p.relative_to(ROOT).parts
            # Research/Done process notes are dated historical records (the ledger stays checked)
            if rel_parts[:2] == ("Research", "Done") and not p.name.startswith("COMPLETED_"):
                continue
        except ValueError:
            rel_parts = ()
        text = read(p)
        seen: set[str] = set()
        for tok in MD_TOKEN_RE.findall(text):
            tok = tok.strip()
            if tok in seen or tok.split("/")[-1] in DEAD_REF_TOKEN_ALLOW or any(ph in tok for ph in ("NNN", "*", "<", "YYYY", "§")):
                continue
            seen.add(tok)
            rel = tok.replace("\\", "/")
            candidates = [ROOT / rel, p.parent / rel, repo / rel, repo.parent / rel,
                          AGENTERS / rel,
                          ROOT / "exemplars" / rel, ROOT / "Research" / rel,
                          repo / "The_400_Tasks" / rel,
                          repo / ".ai_instructions" / "playbooks" / rel]
            if re.fullmatch(r"task\d{3}[A-Za-z0-9_\-]*\.md", rel):
                candidates.append(repo / "The_400_Tasks" / "Task_by_Task" / rel)
            if any(c.exists() for c in candidates):
                continue
            if "/" not in rel and rel in live_names:
                continue
            finding(WARN, "dead-md-refs", f"{display_path(p)} routes to missing `{tok}`")


CHECKS = {
    "dup-sections": check_dup_sections,
    "catalog-stale": check_catalog_stale,
    "catalog-coverage": check_catalog_coverage,
    "index-two-way": check_index_two_way,
    "size-budgets": check_size_budgets,
    "routing-coverage": check_routing_coverage,
    "stub-expiry": check_stub_expiry,
    "done-archival": check_done_archival,
    "changelog-cap": check_changelog_cap,
    "integrated-clean": check_integrated_clean,
    "required-reading": check_required_reading,
    "update-inbox": check_update_inbox,
    "library-db-fresh": check_library_db_fresh,
    "dead-md-refs": check_dead_md_refs,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated check names")
    args = ap.parse_args()
    names = list(CHECKS) if not args.only else [n.strip() for n in args.only.split(",")]
    for n in names:
        if n not in CHECKS:
            print(f"unknown check: {n}", file=sys.stderr)
            return 2
        try:
            CHECKS[n](True)
        except Exception as e:  # a broken check must not hide other findings
            finding(ERROR, n, f"check crashed: {e!r}")
    errors = sum(1 for s, _, _ in FINDINGS if s == ERROR)
    warns = sum(1 for s, _, _ in FINDINGS if s == WARN)
    for sev, check, msg in sorted(FINDINGS, key=lambda f: ("ERROR WARN INFO".split().index(f[0]), f[1])):
        print(f"{sev:5}  {check:18}  {msg}")
    print(f"\nlib_lint: {errors} error(s), {warns} warning(s), {len(FINDINGS) - errors - warns} info — checks: {', '.join(names)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
