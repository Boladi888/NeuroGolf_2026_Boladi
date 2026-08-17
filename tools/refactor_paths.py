#!/usr/bin/env python3
"""Safe file move/rename with repo-wide reference updating.

Use this whenever you relocate or rename a doc/script and want every reference
to follow. Dry-run by default — nothing changes until you pass --apply.

WHY a tool: references in this repo are written two ways — repo-relative
(`Onnx_Advanced/Research/foo.md`) and library-relative (`Research/foo.md`,
relative to Onnx_Advanced/). A move must update BOTH forms. And it must match by
PATH (with a parent dir), never by bare basename — `README.md` / `core.md` etc.
collide across dozens of files (learned the hard way during the 2026-06-01 audit).

Usage (SRC/DST are repo-relative paths):
    # preview (default): shows the git mv + every reference that would change
    python tools/refactor_paths.py --move <SRC> <DST>
    # multiple moves at once
    python tools/refactor_paths.py --move <SRC1> <DST1> --move <SRC2> <DST2>
    # from a JSON plan {"src": "dst", ...}
    python tools/refactor_paths.py --plan moves.json
    # actually do it (git mv + rewrite references)
    python tools/refactor_paths.py --move <SRC> <DST> --apply

After --apply it prints any *bare-basename* mentions it did NOT touch, so you can
eyeball whether a reference was written without its folder and needs a hand-fix.

LIMITATIONS (re-check by hand after a move):
  1. A moved file's OWN internal references are NOT rewritten (the tool skips every
     file in the move batch as a rewrite target). If file A and file B both move and
     A links to B, A's link to B won't be updated — grep the moved files afterwards.
  2. The tool does string-replacement of path forms; it does NOT recompute `../`
     relative links. Moving a file to a different directory DEPTH silently breaks any
     `../` link inside it. On --apply the tool now WARNS when a moved file changes
     depth and still contains `../` links — fix those by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEARCH_DIRS = ["Onnx_Advanced", "The_400_Tasks", "tools", ".ai_instructions", "Logs", "docs"]
EXCLUDE = {".git", ".venv", "__pycache__", "node_modules", "Research_Archive", ".notongit", ".hermes"}
TEXT_EXT = (".md", ".py", ".json", ".txt", ".csv", ".yaml", ".yml", ".toml")
LIB_ROOT = "Onnx_Advanced/"  # files referenced relative to this drop the prefix


def rel(p: Path) -> str:
    return p.resolve().relative_to(REPO).as_posix()


def ref_forms(src_repo: str, dst_repo: str) -> list[tuple[str, str]]:
    """Reference (old, new) string pairs to rewrite, longest first."""
    forms = [(src_repo, dst_repo)]
    if src_repo.startswith(LIB_ROOT) and dst_repo.startswith(LIB_ROOT):
        forms.append((src_repo[len(LIB_ROOT):], dst_repo[len(LIB_ROOT):]))  # library-relative
    # longest first so the repo-relative form is replaced before the lib-relative substring
    return sorted(set(forms), key=lambda t: -len(t[0]))


def corpus() -> list[Path]:
    out = []
    for base in SEARCH_DIRS:
        b = REPO / base
        if not b.exists():
            continue
        for dp, dn, fn in os.walk(b):
            dn[:] = [d for d in dn if d not in EXCLUDE]
            for f in fn:
                if f.endswith(TEXT_EXT):
                    out.append(Path(dp) / f)
    out += [REPO / f for f in os.listdir(REPO) if f.endswith(".md") and (REPO / f).is_file()]
    return out


def anchored(path_str: str) -> re.Pattern:
    # match the path only when preceded by a non-path char (start, space, `, (, /, ", ', =, :)
    return re.compile(r"(?<![A-Za-z0-9_./\\-])" + re.escape(path_str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--move", nargs=2, action="append", metavar=("SRC", "DST"), default=[])
    ap.add_argument("--plan", type=Path, help="JSON {src: dst, ...}")
    ap.add_argument("--apply", action="store_true", help="perform git mv + rewrite refs (default: dry-run)")
    args = ap.parse_args()

    moves: list[tuple[str, str]] = [(s, d) for s, d in args.move]
    if args.plan:
        moves += list(json.loads(args.plan.read_text(encoding="utf-8")).items())
    if not moves:
        ap.error("no moves given (--move SRC DST or --plan file.json)")

    # normalize to repo-relative posix
    norm = []
    for s, d in moves:
        sp, dp = (REPO / s), (REPO / d)
        if not sp.exists():
            print(f"ERROR: source not found: {s}", file=sys.stderr); return 2
        norm.append((rel(sp), rel(dp)))

    files = corpus()
    text = {f: f.read_text(encoding="utf-8", errors="replace") for f in files}
    src_repos = {s for s, _ in norm}
    total_refs = 0
    warns: list[str] = []

    for src_repo, dst_repo in norm:
        print(f"\n=== {src_repo}  ->  {dst_repo} ===")
        forms = ref_forms(src_repo, dst_repo)
        print("  reference forms:", " ; ".join(f"'{o}'->'{n}'" for o, n in forms))
        per_move = 0
        for f in files:
            if rel(f) in src_repos:
                continue
            t = text[f]
            new = t
            for old, repl in forms:
                new = anchored(old).sub(repl, new)
            if new != t:
                hits = sum(len(anchored(o).findall(t)) for o, _ in forms)
                per_move += hits
                print(f"    {rel(f)}: {hits} ref(s)")
                if args.apply:
                    f.write_text(new, encoding="utf-8")
                    text[f] = new
        # bare-basename safety net: mentions of the filename without a folder we did NOT rewrite
        base = Path(src_repo).name
        bare = anchored(base)
        bare_only = []
        for f in files:
            if rel(f) in src_repos:
                continue
            t = text[f]
            # count bare basename mentions NOT part of any path form
            stripped = t
            for old, _ in forms:
                stripped = stripped.replace(old, "")
            if bare.search(stripped):
                bare_only.append(rel(f))
        if bare_only:
            warns.append(f"  {base}: bare-name mention (no folder) in {len(bare_only)} file(s) — REVIEW: {bare_only[:6]}")
        print(f"  -> {per_move} reference(s) {'rewritten' if args.apply else 'would change'}")
        total_refs += per_move
        if args.apply:
            (REPO / dst_repo).parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["git", "mv", src_repo, dst_repo], cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0:
                # fall back to plain move if not git-tracked
                os.replace(REPO / src_repo, REPO / dst_repo)
                print(f"  (git mv failed [{r.stderr.strip()[:60]}]; used os.replace)")
            else:
                print(f"  git mv OK")

    # depth-change relative-link check: a file moved to a different directory depth
    # may contain `../` links that now resolve to the wrong place (the tool does not
    # recompute them). Flag any moved file that changed depth and still has `../`.
    rel_link = re.compile(r"(?:`|\]\()\.\./")
    for src_repo, dst_repo in norm:
        if src_repo.count("/") == dst_repo.count("/"):
            continue
        p = REPO / (dst_repo if args.apply else src_repo)
        if not p.exists():
            continue
        n = len(rel_link.findall(p.read_text(encoding="utf-8", errors="replace")))
        if n:
            warns.append(f"  {dst_repo}: DEPTH CHANGE + {n} `../` link(s) — re-check relative paths by hand (tool does not recompute `../`).")

    print(f"\n{'APPLIED' if args.apply else 'DRY-RUN'}: {len(norm)} move(s), {total_refs} reference(s) {'rewritten' if args.apply else 'to change'}.")
    if warns:
        print("\n[!] BARE-BASENAME REVIEW (paths written without a folder — not auto-rewritten):")
        for w in warns:
            print(w)
    if not args.apply:
        print("\n(dry-run; re-run with --apply to perform)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
