# -*- coding: utf-8 -*-
"""librarian_ctl.py — superseded proposal copy.

Live implementation: Agenters/Librarian/librarian_ctl.py.

The Librarian's drain step, scripted (sibling of captain_ctl.py). The captain relays each shipped
candidate's lesson into Agenters/Librarian/Updates_<Lane>/<UTC>-task<NNN>.lesson.md. The librarian runs
`digest`, reads ONE compact "integrate these" list (task + lane + snippet), integrates the §-numbers into
Onnx_Advanced/ by hand (that's the judgment — NOT scripted), then `archive`s what it integrated.

  digest  [--lane all|claude|codex]             list pending lessons: task, lane, first-line snippet, file
  show    --file <name>                         print one lesson's full text (when integrating it)
  archive --lane <lane> (--file <name> | --all) [--apply]   move integrated lesson(s) -> _integrated/

Token saver: the digest is the librarian's whole input — it never opens the inbox files to triage; it
opens only the lessons it integrates (via `show` or directly). SILENCE IS VALID: empty inbox -> nothing.
"""
import argparse, datetime, os, pathlib, sys

HERE = pathlib.Path(__file__).resolve()
HOME = HERE.parents[2]
LIB_DIR = HOME / "Agenters" / "Librarian"                           # /Updates_<Lane>/, /_integrated/
LANES = ("Claude", "Codex")

def lane_dir(lane): return LIB_DIR / f"Updates_{lane}"
def iter_lessons(lane):
    for p in sorted(lane_dir(lane).glob("*.lesson.md")):
        yield p
def snippet(p, n=160):
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip(): return line.strip()[:n]
    except Exception: pass
    return ""

def cmd_digest(args):
    lanes = LANES if args.lane == "all" else (args.lane.capitalize(),)
    found = 0
    for lane in lanes:
        rows = list(iter_lessons(lane))
        if not rows: continue
        if found == 0: print("== librarian digest (integrate these) ==")
        print(f"[{lane}] {len(rows)} pending:")
        for p in rows:
            task = p.name.split("-")[-1].replace(".lesson.md", "")
            print(f"  {task}  {p.name}\n      {snippet(p)}")
        found += len(rows)
    if not found: print("(no pending lessons — nothing to integrate)")

def cmd_show(args):
    for lane in LANES:
        p = lane_dir(lane) / args.file
        if p.exists(): print(p.read_text(encoding="utf-8", errors="replace")); return
    print(f"(not found: {args.file})")

def cmd_archive(args):
    lane = args.lane.capitalize(); d = lane_dir(lane); done = d / "_integrated"
    targets = [d / args.file] if args.file else list(iter_lessons(lane))
    moved = 0
    for p in targets:
        if not p.exists(): print(f"(skip, not found: {p.name})"); continue
        print(f"ARCHIVE {p.name}" + ("" if args.apply else "  (dry-run)"))
        if args.apply:
            done.mkdir(parents=True, exist_ok=True); os.replace(p, done / p.name)
        moved += 1
    print(f"-- {'archived' if args.apply else 'would archive'} {moved} --" + ("" if args.apply or not moved else "  (--apply to commit)"))

def main():
    ap = argparse.ArgumentParser(description="Librarian inbox drain (superseded proposal copy)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dg = sub.add_parser("digest"); dg.add_argument("--lane", default="all", choices=["all", "claude", "codex"]); dg.set_defaults(fn=cmd_digest)
    sh = sub.add_parser("show"); sh.add_argument("--file", required=True); sh.set_defaults(fn=cmd_show)
    ar = sub.add_parser("archive"); ar.add_argument("--lane", required=True, choices=["claude", "codex"])
    g = ar.add_mutually_exclusive_group(required=True); g.add_argument("--file"); g.add_argument("--all", action="store_true")
    ar.add_argument("--apply", action="store_true"); ar.set_defaults(fn=cmd_archive)
    args = ap.parse_args(); args.fn(args)

if __name__ == "__main__":
    sys.exit(main())
