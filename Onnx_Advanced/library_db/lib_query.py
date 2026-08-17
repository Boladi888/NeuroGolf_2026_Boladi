#!/usr/bin/env python3
"""Query the generated Onnx_Advanced SQLite library.

Agents should use this before opening large Markdown files. Output is compact
and always cites the source path/line for follow-up reading.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "onnx_library.sqlite"
BUILD = HERE / "build_library_db.py"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def con() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SystemExit(f"missing {DB_PATH}; run: python {BUILD}")
    return sqlite3.connect(DB_PATH)


def cite(path: str, line: int | None) -> str:
    return f"{path}:{line}" if line else path


def dtype_list_has(text: str, dtype: str) -> bool:
    token = dtype.upper()
    return re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", text.upper()) is not None


def print_rows(rows: list[sqlite3.Row], fields: list[str]) -> None:
    for r in rows:
        print(" | ".join(str(r[f]) if r[f] is not None else "" for f in fields))


def cmd_status(_args: argparse.Namespace) -> int:
    rc = subprocess.call([sys.executable, str(BUILD), "--check"])
    with con() as db:
        db.row_factory = sqlite3.Row
        for row in db.execute("SELECT key, value FROM metadata WHERE key IN ('generated_at','source_manifest_sha','schema_version') ORDER BY key"):
            print(f"{row['key']}: {row['value']}")
    return rc


def cmd_task(args: argparse.Namespace) -> int:
    task = args.task if args.task.startswith("task") else f"task{int(args.task):03d}"
    with con() as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task,)).fetchone()
        if not row:
            print(f"no task row for {task}")
            return 1
        print(f"{row['task_id']} best_cost={row['best_cost']} routes={row['route_count']}")
        print(f"families: {row['families']}")
        print(f"classes: {row['technique_classes']}")
        print(f"best route: {row['best_route_text']}")
        print()
        for r in db.execute(
            "SELECT cost, family, technique_class, route_text, source_path, source_line FROM task_routes WHERE task_id=? ORDER BY cost IS NULL, cost, source_line",
            (task,),
        ):
            print(f"- cost={r['cost']} [{r['family']} / {r['technique_class']}] {r['route_text']} ({cite(r['source_path'], r['source_line'])})")
        facts = list(
            db.execute(
                "SELECT fact_type, claim, status, confidence, source_path, source_line FROM facts WHERE task_id=? AND status='current' ORDER BY fact_type, source_line LIMIT 12",
                (task,),
            )
        )
        if facts:
            print("\nfacts:")
            for f in facts:
                print(f"- [{f['fact_type']}/{f['confidence']}] {f['claim']} ({cite(f['source_path'], f['source_line'])})")
    return 0


def cmd_op(args: argparse.Namespace) -> int:
    with con() as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM ort_ops WHERE lower(op)=lower(?)", (args.op,)).fetchone()
        if not row:
            print(f"no ORT row for op {args.op}")
            return 1
        print(f"{row['op']} [{row['confidence']}] ({cite(row['source_path'], row['source_line'])})")
        print(f"accepted: {row['accepted_dtypes']}")
        print(f"rejected: {row['rejected_dtypes']}")
        print(f"notes: {row['notes']}")
        if args.dtype:
            accepted = dtype_list_has(row["accepted_dtypes"], args.dtype)
            rejected = dtype_list_has(row["rejected_dtypes"], args.dtype)
            if accepted and rejected:
                verdict = "accepted-and-rejected-listed (opset/notes-dependent)"
            elif accepted:
                verdict = "accepted-listed"
            elif rejected:
                verdict = "rejected-listed"
            else:
                verdict = "not-explicit"
            print(f"dtype {args.dtype}: {verdict}")
    return 0


def cmd_technique(args: argparse.Namespace) -> int:
    section = args.section.lstrip("#")
    with con() as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM techniques WHERE section=?", (section,)).fetchone()
        if not row:
            print(f"no technique section #{section}")
            return 1
        print(f"#{row['section']} {row['title']}")
        print(f"tags: {row['tags']}")
        print(f"source: {cite(row['source_path'], row['source_line'])}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    query = " ".join(args.query)
    limit = args.limit
    with con() as db:
        db.row_factory = sqlite3.Row
        try:
            rows = list(
                db.execute(
                    """SELECT s.kind, s.ref_id, s.title, snippet(search_fts, 3, '[', ']', ' … ', 12) AS body,
                              s.source_path, s.source_line
                       FROM search_fts
                       JOIN search_index s ON s.rowid = search_fts.rowid
                       WHERE search_fts MATCH ?
                       ORDER BY bm25(search_fts)
                       LIMIT ?""",
                    (query, limit),
                )
            )
        except sqlite3.OperationalError:
            like = f"%{query}%"
            rows = list(
                db.execute(
                    """SELECT kind, ref_id, title, body, source_path, source_line
                       FROM search_index
                       WHERE title LIKE ? OR body LIKE ?
                       LIMIT ?""",
                    (like, like, limit),
                )
            )
        if not rows:
            print("no search hits")
        for r in rows:
            print(f"- [{r['kind']}:{r['ref_id']}] {r['title']} — {r['body']} ({cite(r['source_path'], r['source_line'])})")
    return 0


def cmd_facts(args: argparse.Namespace) -> int:
    clauses = []
    params: list[object] = []
    if args.status:
        clauses.append("status=?")
        params.append(args.status)
    elif not args.include_stale:
        clauses.append("status='current'")
    if args.task:
        clauses.append("task_id=?")
        params.append(args.task if args.task.startswith("task") else f"task{int(args.task):03d}")
    if args.op:
        clauses.append("lower(op)=lower(?)")
        params.append(args.op)
    if args.type:
        clauses.append("fact_type=?")
        params.append(args.type)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with con() as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            f"""SELECT fact_id, fact_type, claim, status, confidence, source_path, source_line
                FROM facts {where}
                ORDER BY fact_type, fact_id
                LIMIT ?""",
            (*params, args.limit),
        )
        rows = list(rows)
        if not rows:
            print("no matching facts")
        for r in rows:
            print(f"- {r['fact_id']} [{r['fact_type']}/{r['status']}/{r['confidence']}] {r['claim']} ({cite(r['source_path'], r['source_line'])})")
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    task_rc = cmd_task(args)
    print("\nsearch hits:")
    task = args.task if args.task.startswith("task") else f"task{int(args.task):03d}"
    args.query = [task]
    args.limit = 8
    cmd_search(args)
    return task_rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("task")
    p.add_argument("task")
    p.set_defaults(fn=cmd_task)

    p = sub.add_parser("context")
    p.add_argument("task")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(fn=cmd_context)

    p = sub.add_parser("op")
    p.add_argument("op")
    p.add_argument("dtype", nargs="?")
    p.set_defaults(fn=cmd_op)

    p = sub.add_parser("technique")
    p.add_argument("section")
    p.set_defaults(fn=cmd_technique)

    p = sub.add_parser("search")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("facts")
    p.add_argument("--task")
    p.add_argument("--op")
    p.add_argument("--type")
    p.add_argument("--status", choices=("current", "stale", "disputed", "deprecated", "unknown"))
    p.add_argument("--include-stale", action="store_true")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(fn=cmd_facts)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
