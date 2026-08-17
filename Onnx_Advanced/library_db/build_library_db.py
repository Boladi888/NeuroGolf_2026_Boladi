#!/usr/bin/env python3
"""Build the generated Onnx_Advanced SQLite library.

The Markdown docs remain canonical. This script compiles a compact, cited,
status-aware lookup DB for agents. It stores structured summaries and source
anchors, not full copies of the library prose.

Usage:
    python Onnx_Advanced/library_db/build_library_db.py
    python Onnx_Advanced/library_db/build_library_db.py --check
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "onnx_library.sqlite"
MANIFEST_PATH = HERE / "build_manifest.json"
SCHEMA_PATH = HERE / "schema.sql"
SCHEMA_VERSION = "1"

PARSED_DOCS = {
    "techniques_catalog.md",
    "techniques.md",
    "task_techniques_index.md",
    "ort_reality.md",
    "traps.md",
    "cost_floor_reference.md",
    "scoring_internals.md",
    "WORKER_PROTOCOL.md",
    "core.md",
    "by_symptom.md",
}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def strip_md(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_>#]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clip(text: str, n: int = 420) -> str:
    text = strip_md(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def number_from_cost(text: str) -> int | None:
    m = re.search(r"(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})+|\d{2,7})(?![A-Za-z0-9])", text)
    return int(m.group(1).replace(",", "")) if m else None


def all_source_files() -> list[Path]:
    files: list[Path] = []
    for p in sorted(ROOT.rglob("*.md")):
        parts = p.relative_to(ROOT).parts
        if "_archive" in parts or "_integrated" in parts:
            continue
        files.append(p)
    # Include the scripts/schema that define generated behavior.
    files.extend([SCHEMA_PATH, HERE / "build_library_db.py", HERE / "lib_query.py"])
    return [p for p in files if p.exists()]


def source_manifest() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for p in all_source_files():
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        try:
            path = rel(p)
        except ValueError:
            path = str(p.relative_to(REPO)).replace("\\", "/")
        out[path] = {
            "sha256": sha,
            "short_sha": sha[:16],
            "bytes": len(raw),
            "mtime_utc": dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    return out


def manifest_sha(manifest: dict[str, dict[str, object]]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def line_map(path: Path) -> list[str]:
    return read(path).splitlines()


def source_line_for_header(path: Path, pattern: str) -> int | None:
    rx = re.compile(pattern)
    for i, line in enumerate(line_map(path), 1):
        if rx.search(line):
            return i
    return None


def insert_source_files(cur: sqlite3.Cursor, manifest: dict[str, dict[str, object]]) -> None:
    for path, meta in manifest.items():
        cur.execute(
            "INSERT INTO source_files(path, sha256, short_sha, bytes, mtime_utc) VALUES (?, ?, ?, ?, ?)",
            (path, meta["sha256"], meta["short_sha"], meta["bytes"], meta["mtime_utc"]),
        )


def add_search(cur: sqlite3.Cursor, kind: str, ref_id: str, title: str, body: str, source_path: str, source_line: int | None) -> None:
    cur.execute(
        "INSERT INTO search_index(kind, ref_id, title, body, source_path, source_line) VALUES (?, ?, ?, ?, ?, ?)",
        (kind, ref_id, title, clip(body, 900), source_path, source_line),
    )


def add_fact(
    cur: sqlite3.Cursor,
    fact_id: str,
    fact_type: str,
    claim: str,
    source_path: str,
    source_line: int | None,
    *,
    status: str = "current",
    confidence: str = "documented",
    task_id: str | None = None,
    op: str | None = None,
    technique_section: str | None = None,
    source_anchor: str = "",
    mutable: bool = False,
    reverify_after: str | None = None,
    superseded_by: str | None = None,
) -> None:
    cur.execute(
        """INSERT OR REPLACE INTO facts
           (fact_id, fact_type, claim, status, confidence, task_id, op, technique_section,
            source_path, source_line, source_anchor, mutable, reverify_after, superseded_by)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            fact_id,
            fact_type,
            clip(claim, 900),
            status,
            confidence,
            task_id,
            op,
            technique_section,
            source_path,
            source_line,
            source_anchor,
            1 if mutable else 0,
            reverify_after,
            superseded_by,
        ),
    )


def parse_techniques(cur: sqlite3.Cursor) -> None:
    cat = ROOT / "techniques_catalog.md"
    tech = ROOT / "techniques.md"
    tech_lines = line_map(tech)
    section_lines: dict[str, int] = {}
    for i, line in enumerate(tech_lines, 1):
        m = re.match(r"^#\s?(\d+[a-z]?)\.\s", line)
        if m:
            section_lines[m.group(1)] = i

    row_re = re.compile(r"^\| #(\d+[a-z]?) \| (.*?) \| (.*?) \| (.*?) \|$")
    for i, line in enumerate(line_map(cat), 1):
        m = row_re.match(line)
        if not m:
            continue
        section, title, tags, formerly = [x.strip() for x in m.groups()]
        if section == "§":
            continue
        source_line = section_lines.get(section)
        anchor = f"# {section}."
        cur.execute(
            """INSERT INTO techniques(section, title, tags, formerly, status, source_path, source_anchor, source_line)
               VALUES (?, ?, ?, ?, 'current', 'techniques.md', ?, ?)""",
            (section, strip_md(title), strip_md(tags), strip_md(formerly), anchor, source_line),
        )
        add_fact(
            cur,
            f"technique.{section}",
            "technique",
            f"#{section}: {title}",
            "techniques.md",
            source_line,
            technique_section=section,
            source_anchor=anchor,
        )
        add_search(cur, "technique", section, f"#{section} {title}", f"{title} {tags}", "techniques.md", source_line)


def parse_task_routes(cur: sqlite3.Cursor) -> None:
    path = ROOT / "task_techniques_index.md"
    family = ""
    klass = ""
    routes: list[dict[str, object]] = []
    for i, line in enumerate(line_map(path), 1):
        fm = re.match(r"^##\s+(.+)$", line)
        if fm and not line.startswith("###"):
            family = strip_md(fm.group(1))
            klass = ""
            continue
        cm = re.match(r"^###\s+(.+)$", line)
        if cm:
            klass = strip_md(cm.group(1))
            continue
        m = re.match(r"^- `task(\d{3})`(.*)$", line)
        if not m:
            continue
        task_id = f"task{m.group(1)}"
        rest = m.group(2).strip()
        cost = None
        parens = re.search(r"\(([^)]*)\)", rest)
        if parens:
            if "→" in parens.group(1):
                right = parens.group(1).split("→")[-1]
                cost = number_from_cost(right)
            else:
                cost = number_from_cost(parens.group(1))
        exemplar = ""
        if "→" in rest:
            exemplar = strip_md(rest.split("→", 1)[1])
        route_text = strip_md(line[2:])
        routes.append(
            {
                "task_id": task_id,
                "cost": cost,
                "family": family,
                "klass": klass,
                "route_text": route_text,
                "exemplar": exemplar,
                "line": i,
            }
        )
        cur.execute(
            """INSERT INTO task_routes
               (task_id, cost, family, technique_class, route_text, exemplar, status, source_path, source_line)
               VALUES (?, ?, ?, ?, ?, ?, 'current', 'task_techniques_index.md', ?)""",
            (task_id, cost, family, klass, route_text, exemplar, i),
        )
        add_fact(
            cur,
            f"task_route.{task_id}.{i}",
            "task_route",
            route_text,
            "task_techniques_index.md",
            i,
            task_id=task_id,
        )
        add_search(cur, "task_route", f"{task_id}:{i}", task_id, route_text, "task_techniques_index.md", i)

    grouped: dict[str, list[dict[str, object]]] = {}
    for r in routes:
        grouped.setdefault(str(r["task_id"]), []).append(r)
    for task_id, rows in grouped.items():
        costs = [int(r["cost"]) for r in rows if r["cost"] is not None]
        best_cost = min(costs) if costs else None
        families = ", ".join(sorted({str(r["family"]) for r in rows if r["family"]}))
        classes = ", ".join(sorted({str(r["klass"]) for r in rows if r["klass"]}))
        best_row = next((r for r in rows if r["cost"] == best_cost), rows[0])
        cur.execute(
            """INSERT INTO tasks(task_id, best_cost, route_count, families, technique_classes, best_route_text, source_path)
               VALUES (?, ?, ?, ?, ?, ?, 'task_techniques_index.md')""",
            (task_id, best_cost, len(rows), families, classes, str(best_row["route_text"])),
        )


def parse_ort_ops(cur: sqlite3.Cursor) -> None:
    path = ROOT / "ort_reality.md"
    row_re = re.compile(r"^\| `([^`]+)` \| (.*?) \| (.*?) \| (.*?) \|$")
    seen: set[str] = set()
    for i, line in enumerate(line_map(path), 1):
        m = row_re.match(line)
        if not m:
            continue
        op, accepted, rejected, notes = [strip_md(x) for x in m.groups()]
        if op in seen:
            continue
        seen.add(op)
        confidence = "kaggle_confirmed" if re.search(r"kaggle|grader", notes, re.I) else "probed"
        cur.execute(
            """INSERT INTO ort_ops(op, accepted_dtypes, rejected_dtypes, notes, status, confidence, source_path, source_line)
               VALUES (?, ?, ?, ?, 'current', ?, 'ort_reality.md', ?)""",
            (op, accepted, rejected, notes, confidence, i),
        )
        add_fact(
            cur,
            f"ort_op.{op}",
            "ort_op_dtype",
            f"{op}: accepts {accepted}; rejects {rejected}. {notes}",
            "ort_reality.md",
            i,
            op=op,
            confidence=confidence,
            mutable=True,
            source_anchor=op,
        )
        add_search(cur, "ort_op", op, op, f"accepts {accepted}; rejects {rejected}; {notes}", "ort_reality.md", i)


def parse_traps(cur: sqlite3.Cursor) -> None:
    path = ROOT / "traps.md"
    table_re = re.compile(r"^\| ([^|]+) \| ([^|]+) \| ([^|]+) \|$")
    block_re = re.compile(r"^>\s+\*\*(§[^*]+?)\*\*\s*(.*)$")
    for i, line in enumerate(line_map(path), 1):
        tm = table_re.match(line)
        if tm and tm.group(1).strip() not in {"#", "---"}:
            tid = strip_md(tm.group(1))
            name = strip_md(tm.group(2))
            fix = strip_md(tm.group(3))
            trap_id = f"trap.table.{tid}"
            cur.execute(
                """INSERT OR IGNORE INTO traps(trap_id, name, summary, trap_kind, status, source_path, source_line)
                   VALUES (?, ?, ?, 'table', 'current', 'traps.md', ?)""",
                (trap_id, name, fix, i),
            )
            add_fact(cur, trap_id, "trap", f"{name}: {fix}", "traps.md", i, source_anchor=tid)
            add_search(cur, "trap", trap_id, name, fix, "traps.md", i)
            continue
        bm = block_re.match(line)
        if bm:
            heading = strip_md(bm.group(1))
            body = strip_md(bm.group(2))
            slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:80]
            trap_id = f"trap.block.{slug}"
            cur.execute(
                """INSERT OR REPLACE INTO traps(trap_id, name, summary, trap_kind, status, source_path, source_line)
                   VALUES (?, ?, ?, 'doctrine', 'current', 'traps.md', ?)""",
                (trap_id, heading, clip(body, 700), i),
            )
            add_fact(cur, trap_id, "trap", f"{heading}: {body}", "traps.md", i, source_anchor=heading)
            add_search(cur, "trap", trap_id, heading, body, "traps.md", i)


def parse_cost_floors(cur: sqlite3.Cursor) -> None:
    path = ROOT / "cost_floor_reference.md"
    row_re = re.compile(r"^\| (task\d{3}) \| (.*?) \|$")
    for i, line in enumerate(line_map(path), 1):
        m = row_re.match(line)
        if not m:
            continue
        task_id, note = m.group(1), strip_md(m.group(2))
        cost = None
        bold = re.search(r"\*\*([^*]+)\*\*", m.group(2))
        if bold:
            cost = number_from_cost(bold.group(1))
        if cost is None:
            cost = number_from_cost(note)
        cur.execute(
            """INSERT OR REPLACE INTO cost_floors(task_id, note, cost, status, source_path, source_line)
               VALUES (?, ?, ?, 'current', 'cost_floor_reference.md', ?)""",
            (task_id, note, cost, i),
        )
        add_fact(
            cur,
            f"cost_floor.{task_id}",
            "cost_floor",
            f"{task_id}: {note}",
            "cost_floor_reference.md",
            i,
            task_id=task_id,
        )
        add_search(cur, "cost_floor", task_id, task_id, note, "cost_floor_reference.md", i)


def parse_library_fact_blocks(cur: sqlite3.Cursor) -> None:
    block_re = re.compile(r"(?P<fence>`{3}|~{3})library-fact\s*\n(?P<body>.*?)\n(?P=fence)", re.DOTALL)
    for p in all_source_files():
        if p.suffix != ".md" or rel(p) == "library_db/README.md":
            continue
        text = read(p)
        for m in block_re.finditer(text):
            fields: dict[str, str] = {}
            start_line = text[: m.start()].count("\n") + 1
            for raw in m.group("body").splitlines():
                if not raw.strip() or raw.lstrip().startswith("#") or ":" not in raw:
                    continue
                k, v = raw.split(":", 1)
                fields[k.strip()] = v.strip().strip('"')
            fact_id = fields.get("id")
            claim = fields.get("claim")
            if not fact_id or not claim:
                continue
            task_id = fields.get("task_id") or fields.get("task") or None
            if task_id and not task_id.startswith("task"):
                task_id = f"task{int(task_id):03d}"
            source_path = fields.get("source_path") or rel(p)
            source_line = start_line
            if fields.get("source_line"):
                try:
                    source_line = int(fields["source_line"])
                except ValueError:
                    source_line = start_line
            add_fact(
                cur,
                fact_id,
                fields.get("fact_type") or fields.get("type", "manual"),
                claim,
                source_path,
                source_line,
                status=fields.get("status", "current"),
                confidence=fields.get("confidence", "documented"),
                task_id=task_id,
                op=fields.get("op") or None,
                technique_section=fields.get("technique_section") or None,
                source_anchor=fields.get("source_anchor", ""),
                mutable=fields.get("mutable", "").lower() in {"1", "true", "yes"},
                reverify_after=fields.get("reverify_after") or None,
                superseded_by=fields.get("superseded_by") or None,
            )
            title = fields.get("subject") or fact_id
            body_bits = [claim]
            if task_id:
                body_bits.append(task_id)
            if fields.get("op"):
                body_bits.append(fields["op"])
            if fields.get("technique_section"):
                body_bits.append(f"techniques #{fields['technique_section']}")
            add_search(cur, "fact", fact_id, title, " ".join(body_bits), source_path, source_line)


def build_db() -> dict[str, object]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(SCHEMA_PATH)
    manifest = source_manifest()
    msha = manifest_sha(manifest)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(read(SCHEMA_PATH))
        cur = con.cursor()
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", SCHEMA_VERSION),
                ("generated_at", now),
                ("source_manifest_sha", msha),
                ("source_root", rel(ROOT)),
                ("builder", "library_db/build_library_db.py"),
            ],
        )
        insert_source_files(cur, manifest)
        parse_techniques(cur)
        parse_task_routes(cur)
        parse_ort_ops(cur)
        parse_traps(cur)
        parse_cost_floors(cur)
        parse_library_fact_blocks(cur)
        counts = {}
        for table in ("source_files", "techniques", "tasks", "task_routes", "ort_ops", "traps", "cost_floors", "facts", "search_index"):
            counts[table] = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cur.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", (f"count.{table}", str(counts[table])))
        con.commit()
    finally:
        con.close()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "database": rel(DB_PATH),
        "source_manifest_sha": msha,
        "counts": counts,
        "sources": manifest,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def check_fresh(quiet: bool = False) -> int:
    if not DB_PATH.exists():
        if not quiet:
            print(f"STALE: missing {rel(DB_PATH)}")
        return 1
    current = manifest_sha(source_manifest())
    try:
        con = sqlite3.connect(DB_PATH)
        db_sha = con.execute("SELECT value FROM metadata WHERE key='source_manifest_sha'").fetchone()
        schema = con.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        con.close()
    except sqlite3.Error as e:
        if not quiet:
            print(f"STALE: cannot read {rel(DB_PATH)}: {e}")
        return 1
    if not db_sha or db_sha[0] != current:
        if not quiet:
            print(f"STALE: DB source manifest {db_sha[0] if db_sha else 'missing'} != current {current}")
        return 1
    if not schema or schema[0] != SCHEMA_VERSION:
        if not quiet:
            print(f"STALE: DB schema {schema[0] if schema else 'missing'} != expected {SCHEMA_VERSION}")
        return 1
    if not quiet:
        print(f"fresh: {rel(DB_PATH)} source_manifest_sha={current[:16]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="return nonzero if DB is missing/stale")
    ap.add_argument("--quiet", action="store_true", help="suppress check output")
    args = ap.parse_args()
    if args.check:
        return check_fresh(quiet=args.quiet)
    payload = build_db()
    print(
        f"wrote {rel(DB_PATH)}: "
        + ", ".join(f"{k}={v}" for k, v in payload["counts"].items())
        + f", source_manifest_sha={payload['source_manifest_sha'][:16]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
