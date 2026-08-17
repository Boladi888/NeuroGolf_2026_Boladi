#!/usr/bin/env python3
"""build_techniques_catalog.py - regenerate techniques_catalog.md from techniques.md.

The catalog is the COMPLETE one-line-per-section index of techniques.md, so a worker
(or a captain writing a prompt) can discover every technique by grepping ~15 KB instead
of linear-reading ~340 KB. It is a GENERATED file: never edit it by hand; re-run this
script after any edit to techniques.md (the sorter's job; lib_lint.py flags staleness
via the embedded source-sha).

Usage (from anywhere):
    python Onnx_Advanced/.update/tools/build_techniques_catalog.py
Exit 0 on success. Stdlib only. Windows-safe. UTF-8 throughout.
"""
from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../Onnx_Advanced
SOURCE = ROOT / "techniques.md"
TARGET = ROOT / "techniques_catalog.md"

HEADER_RE = re.compile(r"^#\s?(\d+[a-z]?)\.\s*(.+?)\s*$")
COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
FORMERLY_RE = re.compile(r"formerly\s+(?:an?\s+)?(?:duplicate\s+)?[\"']?#?\s?(\d+[a-z]?)", re.IGNORECASE)

# keyword -> tag, matched against the lowercased title. Deliberately coarse: tags are
# a browse aid, not taxonomy. Titles stay verbatim so grep-by-title always works.
TAG_RULES: list[tuple[str, str]] = [
    (r"einsum", "einsum"),
    (r"qlinearconv|qconv|quantiz", "qlinearconv"),
    (r"convtranspose", "convtranspose"),
    (r"\bconv\b|conv-|conv×|1×1 conv|3×3", "conv"),
    (r"\bbool\b|boolean", "bool"),
    (r"\bf16\b|fp16|float16|\bu8\b|uint8|\bint8\b|i8|dtype|\bcast\b|narrow", "dtype"),
    (r"\bpad\b|padding|negpad|negative.pad|crop|slice", "pad-crop"),
    (r"scatter|gather", "scatter-gather"),
    (r"topk|argmax|arg-max", "topk-argmax"),
    (r"flood|morpholog|dilat|erod|neighbor", "flood-morph"),
    (r"pack|nibble|\bbit\b|bitwise|bitand|bitshift|base-?\d+|place-value|limb", "packing"),
    (r"\brank\b|selector|\bcp\b|factoriz|low-rank|separable", "rank-separable"),
    (r"floor|proof|lower.bound|forced", "floor-proof"),
    (r"free.output|output.*free|scorer-invisible|free.input", "free-cost"),
    (r"one-?hot", "onehot"),
    (r"\bwhere\b", "where"),
    (r"\blut\b|lookup|table", "lut"),
    (r"reshape|transpose|depthtospace|d2s|spacetodepth|tile|stamp|grid", "shape-move"),
    (r"period|repeat|wrap", "periodicity"),
    (r"mask|stencil|template|detect", "detection"),
    (r"reduce|sum|cumsum|count|moment|marginal", "reduce"),
    (r"scoring|cost|price|reprice|credit|profile", "pricing"),
    (r"identity|algebra|fold|cse|dead", "algebra"),
]


def tags_for(title: str) -> str:
    low = title.lower()
    out: list[str] = []
    for pat, tag in TAG_RULES:
        if re.search(pat, low) and tag not in out:
            out.append(tag)
    return " ".join(out[:4]) if out else "-"


def main() -> int:
    if not SOURCE.exists():
        print(f"ERROR: {SOURCE} not found", file=sys.stderr)
        return 2
    raw = SOURCE.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    text = raw.decode("utf-8", errors="replace")

    rows: list[tuple[str, str, str, str]] = []  # (num, title, was, tags)
    seen: dict[str, int] = {}
    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if not m:
            continue
        num, rest = m.group(1), m.group(2)
        was = ""
        for c in COMMENT_RE.findall(rest):
            fm = FORMERLY_RE.search(c)
            if fm:
                was = fm.group(1)
        title = COMMENT_RE.sub("", rest).strip().rstrip("\\").strip()
        # markdown-table-safe
        title = title.replace("|", "/")
        seen[num] = seen.get(num, 0) + 1
        rows.append((num, title, was, tags_for(title)))

    dups = sorted(n for n, c in seen.items() if c > 1)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def sort_key(r: tuple[str, str, str, str]):
        m2 = re.match(r"(\d+)([a-z]?)", r[0])
        return (int(m2.group(1)), m2.group(2))

    rows_sorted = sorted(rows, key=sort_key)

    out: list[str] = []
    out.append("<!-- GENERATED FILE - DO NOT EDIT BY HAND.")
    out.append("     Regenerate after ANY techniques.md edit:")
    out.append("         python Onnx_Advanced/.update/tools/build_techniques_catalog.py")
    out.append(f"     source-sha256:{sha} sections:{len(rows)} generated:{stamp} -->")
    out.append("")
    out.append("# techniques_catalog.md — the complete §-index of `techniques.md` (GENERATED)")
    out.append("")
    out.append("**Use this file FIRST for discovery**: grep/scan here (~15 KB, every technique), then open")
    out.append("`techniques.md` at the §-number (`grep \"^# 89\\.\" techniques.md`). Titles are verbatim from the")
    out.append("source — title-grep works in both files. `was` = old number before a renumber (breadcrumb map).")
    out.append("Tags are coarse keyword auto-tags, a browse aid only. Routing by graph-symptom: `by_symptom.md`.")
    out.append("")
    if dups:
        out.append(f"**LINT: DUPLICATE SECTION NUMBERS DETECTED: {', '.join(dups)} — fix techniques.md.**")
        out.append("")
    out.append("| § | technique (title verbatim) | tags | was |")
    out.append("|---|---|---|---|")
    for num, title, was, tags in rows_sorted:
        out.append(f"| #{num} | {title} | {tags} | {('#' + was) if was else ''} |")
    out.append("")
    out.append("*Named sub-blocks (e.g. `§39 MECHANISM CONFIRMED`, `F16-CONV VERDICT`, `§39 GAP-ledger`) are not*")
    out.append("*top-level §s; grep them directly in `techniques.md`. Doctrine/traps live in `traps.md`, not here.*")
    out.append("")

    TARGET.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"wrote {TARGET.name}: {len(rows)} sections, sha {sha}, {TARGET.stat().st_size} bytes")
    if dups:
        print(f"WARNING: duplicate section numbers: {dups}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
