#!/usr/bin/env python3
"""Turn a repo-less external/partner packet into a task-local reproducible build.

Web-only external agents (Claude-app, ChatGPT web, paid HandOff partners) return
prose + fenced builder code in a chat page or a HandOff deliverable. This tool
parses the return-contract fields, extracts the builder code + any referenced
attachments, and writes a task-local intake folder with `build_next.ps1`,
`first_gate.ps1`, and `review_summary.md`.

A packet missing the graph-op or charged-tensor fields is REJECTED (nonzero exit)
-- an "analysis only" report is not acceptable external work.

Contract: Agenters/External_Intake/intake_contract.md
Path spec: codex_external_partner_onnx_win_actions_2026-07-03.md (P0 repo-less intake)

Usage:
    python tools/external_packet_intake.py --task N --packet PATH [--agent ID] [--out DIR]

Default out dir:
    The_400_Tasks/Task_by_Task/taskNNN/external_intake/<agent>/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Field name -> tuple of accepted aliases (all normalized: lowercase, non-alnum
# collapsed to '_'). Mirrors the table in intake_contract.md.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "graph_ops": ("graph_ops", "chosen_graph_ops", "ops", "op_list", "graph_op"),
    "charged_tensors": (
        "charged_tensors",
        "charged_tensors_expected",
        "expected_charged_tensors",
    ),
    "expected_params": ("expected_params", "params_expected", "param_count"),
    "expected_memory": ("expected_memory", "mem", "memory"),
    "current_bar_used": (
        "current_bar_used",
        "baseline_used",
        "current_bar",
        "bar_used",
        "baseline",
    ),
    "validation_command": (
        "validation_command",
        "first_local_command_to_run",
        "first_local_command",
        "validation",
        "validate_cmd",
    ),
    "decision": ("decision", "packet_class", "class", "verdict"),
    # Optional descriptive fields surfaced in the summary when present.
    "candidate_name": ("candidate_name", "candidate", "onnx_name"),
    "expected_cost": ("expected_cost", "cost", "candidate_cost"),
    "core_idea": ("core_idea", "idea", "summary"),
    "formulation_families_considered": (
        "formulation_families_considered",
        "formulation_families",
        "families_searched",
        "families_considered",
    ),
    "failure_reason": ("failure_reason", "why_not_viable", "blocker", "failure_kind"),
    "next_route": ("next_route", "next_creative_route", "route_worth_trying"),
    "files_returned": ("files_returned", "attachments", "files"),
}

# Fields whose absence is a HARD reject (nonzero exit).
REQUIRED_HARD = ("graph_ops", "charged_tensors")

# Fields required for a paid-accept label (recorded, not a hard reject).
REQUIRED_FOR_ACCEPT = (
    "expected_params",
    "expected_memory",
    "current_bar_used",
    "validation_command",
    "decision",
)

VALID_DECISIONS = ("ready", "probe", "discard")

CURRENT_BAR_CMD = "python tools/ng_current_bar.py --task {task} --json"

CODE_FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]*)\r?\n(?P<body>.*?)```",
    re.DOTALL,
)
# `key: value` on a single markdown line (value must be non-empty on that line
# or continue -- we take the same-line remainder).
KV_LINE_RE = re.compile(r"^\s*[-*]?\s*`?(?P<key>[A-Za-z0-9 _+/-]+?)`?\s*:\s*(?P<val>.*?)\s*$")


def normalize_key(raw: str) -> str:
    """Lowercase, collapse any run of non-alphanumerics to a single '_'."""
    return re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")


def build_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            lookup[normalize_key(alias)] = canonical
    return lookup


ALIAS_LOOKUP = build_alias_lookup()


def parse_json_fields(text: str) -> dict[str, str]:
    """Pull contract fields from any top-level JSON object embedded in the text.

    Handles a whole-file JSON packet or a fenced ```json block. Values are
    stringified (lists joined with ', ').
    """
    fields: dict[str, str] = {}
    candidates: list[str] = []

    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    for m in CODE_FENCE_RE.finditer(text):
        if normalize_key(m.group("lang")) in ("json", ""):
            body = m.group("body").strip()
            if body.startswith("{"):
                candidates.append(body)

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        for key, val in obj.items():
            canonical = ALIAS_LOOKUP.get(normalize_key(str(key)))
            if canonical is None or canonical in fields:
                continue
            if isinstance(val, (list, tuple)):
                sval = ", ".join(str(v) for v in val)
            else:
                sval = str(val)
            sval = sval.strip()
            if sval:
                fields[canonical] = sval
    return fields


def parse_markdown_fields(text: str) -> dict[str, str]:
    """Pull `key: value` contract fields from markdown lines (skipping code fences)."""
    fields: dict[str, str] = {}
    # Blank out fenced code so `key: value` inside code is not misread as a field.
    scrubbed = CODE_FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    for line in scrubbed.splitlines():
        m = KV_LINE_RE.match(line)
        if not m:
            continue
        canonical = ALIAS_LOOKUP.get(normalize_key(m.group("key")))
        if canonical is None or canonical in fields:
            continue
        val = m.group("val").strip().strip("`").strip()
        if val:
            fields[canonical] = val
    return fields


def parse_fields(text: str) -> dict[str, str]:
    """Merge JSON-derived and markdown-derived fields (JSON wins on conflict)."""
    fields = parse_markdown_fields(text)
    fields.update(parse_json_fields(text))  # JSON is more structured -> precedence
    return fields


def extract_builder_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(lang, body)] for builder code fences (python / powershell / ps1)."""
    blocks: list[tuple[str, str]] = []
    for m in CODE_FENCE_RE.finditer(text):
        lang = normalize_key(m.group("lang"))
        body = m.group("body").strip("\n")
        if lang in ("python", "py", "powershell", "ps1", "pwsh", "ps", "bash", "sh", ""):
            if lang == "" and body.strip().startswith("{"):
                continue  # bare JSON, not a builder block
            blocks.append((lang or "text", body))
    return blocks


def extract_referenced_attachments(text: str, fields: dict[str, str]) -> list[str]:
    """Collect referenced attachment names (.onnx / builder .py) from files field + inline."""
    names: set[str] = set()
    files_field = fields.get("files_returned", "")
    for tok in re.split(r"[,\n;]", files_field):
        tok = tok.strip().strip("`").strip()
        if tok:
            names.add(tok)
    for m in re.finditer(r"[\w./\\-]+\.(?:onnx|py)\b", text):
        names.add(m.group(0))
    return sorted(names)


def resolve_out_dir(args: argparse.Namespace, agent: str, timestamp: str) -> Path:
    if args.out:
        return Path(args.out).expanduser().resolve()
    task_dir = REPO_ROOT / "The_400_Tasks" / "Task_by_Task" / f"task{args.task:03d}"
    return task_dir / "external_intake" / agent / timestamp


def write_build_script(out_dir: Path, blocks: list[tuple[str, str]], agent: str, task: int) -> Path:
    path = out_dir / "build_next.ps1"
    lines = [
        "# build_next.ps1 -- runs the builder code extracted from the external packet.",
        f"# task {task:03d} | agent {agent}",
        "# GUARDRAIL: produces a CANDIDATE artifact only. Do not submit, stage, or",
        "# touch team best. Run from the repo root.",
        "$ErrorActionPreference = 'Stop'",
        "",
    ]
    if not blocks:
        lines += [
            "Write-Host 'No builder code block was found in the packet.'",
            "Write-Host 'This is a probe/discard packet -- build the graph by hand from'",
            "Write-Host 'review_summary.md, or see the referenced attachment.'",
            "exit 2",
        ]
    else:
        for i, (lang, body) in enumerate(blocks):
            ext = "py" if lang in ("python", "py") else ("ps1" if lang in ("powershell", "ps1", "pwsh", "ps") else "txt")
            snippet_name = f"builder_{i}.{ext}"
            (out_dir / snippet_name).write_text(body + "\n", encoding="utf-8")
            lines.append(f"Write-Host '--- builder block {i} ({lang}) -> {snippet_name} ---'")
            if ext == "py":
                lines.append(f"& '.venv\\Scripts\\python.exe' '{snippet_name}'")
            elif ext == "ps1":
                lines.append(f"& '.\\{snippet_name}'")
            else:
                lines.append(f"Write-Host 'Non-executable block saved to {snippet_name}; run it manually.'")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_first_gate(out_dir: Path, fields: dict[str, str], task: int) -> Path:
    path = out_dir / "first_gate.ps1"
    cmd = fields.get("validation_command", "").strip()
    lines = [
        "# first_gate.ps1 -- first local validation command for the candidate.",
        f"# task {task:03d}",
        "# Visible pass is only smoke; require truth/adversarial evidence before trusting.",
        "$ErrorActionPreference = 'Stop'",
        "",
        f"# Refresh the live bar before spending gates (reject_stale_bar if not strictly below):",
        f"# {CURRENT_BAR_CMD.format(task=task)}",
        "",
    ]
    if cmd:
        lines += [
            f"Write-Host 'Packet validation command:'",
            f"# From the packet's validation_command / first_local_command_to_run field:",
            cmd,
        ]
    else:
        lines += [
            "Write-Host 'Packet supplied no validation command. Use the standard gate ladder:'",
            f"# python tools/ng_gate.py visible --task {task} --cand <CAND.onnx>",
            f"# python tools/ng_gate.py truth   --task {task} --cand <CAND.onnx> --full",
            f"# python tools/ng_gate.py ab      --task {task} --cand <CAND.onnx> --full",
            "exit 2",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_review_summary(
    out_dir: Path,
    *,
    task: int,
    agent: str,
    timestamp: str,
    packet_path: Path,
    fields: dict[str, str],
    blocks: list[tuple[str, str]],
    attachments: list[str],
    missing_accept: list[str],
) -> Path:
    path = out_dir / "review_summary.md"
    decision = fields.get("decision", "").strip().lower()
    decision_note = decision if decision in VALID_DECISIONS else f"MISSING/invalid (got {decision!r})"

    def fval(name: str) -> str:
        return fields.get(name, "_(not supplied)_")

    lines = [
        f"# External Intake Review Summary -- task{task:03d}",
        "",
        f"- agent: `{agent}`",
        f"- intake timestamp (UTC): `{timestamp}`",
        f"- source packet: `{packet_path}`",
        f"- out dir: `{out_dir}`",
        f"- decision (packet class): **{decision_note}**",
        "",
        "## Current-bar check (RUN THIS BEFORE VERIFYING / PAYING)",
        "",
        "A paid external packet is only accepted if its claimed improvement is",
        "**strictly below** the live local bar at review time. Refresh it with:",
        "",
        "```powershell",
        CURRENT_BAR_CMD.format(task=task),
        "```",
        "",
        f"- current_bar_used (claimed by worker): {fval('current_bar_used')}",
        f"- expected_cost (claimed): {fval('expected_cost')}",
        "",
        "If the claimed improvement is not strictly below the live bar output above,",
        "classify **`reject_stale_bar`** and do not spend full gates.",
        "",
        "## Return-contract fields",
        "",
        f"- graph_ops: {fval('graph_ops')}",
        f"- charged_tensors: {fval('charged_tensors')}",
        f"- expected_params: {fval('expected_params')}",
        f"- expected_memory: {fval('expected_memory')}",
        f"- validation_command: {fval('validation_command')}",
        f"- candidate_name: {fval('candidate_name')}",
        f"- core_idea: {fval('core_idea')}",
        f"- formulation_families_considered: {fval('formulation_families_considered')}",
        f"- failure_reason: {fval('failure_reason')}",
        f"- next_route: {fval('next_route')}",
        "",
        "## Extracted artifacts",
        "",
        f"- builder code blocks extracted: {len(blocks)}"
        + ("".join(f"\n  - `builder_{i}` ({lang})" for i, (lang, _) in enumerate(blocks)) if blocks else " (none)"),
        f"- referenced attachments named in packet: "
        + (", ".join(f"`{a}`" for a in attachments) if attachments else "(none)"),
        "",
        "  NOTE: named attachments must be copied into this folder by the reviewer;",
        "  intake records the names but cannot fetch files from a chat page.",
        "",
        "## Generated run files",
        "",
        "- `build_next.ps1` -- runs the extracted builder block(s) to produce a candidate.",
        "- `first_gate.ps1` -- first local validation command (or the standard gate ladder).",
        "",
        "## Reviewer completeness check",
        "",
    ]
    if missing_accept:
        lines.append(
            "- INCOMPLETE for a paid-accept label -- missing required fields: "
            + ", ".join(f"`{m}`" for m in missing_accept)
            + ". Do not issue `accept_paid_*` until these are supplied."
        )
    else:
        lines.append("- All accept-required contract fields present.")
    if decision not in VALID_DECISIONS:
        lines.append(
            f"- `decision` is missing/invalid -- must be one of {', '.join(VALID_DECISIONS)}."
        )

    lines += [
        "",
        "## Rejection / accept labels",
        "",
        "See `Agenters/External_Intake/intake_contract.md`. This packet cleared the",
        "hard gate (graph_ops + charged_tensors present). Reviewer still chooses among:",
        "`reject_stale_bar`, `reject_duplicate`, `no_packet_value`,",
        "`accept_paid_ready`, `accept_paid_probe`.",
        "",
        "Record the outcome in `Agenters/_proposals/external_partner_spend_ledger.csv`.",
        "",
        "## Guardrails (fixed)",
        "",
        "Do not submit externally / Do not modify team best / Return candidate",
        "artifacts only / Submit captain decides staging after local gates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def reject(msg: str) -> "int":
    print(f"REJECT: {msg}", file=sys.stderr)
    print(
        "  This packet is analysis-only / non-buildable and is NOT acceptable "
        "external work.\n"
        "  Label: reject_unusable. Do NOT mark the paid HandOff task verified.\n"
        "  See Agenters/External_Intake/intake_contract.md for the required fields.",
        file=sys.stderr,
    )
    return 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="external_packet_intake.py",
        description=(
            "Turn a repo-less external/partner packet into a task-local reproducible "
            "build (build_next.ps1 + first_gate.ps1 + review_summary.md). Rejects a "
            "packet missing graph-op or charged-tensor fields."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default out dir:\n"
            "  The_400_Tasks/Task_by_Task/taskNNN/external_intake/<agent>/<timestamp>/\n\n"
            "Contract: Agenters/External_Intake/intake_contract.md"
        ),
    )
    parser.add_argument("--task", type=int, required=True, help="Task number, e.g. 374.")
    parser.add_argument(
        "--packet",
        required=True,
        help="Path to the external packet file (markdown or JSON).",
    )
    parser.add_argument(
        "--agent",
        default="unknown_agent",
        help="External agent id (used in the out-dir path). Default: unknown_agent.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output dir. Default: "
            "The_400_Tasks/Task_by_Task/taskNNN/external_intake/<agent>/<timestamp>/"
        ),
    )
    args = parser.parse_args(argv)

    packet_path = Path(args.packet).expanduser()
    if not packet_path.is_file():
        print(f"ERROR: packet not found: {packet_path}", file=sys.stderr)
        return 2
    try:
        text = packet_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = packet_path.read_text(encoding="utf-8", errors="replace")

    fields = parse_fields(text)

    # HARD gate: reject a packet missing graph-op or charged-tensor fields.
    missing_hard = [f for f in REQUIRED_HARD if not fields.get(f)]
    if missing_hard:
        return reject(
            "packet is missing required field(s): "
            + ", ".join(missing_hard)
            + f" (parsed fields: {sorted(fields) or 'none'})"
        )

    blocks = extract_builder_blocks(text)
    attachments = extract_referenced_attachments(text, fields)
    missing_accept = [f for f in REQUIRED_FOR_ACCEPT if not fields.get(f)]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    agent = normalize_key(args.agent) or "unknown_agent"
    out_dir = resolve_out_dir(args, agent, timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy the raw packet in for provenance.
    (out_dir / f"packet_source{packet_path.suffix or '.txt'}").write_text(text, encoding="utf-8")

    build_path = write_build_script(out_dir, blocks, agent, args.task)
    gate_path = write_first_gate(out_dir, fields, args.task)
    summary_path = write_review_summary(
        out_dir,
        task=args.task,
        agent=agent,
        timestamp=timestamp,
        packet_path=packet_path,
        fields=fields,
        blocks=blocks,
        attachments=attachments,
        missing_accept=missing_accept,
    )

    print(f"OK: intake written to {out_dir}")
    print(f"  - {build_path.name} ({len(blocks)} builder block(s))")
    print(f"  - {gate_path.name}")
    print(f"  - {summary_path.name}")
    if missing_accept:
        print(
            "  WARNING: incomplete for a paid-accept label -- missing "
            + ", ".join(missing_accept)
        )
    decision = fields.get("decision", "").strip().lower()
    if decision not in VALID_DECISIONS:
        print(
            f"  WARNING: decision is missing/invalid (must be one of {', '.join(VALID_DECISIONS)})"
        )
    print(
        f"  NEXT: refresh the live bar with `{CURRENT_BAR_CMD.format(task=args.task)}` "
        "before verifying/paying."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
