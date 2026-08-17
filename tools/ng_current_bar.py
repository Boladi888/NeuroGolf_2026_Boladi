#!/usr/bin/env python3
"""One authoritative "current bar" per task: live cost + sha + owner + freshness.

Problem this solves
--------------------
Workers, reviewers, and external packets keep working off *stale* per-task cost
targets. The team manifest (behind the leaderboard) is the live truth, but the
same number is mirrored — and drifts — across a workbook, staged-pending
sidecars, and local task notes. Three proposals ask for a single command that
reconciles all of these and prints ONE bar to use:

  * ``Agenters/_proposals/codex_p43_onnx_win_actions_2026-07-03.md``
    ("Make Strict-Gain Checks Impossible To Skip")
  * ``Agenters/_proposals/codex_p44_onnx_win_actions_2026-07-03.md``
    ("Enforce One Current-Bar Source")
  * ``Agenters/_proposals/codex_external_partner_onnx_win_actions_2026-07-03.md``
    ("Put All External Work Through A Current-Bar Gate")

Usage
-----
    python tools/ng_current_bar.py --task 50
    python tools/ng_current_bar.py --task 50 --json

Authoritative source
--------------------
The team manifest CSV (``Team_Submission/Backend/manifest.csv``) is the reconcile
source behind the leaderboard, so it wins. The per-task Team_Best ONNX
(``Team_Best_Onnx/{task}.onnx``) is the artifact the manifest sha *should* point
at; when the manifest is missing/incomplete we fall back to the Team_Best hash +
a cost recomputed from bytes. Every other source (workbook, staged-pending
sidecars, local task_state) is labelled and flagged ``stale`` when it disagrees
with the authoritative cost or sha.

This tool is strictly read-only: it reads files and computes sha256 hashes. It
never writes state, touches Kaggle, or edits the board.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# --- Reuse the shared repo-root resolver -------------------------------------
# neurogolf_paths already knows how to find the canonical NeuroGolf root even
# from a worktree checkout; use it instead of hardcoding the repo path.
REPO_ROOT_FROM_FILE = Path(__file__).resolve().parent.parent
if str(REPO_ROOT_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FROM_FILE))

try:
    from tools.neurogolf_paths import resolve_canonical_root
except Exception:  # pragma: no cover - fall back to file-relative root
    def resolve_canonical_root(start: Path | None = None) -> Path:  # type: ignore
        return REPO_ROOT_FROM_FILE


# Cost tolerance: manifest cost and a freshly-recomputed cost can differ by a
# rounding/scorer-version byte or two; treat sources within this many cost units
# of the authoritative value as agreeing (not stale). Sha mismatch always wins
# over the tolerance (a different sha means a different artifact).
COST_TOLERANCE = 0


# ---------------------------------------------------------------------------
# Path resolution for the OnnxlyFans backend (the team submission repo).
#
# The OnnxlyFans repo is a *sibling* of the NeuroGolf repo, and the repo root
# exposes a ``Team_Best_Onnx`` junction that points into its Backend dir. We
# resolve the backend from (in order): an env override, the junction target,
# the conventional sibling path, then a couple of legacy fallbacks. Whatever we
# find is verified by the presence of ``manifest.csv``.
# ---------------------------------------------------------------------------

ONNXLYFANS_ROOT_ENV = "NEUROGOLF_ONNXLYFANS_ROOT"


def _candidate_backend_dirs(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []

    env = None
    import os

    env = os.environ.get(ONNXLYFANS_ROOT_ENV)
    if env:
        env_root = Path(env).expanduser()
        candidates.append(env_root / "Team_Submission" / "Backend")
        candidates.append(env_root)  # in case env already points at Backend

    # The repo-root junction: Team_Best_Onnx -> <onnxlyfans>/Team_Submission/Backend/Team_Best_Onnx
    team_best_junction = repo_root / "Team_Best_Onnx"
    try:
        resolved = team_best_junction.resolve()
        # resolved == .../Backend/Team_Best_Onnx ; its parent is Backend
        if resolved.name.lower() == "team_best_onnx":
            candidates.append(resolved.parent)
    except OSError:
        pass

    # Conventional sibling layout: <parent>/K-NeuroGolf-OnnxlyFans/Team_Submission/Backend
    parent = repo_root.parent
    candidates.append(parent / "K-NeuroGolf-OnnxlyFans" / "Team_Submission" / "Backend")

    # In-repo copy, if a checkout ever nests it (defensive, not expected).
    candidates.append(repo_root / "K-NeuroGolf-OnnxlyFans" / "Team_Submission" / "Backend")

    return candidates


def resolve_onnxlyfans_backend(repo_root: Path) -> Path | None:
    """Return the OnnxlyFans ``Team_Submission/Backend`` dir, or None."""
    seen: set[str] = set()
    for cand in _candidate_backend_dirs(repo_root):
        try:
            key = str(cand.resolve()).lower()
        except OSError:
            key = str(cand).lower()
        if key in seen:
            continue
        seen.add(key)
        if (cand / "manifest.csv").exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def normalize_task(value: str) -> tuple[int | None, str]:
    """Return (task_int, canonical 'taskNNN' string) from '50' / 'task050' / etc."""
    text = str(value or "").strip().lower()
    if text.startswith("task"):
        text = text[4:]
    if text.isdigit():
        n = int(text)
        return n, f"task{n:03d}"
    return None, str(value)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _norm_sha(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text or None


def _as_int_cost(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source readers. Each returns a "source dict" or None if the source has no row
# for this task. Source dicts have: name, cost, sha, owner, updated_utc, path,
# plus optional kind/note. The `stale`/`authoritative` flags are added later by
# reconcile().
# ---------------------------------------------------------------------------

def read_manifest_source(backend: Path, task_id: str) -> dict[str, Any] | None:
    """Team manifest CSV — the reconcile source behind the leaderboard (truth)."""
    path = backend / "manifest.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("task", "")).strip().lower() == task_id:
                return {
                    "name": "manifest",
                    "cost": _as_int_cost(row.get("cost")),
                    "sha": _norm_sha(row.get("onnx_sha256")),
                    "owner": (row.get("creator") or "").strip() or None,
                    "updated_utc": (row.get("updated_utc") or "").strip() or None,
                    "path": str(path),
                    "points": row.get("points"),
                }
    return None


def read_team_best_source(repo_root: Path, backend: Path | None, task_id: str) -> dict[str, Any] | None:
    """Live per-task best ONNX. Hash it; recompute cost from bytes if possible."""
    # Prefer the repo-root junction (what everyone else reads through), then the
    # real backend dir.
    candidates = [repo_root / "Team_Best_Onnx" / f"{task_id}.onnx"]
    if backend is not None:
        candidates.append(backend / "Team_Best_Onnx" / f"{task_id}.onnx")
    onnx_path = next((p for p in candidates if p.exists()), None)
    if onnx_path is None:
        return None
    sha = sha256_file(onnx_path)
    cost = _team_best_cost(onnx_path)
    return {
        "name": "team_best",
        "cost": cost,
        "sha": sha,
        "owner": None,
        "updated_utc": None,
        "path": str(onnx_path),
        "note": "sha256 of live Team_Best ONNX" + ("" if cost is not None else "; cost predictor unavailable"),
    }


def _team_best_cost(onnx_path: Path) -> int | None:
    """Recompute cost (memory+params) from ONNX bytes via cost_predictor, best-effort."""
    try:
        from tools.cost_predictor import predict  # imported lazily; needs onnx installed
    except Exception:
        return None
    try:
        r = predict(str(onnx_path))
    except Exception:
        return None
    if not isinstance(r, dict) or r.get("error"):
        return None
    return _as_int_cost(r.get("cost"))


def read_workbook_source(onnxlyfans_root: Path | None, task_id: str) -> dict[str, Any] | None:
    """OnnxlyFans workbook Tasks sheet — a human-facing mirror of the manifest."""
    if onnxlyfans_root is None:
        return None
    path = onnxlyfans_root / "NG_OnnxlyFans_TaskStatus.xlsx"
    if not path.exists():
        return None
    try:
        import openpyxl  # noqa: WPS433
    except Exception:
        return {
            "name": "workbook",
            "cost": None,
            "sha": None,
            "owner": None,
            "updated_utc": None,
            "path": str(path),
            "note": "openpyxl unavailable; workbook not read",
        }
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "workbook",
            "cost": None, "sha": None, "owner": None, "updated_utc": None,
            "path": str(path), "note": f"workbook read failed: {exc}",
        }
    try:
        if "Tasks" not in wb.sheetnames:
            return None
        ws = wb["Tasks"]
        header = [str(c.value).strip().lower() if c.value is not None else "" for c in next(ws.iter_rows(min_row=1, max_row=1))]
        idx = {name: i for i, name in enumerate(header)}
        task_col = idx.get("task", 0)
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row:
                continue
            key = str(row[task_col]).strip().lower() if row[task_col] is not None else ""
            if key != task_id:
                continue

            def cell(name: str) -> Any:
                i = idx.get(name)
                return row[i] if (i is not None and i < len(row)) else None

            return {
                "name": "workbook",
                "cost": _as_int_cost(cell("cost")),
                # sha12 is only the first 12 hex chars of the manifest sha.
                "sha": _norm_sha(cell("sha12")),
                "sha_is_prefix": True,
                "owner": (str(cell("current_best_by")).strip() or None) if cell("current_best_by") is not None else None,
                "updated_utc": (str(cell("updated_utc")).strip() or None) if cell("updated_utc") is not None else None,
                "path": str(path),
            }
        return None
    finally:
        wb.close()


def read_staged_pending_sources(onnxlyfans_root: Path | None, task_id: str) -> list[dict[str, Any]]:
    """Staged-pending sidecars that may *beat* the manifest but aren't live yet.

    Only the top level of each ``Staged_<creator>`` dir counts as pending; the
    ``_hidden_failed`` subtree holds probes that were rejected/hidden and must
    NOT be presented as a live-or-pending bar.
    """
    out: list[dict[str, Any]] = []
    if onnxlyfans_root is None:
        return out
    staged_root = onnxlyfans_root / "Team_Submission"
    if not staged_root.exists():
        return out
    for staged_dir in sorted(staged_root.glob("Staged_*")):
        if not staged_dir.is_dir():
            continue
        sidecar = staged_dir / f"{task_id}.json"
        if not sidecar.exists():
            continue  # top-level only: ignores _hidden_failed/**
        doc = read_json(sidecar)
        if not isinstance(doc, dict):
            continue
        onnx_path = staged_dir / f"{task_id}.onnx"
        out.append(
            {
                "name": f"staged_pending:{staged_dir.name.replace('Staged_', '').lower()}",
                "cost": _as_int_cost(doc.get("cost")),
                "sha": _norm_sha(doc.get("sha256") or doc.get("probe_sha")),
                "owner": (doc.get("creator") or "").strip() or None,
                "updated_utc": None,
                "path": str(sidecar),
                "kind": "pending",
                "candidate_status": doc.get("candidate_status"),
                "onnx_present": onnx_path.exists(),
            }
        )
    return out


def read_local_sidecar_source(repo_root: Path, task_id: str) -> dict[str, Any] | None:
    """Local task_state.csv row — tactical, frequently stale context."""
    path = repo_root / "The_400_Tasks" / "task_state.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("task", "")).strip().lower() == task_id:
                return {
                    "name": "local_task_state",
                    "cost": _as_int_cost(row.get("current_cost")),
                    "sha": None,  # this CSV carries no per-task ONNX sha
                    "owner": (row.get("current_src") or "").strip() or None,
                    "updated_utc": (row.get("last_attempted_at") or "").strip() or None,
                    "path": str(path),
                    "kind": "tactical",
                    "note": "local task_state.csv is tactical context, treated as likely-stale",
                }
    return None


def read_leaderboard_state(backend: Path | None) -> dict[str, Any] | None:
    """Global leaderboard reconcile snapshot (not a per-task cost)."""
    if backend is None:
        return None
    path = backend / "current_leaderboard_state.json"
    if not path.exists():
        return None
    doc = read_json(path)
    if not isinstance(doc, dict):
        return None
    return {
        "path": str(path),
        "current_public_score": doc.get("current_public_score"),
        "manifest_total": doc.get("manifest_total"),
        "updated_utc": doc.get("updated_utc"),
        "submission_sha256": _norm_sha(doc.get("submission_sha256")),
        "source": doc.get("source"),
    }


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def _sha_matches(authoritative_sha: str | None, src: dict[str, Any]) -> bool | None:
    """Compare a source sha to the authoritative sha. Returns None if we can't
    compare (missing sha on either side)."""
    src_sha = src.get("sha")
    if not authoritative_sha or not src_sha:
        return None
    if src.get("sha_is_prefix"):
        # workbook stores only a 12-char prefix; compare prefix-to-prefix.
        n = len(src_sha)
        return authoritative_sha[:n] == src_sha
    return authoritative_sha == src_sha


def reconcile(sources: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Pick the authoritative source and flag every other source as stale/fresh.

    Priority for "authoritative": manifest > team_best > workbook > anything
    else. Pending/tactical sources never become authoritative — they exist to be
    compared against the live bar.
    """
    by_name = {s["name"]: s for s in sources}
    priority = ["manifest", "team_best", "workbook"]
    authoritative = None
    for name in priority:
        s = by_name.get(name)
        if s is not None and (s.get("cost") is not None or s.get("sha")):
            authoritative = s
            break
    if authoritative is None:
        # No live source at all; fall back to the first source with any signal.
        for s in sources:
            if s.get("cost") is not None or s.get("sha"):
                authoritative = s
                break

    auth_cost = authoritative.get("cost") if authoritative else None
    auth_sha = authoritative.get("sha") if authoritative else None

    for s in sources:
        s["authoritative"] = authoritative is not None and s is authoritative
        if s is authoritative:
            s["stale"] = False
            continue

        reasons: list[str] = []
        sha_cmp = _sha_matches(auth_sha, s)
        cost_ok: bool | None = None
        if s.get("cost") is not None and auth_cost is not None:
            cost_ok = abs(s["cost"] - auth_cost) <= COST_TOLERANCE
            if not cost_ok:
                reasons.append(f"cost {s['cost']} != live {auth_cost}")
        if sha_cmp is False:
            reasons.append("sha != live")

        kind = s.get("kind")
        if kind == "pending":
            # A pending candidate is not "stale" — it's a not-yet-live proposal.
            # Flag it stale only if it merely restates the live bar (nothing new)
            # or is strictly worse; otherwise mark it pending-ahead/behind.
            if s.get("cost") is not None and auth_cost is not None:
                if s["cost"] < auth_cost:
                    s["stale"] = False
                    s["pending_delta"] = s["cost"] - auth_cost  # negative = would beat live
                else:
                    s["stale"] = True
                    s["pending_delta"] = s["cost"] - auth_cost
                    reasons.append("pending not below live")
            else:
                s["stale"] = False
            s["stale_reasons"] = reasons
            continue

        # Non-live, non-pending sources (workbook mirror, local task_state):
        # stale if cost or sha disagrees; if we couldn't compare anything, mark
        # unknown-freshness (stale=None) rather than pretending it's fresh.
        if cost_ok is None and sha_cmp is None:
            s["stale"] = None  # cannot verify
        else:
            s["stale"] = bool(reasons)
        s["stale_reasons"] = reasons

    return authoritative, sources


# ---------------------------------------------------------------------------
# Assembly + rendering
# ---------------------------------------------------------------------------

def build_bar(repo_root: Path, task_input: str) -> dict[str, Any]:
    task_int, task_id = normalize_task(task_input)
    backend = resolve_onnxlyfans_backend(repo_root)
    onnxlyfans_root = backend.parent.parent if backend is not None else None

    sources: list[dict[str, Any]] = []
    manifest = read_manifest_source(backend, task_id) if backend else None
    if manifest:
        sources.append(manifest)
    team_best = read_team_best_source(repo_root, backend, task_id)
    if team_best:
        sources.append(team_best)
    workbook = read_workbook_source(onnxlyfans_root, task_id)
    if workbook:
        sources.append(workbook)
    sources.extend(read_staged_pending_sources(onnxlyfans_root, task_id))
    local = read_local_sidecar_source(repo_root, task_id)
    if local:
        sources.append(local)

    authoritative, sources = reconcile(sources)
    leaderboard = read_leaderboard_state(backend)

    result: dict[str, Any] = {
        "task": task_id,
        "task_id": task_int,
        "live_cost": authoritative.get("cost") if authoritative else None,
        "live_sha256": authoritative.get("sha") if authoritative else None,
        "owner": authoritative.get("owner") if authoritative else None,
        "updated_utc": authoritative.get("updated_utc") if authoritative else None,
        "authoritative_source": authoritative.get("name") if authoritative else None,
        "sources": sources,
    }
    if leaderboard is not None:
        result["leaderboard_state"] = leaderboard
    if authoritative is None:
        result["error"] = f"no live source found for {task_id} (not in manifest / Team_Best / workbook)"
    if backend is None:
        result["warning"] = "OnnxlyFans backend (Team_Submission/Backend) not found; manifest/workbook/staged sources skipped"
    return result


def _fmt(value: Any) -> str:
    return "?" if value in (None, "") else str(value)


def render_human(result: dict[str, Any]) -> str:
    lines: list[str] = []
    task = result.get("task")
    auth = result.get("authoritative_source")
    live_cost = result.get("live_cost")
    live_sha = result.get("live_sha256")
    lines.append(f"Current bar for {task}")
    if result.get("error"):
        lines.append(f"  ERROR: {result['error']}")
    if result.get("warning"):
        lines.append(f"  WARNING: {result['warning']}")
    lines.append(
        f"  authoritative: {_fmt(auth)}  owner={_fmt(result.get('owner'))}  updated={_fmt(result.get('updated_utc'))}"
    )
    lines.append("")
    header = f"  {'source':<26} {'cost':>8} {'stale':>6}  {'sha (12)':<14} detail"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for s in result.get("sources", []):
        sha = s.get("sha")
        sha_disp = (sha[:12] if sha else "-")
        if s.get("authoritative"):
            stale_disp = "LIVE"
        elif s.get("stale") is True:
            stale_disp = "STALE"
        elif s.get("stale") is None:
            stale_disp = "?"
        else:
            stale_disp = "ok"
        detail_bits: list[str] = []
        if s.get("kind") == "pending":
            pd = s.get("pending_delta")
            if pd is not None:
                detail_bits.append(f"pending delta {pd:+d} vs live" + (" (would beat)" if pd < 0 else ""))
            else:
                detail_bits.append("pending")
            if s.get("candidate_status"):
                detail_bits.append(str(s["candidate_status"]))
        if s.get("stale_reasons"):
            detail_bits.append("; ".join(s["stale_reasons"]))
        if s.get("note"):
            detail_bits.append(str(s["note"]))
        detail = " | ".join(detail_bits)
        lines.append(f"  {s.get('name',''):<26} {_fmt(s.get('cost')):>8} {stale_disp:>6}  {sha_disp:<14} {detail}")

    lb = result.get("leaderboard_state")
    if lb:
        lines.append("")
        lines.append(
            f"  leaderboard: public_score={_fmt(lb.get('current_public_score'))} "
            f"updated={_fmt(lb.get('updated_utc'))} (global, not per-task)"
        )
    lines.append("")
    if live_cost is not None or live_sha is not None:
        lines.append(f"USE THIS BAR: cost={_fmt(live_cost)} sha={_fmt(live_sha)}")
    else:
        lines.append("USE THIS BAR: (none - no live source)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--task", required=True, help="Task number or id, e.g. 50 or task050")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of the human summary")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override the NeuroGolf repo root (defaults to the canonical root resolved from this file).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page (cp1252) that can't encode
    # some glyphs; force UTF-8 so output never crashes on a stray non-ASCII char.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve() if args.repo_root else resolve_canonical_root()
    result = build_bar(repo_root, args.task)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_human(result))
    # Exit non-zero when there is no authoritative bar so callers/CI can gate on it.
    return 0 if result.get("live_cost") is not None or result.get("live_sha256") is not None else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
