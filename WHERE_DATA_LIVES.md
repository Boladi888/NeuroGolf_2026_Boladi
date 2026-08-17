# WHERE DATA LIVES — read this before touching the filesystem

> **AGENT NOTICE.** The heavyweight local-only payload lives at `.notongit\NeuroGolf_nogit\` **inside this repo** (gitignored). The owner's checkout carries two Windows directory junctions into it — `.venv/` and `data/`. They look like normal directories to you, your code, and git. **Do NOT rebuild them — the data is fine, it just lives under `.notongit\`.** A fresh clone from GitHub has neither the payload nor the junctions.
>
> **Related:** `CURRENT_STATE.md` (next to this file) is the score / queue / state snapshot. This file is the layout / filesystem map. Read both at session start; they are intentionally small so they fit in any dumb agent's first context window.

## Layout (verified 2026-08-17 — the payload is now nested in-repo, not a sibling folder)

| Repo path | What it actually is | What it holds |
|---|---|---|
| `.notongit/NeuroGolf_nogit/` | **REAL in-repo directory, gitignored** (`.gitignore` → `.notongit/`). 101,995 files / 3.4 GB. | The whole local-only payload: `.venv`, `data`, `submissions` snapshot, `orchestrator`, `worktrees`, `BatchPilotArchive`. Moved here 2026-08-17 from the old sibling `<NOGIT_ROOT>`. |
| `.venv/` | **Junction** → `.notongit\NeuroGolf_nogit\.venv\` | Python 3.14.4 with pinned deps. **Do not re-create.** Run via `${NEUROGOLF_PYTHON}` or `.venv\Scripts\python.exe`. |
| `data/` | **Junction** → `.notongit\NeuroGolf_nogit\data\` | Kaggle source (400 tasks + utils) + competitor ONNX dumps (670 MB). **Do not re-download.** |
| `submissions/` | **REAL in-repo directory — CANONICAL.** Not a junction anymore. | The LIVE `submission.zip`, canonical backups, per-task ONNX builds. ⚠️ `.notongit\NeuroGolf_nogit\submissions\` still exists but is a **STALE ~2026-05-16 snapshot** (different submission.zip!) — never "restore" the junction or copy from it; only its `_alpha_a_scratch`/`_alpha_b_scratch` dirs are still referenced (by `tools/promote_task_work.py`). |
| `local_profiler/` | Ignored local scratch at the repo root. Older dumps may also live in `.notongit\NeuroGolf_nogit\local_profiler\`. | ONNX Runtime profiler dumps. Regenerable; not durable archive content. |
| `BatchPilotArchive/` | Gone from repo root (reference-only; lives only in `.notongit\NeuroGolf_nogit\BatchPilotArchive\`). | Archived pilot artifacts (3020 files). |
| `Team_Best_Onnx/` | **Gone.** Was a junction → `..\K-NeuroGolf-OnnxlyFans\Team_Submission\Backend\Team_Best_Onnx\`. The OnnxlyFans repo is no longer on C: (archived to the NAS), so the link is not recreated. Nothing under it was ever tracked. | The team's per-task best ONNX set (OnnxlyFans merge) — see the team archive on V:. |
| `Handoff\skills\handoff-workflows\` | **Real files now** (12 files), not a junction — the 2026-07 archive pass materialized the junction, and the Boladi_Skillz source folder it pointed at is empty. Still gitignored. | The HandOffz workflow skill the lanes consult (board cleanup, closeout, external packets, lane boundaries). Breadcrumb: `Handoff\skills\README.md`. |

Verify: `Get-ChildItem -Force | Where-Object {$_.LinkType}` from PowerShell (root) shows exactly two junctions, `.venv` and `data`, both targeting `.notongit\NeuroGolf_nogit\`. **Do NOT rebuild the payload; it is right there under `.notongit\`.**

## The ng_gate test corpus (Q: drive — NOT a junction)

> **AGENT NOTICE.** The shared verification harness `tools/ng_gate.py` (+ `tools/ng_harness/`) caches its ARC-GEN ground-truth corpus on a **separate drive**, not under the junctions:
>
> | Logical | Real location | What |
> |---|---|---|
> | corpus cache | `Q:\AIz\BoBrain\Data\domains\neurogolf\argen_cache\taskNNN.npz` | deterministic cached ARC-GEN pairs (uint8 color-index grids), generated once and reused by every agent. **Do NOT regenerate per-run** — call `ng_gate.py`, it caches. |
>
> Override with env var **`NEUROGOLF_ARGEN_CACHE`** (default is the Q: path above). The corpus is deterministic (`seed_offset=2026`), so it is identical across agents/runs. Use `py tools/ng_gate.py fill-corpus --tasks 1-400 --n 200000 --new-window` to pre-build it. Design + decisions: `tools/ng_harness/PLAN.md`.

**Use `ng_gate.py` instead of hand-writing ab/synth/battery scripts**, and run long verifications with `--new-window` (a visible console; closing it kills the whole run — no orphans). It self-throttles to ~80% CPU across all agents.

## What's tracked vs ignored inside `submissions/`

`submissions/` is a real in-repo directory. The tracked files are intentionally minimal; check `git ls-files -- submissions` for the current exact list. Typical tracked files:

- `submissions/submission.zip` — the live Kaggle upload payload
- `submissions/current_best_manifest.json` — current-best package manifest
- `submissions/README.md` — layout and submit-path notes

Everything else inside `submissions/` is gitignored: per-task `task*.onnx` outputs, fruit-named bundle dirs (e.g. `afr1ste_5377_grape/`), scratch dirs (e.g. `_target_slot_scan/`). All of those are regenerable by re-running the build pipeline against `data/`.

## Env vars (optional — junctions make them work without these)

Defaults in code resolve through the junctions, so env vars are not required on the owner's machine. Set them via `setx` (one-time, user-level) if you want belt-and-suspenders or you're running outside the main checkout:

```cmd
setx NEUROGOLF_MAIN_ROOT  "<REPO_ROOT>"
setx NEUROGOLF_NOGIT_ROOT "<NOGIT_ROOT>"
setx NEUROGOLF_DATA_DIR   "<NOGIT_ROOT>\data\competitions\neurogolf-2026"
setx NEUROGOLF_PYTHON     "<NOGIT_ROOT>\.venv\Scripts\python.exe"
setx NEUROGOLF_COMPETITOR_ARTIFACTS_DIR "<NOGIT_ROOT>\data\competitor_artifacts"
setx NEUROGOLF_ARGEN_CACHE "Q:\AIz\BoBrain\Data\domains\neurogolf\argen_cache"
```

Files that historically read these env vars included the retired `build_submission.py` dispatcher plus `tests/test_detect.py`, `tests/test_detector_builder_parity.py`, all of `tools/`, `explore.py`, `scan_tasks.py`, `verify_*.py`, and BatchPilot scripts. Code pattern was always `os.environ.get(VAR, default_under_repo_root)`.

See `.env.example` for the full list. To use `.env` instead of `setx`, copy to `.env` and source manually (no automatic dotenv loader is wired in).

## Future scratch convention

> Why: previous autonomous sessions parked ~10 ad-hoc scratch directories under `submissions/` (`_target_slot_scan/`, `_legalized_*/`, etc.), which mixed regenerable graph dumps with the live payload. Don't do that.

| Output type | Put it here | Tracked? | Promotion path |
|---|---|---|---|
| Analytical CSV / MD summary | `Logs/<date>_<topic>.csv\|md` | **Tracked** | Already the durable format. |
| Per-task ONNX builds | `submissions/<fruit_or_topic>_<date>/` | Ignored | Real dir; archive bundles survive locally (not in git). |
| Raw mining/probe dumps | `Logs/scratch/<topic>_<date>/` | **Ignored** (rule added 2026-05-10) | Promote summaries to `Logs/<date>_<topic>.csv\|md` at session end; rest can be purged. |
| Profiler JSON dumps | `local_profiler/` | Ignored | Regenerable — clear freely. |

If a session produces a single useful CSV out of a 14 MB scratch dir, copy the CSV to `Logs/` (tracked) and leave the rest in `Logs/scratch/`. Don't dump scratch ONNXs into `submissions/` again.

## If everything is on fire (recovery)

Only two junctions exist now: `.venv` and `data`, both pointing into `.notongit\NeuroGolf_nogit\` in this same repo. Re-create them only if you have verified the payload exists and the repo-root paths are missing. Never use `Remove-Item -Recurse` on an existing junction.

```powershell
$src = "<REPO_ROOT>"
$dst = "$src\.notongit\NeuroGolf_nogit"
foreach ($name in '.venv','data') {
  if (Test-Path "$src\$name") { Remove-Item "$src\$name" -Force }   # removes the LINK only, never -Recurse
  New-Item -ItemType Junction -Path "$src\$name" -Target "$dst\$name" | Out-Null
}
```

**⚠️ NEVER junction `submissions/` — it is a real directory holding the LIVE submission state. The `.notongit` copy is a stale 2026-05-16 snapshot; junctioning would effectively replace the live zip with a month-old one.**

If `.notongit/NeuroGolf_nogit/` itself is missing, it is on the NAS archive (`V:\Archive_CX_202607\Tech\Kaggle\K-NeuroGolf\.notongit\NeuroGolf_nogit`, 101,995 files / 3.4 GB). Failing that, the only really expensive thing to rebuild is `data/competitor_artifacts/` (~570 MB pulled from NAS). `data/competitions/` is `kaggle competitions download -c neurogolf-2026` (~15 s). `.venv/` is `python -m venv .venv && pip install -r requirements.txt` (~5 min).

## Don't do this

- ❌ `pip install -r requirements.txt` if `.venv/` already exists (junction or otherwise)
- ❌ `kaggle competitions download` if `data/competitions/neurogolf-2026/` already has 400 `*.json`s
- ❌ Dropping ad-hoc `_my_experiment/` dirs into `submissions/` — use `Logs/scratch/` instead
- ❌ `rm -rf` on a junction expecting to clear its target — `rm` on a junction removes the **junction**, not the target. (`Remove-Item` in PowerShell can recurse into the target if `-Recurse` is used; be careful.)

*Layout established 2026-05-10; re-verified + corrected 2026-06-06 (submissions/ found to be a real dir, not a junction); clarified during 2026-07 archive cleanup; rehomed 2026-08-17 — the local-only payload moved from the sibling `Py_cx\.notongit\` into this repo's own gitignored `.notongit\`, executing `Agenters/Janitor/notongit_rehome_plan.md`. See `.ai_instructions/03_PROJECT.md` for project context.*
