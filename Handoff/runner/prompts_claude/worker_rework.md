You are `{worker_id}` reworking `{task}` per reviewer direction.

Reviewer rework prompt:

```text
{reviewer_rework}
```

Your prior worker packet:

```text
{prior_worker_packet}
```

Address every reviewer point. Re-gate at `--n 8000` (visible + truth/ab; raw `--full`/n=200000 banned) + `profile_real_cost.py`. A win must have profile cost STRICTLY below the team-best bar `{bar}`. If the reviewer's suggested lever does not pay, say so with the exact gate/profile evidence rather than forcing a tie or a regression.

Keep gates FOREGROUND and cheapest-first (WORKER_PROTOCOL gate hygiene). If the locked hypothesis turns out visible-zero or proves a floor, return that certificate instead of burning more full-scale gates — a proven dead end is a valid outcome here, not a reason to keep grinding.

Before returning a reviewer-ready candidate, rerun `python tools/ng_safety.py model --task {task_num} --cand <candidate.onnx> --json` and classify it. `LOCAL_GOLD` is only for `PATH_A_BYTE_EQUAL` / `PATH_A_INHERITED` with AB `--n 8000` byte-equal proof, clean `ng_safety` PASS, `x86_required:false`, and `kaggle_probe_required:false`. Path-B non-byte-equal, fitted/approx, grader-gap, dynamic/profile uncertainty, or any safety warning is `PROBE_READY` unless the reviewer explicitly proves otherwise.

Return an updated worker packet (same fields as the contract), including `candidate_class`, `parent_source`, `parent_sha256`, `candidate_sha256`, `proof_commands_run`, `proof_results`, `safety` (raw `ng_safety.py` JSON), `x86_required`, `kaggle_probe_required`, and `status`.

## Bar + proof tooling (2026-07-03)

- **Authoritative bar:** start with `python tools/ng_current_bar.py --task NNN --json`; build strictly below its `live_cost` (it reconciles manifest/team-best/workbook and flags stale sources) — not a BoBrain / old-doc cost.
- **Name your saving:** before claiming a win, run `python tools/ng_graph_cost_diff.py --task NNN --live <current-best.onnx> --candidate <yours.onnx>` and put the named charged-structure delta in your packet.
