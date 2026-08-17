# Spread-knowledge tools — usage guide

> Built 2026-05-18. Spreads optimization insights gained from individual
> task work (e.g. task000's `ones10 → scalar1` broadcast trick, task230's
> `Constant → graph.initializer` move) across all 401 tasks in
> `submissions/submission.zip` without manual one-by-one application.

## TL;DR

```bash
# 1. Scan: tries every known math-equivalent shrink on every task.
.venv/Scripts/python.exe tools/spread_scan.py
# -> Logs/spread_scan_results/spread_scan_<UTC date>.{json,csv,summary.md}

# 2. Update per-task notes with the findings.
.venv/Scripts/python.exe tools/spread_annotate_notes.py
# -> appends a `<!-- SPREAD:BEGIN -->...<!-- SPREAD:END -->` block to
#    every `The_400_Tasks/Task_by_Task/taskNNN.md`

# 3. Master report (top-N tasks by recoverable bytes, per-pattern
#    breakdown, advisory list).
.venv/Scripts/python.exe tools/spread_master_report.py
# -> Research/notes/spread_knowledge_findings_<UTC date>.md

# 4. Worker / submit-captain polish hook (single ONNX):
.venv/Scripts/python.exe tools/polish_candidate.py path/to/candidate.onnx
# -> writes candidate_polished.onnx + prints applied passes
```

## What gets scanned

`tools/spread_patterns/` holds one module per spreadable optimization
("pass"). Each module exposes a `PATTERN` of type `SpreadPattern`. The
driver auto-discovers them via `pkgutil.iter_modules`. Current set:

| Pattern | Category | Origin lesson |
|---|---|---|
| `onnxsim` | math-equivalent | 2026-05-12 onnxsim audit (+3.84 LB across 107 tasks) |
| `safe_opt` | math-equivalent | onnx_opt_safe + extended (dedupe/CSE/DCE + 18 elim passes) |
| `const_to_init` | math-equivalent | 2026-05-12 task230 (+7.07 LB across 7 tasks) |
| `clip_mul_to_min` | math-equivalent | 2026-05-12 (+0.57 LB across 3 tasks) |
| `scalar_broadcast_ones` | math-equivalent | 2026-05-18 task000 v2_scalar1 rev1 (-99 bytes) |
| `slice_axis_collapse` | math-equivalent | 2026-05-18 task000 v2_scalar1 rev2 (-10 bytes) |
| `fp16_selective` | selective | 2026-05-12 fp16_convert (+1.34 LB) **but ONLY kept if cost actually drops** |
| `crop_to_size_advisory` | advisory | 2026-05-11 task320 (#1 cost-reduction trick) |

The scanner is **conservative**: every "applied" verdict has been verified
byte-exact on 3 random one-hot inputs (or within 1e-3 tolerance for
fp16_selective) before being reported as a real saving. Patterns that
reduce cost on the model but fail verification are reported as
`verify-failed` and never shipped.

### Why these patterns are safe

Per the 2026-05-12 DIAG submissions, **Kaggle scores raw tensor values,
not decoded outputs**. So any rewrite that changes the raw output —
even one that produces identical decoded grids — is a risk. The patterns
in this scanner are all either:

* **Math-equivalent**: same tensor result by construction (onnxsim,
  safe_opt, const_to_init, clip_mul_to_min when mask ∈ {0,1},
  scalar_broadcast_ones when consumer broadcasts, slice_axis_collapse
  when axis 0 IS a no-op for the input shape).
* **Verified**: each "applied" candidate produces byte-exact output to
  the original on N dummy inputs before being reported.
* **Advisory** (`crop_to_size_advisory`): doesn't rewrite anything;
  just flags the opportunity for a worker to consider manually.

The selective pattern (`fp16_selective`) is the one exception — it's
kept only when the cost actually drops (the scanner verifies this via
`score_network_v2`), preventing the 2026-05-12
`onnxconverter_common` regression where Cast intermediates make cost go
UP on high-cost graphs.

## When to re-run

* **After any new candidate lands in `submission.zip`** — new candidates
  may not have had the polish pipeline applied; the scan finds them.
* **When adding a new pattern module** — drop a `pass_<name>.py` into
  `tools/spread_patterns/` and re-run.
* **Daily / weekly** — cheap insurance; takes ~15 min for 401 tasks.

## How a worker uses the per-task `## Spread-knowledge scan` block

The annotator writes a block like this into each `taskNNN.md`:

```
<!-- SPREAD:BEGIN -->
## Spread-knowledge scan (20260518)

Current canonical cost: **9681** (memory=9524, params=157, points≈15.82208).

Pattern sweep verdicts:

| Pattern | Verdict | Δ cost | Notes |
|---|---|---:|---|
| `scalar_broadcast_ones` | ✓ SAVES | -99 | replaced 1 uniform init |
| `slice_axis_collapse` | ✓ SAVES | -4 | collapsed axis-0 no-op |
| `crop_to_size_advisory` | ? | -32000 | max active grid is 10x10 |
| `fp16_selective` | ✗ regress | +67238 | Casts dominate |
...
**Free LB available**: ...
**Advisory** (crop_to_size_advisory): ...
**Do NOT apply**: fp16_selective...
<!-- SPREAD:END -->
```

A worker reopening the task sees at a glance:
* What's been tried automatically and works (free LB).
* What's task-specific work still on the table (advisory).
* What NOT to attempt (the sweep already verified regression).

Re-runs update the block in place — the marker-fenced region is
overwritten without touching the free-form `## Rule` /
`## Algorithm sketch` / `## Attempts` sections.

## When to use `polish_candidate.py`

Before any worker handoff or submission, run:

```bash
.venv/Scripts/python.exe tools/polish_candidate.py path/to/candidate.onnx
```

It runs the full math-equivalent pipeline (in order:
onnxsim → safe_opt → const_to_init → clip_mul_to_min →
scalar_broadcast_ones → slice_axis_collapse) and writes
`candidate_polished.onnx`. Each step is rolled back if verification
fails, so the polished file is guaranteed to produce the same raw
tensors as the input.

This is the kind of pre-flight check that should be in the
submit-captain's review gate.

## Adding a new pattern

1. Drop a new module under `tools/spread_patterns/pass_<name>.py`.
2. Define a function `_apply(model, task_id) -> SpreadResult`.
3. Export a `PATTERN = SpreadPattern(name=..., apply=_apply, ...)`.
4. Add a unit test under `tests/test_spread_patterns.py` showing a
   minimal trigger graph + the expected rewrite.
5. Re-run `tools/spread_scan.py`; the driver auto-discovers it.

The pattern is responsible for ensuring math-equivalence by
construction; the driver handles verification + scoring + reporting.
