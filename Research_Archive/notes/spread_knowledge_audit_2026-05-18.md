# Spread-knowledge audit — 2026-05-18

> Pre-shipping audit of `tools/spread_scan.py` findings before workers
> implement them. Documents what passed, what failed, and how to read
> the report correctly.

## What was verified

### 1. Byte-exact output on REAL task JSON inputs (not just random)

`tools/_spread_audit.py` extracts each task's ONNX from `submission.zip`,
runs the full `polish_candidate.py` pipeline, then compares the original
vs polished output via `onnxruntime.InferenceSession.run()` on each
encoded `data/competitions/neurogolf-2026/taskNNN.json` train+test input.

Result for top 10 tasks by saving: **10/10 byte-exact** on real inputs
(2–5 input pairs per task). The 3-random-input verification used by the
scanner is NOT a substitute, but it's a good filter — and the real-input
spot-check confirms its conclusions on the largest wins.

| Task | Polish steps | Δ cost | Real-input check |
|---|---|---:|---|
| task217 | onnxsim | -36,000 | OK byte-exact (4 pairs) |
| task370 | onnxsim | -8,017 | OK byte-exact (4 pairs) |
| task306 | onnxsim | -47 | OK byte-exact (3 pairs) |
| task020 | onnxsim | -4,198 | OK byte-exact (4 pairs) |
| task000 | scalar_broadcast_ones + slice_axis_collapse | -103 | OK byte-exact (2 pairs) |
| task283 | scalar_broadcast_ones | -899 | OK byte-exact (3 pairs) |
| task062 | onnxsim | -400 | OK byte-exact (5 pairs) |
| task354 | onnxsim | -1 | OK byte-exact (4 pairs) |
| task221 | onnxsim | -358 | OK byte-exact (5 pairs) |
| task036 | onnxsim | -128 | OK byte-exact (3 pairs) |

### 2. IR version + opset preservation

The Kaggle validator accepts opset ≥ 10 (per 2026-05-12 lesson; banned ops
are the real constraint). Our polished ONNXes preserve the same IR
version and opset as the originals — no opset bump, no IR drift.

Spot-checked: task000 (opset 16), task062 (opset 9), most others
(opset 10) all preserved exactly through the polish pipeline.

### 3. Cost re-scoring

For each top task, `score_network_v2` invoked **directly** (not through
the subprocess wrapper) returned the same cost delta as the polish
pipeline reported. No score-source skew.

## What needs correction in the original report

### The 61,498 byte headline was an UPPER BOUND, not the truth

`spread_scan.py` measures each pattern's savings **independently** against
the ORIGINAL model. When applied sequentially via `polish_candidate.py`,
patterns subsume each other — onnxsim's algebraic + DCE passes cover
most of what safe_opt would have done. So the SUM of per-pattern deltas
overstates the achievable savings.

Audit comparison on top tasks:

| Task | Scanner per-pattern sum | Polish (sequential) | Overstatement |
|---|---:|---:|---:|
| task217 | -36,000 | -36,000 | 0 (single pattern, no overlap) |
| task370 | -8,033 | -8,017 | -16 |
| task306 | -6,574 | **-47** | **-6,527** (safe_opt fully subsumed) |
| task020 | -4,404 | -4,198 | -206 |
| task062 | -800 | -400 | -400 |
| task354 | -371 | -1 | -370 |
| task036 | -248 | -128 | -120 |
| task221 | -358 | -358 | 0 |
| task283 | -899 | -899 | 0 |
| task000 | -103 | -103 | 0 (non-overlapping patterns) |

The fix: `tools/spread_polish_sweep.py` runs `polish_candidate.py` on
every task and writes `Logs/spread_scan_results/polish_sweep_<date>.csv`
with the TRUE sequential-pipeline delta per task. Both
`tools/spread_master_report.py` and `tools/spread_annotate_notes.py`
now consume that CSV and prefer it for headline numbers (with a
fallback to the upper-bound when the CSV isn't available).

### The advisory `crop_to_size_advisory` numbers are loose estimates

`pass_crop_to_size_advisory.py` estimates `~36000 - max_h*max_w*40` bytes
per `[...,30,30]` intermediate. The actual savings depend on:
* Whether all intermediates can be cropped (some are upstream-locked to
  full size by Gather/Reshape ops with constant shape arguments).
* The Slice + Pad bracket adds a small overhead.
* Workers need to implement crop-process-pad correctly per task.

Treat the advisory `Δ` column as an **upper bound** for prioritization,
not a guaranteed saving. Per-task implementation is required.

## Edge cases we found and handled

* **task175 onnxsim/safe_opt/fp16 all produce graphs that SIGFPE inside
  ORT** (Windows fatal int divide by zero). The pre-fix driver crashed
  with exit 127; the post-fix driver runs scoring + verification in
  subprocesses so the C-crash is isolated. All three are correctly
  reported as `verify-failed` (subprocess returns non-zero exit).
* **task354 sequential polish reports `scalar_broadcast_ones` as
  `score-failed`** — after onnxsim, the graph survives ORT inference but
  triggers a profiling-pass error in `score_network_v2`. The driver
  correctly rolls back (doesn't apply the failing pass) and ships only
  the onnxsim step. Real-input check confirms byte-exact output.

## What NOT to ship without an additional check

The audit found ZERO false-positive savings. But before shipping a
polished candidate to Kaggle as a real submission, treat the cost delta
as predicted and confirm it as a **diagnostic submission**:

1. Polish one high-impact task (e.g. task217 → -36k) via
   `tools/polish_candidate.py`.
2. Build a delta-only submission containing just that one task.
3. Submit. Confirm Kaggle LB delta matches the local score delta
   (within the public-test noise).
4. If yes → batch the remaining ~100 tasks. If no → reverse-engineer
   what diverged.

This matches the 2026-05-12 lesson: math-equivalent passes preserve
raw tensor values, but it's still cheap insurance to run ONE diagnostic
submission per pattern family before bulk-shipping.

## Tools recap (corrected version)

```bash
# Per-pattern scan (informative; the verdict TABLE is correct, the SUM is upper bound).
.venv/Scripts/python.exe tools/spread_scan.py

# Sequential-pipeline truth (use this for headline numbers).
.venv/Scripts/python.exe tools/spread_polish_sweep.py

# Per-task notes — auto-uses latest polish_sweep CSV if present.
.venv/Scripts/python.exe tools/spread_annotate_notes.py

# Master report — auto-uses latest polish_sweep CSV if present.
.venv/Scripts/python.exe tools/spread_master_report.py

# Pre-submit polish on a single candidate ONNX.
.venv/Scripts/python.exe tools/polish_candidate.py path/to/candidate.onnx
```
