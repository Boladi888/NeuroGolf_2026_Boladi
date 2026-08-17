# diagnostics_recipes.md — when you want X, run Y

Half the diagnostic scripts agents write already exist in `tools/agent_helpers/`. Use this table to find the existing one before writing your own.

All scripts run via:
```
.venv/Scripts/python tools/agent_helpers/<script>
```

Set `PYTHONIOENCODING=utf-8` on Windows if you see encoding errors.

## Recipe table

| When you want to... | Run this | Returns |
|---|---|---|
| **Status of task N** (cost, family, history pointers — the most common question) | `show_task.py --task 110` | One task_state row, readable. NEVER Read task_state.csv whole (1 MB) |
| **Everything about task N in one read** (~14 KB curated: state row, STATUS, rule, don't-dos, prompt, staged, prototypes, cited §s) | `context_pack.py --task 110` | Whale-safe pack — replaces 4-6 separate file reads at session start |
| Check repo invariants (dead refs, retired-system refs, oversize task journals, prompt contract, staging hygiene) | `python tools/repo_lint.py` (root tools/, not agent_helpers) | Findings table + `Logs/lint_last.md`; `--only task-md-size --task N` to self-check your journal before session close |
| Extract a task's canonical ONNX from submission.zip | `extract_task.py 110` | Drops `task110.onnx` into your CWD |
| Recover a probed-but-cleaned candidate | `rg "<cost>" Agenters/Submit_Captain submissions The_400_Tasks/Task_by_Task/taskNNN` then read the `_consumed/*.json` snapshot path and SHA | Pull `taskNNN.onnx` from `Agenters/Submit_Captain/_probe/_snapshots/*/submission.zip`; SHA-match the manifest, then rerun gates live |
| Inspect a graph (op histogram, nodes, inits, intermediates) | `inspect_graph.py task110.onnx` | Full text dump of the graph |
| Find the biggest intermediate (where memory is concentrated) | `largest_intermediate.py task110.onnx --top 10` | Top 10 intermediates by static bytes; auto-runs shape_inference if value_info was cleared |
| Compare two ONNX models structurally | `diff_graphs.py before.onnx after.onnx` | Op-type counts, init add/remove, side-by-side cost |
| Compare costs of two models with LB estimate | `score_compare.py a.onnx b.onnx` | cost/memory/params side-by-side + raw log_e delta + 0.48× "calibrated" line. **NOTE 2026-05-23**: the 0.48× output is for 100+ task megastacks only; single-task wildcards use raw `log_e × 1.0` (see `cost_floor_reference.md` Calibration). The helper's "Expected LB gain (calibrated)" under-predicts single-task wildcards by ~2×. |
| Find specific ops in a graph | `find_op.py task110.onnx ReduceSum` | All nodes of that op type with inputs/outputs |
| Find ops by name regex | `find_op.py task110.onnx --name-regex 'concat.*'` | All matching nodes |
| Show full attribute list per matched node | `find_op.py task110.onnx Conv --show-attrs` | Nodes + their attribute key-values |
| Run model + dump every intermediate tensor (debug verify failures) | `dump_intermediates.py model.onnx --tid 110 --pair-idx 0` | All node-output tensors with their shapes/sums |
| Find where two models diverge | `dump_intermediates.py candidate.onnx --tid 110 --pair-idx 0 --compare-with live.onnx` | First diverging intermediate (often the root cause) |
| Generate synthetic adversarial inputs + check against LIVE (silent-fail trap check) | `adversarial_against_live.py --candidate cand.onnx --tid 110 --n 50` | agreement %, top 3 disagreement details, PASS/BORDERLINE/FAIL verdict. **CANONICAL per-candidate adversarial check** (`tools/silent_fail_check.py` is the package-level arc-gen sweep, not per-candidate) |
| Confirm local data matches Kaggle's current versions | `check_data_freshness.py` | Stale-file list + re-pull command |
| Regenerate the auto-graph-diag section in every task .md | `regenerate_graph_diagnostics.py` | Updates all 401 task .md files (~5-10 min) |
| Regenerate just one task's diagnostic | `regenerate_graph_diagnostics.py task110` | Updates one task .md |

## Common workflows

### Workflow A0 — external/web-lane candidate triage

When a no-repo/web-app worker returns a task pack artifact, classify it before merge:

- **ready**: exact one-task zip (`taskNNN.onnx` at zip root only), ORT CPU loads/runs, original + ARC-GEN validation pass, measured memory/params/cost improve, and no banned constructs.
- **probe**: any validation miss, unsupported kernel, scorer-invalid/dynamic shape, high-theory route needing public check, or hidden-set hypothesis.
- **discard**: duplicate failed hash, known public-zero family, broad validation failure, no measurable gain, or packaging-only variant.

Require the returned note to include: task id, candidate name, baseline/candidate cost + memory + params, validation counts, zip contents, dominant removed tensor family, rule summary, ops changed, risks, and `ready|probe|discard`.

Fast blocker taxonomy before spending build time: full-grid mask floor, 10-channel intermediate floor, candidate-enumeration floor, sparse update table floor, dynamic/profile-invalid floor, param floor, and tie-break/hidden-semantics floor. Name the blocker in the report and choose the route that deletes that tensor family first; dtype cleanup is polish, not a primary plan.

### Workflow A — start a new task

```
# 1. See what's already known
cat The_400_Tasks/Task_by_Task/task<NNN>.md         # has auto-graph-diag

# 2. Extract the canonical
.venv/Scripts/python tools/agent_helpers/extract_task.py <NNN>

# 3. Get the structural picture
.venv/Scripts/python tools/agent_helpers/inspect_graph.py task<NNN>.onnx
.venv/Scripts/python tools/agent_helpers/largest_intermediate.py task<NNN>.onnx
```

### Workflow B — debug a verify_against_live failure

```
# 1. Identify diverging pair index from verify_against_live output (it tells you).
# 2. Dump intermediates from candidate AND live
.venv/Scripts/python tools/agent_helpers/dump_intermediates.py candidate.onnx \
    --tid <NNN> --pair-idx <P> --compare-with live.onnx

# 3. The first diverging intermediate (in node order) is usually the root cause.
# Trace backwards: which node produced it? What were its inputs?
```

### Workflow C — confirm a polish rewrite is safe

```
# 1. Run verify (required gate)
.venv/Scripts/python tools/verify_against_live.py --proto candidate.onnx --tid <NNN> --live-zip submissions/submission.zip
# Required: "OK Prototype matches LIVE" AND "Prototype vs LIVE: N/N"

# 2. Score comparison
.venv/Scripts/python tools/agent_helpers/score_compare.py task<NNN>.onnx candidate.onnx

# 3. Adversarial check (advisory only per core.md 2026-05-22 update — run + report,
#    don't gate on it; tests off-contract inputs Kaggle never sees)
.venv/Scripts/python tools/_score_v2.py --onnx candidate.onnx --adversarial
```

### Workflow D — from-scratch rebuild silent-fail check

```
# After your candidate passes visible verify 100% M/M, check that it agrees
# with LIVE on synthetic inputs OUTSIDE the visible distribution. This catches
# the task080 / task008 class of silent-fail trap.

.venv/Scripts/python tools/agent_helpers/adversarial_against_live.py \
    --candidate candidate.onnx --tid <NNN> --n 50

# Per 2026-05-22 update: advisory only. Run + report. Investigate disagreements
# if low %, but low % is NOT a hard gate — empirically task064 at 9.7% adv shipped
# +1.20 LB clean. Exception: task008-style documented hidden-correctness-sensitive
# tasks where the don't-do log explicitly says adversarial IS a hard gate.
```

### Workflow D2 — multicolor fuzz vs LIVE (task097 trap detector)

When the candidate uses a Conv-based "neighbor" check (any-neighbor / adjacent-cell / connected-region), visible+arc-gen+adversarial can ALL be 100% while the candidate diverges from LIVE on multicolor inputs. See `traps.md §14d`.

```python
# Generate random MULTICOLOR grids (not just visible-style monochromatic)
# Compare candidate vs LIVE byte-exact; reject if divergence > 5%

import numpy as np, onnxruntime as ort, zipfile
def synth_multicolor(seed, min_size=3, max_size=15, min_colors=3):
    rng = np.random.RandomState(seed)
    H, W = rng.randint(min_size, max_size+1, 2)
    n_colors = rng.randint(min_colors, 10)
    return rng.randint(0, n_colors, (H, W))

# Compare 3000 multicolor inputs
candidate = ort.InferenceSession('candidate.onnx', providers=['CPUExecutionProvider'])
with zipfile.ZipFile('submissions/submission.zip') as z:
    z.extract(f'task{NNN}.onnx', '/tmp/')
live = ort.InferenceSession(f'/tmp/task{NNN}.onnx', providers=['CPUExecutionProvider'])

divergences = 0
for seed in range(3000):
    g = synth_multicolor(seed)
    inp = encode_grid(g)  # convert to [1,10,30,30] one-hot
    if not np.array_equal(candidate.run(['output'], {'input': inp})[0],
                          live.run(['output'], {'input': inp})[0]):
        divergences += 1
print(f'Multicolor fuzz divergence: {divergences}/3000 = {100*divergences/3000:.1f}%')
# Required: < 1% divergence to ship. task097 was 84.23% -> -13.02 LB hidden-bad.
```

**Tasks at risk**: any task where the documented rule mentions "neighbor", "adjacent", "connected", or "same-color-region". The visible set is often monochromatic enough that "any-neighbor" and "same-color-neighbor" produce identical outputs; they diverge only on multicolor inputs.

### Workflow E — find a specific structural pattern

```
# Find all Equal nodes (Equal-fold candidates)
.venv/Scripts/python tools/agent_helpers/find_op.py task<NNN>.onnx Equal

# Find all Concat nodes (bool-Concat candidates)
.venv/Scripts/python tools/agent_helpers/find_op.py task<NNN>.onnx Concat --show-attrs

# Find Conv ops + their weights
.venv/Scripts/python tools/agent_helpers/find_op.py task<NNN>.onnx Conv --show-attrs
```

### Workflow F — full diagnostic refresh after canonical change

```
# Run this AFTER a new megastack lands and submission.zip changes
.venv/Scripts/python tools/agent_helpers/regenerate_graph_diagnostics.py

# Then commit the updated task .mds
git add The_400_Tasks/Task_by_Task/
git commit -m "graph diagnostic refresh post-<canonical>"
git push
```

## What's intentionally NOT in agent_helpers/

Some related tools live elsewhere because they're scoring/verification infrastructure, not diagnostics:

| Tool | Location | Purpose |
|---|---|---|
| `verify_against_live.py` | `tools/` | The official byte-exact gate. NEVER skip this. |
| `_score_v2.py` | `tools/` | Compute cost (memory + params) for an ONNX. |
| `spread_score_isolated.py` | `tools/` | Subprocess wrapper around _score_v2 (survives ORT crashes). |
| `spread_scan.py` | `tools/` | Run all 25 auto-detector patterns across all tasks. |

These are higher-level than the diagnostic helpers. Use them when you need authoritative scoring/verification; use the helpers in `agent_helpers/` for exploratory work.

## Add to this file when you create a new helper

If you write a diagnostic script that's broadly useful — drop it in `tools/agent_helpers/` and add a row to the table above. Worker contribution via `.update/` is the formal path.

### ng_fill_hang_safe.py (tools/, 2026-06-04) — hang-safe, resumable ARC-GEN corpus filler

Some ARC-GEN generators infinite-loop on specific RNG seeds (task076/324/018 confirmed) and
others have 20–40 s slow-tail seeds that are HEALTHY (task157); `corpus.build()` has no
per-`gen()` timeout, so one bad seed freezes a fill forever. This tool wraps generation in a
worker subprocess with a heartbeat watchdog: a seed stalling past `--hang-timeout` (default
120 s — do NOT set low, slow tails are valid seeds) is killed, recorded in a durable skip-set
next to the cache (`<cache>/_fill_state/`), and the fill resumes from checkpoint. Time-based
checkpoints (5 min) bank progress even for <2 pairs/s generators. Singleton-guarded
(`manager.alive`) so two managers can never race the same npz.

```
python tools/ng_fill_hang_safe.py --status   # per-task corpus depth + skipped seeds
python tools/ng_fill_hang_safe.py --all      # fill everything below 200k; idempotent, resumable
python tools/ng_fill_hang_safe.py --stop     # graceful halt (~5s)
```
