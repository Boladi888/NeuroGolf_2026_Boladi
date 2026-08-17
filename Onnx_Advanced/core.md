# core.md — NeuroGolf-specific facts and ORT quirks

Everything here is either (a) NeuroGolf-specific, (b) ORT behavior that contradicts the ONNX spec, or (c) competition-specific scoring/encoding details. Nothing in this file is in any general ONNX reference.

---

## Scoring formula

```
points_per_task = max(1.0, 25.0 - log_e(max(1.0, memory + params)))   # natural log
cost            = memory + params                                      # bytes + element-count
total_score     = sum(points_per_task) over the SCORED tasks (task001–task400)
```

- **Scored set = tasks 1–400 (NOT 401).** task000 is the *illustrative/tutorial*
  example and is **not eligible for points** (`00_COMPETITION_SPEC.md:143`; validator
  `neurogolf_utils.load_examples(0)` returns a hardcoded `_TASK_ZERO`; theoretical max
  is "10,000 across **400** tasks"). The submission zip contains 401 entries
  (task000.onnx–task400.onnx) but task000.onnx is ignored by the scorer. The old
  "401 tasks" wording here was the root of a +15.83 phantom in LB accounting — see
  `Logs/lessons.md` 2026-05-29 and `The_400_Tasks/Task_by_Task/task000.md`
  SCORED-STATUS FINDING. (A 1-ship experiment is staged to confirm; treat as
  near-certain pending that result.)
- MACs are NOT scored (removed 2026-05-04). Only `memory + params` count.
- The **inner** `max(1.0, cost)` guard means cost 0 (e.g., `input → Identity → output`) is valid and yields the 25-point ceiling. The outer `max(1.0, ...)` floor means cost below ~e^24 never lowers per-task points below 1.
- Incorrect networks score **literal 0**, not `max(1.0, ...)`. Correctness first, cost second.
- **Calibration by submission regime** (2026-05-23 refinement, supersedes the old blanket 0.48× rule):
  - **Single-task wildcards**: use raw `log_e(cost_before/cost_after) × 1.0`. Confirmed by ~21 single-task ships this session (5926.47 → 5949.01); empirical actuals matched raw log_e predictions within ±0.05 LB.
  - **Small stacks (2-5 tasks)**: closer to raw log_e too; t062+t170 + 5-stack landed exact-additive on Kaggle.
  - **Large megastacks (100+ tasks)**: empirically ~0.48× (megastack153 on 2026-05-20). Many tasks below the per-task LB-floor contribute less than their log_e suggests.
  - **Cascade unlock**: rare step-function exception, not the default model. Only confirmed event was task110 v8 (+12.73 LB unblocking neighbors). Most single-task wildcards land at the cost-curve estimate alone.
  - **Zero-public-weight tasks**: 10 confirmed family members (task366/324/133/182/018/157/255/066/076/277) — any cost reduction via lookup/NN memorization scores 0.00 LB on Kaggle regardless of local cost ratio. See `traps.md §14e`.

## What `params` actually counts

Source: `neurogolf_utils.calculate_params`.

- For `graph.initializer`: `sum(math.prod(init.dims))` — element count of the logical dense tensor.
- For `graph.sparse_initializer`: `sum(math.prod(sparse_init.values.dims))` — **count of STORED values**, NOT the logical dense shape. A sparse `[30, 30]` initializer with 5 non-zero entries counts 5, not 900, IF the consumer op accepts sparse weights (rare — most do not).
- For Constant nodes' `sparse_value`: same — counts `sparse_tensor.values.dims`.
- For Constant nodes' `value` / `value_floats` / `value_ints` / `value_strings`: element count of the dense tensor.
- **Element count, NOT byte count.** A `[30, 30] BOOL` initializer counts the same in `params` as a `[30, 30] F32` initializer: 900.
- Smaller dtype on initializers saves nothing in `params`. It saves only via `memory` (if the init feeds an intermediate that gets profiled).
- **Sparse initializers are not a blanket optimization**: even when params accounting would favor them, `onnx.shape_inference(strict_mode=True)` rejects sparse tensors as `Conv`/`QLinearConv` weights, returning `None` from `score_network`. Test through the official scorer path before relying on this.
- **Sparse initializers are UNUSABLE as Gather indices too** (verified 2026-05-30, task337 sub-floor probe). A sparse `[10]` Gather index would count NNZ params (e.g. 9 instead of 10 for `[0,1,2,3,4,8,6,7,5,9]`), but the scorer rejects it at **three independent gates**, each fatal, none fixable by construction: (a) `onnx.checker.check_model(full_check=True)` → "(Gather): indices typestr Tind has unsupported type sparse_tensor(int64)" — the ONNX Gather spec forbids sparse indices; (b) `shape_inference(strict_mode=True)` → rank mismatch; (c) `sanitize_model` renames the consumer's node input but NOT the `sparse_initializer`, dangling the reference (ORT load + checker fail). ORT *can* run the raw model (it densifies), but `score_network_v2 → None` before any score. **Net: sparse buys nothing for params; never reach for it.** Full proof: `The_400_Tasks/Task_by_Task/task337.md` Floor proof Lemma 4.

## What `memory` actually counts

Source: `neurogolf_utils.calculate_memory`.

- **First re-runs `onnx.shape_inference.infer_shapes(model, strict_mode=True)`**. Cleared `value_info` is repopulated from node outputs; real intermediates are still counted even if the saved model has empty `value_info`. Clearing stale `value_info` before save is only needed to avoid load-time conflicts, not to hide memory.
- Walks `graph.input + graph.value_info + graph.output` of the re-inferred graph. For each tensor, computes `num_elements × dtype.itemsize`.
- **Tensors named exactly `'input'` and `'output'` are SKIPPED**. The model's I/O tensors don't count.
- Then opens the ORT profiler trace JSON and takes `max(static_value_info_bytes, profiler_observed_bytes)` per tensor.
- Initializers do NOT count toward `memory`. They count toward `params`.
- **Scalar value_info tensors (shape `[]`) cost `dtype.itemsize` bytes**, not always 4. `BOOL []` = 1 B, `F16 []` = 2 B, `F32 []`/`INT32 []` = 4 B, `INT64 []` = 8 B. The "scalar costs 4 bytes minimum" claim is only true for 4-byte dtypes.
- A single-op `input → Op → output` graph has zero `value_info` intermediates and `memory = 0`.

## Validator rejection criteria

A model that triggers any of these is dropped (no score):

- More than 1 graph input or 1 graph output.
- Initializer name collides with input/output name.
- Any `dim_param` (dynamic dim), missing `dim_value`, or `dim_value <= 0` on any tensor in input/value_info/output.
- Subgraphs (control flow): `AttributeProto.GRAPH` or `GRAPHS` attribute on any node.
- `sequence_type` on any tensor; **OR any op type that contains the substring `"Sequence"`** (e.g., `SequenceAt`, `SequenceConstruct`, `ConcatFromSequence`, `SplitToSequence`, and `ReverseSequence` — the substring catches it too; both R6 scorer-confirmed "not permitted" 2026-06-19).
- Non-default opset domain (anything outside `""` and `"ai.onnx"`).
- Has `model.functions` (function defs).
- File size > 1,509,949 bytes (1.44 MiB binary, **NOT** 1.44 MB decimal).
- Any banned op type: `LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS`.
- `sanitize_model` returns `None` if any node's first output name contains `"kernel_time"` (collides with ORT profiler-trace key).
- `calculate_memory` rejects duplicate names across `graph.input + graph.value_info + graph.output`.
- `calculate_memory` rejects any non-`output` node output that shape inference cannot place in `value_info` or `output` (i.e., unresolved intermediate dtypes/shapes).
- **`run_network` hard-codes session output `"output"` and feed input `"input"`** — a model can pass static shape checks with other I/O names but fail actual NeuroGolf verification.

`sanitize_model` rewrites every node's `name` to its first output name AND sanitizes all input/initializer/node-output names to `safe_name_<N>`. Original duplicate `node.name` values are no longer a scorer hard failure (sanitization rewrites them). The real hard failures are duplicate/conflicting TENSOR names (input/value_info/output) or name collisions after sanitization.

## NeuroGolf input encoder (`convert_to_numpy`)

```python
benchmark = np.zeros((1, 10, 30, 30), dtype=np.float32)
for r, row in enumerate(grid):
    for c, color in enumerate(row):
        benchmark[0][color][r][c] = 1.0
```

- Input is `[1, 10, 30, 30] F32`, one-hot encoded on the channel axis.
- For a **background cell (color 0) INSIDE the active grid**: ch0 = 1.0, other channels = 0.
- For a **padding cell OUTSIDE the active grid** (row ≥ H or col ≥ W): ALL channels = 0.
- This means `ch0 = 0` is ambiguous between "occupied non-bg" and "padding". **Critical trap** — see "Padding-cell encoding" section below.

## NeuroGolf output decoder (`run_network`)

```python
result = session.run(["output"], {"input": benchmark_input})
return (result[0] > 0.0).astype(float)
```

- The decoder thresholds at `> 0.0` then casts to float. Output dtype on disk can be F32, F16, **BOOL** — all decode identically as long as the `> 0.0` thresholding produces the same {0, 1} mask.
- This is why the **F16-output trick** works: the final F32 Cast is unnecessary.
- This is why declaring graph output dtype as BOOL is sometimes safe (when `(out > 0.0)` is identity on a 0/1 BOOL tensor — which it is).

**Decode-side palette sentinels** (single-homed from the SPEC 2026-06-10): the 12-color palette is 0–9 real ARC colors, **10 = "no color"** (what unset/padding cells decode back to), **11 = "too many colors"** (one-hot decode-error indicator — more than one channel set). Channel exclusivity matters: a multi-hot cell decodes to 11, not to either color.

**Whole-canvas geometry caveat** (from the SPEC probe work): the real grid is embedded at the TOP-LEFT of the 30×30 canvas with trailing padding. `Identity`/`Transpose` preserve that convention for arbitrary grid sizes; **whole-canvas `hflip`/`vflip` via reversed-index Gather do NOT** — they mirror the full 30-wide canvas, moving the grid off the top-left corner and breaking the decoder's trailing-padding assumption. Shape-only solutions must preserve embedded-at-top-left for every grid size the generator emits.

**Pinned environment** (validator/scorer expect exactly): `numpy==2.4.4 · onnx==1.21.0 · onnxruntime==1.24.4 · onnx-tool==1.0.1` (in `.venv/`). Three benchmark sets are evaluated per task — `train` + `test` + bundled `arc-gen` — and the private LB adds a held-out sample from the same generators.

## The "1-ch0 trap" (padding cell encoding)

DO NOT compute "is foreground / non-bg" as `Sub(1.0, Gather(input, [0], axis=1))`. It is wrong.

- For bg cells inside the grid: ch0 = 1 → `1 - 1 = 0` → correct (not foreground).
- For occupied cells: ch0 = 0 → `1 - 0 = 1` → correct (foreground).
- **For padding cells: ch0 = 0 → `1 - 0 = 1` → WRONG. Padding gets marked as foreground.**

Use this instead:
```
full_sum   = ReduceSum(input, axes=[1], keepdims=1)   # sum all 10 channels
ch0_val    = Gather(input, [0], axis=1)
sum_nonbg  = Sub(full_sum, ch0_val)                    # sum of ch1..ch9
nonbg_bool = Greater(sum_nonbg, 0)                     # 1 only for true foreground
```

Returns 0 for bg (sum_nonbg=0), 1 for occupied (sum_nonbg=1), 0 for padding (sum_nonbg=0).

Active-cell detector (any color including bg, but NOT padding):
```
active_bool = Greater(ReduceSum(input, axes=[1], keepdims=1), 0)
```

## ORT-specific behaviors and opset

Kaggle's evaluator uses ONNX Runtime (ORT). ALL ORT facts live in ONE file — **`ort_reality.md`** (the canonical dtype×opset matrix, behavioral quirks, error→fix table, opset support, K2 source-read): `Pad(BOOL)` schema-supported at opset 13+, `Where(bool,bool,bool)` unsupported, `Where(bool,uint8,uint8)` works, OneHot takes INT32 or INT64 (depth dtype must MATCH indices — traps §2), float Equal needs opset 11+, etc. Scorer-side profiling (`calculate_memory` trace semantics, max(static, observed)): `scoring_internals.md`. Don't build against the spec — build against the matrix.

## Cross-task memory cascade bug (Kaggle scorer)

The Kaggle evaluation pipeline leaks memory between consecutive task evaluations.

- A high-memory ONNX in slot N can ERROR slot N+1 (or paired tasks) even though both pass `verify_network` in isolation.
- Host-confirmed (2026-05-04). Known trigger task set: `{45, 129, 130, 171, 172, 210, 211, 220, 239, 240, 241, 242, 256, 257, 258, 259, 260, 261, 262, 309}`.
- **Discovered 2026-05-20**: task110 may be a previously-undocumented trigger. Reducing it from cost 307,174 → 212,871 produced a **+12.73 LB Kaggle delta** vs the small direct cost-reduction prediction. The anomaly is most plausibly explained by cascade unlock — task110's high memory may have been ERRORing one or more neighbor tasks, which recovered when task110 dropped. Treat cost >200k as a cascade-suspect signal only, not confirmation.
- **Local validators (`score_network_v2`, `verify_against_live`) cannot detect this** — requires Kaggle's actual eval infrastructure.
- Workaround: replace high-memory ONNX with cheap `conv_identity_network(3)` stub (~3.7KB, params=900, mem=0) when the cascade triggers.
- When working on these tasks, prefer memory reductions over keeping a complex high-memory candidate even if the cost is similar.
- **Exploit the cascade carefully**: see `cascade_strategy.md` for the full playbook. Key points: cascade-unlock appears step-like and task-specific; single-task wildcards are the only reliable way to see the signal; high cost alone is not enough because some high-cost tasks are zero-public-weight or cost/LB-decoupled. Run the cherry13b Identity/conv-identity diagnostic before spending radical-rewrite effort.

## Shape-only ops are the cost floor

Patterns expressible purely as Slice + Gather + Concat + Pad + Reshape + Transpose + Identity + Cast cost ~1.5k (intermediate-tensor memory; 0 MACs, ~30 params for indices). Score ~17.5 pts/task. Beats any LUT-based or Conv-based representation when the rule allows.

**Order detectors shape-op-first** — they preempt LUT/Conv patterns where applicable.

## Verify gates and hidden-test rules

Three benchmark sets evaluate a network:
1. ARC-AGI public training v1 (`train` + `test` per task).
2. ARC-GEN-100K (bundled into `taskNNN.json` as `arc-gen`).
3. **Private held-out suite** (Private Leaderboard basis).

Functional correctness on (1) + (2) is verifiable; (3) is hidden.

- **Visible verify (`verify_against_live.py`)**: must be 100% M/M PASS. Empirical: 7 of 8 sub-100%-pass wildcards regressed Kaggle by -13 to -17 LB each in the 2026-05-19 wildcard experiment.
- Single exception: tasks with **fewer than 10 visible pairs** can have verify failures and still pass hidden (task000 confirmed, 0/3 visible → no Kaggle regression).
- **100% visible can still be hidden-bad.** task080 v5_bool_slices passed 231/231 visible + 450 adversarial × 9 modes = 100% adversarial, then regressed -13.32 LB on Kaggle. Boolean reformulations of float graphs are particularly prone to this.
- task008's numpy reference passed 266/266 visible but only 3/86 adversarial against LIVE — visible-only correctness ≠ rule-captured.
- **"Agree with LIVE" is asymmetric**: low agree-with-LIVE can mean LIVE is wrong, not that you're wrong. Always check `cand_correct` and `live_correct` independently. (See lessons.md 2026-05-12 task110 v4.)

## Adversarial gate (UPDATED 2026-05-22: advisory, not a hard gate)

```
.venv/Scripts/python tools/_score_v2.py --onnx <candidate> --adversarial
```

**The adversarial gate is ADVISORY** — run it and report the number, but **do NOT block a verify-passing candidate on low adversarial alone.**

### Why it's advisory

Adversarial tests off-contract inputs (shift/rotate ARC content off the (0,0) origin) that NEVER occur in the real Kaggle eval pipeline. Failures on those synthetic inputs don't predict hidden-bad on official data.

**"Agree with LIVE" is asymmetric**: low agree-with-LIVE can mean LIVE is wrong, not that the candidate is wrong. The candidate may be MORE correct than LIVE on out-of-distribution inputs that LIVE itself was never tested against.

### Empirical evidence (2026-05-22 session)

- **task064 `coloridx_exact_minmax`**: 9.7% adversarial-against-LIVE → shipped +1.20 LB clean
- **task064 `coloridx_nz_sentinel`**: 7.8% adversarial → would predict +1.55 LB if shipped
- **task162** (historical): 54.8% adversarial → shipped exactly +0.22 LB as predicted
- **task366 v10**: 97.5% adversarial → 0.00 LB (zero public weight, adversarial passed)
- **task324 cherry13b Identity**: N/A adversarial → 0.00 LB
- **task182**: 100% adversarial → suspect 0.00 LB contribution in 18-stack

Adversarial % does NOT correlate with LB outcome. The signal is noise except in specific documented cases.

### When adversarial IS meaningful (specific don't-do log evidence)

These tasks have DOCUMENTED Kaggle-negative outcomes correlated with adversarial divergence:

- **task008**: numpy ref 266/266 visible but only 3/86 against LIVE → keep ≥30% gate (traps.md §13)
- **task080** v5_bool_slices: 100% visible + 100% adversarial → -13.32 LB Kaggle (covered by `traps.md §12` bool reformulation silent-fail, not by adversarial gate)

The lesson from task008: adversarial-against-LIVE matters ONLY when the algorithm has been reformulated in a way that could silently differ from LIVE on inputs not in the training set (bool reformulations of float graphs, memorized-lookup architectures, etc.). For polish-and-compress candidates that preserve LIVE's algorithm, adversarial is essentially noise.

### Updated workflow

1. **Verify_against_live 100% M/M**: HARD GATE. Don't ship without this.
2. **Adversarial-against-live**: RUN IT, REPORT THE NUMBER. Don't gate on it unless:
   - The task has documented adversarial-correlated Kaggle-negative history (see specific tasks above), OR
   - The candidate is a fundamental algorithm reformulation (bool conversion of float graph, memorized lookup, etc.) — then synthetic-LIVE on 100+ inputs is the better safety check
3. **Synthetic-LIVE on adversarial inputs** (when applicable): the right safety check for hidden-correctness sensitivity. Generate 100+ off-contract inputs and compare cand vs LIVE byte-exact.

## Workflow commands

```bash
# Extract a task's ONNX from the canonical submission for inspection
python -c "import zipfile; zipfile.ZipFile('submissions/submission.zip').extract('taskNNN.onnx', '/tmp/')"

# Score a candidate
.venv/Scripts/python tools/spread_score_isolated.py <path/to/candidate.onnx>

# Verify byte-exact match against live (REQUIRED before staging)
.venv/Scripts/python tools/verify_against_live.py --proto <path> --tid NNN --live-zip submissions/submission.zip

# Adversarial check
.venv/Scripts/python tools/_score_v2.py --onnx <path> --adversarial

# Stage a winning candidate (DO NOT submit; submit-captain owns Kaggle)
mkdir The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/taskNNN_<technique>_cost<NNN>/
cp <candidate.onnx> The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/taskNNN_<technique>_cost<NNN>/taskNNN.onnx
# Then write manifest.json — see .ai_instructions/playbooks/SpreadSweepWorker.md step 7 for the schema.
```

## What you must NOT do

- Submit to Kaggle directly. Submit-captain has exclusive authority.
- Edit `submissions/submission.zip` directly.
- Call `tools/neurogolf_sync_current_best.py`.
- Skip the visible-verify gate (except for tasks with <10 visible pairs — and even then, document the exception in your manifest).
- Use the `1 - ch0` shortcut for foreground detection (see "1-ch0 trap").
- Trust local visible+arc-gen+adversarial as a guarantee against hidden-bad — for any major algorithmic rewrite of float→bool or new-encoding, generate synthetic boundary inputs and verify byte-exact against LIVE.
- **Drop ch0 via Pad-prepend-zero to save memory**. The official encoder explicitly sets ch0=1 for bg cells INSIDE the active grid (see encoder section above). Local `verify_against_live.py` has a softer decode and can mask this bug; the true competition encoder will return 0/265. Confirmed failure: task004 v9 — local decode 265/265 but `nu.convert_to_numpy + nu.run_network + np.array_equal` returned 0/265. Before staging any ch0-dropping candidate, gate with the true encoder pipeline, NOT just `verify_against_live.py`.

## Key paths

Working directory: repository root.

- **Validator code**: `${NEUROGOLF_DATA_DIR}/neurogolf_utils/neurogolf_utils.py` — defines `convert_to_numpy` (zero-padding), `run_network` (binarizes via `> 0.0`), `verify_subset` (`np.array_equal`), `calculate_memory` (intermediate-tensor sum from ORT profile trace), banned ops `{LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS}`, `_FILESIZE_LIMIT_IN_BYTES = 1.44 MB`. Data dir default `data/competitions/neurogolf-2026/`; in the owner's checkout, `data/` was a Windows junction to sibling non-git storage.
- **Per-task data**: `${NEUROGOLF_DATA_DIR}/task${TID:03d}.json` — `{train, test, arc-gen}` lists of `{input, output}` grid pairs.
- **ARC-GEN ground-truth generator (INDEPENDENT ORACLE)**: vendored **inside the repo** at `external/ARC-GEN` (gitignored — 61 MB working tree, 900 generators, 400 training UUIDs; `tools/arc_gen_enrich.py:find_arc_gen_dir` searches it first). This is `google/ARC-GEN`, the *same* generator that produces the hidden test set. Use it to generate FRESH `{input,output}` pairs for any task and verify a rule/ONNX against TRUE ground truth — NOT against the candidate's own numpy oracle (the "synth independence" trap, lessons.md 2026-05-27). Mapping: NG task `N` → `sorted(training/*.json stems)[N-1]` → `tasks/task_<uuid>.py`; call `task_list.task_list()[uuid][0]()` after `random.seed(...)`. This oracle exposed that the shipped task157 struct ONNX diverges from its own (correct) numpy rule on ~2% of real inputs — a lowering bug invisible to self-oracle synth. Re-clone if lost: `git clone --recurse-submodules https://github.com/google/ARC-GEN.git` (Apache-2.0; not vendored into this repo due to 116 MB w/ .git). Generate fresh pairs via `tools/arc_gen_enrich.py --task N --num K --dry-run`.
- **Base submission zip**: `submissions/submission.zip` — the canonical zip with 401 task ONNXes. Gitignored except this specific file via `submissions/**` + `!submissions/submission.zip`. Canonical backups at `submissions/_canonical_*.zip`.
- **Local single-task scorer**: `tools/_score_v2.py:score_network_v2(filename) -> (macs, memory, params)`. Byte-exact equivalent to Kaggle runtime cost per 2026-05-12 verification.
- **Full-zip verification**: `tools/verify_submission.py` — every entry loads + scores; cost/points tally. (The old `pipeline_cost_simulator.py` ref was a phantom — file never existed in git.)
- **Project state snapshot**: `CURRENT_STATE.md` — read first on session start. Replaces, doesn't stack.
- **Lessons**: `Logs/lessons.md` — durable session insights.
- **Per-task prototypes**: `The_400_Tasks/Task_by_Task/taskNNN/prototypes/` — `.py` builders + `.onnx` artifacts (`.onnx` typically gitignored).
- **Adversarial test tool**: `tools/adversarial_test.py` — 8-mode perturbations vs LIVE. Its `encode()` uses ch0=1 padding which differs from validator — used for vs-LIVE comparison, not vs-numpy_reference.
- **Worker flow**: read prompt from `The_400_Tasks/Task Prompts/`, build, gate, stage to task-local `staged_candidates/`, report. Full rules in `.ai_instructions/playbooks/agent_feedback_rules.md`.
- **Submission ledger**: `Logs/submissions_ledger.csv` — Kaggle submit history.
- **Auto-state generator**: `tools/update_current_state.py` — regenerates `CURRENT_STATE.md` (~44 lines) from manifest + Hermes + ledger + credit. DO NOT hand-edit `CURRENT_STATE.md`.
- **One-page dashboard**: `tools/dashboard.py` — predicted-vs-actual LB delta, hidden-bad rate, submit credit, reviewer queue, points-per-worker, unsolved-task ceiling.
- **Pool B target picker**: `tools/pick_unsolved_target.py` — ranks 246 unsolved tasks by tractability heuristic for workers 7-8.
- **Task Prompts are the source of truth** for current worker assignments — see `The_400_Tasks/Task Prompts/1/` (priority) and `/2/` (secondary).
- **Process feedback / worker rules**: `.ai_instructions/playbooks/agent_feedback_rules.md`.
- **Hash-MLP scanner** (2026-05-25): `tools/scan_hash_mlp_signatures.py` — probes every canonical task ONNX for the §28 hash-MLP fingerprint. Output CSV at `Logs/spread_scan_results/hash_mlp_scan_<YYYYMMDD>.csv`. Re-run after any canonical rebuild to identify unshipped hash-MLP unlock candidates. The 2026-05-25 scan identified task157, task182, task255 as unshipped unlock candidates and task219 as a misattributed-artifact case; see `traps.md §27e/§27f`.

## Spread-knowledge tooling (auto-apply math-equivalent shrinks across 401 tasks)

Built 2026-05-18: `tools/spread_scan.py` + `tools/spread_patterns/pass_*.py` + `tools/spread_annotate_notes.py` + `tools/spread_master_report.py` + `tools/polish_candidate.py`. The system applies every known math-equivalent shrink (onnxsim, safe_opt, const_to_init, clip_mul_to_min, scalar_broadcast_ones, slice_axis_collapse, fp16_selective, crop_to_size_advisory) to every task in `submissions/submission.zip`, verifies byte-exact on dummy inputs, and writes:
- `Logs/spread_scan_results/spread_scan_<YYYYMMDD>.{json,csv,summary.md}` per scan
- `<!-- SPREAD:BEGIN -->...<!-- SPREAD:END -->` block in every `The_400_Tasks/Task_by_Task/taskNNN.md`
- Master report at `Research_Archive/notes/spread_knowledge_findings_<date>.md`

**Why this exists**: workers were doing one-task-at-a-time optimization without spreading insights. Task000's manual `ones10 → scalar1` broadcast trick worked on **34 other tasks** when generalized; `slice_axis_collapse` from task000 rev2 hit 53 tasks.

**How to apply**:
- Re-run after any new candidate lands in submission.zip: `.venv/Scripts/python.exe tools/spread_scan.py && tools/spread_annotate_notes.py && tools/spread_master_report.py`.
- Adding a new pattern: drop a module under `tools/spread_patterns/pass_<name>.py` exposing a `PATTERN = SpreadPattern(...)`. Driver auto-discovers via `iter_patterns()`.
- Pre-submit polish for a worker handoff: `.venv/Scripts/python.exe tools/polish_candidate.py path/to/candidate.onnx`.
- Subprocess isolation (`tools/spread_score_isolated.py`, `tools/spread_verify_isolated.py`) is required because ORT C-crashes (SIGFPE on certain fp16 graphs) would otherwise kill the driver.
- `fp16_selective` is the only "selective" pattern: kept only when cost actually drops. The 2026-05-12 lesson says naive fp16 INCREASES cost on high-cost graphs via Cast intermediates; the cost-delta gate prevents that.

**Usage doc**: `Research_Archive/notes/spread_knowledge_tools.md`.

**Tests**: `tests/test_spread_patterns.py` (10) + `tests/test_spread_scan_driver.py` (5).
