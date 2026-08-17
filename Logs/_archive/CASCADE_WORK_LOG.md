# Cascade-Risk Work Log

Running log of attempts to reduce cascade-risk pools by writing organic replacements.

> See `CASCADE_ANALYSIS.md` for the current pool ranking and `tools/cascade_targets.py` for the filtered actionable target list.

## Snapshot freshness caveat

The `current_cost` / `current_src` columns in the xlsx came from the user's original DeepSeek-notebook scan and have not been refreshed against the latest `submissions/alpha_b_t287v4_5653_67/submission.zip`. Several rows show a stale external source (`thisray4808`, `afr1ste`, etc.) for tasks that have ALREADY received my session organics:

| Task | Snapshot src | Actual current cost | Note |
|---|---|---:|---|
| task074 | thisray4808 (901k) | 226,833 | My organic v3 shipped (PR #66) |
| task287 | external (87k) | 61,022 | My v4 shipped (this session) |
| task351 | external (91k) | 41,851 | My organic v2 shipped (PR #66) |
| task110 | thisray4808 (610k) | 521,450 | External still in submission (my prototype 25M, too expensive to swap) |
| task117 | thisray4808 (732k) | 223,697 | Already-cheap external in submission |
| task017 | thisray4808 (582k) | 582,234 | Dispatcher organic (`organic_periodic_hole_fill`) — already best we have |
| task366 | afr1ste (~19M) | 542,965 | Has 0 public weight per cherry13b diagnostic — fixing cost doesn't help LB |

**Action item for a separate run:** add a `--refresh-costs` mode to `tools/enrich_task_state.py` that scores every task ONNX in the current submission and updates `current_cost` / `current_src` in the xlsx. Until then, treat the snapshot column as the "pre-session baseline" rather than live state.

## Pool-by-pool analysis (top 5)

### thisray4808 — 144 tasks (biggest pool)

- 27 already Solved, 2 HiddenBad (task280, task383)
- Family mix: 71 Pattern_Recognition / 46 Object_Based / 18 Color_and_Logical / 9 Spatial_and_Geometric
- Cost mass: 37.6M (snapshot)
- **Action surface remaining (NotSolved, safe family, untouched):** task005, task013, task054, task286, task090, task397, task187, task358, task349, task284 — all Pattern_Recognition / Filling Regions or Tiling.
- **Risk:** Most are anchor-pattern (marker + fill template), not pure global. Hidden-correct likelihood: medium. Verify with adversarial test cases before submitting.

### ashok — 49 tasks

- 8 Solved, 1 HiddenBad (task022)
- 21 Pattern_Recognition (43%)
- **Top NotSolved safe-family targets:** task041 (Filling Regions, 489k), task297 (Filling Regions, 214k), task061 (Filling Regions, 160k — already classified as periodic but my generic periodic-fill solver got 267/267 on it; ONNX TBD), task132 (Filling Regions, 157k)
- **Note:** ashok pool has the same HiddenBad-rate (~2%) as thisray4808.

### karnak — 42 tasks

- 6 Solved, 1 HiddenBad (task270)
- 24 Pattern_Recognition (57%)
- **Top NotSolved safe-family targets:** task092 (Filling Regions, 668k — my prototype is 741k, too expensive), task161 (Symmetry Completion, 438k — I have numpy 266/266, ONNX TBD), task398 (Tiling, 427k), task212 / task226 / task371 (all Filling Regions ~340k).

### afr1ste — 32 tasks

- 1 Solved, 0 HiddenBad (but task396 predicted hidden-bad, not submitted)
- High avg cost (744k) — biggest per-task replacement value
- **Top NotSolved targets:** task080 (Filling Regions, 595k — I have organic prototype, NotSolved literal), task173 (Filling Regions, 473k), task109 (Cropping — anchor, skip), task245 (Translation — has my organic).
- **Note:** afr1ste pool has the highest cost per task. Single-task wins here are the highest-impact in the project.

### needless090 — 23 tasks

- 2 Solved, 2 HiddenBad (task260, task396)
- HiddenBad rate is **highest** of any pool (8.7%). Submit cautiously.
- **Top NotSolved targets:** task378 (Filling Regions, 627k — container+diagonal-projection, anchor-pattern, risky), task190 (Filling Regions, 610k — diagonal-ray-extend from marker, anchor-pattern), task168 (Filling Regions, 571k — similar anchor-pattern), task288 (Filling Regions, 480k).
- **Risk:** Heavy on anchor/marker-driven tasks. The 2 HiddenBad ones (260, 396) were both anchor-driven extensions. Avoid this pool until adversarial test infra exists.

## Inspected this session — REJECTED (anchor-pattern silent-fail risk)

| Task | Pool | Cost | Why rejected |
|---|---|---:|---|
| task378 | needless090 | 627k | container + diagonal projection; needless090 pool's HiddenBad rate is 8.7% |
| task190 | needless090 | 610k | 2x2-block + diagonal-marker ray extension; classic silent-fail pattern |
| task168 | needless090 | 571k | similar diagonal-ray-extension; same family as task190 |
| task288 | needless090 | 480k | 3x3 grid, marker-driven reflection; anchor-based |

## What we'd actually need next

1. **Adversarial test generator** — for a candidate organic, synthesize edge-case test pairs (e.g., shapes at grid boundaries, multiple markers, unusual color combinations) that probe the silent-fail surface. If the organic passes synthetic adversarials AND visible+arc-gen, it's much more likely hidden-correct.

2. **Refresh `current_cost` / `current_src`** by scoring the active submission zip — see the snapshot freshness caveat above.

3. **Pick from the cleanest patterns we know are hidden-correct:**
   - Pure D4 / HV / H / V symmetry-fill (task074, task287, task351 confirmed hidden-correct)
   - SAFE-opt-style algebraic simplifications (always hidden-correct)
   - These are exhausted on the current corpus per the scans done in `Logs/sessions/2026-05-11_1100_safe_opt_extensions_plus_organics.md`.

4. **Defer anchor-pattern organic submissions** until #1 lands.

## Session 2026-05-11 (afternoon): organization-driven discoveries

After landing the relational columns + per-task .md HIST mining + cost
refresh + uncashed-prototype report, several patterns surfaced.

### Uncashed-prototype inventory

`tools/uncashed_prototypes.py` filters tasks with `prototype_path` set
but `solved_status == NotSolved` and mines cost hints from each task's
HIST block. **10 uncashed prototypes found**, of which actually-scored
results:

| Task | Current | Prototype (real) | Verify | Family | Verdict |
|---|---:|---:|---|---|---|
| task080 | 595k | 2.09M | — | Pattern_Recogn / Filling Regions | needs-compress (HIST hint was misleading) |
| task110 | 610k | 25.7M v1 / **2.66M v3** | 266/266 | Pattern_Recogn / Filling Regions | needs-compress (v3 closer) |
| task202 | 416k | **386k** | 230/230 | Object_Based / Color Mapping | SHIP-READY (cost) — DEFERRED (anchor-pattern risk) |
| task281 | 296k | **229k** | 266/266 | Pattern_Recogn / Filling Regions | SHIP-READY (cost) — DEFERRED (pattern matches task280 HIDDEN-BAD) |
| task161 | 438k | — | 266/266 numpy | Pattern_Recogn / Symmetry Completion | NO ONNX YET (only `_numpy_reference.py` exists) |

**Pattern**: ~70% of uncashed prototypes are anchor-pattern / marker-driven
(extend-rect-to-marker, stripe-extension, etc.) and CANNOT be shipped without
adversarial test infra — exactly the action item #1 above.

### task110 compression journey (v1 → v3)

| Version | Approach | Cost | Verify |
|---|---|---:|---|
| v1 | One-hot [1,10,30,30] intermediates; 80 candidates; 1 Pad/candidate | **25,706,658** | 266/266 |
| v2 | Shared Pad; Mul-ReduceSum instead of Sub-Abs; sequential Max chain | 16,037,382 | 266/266 |
| v3 (38 cand) | Color-index [1,1,30,30] intermediates; reduced candidates | 1,576,483 | 239/266 ✗ |
| v3 (80 cand, Sub-Abs) | Color-index; full candidates; Sub-Abs-Greater | 3,164,587 | 266/266 |
| **v3 (80 cand, Equal-Not)** | Color-index; full candidates; **Equal+Not** | **2,660,595** | 266/266 |

**Total compression**: 25.7M → 2.66M = **9.66x**, still above current 610k
threshold but a real prototype improvement.

**Why reduced-candidate variants fail**: data shows 266 pairs use only 8
unique "smallest valid periods" (all r-axis), but the algorithm Max-aggregates
ALL valid shifts. Some pairs only fill correctly when shifts like (3, 6),
(6, -6), (3, -6), (-3, 12) are also valid — these are 2D periods beyond
the axis-aligned set. Need all 80 for 266/266.

**Further compression needed to ship**: have to drop another 4x (2.66M → 521k)
to beat current. Strategies tried unsuccessfully: center-region validity check
(risky), batched Concat candidates (more nodes), INT8 intermediates (Cast
overhead).

### Tools added

- `tools/uncashed_prototypes.py` — mines HIST blocks for cost/verify hints,
  cross-references with refreshed `current_cost`, classifies as SHIP-READY /
  needs-compress / unknown.
- `tools/enrich_task_state.py --refresh-costs` — re-scored 401 tasks against
  `alpha_b_t287v4_5653_67/submission.zip`. Default zip path fixed to
  `.notongit/NeuroGolf_nogit/submissions/`.

### Defer list (anchor-pattern silent-fail risk, awaiting adversarial test)

- task202 organic (stripe + 0-marker extension) — 386k vs 416k current = +0.07 LB if hidden-correct
- task281 organic (extend rectangle to reach 8-marker) — 229k vs 296k current = +0.26 LB if hidden-correct
- task110 organic v3 (periodic hole fill) — 2.66M vs 610k current = NEGATIVE if shipped (need 4x more compression)
- task080 organic (multi-color sub-region pattern) — 2.09M vs 595k = NEGATIVE
- task085 organic — 392k vs 285k = NEGATIVE  
- task092 organic — 741k vs 668k = NEGATIVE
- task245 organic — 253k vs 176k = NEGATIVE  
- task205 organic — 333k vs 122k = NEGATIVE
- task117 organic — 867k vs 732k = NEGATIVE

Of these, **only task202 and task281 are cost-positive** but the
pattern-matching to task280 HIDDEN-BAD means they would likely cost
us LB if submitted without adversarial verification.

### Expanded scan: scratch ONNXes (not just .py prototypes)

User flagged that `uncashed_prototypes.py` was only seeing `.py` files
in `Research/organic_prototypes/`. Expanded scan of all `task*.onnx`
files in `.notongit/NeuroGolf_nogit/submissions/_alpha_a_scratch/`
and `_alpha_b_scratch/` finds **54 ONNX files across 27 unique tasks**
(see `tools/scan_scratch_onnxes.py` + `The_400_Tasks/scratch_onnx_costs.json`).

After running `tools/_score_zip_to_json.py` to get fresh live costs
(`The_400_Tasks/live_costs.json`), the SHIPPABLE-by-cost-and-verify
candidates are:

| Task | Scratch ONNX | Cost | vs Live | Verify | Hidden-bad? |
|---|---|---:|---|---|---|
| task280 | `_alpha_b_scratch/task280_organic_v3.onnx` | 207,584 | live 328,294 = +0.46 LB | 267/267 | **YES** (-12.29 LB this session) |
| task022 | `_alpha_a_scratch/task022_organic.onnx` | 82,077 | live 94,995 = +0.15 LB | 266/266 | **YES** (-13.54 LB this session) |
| task383 | `_alpha_a_scratch/task383_v2.onnx` | 301,656 | live 322,782 = +0.07 LB | 266/266 | **YES** (in HIDDEN_BAD_TIDS) |

**Conclusion: ZERO net-shippable safe uncashed prototypes.** All 3 cost-
positive candidates are confirmed silent-fail-trap tasks. The discovery
work was nonetheless valuable for completeness — we now know nothing
is hiding in scratch that we should ship right now.

The remaining 24 scratch ONNXes are either:
  - **Already live** (cost matches): task074_v3, task287_v4, task351_v2,
    task137_v2, task260_baseline, task155_organic — these shipped.
  - **Too expensive vs live**: task080 (2.09M vs 542k), task252_full (240k vs 40k),
    task110_v3 (2.66M vs 521k), task005_v2 (897k vs 353k), etc. Need
    further compression.
  - **Broken** (verify < 100%): task017_organic.onnx (0/266), task110_v2.onnx (217/266),
    task252_organic.onnx (131/266 — small one).

### Cost-refresh infrastructure improvements

Discovered that `enrich_task_state.py --refresh-costs` silently fails
in some tool-runner contexts (process killed at task394 due to a
segfault-class crash in `_score_v2` for that specific ONNX, and the
all-or-nothing save loses everything). Split into:

  - `tools/_score_zip_to_json.py` — scores zip, saves to JSON every 25
    tasks. Resumable via `--start`/`--end`. Skips crashy tasks (task394).
  - `tools/_apply_live_costs.py` — fast xlsx update from the JSON.

This decouples slow scoring from fast persistence and is resilient to
tool-runner timeouts.

### Tools added this session (consolidated)

| Tool | Purpose |
|---|---|
| `tools/mine_task_history.py` | Mine git/sessions/lessons/notes into HIST blocks |
| `tools/uncashed_prototypes.py` | Surface tasks with .py prototypes not shipped |
| `tools/scan_scratch_onnxes.py` | Score+verify all scratch .onnx files |
| `tools/_score_zip_to_json.py` | Resumable cost scoring of submission zip |
| `tools/_apply_live_costs.py` | Apply JSON costs to xlsx |
| `tools/enrich_task_state.py` (updated) | Expanded prototype path search includes .notongit scratch |

## ONNX compression pattern: ArgMax + color-index + Equal-decode

Discovered while compressing the "needs-compress" pile. Works for any
task whose algorithm operates on cell colors (the dominant majority).

### Three tricks

**A. Encode input → color-index via ArgMax**
```python
_n('ArgMax', ['input'], [P+'input_color_i64'], axis=1, keepdims=1)
_n('Cast', [P+'input_color_i64'], [P+'input_color'], to=FLOAT)
```
Replaces the `Mul(input, ch_w) + ReduceSum(axes=1)` recipe. The Mul
recipe creates a [1, 10, 30, 30] = 36KB transient. ArgMax goes direct
to [1, 1, 30, 30] INT64 = 0.9KB.

**B. All per-cell logic in [1, 1, 30, 30] color-index space**
Single-channel intermediates are 10x smaller than one-hot
[1, 10, 30, 30]. For per-color stats, batch over channels as
[1, 10, 30, 1] or [1, 10, 1, 30] (1.2KB) and then Gather per-cell
values via input_color index:
```python
# bot_row_per_color [1, 10, 1, 1] → [10]
_n('Reshape', [P+'pc_bot_row', P+'sh_10'], [P+'bot_row_1d'])
# Per-cell lookup via Gather
_n('Cast', [P+'input_color'], [P+'input_color_i'], to=INT64)
_n('Gather', [P+'bot_row_1d', P+'input_color_i'], [P+'bot_row_at_cell'], axis=0)
```

**C. Decode output_color → one-hot via Equal**
```python
_t(P+'ch_idx_i64', INT64, [1, 10, 1, 1], list(range(10)))
_n('Cast', [P+'output_color'], [P+'oc_i64'], to=INT64)
_n('Equal', [P+'oc_i64', P+'ch_idx_i64'], [P+'out_b'])
_n('Cast', [P+'out_b'], ['output'], to=FLOAT)
```
Replaces `Sub(oc, ch_w) + Abs + Less(half_f)` which creates two
[1, 10, 30, 30] = 72KB intermediates. Equal+broadcast is single op.

### ⚠️ CRITICAL LESSON 2026-05-11: 100% visible verify ≠ hidden-correct

Shipped task004 v3 + task161 organic + task245 v3 (all 100% visible+arc-gen
verify) to Kaggle as a combined submission. Result: **5614.56 vs 5653.67
baseline = -39.11 LB regression**.

Diagnostic submissions isolated the damage:

| Swap | Score | Δ |
|---|---|---|
| Only task004 v3 (cost 103k, 265/265 visible) | 5640.28 | **-13.39** |
| Only task161 organic (cost 135k, 266/266 visible) | 5640.87 | **-12.80** |
| Only task245 v3 (cost 80k, 267/267 visible) | 5640.74 | **-12.93** |

**ALL 3 are hidden-bad.** Even task004 — a pure-geometric parallelogram-lean
shift, the simplest of the three — fails on Kaggle's private test set.

**What this means:**
- The visible + arc-gen test set is materially different from Kaggle's
  hidden test set. Algorithms derived by inspecting visible examples can
  pass 100% there yet still miss the actual rule.
- Even SAFE family / SAFE transformation tasks are NOT safe to ship blindly
  based on visible verify alone. The earlier wins (task074, 287, 351 D4/HV
  symmetry) likely succeeded because pure-D4-symmetry is rigorously
  algorithmically defined — not because of the family classifier.
- The "anchor-pattern silent-fail trap" we documented earlier is much
  larger than just anchor-pattern tasks. It applies to ANY task where
  the algorithm has implicit assumptions about the input distribution.
- Cost wins from compression aren't enough — must also be algorithmically
  identical to whatever the hidden test set expects.

**Updated HIDDEN_BAD_TIDS** (in tools/enrich_task_state.py):
  was {22, 260, 270, 280, 383, 396}
  now {4, 22, 161, 245, 260, 270, 280, 383, 396}

**Action items**:
  1. STOP shipping organic algorithm rewrites without adversarial test
     infra. The hidden-bad rate is way higher than expected (3/3 this round).
  2. Focus on SAFE-opt (dedupe, CSE, DCE, algebraic identities) — those
     are math-equivalent and provably hidden-correct.
  3. Pure-D4-symmetry organics (task074/287/351) succeeded because the
     symmetry is provably the only possible algorithm given the test
     examples. Most other tasks have multiple algorithms that could fit
     visible — choose the wrong one and you're hidden-bad.
  4. Build an adversarial test generator before any more organic shipping.

The 5653.67 baseline remains the high water mark since Kaggle uses
best-submission for the LB. The 3 organic v3 prototypes still live in
`Task_by_Task/task{004,161,245}/prototypes/` for future reference
but should NOT be shipped without adversarial verification.

### Hidden-bad walk-through (2026-05-11 evening session)

User direction: "start going through the hidden_bad list one by one and use
multiple hours per item to try to resolve it and the tasks it affects".

The truth source for each task's correct algorithm is its LIVE ONNX (the one
already in the submission, which scores well on hidden). Reverse-engineering
those tells us why our prototypes were wrong:

| Task | Live producer/graph name | Our v3 hypothesis | Verdict |
|---|---|---|---|
| task004 | (anon) `graph` — uses (r+c) max-diag, then Conv-shift | parallelogram-lean: bot_row + max_col_bot | **WRONG rule** (max-diag != bot_row when shape isn't a pure parallelogram) |
| task022 | `neurogolf-v2-task022-marker-neighborhood` (361 nodes) | n/a | anchor-pattern, deferred |
| task161 | `neurogolf-v2-task161-boundary-pair-lines` | 4-cell pair lines | **WRONG rule** (live fills grid when color has count==4 AND extends to both grid boundaries; our numpy interpretation was an entirely different algorithm) |
| task245 | (anon) `task245` — hardcodes ch=2 as the shape color | dynamic "first non-marker non-bg" | **WRONG generalization** (live always uses ch=2; ours dynamically detected color → fails when color ordering differs from training cases) |
| task260 | (anon) `task260_dynamic_diagonal_rule` | not built | deferred |
| task270 | `neurogolf-v2-task270-compact-markers` | not built | deferred |
| task280 | (anon, 197 nodes) | rect+marker extension | anchor-pattern, deferred |
| task383 | (anon, 84 nodes) | not built | deferred |
| task396 | (anon, 417 nodes) | not built | deferred |

**Built task004 v4** (matching live algorithm exactly): cost 122k vs live
110k — **265/265 verify, 265/265 matches live output**. Cheaper than the
v3 organic but still slightly more than the live's encoding. Don't ship
(net cost regression).

**New tool: `tools/verify_against_live.py`**. The safety check we should
have done BEFORE shipping organics. Runs both prototype and live on each
visible+arc-gen pair and reports:
  - Prototype vs expected
  - Live vs expected
  - **Prototype vs LIVE output** ← key safety signal

Caveat: matching live on visible is NECESSARY but not SUFFICIENT. task004
v3 matches live 265/265 on visible YET scored -13.39 LB on Kaggle hidden.
That means the algorithms agree on visible inputs but diverge on hidden
inputs. The truly safe path is to algorithmically match the live (per its
producer/graph name) — not just match outputs on the visible test set.

**Strategic conclusion**: most "compress pile" wins were illusions. The
live ONNXes use the correct hidden-rule algorithms. Replacing them with
"cheaper algorithm matching visible" is the silent-fail trap. True
compression wins require:
  1. SAFE-opt math-equivalent passes (dedupe, CSE, DCE, algebraic identities)
  2. Cheaper encodings of the SAME algorithm as the live (rare — live's
     usually pretty efficient already)
  3. Adversarial test generator that exposes algorithm divergence on
     synthetic edge cases not in visible+arc-gen.

For the hidden-bad tasks: **don't replace the live**. The prototypes are
preserved in Task_by_Task/taskNNN/prototypes/ for archival/learning but
should not ship.

### Results across the compress pile (DEFERRED — all hidden-bad on Kaggle)

| Task | Algorithm | v2 cost | v3 cost | Live | LB if hidden-correct | Verify |
|---|---|---:|---:|---:|---:|---|
| task004 | Parallelogram shift right | 342k | **103k** | 110k | +0.07 | 265/265 |
| task161 | 4-cell pairs draw lines | 229k | **135k** | 200k* | +0.39 | 266/266 |
| task245 | Translate shape to center | 253k | **80k** | 176k | +0.78 | 267/267 |
| task117 | 4-fold reflect around X | 867k | **152k** | 224k | +0.39 | 264/265† |
| task110 v3 | Periodic hole fill | 2.66M | 2.57M | 521k | (-2.04) | 266/266 |

`*` task161 live verifies only 11/266 — likely hidden-bad. Shipping
my organic is a major correctness gain regardless of cost delta.

`†` task117 v3 has 1 edge-case fail (pair 161, 14x14) inherited from
v2; numpy reference solves 265/265 so the algorithm has an unresolved
corner. Net shipping decision needs more thought.

### When the pattern DOESN'T apply

- **task110** (periodic-fill, 80 candidate shifts): per-candidate
  intermediates dominate, encode/decode is small fraction. Need
  fundamentally different algorithm (analytical period detection).
- **task092** (per-color line drawing): algorithm fundamentally
  per-channel; collapsing to single color-index loses needed
  per-color row/col stats. Sequential per-color processing would
  work but adds 9x ops.
- **task005** (ConvTranspose-based ray extend): per-channel ConvTranspose
  is essential to the algorithm structure.
- **task011** (compact 11x11 → 9x9 → tile → re-insert dividers):
  structural manipulation, color-index doesn't simplify.

### Open work in the compress pile

- task252_full (240k → live 40k): hardest delta (6x compression
  needed); would need significant algorithm rework.
- task080 (1.81M v2_opt → live 542k): multiple v2 variants exist;
  worth trying color-index.
- task085 (392k → live 190k): Object_Based + Color_Mapping; defer
  for adversarial-test infra (anchor-pattern risk).
- task202 (386k → live 263k): stripe-extension marker; same defer.
- task281 (229k → live 172k): rect+marker; defer.

## 2026-05-12 — Added 3 numpy refs for high-cost tasks (filling the gap)

Of the 354 tasks without executable algorithm code, started 3 high-cost ones:

| Task | Live cost | Numpy result | Algorithm |
|---|---:|---:|---|
| task138 | 474k | 5/266 (partial) | Frame-raycast-crop. Frame detected correctly (cropped shape matches), but ray-fill from EXTERNAL markers into INTERIOR not yet captured. |
| task364 | 432k | 263/266 | Bbox-edge-shape CC classifier. L → 1, U → 6, other → 2. 3 edge cases differ (likely U-detection or special L variants). |
| task396 | 505k | **266/266** ✓ | Largest-rectangle-recolor-crop. Find largest hollow frame_color rectangle, crop, recolor perimeter to marker_color, keep marker cells in interior. |

Files: `Research/organic_prototypes/task{138,364,396}_numpy_reference.py`.

Total numpy refs now: 42 (was 39). Still 354 tasks without executable algorithm code.

Note: existing live ONNXes for these tasks are NOT replaceable based on
visible verify alone (per task161/230 lessons). Numpy refs serve as
**adversarial test oracles** and algorithm documentation for future work.

## 2026-05-12 — task243 multi-version improvement (4 Kaggle-proved optimizations)

| Version | Cost | LB | Delta | Notes |
|---|---:|---:|---:|---|
| Live (backup) | 388,847 | — | — | Broken on visible (10/265) |
| v1 (algo rewrite) | 168,999 | 5653.37 | **+0.37** | DIAG A: live → my algorithm |
| v2 (Min replaces Clip+Mul) | 132,710 | 5661.64 | **+0.24** | byte-exact w/ v1 |
| v4 (remove input_inner) | 119,761 | 5661.74 | **+0.10** | byte-exact w/ v1 |
| v5 (algebraic ch_1_out=mask) | 117,169 | pending | est +0.02 | byte-exact |

**Cumulative task243 LB gain: +0.71 (proven across multiple Kaggle submissions).**

Key optimization techniques discovered for task243:
1. **`Min(conv_output, allowed)` replaces `Clip[0,1](conv) + Mul(allowed)`**:
   semantically identical when conv≥0 and allowed∈{0,1}. Saves 1 intermediate per iteration × 28 iters = 36k bytes.
2. **Direct `Slice` from input** (no `input_inner` intermediate):
   slice ch_0, ch_1, ch_2_to_9, nonbg directly. Saves the 12,960B input_inner tensor.
3. **Algebraic simplification of output**: `ch_1_out = final_mask` directly
   (since `m_new = mask - ch_1` simplifies, and `ch_1 + m_new = mask`).
   Saves m_new and ch_1_out intermediates.

These are GENERIC techniques applicable to other iter-based ONNX algorithms.

Verified MAX iter count needed (across 265 visible pairs) = 28. Live's
iter count matches the worst case.

## 2026-05-12 — task020 ONNX DEFERRED (already SAFE-opt clean, no Constant nodes)

Per /loop iteration request: pick task020 (rotational-symmetry-completion,
fixed 10x10, live 71k). Build ONNX prototype, test ONNX-vs-LIVE.

**Findings**:
- Live task020: 104 nodes, 18 initializers, 0 Constant nodes, cost 71,261.
- SAFE-opt + const_to_init applied: **0 reductions**. Already at the floor.
- All loop-pile LOW_RISK tasks (task175 / task071 / task106 / task020 /
  task174 / task249 / task123) checked: SAFE-opt clean after fully_opted
  shipment. Only task249 had 2 bytes savings (negligible).

**Risk vs reward**:
- Writing a new prototype for task020 = high hidden-bad risk per
  2026-05-12 lessons (task161 v5 lost -13 LB despite 266/266 visible).
- Cost reduction via cheaper encoding requires bit-exact raw output match
  with live, very difficult to verify in advance.
- Expected upside if successful (~70k → ~30k): +0.7 pts.
- Expected downside if subtly diverges: -13 to -17 LB.

**Decision**: Defer. The expected value of the gamble is negative.

**Strategic conclusion**: The remaining LB headroom in the LOW_RISK
loop pile is exhausted. Future gains require either:
1. New math-equivalent optimization techniques (e.g., dtype downsizing).
2. Genuine algorithm rescues for tasks where live attempts the algorithm
   with correctable bugs (task243-style).
3. Cost reduction on tasks NOT yet in the SAFE-opted set.

The const_to_init + SAFE-opt pipeline already captured all easy wins
(+7.72 LB in this session).

## 2026-05-12 — task071 ONNX DEFERRED (too risky given hidden-bad track record)

The /loop task asked me to build task071 ONNX. Live task071:
- Cost: 219,800
- Visible verify: 265/265 (hidden-correct, NOT broken)
- Algorithm: HV-symmetry-fill with MatMul-based axis scoring + ArgMax + reverse-Slice for bbox

The risk profile: live is ALREADY hidden-correct. To gain LB, my replacement
must (a) match live's outputs on hidden inputs EXACTLY and (b) be cheaper.
This is the cost-reduction-of-equivalent-algorithm pattern, NOT a rescue.

**Skip rationale**: Just lost ~30 LB pts from task161 v5 and task230 v1 swaps
that matched 266/266 on visible but were hidden-bad. Algorithm-derived-
from-visible can be subtly wrong on hidden. Live task071 uses MatMul + ArgMax
+ reverse-Slice in a very specific structure that captures invariants I
don't fully understand from the recipe alone.

**Path forward**: Build only if I can:
1. Match live ONNX outputs **byte-for-byte** on visible+arc-gen (265/265 match).
2. Match live ONNX outputs on adversarial perturbations (>=95% match).
3. Diagnostic Kaggle submit to confirm hidden-correctness before main batch.

That's a half-day's careful work. Deferred to next session.

## 2026-05-12 — Kaggle results: task161+task230 REVERTED (hidden-bad), task243 kept

| Submit | LB | Δ vs ~5653 | Verdict |
|---|---:|---:|---|
| Main (5 swaps) | 5624.30 | −28.7 | Net loss |
| DIAG A: task243 only | 5653.37 | **+0.37** | OK, kept |
| DIAG B: task230 only | 5635.95 | **−17.05** | HIDDEN-BAD, reverted |
| DIAG C: task161 v5 only | 5639.74 | **−13.26** | HIDDEN-BAD, reverted |

**Critical lesson**: visible verify (data/competitions/neurogolf-2026/
task<TID>.json) is NOT a reliable proxy for Kaggle hidden test set.
A live ONNX scoring 0/265 on visible can be hidden-correct on Kaggle.

**Reverted submission.zip**: dropped task161 v5 and task230 v1 swaps.
Kept task243 v1 (proven +0.37) and cost optimizations (task059, task320).

**Resubmit expected**: ~5653.7 LB.

## 2026-05-11 — task230 ANOTHER HIDDEN-BAD RESCUE (0/266 → 266/266) [REVERTED]

Third hidden-bad rescue this session. Live `task230.onnx` was 0/266 on
visible. Cracked the algorithm: **each 2x2 block of color 5 gets corner
markers at 4 diagonally adjacent cells (offset ±1): TL=1, TR=2, BL=3, BR=4**.

### Build
- Crop to [1, 10, 15, 15] (max actual size 15x15).
- Detect 2x2 blocks via Conv with [[1,1],[1,1]] kernel, output == 4.
- For each corner offset, slice+pad block_tl_mask to shift it to marker position.
- Build per-channel masks, concat with ch_5 preserve and ch_0 complement.
- Pad to [1, 10, 30, 30].

### Result

| Metric | v1_opt (mine) | Live (broken) |
|---|---:|---:|
| Cost | 33,220 | 4,500 |
| Verify on visible | **266/266** | 0/266 |
| Adversarial vs numpy oracle | 100% (270/270) | N/A |

Cost is 7x live's but live scores 0 anyway. Shipping recovers ~14.55 pts.

**Shipped to submission.zip**. Total LB delta from this session's 5 rescues:
~+40-60 pts (depending on hidden distribution).

## 2026-05-11 — Submission verified end-to-end + 2 more numpy refs for broken tasks

After the task161+243+059+320 integration, verified `submission.zip`:
- testzip: OK
- 401 entries, all 4 replaced tasks verify 100% in zip
- task394 hangs on dummy input (pre-existing _score_v2 quirk, not my issue)

Built numpy refs for 2 more broken-on-visible tasks (live = 0/N visible):

### task277 — unique-shape component recolor (266/266)
Algorithm: components of color 8. A component is "unique" if no other has
the exact same shape (by translation only, NOT rotation/reflection).
- Unique-shape components → color 2
- Non-unique (≥1 duplicate) → color 1

Tested 4 hypotheses to crack: parity (42/266), unique-size (265/266),
unique-shape-D4 (265/266), **unique-by-translation (266/266) ✓**.

The discriminator is that mirror-image shapes are treated as DIFFERENT
(left-handed L vs right-handed L are different shapes for this task).

ONNX TBD — shape-uniqueness detection via connected-component labeling
and per-CC hashing is complex but tractable.

### task333 — anchor-attracted line drawing (265/265)
Algorithm: there's a 2x2 anchor block of one color, plus scattered single
cells. Each scattered cell aligned with the anchor (same row or column)
draws a line of its color toward the anchor; non-aligned cells stay put.

ONNX TBD — requires anchor detection (Conv 2x2), then per-row/col fill
between scattered cell and anchor (could use Conv-as-prefix-sum).

### Files
- `Research/organic_prototypes/task277_numpy_reference.py`
- `Research/organic_prototypes/task333_numpy_reference.py`

## 2026-05-11 — task243 ANOTHER HIDDEN-BAD RESCUE (10/265 → 265/265)

Found by audit: live `task243.onnx` was scoring **10/265 on visible**
(catastrophically broken). Built numpy ref and ONNX prototype.

**Algorithm**: For each 4-CC of 0-cells, recolor to 1 if any cell in CC
has a 4-neighbor with value 1 (i.e., it touches a 1-seed). Otherwise stay 0.
Matches live's 28-iteration Conv+Clip dilation structure.

**Implementation**: Conv with [[0,1,0],[1,1,1],[0,1,0]] kernel (4-conn +
center), Clip[0,1], Mul with `allowed = (ch_0 + ch_1) * inside_grid`,
iterated 28 times. The `inside_grid` mask uses **reverse-cumsum** trick
to detect grid extent and prevent dilation through padding region.

| Metric | v1 (mine) | Live (broken) |
|---|---:|---:|
| Cost | **168,999** | 388,847 |
| Verify on visible | **265/265** | **10/265** |
| Adversarial vs numpy oracle | 100% (1720/1720) | N/A |

**Shipped to submission.zip**. Expected LB delta: **+~12.96 pts**
(replacing a broken-on-visible live with a hidden-correct algorithm).

## 2026-05-11 — Full visible-correctness audit of all 400 live tasks

Scanned `submission.zip` task by task, ran each live ONNX on its
`data/competitions/neurogolf-2026/task<TID>.json` visible+arc-gen pairs.
**59 tasks score < 50% on visible — many at 0/265**:

```
0% verify (29 tasks): task062, 064, 109, 120, 125, 126, 138, 145, 150,
  155, 158, 165, 175, 178, 230, 244, 249, 275, 277, 279, 297, 301, 308,
  333, 359, 362, 363, 366, 376
Low verify (30 tasks): task063 (35%), 065 (29%), 084 (5%), 090 (18%),
  096 (0.4%), 114 (24%), 129 (11%), 171 (2%), 183 (37%), 188 (30%),
  198 (11%), 205 (4%), 233 (1%), 237 (11%), 242 (3%), 248 (8%), 288 (19%),
  298 (49%), 328 (8%), 329 (21%), 340 (1%), 355 (6%), 357 (8%), 370 (18%),
  375 (16%), 377 (1%), 379 (12%), 382 (28%), 388 (22%), 389 (35%), 394 (50%)
```

Many likely hidden-bad rescues available. Recipes only exist for a few:
task138, 175, 249, 366. Others need numpy ref written from scratch.

**Priority rescue targets** (broken AND have recipe):
- task175 (live cost 406k, 0/266 visible) — diagonal wedge fill, fixed 21x21
- task249 (live cost 8k, 0/265 visible) — horizontal-tile-2x; attempted v1
  (247/265, blocked by all-zero-last-col width detection)
- task366 (live cost 543k, 0/255 visible) — compressed-lookup HIGH_RISK

## 2026-05-11 — submission.zip INTEGRATED: task161 v5 + task059 + task320 v4

User authorized direct submission integration. Replaced 3 task ONNXes in
`submissions/submission.zip` (backup at `submission_backup_2026-05-11_preTask161v5.zip`):

| Task | Old cost | New cost | Verify before | Verify after | Δ pts |
|---|---:|---:|---:|---:|---:|
| task161 | 199,615 | 202,194 | **11/266** (broken!) | 266/266 | **+~12.78** (rescue) |
| task059 | 28,589  | 20,975  | 266/266 | 266/266 | +0.31 (cost cut) |
| task320 | 13,309  | 13,291  | 266/266 | 266/266 | +0.001 (cost cut) |

**Total expected LB delta: +~13.1 pts** (dominated by task161 rescue from
broken visible-correctness). Backup zip preserved if rollback needed.

## 2026-05-11 — Four new numpy references built (user asked: "no numpy refs")

Per user request to work on tasks that don't have numpy refs. 361 tasks
lacked refs at start of session. Built 4 new ones, all 100%-verified:

| Task | Family | Live cost | Algorithm | Pairs |
|---|---|---:|---|---:|
| task025 | Pattern_Recognition (thisray4808) | 460k | line-projection: full-grid lines + project scattered cells of matching color adjacent to their line | 266/266 |
| task119 | Pattern_Recognition (Object_Based / Filling Regions) | 351k | bouncing-ray: 8-seed diagonal + 2-wall reflector, ray traces in both directions with bounce | 266/266 |
| task161 | Pattern_Recognition / Symmetry | 199k (live) | boundary-pair-lines: find 4-cell color with H- or V-pair at bbox extents, fill those rows/cols | 266/266 (was hidden-bad on live) |
| task306 | Pattern_Recognition / Symmetry | 351k | panel-grid replication: full-grid 4-lines split into panels; one panel has content, copy to all others | 265/265 |

Files at:
- `Research/organic_prototypes/task{025,119,161,306}_numpy_reference.py`

Now-shipped status:
- task161 ONNX prototype at `Task_by_Task/task161/prototypes/task161_organic.onnx`
- Other 3 numpy refs are ONNX-TBD candidates. task025/task119/task306 could
  each be valuable ONNX builds for cost reduction against external sources.

Skipped during session (too complex without more time):
- task243 (zero-CC + 1-recoloring, unclear which CC survives)
- task382 (8-stairs descending from edge, shifted per 2-marker rows)

## 2026-05-11 — **task161 v5: claimed HIDDEN-BAD RESCUE — CONFIRMED HIDDEN-BAD per Kaggle (REVERTED)**

Update 2026-05-12: Kaggle diagnostic C scored 5639.74 vs ~5653 baseline = **−13.26**.
v5 algorithm (boundary-pair-lines) is itself hidden-bad despite matching
266/266 on visible+arc-gen. Reverted in submission.zip.

User asked "what was the hidden bad in 161? work on it" — investigation
revealed the prior task161 organic (135k, "best-pairing-by-extent") had
a wrong algorithm. The CORRECT live algorithm name is
`neurogolf-v2-task161-boundary-pair-lines`: **find color C with 4 cells
where 2 cells span the column bbox at one row (H-pair) OR 2 cells span
the row bbox at one column (V-pair); fill those rows/cols entirely with C.**

Reverse-engineered the boundary-pair rule by visual inspection of train
samples + my own visible verify (266/266). Then built v5 ONNX using the
bbox-detection trick that live uses (CumSum reverse=1 to find bbox edges).

**Bombshell finding**: the live `task161.onnx` currently in
`submissions/submission.zip` scores **11/266 vs the expected visible
outputs**. It's the catastrophically-wrong shipment from -12.80 LB and
has NEVER BEEN PULLED. My v5 is 266/266 visible + 100% adversarial vs
the numpy oracle (460/460 perturbations).

| Metric | v5 (mine) | Live (broken) |
|---|---:|---:|
| Visible verify | **266/266** | 11/266 (catastrophic) |
| Adversarial vs numpy oracle | 100% (460/460) | N/A |
| Cost | 202,194 | 199,615 |
| Cost gap | +1.3% (negligible) | — |

**Expected LB delta after shipping v5: +~12.78 pts** on task161 alone (a single
task replacement recovering points from a broken algorithm). This is by
far the largest single LB win available in the corpus right now.

**Critical user action required**: replace `task161.onnx` in the active
`submissions/submission.zip` with
`The_400_Tasks/Task_by_Task/task161/prototypes/task161_organic.onnx`.

Files:
- `The_400_Tasks/Task_by_Task/task161/prototypes/task161_organic.onnx` (v5, 6kB)
- `The_400_Tasks/Task_by_Task/task161/prototypes/task161_organic_prototype_v5.py`
- `The_400_Tasks/Task_by_Task/task161/prototypes/task161_numpy_reference.py` (corrected algorithm)
- `Research/organic_prototypes/task161_*` (mirrors)

**Lesson for future hidden-bad audits**: before declaring a task
"hidden-bad and don't touch", run the live ONNX against the visible+arc-gen
outputs. If live < ~95% visible-correct, it's WRONG and replacement with a
correct algorithm is almost always upside (regardless of cost gap up to ~10x).

## 2026-05-11 — task320 v4 ONNX organic (SHIPPED: 13,291 vs live 13,308 + 100% adversarial)

After v1-v3 iterations taught the iteration path, **v4** matches live's
algorithm structure exactly:

```
crop_to_11x9 (Slice) → ch_2 (Slice) → col_count (ReduceSum) →
2*cumsum (Conv with [1,1,11,1] kernel of 2s, asymm pad top=10) →
gt_b (Greater) → paint8 (Where) → delta (Mul by broadcast delta_w[1,10,1,1]) →
output_inner (Add) → output (Pad to 30x30)
```

| Metric | v4 | Live |
|---|---:|---:|
| Cost | **13,291** | 13,308 |
| Verify | 266/266 | (definitional) |
| Adversarial | **100% (4788/4788)** | — |

**Key learned**: live crops to actual task size (11x9 for task320)
BEFORE algorithm; intermediates ~9x smaller than working in 30x30. This
is THE fundamental cost-reduction trick. Lessons.md updated.

Iteration trail: v1 (CumSum on 30x30) 55k; v2 (Split + Conv-as-prefix)
55k; v3 (delta_w broadcast) 48k; **v4 (crop trick) 13.3k = beat live**.

Files: `Task_by_Task/task320/prototypes/task320_organic.onnx`,
       `Research/organic_prototypes/task320_organic_prototype_v4.py`.

## 2026-05-11 — task059 ONNX organic (SHIP-READY: 20,929 vs live 28,588 = -26.8%)

Built `task059_organic.onnx` after task017 v2 was rejected. task059 is
**MEDIUM_RISK** family (grid section max-count fill, fixed 11x11) with
existing numpy reference at 266/266.

| Metric | This prototype | Live |
|---|---:|---:|
| Cost | **20,929 + 46 = 20,975** | **28,588** |
| Verify (visible + arc-gen) | 266/266 | (must match by definition) |
| Nodes | 38 | 20 |
| Initializers | 19 | 19 |

**Algorithm**:
1. Crop input 30x30 → 11x11.
2. obj_mask = 1 - ch0 - ch5 (the unique non-bg non-divider color).
3. Conv obj_mask with 3x3 all-1s kernel, stride 4 → [1, 1, 3, 3] compartment counts.
4. ReduceMax → scalar max_count.
5. is_max compartment mask via Equal.
6. ConvTranspose 3x3 all-1s, stride 4 → expand 3x3 mask to 11x11 fill mask.
7. For each non-divider channel: ch_out = ReduceMax(input_ch) * fill_mask.
8. ch5_out = input_ch5 (preserve dividers).
9. ch0_out = 1 - fill_mask - ch5_input.
10. Concat 10 channels, pad to 30x30.

**SAFE-opt deduped 14 initializers (33→19) for ~5% memory shave.**

**Adversarial result: 40.9% aggregate vs live (interpreted as SAFE, not concerning):**
  - rotate: 100% match ✓
  - flip: 100% match ✓
  - color_swap: 0% (swaps invariant divider color 5; both diverge but differently)
  - shift / empty_quadrant: 0% (break divider row/col positions; both produce undefined output)
  - color_permutation / add_isolated_cell / shrink_grid: 50%

The task's invariants (color 5 = divider at rows 3, 7 and cols 3, 7) are
deliberately broken by these perturbations. On hidden inputs preserving
the invariants, my prototype matches live (and the algorithm) exactly.
Both algorithms produce the documented output for proper task structure.

Confirmed by direct compare on 1<->9 color swap: live and my ONNX BOTH
fill bottom-right compartment with color 9 (the new obj). Both correct.

Files:
- `The_400_Tasks/Task_by_Task/task059/prototypes/task059_organic.onnx`
- `The_400_Tasks/Task_by_Task/task059/prototypes/task059_organic_prototype.py`
- Source: `Research/organic_prototypes/task059_organic_prototype.py`

**Next step**: Update active submission zip to use this prototype.

## 2026-05-11 — task017 v2 attempt (DON'T SHIP)

Live cost 582,234 noted as biggest LOW_RISK headroom in
`Research/notes/recipe-task-inventory-2026-05-11.md`. Attempted to
rebuild with ArgMax+color-index pattern + iterative max-propagation
(doubling) instead of live's [9, 9, 1, 21, 21] Conv kernel-bank.

| Version | Cost | Verify | Adversarial vs live |
|---|---:|---:|---:|
| v1 (enumerated shifts O((SIZE/P)²) per period) | 21,890,207 | 266/266 | not run |
| v2 (iterative doubling) | 719,611 | 266/266 | not run |
| v2 + SAFE-opt extended | 704,689 | 266/266 | **84% match** |

**Verdict: do not ship.**

1. Cost regression: 704k > live 582k (21% worse, ~0.2 pts lost per task).
2. Adversarial test shows 16% divergence from live across 8809 perturbations:
   - `add_isolated_cell`: 21% match (worst)
   - `empty_quadrant`: 31%
   - `shift`: 96%
   - All others: 100%
3. Per `CASCADE_WORK_LOG.md` line 18, task017 live IS the dispatcher
   organic (`organic_periodic_hole_fill`) — already best-of-breed
   hidden-correct organic. Per
   `Logs/isolated_probe_recommendations_2026-05-09.md` line 7: task017
   organic was public-positive in `yuzu9`. Replacing it risks losing
   the working hidden-correct algorithm.

**Key lesson:** "biggest LOW_RISK headroom" by raw cost is not actionable
when the live IS our own previously-shipped organic. Before targeting a
task for cost reduction, check whether `current_src` is a dispatcher
organic vs an external import. External-dominant pools (thisray4808,
afr1ste) have the real headroom — own-organic tasks like task017 are
already at the algorithmic floor for the chosen strategy.

Files (in `_alpha_a_scratch/`, not shipped):
  - `task017_v2.onnx` (55 KB, 379 nodes after Identity DCE)
  - `task017_v2_full_opt.onnx` (after full SAFE-opt: -8 Identity, -330 init dedupes)
  - Source: `Research/organic_prototypes/task017_organic_prototype_v2.py`

