# Deep-read: `neurogolf-locally-solved.ipynb` + the `ashok205/neuro-golf-2026` artifact

The `.ipynb` itself is a **6-cell glue notebook** (~19 KB) that copies pre-built ONNX files from the dataset `ashok205/neuro-golf-2026` into `submission.zip`. The actual content is in the dataset (432 KB compressed → 160 ONNX files). Pulled via `kaggle datasets download ashok205/neuro-golf-2026` (network was previously blocked, came back online 2026-05-02). Files unpack to `data/competitor_artifacts/ashok205-neuro-golf-2026/submission/task*.onnx`.

**Author's caveat:** "This submission only has 160 tasks and doesn't contain all of my solved tasks (I solved around 197 locally). My solved tasks are not 100% optimal as well, so feel free to optimize them." So the 160-file artifact is a strict subset of their full solved set, and per-file cost is not minimized.

## Overlap with our 13 SOLVED_TASKS

12 of our 13 are in their 160 — `{16, 53, 87, 135, 140, 179, 241, 266, 272, 326, 337, 380}`. The one they don't have: **`task129`** (our `const-dominant-color` pattern). This means they likely solve task129 in their full 197 locally but didn't include it in this 160-task artifact, or they use a different approach we haven't seen. Worth pulling the full set if possible, and worth confirming our task129 solution is robust.

**Regression-defense implication:** the 12 overlapping tasks all show ashok205 also solves them, so our patterns for these are at least one independent confirmation away from being silent-fail risks. Three of the 12 (53, 129, 266) appear on competitors' skip lists per `03_PROJECT.md` Open bugs — but ashok205 ships solutions for 53 and 266 (just not 129), suggesting those skip lists are blend-failure-mode-specific, not universal silent-fails.

## Net-new candidates (148 tasks)

Clustered by ONNX op-signature. `sp` = shape-preserved count (input shape == output shape).

| Cluster | Count | Avg size | sp/N | Op-signature | Pattern interpretation |
|---|---|---|---|---|---|
| A | 9 | 5.4 KB | 8/9 | `Conv+Mul+ReduceMax` | Single Conv feature detector + mask + max-aggregate. "Find specific kernel match, place output where matched." |
| B | 8 | 11.2 KB | 4/8 | `Gather+Mul+Reshape×2` | **Pixel-permutation with mask** — exactly `make_gather_onnx(mask=...)` from multi-source-onnx-solver. Already on the Tier-S handoff list. |
| C | 4 | 0.6 KB | 0/4 | `Concat+Constant×4+Gather+Pad+Slice` | **Mirror-h-concat / mirror-v-concat** — `[input | flip(input)]`. Already on Tier-S list. |
| D | 4 | 14.6 KB | 3/4 | `Conv×2+Mul+ReduceMax+Relu` | Two-layer Conv detector with ReLU between. More complex feature extraction. |
| E | 3 | 0.9 KB | 0/3 | `Concat×3+Constant×5+Gather×3+Pad+Slice` | **Quad-mirror** (`[[in, flip_h(in)], [flip_v(in), rot180(in)]]`). Already in my Tier-A #18 list. |
| F | 3 | 1.5 KB | 0/3 | `Add+Concat+Constant×11+Mul+Pad+Slice×3+Sub×5` | "Crop-then-color" / framed-extraction with masked color-add. Likely matches the multi-source `make_cheap_crop_then_color`. |
| G | 3 | 1.9 KB | 3/3 | `Add×2+Conv×3+Mul+ReduceMax×3` | Three-Conv layered feature detection. Heavier than cluster A; probably handles tasks where the rule needs multi-pass evidence. |
| H | 2 | 0.6 KB | 2/2 | `Conv` | **Pure 1×1 Conv color permutation** — see "Color-replace (general)" below. |
| I | 2 | 1.2 KB | 2/2 | `Concat+Constant×7+Max+MaxPool×4+Mul×2+Slice×3+Sub×3` | MaxPool-based **dilate-around-marker** (place 3×3 block at each marker position; cluster I and similar). |
| J | 2 | 1.6 KB | 0/2 | `Abs+Concat+Constant×11+Pad+Slice×3+Sub×4` | Likely shape-changing crop with absolute-value evidence (rare; may not generalize). |
| K | 2 | 15.6 KB | 2/2 | `Add+And+Cast×4+Clip+Gather+Greater+Less×2+...` | Heavy combinatorial logic (boolean chains). Unlikely to fit our pattern style. |
| L | 2 | 19.6 KB | 0/2 | `Add+And+Cast×21+Concat+Conv+Div×2+Equal×19+...` | Even heavier; almost certainly won't fit. |
| M | 2 | 43.7 KB | 0/2 | `Constant×4+Gather+Mul+Reshape×2` | Like cluster B but with explicit mask Constants. ~2× the size — likely larger Gather table (different shape regime). |
| singletons | ~108 | varies | varies | varies | Per-task hand-crafted. Tackle per-task as time allows. |

## Concrete confirmed patterns (from grid inspection of train pairs)

- **task276 (3×4)**: input colors `{6,7}`. Output: 6→2, 7→7. **Single-color permutation**. Why our `_detect_color_perm` doesn't catch: input has only 2 distinct colors (vs our threshold ≥7-of-10 from `lessons.md`).
- **task309 (3×6)**: input `{1,7,8}`. Output: 1→1, 7→5, 8→8. **Single-color permutation**. Same threshold blocker.
- **task164 (3×3 → 3×6)**: output = `[input | flip_h(input)]`. **Mirror-h-concat**.
- **task172 (3×3 → 6×3)**: output = `[input ; flip_v(input)]`. **Mirror-v-concat**.
- **task210 (3×3 → 6×3)**: output = `[input ; flip_v(input)]`. **Mirror-v-concat**.
- **task311 (3×3 → 3×6)**: output = `[input | flip_h(input)]`. **Mirror-h-concat**.
- **task142, 152, 83 (3×3 → 6×6 etc.)**: **Quad-mirror** confirmed visually. `[[in, flip_h(in)], [flip_v(in), rot180(in)]]`.
- **task317 (9×9, sp)**: input has sparse 5-color cells; output has 3×3 blocks of color 1 centered at each 5-position (5 itself replaced with 1). **MaxPool-3×3-then-recolor**.
- **task095 (9×9, sp)**: same but the 5 is preserved (3×3 of 1's surrounds it). **MaxPool-3×3-with-center-preserved**.

## Ranked `[Alpha]` candidate batches

These build on the Tier-S list already in `03_PROJECT.md` (filed in the multi-source-onnx-solver deep-read). Same pattern names; this surfaces specific task numbers and confirmed counts.

### Batch 1 — Color-replace (general, multi-mapping) AND Color-perm with relaxed threshold

Already on Tier-S list. Concrete net-new tasks confirmed: **`task276`** and **`task309`** (both single-mapping but blocked by our `≥7-of-10 colors observed` threshold). Almost certainly more in the singletons list with similar shape. **Action:** loosen `_detect_color_perm`'s `require_all_colors` (use `False` mode) AND lower the colors-observed threshold (e.g., to `≥3` if the input vocabulary itself has only those colors). Re-run and count hits.

### Batch 2 — Mirror-h-concat / Mirror-v-concat (Tier-S Slice + Concat)

Already on Tier-S list. Concrete confirmed: **`task164`**, **`task172`**, **`task210`**, **`task311`**. Plus likely 2-5 more in the singletons that just have additional cells (Pad / Concat variations). Cost ~0.6 KB → trivial. ~14 pts each.

### Batch 3 — Quad-mirror (Tier-A; promote to Tier-S given confirmed hits)

Already on Tier-A list (#18 in multi-source-onnx-solver note). Concrete confirmed: **`task83`**, **`task142`**, **`task152`**. Cost ~0.9 KB → still small. ~14 pts each. **Promote priority.**

### Batch 4 — Pixel-permutation with mask (Tier-A; promote)

Already on Tier-A list (#9 in multi-source-onnx-solver note). 8 net-new tasks here: **`108, 113, 194, 211, 214, 307, 373, 385`**. Cost ~11 KB (length-900 int64 Gather table is ~7.2 KB; mask adds the rest). ~12-13 pts each. Plus the 2 `Constant4+Gather+Mul+Reshape2` variants (`116, 223`). Total ~10 hits if the detector works on these.

### Batch 5 — MaxPool-3×3 dilate-then-recolor (NEW pattern)

Not yet on any list. Concrete confirmed: **`task317`**, **`task95`**. Pattern: detect channel-5 markers; MaxPool 3×3 to dilate; remap 5-channel output to channel 1 (with optional preserve-center for task95). Builder sketch: `Slice(ch5) → MaxPool 3×3 → 1×1 Conv to remap 5→1 → optionally Sub(original 5 channel scaled) → Add(input)`. Cost likely ~50-90k → ~12-13 pts. Estimated 2-5 hits.

### Batch 6 — Single Conv feature detector + Mul + ReduceMax (Tier-B; cluster A)

9 tasks; mostly shape-preserved. **`180, 15, 73, 151, 282, 344, 352, 120, 322`**. Each is a "find specific kernel match, mark detection positions" pattern. Cost ~5 KB ONNX-side (Conv weights + ReduceMax intermediate). Implementation requires per-task feature analysis to pick the Conv kernel; not a single generalized detector but a **template**. Maybe 1-2 fall out as "clean" easy wins (e.g., the smallest ones at 120 / 322 / 282). Defer to per-task `[Alpha]` hand-coding sessions.

### Batch 7 — Crop-then-color masked (Tier-B; cluster F)

3 tasks: **`386, 26, 144`** plus the `Slice4+Sub` variants. Multi-source-onnx-solver has `make_cheap_crop_then_color` (cell 3) — Slice + Gather (channel) + Mul (kill mask) + Pad. Likely transferable directly. ~1.5 KB → ~14 pts. Estimated 3-8 hits.

## Tasks NOT worth attempting (heavy & unlikely to fit)

- Cluster K, L (4 tasks total: `150, 155, 269, 289`) — heavy boolean-logic chains with Cast×21, Equal×19, etc. The kind of thing a brute-search PyTorch-fit produces. Skip unless every other pattern is exhausted.
- Most singletons with ops like `CumSum, Floor, Div, Neg` mixed in — these are bespoke per-task constructions that don't generalize. Worth a per-task look only if specifically targeted.

## Cost-side observations

- The 4 mirror-concat tasks (cluster C) are 0.6 KB each — among the absolute smallest non-trivial ONNX files in the artifact. **Slice + Pad-only operations** are cheaper than our current Gather-based geometry. Consistent with the multi-source-onnx-solver's `make_cheap_*` family observation.
- The 8 Gather-mask tasks (cluster B) are ~11 KB — heavier because the Gather table is `int64[900]` (~7.2 KB) plus float32 mask. Worth checking whether `int32` Gather indices halve this (onnx-tool counts elements not bytes per A1, so dtype switch is cosmetic for `params` but real for `memory`).
- 9 single-Conv tasks (cluster A) at ~5 KB each — Conv kernel `[C, C, k, k]` + bias dominates. For shape-preserved tasks, this is roughly the standard `make_conv_onnx` from multi-source-onnx-solver.

## Action items for Beta in next pass

1. Pull `karnakbaevarthur/neurogolf-2026-task-transformation-library` (DeepSeek per-task primitive CSV referenced in `03_PROJECT.md` Strategic experiments) — now that the network is back, this is a high-leverage data source for verifying cluster interpretations.
2. Pull the higher-LB artifacts (`afr1ste/neurogolf-6285-95-public-score-open-solution`, `neurogolf-6645-39-open-submission-artifact`) to compare against ashok205's 160 — gap = candidates for further patterns.
3. Run our `silent_fail_check.py` against ashok205's 12 overlapping tasks to confirm their ONNXs also pass our visible-pair verify (defensive cross-check).

---

*Filed 2026-05-02 by Beta lane during the followup batch (after silent_fail_check + multi-source-onnx-solver deep-read landed in PR #15). Source dataset: `ashok205/neuro-golf-2026` (160 ONNX, 432 KB compressed). Network came back up after 2026-05-02 outage flagged in `03_PROJECT.md` Open bugs.*
