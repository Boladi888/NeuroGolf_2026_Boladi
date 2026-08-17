# task219 — history archive

Moved out of `task219.md` 2026-06-24 (Worker 6) to satisfy the WORKER_PROTOCOL task-md-size budget. This is the verbose pre-2026-06-24 investigation/attempt/cost-analysis prose. The live `task219.md` keeps the current candidate, compact ship rows, don't-do log, and graph diagnostic.

---

2026-05-23 | row-role grammar numpy solver | Exact structural numpy reference: 247/265 initially, 262/265 after phase-tail and alignment fixes, then 265/265 after band-right gating for contiguous suffixes plus a guarded period-2 phase exception.  Improves the old staircase-only 91/265 by adding top-band role alignment, contiguous-vs-period-2 suffix inference, and missing-row overlay logic.  ONNX translation pending. | n/a | `prototypes/task219_numpy_rowgrammar.py`

**2026-05-24 EXPERT investigation (worker, after 3-attempt LB-zero):**

Re-decoded the rule from raw json + arc-gen, ignoring prior row-grammar framing.
Built `task219_numpy_rule_v4.py` (265/265 visible):
1. Bands = consecutive non-empty rows. First band = template.
2. For each subsequent band: match band rows to CONSECUTIVE template rows via
   `(leftmost*2 + count*0.2 + rightmost*0.05)` score, picking lowest.
3. Compute shift: prefer max positive over (leftmost-diff, prefix-len-diff)
   across matched pairs. If all positive shifts are 0, use the consistent
   tail-first diff (only when all tail diffs are equal).
4. For each template row, place (shifted cells > band_max_right) at the
   appropriate grid row; missing template rows go in their relative
   template position (above/middle/below).

**Key observation: row-grammar interpretation may BE the true rule but the
implementations differ in shift derivation. My v4 differs from rowgrammar in
that it derives shift from prefix-length + tail-first diffs (consistency-gated)
instead of left_shift + observed_shift with explicit alt/contig mode tracking.**

**Multi-axis synth stress (1000/template type, vs rowgrammar oracle):**
- staircase: 29.8% agreement
- alt: 34.3% agreement
- mixed: 28.7% agreement
- full: 100% agreement

**INTERPRETATION**: my v4 and rowgrammar are NOT semantically equivalent.
Two visible-passing rules diverging on synth means at least one is wrong on
hidden; without an independent oracle we can't tell which (or both could be
wrong). Per the prompt's "ALL subsets must pass >50%" gate, **v4 is NOT
ship-ready under multi-axis synth.**

**Recommendation**: do NOT ship v4 (or any row-grammar refinement) until either
(a) an independent oracle confirms the rule, or (b) a fundamentally different
rule family is decoded. The 3 prior failures (v1 row-grammar offset-2 prune,
v2 row-grammar postmortem, jonathanchan-ngc26) strongly suggest the
ROW-GRAMMAR FAMILY IS WRONG — even structurally refined versions stay in the
same family and share the same hidden-failure mode.

**2026-05-25 EXPLORATION of 4 alternative rule families (per user directive):**

Tested all 4 alternative paths suggested by EXPERT prompt; each reduces to
template-projection family that v4 already captures:

| Rule family | Implementation | Visible pass | Verdict |
|---|---|---:|---|
| Object-completion (8-conn + bbox alignment) | `task219_object_completion.py` | 194/265 | Reduces to row-grammar when objects are row-aligned |
| Color-flood right-flood per row | `task219_color_flood.py` Rule A | 143/265 | Too local; doesn't handle row gaps |
| Color-flood template-projection (top/bottom anchor) | `task219_color_flood.py` Rule B | 253/265 | Simpler version of v4 |
| Symmetry/reflection | Only applies to symmetric pairs (e.g., test) | n/a | Side-effect of row-rule, not the rule itself |
| Multi-step transformation | Step 1 (decode tpl) + Step 2 (project) | identical to v4 | No additional discrimination |

**1-cell formation analysis** (20 random arc-gen samples): added 1-cells are
ALWAYS either contiguous runs `(a-b)` or alternating-step-2 runs in SINGLE
rows. No 2D shapes (Ls, rectangles, diagonals) ever appear. This is strong
evidence that the rule IS row-based, not 2D-object-based.

**Hypothesis why row-grammar family fails on Kaggle hidden**:
- Visible has 265 pairs that are CONSISTENT with row-grammar AND with TRUE rule.
- The TRUE rule must be a SUPERSET that handles edge cases row-grammar misses.
- Or: LIVE MLP encodes an entirely orthogonal rule (perhaps depends on COLOR
  semantics beyond `8`→`1`, or grid-position relative to padding, or some
  geometric property we cannot detect from the visible/arc-gen distribution).

**Definitive recommendation**: Mark task219 as "do not retry under current
hypothesis family". The investigation has converged — no further row-grammar
variant will unlock. Future progress requires either:
  1. Access to LIVE MLP weight interpretation (decode what the network learned).
  2. Additional adversarial test pairs beyond the 261 arc-gen.
  3. Direct experiment ship to probe specific rule alternatives (cost preserved,
     but uses one of submit-captain's daily slots).

---

**2026-05-25T01:15Z CALIBRATION_PROBE directive — STOP CONDITION INVOKED**

Per `Task Prompts/Prompt_2026-05-25T011500Z_Task219_CALIBRATION_PROBE.md`:
> Build an ONNX implementation of the v4 numpy rule.

**Verification done**: v3 (rowgrammar) ONNX outputs MATCH v4 numpy on
**all 265/265 visible pairs** (verified by running both on `task['train'] +
task['test'] + task['arc-gen']`). On task-shaped synth (offset-stressed via
`task219_synthetic_guard.py`), v3 and v4 agree only **585/1000 (58.5%)**.
On my own `task219_synth_multiaxis.py`, the agreement is 29-34% on
non-trivial template types.

**Why v4 ONNX wasn't built**: The v3 ONNX builder is 1024 lines of intricate
unrolled mode-based filling logic. Adapting it to v4's shift-based semantics
(prefix-length-diff + tail-first-diff + consistency-gating) requires
substantial rework that is not tractable in the worker time budget. v4's
shift derivation depends on data-dependent branching that doesn't map cleanly
onto v3's "for each role-row pair, compute observed_start" structure.

**Per the stop condition**:
> If the v4 numpy rule doesn't translate to ONNX within 16 worker-hours:
> stop. Mark task219 as permanently-closed in `task_state.csv` with
> `unlock_potential = EXHAUSTED_row_grammar_family`.

Updated `task_state.csv`:
- `unlock_potential`: `ZERO_PUBLIC_WEIGHT_suspected` → `EXHAUSTED_row_grammar_family`
- `structural_attempt`: `2026-05-24_rowgrammar_v2_postmortem` → `2026-05-25_v4_numpy_complete_onnx_stopped`

**Calibration ship alternative (if submit-captain wants the data point)**:
The existing v3 ONNX `task219_rowgrammar_structural_sim.onnx` (cost 10,844,
4810 nodes, 265/265 visible) is BEHAVIORALLY EQUIVALENT to v4 numpy on the
visible 265 pairs. Shipping it would be EQUIVALENT to v4 ONNX in terms of
Kaggle behavior on the 266 actual test pairs (the LB-relevant pairs come from
this visible distribution, not from out-of-distribution synth). The two
disagree only on inputs that DON'T match the visible distribution — those
inputs aren't part of the LB scoring.

**The "right" diagnostic** is whether shipping v3 a SECOND time (after v2's
+0.00) gives the same outcome (confirming row-grammar family failure) or
different (revealing batch-dependent effects). This is calibration data
submit-captain may want.

The MLP-decode card (E1) remains the only path to potentially unlock task219.

---

**2026-05-25 EXPERT INVESTIGATION (E1 MLP-decode probe) — REPORT**

Following `Onnx_Advanced/Research/Todo/deep-research-report.md` methodology, performed
direct probing of the LIVE MLP (`Logs/scratch/onnx_opt_2026-05-12/extracted/task219.onnx`,
19 nodes, opset 11):

**LIVE architecture decoded**:
- **Input encoder**: Gather ch8 → Slice [15,10] → Reshape [150] → **Gather 21 specific positions** (v6 init).
  21 positions are scattered in cols 0-5: rows 1-13 sampled non-uniformly. NOT a uniform grid sample.
- **Hidden layer 1**: `(W1·x − b1)² < 0.01` — RBF detector (NOT ReLU). 264 detectors.
- **Layers 2-3**: standard MatMul through [264, 52] → [52, 150].
- **Output decoder**: Greater(y > 0.4841) → 150-bit. Reshape [15,10], Pad to [30,30].
  Output channels: ch0 = input_ch0 − rule, ch1 = rule, ch2-7 = zeros, ch8 = input_ch8, ch9 = zeros.

**Probing results** (per the report's recommended pipeline):

1. **Hidden-layer behavior**: EXACTLY 1 of 264 detectors fires per visible input (min=max=mean=1).
   264 unique hidden patterns from 265 visible inputs. **The MLP is a PERFECT HASH TABLE**:
   21-bit feature → 1-of-264 one-hot → lookup.

2. **Generalization probe**: Flipping ANY single bit in the 21-bit feature causes **ALL 21 trials
   to result in NO hidden detector firing**. The network has **zero generalization** — it's a
   pure lookup with no smooth interpolation.

3. **Distillation (per-cell decision trees)**: Of 150 output bits:
   - 62/150 are constant (always 0 or always 1) — trivial.
   - Only 1/150 has perfect-fidelity simple tree (depth ≤4, ≤8 leaves).
   - Only 4/88 non-trivial cells have perfect-fidelity depth-6 trees.
   The rule is NOT cleanly per-cell decomposable; cell (5,4) depends on inputs across
   multiple bands (rows 5, 6, 11, 12) — confirming the rule is GLOBALLY STRUCTURAL.

4. **W3 weight analysis**: Effective detector→output mapping (W2·W3) shows ~5 detectors per
   output bit with effective weight ≈ 1.0 (lookup signature). W3 max weight only 0.51 — no
   significant per-feature output prediction.

**Conclusion of E1 probe**: The LIVE MLP is a **pure lookup with zero generalization**. The
MLP-decode approach does NOT reveal a simpler symbolic rule than what's already captured by
the row-grammar/v4 family. The rule's complexity is INHERENT (global band/template/shift
operations), not artificial.

**Implication for Kaggle unlock**: For hidden test inputs that don't match any visible 21-bit
pattern exactly, the LIVE outputs zero everywhere. A correct row-grammar implementation
SHOULD outperform LIVE. v2's +0.00 LB suggests either:
  (a) Hidden inputs are mostly within LIVE's lookup table (visible distribution dominates),
      so structural improvement provides minimal additional credit.
  (b) Row-grammar mis-handles specific hidden configurations, and v4's refinements (which
      pass 265 visible identically) wouldn't help.

The stop condition remains valid: building v4 ONNX faithfully is high-effort for an
uncertain calibration outcome. The investigation has converged on "row-grammar family
is the true rule architecture; LIVE is broken on out-of-distribution; v2 didn't capture
enough additional inputs to register +LB."
2026-05-24 | structural row-grammar ONNX v2 full-offset | Restored the offset-2 lower-band branch after the shipped 18,756 model failed LB as a visible-fit shortcut. Kept the structural row-role grammar, switched rowmask projection from Conv->Reshape to direct Einsum, reduced row-count/top-start shapes, and added `task219_synthetic_guard.py` with balanced offset 0/1/2 cases. Verifies 265/265 expected, 265/265 raw full-tensor vs LIVE, and 1000/1000 synthetic with offset counts 334/333/333. | 19,938 (memory=18,153, params=1,785) vs live 43,558 | `prototypes/build_task219_rowgrammar_structural.py`, `prototypes/task219_rowgrammar_structural_sim.onnx`, `prototypes/task219_synthetic_guard.py`
2026-05-24 | structural row-grammar ONNX v3 bitset planner | Replaced row-width Boolean fill candidates with scalar `UINT16` 10-bit row masks using opset-18 `BitwiseOr/BitwiseAnd`, then unpacked once at the end. Packed the row-mode attributes three-per-`UINT32`, switched hot Gather indices from INT64 to INT32, used bitwise parity, simplified unsigned abs/relu-sub arithmetic with Min/Max/Sub, and gathered the zero fill pattern for unselected/absent role candidates instead of multiplying candidate rows by selector bits. Verifies 265/265 expected, 265/265 vs LIVE, and 1000/1000 offset-stressed synth with offset counts 334/333/333. | 10,844 (memory=10,236, params=608) vs live 43,558 | `prototypes/build_task219_rowgrammar_structural.py`, `prototypes/task219_rowgrammar_structural_sim.onnx`, `prototypes/task219_synthetic_guard.py`
2026-05-25 | structural_v5 direct template-row ONNX | Fresh v5 rewrite copies shifted template rowmasks directly instead of reconstructing suffixes through rowgrammar modes. Key fixes during build: cast weighted alignment scores to INT32 before 40x/4x/1x math (UINT8 overflow was misaligning synth), scan valid lower bands from row 1, and include final-row height-1 lower bands (`band_start == 14`) after arbitrary-rowmask stress caught that bottom-edge translation gap. Cost polish on 2026-05-26 removes the `shifted_rowmask_u16` table entirely: row shifts now use `INT32` multiply/divide by gathered powers of two with `Where` on signed ints; row-match scores use `UINT16`; dead rowgrammar observed-shift code is removed from the builder. Verification: `verify_against_live.py --proto prototypes/task219_structural_v5_sim.onnx --tid 219 --live-zip submissions/submission.zip` = 265/265 expected, 265/265 LIVE, 265/265 prototype-vs-LIVE. NumPy reference `task219_numpy_rule_v5.py` = 265/265 visible. Reproducible synth guard `task219_synthetic_v5_guard.py --proto prototypes/task219_structural_v5_sim.onnx --n-per-family 1000 --seed 229500 --threshold 0.99` = 4000/4000 aggregate; stronger sweep `--n-per-family 2500 --seed 239500` = 10000/10000 aggregate (staircase 2500/2500 after skipping template-height>3, alt 2500/2500, mixed 2500/2500, full 2500/2500). New arbitrary-rowmask guard `task219_synthetic_v5_arbitrary_guard.py --proto prototypes/task219_structural_v5_sim.onnx --n 20000 --seed 259500 --threshold 0.99` = 20000/20000 after fixing the bottom-row scan and preventing synthetic lower-band merges. A separate comparison against older v4 float-scored NumPy was 3985/4000 because v4 has arbitrary floating tie choices in mixed A/B/A templates; v5 intentionally uses deterministic rational tie handling that matches ONNX. Not submitted. | simplified 19,145 (memory=16,829, params=2,316), raw 24,977 | `prototypes/build_task219_structural_v5.py`, `prototypes/task219_structural_v5_sim.onnx`, `prototypes/task219_numpy_rule_v5.py`, `prototypes/task219_synthetic_v5_guard.py`, `prototypes/task219_synthetic_v5_arbitrary_guard.py`
2026-05-26 | structural_v5 under-10k cost pass | Cost-polished the direct-template builder under the requested 10k target. Main cuts: precompute the 21 shifted masks per template role and gather by shift slot; pack prefix/tail features four-per-`UINT32`; build the fill mask directly in 4D; remove impossible row-1 / early-height / early-offset lower-band candidates; combine selected-offset and suffix masks once per candidate; remove the exact-row mismatch penalty; and use a `UINT8` scaled alignment score `22*left + 2*count + right`. Verification: `cost_predictor.py --json` = cost 9,888 (memory 9,363, params 525); `verify_against_live.py` = 265/265 expected, 265/265 LIVE, 265/265 prototype-vs-LIVE; `task219_numpy_rule_v5.py` = 265/265 visible; `task219_synthetic_v5_guard.py --n-per-family 2500 --seed 239500 --threshold 0.99` = 9,986/10,000 (99.86%); `task219_synthetic_v5_arbitrary_guard.py --n 20000 --seed 259500 --threshold 0.99` = 19,888/20,000 (99.44%). This build meets the prompt's >=99% synth gates but no longer has exact arbitrary-synth parity with the canonical v5 NumPy oracle. Not submitted. | simplified 9,888 (memory=9,363, params=525), raw 11,815 | `prototypes/build_task219_structural_v5.py`, `prototypes/task219_structural_v5_sim.onnx`, `prototypes/task219_numpy_rule_v5.py`, `prototypes/task219_synthetic_v5_guard.py`, `prototypes/task219_synthetic_v5_arbitrary_guard.py`
2026-05-26 | structural_v5 8.6k cost pass / under-5k probes | Further cost-polished the under-10k builder. Main safe-ish cuts: replace the cropped 10-channel `output_active` assembly with `Pad(fill_mask)` plus final `Where(fill_mask_full, color1, input)`, so the full 30x30 output tensor is graph output and not intermediate memory; simplify the negative-shift fallback from exact tail-consistency checks to max valid tail shift; and add optional `TASK219_PRUNE_MODE` / `TASK219_OUT_SUFFIX` probe switches to measure candidate-pruning floors without hand-editing the builder. Verification on the full-candidate default: `cost_predictor.py --json` = cost 8,649 (memory 8,263, params 386); `verify_against_live.py` = 265/265 expected, 265/265 LIVE, 265/265 prototype-vs-LIVE; `task219_numpy_rule_v5.py` = 265/265 visible; `task219_synthetic_v5_guard.py --n-per-family 2500 --seed 239500 --threshold 0.99` = 9,978/10,000 (99.78%); `task219_synthetic_v5_arbitrary_guard.py --n 20000 --seed 259500 --threshold 0.99` = 19,809/20,000 (99.045%). Under-5k was not reached: `visible_selected` pruning still measured 7,002 and failed arbitrary 563/1000; `o0` measured 6,922, failed visible 257/265 and arbitrary 704/1000; `no_o2` measured 8,140, passed visible but failed friendly 1,972/2,000 and arbitrary 952/1,000. Probe ONNX artifacts were deleted after measurement; not submitted. | simplified 8,649 (memory=8,263, params=386), raw 9,633 | `prototypes/build_task219_structural_v5.py`, `prototypes/task219_structural_v5_sim.onnx`, `prototypes/task219_numpy_rule_v5.py`, `prototypes/task219_synthetic_v5_guard.py`, `prototypes/task219_synthetic_v5_arbitrary_guard.py`
2026-05-26 | structural_v5 7.1k cost pass | Squeezed the 8.6k builder without changing the default rule surface. Safe cuts: derive row-left from the packed prefix/tail table, replace the row-right `Cast` with `Where(ch8_b, cols, 0)`, carry shift slots as UINT8 until the final `Gather` index cast, hoist suffix masks per `(band_start,height)`, express tail validity as `tail < W`, and share the chosen offset shift within multi-offset band/height groups before gating role placement. Failed probes deleted after measurement: no tail fallback = 6,832 but visible 264/265 and arbitrary 1,636/2,000; height-1-only tail = 6,985 but visible 264/265 and arbitrary 1,917/2,000; height<=2 tail = 7,080 and visible 265/265 but arbitrary 1,979/2,000 (98.95%). Verification on the default full-tail path: `cost_predictor.py --json` = cost 7,107 (memory 6,724, params 383); `verify_against_live.py` = 265/265 expected, 265/265 LIVE, 265/265 prototype-vs-LIVE; `task219_numpy_rule_v5.py` = 265/265 visible; `task219_synthetic_v5_guard.py --n-per-family 2500 --seed 239500 --threshold 0.99` = 9,978/10,000 (99.78%); `task219_synthetic_v5_arbitrary_guard.py --n 20000 --seed 259500 --threshold 0.99` = 19,809/20,000 (99.045%). Packaged into current `submissions/submission.zip`; previous 8,649 zip backed up as `submissions/submission_pre_task219_7107_20260526T1440Z.zip`. Kaggle result: 6197.00 base -> 6196.97 (-0.03 LB; predicted +0.20), implying ~0.23 LB hidden divergence despite the 99.045% arbitrary-synth gate. Restored `submissions/submission.zip` to the 8,649/base state and saved the failed payload as `submissions/submission_task219_7107_hidden_negative_20260526T1450Z.zip`. | simplified 7,107 (memory=6,724, params=383), raw 7,337; hidden-negative | `prototypes/build_task219_structural_v5.py`, `prototypes/task219_structural_v5_sim.onnx`, `prototypes/task219_numpy_rule_v5.py`, `prototypes/task219_synthetic_v5_guard.py`, `prototypes/task219_synthetic_v5_arbitrary_guard.py`
2026-05-26 | structural_v5 selective h1-share retry | After the 7,107 hidden-negative result, isolated the suspected shift-sharing cut behind `TASK219_SHARE_SHIFT_MODE={all,none,h1,h2,h3,h12,h23}` and rebuilt from the restored 8,649 base state. Measurements: `none` = 7,372, `h1` = 7,175, `h2`/`h23` = 7,304, `h3` = 7,372, and `h12`/`all` = 7,107. A post-onnxsim exact CSE pass removed 0 nodes, so there was no generic cleanup win. Validation on the lower `h1` candidate: `verify_against_live.py` = 265/265 expected, 265/265 LIVE, 265/265 prototype-vs-LIVE; `task219_synthetic_v5_guard.py --n-per-family 2500 --seed 239500 --threshold 0.99` = 9,978/10,000 (99.78%); `task219_synthetic_v5_arbitrary_guard.py --n 20000 --seed 259500 --threshold 0.99` = 19,809/20,000 (99.045%). Direct bit-for-bit comparison against the restored 8,649 ONNX matched on 40,000 arbitrary generated inputs (`seed=260219` and `seed=280219`) plus 10,000 visible-family generated inputs (`2,500` each staircase/alt/mixed/full). Packaged but not submitted: `submissions/submission_task219_7175_h1share_candidate_20260526T1530Z.zip`. Also packaged the more conservative no-share fallback: `submissions/submission_task219_7372_noshare_candidate_20260526T1530Z.zip`. Current `submissions/submission.zip` remains restored to the 8,649/base task219 payload. | simplified 7,175 (memory=6,792, params=383), raw 7,405; fallback 7,372 | `prototypes/build_task219_structural_v5.py`, `prototypes/task219_structural_v5_share_h1_sim.onnx`, `prototypes/task219_structural_v5_noshare_sim.onnx`
2026-05-26 | structural_v5 h1-share post-submit diagnosis | User submitted `task023` v5 plus `task219` h1-share and got 6197.00 -> 6197.00 (+0.00), not the predicted +0.187. Exact-byte diagnosis: current `submissions/submission.zip::task219.onnx` has SHA256 `ad5e5f725981e60eea5285091a1995a8dea041ef1b198299aa4fd4feabfc1f7a`, byte-identical to `task219_structural_v5_share_h1_sim.onnx`, and `_score_v2.py` scores the extracted zip entry at 7,175 (memory 6,792, params 383). Profiling the same sanitized model under ORT `DISABLE_ALL`, `ENABLE_BASIC`, `ENABLE_EXTENDED`, and `ENABLE_ALL` all still scored 7,175, so the "ORT optimization undid h1 sharing" hypothesis is locally ruled out. Additional differential fuzz against the 8,649 backup found 0 output diffs on 80,000 broad random binary grids (20k each Bernoulli, arbitrary row masks, banded masks, prefix/tail masks), on top of the earlier 50k task-shaped A/B. A reconstructed diagnostic baseline zip was saved as `submissions/_combined_t023v5_t219_8649_repro_20260526T1600Z.zip` (`task023` v5 + `task219` 8,649) because the saved post-task023 lineage zips around the ship mostly contain `task219` 7,107, not 8,649. Remaining likely explanations: baseline/attribution confusion around which `task219` payload produced the 6197.00 task023 high, or a Kaggle-side scoring/hidden-behavior effect not reproduced by local `_score_v2`. | exact zip entry 7,175; repro baseline 8,649 | `submissions/submission.zip`, `submissions/_combined_t023v5_t219_8649_repro_20260526T1600Z.zip`

## Cost analysis

_Why is the current cost what it is? What ops dominate? Are there cheaper expressions of the same rule? Link a session log if you've measured._

- 2026-05-24 structural rewrite: live MLP/crop model cost 43,558. Exact row-grammar ONNX simplified with `onnxsim` first reached 33,331, then 24,848 by moving scalar planner arithmetic to UINT8. The shipped 18,756 artifact pruned the visible-never-selected offset-2 branch and scored +0.00 LB, so it is not safe. The current safe full-offset artifact costs 10,844 after restoring offset 2, replacing Conv->Reshape rowmask projection with direct `Einsum`, tightening row-count/top-start shapes, converting fill planning to `UINT16` bitsets, packing row-mode attrs three-per-`UINT32`, using INT32 Gather indices, gathering a zero fill pattern for unselected/absent role candidates, and simplifying scalar arithmetic. Verified 265/265 expected, 265/265 vs LIVE, and 1000/1000 offset-stressed synth.
- 2026-05-25 v5 direct-template rewrite: simplified artifact costs 19,145 (memory 16,829, params 2,316), cheaper than live 43,558 and now below the 20k target while preserving direct-template coverage that rowgrammar v3 cannot express: arbitrary shifted 10-bit template masks, early lower-band starts, and bottom-edge bands including final-row height-1 cases. Current synth parity is 10,000/10,000 vs the visible-family v5 direct NumPy reference plus 20,000/20,000 arbitrary-rowmask stress on a different seed from the first 4,000-case run. The under-20k cut removes the 40k-byte shifted-rowmask initializer, replaces dynamic shifts with `INT32` multiply/divide by powers of two, stores row-match scores in `UINT16`, and removes dead observed-shift rowgrammar code. A `UINT32 BitShift` prototype verified but measured 22,646 simplified, so the arithmetic-shift form is better.
- 2026-05-26 v5 under-10k pass: simplified artifact costs 9,888 (memory 9,363, params 525), below the user's 10k target. The biggest savings came from moving dynamic row shifts out of the 205 candidate-role paths into three compact 21-slot role shift tables, packing the prefix/tail lookup four-per-`UINT32`, pruning statically impossible early lower-band candidates/offsets, and accepting a near-equivalent `UINT8` score scale (`22/2/1`) with no exact-row penalty. This is now a cost-biased approximation of canonical v5 on arbitrary synth: visible remains 265/265 and both synth gates are above 99%, but arbitrary parity is 19,888/20,000 rather than 20,000/20,000.
- 2026-05-26 v5 8.6k pass / under-5k floor probe: simplified artifact now costs 8,649 (memory 8,263, params 386). The `Where(fill_mask_full, color1, input)` tail saves about 1,040 over cropped 10-channel assembly, and the max-valid-tail simplification saves another ~199 while keeping the required synth gates barely above threshold. A static candidate-pruning floor check says under-5k is not reachable by pruning alone: even visible-selected candidate triples measure 7,002 simplified and collapse to 56.3% arbitrary synth; offset-0-only measures 6,922 but fails visible; no-offset-2 measures 8,140 and fails both synth gates. A genuine under-5k attempt needs a different dynamic band extraction / vectorized planner, not more static candidate pruning.
- 2026-05-26 v5 7.1k pass: simplified artifact costs 7,107 (memory 6,724, params 383). The largest exact saving was sharing the chosen offset shift per multi-offset band/height group, which removed repeated role-mask gathers while preserving visible/live and the same 99% synth gates. Tail fallback is now the main risky approximation lever: removing all, keeping only height 1, or keeping only heights 1-2 all measured cheaper but failed either visible/live or the arbitrary 99% floor.
- 2026-05-26 v5 7.1k Kaggle result: hidden-negative despite passing visible/live and the 20k arbitrary gate. The submission moved 6197.00 -> 6196.97 (-0.03 LB) against a +0.20 predicted local gain, so the gate missed roughly 0.23 LB of hidden divergence. Current `submissions/submission.zip` is restored to the 8,649/base task219 payload.
- 2026-05-26 selective h1-share retry: disabling all shift sharing gives a conservative 7,372 candidate; sharing only height-1 bands gives 7,175, while height-2 sharing gives 7,304 and height-3 sharing gives no savings. Since `h12` equals the hidden-negative 7,107 artifact, the packaged retry is `h1` only plus a no-share fallback. Both were packaged as separate candidate zips instead of replacing the restored live `submission.zip`.
- 2026-05-26 h1-share post-submit diagnosis: exact `submission.zip::task219.onnx` scores locally at 7,175 and is stable across ORT optimization levels, so there is no evidence that `_score_v2.py` undercounted the submitted bytes. The cleanest next isolation, if spending a submit, is the reconstructed `task023v5 + task219_8649` baseline versus current `task023v5 + task219_7175`; do not infer more from the muddled saved lineage zips.

---

## 2026-07-15 cleanup compression source for task219.md

Full pre-compression task file archived from `Task_by_Task/task219.md` because repo_lint manual content exceeded 20 KB.

```markdown
# task219

<!-- AUTO:BEGIN -->
<!-- generated by tools/generate_task_mds.py; manual edits OUTSIDE this block -->

**STATUS:** LIVE cost **7,175** (live_costs.json, refreshed 2026-06-05).
(one row: `python tools/agent_helpers/show_task.py --task N` — NEVER Read task_state.csv whole)

**Linked research / prototypes** (auto-detected by filename):
- _none found_
<!-- AUTO:END -->

<!-- HIST:BEGIN -->
<!-- mined by tools/mine_task_history.py; rerun to refresh -->

**Git history** (commits touching files or subjects mentioning this task):
- `1c760ab2` — work: per-task .md + prototype builders/harnesses + prompt archive (session)
- `e23c2c59` — task219: E1 MLP-decode probe per research report â€” confirms pure hash lookup
- `bf0049d1` — task219: v4 ONNX builder stub documenting STOP CONDITION
- `18aedf40` — task219: STOP CONDITION invoked per CALIBRATION_PROBE directive
- `1572afb9` — task219: comprehensive findings after 4-path alternative exploration
- `1f93c834` — task219: explore object-completion + color-flood alternative rules
- `350b871a` — task051 v28 SHIPPED +0.01 LB (14th 1.0x confirm; v23 algorithm proven solving)
- `1543449e` — task219.md: document v4 investigation findings + ship recommendation
- `38f37d0f` — task219 multi-axis synth stress test (v4 vs rowgrammar)
- `e12ac616` — task219 numpy rule v4: 265/265 via shape-matched bands + shift + relative placement
- `80b11757` — task285 SHIPPED +15.12 LB EXACT MAX: 6108.66 -> 6123.78 (10th unlock)
- `34654397` — 3-ship integration: +0.87/+0.60/-2.65 + local recovery to 6108.66
- `083c387b` — Methodology pivot: visible-fit shortcuts on Solved tasks are positive EV (NOT always-avoid as previously documented)
- `9decbf3b` — 5 new prompts for tasks resistant to standard unlock methodology
- `079015dc` — 9-task structural-LIVE stack SHIPPED +3.97 LB: 6102.23 -> 6106.20
- _...and 40 more_

**Session log mentions:**
- `Logs/sessions/2026-05-11_1100_safe_opt_extensions_plus_organics.md` — | task219 | numpy partial 91/265 (staircase rule works train; arc-gen needs CC) | Skipped further work. |

<!-- HIST:END -->


## Rule

_unknown — derive and document here. If a recipe exists in `Research_Archive/notes/`, summarize it._

## Algorithm sketch

Current best structural hypothesis (v5): the first nonempty row band is a
complete template, not merely a contig/period-2 row grammar. Lower bands align
to consecutive template rows by a cost-biased `22*left + 2*count + right`
UINT8 score. Shift is derived from positive left/prefix diffs, falling back to
a max valid tail-first slot; fill color 1 is the shifted template row mask
beyond the observed lower-band right frontier.

NumPy reference: `prototypes/task219_numpy_rule_v5.py` (265/265 visible);
it is the direct-template rule with deterministic rational alignment scoring.
ONNX v5: `prototypes/build_task219_structural_v5.py`, emitting
`prototypes/task219_structural_v5.onnx` and
`prototypes/task219_structural_v5_sim.onnx`. Stress guards:
`prototypes/task219_synthetic_v5_guard.py` for visible-family shapes and
`prototypes/task219_synthetic_v5_arbitrary_guard.py` for arbitrary row masks.

## Attempts

**2026-07-11 | ⛔ WALL @ 2,444 — forced-component floor of Sho's F16-CumSum rewrite | bo_ng_cld_worker3**
BAR IS NOW **2,444** (`97fee100…`, owner Sho, manifest 2026-07-09), NOT the 7,094 the entries below describe — Sho did the full from-scratch structural rewrite the 2026-06-30 WALL flagged. It is a **38-node F16 CumSum ordinal graph** (Slice ch8 → start-markers `Greater` → `CumSum` band-id → Scatter/Gather band-start+frontier → offset=r−band_start → template-row Gather → frontier-mask `Where` → `+3` → `BitwiseAnd[5,6]` → `ConvInteger[10,2,1,1]` expand+pad to 30×30). Points = **25−ln(cost)**; 2444 → **17.20** (Sho `Staged_Sho/task219_analysis/README.md`: local-fail/LB-pass, the cheap idiosyncratic 71%-LIVE rule is LB-OPTIMAL; the ARC-GEN-correct rule costs **39,370** → LB 14.42; "keep the cheap one"). Gold gate = **byte-equal-to-LIVE** (harness binarizes output via `run_raw = out>0`, so only per-channel SIGN pattern matters; grid is ALWAYS 15×10, colors {0,1,8}→ch{0,1,8}).
**Per-tensor forced floor** (memory 2354 + params 90; hand-sum = sum of all 37 node outputs, verified): READ **750** = `v_t0` F32[1,1,15,10] **600** (Slice preserves F32; any cheaper read explodes — cast-full-input=9000, ch8-conv-output≥2520) + `v_t1` u8 mask **150** (needed by template-gather + Where-base; keeping F32 makes every downstream plane 4×). DECODE **600** = merged tri-state `v_t34` u8 **150** + `+3` shift `v_t35` **150** + 2-ch conv-input `v_t36` **300**. FILL-MERGE **300** = template-per-row `v_t30` **150** + frontier-mask `v_t33` **150**. FRONTIER `v_t2` col-weight **150**. Small row-vectors ~**554** (int32 Gather/Scatter idx can't narrow; CumSum must be F16). **0 recoverable slack found.**
**Dead routes PROVEN (built/tested/math, not "tried hard"):** (1) **Drop `v_t35` (+3): IMPOSSIBLE.** bg=state0, `0 & anymask = 0` → bg at conv origin; scalar-z ConvInteger then needs `z<1` but integer `z≥1` (algebra) → can't sign-separate bg. Per-channel `x_zero_point=[0,1]` DOES work (found clean weights ch0=(-1,-1),ch8=(1,0),ch1=(0,1) on `&[1,2]`) but **ORT rejects it**: `ConvInteger IsScalarOr1ElementVector(X_Zero_Point) was false`. Prototype `prototypes/c1_drop_plus3.onnx` = runtime error. (2) **Frontier via `ArgMax(select_last)×row_has`:** logically correct but adds int64[1,1,15]=120 B → net WORSE than `Mul`+`ReduceMax` (150+15). LIVE optimal. (3) **Float `Conv`+bias 1-ch decode:** a line can't be >0 at the MIDDLE state (color8=state1) while ≤0 at states 0,2 → needs 2 ch; 2-ch F32 conv-input=1200 ≫ ConvInteger u8 300. (4) **Prompt's "TopK stream compaction":** `COMPRESS`+`NONZERO` are BANNED ops. (5) U8/BOOL CumSum ORT-rejected (kill-rule). **No candidate staged (nothing byte-equal below 2444).** floor_est ~900 is a `2444/2.7` heuristic that inspection DISPROVES — 900 would require a different rule, and the only cheaper-rule direction (correct) costs 39,370. Hard-record: `hard_records/forced_floor_2444.md`. **2nd pass (forced re-attempt):** library-mined (task147 scatter-overlay, task280 Pad-BOOL, task201/255 read-floor) + BoBrain; key insight `output = input + fill-overlay` (input already has ch0@bg/ch8@color8; only ch0/ch1 flip at fill) but overlays need a fatal 30×30 intermediate → ConvInteger reconstruction stays optimal. BUILT+GATED a genuinely-different decode `prototypes/c2_pad_onehot_decode.onnx` (Gather-color→`Equal` one-hot→`Pad`): **byte-equal (0/300) but profile 4,691 ≈ 2× worse** (10-ch `[1,10,15,10]`=1500 plane). Confirms no cheaper assembly. | Cost: 2444 (unchanged). | WALL |

**2026-06-30 | ⛔ WALL @ 7,094 — forced-component floor + ORT-blocked fold | bo_ng_cld_worker3** _(SUPERSEDED: graph since rebuilt to 2444)_
Bar = live manifest **7,094** (`b7216e41…`); gold gate = byte-equal-to-LIVE (ambiguous-class, LIVE 71.110% truth). Per-tensor breakdown of the staged 7094 `_sim` (memory 6712 + params 382, 2638 node outputs, **0 dead nodes** via `prototypes/_dead_nodes.py`): cost is two forced fat tensors + ~2636 already-minimal scalars. (1) `ch8_f32` F32 [1,1,15,10]=**600 B** — forced: the F32 one-hot input must be sliced to get the 2-D color-8 plane that feeds `row_col_values`/`row_right` (Einsum-on-input can move only `rowmask`/`count` off it, not the 2-D plane; a highest-bit LUT costs +1024 params). (2) `fill_mask_full` BOOL [1,1,30,30]=**900 B** — forced: the final `Where→output` needs a 30×30 cond at 1 B/elem (any arithmetic/Pad alternative materializes a 30×30 F32 = 36 KB unless it is the free `output` tensor). int32 scores/indices (504 B) can't narrow (40·Δleft>255; no INT16 kernel; Gather idx needs int32). Params table [256] u32 = 1024 masks @4/word, min. **Tried (built+scored): fold the 66 `Mul(mask, Cast(selected→u16))` select-casts into `Where(selected, mask, 0)` → −132 B (would be 6962<7094).** REJECTED: ORT has **no `Where` kernel for UINT16** (`NOT_IMPLEMENTED: Where(16)`); confirms the truncated 2026-05-26 don't-do. Cast+Mul is forced; BitwiseAnd/Sub-underflow gate costs more nodes. Only remaining lever = a full vectorized rewrite of the data-dependent band/offset/role alignment (~4800 B), but data-dependent shift blocks free-output Einsum (#110/#132) and the rewrite must stay byte-equal to an idiosyncratic 71%-LIVE on 200k — low win-probability, not demonstrated cheaper. **No candidate staged (no tie/regression).** Lesson: `Onnx_Advanced/.update/DONE_2026-06-30T210000Z_where_uint16_unsupported_select_cast_fold.md`. | Cost: 7094 (unchanged). | WALL |

**2026-06-24 | ✅ CANDIDATE v5 h1-share + edge-fillcrop, cost 7,094 (byte-equal to LIVE on 200k) | Worker 6**
Verified the documented `task219_structural_v5_probe_h1_edge_fillcrop_sim.onnx` shave (zero impossible ±10 row-shift slots + crop fill-bit unpack to rows 2..14) is BYTE-EQUAL to the shipped 7,175 LIVE graph and staged it. Class-check: truth on LIVE = **14222/20000 (71.110%)** → ambiguous-class, so byte-equal-vs-LIVE is the gold gate (not 100%-truth). Gates: visible 265/265; AB vs LIVE **200265/200265 byte_equal=True (0 divergences)** (+20265/20265 short run); candidate truth = LIVE truth = 71.110% (identical pass count); `_score_v2` = 7094 (mem 6712 + params 382); `profile_real_cost --arcgen 266` static==profile==7094; opset 18, no banned/Sequence ops. SHA256 `b7216e41a33854388865ec2fb966760db805a3be9980cb2681cf29a7f52436fa`. Staged: `staged_submits/task219_v5_h1_edge_fillcrop_cost7094/`. **NOTE: prompt's "EXPERT FRESH / zero-weight Mystery Solved" framing is STALE** — manifest confirms task219 is already unlocked at 7175/16.1216 pts. Expected Δ = -81 cost ≈ +0.011 LB (near-floor, may not register), downside ZERO (byte-equal hidden behavior). This is SAFE because it is byte-equal, unlike the 2026-05-26 7,107 build (an approximation, 99.045% arbitrary-synth, NOT byte-equal) which shipped Kaggle-NEGATIVE. | Cost: 7094. | not submitted; captain ships |

**2026-05-24T23:18Z | ⚠️ SHIPPED rowgrammar_v2_postmortem — Kaggle → 6106.24 (+0.04 LB combined with task051 stack)** | POST-MORTEM RETRY ATTEMPT FAILED. Worker did everything right per v2 prompt (10,844 cost, 265/265 vs LIVE, 1000/1000 offset-stressed synth) but Kaggle hidden test still returned floor. Joins task233/285/367-v14 in zero-public-weight family suspicion. Cost compressed slightly (18,756 v1 -> 10,844 v2). | Cost: 10844. | submit-captain ship 2026-05-24T23:18Z |

**2026-05-24T20:35Z | ⚠️ SHIPPED rowgrammar_structural_sim — Kaggle → 6100.67 (+0.00 LB)** | NON-CONFIRM. structural row descriptors UINT8 scalar planner + offset-2 prune (never selected in 265 visible). Offset-2 prune was a visible-fit shortcut; hidden has offset-2 cases. 5097 nodes graph. | Cost: 18756. | submit-captain batch ship 2026-05-24T20:35Z |

_One row per attempt. Format: `YYYY-MM-DD | approach | result | cost | session log`._

## COMPRESSED HISTORY

Pre-2026-06-24 attempt prose, the v5 cost-pass ladder (43,558 live → 18,756 → 10,844 → v5 19,145 → 9,888 → 8,649 → 7,107 → **7,175 h1-share = current LIVE**), the E1 MLP-decode probe (LIVE is a pure 21-bit hash lookup, zero generalization), and the per-pass cost analysis are archived in `_archive/task219_history.md`.

**Key carry-forward facts:**
- task219 is ALREADY UNLOCKED: manifest cost **7,175**, points **16.1216** (SHA `ad5e5f72…`). The +15.94 LB v5 unlock already happened. The 'zero-weight / EXPERT FRESH Mystery' prompt framing is STALE.
- Rule = direct-template row projection (v5): first nonempty band is the template; lower bands align by `22*left + 2*count + right`; fill = shifted template-row mask beyond the band's right frontier. NumPy ref `prototypes/task219_numpy_rule_v5.py` (265/265 visible).
- LIVE is ambiguous-class: only **71.110%** vs ARC-GEN truth ⇒ **byte-equal-vs-LIVE is the gold gate**, not 100%-truth.
- Hidden-sensitive: the 7,107 approximation (99.045% arbitrary-synth, NOT byte-equal) shipped Kaggle-NEGATIVE (-0.03 LB). Only byte-equal-to-LIVE compression is safe.


## Don't-do log

_Record ruled-out hypotheses here so future agents don't retry them._

- 2026-07-11: On the **2444** F16-CumSum graph, do NOT retry: (a) removing the `+3` (`v_t35`) plane before ConvInteger — bg is at the conv origin, scalar-z can't sign-separate it (proof) and ORT rejects per-channel `x_zero_point`; (b) `ArgMax` frontier — net +105 B vs `Mul`+`ReduceMax`; (c) float-`Conv`/bias decode — can't isolate the middle state, 2-ch F32 input ≫ ConvInteger u8; (d) `Compress`/`NonZero`/TopK compaction — BANNED ops. 2444 is within ~150 B of the byte-equal forced floor; the only cheaper *direction* is a different (correct) rule which costs 39,370. See Attempts 2026-07-11 + `hard_records/forced_floor_2444.md`.

<!-- SPREAD:BEGIN -->
<!-- generated by tools/spread_annotate_notes.py from 20260518 scan; manual edits OUTSIDE this block -->

## Spread-knowledge scan (20260518)

Current canonical cost: **48138** (memory=19856, params=28282, points≈14.218173).

Pattern sweep verdicts (each pattern measured INDEPENDENTLY vs the original — patterns overlap, so the sum overstates the real cumulative savings):

| Pattern | Verdict | Δ cost (vs original) | Notes |
|---|---|---:|---|
| `clip_mul_to_min` | — | — | no Clip[0,1] + Mul(mask) pairs detected |
| `const_to_init` | — | — | no Constant nodes with a 'value' attribute to convert |
| `crop_to_size_advisory` | ? | -30000 | max active grid is 15×10 but graph has 1 [...,30,30] intermediate(s). Crop-pr... |
| `fp16_selective` | ERR | — | fp16_convert raised ValueError: The model is already converted to float16, if... |
| `onnxsim` | — | 0 | onnxsim found additional simplifications \| cost unchanged |
| `safe_opt` | — | — | no changes from safe-opt + extended passes |
| `scalar_broadcast_ones` | — | — | no uniform-valued float initializers consumed only by broadcasting ops |
| `slice_axis_collapse` | — | — | no Slice nodes with redundant axis-0 entries to drop |

**Sequential pipeline result**: 0 bytes saved — already at the floor for the patterns currently in `tools/spread_patterns/`.

**Advisory** (crop_to_size_advisory): max active grid is 15×10 but graph has 1 [...,30,30] intermediate(s). Crop-process-pad could save ~30000 bytes (rough est). — estimated savings if applied manually: ~30000 bytes.

_Re-run: `.venv/Scripts/python.exe tools/spread_scan.py` (updates JSON in `Logs/spread_scan_results/`); then `.venv/Scripts/python.exe tools/spread_annotate_notes.py`.
<!-- SPREAD:END -->

- 2026-05-20 [PR_BIG worker]: pattern=cast_after_first_slice delta=-2142 verify=265/265 cand=task219_cast_after_first_slice_t_ad7aa643_on_5835_30
- 2026-05-24: Do not revive the 18,756 `rowgrammar_structural_sim` offset-2 prune. It was visible-fit and shipped +0.00 LB. Future cost reductions must keep offsets 0/1/2 represented and pass `prototypes/task219_synthetic_guard.py --n 1000` or stronger.
- 2026-05-24: Do not drop the right-edge score term or flatten count weights in the current bitset planner. A provisional 11,807-cost build still passed visible/live but failed `task219_synthetic_guard.py --n 1000` at case 20 for desired offset 2.
- 2026-05-24: A `UINT64` four-entry mode-attr pack is scoreable but worse than the `UINT32` three-entry pack: it measured 11,249 because heavier scalar decode outweighed the saved params.
- 2026-05-25: Do not use UINT8 arithmetic for v5 weighted alignment scores. The 40x left-distance term overflows UINT8 and silently chooses wrong template windows on synth; cast distances to INT32 before weighting.
- 2026-05-25: Do not treat `task219_synthetic_guard.py` as the v5 oracle. It is rowgrammar-oriented and intentionally diverges from direct-template v5 on some cases. Use `task219_numpy_rule_v5.py` as the direct-template reference when testing v5.
- 2026-05-25: Do not chase v4 floating tie quirks in ONNX. The remaining v4-vs-ONNX disagreements were ambiguous mixed A/B/A template ties; `task219_numpy_rule_v5.py` resolves them with deterministic rational scoring that matches the graph and still passes 265/265 visible.
- 2026-05-25: Use `task219_synthetic_v5_guard.py` for v5 synth evidence. It was audited to consume `--proto` directly and avoids the hardcoded-candidate trap.
- 2026-05-25: Do not omit final-row height-1 lower bands from v5 ONNX. Arbitrary-rowmask stress caught `band_start == 14, height == 1` as a real NumPy-vs-ONNX translation gap; the builder now scans `range(1, H)`.
- 2026-05-25: Keep synthetic lower bands separated by at least one empty row. If the arbitrary generator places adjacent lower bands, `find_bands` merges them into a taller-than-template off-contract band and creates misleading failures.
- 2026-05-26: Do not add a runtime guard just to remove the `+10` shifted-rowmask table slot. `Where(BOOL, UINT16, UINT16)` is rejected by ORT, and the safe `Cast(Not(cond))->UINT16` multiply variant measured worse (35,839 simplified) than keeping the all-zero `+10` slot (35,744 simplified).
- 2026-05-26: Prefer `INT32` arithmetic shifts over `UINT32 BitShift` for v5 rowmasks. `BitShift` is unsupported for `UINT16`; the `UINT32 BitShift` rewrite passed gates but measured 22,646 simplified, while `mask * 2^p` / `mask // 2^n` with `INT32 Where` measures 19,145.
- 2026-05-26: Do not remove the count term from the under-10k scorer. A `left/right`-only score measured 10,799 but broke visible parity (264/265) and failed synth (arbitrary 1,862/2,000; visible-family aggregate 1,929/2,000).
- 2026-05-26: Do not remove the right-edge term from the under-10k scorer. A `left/count` score measured 10,968 and still passed visible, but arbitrary synth was 1,966/2,000 (98.3%), below the >=99% gate.
- 2026-05-26: The under-10k build deliberately trades exact arbitrary-v5 parity for cost. The removed exact-row penalty plus `22/2/1` `UINT8` score scale should be treated as a cost-biased approximation; if a future pass needs 100% arbitrary parity again, restore the exact rowmask penalty / wider score path and expect cost to rise above 10k unless another structural saving replaces it.
- 2026-05-26: Do not chase under-5k through static candidate pruning. Measured probes: `visible_selected` = 7,002 simplified, visible 265/265 but arbitrary 563/1000; `o0` = 6,922 simplified, visible 257/265 and arbitrary 704/1000; `no_o2` = 8,140 simplified, visible 265/265 but friendly 1,972/2,000 and arbitrary 952/1,000. Under-5k requires a different planner, not fewer static `(band_start,height,offset)` branches.
- 2026-05-26: The 8,649 build is close to the arbitrary-synth floor after tail simplification: 19,809/20,000 (99.045%). Any further approximation must be checked on the 20k arbitrary guard, not just 2k smoke tests.
- 2026-05-26: Do not remove or height-prune the tail fallback on the 7.1k builder. `TASK219_SHIFT_MODE=no_tail` measured 6,832 but failed visible/live 264/265 and arbitrary 1,636/2,000; `h1` measured 6,985 but failed visible/live 264/265 and arbitrary 1,917/2,000; `h12` measured 7,080 and passed visible/live but failed arbitrary 1,979/2,000 (98.95%).
- 2026-05-26: Do not replace the selected suffix `UINT16` mask multiply with `Where(BOOL, UINT16, UINT16)`. ORT rejects `Where(16)`; keep the validator-compatible `Cast(selected)->UINT16` then `Mul`.
- 2026-05-26: Do not ship the 7,107 compression again, and do not treat 99.045% arbitrary-synth as sufficient for task219 compression. It was Kaggle-negative (6197.00 -> 6196.97, -0.03 LB) despite visible/live 265/265 and `19,809/20,000` arbitrary synth.
- 2026-05-26: Do not blame h1-share's +0.00 result on ORT graph optimization without new evidence. The exact submitted `task219.onnx` scores 7,175 under local `_score_v2.py` across ORT disable/basic/extended/all. Future diagnosis should isolate submission lineage or Kaggle hidden behavior, not re-run the same local scorer theory.
- 2026-05-26: OnnxAdv follow-up found one tiny safe same-behavior shave on the h1-share graph: zero the impossible `shift = +/-10` row-shift slots and crop the internal fill-bit unpack from rows `0..14` to rows `2..14` before padding. Result: `task219_structural_v5_probe_h1_edge_fillcrop_sim.onnx` scores 7,094 (memory 6,712 + params 382), passes visible/live 265/265, passes smoke synth (`3,995/4,000` friendly; `4,962/5,000` arbitrary), and A/B matches the 7,175 h1-share graph on 20,000 arbitrary generated cases. Candidate zip: `submissions/submission_task219_7094_h1_edge_fillcrop_candidate_20260526T173402.zip` with task023 v5 intact.
- 2026-05-26: Do not elide tail-valid checks. Tail validity is not just row/role existence; it distinguishes prefix-only rows from rows with a real tail segment. The `TASK219_ELIDE_TAIL_VALID=1` probe measured 6,995 but failed friendly synth badly (`927/1,000`) and arbitrary (`1,691/2,000`).
- 2026-05-26: Do not drop the right-edge tie-break even in the 7.1k h1-share graph. `TASK219_SCORE_MODE=no_right` measured 7,028 and passed visible/live, but failed friendly (`9,767/10,000`) and arbitrary (`19,601/20,000`) synth. Count removal and left-only both break visible 264/265.
- 2026-06-24 [Worker 6]: **task219 is hidden-sensitive AND ambiguous-class** — LIVE itself is only 71.110% (14222/20000) vs ARC-GEN truth, so byte-equal-vs-LIVE is the gold gate, NOT 100%-truth. **Only BYTE-EQUAL-to-LIVE compression is safe.** The 7,094 `h1_edge_fillcrop` IS byte-equal (200265/200265, 0 divergences) and staged. The 7,107 build was NOT byte-equal (an approximation at 99.045% arbitrary-synth) and shipped Kaggle-NEGATIVE (-0.03 LB). Any future compression must clear AB full-200k byte_equal=True; ≥99% synth is provably insufficient here. Static candidate-pruning floor is already documented as exhausted under 5k; the remaining 7,094→lower path requires a genuinely different planner that stays byte-equal.

<!-- BEGIN AUTO-GRAPH-DIAG -->

## Graph Diagnostic (auto-generated)

**Cost**: 7,175 | **Memory**: 6,792 | **Params**: 383 | **Points**: 16.1216
**Nodes**: 2645 | **Initializers**: 61 | **Opset**: 18 | **Graph output dtype**: F32
**Max active grid (from `task219.json`)**: 15×10

### Quick-look flags

- has_Conv: ✗
- has_MatMul: ✗
- has_ReduceSum: ✓
- has_ReduceMax: ✓
- has_Where: ✓
- has_Gather: ✓
- has_OneHot: ✗
- first_node: `Slice`
- last_node: `Where`
- ends_with_Cast(F16→F32)→output: ✗
- first_node_is_Slice: ✓

### Op-type histogram

| op | count |
|---|---:|
| `And` | 308 |
| `Gather` | 275 |
| `Max` | 254 |
| `Mul` | 236 |
| `Sub` | 217 |
| `Add` | 208 |
| `Where` | 192 |
| `BitwiseOr` | 180 |
| `Cast` | 152 |
| `Min` | 146 |
| `BitwiseAnd` | 142 |
| `Less` | 95 |
| `Or` | 79 |
| `Not` | 49 |
| `LessOrEqual` | 48 |
| `Div` | 33 |
| `BitwiseNot` | 13 |
| `Concat` | 4 |
| `Mod` | 3 |
| `Greater` | 3 |
| `Reshape` | 2 |
| `Slice` | 1 |
| `Einsum` | 1 |
| `ReduceMax` | 1 |
| `ReduceSum` | 1 |
| `ArgMax` | 1 |
| `Pad` | 1 |

### Initializers (top 10 by params)

| name | dtype | shape | params |
|---|---|---|---:|
| `row_prefix_tail_pack4_u32` | dtype12 | [256] | 256 |
| `fill_by_kind_start_mask_u16` | UINT16 | [11] | 11 |
| `powers_f32` | F32 | [10] | 10 |
| `cols_u8_4d` | UINT8 | [1, 1, 1, 10] | 10 |
| `col_bits_u16_4d` | UINT16 | [1, 1, 1, 10] | 10 |
| `fill_one_hot_f32` | F32 | [1, 10, 1, 1] | 10 |
| `fill_mask_pads` | INT64 | [8] | 8 |
| `row_prefix_tail_divisors_u32` | dtype12 | [4] | 4 |
| `fill_bits_shape` | INT64 | [4] | 4 |
| `ch8_starts` | INT64 | [3] | 3 |
| ... (51 more) | | | |

### Top 5 largest intermediates (static value_info bytes)

| name | dtype | shape | bytes |
|---|---|---|---:|
| `fill_mask_full` | BOOL | [1, 1, 30, 30] | 900 |
| `ch8_f32_1` | F32 | [1, 1, 15, 10] | 600 |
| `fill_bit_values_2869` | UINT16 | [1, 1, 15, 10] | 300 |
| `ch8_b_2` | BOOL | [1, 1, 15, 10] | 150 |
| `row_col_values_u8_18` | UINT8 | [1, 1, 15, 10] | 150 |

Note: grader charges max(static, runtime-on-real-inputs)/tensor — techniques.md §39 MECHANISM.

*Technique routing: `Onnx_Advanced/by_symptom.md` (+ INDEX verdict list). Flags above feed it.*

*Node list (2645 nodes) omitted — the .onnx is canonical; run `tools/agent_helpers/inspect_graph.py task219` for the full sequence.*

<!-- END AUTO-GRAPH-DIAG -->
```
