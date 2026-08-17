# Free-output `>0`-decode graphs are LOCKED to integer weights — non-integer refit/rank-reduction is dead; + einsum reflection-basis floor calibration

**Where:** task117 (4-fold reflection around a detected X-center). Live best = **2380**, Sho's single free-output `Einsum` (`Team_Best_Onnx/task117.onnx`, graph `bianfu3`). Full write-up: `The_400_Tasks/Task_by_Task/task117/hard_records/einsum_frame_and_rank_floor.md`. Repro scripts referenced there.

---

## Insight 1 (MOST transferable): the `out>0` per-channel decode is EXACT-AT-0, so any free-output vote-accumulation graph is locked to INTEGER weights. Continuous refit / rank-reduction / quantization is a dead end even when it is *mathematically exact*.

The official decode (`ng_harness/session.run_raw` = `neurogolf_utils.run_network`) is `(output > 0)` per channel, compared exactly to the one-hot target over the full 30×30 (off-grid = all-zero). Free-output "vote" graphs (Einsum/MatMul that accumulate signed votes and let the sign decide the class) work ONLY because the builder used **integer** weights (0/1/2, ±1000 penalties) so every OFF cell lands at **exactly ≤ 0**.

Measured on task117: I took Sho's `v_w2` reflection basis and applied a **mathematically-exact basis rotation** (`v_w2' = O·v_w2`, `v_w3' = v_w3·O⁻¹`, so the effective operator `M = v_w3'·v_w2'` is preserved to 2e-7). Raw output matched Sho to **max diff 5e-4**. Yet under the `>0` decode it scored **0/265 in BOTH numpy AND ORT** — because ~thousands of exactly-0 off-cells got bumped to `+5e-4 > 0` and decoded "on". (I first misread this as ORT einsum fragility; it is NOT ORT-specific — numpy fails identically. It is the razor-sharp threshold.)

**Consequence for cost-golfing:** you cannot shrink a free-output vote graph's basis by SVD/low-rank/gradient-fit/quantize — the moment weights leave the exact-integer lattice, off-cells drift positive and the whole task fails. Only an *exact integer* re-expression survives, and you cannot span a rank-K integer operator family with < K integer modes. This makes free-output vote graphs' parameter tensors effectively **un-refittable** — treat their element count as a hard floor unless you find an exact integer restructure.

Searched `traps.md` + `techniques.md`: integer-weight *construction* is mentioned (techniques.md:1163, 3192) but the **refit-killer / floor implication of the exact-at-0 threshold** is not stated anywhere. This is the missing piece.

## Sorter note: traps.md — add as new trap "freeoutput_gt0_exact_at_0_integer_lock": `>0` decode is exact-at-0 ⇒ free-output vote graphs are locked to integer weights; any non-integer refit (SVD/rank-reduce/quantize/gradient) drifts off-cells positive → 0/N even for a raw-diff-5e-4 exact rotation (measured task117). Corollary: a free-output vote tensor's element count is an un-refittable floor absent an exact-integer restructure.

---

## Insight 2: free-output Einsum vote-accumulation BEAT the paint-over family on a reflection task — recalibrates an old "einsum dominated" floor note.

task117's own `hard_records/paint_over_floor.md` (2026-07-08) declared a WALL at 3795 for the paint-over (Slice/Gather/Where) family and **dismissed free-output Einsum** as "center-dependent reflect basis ≥ [5,15,30] = 2250 params, dominated". That estimate was wrong: Sho reached **2380** with a single degree-10 free-output Einsum by (a) cropping input 30→15 with a shared `eye(15,30)` resampler, (b) a **shared** `[6,15,15]` reflection basis for BOTH row and col (`gau`,`jcv`), = identity mode + 5 anti-diagonal reflection modes (centers cy∈{5..9} ⇒ a+u∈{10,12,14,16,18}), rank-6 exact. So a 2-D center-dependent reflection is NOT forced to a `[K,15,30]` basis — downsample-to-active-frame + a shared separable anti-diagonal basis is ~1.4k, not 2.25k+. General lesson: before declaring a paint-over/Gather floor, price the free-output Einsum with (downsample resampler + shared separable geometric basis), not the naive full-frame basis.

## Sorter note: techniques.md — extend the free-output-Einsum section with "shared separable anti-diagonal reflection basis + 30→active-frame crop resampler" as the cheap route for data-dependent 2-D reflection/symmetry tasks (exemplar: task117 `bianfu3`, 2380, beat the 3795 paint-over wall).

---

## Insight 3 (dead-end): under the `>0` decode, a shared crop resampler ties EVERY frame index to one size — you cannot partial-crop the frame even when content is smaller.

task117 content (nonzero cells) never exceeds row/col 13, so cropping the 15-frame to 14 *looks* free and is argmax-identical. But `>0` requires the **background outer ring** of size-15 grids (row/col 14, color 0) to decode ch0>0, supplied by the identity component reading+placing input row/col 14 — needs the 15th frame index. And because `v_w0` is a single shared crop feeding input-downsample, output-upsample, AND the aux center-detect crops, every frame letter in the equation is the same size — you can't shrink the reflection frame while keeping the background frame. Net: frame is monolithic; `-214` frame-crop scored visible **212/265** (fails exactly the 53 size-15 grids). Decoupling into two resamplers costs +420 to save 174 = worse.

Measured the 2-node escape concretely (forced 2nd pass): `Einsum(14-frame reflect)→refl` + `Add(refl, input·ch0_mask)→out` is visible **265/265** but profiles **74176** (memory 72000 + params 2176) — the two inter-node `[1,10,30,30]` F32 tensors are charged 36000 B each, obliterating the −214 param save (net +71796). Concrete instance of task163 Tool-4 / task262 (single zero-intermediate Einsum beats any split). **Extra gotcha:** you cannot restore the background with `Max(refl, input_ch0)` — reflected-fg cells were background in the INPUT (input_ch0=1), so Max undoes the −1000 penalty → multi-hot → 0/265; must use `Add` (penalty survives the sum). Background and the fg-overwrite penalty are entangled; a max/OR split is semantically wrong, not just costly.

## Sorter note: traps.md — extend the frame-crop/active-grid trap: under `>0` decode the background/empty class must be produced too, so an all-cells output (30×30) whose empty ring depends on grid size forces the FULL frame; a shared crop resampler makes the frame monolithic (can't shrink reflection frame independently). 2-node escape measured 74176 vs 2380 (task117). Also: background restore after an overwrite must be Add (penalty-preserving), not Max/OR.
