# task117 hard record — free-output Einsum floor ~2380 (frame-crop + rank-reduction are DEAD)

- **Live bar when recorded:** **2380** (authoritative manifest, `ng_current_bar.py --task 117`, owner **Sho**, sha `79dcba6fb80f7b66ba3f4ed64b9c7d3e4e5358a02972745772b1c6fdc3a0f879`, updated 2026-07-09). This is a single **free-output Einsum** (`Team_Best_Onnx/task117.onnx`, graph `bianfu3`) and it **broke the old paint-over "3795 WALL"** — see `paint_over_floor.md`, whose dismissal of free-output Einsum ("≥2250 params, dominated") was wrong.
- **Recorded:** 2026-07-11 by `bo_ng_cld_worker3` (task117).
- **Prompt context:** prompt `Task117_F16CumSumOrdinal` is MISMATCHED (describes a CumSum ordinal task, `floor_est ~900`). task117 is actually 4-fold reflection around an X-center. The `~900` floor_est is a mis-estimate, same failure mode as the old `~1300`.

## Verdict: WALL for the levers explored. 2380 is at/near the structural floor of the free-output-Einsum family.

### What Sho's 2380 graph is (fully reverse-engineered)
ONE `Einsum` node, output `[1,10,30,30]` (FREE ⇒ memory 0), so **cost = sum of 9 initializer element-counts = 2380**. Degree-10 in the input (10 `input` copies), one-hot exploited (x^k=x). Two logical parts sharing indices `s,t` (row/col center hypotheses, each ∈ 5 values = centers cy,cx ∈ {5..9}):
- **Part A** (deg-9): computes a soft center selector from row/col sums.
- **Tail**: applies the selected reflection. `out = Σ_m channelmap(o,p,m)·rowop(m,s)·colop(m,t)·input`, m∈{0,1}.
  - **comp0** = `(I+R_cy)⊗(I+R_cx)` = the 4-fold reflection; `v_w4[:,:,0]` = fg identity(1..9) + ch0 `−1000·fg` penalty.
  - **comp1** = `I⊗I` (identity); `v_w4[:,:,1]` = `e0⊗ones` (total input mass → background ch0).

Weight roles / cost (element counts):
| w | shape | cnt | role |
|---|---|---|---|
| v_w2 | [6,15,15] | **1350** | reflection basis: **identity mode + 5 anti-diagonal reflections** (a+u ∈ {10,12,14,16,18}). rank-6 EXACT (SVD: 6 nonzero sv). SHARED row(`gau`)+col(`jcv`). |
| v_w0 | [15,30] | **450** | crop `eye(15,30)`: input 30→15 frame AND frame 15→30 output. |
| v_w4 | [10,10,2] | 200 | channel map (comp0 near-I + bg penalty; comp1 rank-1 e0⊗ones). |
| v_w1 | [5,30] | 150 | center-detect crop rows 5..9. |
| v_w7,v_w8 | [5,15]×2 | 150 | center-detect aux crops rows 4..8 / 6..10. |
| v_w3 | [2,5,6] | 60 | mode selector (component,center)→modes. |
| v_w5,v_w6 | [10]×2 | 20 | channel gates. |

### GRADING SEMANTICS (the trap that kills the obvious shave)
The official decode is **`out > 0` PER CHANNEL** (`ng_harness/session.run_raw` = `neurogolf_utils.run_network`), compared exactly to the one-hot expected over the **FULL 30×30** (`encode_grid`: off-grid r≥h/c≥w ⇒ all-channels-0). **NOT argmax.** So every within-grid background cell (color 0) must decode **ch0 > 0**, and every off-grid cell must have ALL channels ≤ 0.

### DEAD END 1 — frame-crop 15→14 (would be 2166, −214). INVALID: `ng_gate visible 212/265`.
Foreground content never exceeds row/col 13 (all 265), so cropping the frame dim in v_w0/v_w2/v_w7/v_w8 to 14 *looked* free and is **argmax-identical to Sho on all 265** (numpy AND ORT). But it fails EXACTLY the **53 size-15 grids** (265−53=212): their background **outer ring** (row 14 / col 14, color 0) must decode ch0>0, supplied by comp1 (total mass → ch0) reading+placing input row/col 14 — which needs the **15th frame index**. Cropping ⇒ ch0=0 there ⇒ decodes empty ⇒ mismatch. Because **v_w0 is a shared crop, every frame index ties to 15**; no partial frame-crop is possible. (Builder kept as a marked dead end: `prototypes/build_task117_einsum_frame14.py`.)

### DEAD END 2 — rank-reduce v_w2 (6→5 modes, would be ~2145, −235). INVALID: razor-sharp `>0` grading + integer weights.
The 10 effective reflection matrices `M[m,s]` are **rank-6 EXACT** (SVD sv = [12.9,3.8,3.6,3.5,3.3,2.3,0,0,0,0]) — 5 modes cannot represent them exactly. But the deeper, more general kill: **the `>0` decode is exact-at-0 and Sho's weights are INTEGER (0/1/2), so every off-cell lands at EXACTLY ≤0.** A basis rotation that is mathematically exact — raw output within **5e-4** of Sho — nonetheless scores **0/265 in BOTH numpy AND ORT**, because the ~5e-4 float residue bumps thousands of exactly-0 off-cells to `+5e-4 > 0` ⇒ decoded "on" ⇒ every grid fails. ⇒ **ANY non-integer reweighting of the reflection basis is dead** (rank-reduce, SVD refit, gradient fit, quantization). Only an *exact integer* basis survives, and you cannot span a rank-6 integer operator family with <6 integer modes. This also means v_w2's 6 modes cannot be shrunk by continuous optimization at all.

### DEAD END 3 — 2-node split (14-frame reflection Einsum + background restore). MEASURED cost 74176 (31× worse). [forced 2nd pass]
The one structural move that dodges the 15-frame floor: run the reflection in the cheaper 14-frame (−214 params) and restore the size-15 background ring with a separate term. Built `prototypes/build_task117_2node_frame14_bgmax.py`: `Einsum(14-frame)→refl` then `Add(refl, input·ch0_mask)→output`.
- **Semantic gotcha (worth knowing):** `Max(refl, input_ch0)` is WRONG (visible 0/265) — reflected-fg cells were *background in the input* (input_ch0=1 there), so Max restores ch0=1 and undoes the −1000 penalty ⇒ every cell decodes multi-hot. Must use **Add** so the penalty survives (`−999 + 1 = −998 < 0`). Background and penalty are entangled; you cannot split them with a max/OR. Add version: **visible 265/265 PASS**.
- **Cost (measured, `profile_real_cost --arcgen 266` + `ng_graph_cost_diff`):** `static=74176 profile=74176 (memory 72000 + params 2176)`. The two inter-node tensors `refl` and `bg`, each `[1,10,30,30]` F32 = **36000 B**, are charged (named by cost_diff: `+36,000 B added memory_tensor` ×2). Net vs live **+71796** — the −204 param saving is obliterated 350× over. This is the concrete instance of **task163 Tool-4** ("free channel-identity does not survive folding; a 2nd node makes the copy a charged `[1,10,30,30]` intermediate ≫ any saving") and **task262** ("prefer a single zero-intermediate Einsum at the rank floor over any Slice/Gather route"). Confirms: memory must stay 0 ⇒ single free-output Einsum ⇒ monolithic 15-frame.

### Why the floor is ~2380 (forced-component argument)
1. **Single free-output Einsum is mandatory.** Any 2-node split (Einsum→Add) materializes a `[1,10,≥14,≥14]` F32 intermediate = **≥7840 bytes memory**, dwarfing any ≤~370 param saving. So memory must stay 0 ⇒ everything in one Einsum ⇒ frame-ops must be v_w2 modes (can't route identity via a separate additive term).
2. **Reflection basis 1350 forced.** A data-dependent reflection over 5 centers, in pure Einsum (no Gather → no charged index tensor), needs the 5 anti-diagonal matrices (linearly independent, dense 5×15×15=1125) + identity (needed by comp0's `I+R` and comp1's background) = 6 modes × 15×15 = 1350. Rank-6 is exact-tight. Centering (to make reflection a fixed reversal) just moves the cost into per-center shift tensors ([5,13,30]≈1950) — strictly worse.
3. **15-frame forced** by the size-15 background outer ring (DEAD END 1), and the resampler is shared, so v_w2 can't be 14 while background stays 15 without a second resampler (+420 to save 174 = net worse).
4. **v_w0 [15,30]=450 forced** — must contract input's full 30-spatial down to the 15-frame; 15 forced by (3).
Core (v_w2+v_w0)=1800 forced; remaining 580 = center-detect + channel-map + selector.
5. **All 9 initializers are load-bearing** (ablation, task163 Tool-3): zeroing ANY one weight — including the 10-param gates `v_w5`/`v_w6` and the 75-param `v_w7`/`v_w8` — drops to **0/265**. No dead weight to remove; the graph is minimal. And no weight can be *shape*-reduced without breaking a shared einsum index (every "15" ties to the monolithic frame; every "5" to the 5 centers; "10" to channels; "2" to the two components). So there is no sub-2380 route by pruning or reshaping this graph, and (by 1–4) none by restructuring the family either.

### One remaining hypothesis (low confidence, HIGH risk)
Part A center-detection + channel-map + gates ≈ **500 params** (v_w1 150 + v_w7/v_w8 150 + v_w4 200) are unexplored. They are integer/sparse (so *maybe* ORT-safe to edit, unlike a rotated basis), but (a) Part A is an opaque deg-9 polynomial that must stay correct on the 200k-truth incl. the area-5 center ambiguity (see task md Don't-do 2026-06-12), and (b) any structural change risks silent ORT numeric breakage. Even a full 500→0 (impossible) is only +0.2 LB pts (points = 25−log(cost): 2380→17.22, 2166→17.32). **EV of further grinding is ~0.04–0.1 LB pts; recommend spending budget elsewhere and treating task117 as AT its floor.**
