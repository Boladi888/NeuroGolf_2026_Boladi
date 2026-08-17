# task364 — DEAD ROUTES: structural shaves of the live 12,319 flood-classifier

**Batch:** bo_ng_cld_worker2 (Claude lane) 2026-07-12
**Live bar at run time:** **12,319** (manifest, owner **Sho**, updated 2026-07-09T01:09:17Z),
sha256 `224d157edef9bbb1f98e83345ca1ed6a44610c97d0b18edbd0e58b45a3119808`,
`Team_Best_Onnx/task364.onnx`, graph `task364_qcode_h25_v25_p7`.
profile (worker3, `--arcgen 266`): static==profile==12,319 (fully STATIC).

This note complements `CarrierNarrowEqualTail_900_forced_floor.md` (which killed the prompt's
~900 carrier-narrow Direction). Here I audit whether the live graph can be **shaved below 12,319**
by any structural change, and refute 5 more independent routes with priced/empirical evidence.
**No gates were run** — every route falsifies at the numpy fast-probe / construction level, so a
`truth`/`ab`/`profile` gate would be pure burn (protocol: cheapest-first, kill pre-build).

## EXACT per-tensor census (measured 2026-07-12 from the live ONNX; STATIC ⇒ profile==static)

Sum below = **12,319** exactly (memory 12,214 + params 105). Every tensor is [1,1,20,22] U8/BOOL
= 440 B unless noted. Node order per `inspect_graph.py Team_Best_Onnx/task364.onnx`:

| stage | tensors | bytes |
|---|---|---:|
| F32 input read | `mask_f` F32[1,1,20,22] | 1,760 |
| masks | `mask_bool` + `mask_u8` | 880 |
| H/base detect | `nbr_center_code_u8` + `branch_bool` (`ng_has_hv` folds to turn) | 880 |
| corner re-encode | `ng_has_hv` + `turn_code_u8` | 880 |
| **U detect** | `pair2_score_u8` U8[1,2,20,22]=880 + `pair_score_max_u8` 440 | 1,320 |
| seed | `seed_label` | 440 |
| **flood** | `prop_pool_0..4` (5×440) + `prop_0..3` (4×440) | 4,400 |
| background sentinel | edge slices ~294 + `scalar0_row`/`task364_scalar0` 460 | 754 |
| Equal-tail canvas | `task364_scalar6` 440 + `task364_scalar30` U8[1,1,30,30] 900 | 1,340 |
| params | initializers | 105 |
| **total** | | **12,319** |

Rule realized by the graph (decoded from the initializers): `nbr_center_code` 3×3 kernel
`[[0,3,0],[2,6,2],[0,3,0]]` ⇒ shape-cell code = `6 + 2·h_nbr + 3·v_nbr`; **code 11 = elbow/corner**,
**code >12 = junction ⇒ H**. `pair2_k_q` [2,1,5,5] weights `[6,6,4,2,1]` along a H-line (ch0) and
V-line (ch1) on `turn_code` (corner→2, shape→1, bg→0); `ReduceMax` merges the two orientations →
U-seed. `seed_label`→5×masked-MaxPool flood→`scalar6`→Pad 30×30→`Equal(colors10)` free output.

## Routes refuted (5, independent — all pre-build, no gate burn)

**1. Bigger flood kernel, fewer rounds (target −1,760).** DEAD by construction.
Flood is 5×`MaxPool(3×3)`+mask = 10 tensors = 4,400. A 5×5 MaxPool would halve the rounds, BUT the
generator places sprites with `common.overlaps(brows,bcols,wides,talls, spacing=1)` ⇒ bbox gap ≥1 ⇒
**cells of two different sprites can be exactly Chebyshev-2 apart**. A 5×5 MaxPool (Chebyshev-2 reach)
bridges them → the flood label leaks across components → misclassification. Masking each round does
not help (the leaked cell is itself a shape cell). Only a 3×3 (Chebyshev-1) kernel is safe, and the
don't-do log already verified drop-1-round → 271/4000 mismatch (geodesic-diameter-5 wall). **Forced.**

**2. 9×9 single-conv corner-count U-seed (target −880).** DEAD.
D3 (corner-pair ≤4 apart) is logically 100% but needs a 9×9 receptive field (max U corner-pair
Chebyshev dist = 4). A 9×9 conv summing a corner-marker would fire on any cell that sees 2 corners —
but adjacent sprites' corners are within Chebyshev-4 of a shared cell (sprites can be Chebyshev-2
apart), so an L-cell that sees its own 1 corner + 1 neighbour-sprite corner false-fires → L floods as
U. Cross-contamination. The live 5×5 *contiguous-arm* detector avoids this because the arm must
connect within one component.

**3. 2→1 channel U-seed / `pair2` collapse (target −440 to −880).** **EMPIRICALLY DEAD.**
`prototypes/probe_cheap_useed.py` (3000 fresh ARC-GEN, seed_offset 920000, integer conv model that
reproduces the live clean split): a U-seed is viable iff `min over U-comps of (best cell conv)` >
`max over any non-U shape cell`. Results:

| candidate | min U-best | max nonU | clean? |
|---|---:|---:|---|
| K_h5 (live ch0 only) | 20 | 25 | NO overlap |
| K_v5 (live ch1 only) | 20 | 25 | NO overlap |
| K_plus (1-ch symmetric cross) | 14 | 18 | NO overlap |
| **K_2ch (live 2-ch + ReduceMax)** | **26** | **25** | **YES (margin 1)** |

The live 2-channel design clears by a razor-thin margin of **exactly 1** (26 vs 25). Dropping either
channel drops min-U-best to 20 (below the non-U ceiling 25) ⇒ some U's never seed AND some L/H cells
false-seed. Both channels are load-bearing; the `ReduceMax` is required to merge them. Confirms
worker3's "2→1 is correctness-risky" as a hard NO.

**4. 3×3 pocket U-seed (target: replace `pair2` with one 3×3).** DEAD.
`probe_cheap_classifier.py` D1 = 82.98% (3000 pairs). Wide U (`random_you`, width 5) has mouth cols
1..w-2; no background cell has ≥3 shape neighbours, so no 3×3 pocket exists. All D1 mismatches are
true-U→pred-L. A 5-wide detector is required — which is exactly `pair2`.

**5. Fold masks / `turn_code` (target −440 each).** DEAD (correctness-risky, no clean path).
`mask_bool` (flood cond + edge slices + final Where) and `mask_u8` (conv input + turn_code) are
distinct dtypes both consumed; replacing the flood `Where` with `Mul(mp, mask_u8)` etc. re-plumbs
every consumer and risks byte-equality on a graph tuned to a 1-count margin. `ng_has_hv`+`turn_code`
re-encode corners as the exact `{0,1,2}` the `pair2` quant (scale 17) expects. No net-negative,
safe folding exists.

## Route 6 (2nd pass, BUILT + GATED): delete the background-sentinel machinery — REFUTED BY GATE

Hypothesis H-a: the ~766 B grid-extent / background-sentinel machinery (`task364_scalar0`
U8[1,1,20,22]=440 + edge-validity slices ~294 + edge initializers 12) exists only to force
outside-grid cells to **all-zero**. If the grader instead scored outside-grid as **channel-0
background**, the whole machinery could be deleted by painting every non-shape cell (in-grid AND
outside) as color 0.

BUILT: `prototypes/build_nobg_variant.py` surgically rewires the live 12,319 graph —
`task364_scalar6`'s else-input `task364_scalar0`→`zero_u8`, `Pad` value `h_th_code(25)`→`zero_u8(0)`,
prune 10 now-dead nodes + 4 inits (32→22 nodes). `ng_graph_cost_diff` names the delta:
**−766 B** (`task364_scalar0` −440, 9 edge slices −294, 3 edge inits + h_th_code −12) → 11,545.

GATED: `ng_gate visible --task 364 --cand task364_nobg.onnx` → **0/266 (0.00%) FAIL.**
EVERY pair fails — the 30×30 output pad region (present in every example, native grid ≤20×22) must
be **all-zero**, but H-a paints it channel-0. So outside-grid == all-zero is a HARD grader
requirement, and the 766 B sentinel machinery is **FORCED**. (Per §VISIBLE-FAIL, stopped here — no
truth/ab/profile on a visible-zero.) This was the only "soft" component in the census; it is now
gate-confirmed load-bearing.

## Verdict
**WALL / DROP.** 12,319 is at the algorithmic floor for the forced idea-class: F32 input read 1,760
(dtype-locked) + 2-channel corner-pair U-detector (empirically minimal, 26-vs-25 margin) + junction
H-detector (reuses one conv) + BFS-diameter-5 masked 3×3 flood 4,400 (spacing forbids a bigger
kernel) + in-grid-vs-outside background sentinel + minimal broadcast-`Equal` tail. Every byte is
load-bearing (census sums exactly). No positive-cost candidate exists.

**Remaining hypothesis (weak, unpursued):** none structural. The only conceivable shave is a
re-quantization of the background-sentinel edge machinery (~294 B of thin edge slices) into a single
cheaper grid-extent op — but the `scalar0` 20×22 canvas (440) is forced and the edges are already
thin strips; expected upside < a few hundred bytes at high byte-equality risk on a 1-margin graph.
Not worth a build.

Probe scripts: `prototypes/probe_cheap_useed.py`, `prototypes/probe_cheap_classifier.py`
(re-run: `& $env:NEUROGOLF_PYTHON probe_cheap_useed.py 3000`).
