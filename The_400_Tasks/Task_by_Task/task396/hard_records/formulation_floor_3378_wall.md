# task396 hard record: 3378 is the SAFE formulation floor (WALL) — per-tensor proof

**Recorded:** 2026-07-12 by bo_ng_cld_worker1 (Claude lane).

## Live bar (source of truth)
- `python tools/ng_current_bar.py --task 396 --json` → **live_cost 3378**, authoritative_source `manifest`,
  owner Boladi, `updated_utc 2026-07-09T02:03:48Z`, **not stale**. No sub-3378 staged sidecar
  (`Team_Submission/Staged_Boladi/task396.json` absent).
- Live SHA-256 `0f56efef5a391212c94950780a3d7245ef0bedf2ed96cd6a15b625e9c9ffc567`.
- **The live 3378 IS the run-detection graph** (staged `task396_fused_f32_prefix_cost3378`,
  `Team_Best_Onnx/task396.onnx` — all SHAs identical). Its manifest proves it is **byte-equal**
  (200266/200266 A/B, `byte_equal=True`) to the accepted team graph and full-truth 99.960%
  (the known live ambiguity band). It is a legitimate generalizing model, NOT the hidden-bad count-hash.

## Cost model (verified from `tools/_score_v2.py` + `neurogolf_utils`)
`cost = memory + params`. `memory` = intermediate tensor **bytes** (max(static, real-input runtime)/tensor).
`params` = initializer **element count** (dtype-independent — lowering an initializer's dtype does NOT
reduce params). Verified breakdown of the 3378 graph: memory 3047 + params 331 = 3378.
(memory 3047 = 2639 value_info-declared bytes + ~408 bytes of 26 no-value_info node outputs charged at
runtime shape, e.g. `selected_nonzero_raw_bool` [1,1,8,8]=64, `prefix_colmax` [1,1,15]=60, color/count vectors.)

## Formulation 2 — byte-equal compression of the run-detection graph: FLOOR = 3378 (every tensor forced)
The graph (58 nodes): width-search (dynamic Conv on the F32 input) → height-search (single-column U8
QLinearConv) → compose (perimeter mask + marker slice → one Einsum scatter to the 30×30 F16 output).
Each charged item is forced by rule semantics + ORT dtype kernels + the ARC one-hot output encoding:

| Tensor | Bytes | Why FORCED (no cheaper equivalent) |
|---|---:|---|
| `prefix_score` F32 [1,1,18,15] | 1080 | Horizontal run map. **F32 forced**: Conv reads the F32 graph input; Conv F16-output on F32 input is impossible (ORT), and casting the input to F16/U8 materializes [1,10,30,30]=18000/9000 B. **Shape minimal**: 18 rows (rectangle may sit in any active row) × 15 window-start positions (a width-4 run can start at cols 0..14, and the binary kernel [128..1] only scores a run aligned to the window's first col → all 15 starts needed). Materializing the frame channel instead ≥1296 B (dead: `u8_crop_at_source.md`). |
| `hprefix_dyn_w_f32` F32 [1,10,1,8] | 320 | Dynamic Conv weight (`frame_onehot`×binary kernel). 10 in-channels forced (Conv weight in-ch == input 10); F32 forced. Selecting the frame channel *inside* the weight is exactly what avoids materializing the full frame map. |
| `marker_slice_4d` F32 [1,1,8,8] | 256 | 8×8 marker-pixel read; Slice preserves F32; 8=max rectangle. |
| `planes_f16` F16 [1,2,8,8] | 256 | Einsum inputs must be F16 (ONNX Einsum out-dtype == in-dtype; graph output is F16; a pre-cast [1,10,30,30] intermediate = 18000 B). 2 planes required to distinguish inside-crop-bg (ch-0 one-hot) from outside-crop (all-zero) — see don't-do log. |
| `planes_bool` BOOL [1,2,8,8] | 128 | Concat-in-BOOL then one Cast→F16 is cheaper than casting the two planes to F16 first (checked: 512 vs 384). |
| 4× selection BOOL [1,1,8,8] + raw | 320 | valid / frame-border / marker-interior / combined; each a distinct required mask. |
| `row_embed_830` [8,30] (params) | 240 | First-8 identity scatter matrix; **30 forced by output width, 8 by max rectangle**; used for BOTH Einsum axes (`bphw,pc,hr,ws->bcrs`). Building it at runtime (range+Equal) makes a 240-elem intermediate (worse). |
| vertical path (`vcol_pre` 72 + `vcol_u8` 18 + `vcol_score` 15 + idx 48) | 153 | Already the cheap route: single column (best_col known) → U8 QLinearConv. |
| small runtime scalars/vectors + color/threshold inits | rest | color counts, TopK, argmax scalars, marker/color vectors, thresholds — all necessary. |

**No byte-equal-preserving compression remains.** Verified negatives: prefix_score width 15 is minimal
(trimming drops width-4 runs at cols 11..14 → not byte-equal); planes cannot be built cheaper; the
dynamic weight cannot shed its 10 channels (Slice-to-channel is dynamic); initializer dtype changes do
not reduce params (element count).

## Formulation 3 — bbox / column-count geometry: PRICED-OUT **and** hidden-bad (not built — priced)
Idea: replace the run scan with per-column / per-row frame **counts** (bounding-box width/height).
- Summing all 18 active rows/cols requires a **fat dynamic Conv weight** `[1,10,18,1]` F32 = **720 B**
  (must select the runtime frame channel). Horizontal = 720 (weight) + 120 (count map) = 840; vertical
  likewise ≈ 792. Geometry ≈ **1584 B vs the current 1505 B** (thin 320 weight + 1080 map + 105 cheap
  vertical). The current asymmetric design (fat *output*, thin *weight* horizontal; cheap single-column
  vertical) is strictly cheaper — the fat row-summing kernel costs more than the smaller output it buys.
- **Also hidden-bad**: bbox-width ≠ longest-run width whenever frame color is scattered / multi-candidate
  (which LIVE's original 25-conv search explicitly handles). A visible/A/B match would be the
  task004-style "265/265 visible yet −13 LB hidden" trap. NOT safe on a task with confirmed public weight.

## Off-limits (prior evidence)
- **count-hash lookup** (3294 / pre-packed 4033): submitted 2026-05-23 → **−13.40 LB hidden-bad regression**.
  OFF-LIMITS (`task396.md` don't-do log). Any memorized count-vector selector shares this failure mode.

## Verdict: WALL
**3378 is the safe formulation floor.** No correct (generalizing) candidate below 3378 exists via
byte-equal compression; the only cheaper *geometry* is both cost-worse and hidden-bad. Three genuinely
independent formulations investigated to priced/feasibility evidence: (1) DepthToSpace prompt route —
infeasible (`depthtospace_block_stamp_mismatch.md`); (2) run-detection byte-equal compression — floor
proven above; (3) bbox/count geometry — priced-out + hidden-bad.

## One remaining hypothesis (for a future batch)
A safe sub-3378 would require an ORT op that computes a windowed run-scan producing a **sub-F32 output
from an F32 input in one op** (so `prefix_score` drops below 1080 without materializing a crop). No such
kernel is known (Conv F32→F32; QLinearConv needs U8 input; no F32→U8 windowed op). Absent that, 3378 stands.

---

## SECOND PASS (2026-07-12) — library-mined + EMPIRICALLY CONFIRMED

Forced second pass. Mined `INDEX.md`→`techniques.md`/`traps.md` + BoBrain for cheaper assemblies, read
the TRUE rule, and numerically tested the cheapest genuinely-different formulation.

### The TRUE rule (from `prototypes/task396_numpy_reference.py`, 266/266 vs expected)
NOT "longest run" and NOT bbox. The generator plants **MULTIPLE hollow frame-color rectangles** (interiors
hold 0s and/or marker cells); the answer is the rectangle maximising **(enclosed_area, total_area)
lexicographically**, subject to **full-perimeter = frame** AND **interior ∈ {0, marker}** validity. This is
an area-maximising search with interior filtering — no marginal/count/bbox can express it.

### Library levers mined — all already reflected in the 3378 graph, none unlock a safe cut
- **#42 free-input Einsum marginals** (Einsum on the FREE input charges only its output; `bchw,c->bh` = 120B;
  task377 dynamic OneHot-in-contraction). → Gives per-row/col **counts**, which are a DIFFERENT function
  than the area-max rule. **Empirically 0/266** (see test below). Also: Einsum cannot do a sliding window,
  so it cannot replace the run-scan Conv. Dead for width/height.
- **#4h CumSum→ArgMax run-length in U8/INT** — computes a run from a *known start*; the width needs the max
  over *all* starts (a 2D map). The binary-kernel Conv already encodes run-length-from-window-start in the
  bit pattern in one op; #4h would need per-position Gather/ArgMax → more tensors. No gain.
- **#41 QLinearConv U8-in/U8-out prefix plane (~1/4 F32 cost)** — needs a **U8 input plane**. The frame
  channel is **DYNAMIC** (frame_idx from runtime counts); reading it as a plane costs ≥1296B F32 before any
  U8 cast (dead route `u8_crop_at_source.md`). The current dynamic-weight F32 Conv avoids that plane
  entirely, outputting only the [18,15] scan (1080B). #41's U8 path is unreachable here **because the
  channel is dynamic** — this is the root reason `prefix_score` is forced F32.
- **#24 merged run-detection Conv** — the graph already fuses selection+window into one Conv; #24 doesn't
  shrink the 2D output.

### Empirical test (`scratchpad/test_cheap_geom.py`, all 266 train+test+arc-gen pairs)
```
TRUE rule vs expected:        266/266
bbox / marginal geometry:       0/266   (diverges on ALL pairs — proven hidden-bad; visible-fail => hidden 0)
longrun geometry (= 3378 ONNX):266/266
```
`bbox` fails because inputs contain several frame rectangles + scattered marker cells (pair 0: a 7×7 and a
4×3 rectangle of color 2, marker 4 scattered) — the union bbox is meaningless. `longrun` coincides with the
area-max rule on the generator (the dominant rectangle has both the longest runs and the max area), which is
exactly why the current graph is byte-equal to the anchor / ships clean (+1.70 LB history) despite <100%
adversarial. **Any marginal/bbox/count route is a certain visible-fail (hidden-0), not a candidate.**

### Gate anchors on my box (floor candidate, sha 0f56efef…)
- `ng_gate.py visible` → `[visible] 266/266 (100.00%) PASS`
- `profile_real_cost.py --arcgen 266` → `static=3378 profile=3378 (memory 3047 + params 331)`
- (heavy `truth`/`ab --n 2000` NOT re-run on the unchanged existing best — gate hygiene; manifest already
  records truth 2000/2000 + ab byte-equal 200266/200266.)

### Verdict UNCHANGED — WALL at 3378
The only correct geometry is longest-run (bbox 0/266); the current longrun implementation is at its
per-tensor floor; the dynamic frame channel forces the F32 scan (blocking every U8/Einsum cost trick).
No safe candidate strictly below 3378 exists by any route mined.
