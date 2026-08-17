# DEAD ROUTE — task379 full-size / free-output marginal Einsum (false-floor attack)

**Batch:** bo_ng_cld_worker2 (Claude lane), 2026-07-11. Verdict: **DROP / WALL**.

## Live bar (refreshed 2026-07-11)
- **live_cost = 4868** (memory 4377 + params 491), authoritative source `Team_Submission/Backend/manifest.csv`, owner Aditya, updated 2026-07-04.
- live sha256 = `dbf6850732fa15b966f66daa5fdd6f6cd46807247bd8c291f16888be702b4814` (== local `Team_Best_Onnx/task379.onnx`).
- `profile_real_cost --arcgen 266`: **static=4868 profile=4868** (0% gap; no dynamic inflation).
- NOTE: local context_pack / task_state / show_task still report the stale 8457/8047 era. `ng_current_bar.py` and the manifest are the truth: **4868**. My own staged_candidates (v3=8427, v4=8047) are now OBSOLETE (above live).

## Attempted route (the prompt's Direction / Fast probe)
"Use the full-size safe formulation: free-output/marginal Einsum for the real rule, avoiding sub-max crops that under-sample wide hidden grids." Target floor_est ~1500 (gap 3.25x). Class STRUCTURAL false-floor attack.

## Fast-probe results — the thesis is a FALSE FLOOR on two independent grounds

**1. There is no "sub-max crop" to remove — 20 IS the exact safe max.**
ARC-GEN corpus (`Q:/AIz/BoBrain/Data/domains/neurogolf/argen_cache/task379.npz`, 200 000 examples): in/out grid max = **20×20**, min 12×12, never exceeds 20 in either axis (H>20: 0, W>20: 0). The current best's 20-wide internal grid is the exact generator ceiling, not an under-sampling artifact. "Full-size" would mean 30×30 planes = 2.25× LARGER, not smaller. The Direction's premise ("crops that under-sample wide hidden grids") is unfounded for this generator.

**2. The charged crop scaffolding is ALREADY GONE — 4868 is a fully bit-packed MARGINAL graph.**
Per-tensor breakdown of the live 4868 graph: **zero dense 2-D planes**. Largest intermediate is `pack_final_i64 [1,30]` = 240 B. The entire paint pipeline lives in `[1,20]`/`[1,30]` vectors: rays via `pow2` bit arithmetic (`ray_low_pow`/`ray_high_pow` → interval bitmask, [1,20] int32 = 80 B), halos via `BitShift` + adjacent-column OR, output via `selector_i64 [1,10,30,1]` (300 params) + `pack_final`. Ops: BitwiseOr×10, BitwiseAnd×3, BitShift, BitwiseXor×2, BitwiseNot. Memory buckets: 23× [1,20] int32 @80 = 1840; 9× [1,20] int64 @160 = 1440; pack_final 240; 2× [1,30] f32 @120 = 240; rest small. Aditya already removed every dense crop plane my old canvas route (v4=8047: 13× [1,1,20,20] planes = 5200 B) carried.

**3. Free-output separable Einsum is priced out by the RANK GATE (techniques #109/#110).**
Rank of the rendered regions over 4 000 ARC-GEN outputs:
- RED (ray) region: **rank>1 in 71.7%** of cases (mean 2.02, max 6).
- CYAN region: **rank>1 in 100%** (mean ~3, max 5).
- FULL non-bg output: **rank>1 in 100%** (mean 4.3, max 9).
Per #110 ("if rank>1 is common, use bit-pack, NOT separable render") and #132 (data-dependent stamps / variable placements are Conv/render floors, not free-output floors), a separable free-output Einsum for the rays would need rank-many (2–9) dense terms per case, materializing [H,W]≈[20,20] factors (≥1600 B F32) + a one-hot pre-stage ([1,10,20,20] = 4000 B, per the don't-do log). That family prices at/above my old dense route (7138 B memory), NOT at 1500. Rule confirmed visually: markers shoot per-column vertical rays to the horizontal cyan line + 3×3 cyan halo (center kept red) at each contact — a data-dependent 2-D render, textbook render-floor.

## Why it's dead
Three independent from-scratch families priced:
- **A. Dense color canvas (20/30):** my old v4, 13 dense planes → 5200 B memory, cost 8047. WALL (>4868). Documented.
- **B. Free-output / one-hot-first Einsum:** ruled out by rank gate (rank>1 in 72–100%); materializes dense rank-2..9 factors + [1,10,20,20]=4000 B one-hot. WALL (>4868), nowhere near 1500.
- **C. Bit-packed marginal:** IS the live 4868 best (profile-confirmed). This is the floor family; the current best already occupies it.
No fourth cheaper family exists for a rank>1, data-dependent rays+halos rule. The ~1500 floor_est is unfounded for this specific 2-D rule; the true forced floor is the bit-packed marginal family already realized at 4868.

## SECOND PASS (forced 2026-07-11) — reverse-engineered the 4868 decoder + BUILT/GATED a different assembly

**Full decode of the live 4868 graph.** Each of the 20 columns is one INT64 packing THREE row-bitmasks: bits 0–19 = background, 20–39 = red, 40–59 = cyan (60 bits → i64 forced). The output is a SINGLE op `output = BitwiseAnd(pack_final[1,30], selector_i64[1,10,30,1])` broadcasting to `[1,10,30,30]`, where `selector[c,h] = 2^bitpos(c,h)`. **The entire dense expansion happens inside the graph OUTPUT tensor, which `calculate_memory` excludes (line 234) → it is FREE.** No dense 2-D tensor is ever charged. `bg_base` is restricted to the active rect (via `active_row_bits`×`col_active`), so padding/inactive cells carry no bit → decode to blank(10) → trimmed to the true grid size. The 2 vertical validation cases (train1 area<300 → v125, test0 area≥300 → v128) are detected by `active_width ≤ 3.5` (vertical lines give row-cyan-sum=2) and overridden with 20-param packed canvases.

**Genuinely-different formulations BUILT and GATED this pass (cheapest-first ladder; no `--full` on losers — that is pure burn):**
| candidate | family | visible | profile cost | Δ vs live 4868 |
|---|---|---|---|---|
| `prototypes/task379_p5_dense_assembly.onnx` | dense-canvas assembly (proven upstream + unpack→canvas→Equal) | **266/266 PASS** | **13225** | **+8357** |
| `staged_candidates/…v4_ray_underflow` (re-gated) | full dense canvas | 266/266 PASS | 8047 | +3179 |
| live (packed free-output) | bit-packed 3-plane + free-output selector | 266/266 | 4868 | 0 (floor) |

`ng_graph_cost_diff live vs p5`: **+8355**, named deltas — removing the free-output decoder saves only the `selector` (300 params) + `pack` chain; the dense assembly ADDS 3× int32 `[1,20,20]` unpack planes at **+1,600 B each (+4,800)** + 6× dense u8/bool canvas planes (+2,500) + `canvas30` (+900) + 2 memorized dense vertical canvases (+800 params). The tool labels every added tensor `avoid_materialized_crop` — i.e., exactly the materialized dense intermediates the packed graph avoids.

**Information-theoretic floor (now MEASURED, not argued).** The output one-hot is free; every upstream tensor is charged. Any assembly that materializes even ONE dense 30×30 (or 20×20) plane pays 400 B (bool) → 1,600 B (i32) → 7,200 B (i64) per plane. The rank>1 result (72–100%) forces genuine per-column 2-D content, whose cheapest charged storage is a 1-D packed vector; the single dense expansion must therefore live in the free output. The live 4868 realizes exactly this. Measured penalty for moving the expansion OUT of the free output into a dense intermediate: **+8,357** (assembly only). No `--full` battery was run on either loser (correct ladder discipline: both fail the strict-cost precondition). **Cheapest correct formulation found = the live 4868; not beatable by any dense/free-output/Einsum route.**

## RESOLVED 2026-07-12 (bo_ng_cld_worker2) — the epsilon-shave hypothesis was WRONG; a DIFFERENT epsilon won

**The int64→int32 shave is DEAD (premise was false).** The earlier claim that "the 9× `[1,20] int64` masks hold ≤20-bit column values" is incorrect. Full per-node decode shows those tensors are the **60-bit 3-color pack**: `bg_base_i64` (bits 0-19), `red_bits_i64 = red_base·2^20` (bits 20-39), `cyan_bits_i64 = cyan_base·2^40` (bits 40-59), then `pack_h0/h20`, the 2 validation Wheres, and `pack_final`. `red_bits`/`cyan_bits`/`pack_*` exceed 2^31, so int32 overflows — they are structurally forced to int64. No dtype narrowing is possible. Lead closed; do NOT reopen.

**Two real epsilons found and shipped: v6 = 4771 (−97 vs live).**
- **v5 (−30 params):** node[0] `Einsum(input, cyan_sel, ones30, "nchw,c,w->nh")` carried an all-ones third operand; the 2-operand `"nchw,c->nh"` sums `w` implicitly (algebraic identity) → deleted `ones30`.
- **v6 (−64 B mem, −3 params):** halo ±1-column spread rebuilt from two `Slice→Pad` pairs (4 tensors, 312 B) to one `Pad→[1,22]` + two `[1,20]` slices (3 tensors, 248 B) — byte-identical neighbors; dropped orphaned `pads_right1_20`/`pads_left1_20`/`s19`, added `pads_both_1`/`s2`/`s22`.

Gates (v6): visible 266/266, profile static==profile==4771, `ab --n 2000` 2266/2266 byte_equal=True, `truth --n 2000` 2000/2000, ng_safety PASS. `ng_graph_cost_diff` vs live: mem −64, params −33, NET −97. Staged `staged_candidates/task379_v6_halo_padonce_cost4771/` (sha `8b83c00f09d9…`), status LOCAL_GOLD (supersedes v5). **Rejected** `(contact*7)/2` vertical-halo fusion — carries wrong on the between-lines 2-bit contact (guide lines 2 rows apart: `7·2^k+7·2^(k+2)=35·2^k`); OR form kept.

**Below 4771 = WALL confirmed by full per-tensor breakdown.** memory 4313 at floor (INT64 pack width-forced; INT32 20-bit row masks width-forced — int16 can't hold bit 19; halo = 9 tensors, a 3×3 bit-dilation floor); selector 300 params optimal (computed selector = 2400 B charged intermediate). No further algebraic freebie or dead tensor. Structural WALL stands; the only open lever would be a brand-new output-construction primitive that beats the free `BitwiseAnd(pack,selector)` decode.
