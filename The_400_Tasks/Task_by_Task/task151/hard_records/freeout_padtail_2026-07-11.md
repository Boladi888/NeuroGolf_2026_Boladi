# DEAD ROUTE — task151 FreeOutputPadTail (SWEEP crack, 2026-07-11)

**Worker:** bo_ng_cld_worker1 (Claude lane). **Verdict: DROP / WALL. No candidate staged.**

## Live bar (refreshed this session)
- **cost 900** — authoritative source = manifest (`ng_current_bar.py --task 151`); reconciled with team_best + workbook + local, all agree 900, none stale.
- sha256 `923f0ee2c55298f44b9f8669305997a4277a7143c4a847c20d536b039d0e163f` (owner Aditya, 2026-06-19). `Team_Best_Onnx/task151.onnx` sha matches exactly.
- Graph = **ONE node**: `Conv(input, W[10,10,3,3]) -> output`, no bias. score_v2 `(macs 0, mem 0, params 900)`; profile `static=900 profile=900` over 266 real inputs (no dynamic gap).
- Leaderboard state 2026-07-11: public 7559.16 / manifest_total 7555.99. No task151 entry pending in `Staged_Boladi/` (only task338/task377).

## Attempted route
Prompt "FreeOutputPadTail": dissolve the **FLOAT16 1x10x12x12 Where/Equal prepad canvas** (the "charged 2880-byte prepad tensor") into a `Pad`/`Where`/`Equal`-as-output free-output tail; target ~420, gap 2.14x.

## Why it's dead (two independent kills)
1. **FALSE FLOOR — the charged component is already gone.** The 2880-byte F16 `1x10x12x12` prepad exists ONLY in the *stale Kaggle LIVE extract* (`live_costs.json` cost 17,046, refreshed 2026-06-05) — the OLD Slice/Conv/Where/Gather/Pad F16 structure. The team already superseded it with the 900 no-bias Conv. The team-best graph pays **no** prepad canvas / carrier / template. Per the 2026-07-10 workflow rule ("if [the charged component] is already gone... return DROP"), the shave premise is void. The ~420 floor_est was computed against a tensor that no longer exists.

2. **Forced-cost wall — every free-output route prices > 900** (re-derived fresh with the exact cost model `cost = memory_bytes + params_elements`):
   - Fast probe re-priced LIVE this session: `task151_coord_fused.onnx` = **7,713** (mem 7,636 + params 77); `task151_coord_route.onnx` = **12,213**. Both 266/266 correct but ≫ 420 target and ≫ 900 bar. **Kill rule triggered → stopped, did not grind.**
   - The dense Conv emits the full `[1,10,30,30]` output in ONE node whose output *is* `output` ⇒ **0 memory**; total 900 = params only.
   - ANY detect+overlay route must (a) find the line-crossing (r*,c*) — a channel reduction of the one-hot input forces a `[1,1,30,30]` nonbg/activity intermediate (3,600 B F32; 900 B best-case 1-byte) OR marginal Convs (~600 params); AND (b) feed `Where` a **2-D ring condition** `[1,1,30,30]` BOOL = **900 B** (a 3×3 box = AND(row-band, col-band); a 1-D condition paints whole rows/cols, not a box — cannot stay 1-D). So route floor ≥ 900 B (ring cond) + detection > 900. Even the theoretical best-case (1-byte plane 900 B + BOOL ring 900 B ≈ 1,800 B) exceeds 900. Measured floor 7,636 B.
   - Corroborating (prior, consistent): `W.reshape(10,90)` full rank 10 (σ tail flat) → rank-9 factor (900 params, no saving) already only 98.1% >0-threshold truth; IC-reduction to `[10,9,3,3]` needs a Slice (`[1,9,30,30]` = 32,400 B); ScatterND ring indices are data-dependent and exceed 900.

## One remaining hypothesis (judged unreachable)
A route that emits the 10-ch `[1,10,30,30]` output as the graph output node (0-mem) WITHOUT first materializing any `[1,1,30,30]` detection plane. Unreachable: line detection from a one-hot input inherently needs a per-cell channel reduction (≥900 B plane), and the ring overlay needs a 2-D condition (≥900 B). Do not re-explore unless the rule itself changes or ORT starts crediting sub-900-B 2-D BOOL conditions AND detection can be done in <1 B — the arithmetic still lands ≥ ~1,800 B.

**900 is the hard formulation floor. WALL re-confirmed. Do not re-run truth/ab/battery on this route.**

---

## SECOND PASS (forced, 2026-07-11) — fresh decode + genuinely-different formulation, MEASURED

Re-decoded the rule from raw pixels (not inherited): input has one H-line + one V-line crossing at (r*,c*); the **8-cell ring** (3×3 minus center) → **color 4**; center + all else = passthrough. Exactly 8 cells change, always color 4, every pair. **Grids are ≤12×12** (H,W ∈ {4,6,8,12}); **color 4 never appears in input** ⇒ output channel 4 *is exactly* the ring mask. Team-best `W[10,10,3,3]` inspected: all 10 output channels carry genuine 3×3 taps (none center-only) — the ring-suppression is baked independently into every channel, which is *why* a single Conv costs 900 and cannot share the detection internally.

**Genuinely-different candidate built + gated** (`prototypes/task151_marginalconv_where.onnx`, builder `build_task151_marginalconv_where.py`): marginal-Conv detection (NO F32 activity plane) → row/col count vectors → ArgMax(r*,c*) → ONE Chebyshev-1 distance-predicate ring mask → `Where(ring, e4, input)` as graph output (#427-433, 0-mem output).
- `ng_gate visible` = **266/266 PASS** (correct).
- score_v2 = `(macs 0, mem 4936, params 673)` · profile_real_cost `--arcgen 266` = **static=5609 profile=5609** (no dynamic gap).
- `ng_graph_cost_diff` vs team-best: **NET +4706 (regression)**. Reconcile: **params −230, memory +4936 B**. Named charged structures: 4× `[1,1,30,30]` BOOL planes (box/center/notc/ring = 3,600 B) + marginal-Conv weights (Wr/Wc, +600 params) + iota/band tensors.

**This is the empirical forced-cost proof:** moving work off the Conv *lowered params by 230* (673 < 900) but *raised memory by 4,936 B* — because every non-Conv assembly must materialise the ring as an intermediate. Even the theoretical best-case (fuse the 4 BOOL planes into ONE 900-B ring + keep the ~673 detection params) ≈ **1,570 > 900**.

### Final-op-family enumeration (forced-component standard #68/#134) — every family ≥ 900
| final op | why it can't beat 900 |
|---|---|
| **Conv** (team-best) | `W[10,10,3,3]`=900, output IS graph output ⇒ 0 mem. **Unique 0-memory family.** Kernel can't shrink: <3×3 loses the ring receptive field (proven visible conflicts); grouped/sparse QConv is scorer-rejected (#393, dense-only); IC-reduction needs a Slice (`[1,9,30,30]`=32,400 B). |
| **Where(mask,e4,input)** | mask is a forced 2-D `[1,1,30,30]` ring condition = **900 B** (L200-203 / #427-433: overlay tax ties 900 even for a small active region), + detection > 0 ⇒ **>900**. Measured 5,609 (this pass) / 7,713 (first pass). |
| **Gather / GatherElements** | single-Gather sub-900 exists ONLY for a spatially-UNIFORM channel permutation (recolor floor, `cost_floor_reference.md` L156-161); task151 is spatially-varying ⇒ per-cell index `[1,10,30,30]` = huge. |
| **ScatterND** | 8 ring cells are data-dependent; index/update tensors + crossing computation exceed 900. |
| **Add/Mul(input, Δ)** | Δ or its mask is `[1,1,30,30]`≥900 B (L157: every `[1,10,30,30]`-output data-carrying intermediate ≥900 B). |
| **Pad(inner12×12)** | needs `Slice(input)→[1,10,12,12]`=5,760 B intermediate; the "FreeOutputPadTail" premise dies on the input slice. |

**Information-theoretic core:** the ring is a data-dependent `[30,30]` spatial decision. It is either (a) recomputed per-output-channel inside a 0-mem Conv = 900 params, or (b) materialised once as a `[1,1,30,30]` intermediate = 900 B. Both = 900; route (b) additionally pays detection ⇒ strictly >900. The Conv sits exactly at the 900 knee. **No sub-900 route exists.**
