# task173 hard record — U8-scatter FreeOutputTail empirical floor + live-best correctness finding (WALL @ 8710)

- **worker / date**: bo_ng_cld_worker3 (Claude lane), 2026-07-12.
- **bar used (refreshed live)**: `python tools/ng_current_bar.py --task 173 --json` → authoritative
  **manifest live_cost `10853`** (owner **Sho**, sha256 `0456dd7533e95cdfd832a87ec0ea8ee56174eee8dfb7862e3d273faca40e9a62`,
  updated 2026-07-09T01:09:17Z). Effective best ceiling = local staged **`8710`**, sha256
  `27f31830cf75f60bdcf0273b403dcb804b8f4b6950817091f50e761be3a57940`. Leaderboard fresh: public 7569.92 (2026-07-12).

## ⚠️ HEADLINE — the LIVE team-best 10853 is LESS CORRECT than the staged 8710
Direct measurement this batch (both models, same corpus):
- `ng_gate truth --task 173 --n 2000` on **8710** → **2000/2000 (100.000%)**.
- `ng_gate truth --task 173 --n 2000` on **Sho 10853 (live)** → **1998/2000 (99.900%)** — **2 latent misses**.
- `ng_gate ab --cand 8710 --ref Team_Best_Onnx/task173.onnx --n 2000` → `match 2264/2266 | divergences=2`.
  The 2 A/B divergences ARE the 2 samples where **live 10853 is wrong and 8710 is right**.

⇒ **8710 dominates the live team-best on BOTH axes: −2,143 B cheaper AND strictly more correct** (100% vs 99.9%).
Shipping 10853 forfeits ~0.1% of the hidden distribution that 8710 gets right. 8710 is byte-equal to the 8829
oracle (an OLD team-best), NOT to the current 10853 (different architecture) — so it is a more-correct **Path-B**
candidate (status PROBE_READY), not a byte-equal inherit. Promotion decision is the captain's; the correctness
delta is a strong new argument to promote it (or to spend one Kaggle probe confirming it).

## CORRECTION to prior hard records (carrier_narrow_2026_07_10 / single_plane_2026_07_10)
Those records claimed the **U8 color-index carrier is "impossible / a net-add"** ("no single op gives U8 from
F32 one-hot without the ≥9,000 B tax"). **That was WRONG.** The live team-best (Sho 10853) IS a U8-scatter
FreeOutputTail graph: `Conv(input, conv_w[1,10,1,1], pads=[0,0,-3,-3]) → label_f F32[1,1,27,27]` (color+1,
cropped 30→27) → `Cast → U8 grid` → all detection/fill in U8/INT32 index space (TopK-21 active pixels,
Gather neighbor checks, ScatterElements fills) → `Pad 27→30 → Equal(padded, class_ids[1..10]) → output` (free).
The F32→U8 step is a plain `Cast` of the *already-collapsed* 1-channel `label_f`, not a Quantize of the 10-ch
input — cost 2,916 B (the F32 collapse plane), not 9,000. The U8 FreeOutputTail family is **viable**, just not cheaper.

## Attempted route this batch (genuinely independent formulation): golf the U8-scatter family below 8710
**Empirical fast-probe** (`prototypes/bo_ng_cld_worker3_u8_skeleton_floor_2026_07_12.py` →
`prototypes/u8_skeleton_floor.onnx`): the irreducible read+enumerate+tail skeleton at the MOST aggressive
25×25 crop (collapse → Cast U8 → Cast F16 → TopK-21 → ScatterElements → Reshape → Pad 25→30 → Equal-tail),
with a **trivial identity fill (zero real detection logic)**:
```
  label_f  F32 [1,1,25,25] 2,500   (collapse — F32 forced by the F32 input)
  gridf16  F16 [625]       1,250   (TopK — F16 forced; TopK[U8]/[I8] = NOT_IMPLEMENTED in ORT 1.24, verified)
  padded   U8  [30,30]       900
  grid4d/gridflat/out_flat/out2d  U8 625 ×4  = 2,500
  TopK pos/casts             ~315
  TOTAL skeleton floor (no detection logic) = 7,465 B
```
A CORRECT graph must ADD the per-pixel neighbor-check + fill scaffolding on top. Sho's live graph spends
**~2,000–2,400 B** on exactly that (≈20× INT32[21] neighbor-offset arrays, 2nd TopK-7, fam-table Gather,
`fill_dest`/`fill_col`, T_h/T_v/T_dr scatters). ⇒ realistic U8-scatter floor **≥ 9,465 B** at a *risky* 25×25
crop; Sho's safe 27×27 makes it larger (skeleton 8,505). **> 8710 by ≥755 B in every case.**

## Why lower is unreachable (cross-architecture forced-component argument)
The task forces ONE expensive tensor family; the two architectures sit at the two corners of the trade-off and
neither can borrow the other's cheap half without paying its expensive half:
- **Dynamic-Conv (8710)**: cheap sparse base-10 decode + **dense render weight `dynamic_w` F32[10,10,3,3]=3,600**
  (position-FREE Conv — no per-pixel enumeration) + detect `detect_map` F32[1,1,23,23]=2,116 (k=8 minimal;
  k=9→visible 262/266). Core 5,716 + scaffolding/params ⇒ **8,710**.
- **U8-scatter (10853)**: cheap-ish read `label_f`=2,916 + **expensive position enumeration** (`gridf16` F16
  TopK grid 1,458 + U8 grids 2,916 + INT32 neighbor scaffolding ~2,000) — sparse render is only "cheap" because
  it pushes the cost into enumerating WHERE to write. Total 10,853; best-case golf ≥ 9,465.
- **The render is the pivot**: a dense Conv render (3,600, one tensor, applies everywhere for free) is cheaper
  than any sparse scatter render (which must enumerate positions via a TopK-able grid = F16 1,458 + U8 grids).
  You cannot fuse "cheap read (2,116 detect Conv)" with "sparse render" — sparse render needs the U8 grid the
  detect Conv never builds. Splitting the dense render into 1×1 color-map + 3×3 spread materialises a
  `[1,10,30,30]`=12,000 B inter-stage canvas (only final `output` is free). ⇒ **8,710 is the global floor.**

## 2nd-pass addendum — #85 INT8-weight QLinearConv render (the last untested free-output lever): EMPIRICALLY DEAD
Library mine (`techniques.md` #67/#78/#81/#85) surfaced one render lever not yet built: replace the F32
`dynamic_w[10,10,3,3]`=3,600 with an **INT8** dynamic weight = **900 B** (−2,700) via a free-output QLinearConv.
Built + measured (`prototypes/bo_ng_cld_worker3_qlc_int8_render_floor_2026_07_12.py`):
- INT8 weight `w[10,10,3,3]` = 900 B — the lever loads (QLC accepts INT8 weight). ✓
- **But QLC needs a 10-channel U8 render INPUT** (color routing forces 10 in-channels): cheapest reachable is
  `Cast(F32 one-hot → U8)` = **9,000 B** (or a 25,000 B F32 Slice if cropped-then-cast — measured). ✗
- **And QLC emits a U8 output** [1,10,30,30] that must become BOOL → another **9,000 B** charged tensor. ✗
- Net render ≈ 900 + 9,000 + 9,000 = **18,900 B** vs 8710's 3,600 (F32 weight, FREE F32 input, FREE output).
⇒ The **F32-input tax** ("kills every dtype-narrowing escape", floor proof) defeats the INT8 lever. 8710's render
reads the free F32 one-hot with an F32 weight and writes the free output, paying ONLY the 3,600 weight. No U8/INT8
render route can undercut it. All other library free-output levers (#67 Equal-tail / #78 placer-Conv & Pad-as-output
/ #81 pad-floor / #79 code-vector) reduce to routes already priced: Equal-tail/placer = Sho's U8-scatter (10,853),
fused-Conv = 8710. None yields a data-dependent color-routing render < 3,600 B.

## Verdict
**WALL at 8710 — now confirmed across BOTH architectures** (dynamic-Conv AND U8-scatter), with empirical priced
evidence (Sho's real 10,853 graph + measured 7,465 skeleton floor + 8710's real gates). The prompt's ~900
floor_est is a **fantasy**: `label_f` alone (2,500 at 25×25) is ~3× the entire target and is a single forced
tensor. No route across Conv / ConvTranspose / Einsum / Scatter / plane / U8-carrier materialises a
data-dependent render < 3,600 B while writing the free one-hot output.

## Gates run (exact result lines, this batch)
- `ng_gate visible --task 173 --cand <8710>` → `[visible] 266/266 (100.00%)  PASS`
- `ng_gate truth  --task 173 --cand <8710> --n 2000` → `2000/2000 (100.000%)`
- `ng_gate truth  --task 173 --cand <10853 live> --n 2000` → `1998/2000 (99.900%)`  ← live is buggy
- `ng_gate ab --task 173 --cand <8710> --ref <10853 live> --n 2000` → `match 2264/2266 | divergences=2`
- `profile_real_cost --task 173 --cand <8710> --arcgen 266` → `static=8710 profile=8710`
- `ng_safety model --task 173 --cand <8710> --json` → `verdict PASS, 0 failures, 0 warnings, runtime_integer_denominators=[]`
- `ng_graph_cost_diff --live <10853> --candidate <8710>` → `NET COST DELTA -2145 (candidate CHEAPER)`
- ORT dtype probe (`prototypes/bo_ng_cld_worker3_dtype_probe_2026_07_12.py`): TopK[U8]/[I8] NOT_IMPLEMENTED;
  TopK[I32]/[F16] OK; ScatterElements[U8]/[I8]/[I32] OK.

## One remaining hypothesis (future batch only — do NOT re-open the priced routes)
Unchanged and unfound: a data-dependent render op that writes the one-hot output for free AND materialises
< 3,600 B without a `[10,10,3,3]` F32 weight, a full `[1,10,30,30]` canvas, OR a TopK-able U8/F16 grid for
position enumeration. Absent such an op, 8710 is the floor. Any sub-8710 attempt must first exhibit that op's
priced tensor — not re-grind carrier-narrow, single-plane, pair-code, Einsum, k9-detector, or full U8-scatter.
