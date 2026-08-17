# FREE-OUTPUT sweep wave-3 findings (worker-7, 2026-06-07) — 12 boladi conversions, +10.24 pts staged

Full results table: `Logs/spread_scan_results/freeout_sweep_20260607_worker7.md`.
Scanner now PERSISTED at `tools/scan_freeout_candidates.py` (CSV + team_split join);
profile gate now PERSISTED at `tools/agent_helpers/profile_real_cost.py` (WORKER_PROTOCOL
gate-5 had no tool — it replays officials through ONE profiling session; `calculate_memory`
maxes per-tensor across ALL trace events, so one session = the credited cost).

## 1. The PAD-FLOOR CROSSOVER LAW (extends #67 — price BEFORE building)

The Equal-tail costs a FIXED ~900 B (padded [1,1,30,30] u8 canvas) + ~30-60 B canvas chain
+ ~30 params. Converting `X[1,10,h,w] → Pad → output` pays only when killing X beats that:

| X dtype | wins iff h·w > | example losers (sweep-verified) |
|---|---|---|
| BOOL | 100 | 031 (56), 267 (49), 127 (77), 360 (40) |
| F16  | 47  | 272 (25) |
| F32  | 23 (≈ but chain overhead pushes real bar to ~+0.2-worthy: prepad ≥ ~1,000 B) | 373/244/236/266 marginal-or-negative |

Tasks already ENDING `canvas → Pad → Equal == output` with small canvases are at their floor —
do not "convert" them again. The 79-task corpus map (54 deferred / 22 aditya / 3 boladi ≥1,800 B)
lives in `Logs/spread_scan_results/freeout_scan_20260607.csv` for the team merge.

## 2. ACTIVE-RECTANGLE MASK is mandatory for same-size Equal-tail rewrites (the 353 lesson)

Canvas value 0 decodes ch0=True. For tasks whose grid size VARIES, out-of-active-grid cells
inside the canvas region MUST get the sentinel: `Where(act_b, values, 10u8)` with
`act_b = Greater(Conv(input, ones[1,10,1,1], negpads), 0.5)` (+~870 B for a 14x12 canvas).
353 v1 without it: officials 0/9 with all intermediates correct. Two immune classes:
- **Passthrough tails** (#78 Where-as-output, e.g. 292): untouched cells come from `input` itself.
- **LIVE-matches-blindly** (047-class): when LIVE itself writes ch0 across the full fixed canvas,
  matching LIVE byte-equal needs no mask (the corpus never exercises the difference).

## 3. Where-as-output beats Equal-tail for sparse SUBSTITUTION rules (292: 6,301 → 1,310)

If output == input except at maskable cells, never rebuild the canvas:
`cond_small → Pad(BOOL, False) → [1,1,30,30] 900 B → Where(cond30, e_k [1,10,1,1], input) == output`.
Replaced cells get the FULL e_k channel vector (kills the old color, births the new one) — and
no active-mask is needed. Same 900 B floor, but the entire value-canvas chain disappears.
Discriminator vs 312's einsum-final: substitution-with-identity → Where; fully-generated → contraction.

## 4. Einsum scalar reads + Equal-ramp PLANE REBUILDS compose (353/047/389)

Singleton-object tasks need NO input plane at all: read each coordinate as
`Einsum('bchw,c,h->b', input, sel_c, ar30)` (4 B, #42), rebuild the object plane as
`And(Equal(posr, r), Equal(posc, c))` (ramps are params). 353 also confirms a CENTROID-singleton
rebuild of a passthrough plane held byte-equal over 200k. k-VALUE extraction (389):
`Einsum('bcij,c->b', ReduceMax(input,[2,3]), arange-with-banned-zeroed)` — presence-weighted id.

## 5. Builder-hygiene traps re-confirmed (cheap to avoid, costly to debug)

- READ LIVE INITS VERBATIM, never synthesize from 1-2 train pairs: 104's mask01/10 are
  ANTI-diagonal (first cut 2/3); 194's BL quadrant is rot90CCW, not transpose — train0's
  palindromic column masked it (first cut 1/4). Officials catch these only if the pairs
  exercise the case — the 200k ab gate is the real net.
- D2S(CRD) u8 + broadcast `Min(idx3, Reshape(idx3,[1,9,1,1]))` builds fractal self-products
  in 2 ops (001; extends #80 to outer-MIN block stamps).
