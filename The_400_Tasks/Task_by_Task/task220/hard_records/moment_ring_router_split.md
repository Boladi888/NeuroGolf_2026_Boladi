# hard_record: task220 SparseConvRingRouterSplit — WALL at 540

- **Date:** 2026-07-03
- **Worker:** bo_ng_cld_worker2 (Claude lane), batch 48 slot 04
- **Live bar used:** cost **540**, sha256 `85ffdce12a173aa2ee5d07ecff8fa32e67502d25d141d8536d8d6fce59d17142`,
  owner Vamshi, source = **manifest** (authoritative), reconciled by `tools/ng_current_bar.py --task 220`.
  (Local `live_costs.json`/task_state said 900 — STALE; live wins.)

## Rule (verified exact, 267/267)
Per-color 3×3 ring stamp: nonzero pixel v∈{2,3,8} → keep center v, draw full
8-neighbourhood ring of color ring[v] (2→1, 3→6, 8→4). Colors ⊆{2,3,8}, ≤1 per
color, never on grid edge, ≤3 objects.

## Attempted route (prompt's direction) + why it's dead
Prompt asked to factor "the dense [10,10,3,3] Conv (52/900 nnz)" into a grouped
ring detector + `[10,10,1,1]` color-router, target ~260 = floor_est.

**The premise is false.** LIVE 540 is NOT a dense Conv — it is
`task220_poly540`, a moment/Einsum decoder (15 inits; 7 Einsum + 9 Pow + 3 Sign;
`inspect_graph` confirms). The dense `[10,10,3,3]` Conv is the *900*-cost artifact
(needless090), already superseded.

### Exact failed gates / measurements
- `ng_gate visible` LIVE (`Team_Best_Onnx/task220.onnx`): **267/267 (100%) PASS** → LIVE is exactly correct (not an ambiguous class).
- `score_v2` LIVE: `(macs=0, mem=132, params=408)` → **540**.
- `score_v2` my prototype `prototypes/task220_momentring.onnx`: `(mem=180, params=303)` → **483** (< 540) but `ng_gate visible` **0/267 FAIL**.

### Forced-floor argument (why <540 is unreachable without shaving LIVE)
Cost = `params + Σ(intermediate-tensor bytes)`. Exact output must be a full
one-hot (background channel-0 = 1 off every stamp).
1. **Single dense Conv** = 900 params (routing not block-diagonal → no grouping win).
2. **Any factored Conv** bills a full-res intermediate: `[1,10,30,30]`F32 = 36,000 B,
   even one `[1,1,30,30]` = 3,600 B — each alone > 540.
3. **Exact one-hot planes** (ReduceMax/MaxPool `[3,30]` per color) sum to ~1.4–5.8 kB
   memory (`Σ`, not max) — over budget.
4. Only poly540's **single terminal Einsum** (coord expansion folded into a
   `coord_basis` param, moments kept as scalar intermediates) reaches ~132 B memory.
   540 is a tight instance of that regime.

### Sub-traps recorded (save the next batch the detours)
- **Sparse-init Conv weight (27 nnz)**: DEAD — `SparseTensorProto.name` missing →
  `calculate_memory` raises → `score_network=None`; sanitizer/shape-infer also break
  (core.md §sparse, traps §8e). Not probed further.
- **Degree-4 monomial render** (`Σ K·rⁱ·rc^j`): float32-unstable (6e8-scale terms,
  catastrophic cancellation → far cells round to exactly 1.0). Numpy with stable
  `(r-rc)²` arithmetic was 267/267; the ONNX monomial fold is the failure. Any
  degree-4 must **multiply two `(r-rc)²` factors** (poly540's trick), not expand monomials.
- **Polynomial render can't be exactly 0 off-stamp** → background/one-hot completeness
  forces the input-passthrough (`nihw`) + `channel_map[7,10]` structure LIVE already has.

## One remaining hypothesis (if ever revisited — build cost may not justify)
A **shared-slot** poly540 variant: reuse poly540's stable factor-mult stamp +
input-passthrough background, but share ONE render across the 3 colors via a `[3,10]`
colour selector instead of replicating poly0/1/2 (=126) + sources (=30) + `channel_map[7,10]`
(=70). Rough price ~300 params + ~132 mem ≈ **~432** (≈ +0.2 pt). This is effectively a
shave of LIVE's specific construction — high build cost, high regression risk under the
exact-one-hot constraint. Must pass `ng_gate truth --full` before any stage. Not attempted.

## Verdict
**WALL @ 540. Nothing staged (never stage a tie; no strict win found).**
