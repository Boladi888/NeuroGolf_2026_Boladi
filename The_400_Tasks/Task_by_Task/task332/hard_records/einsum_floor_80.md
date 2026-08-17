# task332 — WALL at 80 (fused Einsum floor)

- **Live bar:** cost **80**, sha256 `ea1e530cde02e3cfc8aa793b77e574cf2e3b91a9c66bb809f44213dd4b8fba6f`
  (manifest, owner Vamshi, authoritative per `ng_current_bar.py`, not stale, 2026-07-02).
- **Verdict:** WALL. No sub-80 candidate is structurally possible. Do not re-explore.

## The real rule (verified 267/267, NOT identity/transpose)
A gray `5` becomes `3` iff its column `c` satisfies `c ≡ (W-1) mod 2` — i.e.
**right-anchored column parity**: the rightmost occupied column always flips,
alternation runs leftward, and the anchor phase depends on grid width `W`
(which varies per example). `0` stays `0`; only colors {0,5}→{0,3,5} ever appear
(channels 1,2,4,6,7,8,9 are always-zero across all 267 examples).

The prompt's "perfect-25 / cost ≤1 / identity-transpose" premise is FALSE for
this task. Local pre-scan already found no whole-grid identity/flip/rotate; this
is a genuine data-dependent color remap. The legal zero-param family is dead.

## Why 80 is forced (cost model: params = init elements; memory = intermediate BYTES; `input`/`output` tensors excluded)
The LIVE graph is a **single fused Einsum** (`bkrc,tk,blij,tl,tj,to,tc->borc`),
memory **0**, params **80** = `column_basis[2,30]`(60) + `color[2,10]`(20).
Its elegance: the tied *second* `input` copy is globally reduced (over channels,
rows, cols) to detect **W-parity** (the right-anchor phase) from the content
itself, reusing the SAME 60 column params for both the forward color-transfer
AND the phase detection. Zero wasted structure.

Forced-component argument for the ≤80 floor:
1. **Any op that materializes even one full-resolution plane loses.** The output
   is `[1,10,30,30]`; a single `[1,1,30,30]` intermediate = 900 elems × ≥1 B =
   ≥900 bytes >> 80. So a multi-node/mask-multiply/scatter route is memory-bound
   and cannot reach ≤80 (measured: 40,365 — see below). Only a fully-fused
   single op with `memory=0` qualifies.
2. **Conv is out.** Absolute/right-anchored column parity is position-dependent;
   Conv is translation-equivariant and cannot compute it. Einsum is the only
   fused op left.
3. **Einsum params are forced at 80.**
   - The transform needs a per-column vector over the 30-wide canvas. Two rows
     are required: a constant/ones row (channel-0 passthrough `W[0,0,c]=1` for
     all `c`, which is c-uniform and cannot come from the globally-reduced tied
     input) and an alternating parity row → `[2,30]` = 60. The 30-length column
     axis cannot shrink without a Reshape/Slice that materializes a charged
     intermediate (period-2 reshape → `[1,10,30,15,2]` = 9000 elems, dead).
   - `color` must be `[2,10]` to align with the input's 10-channel `k` axis.
     Slicing input to the 3 active channels {0,3,5} materializes `[1,3,30,30]`
     (2,700 elems) >> the 14 params saved. Locked at 20.
   - Single-input rank-2 symmetric tie (`W[o,k,c]=Σ_t color[t,o]color[t,k]colb[t,c]`)
     is algebraically infeasible (needs `a0=1,b0=0,a5=0` ⇒ `W[5,5]=b5²·s(c)`
     cannot equal `even(c)=(1+s)/2`). The second input tie is mandatory and adds
     no params. No room to shrink either the 30 or the 10 axis.

## Attempts run this batch
| formulation | correctness | cost | result |
|---|---|---|---|
| LIVE fused Einsum (baseline) | `visible 267/267 PASS` | static=80 profile=80 (mem 0 + params 80) | the bar |
| F1: multi-node geometric right-anchor (ReduceSum→Greater→CumSum(reverse,excl)→Mod→Slice/Mul/Concat) | `visible 267/267 PASS` | static=40365 profile=40365 (mem **40350** + params 15) | LOSE (memory-bound; validates #1) |
| F2: single-Einsum family (analysis) | — | forced ≥80 | LOSE (forced-component proof above) |
| F3: 1×1 Conv + column mask | — | ≥900 (mask-multiply materializes a plane) | infeasible by #1/#2 |

## Remaining hypothesis
None with a credible sub-80 path. A different fused single op that both (a)
carries a length-30 column signal and (b) detects W-parity for the anchor, in
< 80 charged units, would be needed — no such op exists in the allowed set.
Treat 80 as the floor unless the cost model changes.
