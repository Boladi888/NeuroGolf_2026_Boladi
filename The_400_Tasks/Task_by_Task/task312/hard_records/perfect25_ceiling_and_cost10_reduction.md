# task312 dead-route record — perfect-25 rumor + sub-20 contraction reduction

**Batch:** bo_ng_cld_worker3 (Claude lane), 2026-07-07
**Live bar at write time:** cost **20**, sha256 `e34e78a1f9514b4d1cbcbc936a665ca33aa4d02902b47fb0208e24a9f9d4d035`
(manifest, owner Aditya, authoritative via `tools/ng_current_bar.py --task 312 --json`).
Team_Best_Onnx/task312.onnx = the cost-20 graph, profile=20 confirmed.

## Prompt premise was a MISCLASSIFICATION
The task312 prompt (`.../53/...ZeroopsweepCeilingAudit.md`) frames this as a perfect-25
zero-op task (floor ~1, Identity/Transpose/flip/rotate sweep). **False.** The verified rule
is a per-row color map — `out[r,c] = in[r,0] if in[r,c]==5 else in[r,c]` — which *recolors*
every color-5 cell. Officials contain 5s (4 cells at col 1). So output ≠ input on real grids.

### Zero-param spatial family — DEAD (gate-backed)
- `probe_identity.onnx` (Identity, 0 params): **visible 0/265 FAIL**.
- Identity is the *best case* of the spatial family (it alone preserves cell positions, which
  a position-preserving recolor rule needs). Since even identity scores 0 on the 5-recoloring,
  every position-permuting op (transpose/flip/rotate) is strictly ≤ identity ⇒ also 0.
- Conclusion: **no cost-≤1 spatial candidate can be visible-correct.** Perfect-25 rumor killed.
  Do NOT re-run a one-node identity/transpose sweep here.

## Sub-20 contraction reduction — DEAD (floor 20 confirmed)
Live cost-20 graph = single Einsum `nkra,nbrc,k,b->nkrc`, two F32 length-10 selector vectors
`g=[1,-1,-1,-1,-1,0,-1,-1,-1,-1]`, `s=[1,-1,-1,-1,-1,-1,-1,-1,-1,-1]` (20 params, memory 0).
Decomposition: `output[n,k,r,c] = g[k]·RowCount[n,k,r]·(Σ_b s[b]·input[n,b,r,c])`.
The construction uses a whole-row count (Σ_a), so it **never indexes col-0** — that is exactly
why it costs 20 and not 40+ (a length-30 col-0 selector alone would be 30 params, strictly worse).

Tested every way to drop below 20 with a single length-10 vector (cost 10), all **visible 0/265 FAIL**:
| candidate | formulation | equation / inputs | result |
|---|---|---|---|
| `probe_shared_g` | one shared vector = g, used for both k and b | `nkra,nbrc,k,b`, inputs `input,input,w,w` (w=g) | 0/265 FAIL |
| `probe_shared_s` | one shared vector = s, used for both k and b | same, w=s | 0/265 FAIL |
| `probe_drop_g`   | drop g (implicit all-ones on k), keep s | `nkra,nbrc,b->nkrc` | 0/265 FAIL |
| `probe_drop_s`   | drop s (implicit all-ones on b), keep g | `nkra,nbrc,k->nkrc` | 0/265 FAIL |

⇒ g and s are **independently forced**: g≠s (differ at index 5: g[5]=0 suppresses the 5-channel,
s[5]=−1), neither collapses to a shared vector, neither reduces to a free all-ones partner.

## Verdict: WALL at cost 20
Floor argument (forced components): the RowCount-contraction needs two non-constant per-color
selectors — (1) `g` gates the output channel and zeroes the 5-channel; (2) `s` sign-weights the
source for 5-detection. Each is length-10 (color space) and non-mergeable (proven above). Any
alternative that indexes col-0 directly pays ≥30 (length-30 column selector), strictly worse.
No cheaper legal correct formulation found.

## Remaining hypothesis (low confidence, for a future deep-research pass, NOT a quick win)
Could a signed construction encode both roles in <20 params via a shorter shared basis (e.g. a
rank-1 factor set that regenerates g and s from a length-<10 code)? Would need an exact integer
factorization proof + full byte-equal gates; ALS/CP-SAT tooling exists locally but the analogous
570-era rank search already timed out UNKNOWN. Treat as open, not promising.

## Prototypes (evidence, kept):
`prototypes/probe_identity.onnx`, `probe_shared_g.onnx`, `probe_shared_s.onnx`,
`probe_drop_g.onnx`, `probe_drop_s.onnx`.
