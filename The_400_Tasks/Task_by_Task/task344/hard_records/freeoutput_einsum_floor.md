# task344 — WALL: live 492 is the forced free-output floor (full decode + tested escape hatches)

- **Live bar:** 492 (manifest, owner Vamshi, sha256 `f803acc557b5…e077aa2`), static=492 / profile=492
  (memory 0 + params 492). Re-verified 2026-07-09 (`ng_current_bar.py`, `profile_real_cost.py --arcgen 266`).
- **Rule:** input one-hot [1,10,30,30]. Cell `3` orthogonally adjacent to a `2` → `8`; cell `2`
  orthogonally adjacent to a `3` → `0` (bg). 5s and isolated 2s/3s unchanged. Degree-2 (bilinear).
- **Prompt premise (492→196, "delete charged 10-ch prepad") is a STALE-graph misread.** The live has
  **memory 0** — there is no charged prepad. The 909/15,111 Conv graph in the auto-diag is a stale LOCAL
  artifact, not the team-best. Confirmed batch48; re-confirmed here by inspecting the 492 onnx directly.

## Cost metric (read from `neurogolf_utils.py`, authoritative)
- `params = Σ math.prod(init.dims)` — **element count, dtype-INDEPENDENT** (F16/int8 narrowing saves 0 params).
- `memory = Σ bytes of intermediate tensors` (every node output except graph `input`/`output`; dtype-dependent).
- Decode: `run_network` returns `(output > 0)` — **THRESHOLD at 0, not argmax.**
- ⇒ The only lever is *fewer initializer elements at memory 0*. Any intermediate ≥ hundreds of bytes ≫ the
  whole 192-elem recolor budget, so **memory-0 is mandatory** and the graph must be a single input→output op.

## Live is a single degree-2 free-output Einsum (fully decoded)
`bdpq,rd,sr,pa,qk,ua,tu,vk,tv,st,ha,wk,bchw,sc,so->bohw` (input twice: `bdpq` neighbour read, `bchw` centre read).
`output[o,h,w] = Σ_s CV[s,h,w]·PF[s,h,w]·output_slot[s,o]` over 5 slots s:

| s | centre read CV | neighbour field PF | output_slot row | role |
|---|---|---|---|---|
| 0 | Σ_c m_c = 1 (const) | const1 · identity = 1 | `1−2o²` | base const |
| 1 | Σ_c c·m_c = κ | 1 | `4o` | base linear |
| 2 | Σ_c c²·m_c = κ² | 1 | `−2` (∀o) | base quadratic |
| 3 | m2 | shift·m3 = N3 | `+8@0, −2@2` | recolor 2→0 (T_B) |
| 4 | m3 | shift·m2 = N2 | `−2@3, +50@8` | recolor 3→8 (T_A) |

Slots 0-2 sum to `base[o]=1−2(o−κ)²` → `>0` **iff** o=κ (threshold-identity reconstruction of the input
one-hot). Slots 3-4 push the target channel above 0 and the source channel below 0 for recoloured cells.
`spatial_basis[30,10]` = DST-I eigenbasis of the path-P10 shift (rows 0-9), rows 10-29 zero. `spectral_values`
= {identity eigenvalues (ones), shift eigenvalues 2cos(πk/11)}. Params: 300 + 50 + 50 + 30 + 20 + 15 + 15 + 6 + 6 = **492**.

## Forced-component proof (Δ ≥ 0 for every alternative)
1. **Family:** memory-0 ⇒ single input→output op; degree-2 rule ⇒ Einsum (input read twice). Conv is linear —
   batch48's from-scratch Conv(cross)+Clip+Mul/Sub = **79,472** (memory-walled). Only the free-output Einsum is memory-0.
2. **Spatial 300 forced.**
   - Rank: orthogonal-neighbour = shift on a line; on the ≤10 active grid (support10, max-active 10×10) the
     P10 adjacency is rank 10. Symmetric eigendecomp ⇒ ONE shared basis `[·,10]` (non-symmetric UVᵀ = 2×).
     Probe (batch48): freq 10→9 = visible **1.13%**, 10→8 = **0%**. Rank-10 empirically pinned.
   - Width 30 forced by the official 30-wide input axis (Einsum contracts `pa/qk/ha/wk` over it). Rows 10-29 = zero-pad.
   - Zero-pad (200 elems) **unremovable at memory 0** — all three escapes TESTED dead this session:
     - Sparse init → 100 NNZ but grader returns **None**: `sanitize_model` renames node inputs but *not*
       `sparse_initializer` names → Einsum input `safe_name_8` has no matching init → InferenceSession fails.
       (Raw ORT runs it fine — the kill is purely the grader. Also general: DONE_task326.)
     - `Constant(sparse_value)` → params 292 but the dense [30,10] output charges **1200 B memory** → **1492**.
     - Slice→[1,10,10,10] charges 4000 B; crop-process-pad ≈ 8000 B. All ≫ 200 saved.
3. **Recolor 192 forced.** 5 slots = 3 base + 2 recolor.
   - Base = threshold-identity over colours 0-9. Under `(>0)` decode this needs `base[o]>0 ⇔ o=κ`, i.e. the
     sign-pattern of I₁₀ (+diag, −off). **Sign-rank(I₁₀) > 2**: K=2 infeasible (geometric — 10 output vectors
     can't each be origin-half-plane-isolated from the other 9; + Adam 12-seed search fails), K=3 = the quadratic
     (const/linear/quadratic in κ, margins ±1). **3 base slots forced.**
   - 2 recolour slots (T_A, T_B) irreducible: each has a distinct centre mask (m3/m2), distinct neighbour source
     (m2/m3) and distinct output routing — 3 differing rank-1 factors, cannot merge into one slot.
   - Every sub-factor at its rank floor: sources {const1,m2,m3}=rank3→`[3,10]`; spectral {id,shift}=rank2→`[2,10]`;
     3 spatial terms {centre,v-nbr,h-nbr} forced by non-separability of the 4-neighbour stencil (N = shift⊗δ + δ⊗shift,
     a sum of 2 rank-1)→ term-selectors `[3,2]`. Collapsing any factor pair to dense costs MORE (e.g. source_mix·source_basis
     dense [5,10]=50 > 30+15=45).

⇒ **492 is the forced floor.** Spatial 300 (rank-10 × width-30, zero-pad unremovable) + recolor 192 (5 rank-floor slots).

## Attempts (batch56, 2026-07-09)
| # | route | result | verdict |
|---|---|---:|---|
| baseline | live free-output Einsum | static/profile **492** | the bar |
| A | sparse_initializer for spatial_basis (300→100 NNZ) | raw ORT OK; **grader None** | DEAD (sanitize drops sparse names) |
| B | Constant(sparse_value) for spatial_basis | params 292 + **1200 B mem = 1492** | DEAD (dense output charged) |
| C | rank-2 (2-slot) threshold-identity base | K=2 **infeasible** (sign-rank>2) | DEAD (base is 3-slot forced) |
| — | (batch48) Conv(cross)+Clip+Mul/Sub, from scratch | 79,472 | DEAD (memory-walled) |
| — | (batch48) Einsum freq-truncate 10→9 / 10→8 | visible 1.13% / 0% | DEAD (rank-10 forced) |

## Verdict: WALL at 492. Do NOT re-explore
sparse/Constant-sparse init (grader-dead), dtype-narrowing (params=element count, saves 0), freq-rank reduction
(visible-dead), slice/crop (memory-walled), rank<3 base (sign-rank>2), dense-collapse of any recolor factor
(costs more). No sub-492 route exists for this bilinear free-output formulation.
