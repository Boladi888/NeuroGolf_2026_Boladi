# Transferable insight — free-output RECOLOR/RELABEL floor: threshold decode + the 3-slot identity base

**Source**: task344 batch56 2026-07-09 (`bo_ng_cld_worker3`). LIVE 492 single free-output degree-2 Einsum
(`allcolor_source_factored_support10`, memory=0). Verdict: **WALL at 492** — prompt's "492→196 delete
charged prepad" was a stale-graph misread (live has no charged tensor). These tools close the *recolor/relabel*
branch of the free-output family (sibling to DONE_task163's *copy/render* tools; shares DONE_task326's sparse ban).

**Task shape**: bilinear neighbour recolor — cell `3` next to a `2` → `8`; `2` next to a `3` → `0`. One Einsum
reads the input twice (centre + neighbour) and emits `[1,10,30,30]` directly. Applies to any free-output
graph that RELABELS colours by a local/degree-2 rule (Object_Based / Color-Mapping families).

**Tool A — The decode is `(output > 0)`, NOT argmax. Verify it first.**
`neurogolf_utils.run_network` returns `(result > 0).astype(float)`. So a free-output relabel Einsum need not
emit clean 0/1 — it emits signed logits and the grader thresholds per channel. The live base is
`base[o,h,w] = 1 − 2(o − κ)²` (κ = integer colour at the cell): `>0` **iff** o=κ, `≤ −1` elsewhere. Recolor =
additive slots that push the target channel above 0 and the source channel below 0. If you assume one-hot
output you'll over-constrain and miss this whole family. (Same threshold holds task163/197.)

**Tool B — The identity/passthrough base costs exactly 3 slots (sign-rank of I₁₀ > 2).** Reconstructing
"channel κ on, all others off" from a one-hot input under `(>0)` means realising `base[o,κ] > 0 ⇔ o=κ` — the
sign pattern of the 10×10 identity (+diag, −off). Its sign-rank is **> 2**, so the base needs **≥3 rank-1
slots**: the quadratic `{1, κ, κ²}·{1−2o², 4o, −2}` (const/linear/quadratic in colour, margins ±1).
*Proof that K=2 fails*: with threshold at 0, `sign⟨g_o,f_κ⟩` is purely angular; realising "positive iff o=κ"
needs each of 10 output vectors isolated from the other 9 by an origin-line (half-plane) — impossible for 10
points (a middle point can't have the other 9 in one half-circle). Adam over 12 seeds confirms K=2 infeasible,
K=3 feasible. **Don't try to golf a relabel base below 3 slots.** (Note the cheap trick: a TRUE δ passthrough
is rank-10 = 10 slots; the quadratic collapses it to 3 — and 3 is the floor, not 2.)

**Tool C — Each recolor relation is one irreducible bilinear slot.** A relation `centre_mask · (shift · source_mask)`
(e.g. T_A = m3·N2) needs a distinct centre read, a distinct neighbour source, AND a distinct output routing —
three differing rank-1 factors. Two relations (2→0, 3→8) can't share a slot. So recolor slots = #relations,
on top of the 3 base slots. Spatial side is the usual shift eigenbasis: 4-neighbour stencil `N = shift⊗δ + δ⊗shift`
is a sum of 2 rank-1 spatial terms (+ 1 centre term) → 3-term selector; rank forced = grid support (10 here,
probe: dropping 1 freq → visible 0). Same rank/width/zero-pad logic as DONE_task163 tools 1-2.

**Tool D — Zero-padded basis params are REAL; both sparse escapes are dead (extends DONE_task326).**
A `spatial_basis[30,10]` with rows 10-29 zero (support-M free-output pattern) counts all 300 dense elements
(`params = element count`, dtype-free — narrowing saves 0). Two ways to "not count" the 200 zeros, both TESTED dead:
- graph `sparse_initializer` → grader returns `None`. **NEW mechanism**: `sanitize_model` renames node inputs
  but NOT `sparse_initializer` names → the Einsum input becomes `safe_name_8` with no matching init → session
  fails. (Raw ORT runs it — the kill is purely the grader. task326 saw the check_model kill; either way dead.)
- `Constant(sparse_value)` → params drop to NNZ BUT the dense [30,10] output is a **charged intermediate**
  (1200 B) → net huge loss (1492). Slice/crop to shrink the basis charges ≥4000 B. Zero-pad is stuck.

**Recipe to close a free-output recolor/relabel as WALL**: (1) confirm threshold decode (Tool A); (2) decode the
slot algebra — per slot, `CV=Σ_c input·centre_slot[s,c]`, `PF`=propagated neighbour source, `out += CV·PF·output_slot[s,o]`;
(3) split base (identity, 3-slot sign-rank floor, Tool B) from recolor slots (1/relation, Tool C); (4) apply
task163 tools 1-2 to the spatial basis (rank=support, width=input, zero-pad unremovable, Tool D). Land the
forced-component sum; drop a `hard_records/<route>.md`.

**Applies to**: task052/069/073/085/097/098/160/167 (+ Object_Based Color-Mapping neighbours) and any
free-output relabel Einsum. Pairs with DONE_task163 (copy/render CP tools), DONE_task326 (sparse ban),
DONE_task262 (single zero-intermediate Einsum > slice-to-shrink-params).
