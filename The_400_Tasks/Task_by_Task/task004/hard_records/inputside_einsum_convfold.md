# Dead route — InputSideEinsumConvFold (batch54, 2026-07-07)

**Live bar:** 2367 (manifest, owner Boladi, `ng_current_bar --task 4`).
**Live SHA-256:** `0db63c0f7b1435d0874cf6a1afc952d0992473f8fca22b20d28bedb2bf567c5b`
(`staged_candidates/task004_factored_bcur_cost2367/task004.onnx`; profile==static==2367 = mem 360 + params 2007).

**Prompt target:** 2367 → ~700 (EV +1.22), "fold separable Conv/read logic into
input-side Einsum operands, keep final output free." **Verdict: WALL at 2367.**
The current live build ALREADY is the input-side free-output Einsum fold. The ~700
floor_est is a heuristic that omits the forced column-placement remap.

## The forced component (why ~700 is unreachable)
Rule (re-derived fresh from `prototypes/verify_rule.py`): every non-bg cell shifts
**right by 1 column**, except box bottom-row + corner-above cells (stay). The **row
axis is unchanged** — the only spatial remap is the horizontal shift-by-1. In a
single free-output Einsum the colored source pins the contracted input-col axis `c`
to 30 and the output pins `q` to 30, so the placement operand `S_col[t,q,c]` is
`[2,30,30]` = **1800 params** (slice0 = δ(q,c) identity, slice1 = δ(q,c+1) shift;
59 nonzeros). This is the cost dominator and is forced.

## Three independent from-scratch formulations attempted — all dead
1. **Sparse storage of S_col** (store the 59 nonzeros as `graph.sparse_initializer`;
   params would count NNZ=59 → whole graph ≈ 626). **Built + scored this session.**
   `score_network_v2 → None` at TWO fatal gates on onnx 1.21 / ort 1.24:
   - no value_info: `onnx.checker(full_check=True)` and `shape_inference(strict_mode=True)`
     → `Einsum: Rank of input 3 (0) does not match equation indices (3)`.
   - with value_info: `TypeInferenceError: type case mismatch existing=tensor_type
     inferred=sparse_tensor_type` (and per traps §8e a fixed value_info bills DENSE
     memory 7200 > 1800 anyway).
   ORT *runs* both (densifies) byte-equal to dense, but the scorer rejects pre-score.
   Matches traps §8e + core.md L44-45 (task337/136/042). **Sparse is triple-dead.**
2. **Structured / Kronecker factorization of the shift** (split col-30 = 6×5,
   shift = I₆⊗Pnc₅ + P₆⊗carry₅ ≈ 122 params; or CP factorization). Dead by the
   exact cost model (`neurogolf_utils.calculate_memory`): to split the col axis the
   `[1,10,30,30]` operand must be Reshaped → a charged intermediate of 9,000 elems
   (BOOL 9,000 B / F32 36,000 B) ≫ the 1800 saved. CP is separately dead: mode-1 &
   mode-2 flattening ranks are both 30 → CP rank ≥ 30 → ≥ 30·(2+30+30)=1860 > 1800.
3. **Split-branch (stay rides output-col q, only shift needs a 900 matrix)** — the
   routeA family. Additive `stay·in[q] + shift·in[c]` is a SUM; one Einsum monomial
   is a PRODUCT, so it needs 2 Einsums + Add → materialized `[1,10,30,30]` planes.
   Prior build `task004_routeA_shift900.onnx`: 100% correct, params 1402, but cost
   **73,762**. routeB (compute S_col via EyeLike) → float intermediate 7200, cost
   **26,065**. Both measured, both far above 2367.

## Residual audit (params 207 + mem 360 beyond S_col — all forced)
- `D_channel[10,10]`=100: 9 identity rows + row-0 `[1,-2,-2,…]` bg-suppression
  predicate. Full-rank; the additive bg-correction can't leave the single Einsum
  without a 2nd Einsum + Add. Not reducible below 100 (rank 10).
- `P_basis[2,30]`=60 (2⁰..2²⁹ + ones), `H_feature[10,2]`=20, `C_coef[2,2,2,2]`=16,
  shift/bias/selectors ≈ 11: all at minimal pinned shapes (col axis 30 pinned;
  dead cols 17..29 share the output/input labels, uncroppable).
- memory 360 = Bcur[1,1,30,1] F32 120 + Baug[1,2,30,1] F32 240. Row axis 30 passes
  through the output (uncroppable). F16 dead: Conv-F16-on-F32-input impossible (Cast
  [1,10,30,30]=18,000), and a post-Einsum Cast keeps the F32 output charged + adds
  the F16 tensor (net worse). Row-context via Conv (240 B) is cheaper than an Einsum
  row-shift matrix (≥900) — architecture is already optimal.

**No sub-2367 candidate exists; none staged.** Every byte is forced.

## One remaining hypothesis (for a future batch, low confidence)
If a future opset/ORT ever accepts a `sparse_initializer` through
`checker(full_check)` + `shape_inference(strict_mode)` as an Einsum operand
WITHOUT a dense value_info, sparse S_col (59 params) would drop the graph to ≈626.
Re-run `probe_sparse_scol.py` against the then-current onnx/ort before assuming.
Nothing else in the residual is shavable. Do NOT re-explore CP / Kronecker / routeA
/ canvas / GridSample — all measured or forced-component dead above.
