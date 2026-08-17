# task134 marginal-crop floor — superseded tail note

**2026-07-13 bo_ng_worker1 update:** the 890 wall below remains valid for the
plain marginal-crop `Where -> Pad` tail, but it missed a free-output tail
rewrite. `staged_candidates/task134_QLinearCarrierCollapse_cost863/` replaces
the charged 3x3 one-hot `combined` tensor plus final `Pad` with a small U8
`QLinearConv` output placer. Cost/profile is **863**, full A/B is byte-equal to
LIVE, and status is **PROBE_READY** because `ng_safety` warns on QLinearConv.

# historical record: task134 marginal-crop floor — WALL at cost 890 (batch50, bo_ng_cld_worker2)

## Bar used
- Authoritative bar: **cost 891** (`tools/ng_current_bar.py --task 134` → manifest, sha `1913cef3e00d…`, owner Aditya).
- Lower local staged rung: **cost 890** (`staged_candidates/task134_padaxes_cost890`, pad-axes micro-shave, byte-equal to LIVE).
- Prompt floor_est: ~400 (STRUCTURAL). **This is INFEASIBLE — see forced-component proof.**

## Rule (verified 266/266 on train+test+arc-gen)
Two non-zero colors: one compact "shape", one scattered "dots".
1. dense (shape) color = **max fill-ratio = min column-variance**. Selector `score[c]=N²·(1−Var_col(c)/32)`, argmax → dense. (Dots spread in columns → low score.)
2. dot color = the other present color (exclude background ch0).
3. dense bbox is **always square** (0/266 non-square).
4. output = 3×3 strided sample of the dense mask over its bbox (stride K=side/3), rendered in the dot color, one-hot into `(1,10,30,30)` uint8 at top-left; **empty cells MUST carry ch0=1** (grader requires full one-hot — argmax-only output fails visible 0/266).

## What the prompt got wrong
Prompt assumed LIVE reads the bbox via "4× ArgMax over 30×30 + A_f/B_f[2,30] planes" and that swapping to "2 row/col marginals + Gather crop" would drop 891→~400. **LIVE ALREADY uses that formulation**: 2 marginal Einsums (`bchw,bc->bh` / `->bw`) + a strided `Slice` 3×3 crop + a rank-2 variance `Einsum` selector. The "new formulation" is the existing one. So there is no 2× structural gap to capture.

## Independent from-scratch rebuild → same floor
`prototypes/build_marginalcrop.py` builds the formulation from scratch (no copy of LIVE nodes; reuses only the proven A_f/B_f selector basis).
- `task134_marg_ch0.onnx`: **cost 890** (memory 743 + params 147). visible 266/266; profile static=890 profile=890 (no gap); truth 2995/3000 (99.83%, matches LIVE's own ~99.86% ambiguity); **ab byte-equal vs LIVE 2266/2266 divergences=0**. sha256 `965b0c43…`.
- Reproducing 890 byte-equal from an independent construction is direct evidence 890 is the representation floor, not a formulation artifact.

## Forced-component floor proof (why memory ≥ ~743, params ≥ ~147)
Every intermediate tensor is charged (`max(static,runtime)` bytes); `input`/`output` are FREE.
- **2 F32 marginals** `(1,30)` = 2×120 = **240 B**: Einsum outputs are F32; needed for bbox first/last-present.
- **uint8 occupancy** for both marginals (bool+uint8 per marginal) = 4×30 = **120 B**: `ArgMax` rejects BOOL (hard record `bool_argmax.md`); the uint8 hop is mandatory to get first/last-present index. (ReduceMin-of-masked-index / CumSum alternatives add F32 `(1,30)` tensors → net loss.)
- **10-channel 3×3 output one-hot** `(1,10,3,3)` uint8 = **90 B**: forced for a 10-channel one-hot; padding a `(1,1,30,30)` plane instead is 900 B (worse).
- selector/present `(1,10)` F32 + sampled `(1,1,3,3)` F32 + int32 bbox arithmetic + concats ≈ **~290 B**.
- **params: A_f/B_f `[2,30]` = 120** is the minimal spread-based selector. Column-variance needs Σw AND Σw² ⇒ inherently **rank-2**; a rank-1 form is translation-sensitive (position varies) and fails. Moving the two moments out of the Einsum into separate weighted `ReduceSum`s cuts 60 params but adds ~5 `(1,10)` F32 moment/arithmetic tensors ≈ +240 B → **net +180**. So packing them in the `[2,30]` Einsum (one `(1,10)` output) is Pareto-optimal.
- Any per-color spatial feature (per-color marginal / erosion / conv) materialises a `(1,10,30,·)` tensor ≥ 1,200 B and blows the budget — the scalar Einsum selector is the only cheap path.

∴ memory ≈ 743, params ≈ 147 ⇒ **cost floor ≈ 890.** ~400 is unreachable in this or any nearby representation.

## Remaining lever (epsilon only — NOT staged, per anti-epsilon directive)
`task134_marg_ch0_dedup.onnx` shares one int64 `[2,3]` initializer for both `Unsqueeze` axes and `Pad` axes → **cost 888** (743+145), visible 266/266, byte-equal to LIVE. This is a 2-byte initializer-dedup (applies equally to the staged 890 pad_axes), not a structural path to the floor. Flagged for the captain if every byte counts; deliberately NOT staged (prompt: "do not grind an epsilon shave"; the 890 rung is already held byte-equal).

## Verdict
**WALL.** The marginal-crop formulation floors at cost 890 (byte-equal to LIVE). ~400 floor_est is infeasible (forced-component proof above). Do not re-explore marginal/crop variants for a 2× win. Only a fundamentally new charged-tensor model (not evident) could go lower; the sole sub-890 move is a 2-byte initializer dedup (epsilon).
