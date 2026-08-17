# Reviewer locked fixed-shift certificate

Live/team-best bar: cost 122, SHA-256 `61c58f8fbb942e5a13ed755b3fdc0f7e68c58c405f091240a235d4b0a85359be`, from `python tools/ng_current_bar.py --task 261 --json` / manifest.

Reviewer hypothesis tested: direct fixed down-1 shift plus `8 -> 2` recolor, not another polynomial-factor shave.

Variant 1: `prototypes/task261_literal_slice_pad.onnx`
- Graph: `Slice` channel 8 rows 0..28, `Pad` down by one row, `ReduceMax` active-grid mask, `Mul` to suppress bottom-row spill into padding, `Sub` to reconstruct channel 0, `Concat` to emit full 10-channel one-hot.
- Cost: `(0, 21480, 17)`, total `21497`.
- Priced math: `slice8 1*1*29*30*4 = 3480`, `shifted8 = 3600`, `active = 3600`, `fg = 3600`, `bg = 3600`, `zero = 3600`, memory `21480`; params `slice_starts 4 + slice_ends 4 + pad_pads 8 + pad_value 1 = 17`.
- Gate: `[visible] 265/265 (100.00%)  PASS`.
- Verdict: correct formulation, not a candidate; `21497 - 122 = +21375`.

Variant 2: `prototypes/task261_dense_fixed_shift_final_einsum.onnx`
- Graph: one final `Einsum('njrc,nisc,it,trs,o->norc')` with a dense fixed row table. This uses #67/#89 final-output accounting but no six-factor current polynomial.
- Cost: `(0, 0, 1830)`, total `1830`.
- Priced math: `row_term [2,30,30] = 1800`, `input_term [10,2] = 20`, `output_sign [10] = 10`; output is the only node result, so memory is `0`.
- Gate: `[visible] 265/265 (100.00%)  PASS`.
- Verdict: correct fixed row relation, not a candidate; `1830 - 122 = +1708`.

Route citations used:
- `Onnx_Advanced/techniques.md #65`: Where beats Equal only when passthrough exists. This task is not a clean passthrough because original 8s must be erased and shifted cells painted.
- `Onnx_Advanced/techniques.md #67`: output node is uncharged; useful for the final `Concat`/`Einsum`, but variant 1 still pays all pre-output planes.
- `Onnx_Advanced/techniques.md #89`: final `Einsum` in-kernel temporaries are scorer-invisible; variant 2 confirms the dense fixed relation then loses on params.
- `Onnx_Advanced/by_symptom.md` dense one-hot overlay route: `Where(cond, fill, input)` is not applicable as a win here because the input cannot be the unchanged branch for a move/recolor rule.

Remaining hypothesis: beating 122 requires a more compact dense/separable final-output encoding than a literal row table, or a different way to create a row-local negative baseline without charged Slice/Pad planes. The reviewer-locked direct fixed-shift forms are priced out.
