# Term-axis-last layout removes reshape scaffolding in free-output factor graphs

## What I discovered

In task092, the deployed 4,033 graph was already a free-output rank-term Einsum:

`Einsum(row_comp, coeff_comp, col_comp) -> output`

The apparent floor still had small charged scaffolding tensors because the span factor operands were laid out as `[1,5,30]` / `[1,6,30]`. The graph first reshaped gathered moment vectors from `[1,1,5]` to `[1,5,1]`, built span masks against `[1,1,30]` coordinate grids, then reshaped `is_v` back to `[1,1,5]` for coefficient broadcasting.

Keeping the term/color axis last avoids that whole scaffold:

`old: row_comp/col_comp [1,6,30], final "bir,bci,bik->bcrk"`

`new: row_comp/col_comp [1,30,6], final "bri,bci,bki->bcrk"`

The moment tensors remain `[1,1,5]`, coordinate grids become `[1,30,1]`, validity is unsqueezed to `[1,30,1]`, and the final Einsum contracts the last axis as the term index.

## Where I encountered it

- Task: task092
- Builder: `The_400_Tasks/Task_by_Task/task092/prototypes/work_task092_w19/build_task092_term_axis_last.py`
- Staged candidate: `staged_submits/task092_term_axis_last_cost3980/`
- Base: `staged_submits/task092_double_span_no_half_cost4033/task092.onnx`

## Why it matters

This was a strict, gate-clean improvement from 4,033 to 3,980:

- deleted four `F16[1,5,1]` moment reshapes: 40B
- deleted one `BOOL[1,1,5]` `is_v` reshape: 5B
- removed three 3-element shape initializers and added one 1-element axes initializer: net -8 params
- total: -53B

Gates on the staged candidate:

- visible: 265/265
- truth full: 200000/200000
- AB full: 200265/200265, byte_equal=True
- score_v2: memory 3823 + params 157 = 3980
- real-input profile (`--arcgen 266`): static=3980 profile=3980

This does not remove any forced span/validity factors; it removes only formulation-specific shape scaffolding. It is most useful for #109/#119-style output builders where a runtime-computed term axis can be placed on either side of a spatial axis.

## What's missing from the library

`techniques.md` #109/#119 explain folding factors into a final free-output Einsum, operand order, and selector costs. I did not find a note about choosing operand axis layout to preserve already-available singleton dimensions and avoid `Reshape`/shape-initializer scaffolding before the final contraction.

## Sorter note: techniques.md #119 — extend existing "deep separable output" section with a micro-idiom about term-axis layout / singleton-preserving operand shapes
