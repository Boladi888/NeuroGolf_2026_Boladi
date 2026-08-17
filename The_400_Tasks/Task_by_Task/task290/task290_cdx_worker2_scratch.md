# task290 cdx_worker2 scratch

Bar: current team-best from shared workbook/manifest is cost 222, SHA-256
5fed3434f485dc5f2c65843c938d93b364e044b5983203ef7f8a8a5ef39ba5ef
(memory 176, params 46). Stage only strict-below 222.

Decoded rule: find the single nonzero square/rectangular object, crop to its
bbox, and swap the two nonzero colors. Background stays absent. Official/ARC-GEN
examples in this task emit 3x3 through 6x6 square crops.

Output-size relation: the existing 222 graph counts the two non-background
colors with an Einsum, sums the two top counts to area, takes sqrt(area), and
uses size-3 as an index into packed 3..6 spatial masks. This assumes generated
objects are square, matching visible and ARC-GEN preview samples.

Color roles: TopK returns the two present nonzero color channels; ScatterElements
writes two role weights into a zero color-weight vector, so the later QLinearConv
emits the swapped channels.

Falsification cases:
- Non-square object area would make sqrt(area) non-integral and break packed
  size lookup.
- More than two nonzero colors would make TopK role assignment ambiguous.
- Output size outside 3..6 would index outside packed_spatial.

Candidate families:
- Micro-shave current 222 graph: remove any initializer/input that is redundant
  under opset-14 semantics. First target is `reduce_axes` on ReduceSum, because
  omitting axes reduces all axes and should produce the same scalar `area`.
- Rewrite family: replace TopK/Scatter role assignment with arithmetic color
  formulas, but rough cost is likely worse because it needs extra F32/INT64
  intermediates or channel weights. Only pursue if the micro-shave fails.

Charged tensor/op to delete first: initializer `reduce_axes` [3] (3 params) from
the `ReduceSum` node. Expected score cost if valid: 219.
