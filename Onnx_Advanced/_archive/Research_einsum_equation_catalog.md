# einsum_equation_catalog.md

[PROMPT] Assumptions fixed by the project: ONNX graphs on onnxruntime CPU with `ORT_DISABLE_ALL`, opset ≤ 20; score = trace-visible intermediate bytes + initializer element count; the free graph input is F32 `[1,10,30,30]`; graph outputs are free. [DERIVED] Unless I explicitly say otherwise, every byte figure below is the size of an intermediate F32 tensor, so a tensor that is the final graph output contributes `0` bytes under your scorer.

## Repeated-subscript legality

[SPEC] Yes: one-operand repeated subscripts are part of the intended ONNX/NumPy Einsum semantics. [SPEC] The ONNX Einsum page says that when a dimension character is repeated on the left-hand side it denotes summation, but the same official operator docs also include an example named `einsum_batch_diagonal` with equation `...ii ->...i`, which is direct spec-level evidence for diagonal extraction from a single operand in explicit form. [SPEC] NumPy’s reference removes the ambiguity by stating that repeated labels in one operand take the diagonal, with `ii->i` behaving like `diag` and `...ii->...i` giving the broadcasted diagonal form. [DERIVED] On that basis, `ii->i`, `bcii->bci`, and `bchh->bch` are legitimate equations whenever the repeated axes have equal extent.

[DERIVED] In equations like `bchh->bch`, I am intentionally relabeling the last two spatial axes with the same symbol because your board is square. [ORT-SRC] ORT’s diagonal helper checks equality of the repeated-axis sizes; it does not care that your original semantic names were `h` and `w`.

[ORT-SRC] Yes on ORT CPU too, but the support lives one phase earlier than `EinsumTypedComputeProcessor` itself. [ORT-SRC] In `onnxruntime/core/providers/cpu/math/einsum.cc::Einsum::DeviceCompute`, ORT injects `CpuDeviceHelpers::Diagonal` into `EinsumComputePreprocessor`. [ORT-SRC] Then `onnxruntime/core/providers/cpu/math/einsum_utils/einsum_compute_preprocessor.h::EinsumComputePreprocessor::PreprocessInputs` says “Diagonal needs to be parsed along the repeated axes” and calls `device_diagonal_func_` when the same subscript label appears again in the same input. [ORT-SRC] The helper declaration in `einsum_auxiliary_ops.h` says diagonal parsing reduces rank by one and squeezes the greater repeated axis, and `onnxruntime/core/providers/cpu/math/einsum_utils/einsum_auxiliary_ops.cc::CpuDeviceHelpers::Diagonal` enforces `rank >= 2`, distinct repeated axes, and equal repeated-axis sizes before extracting the actual diagonal.

[ORT-SRC] For the specific project case `bchh->bch` on `[1,10,30,30]`, the repeated axes are already the two innermost dimensions. [ORT-SRC] ORT’s `IsTransposeRequiredForDiagonal` returns `false` when the repeated axes are the two innermost dimensions, so this common grid-diagonal read avoids even an internal transpose inside the kernel. [DERIVED] That makes `bchh->bch` about as favorable a diagonal-read form as ORT’s implementation can offer.

[SPEC] EyeLike produces a 2-D identity-shaped tensor with the same shape as its 2-D input. [DERIVED] So, on a 30×30 board, an EyeLike-based diagonal-read route materializes a `30×30` mask (900 F32 elements = 3,600 B) before the broadcast multiply.

[DERIVED] The comparison below is pure scorer arithmetic under your project rules, assuming the EyeLike route generates the eye dynamically rather than storing a constant eye initializer.

| route | graph ops | visible graph tensors | trace bytes if the diagonal tensor is an intermediate | trace bytes if the diagonal tensor is the final graph output | params |
|---|---:|---|---:|---:|---:|
| `Einsum("bchh->bch")` | 1 | `[1,10,30]` | 1,200 B | 0 B | 0 |
| `EyeLike + Mul + ReduceSum` | 3 | `EyeLike[30,30]`, `Mul[1,10,30,30]`, `Reduce[1,10,30]` | 3,600 B + 36,000 B + 1,200 B = 40,800 B | 3,600 B + 36,000 B = 39,600 B | 0 |

[DERIVED] So the repeated-subscript read is a **34×** graph-visible byte reduction when the diagonal vector is itself an intermediate, and it is effectively **free** when it is the final model output while the EyeLike route still burns `39.6 KB`.

## Diagonal and trace family

[SPEC] The right mental model is: repeated labels read equality of two axes; they do not create off-diagonal predicates by themselves. [ORT-SRC] ORT implements that by diagonalizing a repeated-label input before the later reduction/matmul stage. [SPEC] ONNX Slice allows negative `steps`, so a reverse-flip needed for anti-diagonals is legal in-spec.

[DERIVED] In the table below, every byte count assumes F32 and treats the listed result as an intermediate. [PROMPT] If the listed result is the final graph output, subtract that result tensor from the score.

| family | equation or route | output shape | trace bytes | notes |
|---|---|---|---:|---|
| main-diagonal extract | `bchh->bch` | `[1,10,30]` | 1,200 B | direct one-op diagonal read |
| main trace per color | `bchh->bc` | `[1,10]` | 40 B | diagonal read + sum in one op |
| anti-diagonal extract | reverse last spatial axis, then `bchh->bch` | `[1,10,30]` | 36,000 B + 1,200 B = 37,200 B | reverse can be negative-step `Slice` |
| anti-diagonal trace per color | reverse last spatial axis, then `bchh->bc` | `[1,10]` | 36,000 B + 40 B = 36,040 B | cheapest anti-diagonal count form |
| `k`-offset diagonal extract | slice rows/cols to a square `[30-|k|,30-|k|]`, then `bchh->bch` | `[1,10,30-|k|]` | `40(30-|k|)^2 + 40(30-|k|)` B | for `|k|<30` |
| `k`-offset trace per color | same square slice, then `bchh->bc` | `[1,10]` | `40(30-|k|)^2 + 40` B | one op per offset |
| width-`m` band sum | sum the traces for offsets `k=-m..m` | `[1,10]` | sum of the per-offset costs above | no single 1–2 operand Einsum expresses `abs(i-j)<=m` |

[ORT-SRC] The main-diagonal rows are directly backed by ORT’s repeated-axis preprocessing and diagonal helper, and the `bchh` case is especially favorable because the repeated axes are already innermost.

[DERIVED] For read-side offset diagonals, slicing to a smaller square is usually better than full-size pad/shift under your scorer, because slicing shrinks the visible intermediate from `36,000 B` to `40(30-|k|)^2 B`. [DERIVED] A full-size shift/pad keeps the whole 30×30 board alive and therefore throws away much of the benefit of the repeated-subscript read.

[ORT-SRC] Einsum does **not** give you a diagonal-write primitive. [ORT-SRC] ORT explicitly rejects repeated letters in the output subscript, so `i->ii` and `i,i->ii` are invalid. [SPEC] The valid outer-product form is `i,j->ij`; with ones vectors it produces an all-ones outer, not an identity matrix. [DERIVED] So repeated-subscript Einsum is a **diagonal READ** primitive, not a **diagonal WRITE** primitive. If you need an actual diagonal or band mask tensor, you still need EyeLike/Trilu or another explicit mask construction route.

## Two-operand workhorse catalog

[SPEC] Every equation below follows standard ONNX/NumPy Einsum semantics, including explicit-output reductions, weighted contractions, broadcasts, and outers. [ORT-SRC] Every row stays on ORT’s supported 1- or 2-operand CPU path; ORT enters the pairwise loop only once there is more than one input, and the graph-level decomposition concern starts beyond 2 operands. [PROMPT] `params` below means initializer **elements**, because that is the parameter term in your scorer. [DERIVED] `DIRECT` means “no upstream tensor-producing nodes are needed beyond the free input and any fixed initializers.”

| need | equation | output shape | trace bytes | params | caveats |
|---|---|---|---:|---:|---|
| per-color counts | `bchw->bc` | `[1,10]` | 40 B | 0 | DIRECT from the free input. |
| per-row color marginals | `bchw->bch` | `[1,10,30]` | 1,200 B | 0 | DIRECT from the free input. |
| per-column color marginals | `bchw->bcw` | `[1,10,30]` | 1,200 B | 0 | DIRECT from the free input. |
| main diagonal per-color vector | `bchh->bch` | `[1,10,30]` | 1,200 B | 0 | DIRECT on square grids; repeated-subscript diagonal read. |
| main diagonal trace per color | `bchh->bc` | `[1,10]` | 40 B | 0 | DIRECT on square grids; cheapest per-color diagonal count. |
| weighted row sums for centroid numerator | `bchw,h->bc` | `[1,10]` | 40 B | 30 | DIRECT from free input + 30-element row-coordinate initializer. |
| weighted column sums for centroid numerator | `bchw,w->bc` | `[1,10]` | 40 B | 30 | DIRECT from free input + 30-element column-coordinate initializer. |
| axis-broadcast multiply-reduce, keep columns | `bchw,h->bcw` | `[1,10,30]` | 1,200 B | 30 | DIRECT from free input + 30-element row weights; useful separable stage. |
| axis-broadcast multiply-reduce, keep rows | `bchw,w->bch` | `[1,10,30]` | 1,200 B | 30 | DIRECT from free input + 30-element column weights; useful separable stage. |
| masked color counts | `bchw,bhw->bc` | `[1,10]` | 40 B | 0 if dynamic mask / 900 if fixed `[1,30,30]` mask | DIRECT only if the mask is a constant initializer; otherwise it needs a second tensor. |
| masked full-grid selection | `bchw,bhw->bchw` | `[1,10,30,30]` | 36,000 B | 0 if dynamic mask / 900 if fixed `[1,30,30]` mask | Legal, but plain `Mul` is usually the simpler equivalent if no reduction is needed. |
| outer product for row-mask ⊗ col-mask | `bh,bw->bhw` | `[1,30,30]` | 3,600 B | 60 | Not from the free input itself; ideal for separable rectangle masks. |
| aligned color-aware template correlation | `bchw,chw->b` | `[1]` | 4 B | 9,000 | DIRECT from free input + constant template, but only for an already aligned full-grid dot, not a sliding correlation. |

[ORT-SRC] The diagonal rows are backed by the same repeated-axis preprocessing described earlier. [DERIVED] The dense-template row is score-hostile because the `9,000` initializer-element cost dominates its `4 B` output. [DERIVED] The masked full-grid row is usually a convenience form, not a score winner, because it keeps the full `36 KB` tensor alive. [DERIVED] The best score-density rows are the pure reductions and diagonal/trace reads: they emit only `40 B` to `1,200 B` with zero or tiny initializer budgets.

## Integer Einsum

[SPEC] ONNX Einsum uses one shared type variable `T` for all operands and the output. [ORT-SRC] ORT’s CPU kernel only registers `float`, `double`, `int32`, and `int64`. [PROMPT] Your free input is F32 `[1,10,30,30]`. [DERIVED] That means integer Einsum is not plug-compatible with the current input: if you want `int32` or `int64` Einsum, you must cast the free input first.

[SPEC] ONNX Cast preserves shape. [DERIVED] So a cast from your F32 input to `int32` materializes a `[1,10,30,30]` intermediate of `9,000 × 4 = 36,000 B`, and a cast to `int64` materializes `9,000 × 8 = 72,000 B`. [DERIVED] That cast cost alone is larger than most of the useful Einsum outputs in this catalog.

[DERIVED] Your `2^24` reasoning is correct. A 30×30 grid has only 900 cells. So per-color counts, row/column marginals, and diagonal traces are all at most `900`. If row/column coordinates are `0..29`, then the largest exact weighted coordinate sum is `29 × 900 = 26,100`. [DERIVED] All of those integers are far below `2^24 = 16,777,216`, so IEEE-754 `float32` represents them exactly. [DERIVED] In a centroid pipeline, the numerator and denominator Einsums are exact in F32; the only non-exact step is the later division.

[DERIVED] Prefer integer Einsum only if some downstream subgraph specifically requires integer-typed outputs and you can keep the whole path integer after a single cast. [DERIVED] `int32` gives no accuracy win over F32 for this project’s counts and moments; it only changes type. [DERIVED] `int64` is usually worse: it doubles output bytes with no numeric benefit here, turning `[1,10]` from `40 B` into `80 B` and `[1,10,30]` from `1,200 B` into `2,400 B`. [SPECULATIVE] On pure score grounds for this puzzle setting, `int64` Einsum is almost always wasted budget.

## Failure modes

**Output repetition.** [ORT-SRC] ORT explicitly rejects repeated output labels with the error “Output subscript contains repeated letters.” [DERIVED] So `i->ii`, `i,i->ii`, and any other repeated-output-label form are out.

**Square-only repeated axes in practice.** [ORT-SRC] ORT’s subscript processing keeps a broadcast-style global dimension map and can temporarily tolerate a repeated label when one occurrence has size `1`. [ORT-SRC] But the actual diagonal helper later enforces exact equality of the two repeated-axis sizes before extracting the diagonal. [DERIVED] So one-operand repeats are effectively **square-only at execution time**, which is fine for your 30×30 board but a real trap on rectangular inputs.

**Repeated subscripts on both operands.** [ORT-SRC] ORT preprocesses each input independently in `EinsumComputePreprocessor::PreprocessInputs`, diagonalizing repeated labels before later contraction. [DERIVED] That strongly suggests forms like `ii,jj->ij` and `ii,ii->i` should work as long as each repeated pair is square and the final output labels are unique. [SPECULATIVE] I would still treat those “repeat-on-both-sides” forms as source-supported rather than grader-proven, because they are less common than the straight one-input diagonal cases.

**Non-innermost diagonals may be slower.** [ORT-SRC] ORT avoids an internal transpose only when the repeated axes are already the two innermost dimensions; otherwise `CpuDeviceHelpers::Diagonal` may transpose first and then extract the diagonal. [DERIVED] That internal transpose is invisible to your graph-byte scorer because it stays inside the kernel, but it still matters for runtime.

**Portability and dtype limits.** [SPEC] ONNX’s published operator docs describe lower-case labels and allow a much broader numeric type set than ORT’s CPU kernel does. [ORT-SRC] ORT’s parser accepts both lower- and upper-case letters, and its CPU kernel only supports `float`, `double`, `int32`, and `int64`. [DERIVED] So equations using uppercase may work on ORT but are less portable to the published ONNX wording, and equations using `float16`, `int8`, or unsigned integer Einsum are spec-level ideas that this CPU kernel does not implement.

**Pairwise decomposition beyond two operands.** [ORT-SRC] `EinsumTypedComputeProcessor::Run` loops through extra inputs and calls `PairwiseOperandProcess` for each pair, while `PairwiseOperandProcess` uses permutations, `ReduceSum`, and `MatMul`-style reshaping internally. [ORT-DOC] ORT issue #5897 documented a CPU case where the transposes in such a decomposition dominated runtime cost. [DERIVED] That is exactly why this catalog stays disciplined about 1- and 2-operand forms.

**Historical odd-shape bug.** [ORT-DOC] A released ORT 1.16.1 package broke for the simple one-input equation `ijk->ki`, failing with a `DataCopy` size-mismatch at finalization; that issue was later closed. [DERIVED] The practical lesson is that conventional reductions, marginals, diagonals, and simple 2-input contractions look safe from current source, but unusual transpose/reduce shapes deserve caution on whatever pinned grader build you actually ship against.

## Blunt verdict

[DERIVED] The equations worth adopting immediately are `bchw->bc`, `bchw->bch`, `bchw->bcw`, `bchh->bch`, `bchh->bc`, `bchw,h->bc`, `bchw,w->bc`, and `bh,bw->bhw`; after that, add `bchw,bhw->bc` whenever a mask already exists. [SPEC][ORT-SRC] The repeated-subscript diagonal is **real** in both the ONNX semantics and the ORT CPU implementation, and for your scorer it turns a main-diagonal read from roughly `39.6–40.8 KB` of graph-visible work in an EyeLike+Mul+Reduce route into either `1.2 KB` or effectively `0 B` if that diagonal is the final output. [DERIVED] So the blunt answer is yes: repeated-subscript Einsum should replace EyeLike/Trilu for **diagonal reads and traces** on square grids. [DERIVED] Keep EyeLike/Trilu only for **diagonal or band writes**, explicit mask tensors, and anti-/offset diagonals that still need a flip or slice first. 