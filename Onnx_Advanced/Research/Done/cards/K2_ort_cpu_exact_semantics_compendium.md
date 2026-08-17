# Card K2 — ORT 1.24 CPU exact-semantics compendium (ties, rounding, NaN, integer ops)

**Deliverable**: a single copyable Markdown research note (no file writes, no code execution). Suggested save name: `k2_ort_exact_semantics.md`.

**Researcher contract**: internet access, NO repo access; everything needed is in this card. Cite ONNX operator spec versions AND onnxruntime source (`file::function` or kernel class names, browsable at github.com/microsoft/onnxruntime, tag v1.24.x — if that tag is not browsable, use the nearest available 1.2x tag and SAY SO). Tag every claim `[PROMPT]`/`[PRIMARY]`/`[SECONDARY]`/`[DERIVED]`/`[SPECULATIVE]`.

## Environment (all facts [PROMPT])

- onnxruntime **1.24.4**, CPU execution provider only, graph optimizations DISABLED (`ORT_DISABLE_ALL`), onnx 1.21.0, opsets 10–18 default domain.
- Our models must reproduce a reference output **bit-exactly after binarization** (`output > 0.0` vs expected boolean grid). A single divergent cell on a single input = that task scores zero. So we care about EXACT kernel semantics: tie-breaking, rounding direction, NaN propagation, integer overflow/sign behavior — not approximate numerics.
- Tensors are integer-valued floats (one-hot 0.0/1.0 planes and small-integer intermediates), plus bool/uint8/int32/int64/float16 dtypes.
- Why this card: our internal notes document some behaviors empirically (listed below as claims TO VERIFY); we need an authoritative, source-cited compendium, including where ORT's CPU kernels deviate from the ONNX spec.

## Claims to VERIFY or REFUTE against source (each gets a verdict + citation)

1. `ArgMax` (opset 13) ties: default returns the FIRST index; `select_last_index=1` flips to last. True in the CPU kernel?
2. `TopK` with `sorted=1`: ties broken by lower original index first (stable). Spec guarantee or ORT implementation detail? Is `sorted=0` order deterministic in practice on CPU?
3. ONNX `Pad` with negative pads (crop behavior): supported by the CPU kernel across modes? Which opsets?
4. `Where(cond, X, Y)` with X,Y of type BOOL: rejected by ORT CPU (no kernel). Confirm and list which dtype combinations ARE registered.

## Required answer order

1. **Selection & ordering ops** — ArgMax/ArgMin (`select_last_index` per opset), TopK (tie rule, `sorted` attr, `largest` attr), Hardmax tie rule, Max/Min elementwise NaN handling per opset (the opset-13 change), Clip NaN behavior.
2. **Rounding & conversion** — `Cast`: float32→float16 rounding mode (round-to-nearest-even?), float→int rounding (truncation toward zero?), out-of-range/saturation behavior per target dtype incl. uint8, float16→float32 exactness; `Resize`: exact semantics of `nearest_mode` values (`round_prefer_floor` default etc.) and `coordinate_transformation_mode` values at opsets 11/13/18, with the formula for each; `QuantizeLinear`/`DequantizeLinear` rounding and clamping.
3. **NaN/Inf propagation** — comparisons (Greater/Less/Equal) with NaN; Min/Max NaN per opset; ReduceMax/ReduceMin with NaN; whether `Equal(+0.0, −0.0)` is true; float16 subnormal handling in comparisons on CPU.
4. **Integer & modular ops** — `Mod` with `fmod=0/1` sign-of-result rules for negative operands; integer `Div` truncation direction; integer overflow behavior (int32 wraparound? UB?); `Pow` with integer exponents; negative-step `Slice` exact index semantics; `CumSum` `exclusive`/`reverse` exactness.
5. **Spec-vs-ORT deviation list** — every place the ORT 1.24 CPU kernel is known to differ from the ONNX operator spec or to be under-specified (consult ORT GitHub issues, the operator-kernels documentation, kernel source). This is the highest-value section.
6. **Determinism** — for single-threaded CPU inference with optimizations off: are kernel results run-to-run deterministic? Any ops with documented nondeterminism or parallelism-dependent reduction order even at intra_op_num_threads=1?
7. **Overall verdict** — one paragraph: which 3 behaviors are most likely to silently break a bit-exactness contract, ranked.

## Required table schema (used in sections 1–4)

| op (opset) | attribute & default | exact CPU behavior | spec source | ORT source citation | deviation? | bit-exactness implication |

## Demands

- Every behavioral claim must cite EITHER the ONNX spec section + an ORT conformance statement, OR the ORT kernel source location. Pure training-data recall without a citation is not acceptable — mark anything you cannot source as `[SPECULATIVE]`.
- Counterexample-or-proof: for each tie/rounding rule, give a 1-line concrete example (inputs → output index/value).
- No code except ≤10-line illustrative snippets.

## Final verification checklist

- [ ] All 4 VERIFY claims have explicit verdicts with citations.
- [ ] Resize nearest/coordinate modes covered with formulas, not prose.
- [ ] Deviation list (section 5) present, even if it concludes "none found — searched issues X, Y, Z".
- [ ] Every table row has a source citation a human can find without link tokens.
- [ ] Verdict paragraph ranks the top-3 silent-divergence risks.
