# Research Card M3 — Minimum-cost exact ONNX realization of a fixed integer linear map y = Wx

**Deliverable**: a single copyable Markdown research note (no file writes, no code execution). Suggested save name: `m3_min_cost_integer_linear_map.md`.

## Researcher contract

Internet-research assistant, web access, NO repo access. ONE self-contained Markdown note, no questions, assumptions inline. Source-tag every claim; human-readable citations (ONNX op spec pages, linear-algebra references). Short numpy/ONNX-builder snippets allowed for recommended patterns only.

## Why this card exists (the gap)

My current optimization frontier is large old "blend" graphs (cost 15,000–35,000) that implement, at heart, a **fixed integer linear map** `y = W x` (W known at build time, integer/{0,1}). The cost is dominated by how that map is realized in ONNX. I need a decision procedure: **given a fixed integer W ∈ ℤ^{M×N} (and its structure), which ONNX realization minimizes the element/byte cost** under the model below — plain MatMul/Gemm, a (grouped/1×1) Conv, a Gather-LUT, a factored two-stage map, or a structural/zero-param generator?

This is distinct from card L1/M2 (which decompose 3-D *selector banks* by CP rank). M3 is about the **2-D linear map op-choice and structure-exploitation** for the big-fish tier.

## Cost model (memorize — this drives every recommendation)

```
Score = intermediate_memory_bytes + param_elements   (LOWER better)
- param_elements = element count of all initializer/Constant tensors (unique). DTYPE-BLIND:
  a [M,N] weight costs M*N regardless of int8/f16/f32. Narrow dtype does NOT cut params.
- intermediate_memory_bytes = byte sizes (elements x dtype_size) of materialized value_info
  between input and output. Graph input AND graph output tensors are FREE (excluded).
- A node whose output is the graph output is free. ORT kernel-internal temporaries are not
  counted (only declared value_info/outputs).
- Ignore ORT fusion unless documented.
```

Key consequences to reason from: a dense `MatMul` weight costs `M*N` params (e.g. a 30×30→30×30 map = 810,000 — catastrophic). A `Conv` exploits translation invariance: a `[Co,Ci,kh,kw]` kernel costs only `Co*Ci*kh*kw` (e.g. 10×10×3×3=900) — the win is when W is a *convolution* (shared local stencil), not an arbitrary dense map. A `Gather`/LUT costs the table size. So the realization cost depends entirely on W's STRUCTURE.

## Environment & constraints (fixed)

```
- ONNX opset <=18 target on ONNX Runtime 1.24.4 CPU EP, static shapes, NCHW, [1,C,H,W], C<=10, H,W<=30.
- Integers are exact in f32 up to 2^24 and in int32 natively; uint8 exact to 255.
- Banned: Loop, Scan, NonZero, Unique, Compress, FunctionProto/If.
- params dtype-blind (element count); intermediate memory dtype-aware (bytes).
- check_model passing != has-CPU-kernel; flag any kernel-absent op.
```

## Required answer order

1. **Op-cost formulas**: for each realization of `y=Wx` — Gemm/MatMul, 1×1 Conv, k×k Conv (+ groups), Gather/ScatterND LUT, two-stage factored `W=AB` — give the exact param + intermediate-memory cost as a formula in (M, N, structure params).
2. **Structure → cheapest-op decision table**: for each structural class of W — dense/unstructured, low-rank (rank r), separable/Toeplitz/convolutional, permutation, sparse (nnz), block-diagonal, circulant — state the minimum-cost realization and its cost, with the kill-number for the alternatives.
3. **The decision procedure**: given an arbitrary integer W, the ordered tests to run (rank? sparsity? shift-structure? block structure?) to pick the cheapest realization, and how to verify exactness.
4. **Worked examples**: 3 concrete W's (a 900×900 dense-looking map that is actually a 3×3 convolution; a permutation; a rank-2 map) showing the cost collapse.

## Required table schema

`| W structure | detection test | cheapest ONNX op | param cost | intermediate cost | kill-number vs dense |`

## Source-tag standard

`[PROMPT]` · `[PRIMARY]` · `[SECONDARY]` · `[DERIVED]` · `[SPECULATIVE]`.

## Mandatory caveats

- Conv only beats dense when W is genuinely a shared-stencil convolution (or block-Toeplitz) — quantify when the kernel-size/groups tradeoff stops paying.
- Factoring `W=AB` (M×N → M×r + r×N) helps params only if `r(M+N) < MN` AND the intermediate `[1, r, H, W]` byte cost doesn't eat the param saving (the free-output-separability trap: a final-stage factorization materializes a charged middle tensor). State the crossover.
- params dtype-blind: never recommend "use int8 weights to cut params."
- Exactness: every realization must reproduce Wx exactly under f32≤2^24 / int32; flag any that risks rounding (e.g. f16 accumulation in native Gemm rounds every partial).

## Verification checklist (end with this)

- [ ] Cost formulas given for all 5 realization families.
- [ ] Structure→op decision table with detection tests + kill-numbers.
- [ ] Factorization crossover (param saving vs charged intermediate) stated explicitly.
- [ ] dtype-blind-params and f32≤2^24 exactness caveats applied.
- [ ] **One-paragraph blunt verdict**: the ordered decision procedure a builder should run on any fixed integer W to pick the minimum-cost ONNX realization in under 5 minutes.
