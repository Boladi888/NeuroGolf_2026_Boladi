# Transferable insight — a free-output Einsum that takes the F32 graph `input` as an operand is HARD dtype-locked to F32 (ORT rejects mixed F16/F32 Einsum) — the biggest narrowing prize (F16 the other operands) is unreachable

**Source**: task198 batch (`bo_ng_cld_worker2`, Claude, 2026-07-12). Audited LIVE **2898** (Sho v52
`task198_v52_opset18_padmaxpool`, ring-Conv template + one free-output Einsum, sha `ce164130…`). Floor
**re-confirmation (WALL @ 2898)**, not a new win. **Sharpens** the already-recorded aside "F32-input
locks the Einsum to F32" (see `DONE_2026-07-09T201927Z_...resize_beats_freeoutput.md` line ~34) with an
**empirical ORT rejection proof** + a concrete third task where it kills the top lever.

## The mechanism (empirically verified this session)
When the graph output is produced by a single fused Einsum `…,bkhw->bchw` whose operands include the
**F32 graph `input`** (the one-hot, used for data-dependent passthrough/remap), ALL other Einsum
operands are forced to F32 — you cannot F16-narrow the small charged operands to halve their memory.

- ONNX Einsum has a **single type parameter T** bound across all inputs + output. Minimal repro
  (`einsum_dtype_test.py`, opset 18, CPU EP):
  - `F16 × F16 → F16`: **OK**.
  - `F16 × F32 → F32`: **ORT FAIL** — `Type Error: Type parameter (T) of Optype (Einsum) bound to
    different types (tensor(float16) and tensor(float)) in node ()`.
  - `F16 × F32 → F16`: same FAIL. (onnx.checker PASSES all three — the block is an **ORT runtime**
    type-inference error, invisible to the checker. Local static `_score_v2` would also "cost" it
    happily; you only catch it at `ng_gate visible`/profile. Build+gate, don't trust the checker.)
- The graph `input` is F32 (grader feeds F32 one-hot; declaring the graph input F16 mismatches the fed
  tensor). Casting `input` F32→F16 to make the Einsum uniform materialises `[1,10,30,30]` F16 =
  **18000 B** — dwarfs any operand saving.

## Why this is the whole ballgame for template/decode tasks
The dominant charged *memory* in a downsample→expand→fold decode is the **F32 Einsum operand(s)**, e.g.
task198 `score_bucket[1,1,8,8]` F32 = **256 B** (the #1 memory item). F16 would give 128 B (+ sp_onehot
12→6) = **−134 B → 2765, a clean win over 2898**. It is unreachable purely because of the F32-input
lock above. So on any such graph:
- **You cannot split** the fold to F16-the-paint and F32-the-passthrough: splitting materialises the
  per-pixel scalar `P[1,1,H,W]` (≥1800) AND the per-channel field `W[1,10,H,W]` (≥18000). The single
  fused Einsum (output = `P[h,w]·W[c,h,w]`, both folded into the FREE named output, zero full-res
  intermediates) is uniquely optimal. Confirmed again here.
- **Params don't care about dtype**: `_score_v2` counts initializer **elements**, so F16/U8-narrowing
  the big tables (`membership_diff[3,30,8]`=720, ring kernels 810, `transform_s[10,10]`=100) saves 0.
- Net: on task198 the only ≥100 B lever (F16 the score operand) is dead, and every other tensor is
  load-bearing & min-sized → **2898 is the genuine architecture floor** (forced read 1350 + expand 720
  + remap 100 + score 256 = 2426; minimal cell-select tail 473; = 2899 local, manifest records 2898,
  i.e. 1 B *below* the champion's own `_score_v2`).

## Rule of thumb
Before estimating a F16/U8 memory-halving win on a free-output Einsum (or MatMul/Gemm) decode: check
whether the **raw F32 graph input is one of its operands**. If yes, the dtype is locked to F32 and the
narrowing is a phantom — don't stage a `floor_est` that assumes it. The narrowing is only real when the
Einsum operates purely on already-narrow/derived tensors and the input passthrough is handled elsewhere
*without* a full-res intermediate (rare — usually impossible, because avoiding the input operand forces
a materialised passthrough plane). Also: this ORT type error is checker-invisible — a mixed-dtype
Einsum only fails at visible/profile gate, so always build+gate, never trust static cost alone.
