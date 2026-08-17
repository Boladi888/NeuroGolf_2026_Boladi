# task399 hard-record — perfect-25 rumor DISPROVEN + cost-43 is the forced floor

**Live bar:** cost **43** (manifest, owner Vamshi) · sha256 `10881f0db7039e00485832855525080264e0f40c9a419819b16fc35af70e7c5f`
· profile_real_cost = 43 (static==profile, no dynamic gap) · visible 273/273.
Recorded 2026-07-07 by `bo_ng_cld_worker2`.

## Task is content-dependent counting, NOT a shape-only transform
Outputs are ALWAYS 3×3; inputs vary (3×3, 5×5, 6×6, 7×7). Rule = count the N solid 2×2
red (color-2) blocks and render N as a dice-pip pattern on a fixed 3×3:
pip cell order (0,0),(0,2),(1,1),(2,0),(2,2); cell k lights iff N≥k (verified N=1..5 across
all 8 train + 3 test pairs). This is a data-dependent count→render, so:

### DEAD ROUTE 1 — perfect-25 / cost-≤1 zero-param family (the prompt's premise)
Identity / all Transpose perms / hflip / vflip / rot90/180/270 checked against every
supplied pair: **0 matches** (shapes don't even align: variable N×N → fixed 3×3).
A no-payload direct-output op cannot map a variable-size input to a content-dependent
3×3 count. The "perfect 25, forced floor ≤1" framing is a rumor and is disproven — do NOT
re-run a one-node sweep here.

## The cost-43 graph (current best) and why every initializer is at its floor
`Einsum(nchw,c->n)` with one-hot@2 selector → 4N scalar; `Cast`→i8; `Sub` odd-threshold
patch `[3,127,7,127,11,127,15,127,19]`; `ConvInteger` weight `[10,1,1,1]=[-1,1,0…]`
(free `pads` attr places the 3×3 at top-left of 30×30, expands to 10-ch one-hot).
Cost 43 = memory 14 (red_f32 4 + red_i8 1 + signed_patch 9) + params 29
(red_selector 10 + thresholds 9 + channel_weights 10).

- **red_selector 10 — FORCED.** Einsum contracts the length-10 channel axis, so the
  selector operand must have 10 elements. Every cheaper selector materialises a plane/
  vector intermediate that costs MORE memory than the params saved: `Einsum(nchw->nc)`+
  `Gather` = 1 param but a [1,10] f32 intermediate (+40 B) ⇒ net +31; `Gather`/`Slice` a
  [1,30,30] plane = +3600 B. Scalar-reduce with a 10-vector is strictly optimal.
- **thresholds 9 — FORCED.** Elementwise `Sub` over the 3×3 output geometry needs a value
  at all 9 positions (5 pip thresholds + 4 blocker cells at 127). Not separable/broadcastable
  (X-pattern values vs +-pattern blockers). Odd thresholds are load-bearing: 4N is always
  even, so `4N−thr` is never 0 → the decoder's "exactly one channel >0" rule always holds.
- **channel_weights 10 — FORCED by the local visible gate.** `mode_visible` compares the
  raw output byte-for-byte to `encode_grid(expected)` = **[1,10,30,30]**. Output must be
  exactly 10 channels → 1×1 ConvInteger weight is `[M=10,1,1,1]` = 10 params (zeros still
  count via `math.prod(dims)`).

### DEAD ROUTE 2 — 2-channel output (drop channel_weights 10→2, would be cost 35)
Prototype `task399_ceiling_2ch.onnx` (sha `10363cbb…`) scores 35 and DECODES correctly
(`decode_to_grid`/official `convert_from_numpy` read channel count dynamically) — but
`ng_gate visible` = **0/273** because the gate does `np.array_equal(out, [1,10,30,30])`;
a [1,2,30,30] output shape-mismatches. Local gate is the staging ground truth → 2-ch is
UNSTAGEABLE. Do not resurrect unless the visible gate's comparison changes.

### DEAD ROUTE 3 — sparse_initializer to make channel_weights cost only nnz (=2)
`calculate_params` counts `math.prod(sparse.values.dims)` = nnz, so a sparse [10,1,1,1]
with 2 nonzeros would score as 2. BUT: (a) `onnx.checker.check_model(full_check=True)`
— which `calculate_memory` runs — raises `TypeInferenceError` on ConvInteger with a sparse
weight input ⇒ score returns None; (b) ORT loads it but applies no weight (output all-zeros);
(c) `sanitize_model` renames node inputs but not `sparse_initializer` names ⇒ dangling ref.
Triple-dead. Reconstruction `task399_ceiling_10ch.onnx` (sha `ea4eb06d…`) reproduces 43
exactly and passes visible 273/273 (proves the rebuild is faithful).

## Forced-floor lower bound (argument, not "tried hard")
Any correct graph must: (1) render a 10-channel one-hot output ⇒ ≥10 params for the 1×1
channel expansion; (2) encode the 3×3 pip geometry ⇒ ≥9 params; (3) reduce channel-2 to a
count ⇒ ≥10 params (10-vector selector) OR pay >10 B of intermediate memory. Min ≈ 29
params + ~11 B memory ≈ 40; the achieved 43 (memory 14) is within the irreducible
red_f32(4)+cast(1) overhead of the arithmetic count path. **43 is the forced floor. WALL.**

## One remaining hypothesis (low confidence)
If a future ORT/opset exposes a channel-expansion op whose weight is an *attribute* (0-param)
or a 0-param broadcast that emits 10 channels with only channels 0/1 populated, the
channel_weights 10 could drop toward 2 → cost ~35. No such op exists at opset 14 today.
