# ch0 penalty elimination via Slice(ch1-9) + area formula savings (task291, 2026-06-12)

## What I discovered

For hollow-box detection (and any multi-channel argmax where ch0 is always background), the
standard `ch0_pen = [-1e9, 0, ..., 0]` parameter (10 params = 10 units) can be replaced by
a Slice operation that excludes ch0 entirely. This saved 12-15 units net.

**Three independent param savings combined for -15 units (1756 → 1741):**

1. **ch0 Slice trick** (-12 units net):
   - Instead of penalizing ch0 with `hollow + ch0_pen[-1e9,0,...,0]` (10 params),
   - Use `Slice(hollow, starts=[1], ends=[10], axes=[1])` → [1,9,1,1] F32 = 36 B
   - Then ReduceMax + Equal over ch1-9 only, Pad output with C_begin=1
   - New pads=[0,1,0,0,0,0,29,29] (same 8 params), removes ch0_pen (10 params), adds 3 Slice params
   - Net: -10 + 3 = -7 params; -40 (score intermediate) + 36 (hollow_9) - 1 (oh shrinks 10→9 B) = -5 B memory
   - CORRECTNESS: Among ch1-9, the hollow box has hollow > 0 and all solids have hollow = 0,
     so argmax over ch1-9 correctly identifies the hollow box. ch0 (background) may have hollow > 0
     but is excluded by the Slice.

2. **Area formula without one_f** (-1 param):
   - Instead of: `hm1_f * width → hm1_times_w; hm1_f + one_f → height_f; width * height_f → area`
   - Use: `hm1_f * width → hm1_times_w; hm1_times_w + width → area`
   - Because: `width * height = width * (height - 1) + width = hm1_times_w + width`
   - Eliminates the `one_f` float scalar param (1 param). Same memory cost.

3. **Shared initializer for Slice starts/axes** (-1 param):
   - `Slice(hollow, starts=[1], ends=[10], axes=[1])` uses starts=[1] and axes=[1].
   - Both can share the SAME `one_i64=[1]` INT64 initializer, saving 1 param.
   - Verified: ORT correctly interprets the same tensor for both inputs.

4. **Pad without constant_value** (-1 param):
   - `Pad(oh, pads)` without a 3rd input `constant_value` defaults to BOOL False (0).
   - Eliminates the `pad_false = [False]` initializer (1 param). Verified correct.

## Transfer conditions

- **ch0 Slice trick**: applies whenever ch0 (background channel) is always disqualified from
  a per-channel argmax/one-hot. Instead of penalizing ch0, just slice it out and Pad it back
  in the output with C_begin=1. Requires C=10 total channels and 9 valid channels.
  
- **Area without one_f**: applies to ANY computation of `hm1 * scale + scale` where `scale`
  is already computed. Algebraic identity: `hm1 * w + w = (hm1 + 1) * w = h * w`.
  
- **Pad default fill**: Pad(data, pads) without constant_value fills with dtype-zero (0/False).
  Verified for BOOL at opset 13+. No need for a scalar `pad_false` init. Saves 1 param whenever
  you would use pad_value=0 or pad_value=False.

- **Shared [1] tensor**: any graph with multiple ops that independently need `[1]` (INT64) as input
  can share a single `one_i64` initializer across all uses. Commonly: Slice starts, Slice axes,
  Unsqueeze axes, ReduceSum axes (at opset 13+).

## Dead ends also confirmed

- Can't use opset-12 attribute-based ReduceSum (0 params for axes) in the same model as opset-13
  BOOL Pad: they require different minimum opsets. Dead end.
- Can't eliminate ax_w or ax_h (each 1 param) since ReduceSum at opset 17 requires axes as INPUT.
  Einsum can't produce keepdims=1 outputs, so can't cleanly replace ReduceSum with Einsum for rowsum.
- ConstantOfShape for pads: 4 params + 112 B memory intermediates vs 8 params + 0 B. Worse.
- INT32 ArgMax: not possible (ArgMax always outputs INT64 per ONNX spec).

## Where encountered

- Task 291 (HollowBox), 2026-06-12
- Builder: `The_400_Tasks/Task_by_Task/task291/prototypes/build_task291_slice9_v3.py`
- Staged: `staged_submits/task291_hollowbox_cost1741/task291.onnx`
- Final cost: 1741 (beat team-best 1756 by 15 units)
- All gates: visible 265/265, truth_full 200000/200000, ab_full 200265/200265 (byte-equal)
- Profile cost matches static: 1741

## Library location

The ch0 Slice trick and shared-[1] pattern should be added to techniques.md (new § or extension
of §67 FREE-OUTPUT catalog). The "Pad without constant_value" should go into traps/techniques
as a free param savings note. The area formula (hm1*w+w) is a local algebraic trick not worth
a dedicated §, but should note the general principle of eliminating +1 scalars by rewriting
the multiplication order.
