# task117 — crop15 + F16-output (the cornerstone of crop + F16)

| Field | Value |
|---|---|
| Task | task117 |
| Family | Pattern_Recognition / Filling Regions |
| Cost reduction | -100,525 (148,857 → 48,332) |
| Estimated LB | +1.1249 |
| Patterns | `crop_to_size_advisory` (manual application) + F16-output trick |
| Active grid | 15×15 |

## Why this exemplar matters

task117 is the canonical example of stacking TWO complementary techniques on a single task. Each technique alone saves bytes; together they multiply. Workers tackling any Pattern_Recognition / Filling Regions task should study this first.

## Technique 1: Crop to active grid

The original task117 graph operates on `[1, 10, 30, 30]` intermediates throughout, even though task117's max input/output grid is `15×15` — half the canvas is wasted padding.

### Before (graph operates on full 30×30)

```
input [1, 10, 30, 30] F32
  → Cast(input, F16) → input_f16 [1, 10, 30, 30] F16 = 18,000 bytes
  → various ops on [..., 30, 30] intermediates
  → 52 intermediates each [..., 30, 30]
  → Concat → output_f16 [F16, [1,10,30,30]] = 18,000 bytes
  → Cast(output_f16, F32) → output [F32, [1,10,30,30]] = 36,000 bytes (excluded as output)
```

### After (graph operates on cropped 15×15 plus Pad back)

```
input [1, 10, 30, 30] F32
  → Slice(input, ends=[1, 10, 15, 15]) → input_cropped [1, 10, 15, 15] F32 = 9,000 bytes
  → Cast(input_cropped, F16) → input_f16 [1, 10, 15, 15] F16 = 4,500 bytes
  → various ops on [..., 15, 15] intermediates (52 of them, each 4× smaller)
  → Concat → output_f16 [F16, [1,10,15,15]] = 4,500 bytes
  → Pad(output_f16, pads=[0,0,0,0, 0,0,15,15]) → output_padded [F16, [1,10,30,30]] = 18,000 bytes
  # ↑ output_padded becomes the model output (declared F16); excluded from memory
```

## Critical step: rewrite dimension-tied initializers

The graph contains initializers tied to the 30×30 dimension:
- `range30` (INT64 [30]) — used for reflection/period ops.
- `c29i` (INT64 scalar) — typically 29.
- `c30i` (INT64 scalar) — typically 30.

After cropping to 15×15, these MUST be rewritten in lockstep:

```python
# Cropped equivalents
new_inits.append(helper.make_tensor("range30", TensorProto.INT64, [15], list(range(15))))
new_inits.append(helper.make_tensor("c29i", TensorProto.INT64, [], [14]))   # H-1
new_inits.append(helper.make_tensor("c30i", TensorProto.INT64, [], [15]))   # H

# New inits for the Slice + Pad surgery
new_inits.append(helper.make_tensor("slice_starts", TensorProto.INT64, [4], [0, 0, 0, 0]))
new_inits.append(helper.make_tensor("slice_ends",   TensorProto.INT64, [4], [1, 10, 15, 15]))
new_inits.append(helper.make_tensor("slice_axes",   TensorProto.INT64, [4], [0, 1, 2, 3]))
new_inits.append(helper.make_tensor("pad_pads",     TensorProto.INT64, [8], [0,0,0,0, 0,0,15,15]))
```

`range30` keeping its NAME but changing its CONTENTS is correct — every downstream Gather/Slice that consumes `range30` still works because they use the name, not the size.

## Critical step: `graph.ClearField("value_info")`

ORT's load uses cached `value_info` for shape validation. After cropping, intermediates that USED to be `[1, 1, 30, 30]` are now `[1, 1, 15, 15]`. Without clearing value_info, ORT load fails with:

```
Op (Max) [ShapeInferenceError] Incompatible dimensions
```

at the first 4-input Max where the cached shape disagrees with runtime. **One line of code or the model is dead**:

```python
g.ClearField("value_info")
```

See `ort_reality.md §10`.

## Technique 2: F16-output trick

The original graph ends:

```
... → Concat → output_f16 [F16, [1,10,30,30]] = 18,000 bytes intermediate
      Cast(output_f16, F32) → output [F32, [1,10,30,30]] = excluded as output
```

The `output_f16` is a counted F16 intermediate (18,000 bytes). The `output` is the model output (excluded).

After the F16-output rewrite:

```
... → Concat → output [F16, [1,10,15,15]] = excluded as output (renamed)
```

Drop the final F32 Cast. Concat's output becomes the model output directly, declared as F16. Saves the F16 intermediate (which used to be Concat's output).

```python
for n in g.node:
    if n.op_type == "Cast" and n.output and n.output[0] == "output":
        continue                                      # drop the final Cast
    if n.op_type == "Concat" and n.output and n.output[0] == "output_f16":
        n.output[0] = "output"                        # rename to be the model output
g.output[0].type.tensor_type.elem_type = TensorProto.FLOAT16
```

After the crop (Technique 1) the `output_f16` Concat output is at `[1,10,15,15]` = 4,500 bytes, but the Pad's output (the actual model output) is at `[1,10,30,30]` F16 = 18,000 bytes (excluded). Net: the Concat's `output_f16` intermediate disappears (saves another 4,500 bytes).

## Combined savings

- Crop alone: -50,000 bytes (52 intermediates, each [..., 30,30] → [..., 15, 15] saves 75% per intermediate at F16).
- F16-output alone: -18,000 bytes.
- Stacked: -100,525 bytes (148,857 → 48,332). +1.1249 LB.

## Why it's safe under official scoring

- **Crop semantics**: Slice extracts the active 15×15 region; compute on it; Pad back to 30×30 with constant 0. The output is identical to the original graph's output on every active cell. Padding cells (outside the active 15×15) are 0 in both versions.
- **F16 representation**: {0.0, 1.0} F16 is exact. `(out > 0.0)` thresholding produces the same {0, 1} mask under decoded.

Verify: `tools/verify_against_live.py --tid 117 --live-zip submissions/submission.zip` returns 265/265 OK.

## Generalizes to

`techniques.md` §2 (Crop-to-active-grid) + §3 (F16-output trick). Apply to any Pattern_Recognition task where:

- Max active grid is significantly < 30×30 (check `data/competitions/neurogolf-2026/taskNNN.json` input/output shapes).
- Graph has many `[..., 30, 30]` F32 or F16 intermediates.
- Graph uses dimension-tied initializers (range30, c29i, c30i) that can be rewritten.
- The graph's first node ISN'T already a Slice to the active region (false-positive guard — see `traps.md §21`).

## Reproducibility

The 5-line surgery template:

```python
import onnx
from onnx import helper, TensorProto

m = onnx.load_from_string(live_bytes)
g = m.graph
H = max_active_grid  # e.g. 15 for task117

# 1. Rewrite dimension-tied initializers
new_inits = [...]   # as above

# 2. Prepend Slice + insert before the first Cast(input)
slice_node = helper.make_node("Slice", ["input", "slice_starts", "slice_ends", "slice_axes"], ["input_cropped"])
new_nodes = [slice_node]
for n in g.node:
    if n.op_type == "Cast" and n.output and n.output[0] == "input_f16":
        n.input[0] = "input_cropped"
    elif n.op_type == "Concat" and n.output and n.output[0] == "output_f16":
        n.output[0] = "output_unpadded"
    elif n.op_type == "Cast" and n.output and n.output[0] == "output":
        continue   # drop final Cast
    new_nodes.append(n)
# Append Pad to restore [1,10,30,30]
new_nodes.append(helper.make_node("Pad", ["output_unpadded", "pad_pads"], ["output"], mode="constant"))
g.ClearField("node"); g.node.extend(new_nodes)

# 3. Set graph output dtype to F16
g.output[0].type.tensor_type.elem_type = TensorProto.FLOAT16

# 4. CRITICAL: strip stale value_info
g.ClearField("value_info")

onnx.save(m, "task117_cropped.onnx")
```

Then verify and score.

## Cross-ref

- `task384_f16_drop.md` — F16-output trick alone on a task that's already cropped.
- `task055_per_channel.md` — F16 placement when per-channel Gather happens.
- `Algorithm_Library/per_family_starting_points.md` — Pattern_Recognition family entry recommends starting with this stack.

## Stacks with other techniques

After crop + F16-output, additional savings can come from:
- `bool_andfold` / `mul_mask_to_where` on remaining F32 mask intermediates.
- `separable_mask_collapse` on 2D mask initializers within the cropped region.
- input → BOOL whole-graph (rule 8) if downstream is fully bool-tolerant.
