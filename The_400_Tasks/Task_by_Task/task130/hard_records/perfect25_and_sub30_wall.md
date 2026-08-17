# task130 hard record — perfect-25 rumor + sub-30 WALL

- **Live bar:** cost **30**, sha256 `aaf291bab4664270ac1030a0f0a0efa703f037c864bb9dea18014975c16b0dc3`
  (manifest/workbook, owner Aditya, 2026-06-19). profile==static==30 (memory 0 + params 30).
- **Champion graph:** single depthwise `Conv`, kernel `[10,1,1,3]` all-ones except gray
  channel 5 = `[0,0,0]`, `group=10`, `strides=[3,3]`, `pads=[0,0,58,60]`.
  Sums each 3×3 cell's top row (3 taps), zeroes gray, downsamples ×3, pads to 30×30 — all in ONE node ⇒ memory 0.

## Routes proven dead (2026-07-07, bo_ng_cld_worker1)

| route | params | gate | result |
|---|---:|---|---|
| Identity (zero-param) | 0 | visible | **0/265 FAIL** — perfect-25 rumor false |
| Transpose(0,1,3,2) (only shape-legal perm) | 0 | visible | **0/265 FAIL** |
| Conv width-2 (2 taps/cell) | 20 | visible | **253/265 FAIL** — 2 gray can kill both sampled taps |
| MaxPool single-node, big pads | 0 | build | **ORT reject**: pad must be `< kernel_shape` — pool can't reshape to 30×30 in one node |
| MaxPool→Mul→Pad (param-free reducer, correct) | 18 | visible PASS | **cost 8018** (mem 8000 = two `[1,10,10,10]` intermediates) — 267× worse |

## Why 30 is the forced floor (memory model = `neurogolf_utils.calculate_memory`)
1. `calculate_memory` charges **every** node-output tensor its full byte size **except** the two
   tensors literally named `input`/`output`. ⇒ any 2+ node graph pays for ≥1 charged intermediate.
   The smallest full-canvas intermediate is `[1,10,30,30]` F32 = 36,000 B; a downsampled
   `[1,10,10,10]` F32 = 4,000 B; even `[1,10,3,3]` INT8 = 90 B — all ≫ 30.
   ⇒ **the champion must be a single input→output node (memory 0).**
2. Single param-free ops (Identity/Transpose/Slice/Pool) cannot both (a) suppress gray per-channel
   and (b) per-cell-reduce 3 taps. Pools are channel-blind (can't zero gray); Identity/Transpose
   can't reduce. ⇒ the single node must carry per-channel weights ⇒ **Conv**.
3. The generator may gray ≤2 of a cell's 9 pixels anywhere, so a robust witness needs ≥3 distinct
   taps (any 2-tap set can be fully grayed — empirically 253/265). 3 taps × 10 output channels
   (output contract is 10 color planes) = **30 weight elements**, and Conv weight dim-0 must equal
   the 10 output channels (gray channel can't be dropped to 0 taps; kernel dims are uniform).

⇒ **30 = 3 forced taps × 10 forced channels is the exact floor. Sub-30 is impossible. WALL.**

## One remaining hypothesis (only if the cost model changes)
A sub-30 win would require a single-node op that outputs `[1,10,30,30]` from a `<30`-element
initializer while still reducing 3 taps per channel. None exists in the ai.onnx op set: Conv is the
unique reduce+per-channel-weight single node, and its weight tensor is floored at `M·(C/g)·kH·kW =
10·1·1·3 = 30`. Revisit only if (a) the grader stops charging intermediate memory, or (b) the
generator guarantee weakens to ≤1 gray/cell (then width-2 = 20 params becomes legal).
