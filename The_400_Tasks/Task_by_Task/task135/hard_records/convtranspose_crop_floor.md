# task135 — DEAD ROUTE: sub-30 crop (forced floor 30)

- **Live bar:** cost **30** (manifest authoritative), sha256 `84b2d9b713db8abb8c24001aef5e9a1513f203d2714563b4d747bee27a0b1a4a`, owner Aditya.
- **Task:** crop top-right 3×3 of the 9×9 input → output[0:3,0:3], rest zero. (Verified 4 train + 1 test + 266 arc-gen.)
- **Class:** the prompt framed this as a "perfect-25 zero-cost" rumor slot (forced floor <=1). **That premise is false** — it is a genuine crop that deletes a data band, not a whole-grid identity/transpose/flip/rotate.

## Attempted routes & exact failure

| route | gate | result | why dead |
|---|---|---|---|
| Identity / Transpose / flip (perfect-25 family) | visible | **0/266** | output is a 3×3 corner crop, not a whole-grid permutation |
| ConvTranspose depthwise kh=2 (20 params) | visible | **0/266** | 2 kernel taps select only 2 of the 3 output rows |
| Slice→Pad (two-op, params 14) | visible PASS / profile | **cost 374** | 3×3×10 F32 crop intermediate = 360 B `memory` ≫ 30 |
| ConvTranspose depthwise kh=3 (rebuild of LIVE) | visible PASS / profile | **cost 30** | ties the live floor — not stageable (never stage a tie) |

## Forced-floor argument (why 30 is minimal)

cost = `memory` (non-output intermediate BYTES, F32×4) + `params` (initializer ELEMENT count, dtype-independent — #109).

1. **Single op is mandatory.** Output must be [1,10,30,30] (encode_grid pads the 3×3 to a 30×30 canvas). Any op that emits the bare 3×3 (Slice/Gather) has the wrong shape and needs a second op to re-pad → the 3×3×10 crop is materialized as a non-output intermediate = 90 elems × 4 B = **360 B memory** (measured 374 total). Only a single op keeps memory 0 (its lone tensor is the graph output).
2. **Horizontal shift is free.** Moving cols 6-8 → 0-2 is a pad-shift (`pads` W begin=6/end=-6); the shifted-in cols 3-29 come from input cols 9-29 which are already zero (grid is only 9 wide). No payload.
3. **Vertical band-removal is NOT free and forces 3 taps.** Rows 3-8 hold real grid data (grid is 9 tall) that must be deleted while keeping a 30×30 output. A monotonic shift/crop cannot delete an interior band and restore size. The ConvTranspose dilated-tap trick (strides=[30,1], dilations=[29,1], pads H=[58,841]) serves each surviving output row 0,1,2 from a **distinct** kernel tap and maps rows 3-8 out of the cropped range → **kernel height 3 forced** (kh=2 = visible 0/266).
4. **10 channels forced.** Output must be exactly 10 one-hot channels; group=10 depthwise requires one [1,3,1] weight slice per channel. All 10 colors appear in the corpus.
5. Therefore weight volume = 10 × 1 × 3 × 1 = **30 elements = 30 params**, memory 0 ⇒ **cost 30 is the forced floor.** dtype narrowing saves nothing (params = elements).

## Remaining hypothesis (thin)

The only conceivable sub-30 route would be a single ONNX op that (a) has no weight payload (pooling/Slice/Pad have none) AND (b) can delete the interior data band rows 3-8 while emitting a full 30×30 canvas. No such op exists: pooling can't negative-crop or non-monotonically gather, Slice/Pad change shape or can't zero interior data. **Verdict: WALL at the forced floor 30. Current live is already optimal.**
