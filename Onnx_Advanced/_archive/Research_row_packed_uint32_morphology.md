# row_packed_uint32_morphology.md

## Pack

[PROMPT][SPEC][ORT-SRC][DERIVED] The cheapest exact pack I found is **not** the float-weighted idea that your prompt already rules out, and it is also cheaper than the stated `int64` weighted-sum variant: it is a **pure `uint32` weighted row dot-product**. Concretely, take the `[1,1,30,30]` binary plane, `Cast` it to `uint32`, apply `MatMul` against a length-30 powers-of-two vector `P2 = [1,2,4,...,2^29]`, and reshape `[1,1,30] -> [30]`. ONNX `MatMul`-13 accepts `tensor(uint32)` and “behaves like `numpy.matmul`,” so `[1,1,30,30] @ [30]` contracts the last axis to `[1,1,30]`; ORT CPU registers `MatMul` with `BuildKernelDefConstraints<int32_t, uint32_t>()` at opsets 9–12 and 13, so this is CPU-legal on ORT. Exactness is straightforward: each row computes `Σ b_j 2^j` with `b_j∈{0,1}`, and the maximum possible row value is `2^30-1`, which fits safely inside `uint32`. (ONNX MatMul-13; ONNX Cast; ONNX Reshape; onnxruntime/core/providers/cpu/math/matmul.cc)

[PROMPT][DERIVED] Under your cost model, the pack-byte question is purely “sum every node output,” with graph inputs/outputs and initializers free. That makes the weighted-`uint32` `MatMul` form unusually attractive because it creates only one large temporary: the casted `[1,1,30,30]` plane. The row-dot-product output is only 30 lanes.

| family | exact construction | bytes at 30×30 | comments |
|---|---|---:|---|
| (a) log-tree `BitShift` + `BitwiseOr` on halved groups | `[u8] 30→15→8→4`, `Cast` `[30,4]` to `u32`, then `[u32] 4→2→1`, final `Reshape` | `5,430 [DERIVED]` | `[DERIVED]` Cheapest tree I found. The exact irregular tail works as widths `[2]*15 -> [4]*7 + [2] -> [8]*3 + [6] -> [16] + [14] -> [30]`. |
| (b) weighted sums in `int64`, then `Cast→u32` | `Cast(u8/bool→i64)` + batched `MatMul` with `[30]` `int64` powers-of-two vector + `Cast→u32` + `Reshape` | `7,680 [DERIVED]` | `[SPEC][ORT-SRC][DERIVED]` Legal and exact; still loses to the tree. |
| (b′) same weighted-sum idea, but in `uint32` | `Cast(u8/bool→u32)` + batched `MatMul` with `[30]` `uint32` powers-of-two vector + `Reshape` | `3,840 [DERIVED]` | `[SPEC][ORT-SRC][DERIVED]` **Cheapest exact pack overall.** Not one of your listed subfamilies, but same idea and fully legal. |
| (c) 30-step `Slice` + shift-or accumulation | first column `Slice+Cast`, then 29 times `Slice+Cast+BitShift+BitwiseOr`, final `Reshape` | `11,580 [DERIVED]` | `[DERIVED]` Small per-step tensors, but too many steps. |

[SPEC][ORT-SRC][DERIVED] For legality: `BitShift`-11 is elementwise and allows `uint8/uint16/uint32/uint64`; ORT CPU registers `BitShift` for `uint8`, `uint32`, and `uint64`. `BitwiseAnd/Or/Not`-18 allow integer tensors including `uint32`, and ORT CPU registers `uint32` kernels for them. `MatMul`-13 allows `int64` and `uint32`, and ORT CPU registers both the `int64/uint64` and `int32/uint32` families. `Cast` and `Reshape` are fully generic enough for these paths. (ONNX BitShift-11; ONNX BitwiseAnd-18; ONNX BitwiseOr-18; ONNX MatMul-13; onnxruntime/core/providers/cpu/math/element_wise_ops.cc; onnxruntime/core/providers/cpu/math/matmul.cc; onnxruntime/core/providers/cpu/tensor/cast_op.cc)

## Flood round on the packed vector

[SPEC][ORT-SRC][PROMPT][DERIVED] The exact 5-term packed round specified by your Q2/Q5 is
`step(x, m) = (x | E(x) | W(x) | N(x) | S(x)) & m`,
with one `uint32` lane per row, one bit per column, and `m` the packed mask. `BitShift` is elementwise, so horizontal motion is by bit shifts inside each `uint32` lane, while vertical motion must be done by lane-axis shifts using tensor ops, not bitwise shifts. ONNX `BitShift`-11 is elementwise with unsigned integer types; ONNX `BitwiseAnd/Or`-18 admit `uint32`; ORT CPU registers `uint32` kernels for all of them. `Pad`-18/19 admits `uint32`, and its spec explicitly allows negative pads to crop and zero-fill, which gives a one-node exact lane shift. (ONNX BitShift-11; ONNX BitwiseAnd-18; ONNX BitwiseOr-18; ONNX Pad; onnxruntime/core/providers/cpu/math/element_wise_ops.cc; onnxruntime/core/providers/cpu/tensor/pad.cc)

[SPEC][ORT-SRC][DERIVED] Using two `BitShift`s, two 1-D `Pad`s for lane shifts, four binary `BitwiseOr`s, and one final `BitwiseAnd` with the packed mask, the **exact per-round byte cost is 1,080 B**: `120 + 120 + 120 + 120 + 4×120 + 120`. That is **not** the “~120 B per round” promise once all temporaries are charged; it is only the size of the final packed state tensor itself. It is also **not** lower than a single 900 B unpacked plane, but it **is** lower than your Q4 unpacked round baseline of `900 + 900 = 1,800 B`. (ONNX BitShift-11; ONNX BitwiseAnd-18; ONNX BitwiseOr-18; ONNX Pad; onnxruntime/core/providers/cpu/math/element_wise_ops.cc; onnxruntime/core/providers/cpu/tensor/pad.cc)

[ORT-SRC][SPEC][DERIVED] If you avoided negative-pad lane shifts and used the explicit `Slice + Concat` fallback instead, north and south would each cost `116 + 120 = 236 B`, so the round would rise to **1,312 B**. That fallback is still legal because `Slice` and `Concat` both admit `uint32`, ORT CPU dispatches `Slice` on `uint32`, and ORT CPU `Concat` is registered for all tensor types. (ONNX Slice; ONNX Concat; onnxruntime/core/providers/cpu/tensor/slice.cc; onnxruntime/core/providers/cpu/tensor/concat.cc)

[SPEC][DERIVED] The shift-union algebra itself is standard bitboard flood-fill: repeated one-step directional shifts with unions, optionally intersected by masks to stop or confine propagation. That is exactly the flavor described in `Dumb7Fill` and in the Chessprogramming fill overview, and it matches Tony Finch’s word-parallel “Life” discussion of register-wide shifts and boolean combination over 32-bit groups. (Chessprogramming `Dumb7Fill`; Chessprogramming `Fill Algorithms`; Finch 2003)

## Unpack

[SPEC][ORT-SRC][DERIVED] The cheapest exact unpack is your candidate **(a)**: reshape the `[30]` packed vector to `[1,1,30,1]`, broadcast `BitwiseAnd` against the 30-element powers-of-two vector, then `Greater(..., 0)` to land in `bool` at `[1,1,30,30]`. Exact byte cost is **4,620 B** = `Reshape 120 + BitwiseAnd 3,600 + Greater 900`. The big intermediate is indeed a `[1,1,30,30] uint32` tensor, i.e. `900 × 4 = 3,600 B`. `Greater`-13 accepts `uint32` inputs and returns `bool`; ORT CPU registers `Greater` for `uint32` at opset 13. (ONNX BitwiseAnd-18; ONNX Greater-13; ONNX Reshape; onnxruntime/core/providers/cpu/math/element_wise_ops.cc)

[SPEC][ORT-SRC][DERIVED] The cheapest version of candidate **(b)**, if interpreted as a **right-shift ladder** instead of 30 independent shifts, is **9,000 B**: reshape once (`120`), then for bit 0 do `And+Greater` (`150`), and for bits 1–29 do `ShiftRight by 1 + And + Greater` (`29×270 = 7,830`), then `Concat` the 30 `[1,1,30,1] bool` columns into `[1,1,30,30] bool` (`900`). Candidate **(c)**, a `Div/Mod` ladder, is also **9,000 B** by the same logic: 30 `Mod`, 29 `Div`, 30 `Greater`, one reshape, one concat. `Div`-14 and `Mod`-13 both admit `uint32`, and ORT CPU registers `Div` for `uint32` and enables `Mod` over `uint32` in its type-control tables. (ONNX Div; ONNX Mod; ONNX Concat; onnxruntime/core/providers/cpu/math/element_wise_ops.cc)

| family | exact construction | bytes at 30×30 | comments |
|---|---|---:|---|
| (a) broadcast mask + `Greater` | `Reshape [30]→[1,1,30,1]` + broadcast `BitwiseAnd` with `P2[30]` + `Greater(,0)` | `4,620 [DERIVED]` | `[DERIVED]` **Cheapest exact unpack.** Final tensor is `bool` at 900 B. |
| (b) per-bit shift+mask ladder | repeated `ShiftRight by 1`, `BitwiseAnd 1`, `Greater`, then `Concat` | `9,000 [DERIVED]` | `[DERIVED]` Reusing the ladder state is cheaper than 30 independent shifts, but still much worse than broadcast-mask. |
| (c) `Div/Mod` ladder | repeated `Mod 2`, `Greater`, `Div 2`, then `Concat` | `9,000 [DERIVED]` | `[DERIVED]` Same asymptotic shape as (b), no byte advantage. |

## End-to-end pricing

[PROMPT][SPEC][ORT-SRC][DERIVED] Using the **cheapest exact pack** (`uint32` weighted `MatMul`, `3,840 B`), the **exact packed round** with two bit shifts + two lane `Pad`s + four `Or`s + final mask `And` (`1,080 B`), and the **cheapest exact unpack** (`4,620 B`), the packed-path total is
`3,840 + K×1,080 + 4,620 = 8,460 + 1,080K` bytes.
Following your required comparison, the unpacked baseline is `K × (900 + 900) = 1,800K` bytes. Shared initializer count for this construction is **41 elements**: one 30-element powers-of-two vector shared by pack and unpack, one `[30]` reshape shape, one shift-amount scalar, two 2-element pad vectors, one 4-element unpack reshape shape, and one zero scalar for `Greater`. (ONNX MatMul-13; ONNX BitShift-11; ONNX BitwiseAnd/Or-18; ONNX Pad; ONNX Greater-13; ONNX Reshape; onnxruntime/core/providers/cpu/math/matmul.cc; onnxruntime/core/providers/cpu/math/element_wise_ops.cc; onnxruntime/core/providers/cpu/tensor/pad.cc)

[DERIVED] The crossover for that one-pack accounting is where `8,460 + 1,080K < 1,800K`, i.e. **`K > 11.75`**, so the first integer win is **`K = 12`**.

| K | packed total bytes | unpacked total bytes | ratio | params |
|---|---:|---:|---:|---:|
| 5 | 13,860 [DERIVED] | 9,000 [PROMPT][DERIVED] | 1.54 [DERIVED] | 41 [DERIVED] |
| 11 | 20,340 [DERIVED] | 19,800 [PROMPT][DERIVED] | 1.03 [DERIVED] | 41 [DERIVED] |
| 14 | 23,580 [DERIVED] | 25,200 [PROMPT][DERIVED] | 0.94 [DERIVED] | 41 [DERIVED] |
| 25 | 35,460 [DERIVED] | 45,000 [PROMPT][DERIVED] | 0.79 [DERIVED] | 41 [DERIVED] |

[SPECULATIVE][DERIVED] One important accounting caveat: the table above follows your wording literally as **one online pack + K packed rounds + one unpack**. If the topology mask is another **dynamic** plane that must also be packed online rather than being pre-packed or constant, add **one more pack cost** of `3,840 B`. That moves the crossover to about **`K = 18`** while leaving params unchanged, because the same initializer set can be shared.

## Exactness audit

[DERIVED] Bit convention for the formulas above: **lane `r` stores row `r`, and bit `j` stores column `j`**. If your project uses the opposite within-row bit ordering, swap LEFT and RIGHT everywhere; the byte totals do not change.

[SPEC][PROMPT][DERIVED] The 5-term packed round is **exactly one 4-neighbor geodesic dilation step plus persistence**, with no doubling shortcuts. Each generated term is a **single-cell** neighbor transform of the original `x`: `x<<1`, `x>>1`, one-lane north shift, or one-lane south shift. `BitShift` is elementwise, so bits never cross tensor elements; that prevents any hidden inter-row carry or word-to-word propagation. The `Or` chain only unions those immediate-neighbor sets. There is no `x<<2`, no recursive in-round saturation, and no Kogge-Stone-style prefix propagation, so one round cannot jump down a corridor by more than one graph step. (ONNX BitShift-11; Chessprogramming `Dumb7Fill`; Chessprogramming `Fill Algorithms`)

[DERIVED] The boundary conditions and masks are:

- [DERIVED] **Spare-bit hygiene**: because only bits `0..29` are used, the canonical validity mask is `ROW_VALID_30 = 0x3fffffff`.
- [DERIVED] **East edge**: if east means `LEFT shift by 1`, an original bit at column 29 would move to bit 30. You can handle that either by an explicit post-mask `((x << 1) & ROW_VALID_30)` or, more cheaply, by relying on the final `& packed_mask`, provided that the packed mask has bits 30 and 31 cleared.
- [DERIVED] **West edge**: if west means `RIGHT shift by 1`, no explicit mask is needed; bit 0 simply shifts out to zero.
- [DERIVED] **North edge**: use a lane shift that drops row 0’s predecessor and zero-fills the new boundary lane. With top-to-bottom lane order, one exact form is `Pad(x, pads=[-1, +1])`.
- [DERIVED] **South edge**: symmetrically, `Pad(x, pads=[+1, -1])` zero-fills the other boundary.
- [DERIVED] **Lane wrap**: there is none, because north/south are tensor-axis shifts, not bit shifts.
- [DERIVED] **Mask exactness**: the final `& packed_mask` enforces the same geodesic restriction as the unpacked `∧ mask`, assuming the mask was packed with the same row/bit convention.

[PROMPT][DERIVED] There is one important scope mismatch to keep explicit: the round priced above is the **5-term 4-neighbor** step that Q2/Q5 explicitly asked for. It is **not** the same operator as a full **`dilate3x3` / 8-neighbor** step from your project-context sentence unless you add the four diagonal terms `NE/NW/SE/SW`. So this note is an exact answer to the card as written, but not an automatic proof that the current full 3×3 unpacked flood can be replaced 1-for-1 without re-pricing diagonals.

## Blunt verdict

[DERIVED] **Adopt conditionally.** If this card is truly about the exact **5-term 4-neighbor** row-packed step, then yes: exact ONNX/ORT CPU constructions exist, the best pack is a `uint32` weighted `MatMul` at **3,840 B**, the packed round is **1,080 B** under your cost model, the best unpack is **4,620 B**, and the one-pack crossover is **`K = 12`** with clear wins by `K = 14` and `K = 25`. If the mask must also be packed online, move that crossover to about **`K = 18`**. But if you actually need the project-context **full 3×3 / 8-neighbor dilation**, do **not** treat this as a drop-in green light yet: the 4-neighbor card is exact, the “~120 B per round” slogan is false once temporaries are charged, and the diagonal extension still needs its own pricing before you commit.