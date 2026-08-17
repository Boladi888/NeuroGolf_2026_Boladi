# Task286 55,933 Cost Breakdown

Model: `.hermes/board/staged_submits/task286_v4_u8tail_croplabel_t_06514cd5_on_63211/task286.onnx`

Note: this is the originally staged 55,933 crop-label artifact. The builder now removes a dead
`s25lab` initializer on crop-label rebuilds, producing a 55,929 strict rebuild; the tensor proof below is
kept for the 55,933 model requested in the prompt.

Score: memory `53,832` + params `2,101` = cost `55,933`. The memory sum is every inferred intermediate excluding `input`/`output`; params are initializer element counts, matching `score_network_v2`.

## Class Totals

| Class | Tensors | Memory | Params | Total |
|---|---:|---:|---:|---:|
| mandatory | 422 | 45,627 | 741 | 46,368 |
| removable by reordering | 9 | 5,900 | 0 | 5,900 |
| shared/duplicated | 20 | 1,680 | 1,000 | 2,680 |
| removable by dtype | 1 | 625 | 0 | 625 |
| removable by crop | 1 | 0 | 360 | 360 |

## Family Totals

| Class | Family / reason | Tensors | Memory | Params | Total |
|---|---|---:|---:|---:|---:|
| mandatory | reach/flood stack: 100-byte U32 row-bit state; removable only by replacing reach encoding | 400 | 40,000 | 0 | 40,000 |
| removable by reordering | UINT8 color-tail plane; candidate direct-output/tail reorder target | 8 | 5,000 | 0 | 5,000 |
| mandatory | full 25x25 unpack from U32 rows for final per-cell mask | 1 | 2,500 | 0 | 2,500 |
| mandatory | Conv output for cropped 25x25 label plane; already paid params to remove 30x30 label runtime crop | 1 | 2,500 | 0 | 2,500 |
| shared/duplicated | row-pack setup for open/seed; overlapping kernels but current direct two-pack is cheaper than deriving seed from open | 16 | 1,680 | 0 | 1,680 |
| shared/duplicated | row-pack kernel table; open/seed duplicate most channel weights but direct pack avoids extra runtime masks | 4 | 0 | 1,000 | 1,000 |
| removable by reordering | single-channel 30x30 canvas before free Equal; direct one-hot output likely costs more but this is a tail target | 1 | 900 | 0 | 900 |
| mandatory | BOOL reach mask consumed by UINT8 tail | 1 | 625 | 0 | 625 |
| removable by dtype | casted UINT8 label plane; dtype-optimized, but tied to label output path | 1 | 625 | 0 | 625 |
| mandatory | checkerboard parity table for even/odd seed color split | 1 | 0 | 625 | 625 |
| removable by crop | crop-label trade: +350 params vs 1x1 label to remove 2000 runtime bytes | 1 | 0 | 360 | 360 |
| mandatory | shared reach unpack/vertical-shift table | 3 | 0 | 75 | 75 |
| mandatory | shape/output glue initializer | 6 | 0 | 34 | 34 |
| mandatory | scalar glue initializer | 7 | 0 | 7 | 7 |
| mandatory | small scalar or glue tensor | 2 | 2 | 0 | 2 |

## Largest Memory Tensors

| Tensor | Op | Dtype | Shape | Bytes | Class |
|---|---|---|---|---:|---|
| `reach_bits_417` | BitwiseAnd | UINT32 | 1x1x25x25 | 2,500 | mandatory |
| `lab_f_419` | Conv | FLOAT | 1x1x25x25 | 2,500 | mandatory |
| `padded_u8_431` | Pad | UINT8 | 1x1x30x30 | 900 | removable by reordering |
| `reach_b_418` | Greater | BOOL | 1x1x25x25 | 625 | mandatory |
| `lab_u8_420` | Cast | UINT8 | 1x1x25x25 | 625 | removable by dtype |
| `gt_one_421` | Greater | BOOL | 1x1x25x25 | 625 | removable by reordering |
| `is_wall_422` | Equal | BOOL | 1x1x25x25 | 625 | removable by reordering |
| `non_bg_423` | Where | UINT8 | 1x1x25x25 | 625 | removable by reordering |
| `seed_label_424` | Where | UINT8 | 1x1x25x25 | 625 | removable by reordering |
| `seed_even_425` | Where | UINT8 | 1x1x25x25 | 625 | removable by reordering |
| `seed_odd_426` | Where | UINT8 | 1x1x25x25 | 625 | removable by reordering |
| `painted_u8_429` | Where | UINT8 | 1x1x25x25 | 625 | removable by reordering |
| `outlab_u8_430` | Where | UINT8 | 1x1x25x25 | 625 | removable by reordering |
| `lo_open_1` | Conv | FLOAT | 1x1x30x1 | 120 | shared/duplicated |
| `hi_open_2` | Conv | FLOAT | 1x1x30x1 | 120 | shared/duplicated |
| `lo_seed_9` | Conv | FLOAT | 1x1x30x1 | 120 | shared/duplicated |
| `hi_seed_10` | Conv | FLOAT | 1x1x30x1 | 120 | shared/duplicated |
| `los_open_3` | Slice | FLOAT | 1x1x25x1 | 100 | shared/duplicated |
| `his_open_4` | Slice | FLOAT | 1x1x25x1 | 100 | shared/duplicated |
| `lou_open_5` | Cast | UINT32 | 1x1x25x1 | 100 | shared/duplicated |

## Largest Initializers

| Tensor | Dtype | Shape | Param units | Class |
|---|---|---|---:|---|
| `even_b` | BOOL | 1x1x25x25 | 625 | mandatory |
| `label_w_crop6` | FLOAT | 1x10x6x6 | 360 | removable by crop |
| `klo_open` | FLOAT | 1x10x1x25 | 250 | shared/duplicated |
| `khi_open` | FLOAT | 1x10x1x25 | 250 | shared/duplicated |
| `klo_seed` | FLOAT | 1x10x1x25 | 250 | shared/duplicated |
| `khi_seed` | FLOAT | 1x10x1x25 | 250 | shared/duplicated |
| `prev_idx` | INT64 | 25 | 25 | mandatory |
| `next_idx` | INT64 | 25 | 25 | mandatory |
| `bit2pow` | UINT32 | 1x1x1x25 | 25 | mandatory |
| `rg` | UINT8 | 1x10x1x1 | 10 | mandatory |
| `padp` | INT64 | 8 | 8 | mandatory |
| `s0` | INT64 | 4 | 4 | mandatory |
| `s25` | INT64 | 4 | 4 | mandatory |
| `sax` | INT64 | 4 | 4 | mandatory |
| `s25lab` | INT64 | 4 | 4 | mandatory |
| `sh13` | UINT32 | scalar | 1 | mandatory |
| `one_u32` | UINT32 | scalar | 1 | mandatory |
| `zero_u32` | UINT32 | scalar | 1 | mandatory |
| `zero_u8` | UINT8 | scalar | 1 | mandatory |
| `one_u8` | UINT8 | scalar | 1 | mandatory |

## Audit Notes

- The flood/reach core is 400 U32 row-state tensors at 100 bytes each = 40,000 bytes. This is 71.5% of total cost and 74.3% of runtime memory. Reordering and dtype changes inside the existing row-bit representation do not remove it; only a different wall-safe reach encoding can.

- The no-flood floor for this exact tail is 15,933 = 13,832 memory + 2,101 params. The m20 stack adds exactly 40,000 bytes.

- The crop-label trick is visible in `label_w_crop6`: +360 params total for label weights, but it removes the old 30x30 label cast/slice runtime path.

- Full tensor-by-tensor rows are in `cost_breakdown_55933.tsv`.
