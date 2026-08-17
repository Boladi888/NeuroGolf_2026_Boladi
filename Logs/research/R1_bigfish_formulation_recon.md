# R1 Big-Fish Formulation Recon

Generated from repo-local sources only. Scope filter: `boladi_workable=Y` and `current_cost >= 10000` in `The_400_Tasks/task_state.csv`. Generator IDs use the repo mapping in `tools/arc_gen_enrich.py`: task001..task400 equals sorted ARC-AGI training UUIDs, so source is `external/ARC-GEN/tasks/task_<generator_id>.py`.

Floor estimates are skeleton-level estimates, not built or gated. Point gain is `25 - ln(floor_estimate) - current_points`. Live ONNX tensor dominators are static `value_info`/shape-inference profiles from `submissions/submission.zip`; the per-task worker still must run real-input profiling before staging.

- Big-fish rows found: 66
- Active decoded worklist rows: 40
- Skips: 26 compaction/proven rows (26 compaction-owned, 0 floor/proven)

## 10 Highest-EV Tasks

| rank | task | current | floor est | est gain | formulation | rule |
|---:|---|---:|---:|---:|---|---|
| 1 | task342 | 12836 | 250 | 3.939 | quadrant marker Gather + 2x2 free-output Pad | Map the four colored quadrant markers around a cyan 2x2 block into the corresponding four cells of that block. |
| 2 | task371 | 17340 | 350 | 3.903 | two-marker midpoint arithmetic + free-output plus mask | Find two blue dots on a row/column, draw a green plus at their midpoint, and preserve the blue dots; optional transpose. |
| 3 | task362 | 11536 | 300 | 3.649 | gray-count offset + row/column line redraw as output | Shift a same-colored cross down and left by the number of gray markers on the right edge. |
| 4 | task253 | 23974 | 700 | 3.534 | corner-triomino detector + small-grid free-output Pad | Detect four colored 2x2 corner triominoes and assemble their colors into fixed quadrants of a 4x4 output. |
| 5 | task183 | 15138 | 500 | 3.410 | small-grid crop + quadrant mask/color Gather + free-output Pad | Crop the interior of a blue-framed board and recolor cyan cells by the color marker in their quadrant corner. |
| 6 | task019 | 21779 | 900 | 3.186 | 2x ConvTranspose/block upsampler + cyan halo Conv/Where | Replicate each colored input cell into the four quadrants of a 2x enlarged output and add cyan diagonal halos around each replica. |
| 7 | task289 | 21139 | 900 | 3.156 | block-upsampler ConvTranspose bank (#80-inverse) selected by color-count k | Upsample each colored cell of a 3x3 sparse grid into a kxk block where k is the number of distinct colors present. |
| 8 | task046 | 25984 | 1200 | 3.075 | segment-boundary detector + row-offset Gather/Scatter reconstruction | Undo per-segment vertical offsets in a 3-row serpentine colored path, dropping gray separators and reassembling the original contiguous path. |
| 9 | task343 | 12872 | 600 | 3.066 | period extractor + Tile/Concat free-output extension | Continue a periodic vertical-bar pattern across the fixed 15-column canvas, optionally flipping every other period. |
| 10 | task029 | 14545 | 700 | 3.034 | bbox from border color + dynamic fixed-size crop (Gather not Slice) | Use the colored border as a zoom box and output the interior crop. |

## Ranked Active Worklist

| rank | task | cost | gen | misnomer | floor | gain | confidence | formulation | dominant tensors |
|---:|---|---:|---|---|---:|---:|---|---|---|
| 1 | task342 | 12836 | d89b689b | N | 250 | 3.939 | decoded-clean | quadrant marker Gather + 2x2 free-output Pad | Slice [1, 10, 10, 10] F32 4000B x10; Equal [1, 10, 10, 10] BOOL 1000B y10_b; Where [1, 1, 10, 10] F32 400B xq_tr |
| 2 | task371 | 17340 | e9614598 | Y | 350 | 3.903 | decoded-clean | two-marker midpoint arithmetic + free-output plus mask | Slice [1, 10, 14, 14] F32 7840B input_inner_f32_internal; Cast [1, 10, 14, 14] F16 3920B input_inner; Concat [1, 4, 14, 14] F16 1568B inner4 |
| 3 | task362 | 11536 | e48d4e1a | Y | 300 | 3.649 | decoded-clean | gray-count offset + row/column line redraw as output | Slice [1, 9, 10, 10] F32 3600B v10_f32_internal; Concat [1, 10, 10, 10] F16 2000B v36; Cast [1, 9, 10, 10] F16 1800B v10 |
| 4 | task253 | 23974 | a61ba2ce | N | 700 | 3.534 | decoded-clean | corner-triomino detector + small-grid free-output Pad | Conv [1, 40, 12, 12] F16 11520B v15; Slice [1, 10, 13, 13] F32 6760B input_cropped_f32_internal; Cast [1, 10, 13, 13] F16 3380B input_cropped |
| 5 | task183 | 15138 | 77fdfe62 | N | 500 | 3.410 | decoded-clean | small-grid crop + quadrant mask/color Gather + free-output Pad | Slice [1, 10, 10, 10] F32 4000B input_cropped_f32_internal; Cast [1, 10, 10, 10] F16 2000B input_cropped; Gather [1, 10, 6, 10] F16 1200B v14 |
| 6 | task019 | 21779 | 10fcaaa3 | N | 900 | 3.186 | decoded-clean | 2x ConvTranspose/block upsampler + cyan halo Conv/Where | Slice [1, 10, 12, 12] F32 5760B tdnmb_input12; Cast [1, 10, 12, 12] F16 2880B tdnmb_output_12_fp16; Cast [1, 10, 12, 12] F16 2880B tdnmb_input_fp16 |
| 7 | task289 | 21139 | b91ae062 | N | 900 | 3.156 | decoded-clean | block-upsampler ConvTranspose bank (#80-inverse) selected by color-count k | Slice [1, 10, 15, 15] F32 9000B input_inner; MatMul [1, 10, 15, 15] F32 9000B upscaled; MatMul [1, 10, 15, 3] F32 1800B step1 |
| 8 | task046 | 25984 | 234bbc79 | Y | 1200 | 3.075 | decoded-clean | segment-boundary detector + row-offset Gather/Scatter reconstruction | Slice [1, 10, 3, 20] F32 2400B v38_f32_internal; Cast [1, 10, 3, 20] F16 1200B v38; Concat [1, 10, 3, 16] F16 960B v113 |
| 9 | task343 | 12872 | d8c310e9 | Y | 600 | 3.066 | decoded-clean | period extractor + Tile/Concat free-output extension | Slice [1, 10, 5, 15] F32 3000B safe_name_19_f32_internal; OneHot [1, 10, 5, 15] F16 1500B safe_name_105; Cast [1, 10, 5, 15] F16 1500B safe_name_19 |
| 10 | task029 | 14545 | 1c786137 | N | 700 | 3.034 | decoded-clean | bbox from border color + dynamic fixed-size crop (Gather not Slice) | Einsum [1, 30, 30] F32 3600B ba; ReduceMax [1, 10, 30] F32 1200B n; ReduceMax [1, 10, 30] F32 1200B m |
| 11 | task074 | 21634 | 3631a71a | N | 1200 | 2.892 | decoded-clean | D4 orbit gather/ReduceMax + einsum-final one-hot reconstruction | Cast [1, 30, 30] INT64 7200B t074_u8m_idx_out; ArgMax [1, 30, 30] INT64 7200B t074_u8m_idx; Transpose [1, 30, 30] UINT8 900B t074_u8m_t |
| 12 | task034 | 13273 | 1f0c79e5 | N | 900 | 2.691 | decoded-clean | corner seed detector + four directional ray masks | Slice [1, 10, 9, 9] F32 3240B safe_name_23; Where [1, 10, 9, 9] F16 1620B safe_name_103; Mul [1, 10, 9, 9] F16 1620B safe_name_102 |
| 13 | task264 | 34606 | a8c38be5 | Y | 2500 | 2.628 | decoded-clean | template-detector bank + free-output one-hot/Where tail | Slice [1, 10, 16, 16] F32 10240B crop_f32; Cast [1, 10, 16, 16] F16 5120B input_cropped; Conv [1, 8, 14, 14] F16 3136B v_match_sum |
| 14 | task169 | 16106 | 6e82a1ae | N | 1200 | 2.597 | decoded-clean | component-size count + small color LUT/einsum output | Concat [1, 4, 10, 10] UINT8 400B out4; Slice [1, 1, 10, 10] F32 400B ch5_10; Sub [1, 1, 10, 10] UINT8 100B bg10 |
| 15 | task310 | 20112 | c909285e | N | 1500 | 2.596 | decoded-clean | bbox crop + modular wire-period detector + free-output crop | Gather [1, 10, 8, 30] F32 9600B shifted_rows; Gather [1, 1, 30, 30] F32 3600B lcc_mask_5d; Mul [1, 10, 8, 8] F32 2560B cropped |
| 16 | task035 | 11984 | 1f642eb9 | N | 900 | 2.589 | decoded-clean | cyan bbox + marker-to-edge projection with Where output | Slice [1, 10, 10, 10] F32 4000B v93; Equal [1, 10, 10, 10] BOOL 1000B v208; ArgMax [1, 1, 10, 10] INT64 800B t035_input_color_i64 |
| 17 | task278 | 20348 | b27ca6d3 | N | 1600 | 2.543 | decoded-clean | adjacent-red pair detector + MaxPool/Where green neighborhood fill | Concat [1, 10, 18, 18] F16 6480B output_inner; ReduceMax [1, 1, 30, 30] F32 3600B rmax_full_f32_pre_f16; Cast [1, 1, 30, 30] F16 1800B rmax_full |
| 18 | task088 | 14797 | 3de23699 | N | 1200 | 2.512 | decoded-clean | corner-marker bbox crop + recolor Pad tail | Conv [1, 1, 30, 30] F32 3600B combined_sum; Pad [1, 1, 40, 40] BOOL 1600B padded_mask; Gather [1, 1, 24, 40] BOOL 960B shifted_rows |
| 19 | task341 | 11565 | d6ad076f | N | 1000 | 2.448 | decoded-clean | bar bbox detector + cyan bridge fill + orientation normalize | Slice [1, 10, 10, 10] F32 4000B input_inner_f32_internal; Where [1, 10, 10, 10] F16 2000B output_inner; Cast [1, 10, 10, 10] F16 2000B input_inner |
| 20 | task112 | 14792 | 4938f0c2 | Y | 1300 | 2.432 | decoded-clean | D4/reflection copy around center + output Where tail | Conv [1, 1, 30, 30] F32 3600B color_p1; Where [1, 1, 30, 30] UINT8 900B out_canvas; Where [1, 1, 30, 30] UINT8 900B canvas_r |
| 21 | task349 | 27955 | db93a21d | Y | 2500 | 2.414 | decoded-clean | local radius detector + ray/halo renderer with QLinearConv/Where output | Slice [1, 1, 30, 30] F32 3600B _ch9_f32; Slice [1, 1, 30, 30] F32 3600B _ch0_f32; Where [1, 1, 30, 30] UINT8 900B v145 |
| 22 | task069 | 10000 | 321b1fc6 | N | 900 | 2.408 | decoded-clean | source template extraction + multi-anchor placer/free-output tail | Conv [1, 1, 30, 30] F32 3600B color_full; Equal [1, 10, 10, 10] BOOL 1000B pre_pad_bool; Sub [1, 1, 19, 19] F16 722B score_diff |
| 23 | task390 | 22766 | f8a8fe49 | N | 2200 | 2.337 | decoded-clean | red-frame bbox + affine index transform + sparse Where/Scatter tail | Slice [1, 10, 15, 15] F32 9000B input_cropped_f32_internal; Concat [1, 10, 15, 15] F16 4500B output_active; Cast [1, 10, 15, 15] F16 4500B input_cropped |
| 24 | task059 | 15351 | 29623171 | N | 1600 | 2.261 | decoded-clean | occupancy-pack + ReduceSum/Equal(max) + block fill (#86) | Slice [1, 10, 11, 11] F32 4840B t59_inp11_f32_internal; Concat [1, 10, 11, 11] F16 2420B t59_output_inner; Cast [1, 10, 11, 11] F16 2420B t59_inp11 |
| 25 | task063 | 17147 | 2bee17df | N | 1800 | 2.254 | decoded-clean | row/column occupancy reductions + green Where fill | Slice [1, 6, 14, 14] F32 4704B ch4_9_f32_internal; Concat [1, 10, 14, 14] F16 3920B inner; Cast [1, 6, 14, 14] F16 2352B ch4_9 |
| 26 | task243 | 21328 | 9edfc990 | N | 2500 | 2.144 | decoded-clean | row-byte/u8 packed flood (#88) with output-as-Where | Pad [1, 1, 30, 30] BOOL 900B fill_full; Concat [1, 1, 18, 18] BOOL 324B fill_inner; BitwiseAnd [1, 1, 18, 8] UINT8 144B fill_masked_mid |
| 27 | task206 | 13571 | 88a10436 | N | 1600 | 2.138 | decoded-clean | sprite mask extraction + translation/placer Where | Concat [1, 10, 12, 12] BOOL 1440B t206ci_active_out; Slice [1, 1, 12, 12] F32 576B t206ci_ch6_f32; Slice [1, 1, 12, 12] F32 576B t206ci_ch5_f32 |
| 28 | task365 | 29489 | e50d258f | N | 3500 | 2.131 | decoded-clean | bbox components + red-count selector + dynamic fixed-size crop | Slice [1, 10, 10, 10] F32 4000B t365_input10; Mul [1, 10, 10, 10] F32 4000B output_crop; Gather [1, 10, 10, 10] F32 4000B v250_nm |
| 29 | task238 | 15085 | 9aec4887 | N | 1800 | 2.126 | decoded-clean | bbox sprite crop + diagonal quadrant masks + free-output frame | Slice [1, 1, 22, 22] F32 1936B inp_ch8; ReduceMax [1, 10, 30, 1] F32 1200B rp; ReduceMax [1, 10, 1, 30] F32 1200B cp |
| 30 | task163 | 23679 | 6d0160f0 | Y | 3000 | 2.066 | decoded-clean | mini-grid selector + D2S/SpaceToDepth-style placement + free-output tail | Slice [1, 10, 11, 11] F32 4840B input_N_f32_internal; Sum [1, 10, 11, 11] F16 2420B out_N; Mul [1, 10, 11, 11] F16 2420B input_masked |
| 31 | task279 | 14067 | b2862040 | N | 1800 | 2.056 | decoded-clean | box-closure detector + cyan/blue recolor mask | Slice [1, 1, 16, 16] F32 1024B slice_2; Slice [1, 1, 16, 16] F32 1024B slice_1; Pad [1, 1, 30, 30] UINT8 900B pad_46 |
| 32 | task392 | 13546 | f8c80d96 | N | 1800 | 2.018 | decoded-clean | ring-distance generator + periodic square renderer | Slice [1, 10, 10, 10] F32 4000B v36_f32_internal; Concat [1, 10, 10, 10] F16 2000B all_channels_small; Cast [1, 10, 10, 10] F16 2000B v36 |
| 33 | task064 | 14706 | 2c608aff | Y | 2000 | 1.995 | decoded-clean | axis-aligned ray fill to bbox using cumulative masks | Conv [1, 1, 24, 24] F32 2304B dot_score; Pad [1, 1, 30, 30] BOOL 900B paint30; Or [1, 1, 24, 24] BOOL 576B paint24 |
| 34 | task364 | 14162 | e509e548 | N | 2200 | 1.862 | decoded-clean | small-shape detector bank + per-component recolor Where | QLinearConv [1, 4, 20, 22] UINT8 1760B W4; Slice [1, 1, 20, 22] F32 1760B v56_f32; Pad [1, 1, 30, 30] UINT8 900B cv_pad |
| 35 | task101 | 18236 | 447fd412 | Y | 3000 | 1.805 | decoded-clean | source-sprite extraction + scale-aware ConvTranspose/placer tail | Slice [1, 1, 17, 20] F32 1360B c2_2; Slice [1, 1, 17, 14] F32 952B c1_1; Pad [1, 1, 30, 30] BOOL 900B mask30_421 |
| 36 | task044 | 27232 | 228f6490 | N | 4500 | 1.800 | decoded-clean | box/slot detector + placer Where tail | Slice [1, 10, 10, 10] F32 4000B input10; Concat [1, 10, 10, 10] F16 2000B out10; Cast [1, 10, 10, 10] F16 2000B input10_f16 |
| 37 | task066 | 11769 | 2dd70a9a | N | 2200 | 1.677 | decoded-clean | path/wall flood or directional connector over cyan blockers | Slice [1, 1, 20, 20] F32 1600B m8f; ReduceSum [10, 30] F32 1200B row_counts_all; ReduceSum [10, 30] F32 1200B col_counts_all |
| 38 | task382 | 13298 | f15e1fac | Y | 2500 | 1.671 | decoded-clean | prefix-count/CumSum diagonal drift + orientation normalize | Gather [1, 10, 30, 1] F32 1200B maxcolfull12; Gather [1, 10, 1, 30] F32 1200B maxrowfull11; Pad [30, 30] INT8 900B colorfull168 |
| 39 | task216 | 13995 | 8efcae92 | N | 3000 | 1.540 | decoded-clean | bbox components + red-count selector + crop | Slice [1, 1, 20, 20] F32 1600B ch2c; Slice [1, 1, 20, 20] F32 1600B ch0c; Pad [1, 1, 30, 30] UINT8 900B pcanv |
| 40 | task017 | 11582 | 0dfd9992 | N | 3500 | 1.197 | decoded-clean | zero-param coordinate field + modulo LUT + cutout Where | Conv [1, 1, 21, 21] F32 1764B v26_label_f32; Pad [1, 1, 30, 30] INT8 900B v25_label_padded_i8; MatMul [1, 1, 21, 21] F16 882B v4_selected_kernel_base |

## Compaction-Owned / Skips

These match the card skip doctrine: legacy whale journals at or above 20KB are already owned by the deferred-tail compaction program. They are included in the CSV as `confidence=skip`, but not EV-ranked.

| task | cost | generator | reason |
|---|---:|---|---|
| task286 | 53464 | b782dc8a | task md is 58.8KB |
| task366 | 52001 | e6721834 | task md is 138.3KB |
| task173 | 35163 | 72322fa7 | task md is 25.5KB |
| task158 | 34217 | 6aa20dc0 | task md is 35.3KB |
| task054 | 26043 | 264363fd | task md is 36.3KB |
| task284 | 25855 | b7249182 | task md is 34.0KB |
| task233 | 25658 | 97a05b5b | task md is 186.4KB |
| task191 | 24807 | 7df24a62 | task md is 33.5KB |
| task025 | 23495 | 1a07d186 | task md is 24.5KB |
| task133 | 22293 | 57aa92db | task md is 48.3KB |
| task076 | 22143 | 36d67576 | task md is 28.9KB |
| task110 | 22059 | 484b58aa | task md is 55.0KB |
| task157 | 21938 | 6a1e5592 | task md is 45.9KB |
| task367 | 19465 | e73095fd | task md is 24.4KB |
| task018 | 19175 | 0e206a2e | task md is 239.2KB |
| task324 | 18545 | d07ae81c | task md is 30.9KB |
| task285 | 17654 | b775ac94 | task md is 43.1KB |
| task005 | 16738 | 045e512c | task md is 30.7KB |
| task319 | 16716 | ce602527 | task md is 25.2KB |
| task145 | 16223 | 6455b5f5 | task md is 28.4KB |
| task255 | 13983 | a64e4611 | task md is 76.9KB |
| task182 | 13881 | 776ffc46 | task md is 32.8KB |
| task118 | 13260 | 50846271 | task md is 29.9KB |
| task138 | 13204 | 5daaa586 | task md is 27.4KB |
| task383 | 13112 | f1cefba8 | task md is 26.3KB |
| task187 | 11979 | 7b6016b9 | task md is 56.2KB |

## Batch-Transfer Opportunities

- **Small free-output / marker-to-output tasks**: task342, task362, task371, task183, task253, task343, task029, task035, task034, task019. These should start from a tiny output declaration, one-hot/Where-as-output, and avoid materializing full 10-channel canvases.
- **Block expansion / ConvTranspose family**: task289, task019, task101, and part of task341. Reuse the #80-inverse grouped ConvTranspose mental model; dynamic stride/scale is the only hard part for task289/task101.
- **BBox/count/crop selector family**: task365, task216, task310, task088, task029. Common scaffold: component bbox extraction, scalar count/select, then fixed-size Gather crop instead of dynamic Slice.
- **Flood/ray/line-fill family**: task243, task064, task066, task278, task063, task341. Prefer row-byte or u8 masks where widths are small; use output-as-Where to avoid a 10-channel tail.
- **Symmetry/periodic restoration family**: task074, task112, task343, task017, task392. These look expensive when treated as pixel lookup, but generator source exposes low-rank coordinate/orbit formulas.
- **Mini-grid/occupancy selectors**: task163, task059, task183, task342. They share 3x3 subgrid occupancy/marker indexing; a single SpaceToDepth-style skeleton should transfer.

## Newly Suspected FAMILY-LABEL Misnomers

| task | label says | generator behavior contradiction |
|---|---|---|
| task371 | Pattern_Recognition / Filling Regions | ESTIMATE; generator places blue at col±space and green plus at the center; not filling regions. |
| task362 | Pattern_Recognition / Filling Regions | ESTIMATE; generator uses gray marker count as offset and redraws axes at row+offset and col-offset; not region filling. |
| task046 | Object_Based / Cropping | Object Detection | ESTIMATE; generator offsets each segment in input via offsets[idx] and writes unoffset rows/cols to output; this is dewarping, not object cropping. |
| task343 | Pattern_Recognition / Filling Regions | ESTIMATE; generator reveals only a prefix in input and draws the full repeated pattern to output. |
| task264 | Object_Based / Cropping | Object Detection | ESTIMATE; generator hard-codes 9 sprite shapes and writes output[minisize*(idx//3)+dr][minisize*(idx%3)+dc], so the crop/object label misses the fixed glyph-catalog behavior. |
| task112 | Pattern_Recognition / Filling Regions | ESTIMATE; generator always writes four diagonal copies to output but may show only one/all in input. |
| task349 | Pattern_Recognition / Filling Regions | ESTIMATE; generator draws halos/beams from center/radius tuples, not generic region filling. |
| task163 | Pattern_Recognition / Symmetry Completion | ESTIMATE; generator keys on the yellow cell location, not symmetry completion. |
| task064 | Color_and_Logical / Color Mapping | ESTIMATE; generator explicitly skips diagonal dots and while-fills toward boxcolor; not color mapping. |
| task101 | Pattern_Recognition / Filling Regions | ESTIMATE; generator hides blue pixels from magnified copies in input while output restores them. |
| task382 | Pattern_Recognition / Filling Regions | ESTIMATE; generator increments inc on red rows and places cyan at c+inc; not a fill-region rule. |

Add these to traps `FAMILY-LABEL`: task264, task046, task163, task349, task101, task371, task112, task064, task382, task343, task362. The common failure is reading a broad DeepSeek label as the rule when the generator encodes a specific symbol grammar, marker arithmetic, dewarping, or periodic extension.

## Discipline Notes

- Every active row above has a generator-source one-liner. For misnomers, the notes cite the generator behavior that contradicts the label.
- `floor_estimate` is an **ESTIMATE** for mandatory tensor skeletons only; no ONNX was built, gated, staged, or submitted by this card.
- Real-input profile remains mandatory before staging because dynamic/runtime surfaces can reprice.

## Final Required Closeout

**10 highest-EV tasks:** task342 (+3.939, floor ESTIMATE 250), task371 (+3.903, floor ESTIMATE 350), task362 (+3.649, floor ESTIMATE 300), task253 (+3.534, floor ESTIMATE 700), task183 (+3.410, floor ESTIMATE 500), task019 (+3.186, floor ESTIMATE 900), task289 (+3.156, floor ESTIMATE 900), task046 (+3.075, floor ESTIMATE 1200), task343 (+3.066, floor ESTIMATE 600), task029 (+3.034, floor ESTIMATE 700).

**Cross-task batch-transfer patterns:** small free-output/marker outputs (task342, task362, task371, task183, task253, task343, task029, task035, task034, task019); block expansion/ConvTranspose (task289, task019, task101, task341); bbox/count/crop selectors (task365, task216, task310, task088, task029); flood/ray/line fill (task243, task064, task066, task278, task063, task341); symmetry/periodic restoration (task074, task112, task343, task017, task392); mini-grid/occupancy selectors (task163, task059, task183, task342).

**New FAMILY-LABEL misnomers to add to traps:** task371, task362, task046, task343, task264, task112, task349, task163, task064, task101, task382.


## CAVEAT (added 2026-06-08 after task342 shipped): floor_estimate UNDERCOUNTS the output canvas

task342 shipped at 1,654 vs its R1 floor_estimate of 250. The estimate priced only the RULE EXTRACTION (the 2x2 block logic) and missed the mandatory output contract: the grader validates the full [1,10,30,30] one-hot, which forces a ~900 B label canvas (the #81 Equal-tail floor) PLUS routing buffers. **When reading any R1 floor_estimate for a task whose output is a full-grid [1,10,30,30] one-hot, ADD ~900 B (canvas) + routing.** The estimates are skeleton-level (rule-extraction bytes), useful for RANKING tasks by EV but NOT as a literal floor. Tasks whose output is a tiny grid (<~9.5x9.5, where the small-grid-inversion #81 wins) or a passthrough/substitution (#78 Where-as-output, no canvas) are the exceptions. Do not re-arm a worker to chase an R1 estimate below the output-canvas floor.


## CAVEAT 2 (added 2026-06-08 after task019 shipped): recommended_formulation can name the wrong OP

task019 R1 row recommended a '2x ConvTranspose/block upsampler', but the decoded rule is TILE-REPEAT the grid 2x2 (np.tile), not block-upsample (each cell -> 2x2 block) — different ops (ConvTranspose is wrong for tile-repeat). The worker correctly used Gather-mod tiling. **Treat recommended_formulation as a STARTING HYPOTHESIS, not a spec — verify it against the generator's actual operation before building.** The decoded_rule_1liner is more reliable than the recommended_formulation. (Same spirit as the floor-estimate caveat: R1 is for ranking + a head-start, not gospel.)
