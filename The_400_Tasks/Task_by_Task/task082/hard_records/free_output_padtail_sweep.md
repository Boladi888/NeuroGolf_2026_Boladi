# hard_record: task082 - free-output Pad/Equal tail route is DEAD against the 190 bar

- **Date:** 2026-07-09 (`bo_ng_cdx2_worker2`, task082 only)
- **Live bar used:** 190 from `python tools/ng_current_bar.py --task 82 --json`, authoritative source `manifest`, sha256 `e050837db177e7d6b2e5ad5c4cd76bcf19e85a0aedf3fc265056d75870d80446`, owner Sho, updated `2026-07-09T01:09:17Z`.
- **LIVE graph checked:** `Team_Best_Onnx/task082.onnx` is one grouped `Conv` writing graph `output`, `conv_w[10,1,6,3]` (180 params) + `conv_b[10]` (10 params), memory 0, profile cost 190.

## Attempted route

Prompted route: delete the charged 10-channel prepad/output tail and replace it with a free-output Pad/Equal family: build a one-channel label or mask canvas, then make the final `Equal`/`Pad` be graph `output`.

## Exact failed gate

Failed the prebuild cost-floor screen before any candidate was worth emitting:

- Equal-tail requires a data-dependent predecessor tensor. For a full `[1,1,30,30]` label canvas, the predecessor alone costs at least **900 bytes** even if it is `BOOL`/`UINT8`.
- The small-grid inversion does not save this task: max active grid is `6x15`, so a small decoded one-hot predecessor is `10*6*15 = 900` bytes before the final free `Pad`.
- Either 900-byte tail floor is already **+710 over live 190**, before adding the detector/label producer.

Cheap confirmation on live/current best:

- `_score_v2`: `(macs=0, memory=0, params=190)`, cost `190`.
- `profile_real_cost --arcgen 266`: `static=190 profile=190 (memory 0 + params 190; 265 real inputs)`.
- `ng_gate visible`: `[visible] 265/265 (100.00%) PASS`.
- `prototypes/official_verify.py`: `TOTAL: 265 pass, 0 fail`.

No `truth --full`, `ab --full`, or battery gate was run because there is no candidate with profile cost strictly below live.

## Why it is dead

The free-output trick only makes the final node free. Its input still charges. On task082, the current best already writes the entire `[1,10,30,30]` one-hot as the single graph output with **zero** charged intermediates. Any Pad/Equal tail introduces at least one charged data-dependent canvas, whose minimum possible size is 900 bytes for this output bound. That cannot beat 190, even under the most favorable dtype and before detector costs.

This is separate from the existing `drop_bias_single_conv.md` wall: even if a one-channel label producer existed, the Pad/Equal tail's predecessor memory alone loses to live.

## Remaining hypothesis

Only a new single-output-node formulation with **less than 190 params and zero intermediates** could improve this task. Known candidates are unpromising: Conv/ConvTranspose need the forced `[10,1,6,3]` stencil plus bias; free-output Einsum loses the convolutional shift sharing because the horizontal shift matrices are high-rank over the active width.
