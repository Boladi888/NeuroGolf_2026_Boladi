# task173 hard record - free-output Einsum renderer lower bound

- **bar used**: team-best/profile cost `8710`, sha256 `27f31830cf75f60bdcf0273b403dcb804b8f4b6950817091f50e761be3a57940`.
- **bar source**: `python tools/ng_current_bar.py --task 173 --json`, authoritative source `manifest`; workbook agrees at `8710` and local task_state/live_costs/library DB are stale.
- **attempted route**: replace the runtime F32 `dynamic_w [10,10,3,3]` render with a free-output separable `Einsum` that uses decoded `(shape, center_color, frame_color)` triples as small operands and writes the final one-hot output directly. This is the natural #67/#109/#119/#132 alternative to a terminal data-dependent Conv.
- **exact failed gate**: pricing/lower-bound gate before build. Keeping the current detector/decode surface and deleting the current render-specific pieces gives only this optimistic base:
  - remove memory: `dynamic_w 3600 + weight_updates 432 + weight_indices 192 + weight_indices_i32 96 + 4 index blocks 96 = 4416`.
  - remove params: `base_w 900 + update_table 180 + bias 10 + zero_col 3 = 1093`.
  - non-render detector/decode lower bound: `8710 - 4416 - 1093 = 3201`.
- **why dead**: the cheapest separable-Einsum render needs full 30-wide translation bases for local offsets `{-1,0,+1}` in both row and col:
  - row/col shift bases: `(3 * 30 * 30) + (3 * 30 * 30) = 5400` params.
  - dynamic F32 channel selectors for up to 3 triples: at least `center_sel [3,10] + outer_sel [3,10] = 60` F32 values = `240` bytes, because a final Einsum that reads the F32 input cannot mix U8/BOOL selector operands.
  - dynamic F32 shape/offset selector: at least `[3,3,3] = 27` values = `108` bytes.
  - optimistic lower bound: `3201 + 5400 + 240 + 108 = 8949`, already `+239` above the `8710` live bar before adding any node scaffolding, background suppression, validity masking, or profile overhead.
- **prompt constraint note**: this route also keeps the current full `detect_map [1,1,23,23]` F32 detector, so it would fail the prompt's fast-probe kill rule even if the renderer priced better.
- **one remaining hypothesis**: a non-Einsum final op that keeps translation invariance, writes output for free, and routes dynamic colors without either a full F32 `dynamic_w` or dense color/spatial carriers. No scoreable ONNX op class identified in this pass.
