# packed_u8_position_decode dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (`python tools/ng_current_bar.py --task 355 --json`, authoritative source `manifest`).
- Attempted route: replace the two row/column `Log -> QuantizeLinear` decodes with one separable packed position scalar, then recover row/column via U8 `Div`/`Mod`. Estimated exact-pack cost was about 378 if U8 packing worked.
- Failed pre-gate: offline exactness check over 267 visible/arc-gen examples found no exact full-output match for row-low or col-low U8 bases `K=2..30`; bases that fit U8 alias row/col bins, while exact bases for 20x20 geometry exceed U8 range.
- Why dead: exact packing needs `K > max(row_or_col)` and `K*max(other_axis)+max(low_axis) <= 255`; task355 can hit 20x20, so no exact U8 packed coordinate exists. U16/INT32 unpack reintroduces enough charged tensors to price above the current two-axis decode.
- Remaining hypothesis: only a non-positional formulation that computes the rectangle score without recovering both row and column extents could beat the 417 graph.
