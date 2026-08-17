# task076 — no-sentinel ScatterElements tail dead

- **Live bar**: 12,539 profile cost, owner Sho, SHA `ec95ba95e46eeb94984d1a20b525eaf4f94851b8fd63b04f53a9618ddabdd773`, from `python tools/ng_current_bar.py --task 76 --json`.
- **Effective task-local bar**: 12,030 profile cost, staged `task076_int8coords_i32idx_cost12030`, SHA `aa5292e62462b76d974cb8b9527b16c475307f03836bd21b0d4659944846df32`.
- **Attempted route**: remove the 16th sentinel row in the final flat color-map scatter, reshape the scatter output as `15x15`, and change final `QLinearConv` pads from `[0,0,14,15]` to `[0,0,15,15]`.
- **Failed gates**:
  - `w4_scattermax_no_sentinel`: `visible 266/266`, profile `11713`, but `ab --n 2000` vs 12,030 parent = `match 2242/2266 | divergences=24 | byte_equal=False`; `truth --full` = `195605/200000 (97.802%)`, below the parent's documented `197902/200000`.
  - `w5_scatter0_restore`: inactive writes to index 0 with original index-0 value; `visible 263/266 (98.87%) FAIL`.
  - `w6_scatter_original_restore`: inactive writes original value at each `selected_safe_indices`; `visible 244/266 (91.73%) FAIL`.
- **Why dead**: the sentinel row is not just padding cost. It prevents clipped/out-of-active and inactive scatter writes from perturbing cells that the current ambiguous-class parent either keeps byte-equal or at least misses in its own way. Max-reduction changes hidden behavior on the full ARC-GEN corpus; overwrite-restoration variants fail visible.
- **One remaining hypothesis**: a no-sentinel tail would need a deterministic duplicate-safe scatter that writes active updates while treating inactive/out-of-bounds positions as true no-ops without materializing a 15x15 mask. Current ONNX `ScatterElements` forms tested here do not provide that at lower cost.
