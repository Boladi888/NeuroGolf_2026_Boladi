# period6_tail_gate_floor

- live bar: cost 168/profile 168, sha256 `467a0931c6aebf9800e570cf741cfed56e55136807e177fe6a4418a74104fd80` (`python tools/ng_current_bar.py --task 231 --json`, manifest source)
- attempted route: exploit the generator invariant that every nonzero row has period 1, 2, or 3, so the first six input columns can reproduce the color pattern across the output; then avoid the current dynamic `[30]` INT32 Gather index.
- exact failed gate: prebuild floor check. A fixed-period output still needs a dynamic tail gate at column `2W`; formula-generated indices price above 286 (`Mod/Where` route), a depthwise fixed-shift Conv needs at least 190 weight params before tail gating, and any Gather-then-mask/Where tail makes the full `[1,10,30,30]` F32 Gather result a charged intermediate.
- why dead: period reuse removes color selection, but not the width-dependent blank tail. The current 168 graph pays exactly one `[30]` INT32 index vector plus scalar selectors; indexless variants either visibly overpaint the tail or charge a full output-sized plane.
- one remaining hypothesis: a terminal op with native wrap/tile plus dynamic crop/blank semantics would be needed to beat 168; ONNX `Pad`/`Tile`/`Resize` do not provide that without producing a charged large intermediate or wrong output shape.
