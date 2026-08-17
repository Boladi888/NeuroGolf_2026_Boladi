# task217 SpaceToDepth shape-param shave dead

- Live bar used: 366, sha256 `60156f39344ce78300e20f571abb526e298406a2babfc2acecd22ac8145c999b`, source `python tools/ng_current_bar.py --task 217 --json` / team manifest, owner Sho, updated 2026-07-09T01:09:17Z.
- Attempted route: replace live `Reshape(v_t2, [1,9,1,1])` with `SpaceToDepth(blocksize=3)` on `v_t2 [1,1,3,3] UINT8`. If legal, this would preserve row-major 3x3-to-channel packing and delete the 4-element shape initializer.
- Exact failed gate: in-memory ORT kernel smoke before model build. `SpaceToDepth(UINT8)` failed at session creation with `NOT_IMPLEMENTED : Could not find an implementation for SpaceToDepth(13)`.
- Why dead: the live graph needs the packed tensor in UINT8 to keep the D2S/ConvInteger carrier at 1 byte per cell. A F32 workaround would add 27+ bytes of activation to save 4 params, so it loses even before gates.
- One remaining hypothesis: a legal attribute-only pack op for `[1,1,3,3] -> [1,9,1,1]` would be a direct -4 param win, but `SpaceToDepth` is not that op under local ORT.
