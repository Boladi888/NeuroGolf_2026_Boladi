# Stale 900B Line-Mask Route

- Live bar used: cost 262, sha256 `0394d8786c55b024ef6b7188817b8267b0c4d5c834cd35744a8e4ed83454a978`, source `python tools/ng_current_bar.py --task 51 --json` / manifest, owner Vamshi, updated 2026-07-02T02:35:15Z.
- Attempted route: narrow or delete the old 30x30 BOOL `line_mask`/comparison-mask cluster described by the older 1994-cost task note.
- Exact failed check: graph inspection of `Team_Best_Onnx/task051.onnx` shows the promoted 262 graph has only scalar intermediates and five Concat basis vectors; `largest_intermediate.py Team_Best_Onnx/task051.onnx --top 30` reports total static intermediates 128 bytes and no 900-byte `line_mask_b`.
- Why dead: the current best already moved the full output assembly into the final free-output Einsum, so there is no charged 900-byte mask left to narrow.
- Remaining hypothesis: continue scalar/free-output parameter factoring only, especially `term_t_sel_swapped`, `term_p_sel`, `color_basis`, `linear_pick`, and `one_m`; avoid rebuilding old line-mask families unless a future live graph reintroduces a charged canvas mask.
