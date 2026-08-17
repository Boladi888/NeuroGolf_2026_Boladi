# small_output_shape dead route

Live bar used: cost 190, sha256 `b3dda820c5b93cb43f60e655a297068590e5a584d36466d5b0201784d0002861` from `python tools/ng_current_bar.py --task 287 --json` / manifest.

Attempted route: emit only `[1,10,16,16]` from the bit-factor `Einsum`, replacing the current `[30,2]` mirror-bit maps with `[16,2]` maps. Estimated params would be about 106 if accepted.

Exact failed gate: source-read shape gate. `neurogolf_utils.convert_to_numpy` encodes expected outputs as `[1,10,30,30]`, and `verify_subset` compares `run_network(...)` to that tensor with `np.array_equal`. A raw `[1,10,16,16]` output therefore shape-mismatches every official visible pair.

Why dead: padding the smaller `Einsum` result back to 30x30 would make the `[1,10,16,16]` F32 `Einsum` output a charged intermediate of 10,240 bytes before the terminal `Pad`, far above the 190 bar.

One remaining hypothesis: only an output ABI or scorer change that allowed 16x16 tensor outputs would revive this route.
