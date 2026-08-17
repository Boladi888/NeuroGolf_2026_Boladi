# Direct/free-output tail infeasibility for task238

Live bar used: 1999 bytes from `python tools/ng_current_bar.py --task 238 --json`, authoritative source `manifest`, live SHA `14e56fdc4e64dacd34b96212744bdf3aca2c865da71421ff3c95077a27c78813`.

Locked route attempted: rewrite the output tail so the graph does not materialize `tile_bool BOOL [1,10,7,7]` from `Equal(color_idx, ch_values)` before final `Pad`. Candidate `prototypes/task238_worker3_direct_equal_tail.onnx` changes only the v7 tail to `Pad(color_idx, sentinel=10) -> Equal(ch_values) == output`.

Failed gate: cheap score/profile after visible. The candidate passes visible, `[visible] 266/266 (100.00%) PASS`, but costs `score_v2 (0, 1789, 416) cost 2205` and `profile_real_cost static=2205 profile=2205 (memory 1789 + params 416; 266 real inputs)`. It is `+206` vs the live 1999 bar and cannot be reviewer-ready.

Why this direct-output tail is dead:

- Existing pad-last tail pays `tile_bool BOOL [1,10,7,7] = 490` bytes, then the final `Pad` output is free. The output-folded Equal route removes that 490-byte tensor but forces `color_idx30 UINT8 [1,1,30,30] = 900` bytes plus a sentinel scalar. Exact observed tail delta from v7 is `+900 -490 +1 = +411`, taking v7 from 1794 to 2205.
- Even an impossible zero-cost replacement tail cannot hit the reviewer target inside this locked scope: v7 score is 1794 and removing `tile_bool` entirely leaves `1379 - 490 + 415 = 1304` bytes before any replacement tail.
- OneHot-as-output has the same or worse floor: to make padding decode all-false, it still needs a padded sentinel index plane, and legal OneHot indices would require a full INT32/INT64 cast/index plane before the free output. That is no cheaper than the 900-byte Equal value canvas and worse than the existing 490-byte small BOOL tile.
- Where-as-output/free-output substitution does not apply because task238 output is fully generated in the top-left, not `input` with a sparse masked substitution. A final `Where` would need either a full correct base canvas or a full replacement/channel selector, which is equivalent to the one-hot/value tile it is trying to avoid.
- #109/#119 separable free-output rank gate fails. Probe `prototypes/probe_task238_direct_tail_rank.py --n 3000` found max per-channel rank distribution `{2:25, 3:230, 4:1127, 5:1017, 6:601}` and sum ranks `min=9 p50=14 max=22 mean=14.71`. Exact full-output separable factors need at least `T=6` terms on observed generator-valid cases; the F16 factor floor alone is `6 * (10 + 2*30) * 2 = 840` bytes before detector, palette, or any computed selectors. With F32 input-carrier factors it is higher. The arbitrary multi-color scatter is not a low-rank tail.

Remaining hypothesis: only a non-tail rewrite that deletes the upstream detector/palette/slot-map construction as well as the tail could plausibly move below 700. That is outside this locked rework request; within direct/free-output tail rewrites, the route is dead.
