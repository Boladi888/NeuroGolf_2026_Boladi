# task081 hard record — direct output QLinearConv

- live bar used: cost 395, sha256 `455d15b1b8ab854784d8462e31b376b7bd028655ab69505c6aeba85beeb663b2` from `python tools/ng_current_bar.py --task 81 --json` (authoritative source: manifest, owner Boladi, updated 2026-06-22T16:50:27Z).
- attempted route: replace the 2-feature hidden stage with `Slice(ch0|ch8, 7x7) -> Cast(UINT8) -> one final kxk QLinearConv output`, scanning k=2/k=3 alignments, all UINT8 input zero-points, and an optional real-valued bias term.
- failed gate: `python The_400_Tasks/Task_by_Task/task081/prototypes/probe_direct_qlc_linear.py` returned `direct_family_feasible=False`; every scanned `source={cyan,bg}` / k / pad alignment printed `no full separator`.
- why dead: even over the 264 bundled task-shaped pairs, no real-valued linear separator exists for all required output channels `{0,1,8}` from a single local ch0/ch8 patch. Bounded int8 QLinearConv weights cannot rescue a real-infeasible family.
- remaining hypothesis: only a non-linear intermediate basis can still beat 395; it must either avoid the 98 B two-feature tensor entirely or reduce the forced `x8`/`cyan_u8` read, because the known #98 dilated read is correct but costs 427.
