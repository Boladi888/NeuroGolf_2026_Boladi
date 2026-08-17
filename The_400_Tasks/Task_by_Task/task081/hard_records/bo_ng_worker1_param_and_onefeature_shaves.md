# task081 hard record — bo_ng_worker1 param and one-feature shaves

- live bar used: cost 395, sha256 `455d15b1b8ab854784d8462e31b376b7bd028655ab69505c6aeba85beeb663b2` from `python tools/ng_current_bar.py --task 81 --json` (authoritative source: manifest, owner Boladi, updated 2026-06-22T16:50:27Z).
- attempted route A: keep the staged two-feature architecture but delete cheap params: final `feat_zp`, hidden bias, or hidden-bias replacement by a shared hidden-output/final-input zero-point.
- failed gate/probe A: `python The_400_Tasks/Task_by_Task/task081/prototypes/probe_task081_param_shaves.py` (after an initial unvectorized timeout) returned `shared_hidden_yzp_no_bias_hits: []`; baseline `final_zp_0.all_channels_separable=false` because channel 1 has no separator; `drop_hidden_bias.n_final_zp_hits=0`.
- why A is dead: the current two-feature basis needs a nonzero final feature zero-point to separate the hole class, and removing the two-element hidden bias destroys all final-zero-point separators. No initializer element drops without changing the basis.
- attempted route B: skip the hidden feature stage entirely with one final local QLinearConv over cyan/background patches.
- failed gate/probe B: `python The_400_Tasks/Task_by_Task/task081/prototypes/probe_direct_qlc_linear.py` returned `direct_family_feasible=False`; every scanned `source={cyan,bg}`, `k={2,3}`, pad alignment had no full real-valued separator.
- why B is dead: a single local linear output kernel cannot distinguish all required bg/hole/cyan states; bounded int8 QLinearConv weights cannot rescue a real-infeasible family.
- attempted route C: use only one of the existing hidden features plus a boundary-safe 2x2 final QLinearConv, priced around 356 if feasible.
- failed gate/probe C: `python The_400_Tasks/Task_by_Task/task081/prototypes/probe_task081_one_feature_final2x2.py` returned `one_feature_final2x2_hits=[]` for feature 0/1, final zero-points 0..31, and pad starts 0/1 including all 30x30 outside cells as no-channel negatives.
- why C is dead: neither current hidden feature alone contains enough information for a 2x2 final linear kernel, and the padded outside constraint removes the edge-leak alignments.
- remaining hypothesis: a genuinely new learned one-feature basis plus 2x2 final could still be searched with MILP/CP-SAT, but it is not the current basis and must prove boundary safety; the known ways to remove `x8`/`cyan_u8` are direct-output infeasible or the #98 dilated read at cost 427.
