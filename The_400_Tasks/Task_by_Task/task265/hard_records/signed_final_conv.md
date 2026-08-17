# signed_final_conv

- live bar: 3637, sha `0c5b087e1708ee1e348809c46c3bc1e5dcb94234fc4c11444875f9696cc4dcf3` (`ng_current_bar.py --task 265 --json`, manifest source).
- attempted route: signed/local final-output Conv skeleton, folding the red mask into the graph output to avoid the 900-byte `Where` condition.
- exact failed gate: pre-build visible feasibility. A 3x3 output Conv has two official visible local-context conflicts (`same 3x3 input patch -> labels {0,2}`), so it cannot pass visible. 5x5 and 7x7 local linear classifiers have no context conflicts but LP feasibility for channels 0 and 2 is infeasible on the official set.
- why dead: one linear output node cannot express the red/background nonlinear association, even when the visible local context is widened beyond 3x3.
- one remaining hypothesis: a two-stage signed/bitset association could still work if the nonlinear association state is computed in one compact plane and the final assembly avoids reintroducing an equivalent full output mask.
