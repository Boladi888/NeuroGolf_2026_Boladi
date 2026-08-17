live bar: cost 100, sha256 91294db3a76db1cf55efd0eeb6e6c475503684a7d282ec1b5611be5a48729242

attempted route: reviewer-locked #89 free-output contraction final for center-copy plus same-channel left*right and top*bottom endpoint products.

fresh separability probe: `python The_400_Tasks/Task_by_Task/task314/prototypes/task314_einsum_separability_probe.py` generated 64 fresh ARC-GEN pairs with seed_offset 731400. It found 113 fill cells, all 113 requiring background-channel suppression; center+endpoint products without suppression matched only 10/64 pairs, while suppression matched 64/64.

priced skeleton: `python tools/agent_helpers/profile_real_cost.py --task 314 --cand The_400_Tasks/Task_by_Task/task314/prototypes/task314_einsum_channel_route_skeleton.onnx --arcgen 266` -> `task314 task314_einsum_channel_route_skeleton.onnx: static=100 profile=100 (memory 0 + params 100; 266 real inputs)`

why dead: the first nonseparable state is `fill_any_nonbg -> output channel1 off` while also routing each same-channel endpoint product to its own output channel. The cheapest dense channel-route operand for this is `[10,10]` = 100 params; its matrix rank is 8, so factorizing it costs at least 8*(10+10)=160 params. This reaches the live 100 bar before any row/column/cell-phase shift selectors or center-copy selector tail are added.

one remaining hypothesis: only a non-ONNX-accounted way to share the live Conv's threshold/routing with fewer than 100 charged selector elements would reopen this path; #87 warns that splitting the free-output Conv materializes intermediates, and the skeleton confirms the selector bank itself is already at the bar.
