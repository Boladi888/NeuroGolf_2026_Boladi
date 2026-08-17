# task048 no-candidate packet - worker-w048 - 2026-06-24

## Verdict
- verdict: no-candidate
- reason: the only local below-team-best artifact, v8 full-box k5 cost642, is already confirmed public hidden-failed in the team tree.
- staged path: none for a candidate; this folder records the no-candidate audit.

## Source of truth
- team-best source: `../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`
- task048 team-best: cost 686, creator Boladi, SHA256 `6d123e4a89db56cf2db605f86ab756c341af1756e4728172361c15eb5ad67299`
- team-best artifact: `../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task048.onnx`
- local below-best artifact checked: `staged_submits/task048_v8_fullbox_k5_cost642/task048.onnx`
- v8 SHA256: `9791950272084368dc499a9caf1283c4bde835b5c3b64d837769575a96b86e82`
- disqualifying evidence: same SHA exists at `../K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Boladi/_hidden_failed/task048.onnx`; sidecar says `HIDDEN-FAILED: expected +0.15 if correct, landed -36.00`.

## Gate Results
- team-best visible: `python tools/ng_gate.py visible --task 48 --cand ../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task048.onnx` -> 270/270 PASS.
- team-best profile: `python tools/agent_helpers/profile_real_cost.py --task 048 --cand ../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task048.onnx --arcgen 266` -> static=686 profile=686, memory=578, params=108.
- v8 visible recheck: `python tools/ng_gate.py visible --task 48 --cand staged_submits/task048_v8_fullbox_k5_cost642/task048.onnx` -> 270/270 PASS.
- v8 score_v2 recheck: `(0, 534, 108)`, cost=642.
- v8 profile recheck: `python tools/agent_helpers/profile_real_cost.py --task 048 --cand staged_submits/task048_v8_fullbox_k5_cost642/task048.onnx --arcgen 266` -> static=642 profile=642, memory=534, params=108.
- prior v8 full gates from staged manifest: truth_full 199,434/200,000; A/B vs v7 divergences 745, byte_equal=False; 715 v7 misses fixed, 30 new misses introduced, 536 shared misses.

## Dominant Charged Tensors
- v7/team-best cost: memory 578 + params 108 = 686.
- `packed_f` F32 [1,2,8,1] = 64 B forced F32 pack bridge.
- `packed_u8` + `reds` + `cyans` + `pass_m` = 40 B around the packed red/cyan substrate.
- six row-byte flood rounds dominate memory: 5 masked rounds at 72 B each plus final unmasked expansion, with every lane tensor U8 [1,1,8,1] = 8 B.
- params 108: `w_pack` 80, `w_out` 10, `zeros8` 8, pad axes/pads 5, scalar constants 5.

## Techniques Tried And Rejected
- v8 full-box k5 cost642: rejected because exact SHA is hidden-failed publicly, despite local visible/profile/truth gates.
- k4 full-box: already measured visible-fail 266/270; train#2 needs exactly 5 full-box 8-conn rounds by BFS.
- 5x5 accelerated flood schedules on 20k corpus: rejected. Best tested schedule `5_3_3_3` got 18,430/20,000 correct, far below v7's 19,858/20,000 on the same slice.
- Path-B non-flood formulations: archived price check still applies. Plain 8x8 MaxPool flood is at least 192 B/round, CC labeling needs >=128 B label planes plus histogram/index machinery, and MatMulInteger reachability needs INT32[64] vectors or 64x64 adjacency materialization.

## Floor Scope
- No algorithm-independent floor is claimed.
- The safe byte-equal/routed row-byte floor remains 686 after the v8 hidden-fail.
- Any future sub-686 attempt must be either byte-equal to the shipped 686 artifact or a genuinely different formulation with stronger evidence than local filtered-clean truth.

## Librarian Lesson
- None. This is task-specific hidden-fail bookkeeping, not a durable general technique.
