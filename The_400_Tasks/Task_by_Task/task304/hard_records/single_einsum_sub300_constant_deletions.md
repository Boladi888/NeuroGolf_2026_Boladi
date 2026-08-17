# task304 single-Einsum sub-300 constant-deletion probes

## Live bar
- Authoritative team-best bar: cost/profile_cost 300 from `python tools/ng_current_bar.py --task 304 --json`.
- Source: manifest, owner Vamshi, updated 2026-07-12T15:15:54Z.
- Live/team-best SHA-256: `33276d90fabe23c8b107ab4823b515e1f6f91e010712549c99ce85e44fd87807`.
- Team-best local checks run this pass:
  - `python tools/agent_helpers/profile_real_cost.py --task 304 --cand Team_Best_Onnx/task304.onnx --arcgen 266` -> `task304 task304.onnx: static=300 profile=300 (memory 0 + params 300; 266 real inputs)`
  - `python tools/ng_gate.py visible --task 304 --cand Team_Best_Onnx/task304.onnx` -> `[visible] 266/266 (100.00%)  PASS`
  - `python tools/ng_gate.py truth --task 304 --cand Team_Best_Onnx/task304.onnx --n 200 --json` -> `correct=200 total=200 pct=100.0`
  - `python tools/ng_gate.py ab --task 304 --cand Team_Best_Onnx/task304.onnx --n 200 --json` -> `match=466 total=466 divergences=0 byte_equal=True`

## Attempted routes
- `prototypes/probe_single_selector_modonly.onnx`: one-node Einsum, removed the 90-param `floor` selector and used `mod` everywhere. Static cost 210 = params 210, memory 0. SHA-256 `70aaab9c8abe6a85f94424f5e1e9a3f735b1c112ba71c340c447781d86da1257`.
  - Failed gate: `python tools/ng_gate.py visible --task 304 --cand The_400_Tasks/Task_by_Task/task304/prototypes/probe_single_selector_modonly.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
  - Dead because one selector cannot carry both output block-coordinate and inner-tile coordinate roles in the accepted one-Einsum representation.
- `prototypes/probe_no_channel_mixer.onnx`: one-node Einsum, removed the 100-param `channel` mixer and tied tile input channel directly to output channel. Static cost 200 = params 200, memory 0. SHA-256 `46b9f8627791a8c5403588da91d9c89132b04a34b6bfd85e91b9b507013954e3`.
  - Failed gate: `python tools/ng_gate.py visible --task 304 --cand The_400_Tasks/Task_by_Task/task304/prototypes/probe_no_channel_mixer.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
  - Dead because the channel mixer is the only charged structure that emits the required positive background channel while preserving foreground channels.
- `prototypes/probe_positive_center_only.onnx`: one-node Einsum, removed the negative all-ones center/sign basis and kept only the positive 3x diagonal center term. Static cost 289 = params 289, memory 0. SHA-256 `25b594b0efd292ff656f6a1c994ad475a28a96fdcbafc4978e79c3e99e8d1216`.
  - Failed gate: `python tools/ng_gate.py visible --task 304 --cand The_400_Tasks/Task_by_Task/task304/prototypes/probe_positive_center_only.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
  - Dead because the negative center/sign basis is load-bearing for the cubic unique-mode test; the positive term alone cannot reject non-mode cells.

## Remaining hypothesis
The only visible-plausible sub-300 route left is not a deletion of one of the current constants; it would need a new final-op family or an ORT-accepted structured/sparse constant mechanism that preserves the current one-node behavior while pricing below `mod(90) + floor(90) + channel(100) + center/sign(20) = 300`. Sparse Einsum inits were already rejected in the prior hard record, so the next viable test should be a genuinely different final op, not another selector/channel/center deletion variant.
