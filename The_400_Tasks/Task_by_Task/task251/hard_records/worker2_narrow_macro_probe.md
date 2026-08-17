# task251 worker2 narrow macro probe

- Live bar used: 3740 bytes, sha256 `e99be2e84e3dde0bdc065418034917aa39cada740a2507ba38ac11b719db0844`, from `python tools/ng_current_bar.py --task 251 --json` manifest source.
- Attempted route: keep Vamshi's row-packed UINT16 R10 flood depth, then shrink the per-round/tail macro using task-local probes in `prototypes/worker2_narrow_probes/`.
- Exact failed gates:
  - `bitshift_rounds` and `bitshift_rounds_seed`: ONNX checker OK, `score_network_v2` returned `None`; direct ORT gate error was `NOT_IMPLEMENTED : Could not find an implementation for BitShift(11) node with name 'left0'`.
  - `slice_concat_neighbors`: ONNX checker OK, `score_network_v2` = memory 4042 + params 120 = cost 4162, which is +422 over live before visible.
  - `tail_pad_two_stage` and `tail_pad_one_stage`: ONNX checker OK, `score_network_v2` returned `None`; direct ORT gate error was `NOT_IMPLEMENTED : Could not find an implementation for Pad(18) node with name 'packed_1x10x12x1'` / `packed_1x10x30x1`.
  - `max_vertical` and `max_vertical_horizontal`: ONNX checker OK, `score_network_v2` returned `None`; direct ORT gate error was `NOT_IMPLEMENTED : Could not find an implementation for Max(13) node with name 'vertical0'`.
- Why dead: the only exact supported row-neighbor replacement increases memory; the attractive single-op shrinks (`BitShift`, `Pad`, `Max`) are not executable on UINT16 in the scoring runtime, and casting to wider supported integer types would more than erase the theoretical savings.
- One remaining hypothesis: a true segmented rewrite would need to compute horizontal run closure with only scorer-supported UINT16 `Bitwise*`, `Mul`, `Div`, `Gather`, and `Concat` primitives, and must first beat the live 216-byte per-round macro in a one-round price probe without reducing R10.
