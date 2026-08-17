# Dead route - no-keep zero-sum phase kernel

- Live bar: cost 78 from `python tools/ng_current_bar.py --task 7 --json`; source `manifest`, owner Boladi, sha256 `25e729678c62f5f7472cd86857ada940dd8dde5a6e5a734b79e227ca5ba31089`, updated `2026-06-10T01:24:21Z`.
- Attempted route: remove v11's `keep_nonzero_color[10]` mask and rely on a two-row trig phase kernel to keep channel-0 background logits nonpositive. Existing probe: `The_400_Tasks/Task_by_Task/task007/prototypes/task007_v12_direct_trig_no_keep_probe.onnx`.
- Candidate/probe cost: `score_network_v2` reports `(macs=0, memory=0, params=68)`, so this would beat 78 if correct.

## Exact failed gate

```text
python tools/ng_gate.py visible --task 7 --cand The_400_Tasks/Task_by_Task/task007/prototypes/task007_v12_direct_trig_no_keep_probe.onnx
[visible] 0/266 (0.00%)  FAIL
```

One visible miss is hidden 0 by traps `VISIBLE-FAIL`; 0/266 kills the no-keep representation family. No truth/ab/profile gates were run after the visible-zero.

## Why this family is dead

With only the existing two phase rows (`cos`, `sin`), any per-residue phase kernel has zero sum across the three residue offsets. Normalize the true-residue score to `K0 = 1`; then the two wrong-residue scores are `K1 = -a`, `K2 = -(1-a)` for some `0 <= a <= 1`.

For the 7x7 generator, total cell counts by `(r+c) mod 3` are `[17, 16, 16]`. The selected visible diagonal lengths can be:

```text
residue 0: {1, 4, 7}
residue 1: {2, 3, 5, 6}
residue 2: {2, 3, 5, 6}
```

So channel-0 background counts can include `[16, 14, 14]` when residue 0 selected length is 1 and residues 1/2 selected lengths are 2. The channel-0 output for residue 0 is:

```text
16*K0 + 14*K1 + 14*K2
= 16 - 14*a - 14*(1-a)
= 2
```

That is strictly positive for every zero-sum two-row kernel, so channel 0 lights and the official decoder produces a wrong multi-hot/extra color. This is not just the specific v12 coefficient choice; it rules out the whole no-keep two-row phase family.

A non-zero-sum kernel such as `2*cos(diff) - 1` would fix this correctness issue, but it needs a constant active-region term. In one output `Einsum`, that requires either a third `[30]` active basis row or a separate `[30]` active vector used on output row/col, adding at least 30 counted params while saving only the 10-param keep mask. That cannot beat the current 78 bar.

## Remaining hypothesis

Sub-78 would need a genuinely different counted representation for the forced active-output basis or the channel-0 suppressor, not another two-row trig/no-keep coefficient search. Sparse/dtype packing is already blocked by traps `8e` and `15`, and the task note records sparse initializer scoring as `None`.
