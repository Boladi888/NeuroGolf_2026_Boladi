# task370 hard record — kernel_dropfar_local_shave

- live bar: 6204, SHA `d9643e949d751f6d3c5266908a35b400ce280ed185d96db71c1b3037ce9c5a94` (`python tools/ng_current_bar.py --task 370 --json`, manifest source, 2026-07-04T05:54:11Z).
- attempted route: shrink 31x31 dynamic projection kernel to 29x29/27x27/25x25, preserving visible support by dropping far ray taps.
- best artifact: `The_400_Tasks/Task_by_Task/task370/prototypes/task370_kernel25_dropfar.onnx`; profile cost 5532.
- exact gates:
  - visible K25: `[visible] 266/266 (100.00%)  PASS`.
  - profile K25: `task370 task370_kernel25_dropfar.onnx: static=5532 profile=5532 (memory 4683 + params 849; 266 real inputs)`.
  - capped truth K25: `[truth] vs ARC-GEN ground truth: 4994/5000 (99.880%)`.
  - capped A/B K25: `[ab] vs submission.zip: match 5263/5266 | divergences=3 | byte_equal=False (first_div=542)`.
- why dead as a reviewer-ready candidate: live on the same 5k truth window was 4997/5000; this route introduces the same 3 A/B divergences and is strictly worse than live truth, so it is only a local-live shave, not a staged win under the ambiguous-LIVE byte-equal/dual-truth rule.
- one remaining hypothesis: a true improvement would need to keep the full distance-15 ray support while saving cost elsewhere, or produce a dual-truth more-correct rewrite rather than clipping far taps.
