# task354 worker2 tail/microvariant dead record

- live bar: 2,495, SHA256 `9a67e0de227b835f393ff79c39ef3243c59c2ca52bb8cfe15fea3ddade3bc8f8`
  from `python tools/ng_current_bar.py --task 354 --json` (`authoritative_source=manifest`).
- attempted route: cloned v12 pad-axes equal-tail and probed (a) `ScatterElements`
  all-zero index shape reductions, and (b) the tail reorder
  `Equal(labels, ch_ids) -> Pad(output)` instead of `Pad(labels) -> Equal(output)`.
- exact failed gates:
  - `task354_worker2_full_pad_equal.onnx`: visible `[visible] 266/266 (100.00%)  PASS`;
    score `(0, 2450, 45)`, cost 2,495, tie only.
  - `task354_worker2_full_equal_pad.onnx`: visible `[visible] 266/266 (100.00%)  PASS`;
    score `(0, 2550, 45)`, cost 2,595, losing by +100.
  - `task354_worker2_{scalar4d,rank1,scalar}_{pad_equal,equal_pad}.onnx`:
    `score_network_v2(...) -> None`; ORT rejects `ScatterElements` because
    `indices` and `updates` must have the same rank/shape
    (`1 vs 10` at the last dim for the 4-D scalar case).
- why dead: the only legal `ScatterElements` top-row restore keeps the 10-element
  all-zero index initializer. The tail reorder obeys #81 exactly: it deletes the
  900 B U8 `labels30` canvas but materializes a 1,000 B BOOL `onehot10`, so the
  named charged-structure delta is net +100 B. Reduced-index Scatter shapes pass
  static inference but fail in ORT/scorer runtime, so their apparent param shave is
  not score-valid.
- one remaining hypothesis: a real sub-2,495 route must delete part of the 400 B
  F32 `mask5` read or the 800 B four-round masked flood. Tail/index-only edits do
  not have enough legal slack left.
