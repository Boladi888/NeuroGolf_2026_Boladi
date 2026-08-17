# task342 worker2 scratch plan

Bar: workbook + team manifest both report current team-best cost 126, creator Vamshi, sha256 `9a749314b24fdbf582eee9c05b47dc9361a441c5baf00c18afac8f7c6d499d27`, updated 2026-07-02T02:35:15Z. Strict staging bar is therefore `<126`; stale local notes at 1654/606 are not the bar.

Decoded rule: input is a fixed 10x10 active grid embedded in the 30x30 one-hot canvas. One cyan `8` 2x2 block is present, with one non-cyan marker in each quadrant around it. Output is a full `[1,10,30,30]` one-hot canvas whose active 10x10 grid is black except the cyan block cells become TL/TR/BL/BR marker colors.

Falsification cases:
- cyan block near a 10x10 edge: output must still be embedded at top-left and zero outside active grid;
- marker colors include channel 0? Generator/visible examples use nonzero markers only; final renderer must still not leak padding/ch0 outside active cells;
- two markers share row or column with the block boundary: quadrant classification must use strict side of the 2x2 block, not nearest marker order.

Families:
- A. Current best family: one `Einsum` writes graph output directly, memory 0, params 126. Try only parameter-count compression: sparse initializer acceptance, selector/mode factor removal, or equation refactor that keeps one output op. Target `<126`; charged old `Where/ReduceSum/Conv` surface is already deleted.
- B. Alternative one-op renderer: a new single `Einsum`/Conv-like formula. Any legal formula still needs output color selection, local/background separation, row/col basis, and cyan block coordinates; rough lower bound appears near current 126 unless a selector can be algebraically folded.
- C. Multi-node structural rebuild: dead versus this bar unless it somehow has total intermediate+params `<126`. Prior slot-scatter tail alone has 900B `label30`, so it cannot be reviewer-ready below 126.

Gate plan: cheapest-first. For probes, run static score first; run visible/profile only if static is `<126` or the probe is needed to prove a route dead. No full truth/AB unless a candidate visibly passes and profile cost strictly beats 126.
