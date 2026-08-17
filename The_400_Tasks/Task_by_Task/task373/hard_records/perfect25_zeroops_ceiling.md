# task373 — perfect-25 cost<=1 ceiling: WALL (floor = 30)

- **Live bar at audit**: cost **30** (manifest, owner Vamshi), sha256 `efaa28a5365fb08d5a3f9290960878f7382fbbaf6c9a3a5dac85bb8dbff3ba3c`.
  Source: `python tools/ng_current_bar.py --task 373` (authoritative=manifest, not stale).
- **Audit date**: 2026-07-07 | worker `bo_ng_cld_worker2` (Claude lane).

## Transform (decoded from `GUI/client/public/tasks/task373.json`)
Input 2×6: row0 = all color A, row1 = all color B.
Output 2×6: row0 = `[A,B,A,B,A,B]`, row1 = `[B,A,B,A,B,A]`.
So **`Output[i][j] = Input[(i+j)%2][j]`** — a column-parity row-swap (checkerboard
interleave of the two constant rows). NOT identity/transpose/flip/rotate of the
whole grid.

## Cost model recap (neurogolf_utils)
`cost = memory + params`. `params` = element count of initializers + Constant-node
values only (other nodes' attributes — Einsum equation, Transpose perm, opset≤9
Slice starts/ends, Pad pads — are FREE). `memory` = every intermediate tensor
charged in **bytes** (×itemsize), input/output excluded. Live best = single Einsum,
0 intermediates → memory 0, params 30.

## Attempted routes (3 independent formulations)
1. **Zero-param single-node family (the cost<=1 rumor)** — Identity, Transpose(H↔W,
   perm 0,1,3,2), flip_W, flip_H (Slice step −1). Built in `prototypes/zeroops_probe.py`.
   `ng_gate visible --task 373`: **all 0/75 (0.00%) FAIL**. One visible miss = certain
   hidden 0 (traps §VISIBLE-FAIL) → **family DEAD**. The cost<=1 perfect-25 route does
   not exist: the output is a genuine checkerboard, not a whole-grid symmetry.
2. **Sub-30 single-Einsum (shorten the parity vector)** — length-6 `sign` in the live
   equation `ncab,a,h,w,ndhe->nchw`. **Rejected by ONNX shape inference**: operands
   indexed by canvas axes (a/h/w) MUST be length 30 to match [1,10,30,30]. The period-2
   parity vector is forced to length 30 → params ≥ 30 for the whole single-Einsum family.
3. **Procedural / multi-node parity** — Range(0,30)+Mod(2) materializes a [30] parity
   intermediate (≥120 B F32 / ≥30 B int8) PLUS scalar params ⇒ ≥124 (or ≥34) > 30.
   Any elementwise mask·input blend materializes a full-canvas [1,10,30,30] F32
   intermediate = 36 000 B (historical crop/concat routes cost 747–1559). All ≫ 30.

## Forced-component floor argument (why 30 is the wall)
The output alternates with period 2 along columns. The input carries **no** column-varying
signal (each row is a single constant color), so the parity must be injected externally.
The only memory-0 path is a single output-producing node (any intermediate is charged in
bytes; the smallest useful full-canvas op yields ≫30 B). A single node that emits the
checkerboard needs a data payload = the length-30 parity, minimally 30 chargeable elements
(initializer or Constant). Procedural generation trades those 30 params for a ≥30 B
intermediate + scalars (no better). **Hence cost ≥ 30; the live Einsum already sits on
that floor.** No candidate ships below 30.

## Verdict
**WALL at 30. No stage.** Do not re-run the zero-op / identity / transpose sweep — it is a
proven visible-zero. Do not retry crop/concat (see crop_to_size_advisory). Only revisit if
the cost model changes (e.g. attributes start carrying free data payloads) or a genuinely
new memory-0 single-op emitter for period-2 selection appears.
