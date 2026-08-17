# task018 — DEAD ROUTE: MorphologyPool / local-neighborhood rewrite (floor_est ~1500 is fantasy)

**Worker:** bo_ng_cld_worker3 (Claude lane) · **Date:** 2026-07-12 · **Verdict: DROP / WALL at 19,125**
**Sibling record:** [[pad_concat_predicate_false_floor_2026-07-10]] (same task, Pad/Concat SWEEP angle, also DROP).

## Live bar (re-measured this session)
- **live_cost = 19,125** — `ng_current_bar.py --task 18`: authoritative_source=`manifest`, owner Boladi,
  sha256 `d6567a94331bf690c8e85c3361b13713e4f77e327fef0d552070f9dfe76da3fb`, updated 2026-07-02.
- Own `score_v2` on the live/anchor4 onnx: **memory 18486 + params 639 = 19125** (matches manifest exactly).
  (`team_best` source shows 19091 for the SAME sha — a stale measurement artifact; authoritative manifest and
  my re-score both = 19125.) No `Team_Submission/Staged_Boladi/task018.json` sidecar exists → manifest live IS the effective best.
- Live == staged `task018_anchor4_cost19125_probe`, byte-equal to the shipped submission (prior `ab --full` 200266/200266).
  Path-B solver (truth ~98.9%), so byte-equal-vs-submission is the gold gate.

## Card hypothesis (this prompt) and why it's the wrong shape for the rule
Prompt 6411 (MorphologyPool) is a STRUCTURAL card triggered by corpus hits **maxpool / component / flood**, proposing a
from-scratch rewrite via "binary/colored morphology: MaxPool dilation, signed-int8 erosion, local 3x3 window tests,"
targeting floor_est ~1500 (12.8× below live). **The corpus trigger matched a SUB-COMPONENT (the flood), not the rule.**

The decoded rule (from the ARC-GEN generator `task_0e206a2e.py`, same as hidden test): 1–2 color-identical "creatures",
each = a full ORIGINAL + a CLONE rotated by rot∈{1,2,3,4} with body hidden (only 3 anchors shown). Output = each clone
reconstructed FULLY under the rigid D4 transform aligning its 3 anchors; originals removed. This is **object detection +
per-object rigid-transform reconstruction + colored scatter** — NOT a morphology rule. A rigid-transform body
reconstruction is not expressible as pooling/erosion. **floor_est ~1500 is fantasy**; the output canvas alone is 900 B.

## The live graph already IS the morphology the card asks for
The live 237-node graph (`task018_structural_v54` lineage) already uses MaxPool morphology where morphology applies:
- Flood CC-separation = **3 clamped `MaxPool`+`Min` rounds** (nodes 15/16 region-dilate, 26/27 + 28/29 flood) — this is
  exactly "MaxPool dilation + local-window clamp." It is the minimal CC-separation of two color-identical blobs.
- Color classification = a **1×1 `Conv`** (node 0) = a per-cell local test, already optimal (see decode table below).

So the card's suggestions are already the live approach; the remaining question is only whether any charged component is
narrowable. Every one is dtype-locked or structurally forced:

## Priced per-component floor (where the 18,486 charged bytes live)
| bytes | tensor | forced by | this-session evidence |
|---:|---|---|---|
| 2,304 | `color_idx_full_f32` (Conv, F32) | F32 one-hot input → F32 reduce out | **decode table below — Conv is the floor, all alts 4–17× worse or ORT-invalid** |
| 1,152 | `nonbg_score_f16` (TopK input, F16) | U8 TopK is ORT NotImplemented; INT32 = 2× | verified failed-probe `v125_u8_topk`; F16 is the 2-byte floor |
| ~3,456 | 3× flood `MaxPool`+`Min` rounds | no sparse-CC op in ONNX (NonZero/Unique banned); wider/fewer rounds bridge the 2 color-identical creatures → `ScatterElements idx=576 out of bounds` crash or visible fail | **documented-dead — do-not-rewalk (don't-do log + v126_k10/k11)** |
| 900 | `raw_label_masked_padded` [30,30] | forced 30×30 output contract | prior record |
| ~10,944 | ~19 × 576-B U8/BOOL masks + flat↔spatial reshapes | each a distinct semantic mask; reshapes forced because ORT needs flat layout for Gather/Scatter/TopK vs spatial for MaxPool | fresh full-graph teardown this session |

## Empirical decode pricing (this session — the one NEW lever this card raised)
Isolated color-decode `input[1,10,30,30]F32 → color_idx_u8[1,1,24,24]`, scored by `score_v2` (memory+params):

| formulation | charged bytes | verdict |
|---|---:|---|
| **A: 1×1 Conv F32 → Cast U8 (LIVE)** | **2,880** | FLOOR (2,304 F32 + 576 U8) |
| B: Slice→Cast U8→Mul→ReduceSum (integer) | **None** | ORT-INVALID (U8 reduce path fails score_network_v2) |
| C: ArgMax(axis=1) INT64 → crop → Cast | 12,384 | 4.3× worse (INT64 planes) |
| D: Slice→Mul→ReduceSum (F32) | 48,960 | 17× worse (materializes [1,10,24,24] F32 = 23,040) |

**Root cause:** the Conv uniquely fuses channel-reduction + the 30→24 crop (`pads=[0,0,-6,-6]`) into one op and never
materializes a multi-channel intermediate. Any op that touches the 10-channel input (Slice/Cast/ArgMax) creates a
[1,10,*] intermediate ≥5,760 B. The 2,304-B F32 is forced by the F32 one-hot input dtype — a "local window test" cannot
beat it. Card hypothesis falsified on its own biggest lever.

## Independent from-scratch formulations evaluated (all wall)
1. **Pure local-morphology / MaxPool-pooling solver (the card):** falsified — rule needs rigid-transform reconstruction,
   not pooling; the morphology sub-component (flood) is already used and structurally minimal.
2. **Integer-only decode + selection (dodge F32/F16 locks):** falsified empirically — integer decode ORT-invalid or
   4–17× worse (table above); U8 TopK ORT-blocked (v125).
3. **Constant/anchor shave route:** exhausted by prior second-pass — anchor cap 4 is the floor (anchor3 fails visible 53%
   `v129`; body K=12 / nonbg K=30 visible-tight; k10 crashes). 19,125 is the shave floor.

## Verdict + remaining hypothesis
**WALL at 19,125.** Re-confirmed from the MorphologyPool/local-window angle, independent of the 2026-07-10 Pad/Concat
angle. Every charged component is individually dtype-locked (F32 color Conv, F16 TopK) or structurally forced
(no-sparse-CC flood, forced 30×30 canvas, semantically-distinct masks, ORT-layout-forced reshapes). No sub-19,125
structural route survived. Only a fundamentally different *algorithm class* (a lookup/table or public-weight-aware
strategy — NOT morphology, NOT a constant shave) could beat 19,125, and the task's cost→public-score link is broken for
the lookup family (task018.md don't-do log; the "8154" fullcount/metapack figure is a zero-public-weight memorized-lookup
artifact, not a valid structural bar). **Do not re-run the MorphologyPool / local-window / integer-decode probes on this task.**

---

## SECOND PASS (2026-07-12, forced pre-review) — library-mined + data-backed re-confirmation

Runner forced a second pass demanding a *genuinely different* from-scratch formulation (different dominant
intermediate shape/dtype), library mining, and gating. Did all three; every genuinely-different lever is now
falsified with **empirical/priced** evidence, not just analysis.

### Library levers mined (INDEX → cost_floor_reference / ort_reality / techniques)
- `cost_floor_reference.md`: the only cited flood-reduction levers are **bit-packed / column-packed flood**
  (task251 R10 = 3,740; task286 column-packed U32 = 21,558). Both win by shrinking the *round-count × plane*
  product of **LONG** floods (task251 = 10 rounds; task286 = geodesic horizontal sweep).
- `ort_reality.md`: `MaxPool` accepts U8/I8 (1 B) — already used; negative end-pads crop in-node (already used).

### Why bit-packed flood is NET-NEGATIVE here (priced)
task018's flood is **2 rounds with WIDE kernels** (`fexp0` 7×7 reach-3, `fexp1` 5×7 reach-3; `dil2_f` 5×5).
A bit-flood propagates exactly **1 cell/round**, so matching reach-3 needs ~3 sub-rounds ⇒ 2 rounds → ~6 bit-rounds,
each needing H-shift + V-roll + OR + region-AND (~6 ops × [24]U32=96 B ≈ 576 B/round) ≈ 3,456 B, PLUS pack + the
region/seed re-pack + sparse unpack. That EXCEEDS the current pure-propagation cost (~2,304 B, nodes 26–29).
Bit-packing amortizes only over long floods; task018 is the opposite regime (short + wide-kernel). Not built — priced-out.

### Data-backed proof the flood itself is FORCED (new this pass — decoded rule from the 266 real inputs)
- All 266 inputs: exactly **3 distinct anchor colors**; `color-set(input) == color-set(output)` on all 266 (pure relocation).
- **124/266 (47%) inputs have 2 creatures that are COLOR-IDENTICAL** — same filler + all 3 anchor colors shared
  (each anchor color appears 4× = original+clone in *both* creatures; the other 142 are 1-creature, each anchor 2×).
  Concrete: input[0] = two filler-8 blobs, BOTH with anchors {1,3,4}. **No color/sparse grouping can separate them** →
  spatial connectivity (the flood) is information-theoretically required in ~half the corpus. "Skip the flood via
  color-grouping" FALSIFIED by data.

### `inp_active` / output-`Where` is FORCED (killed the one redundancy hypothesis)
Input grids vary **12×15 … 24×24** (output size = input size), so `inp_active = (color_idx≠0)` is NOT all-True — it
masks out-of-grid cells (color_idx=0, all-zero one-hot) to background for sub-24×24 grids. The `Where`+`inp_active`
(1,152 B) is load-bearing. The 30×30 output pad (900 B) is the grader canvas. Both forced.

### Sparse-CC (adjacency-matrix reachability) — priced worse
Separating the ≤30 nonbg cells via a `[30,30]` adjacency + iterated `MatMul` reachability (MACs are free post-2026-05-04)
needs the adjacency built from pairwise dr/dc ⇒ several `[30,30]`=900-B tensors (900 > the 576-B plane). nonbg K=30 is
visible-tight (can't shrink so [K,K] < 576). Net worse than the plane flood.

### Verdict (unchanged, now data-backed): WALL at 19,125
Cheapest correct formulation found = the live `anchor4` graph, **profile cost 19,125** (static=profile, no grader-gap).
Information-theoretic floor argument: per input the graph MUST (a) decode a ≤24×24 color grid (2,304 F32, dtype-forced),
(b) spatially CC-separate 2 color-identical creatures in 47% of cases (flood, ~2,304+ forced, wide-kernel regime beats
bit-packing), (c) select ≤30 sparse nonbg cells (TopK, 1,152 F16, U8-blocked), (d) mask out-of-grid for 12–24 variable
grids (inp_active, forced), (e) emit a 30×30 canvas (900, grader contract). These forced components alone sum past most
of the 18,486 charged bytes; the sparse transform middle (~2–3 KB of [3]/[4]/[12] INT8/16 tensors) and 5 forced
flat↔spatial reshapes are individually <50 B or layout-forced. No sub-19,125 route survives. No candidate gated (none
below bar; ship-gates on non-winners are banned burn).
