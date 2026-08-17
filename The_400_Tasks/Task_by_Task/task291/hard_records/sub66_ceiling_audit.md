# task291 hard_record — sub-66 ceiling audit (WALL)

- **Live bar:** cost **66** (`pad-axes-v1`), SHA `8a7573530530236fe87f01bca7cd68d84f804675c4fb705007a4cb03d65bd545`
  (manifest authoritative, owner Boladi, 2026-06-22). Verified live this batch:
  `profile_real_cost` = static=66 profile=66 (memory 40 + params 26; 265 real inputs).
- **Audit date:** 2026-07-07 (batch53, worker `bo_ng_cld_worker1`).
- **Prompt framing rejected:** "perfect-25 / zero-cost ≤1 ceiling" is DISPROVEN. This is a
  genuine hollow-box **detector**, not an identity/transpose. No-Pad direct 1×1 output
  scores (0,0,20) but visible is **0/265** — the validator needs the full `[1,10,30,30]`
  encoded output (`neurogolf_utils.py:546`, convert_to_numpy space). No legal zero-param
  family exists. Do NOT re-run the perfect-25 sweep here.

## Routes attempted / disproven this audit
1. **Basis-vector free-output einsum** (BUILT + GATED, independent structure):
   `abrd,abec,afrc,f,b,x,y->abxy` with two size-30 corner basis vectors row0/col0 —
   emits the final `[1,10,30,30]` directly, **0 intermediates, no Pad**.
   - score_v2 = (0, 0, **80**); visible **265/265 PASS** (correct but dominated).
   - builder: `prototypes/build_task291_basis_freeout_v1.py`; onnx: `task291_basis_freeout_v1.onnx`.
   - Lesson: the free-output branch trades the 40 B `score` intermediate + 6 Pad params
     (= 46 units in the reigning graph) for 60 spatial-basis param units. 60 > 46 ⇒ +14, loses.

## Forced-floor argument (why 66 is the wall)
Output contract forces a full `[1,10,30,30]` tensor. Detection is per-channel (which of
colors 1–9 is hollow), so ≥1 F32 scalar per candidate channel is required. Two — and only
two — placement branches exist:
- **(a) intermediate + place:** einsum → `score` F32[1,10,1,1] = **40 B** (F32-in ⇒ F32-out,
  10 channels tied to input; cannot be BOOL/INT8 without a Cast that first materializes the
  40 B F32). Minimal scalar→corner placement = **Pad-18 axes** `pads_hw[4]+axes_hw[2]` = **6**.
  Subtotal 46.
- **(b) free-output basis:** no intermediate, but two size-30 corner vectors = **60**. Loses.
- min(46, 60) = 46.

Channel machinery is forced at **20**: `black[10]` (selects channel-0 background plane as the
intersection template — the cheapest way to get "hole area = Σ_bbox background"; any Slice of
input[:,0] adds a 3600 B intermediate) + `colors[10]` (kills output channel 0; einsum b-axis
is pinned to input's 10 channels, so a 9-channel output would need a [9,10]=90-param map or a
post-Slice that ADDS the 36 B intermediate on top of the 40 B score).

**Floor = 20 + 46 = 66.** No sub-66 route exists in either branch.

## One remaining hypothesis (unlikely)
An op that (i) performs the input³ rowsum×colsum×background contraction AND (ii) emits a
sub-F32 dtype `[1,10,1,1]` as its FIRST materialized tensor (no F32 intermediate). No such
ONNX op is known (einsum/matmul are F32-in→F32-out; comparison/cast ops require the F32
tensor to already exist). If one is ever found, memory 40→10 would give cost 36 — the only
conceivable strict-below path. Until then: **WALL @ 66.**
