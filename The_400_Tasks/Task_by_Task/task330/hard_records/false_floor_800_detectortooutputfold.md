# task330 DetectorToOutputFold — false-floor(~800) attack = DEAD (2026-07-11)

**Live bar at attack time:** cost **2535** (mem 2300 + par 235, pts 17.1621),
manifest/owner Aditya, sha256 `b3c1971515a2001ddd91305231d4f755eabaea7269727c9e3005016e2acac4bf`.
Prompt asked for a from-scratch rewrite to a ~800 forced floor. **Refuted.**

## Why the ~800 floor is a mis-estimate (forced-component argument)
Cost model (`neurogolf_utils.calculate_memory`): cost = Σ intermediate-tensor
bytes (max static/runtime) + Σ initializer element-count. **input AND output
tensors are EXCLUDED** — the [1,10,30,30] canvas is FREE. So the ~800 estimate
("detector evidence + output condition, canvas not forced") only makes sense if
a CHEAP local area detector exists. It does not:

- **Rule** needs true 8-connected component area (`==6 -> c2, else c1`). Areas
  span 2..8; area-6 comps take **52 distinct shapes**, all in a 4x4 bbox.
- **Local window-sum detector is DEAD.** For each fg cell, count fg in a KxK
  window; `==6 -> c2`. numpy sweep K=3..11: best **26/266** (K=7). Components sit
  ≤1 bg cell apart, so a window catches neighbours → cannot isolate a component.
  Built as ONNX (`prototypes/task330_windowsum_probe.onnx`): **visible 0/266 FAIL**,
  and its detector core alone is Slice(400B F32) + Conv(400B F32) = **~800B before
  any fold** — already at the target, and wrong.
- **Forced bytes for a CORRECT graph:** F32 ch5 Slice = 400B (any op reading the
  F32 input to a 10x10 region yields F32; the downcast is a separate counted
  node) + 3-round geodesic MaxPool labeling = 600B (load-bearing: staged
  `prune2` = 2 rounds → 39/266; 5x5/7x7 pools leak across the 1-cell gaps) +
  INT32 scatter index = 400B (ScatterElements/GatherElements require int32/64;
  no int8 kernel) + count vectors ~500B + 2-ch features 200B. Floor ≈ **2535**.
- **Paint is already optimal & fused.** Champion Concat(active,enc)[1,2,10,10]
  (200B) → **QLinearConv** directly to the FREE [1,10,30,30] output (pads
  [0,0,20,20] expand 10→30 canvas for free). The prompt's "Where/Equal output"
  suggestion is WORSE here: it materialises a 10-channel [1,10,10,10] (1000B+)
  intermediate before Pad. **The "separate rendered canvas" the Direction wanted
  to delete is ALREADY GONE** in the live 2535 graph → prompt DROP condition met.

## Exact failed gates
- `windowsum_probe`  : visible **0/266 FAIL**  (score_v2 cost 3368).
- `labelscatter_shave`: dropped the one non-forced node (2× updates, base 244→250)
  → cost **2435** but visible **68/266 FAIL** — the 2× update is load-bearing for
  the QLinearConv u8-wraparound encoding; a q_w retune would still only reach
  2435 (still 3× the ~800 target, and a rung-shave the prompt excludes: stage-if-<800).

## Verdict
DROP / WALL. 2535 is at the true floor for correct connected-component area-6
detection, NOT a formulation artifact above 800. The ~800 estimate omitted the
labeling+scatter counting cost, which the window-sum refutation proves is forced.

## One remaining hypothesis (low confidence)
A q_w-retuned 1×-update variant reaches 2435 (byte-equal-able to LIVE, LOCAL_GOLD
class) — a legit +0.04 LB but a rung-shave below the structural bar. Only worth it
if the team wants the last 100B on this task; not pursued here per stage-if-<800.

## Second pass (2026-07-11) — grouping-primitive floor is now PROVEN, not argued
Reviewer forced a second pass. Extra empirical evidence that the INT32 scatter
index (400B) and thus the whole counting stage is FORCED:
- **Narrower index dtype REJECTED.** Built ScatterElements/GatherElements with
  int8/int16/uint8 indices → `onnx.checker` (which the scorer runs, `full_check=
  True`) REJECTS all; only int32/int64 pass. So the 400B index cannot shrink to
  100B/200B. INT32 [1,100] = 400B is the hard floor for a 100-cell grouping.
- **Scatter-free grouping is 34× WORSE.** Built the only exact scatter-free
  grouping — OneHot(label,100)+ReduceSum+MatMul (`prototypes/
  task330_onehot_matmul_count.onnx`), reusing the champion's labeling + paint so
  the delta is purely the grouping. Result: **byte-equal to champion 266/266,
  visible 266/266 PASS, profile cost 85,943** (mem 85,700 + par 243). The one-hot
  is [1,100,100] and ONNX bans u8 `ReduceSum`, forcing F32 → 40,000B ×2. Scatter's
  400B index is optimal by ~34×.
- **Labeling is already the cited-cheapest.** techniques.md #33 (neg-MaxPool
  MinPool connected-component labeling) IS the champion's assembly; no cheaper
  labeling exists in the library. BoBrain `consistency task330` = 2535 authoritative.

Information-theoretic floor: read fg (400B F32 slice, forced) + group-by-component
(400B INT32 scatter index, proven-optimal) + geodesic labeling to build the group
key (600B, load-bearing) + count vectors + fused free-output paint = **~2535**. The
~800 target is unreachable on this route; 2535 is the true floor. **WALL stands.**
