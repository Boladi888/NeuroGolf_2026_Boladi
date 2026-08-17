# DEAD ROUTE: zero-param direct-output op (the "perfect-25 cost≤1" rumor)

- **Live bar**: cost **30** (manifest, authoritative), sha256 `356a5f1e2200cf2560f6c836ce7d21eca6b6cad9e68015c2ae31639613ef9b09`.
- **Attempted route** (ZeroopsweepCeilingAudit, worker1 Claude, 2026-07-07): one-node / attr-only
  direct-output ops that would price ≤1 — `Identity`, `Transpose(perm=0,1,3,2)` (only
  shape-preserving perm). (flip/reverse already dead per Don't-do: Slice neg-step + not a stack.)
- **Failed gate**: `ng_gate visible` → **0/265 FAIL** for BOTH (profile cost 0 each).
  A visible-zero at cost 0 is a certain hidden 0 (§VISIBLE-FAIL AXIOM).
- **Why it's dead**: the rule is a vertical mirror-**STACK** — `out[r]=in[idx[r]]` with
  idx=`[0,1,2,3,4,4,3,2,1,0, 10×20]`. It DUPLICATES each input row into both the top and the
  bottom half. No zero-param reshape/transpose/flip/resize op can duplicate a row. Identity leaves
  the bottom-mirror rows empty; a shape-preserving Transpose only swaps spatial axes. Any
  repeated-index row map forces a Gather with a 30-element index ⇒ params ≥ 30. The dense
  INT64/INT32 initializer is the only memory-0 realization ⇒ exactly 30. cost=30 is the forced floor.
- **One remaining hypothesis** (already disproven upstream, do NOT re-run): every non-dense index
  realization (Constant node / Range / Pad / ConstantOfShape / sparse) either produces a counted
  `[30]` node-output intermediate (≥120 B ⇒ ≥133) or is rejected by `score_network_v2` (=None).
  See `../../task113.md` Attempts + "Why 30 is the hard floor" proof. **No sub-30 candidate exists.**
