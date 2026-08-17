# task243 — U8-byte-lane bit-packed flood FLOOR (no-candidate floor proof)

**Where:** task243 (Filling Regions / 4-conn seed flood-fill, active grid ≤18×18).
Team-best ONNX `Team_Submission/Backend/Team_Best_Onnx/task243.onnx` SHA256
`2761FFEC56E1D94FF68619E27D6A8C0608619E60BF673B3047FB37F32EFEF871`,
score_network_v2 = (0, 10908, 407) ⇒ cost **11,315**, profile==static, visible 265/265.
This is Aditya's `task243_insert_pruned` — NOT recorded in any routed library doc as of 2026-06-24
(task_techniques_index still lists task243 at 117,161; recent_ships has no entry).

## What I discovered (transferable)

**The U8-byte-lane bit-packed flood is the strict successor to #47's row-packed U32 flood for widths 9-24.**
At width 18 you need 18 bits/row = 3 UINT8 lanes (8+8+2). #47 (U32, 4 B/elem) cost 21,328 here;
splitting the same packing into U8 lanes (1 B/elem) cut it to 11,315 — roughly halving the per-tensor cost.
This is the #157 "U8-lane narrowing" verdict applied to a whole flood chain, not just an exit plane.

**Byte-split trick (cheap, reusable):** pack a row into one u32-as-F32 via Conv(taps 2^c), then extract the
3 bytes with `Cast(F32→u8)`=low byte, `Mul(1/256)+Cast`=mid, `Mul(1/256)+Cast`=hi. No BitShift needed for the split.

**Alternating-axis beats combined-axis at byte-lanes — the cross-byte carry is the cost driver.**
A combined 4-conn step (H+V together) pays the expensive cross-byte horizontal carry (~7 BitShift + 7 OR on 3 lanes)
EVERY round. Splitting into separate V rounds (Pad up/down, ~149 B, no carry) and H rounds (~250 B, carry) lets you
use MANY cheap V rounds + FEW expensive H rounds. task243 tuned to 20V+16H=36 rounds; the H count (16) is near the
14 floor measured by "saturate-V-then-1-H" simulation. Combined-step (28 rounds × full carry) LOSES decisively.
Orientation-swap (pack rows, explicit cols) is symmetric → no gain.

**Floor-proof recipe that gave a clean NO-CANDIDATE:**
1. §37 round floor: drop ANY one round from the tuned schedule → fails official corpus (drop-1-H: 1 fail, drop-1-V: 1 fail).
2. 8-conn (3×3 MaxPool) flood differs from 4-conn on 264/265 official pairs — diagonals leak through blockers. Always
   corpus-check 4conn-vs-8conn equality BEFORE substituting MaxPool (here it's NOT equal, unlike task048's resampled generator).
3. Setup (540 B) = F32-locked Conv bridge outputs (input-dtype-lock). Unpack (1,872 B) dominated by `fill_full`
   BOOL[1,1,30,30]=900 B = the irreducible non-separable 30×30 Where condition. Fusion saves only ~36 B.

## Why it matters
Saves a future worker a full siege on task243: 11,315 is the floor for the iterative-bit-packed-flood family.
Extractable slack ≈ 100-300 B (micro-fusion), ln-gain ≈ +0.065 — below a confident strict beat.

## What's missing from the library
- No routed doc captures the U8-byte-lane flood as the width-9-24 successor to #47 (only #88 covers width≤8 single-byte rows).
  Suggest adding to #47/#88 or by_symptom "iterated flood narrow grid": "width 9-24 → 3-lane U8 byte-pack, alternating-axis
  schedule (cheap-Pad-V + carry-H), NOT combined-step."
- task_techniques_index.md task243 row is stale (117,161 → should be 11,315 insert_pruned).
