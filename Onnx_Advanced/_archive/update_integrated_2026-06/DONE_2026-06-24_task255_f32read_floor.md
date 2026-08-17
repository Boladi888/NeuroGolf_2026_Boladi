# task255 structural floor: the F32 one-hot read is a hard 3,600-byte memory floor

Session: W2 task255 re-confirm 2026-06-24. Routing hint: cost_floor_reference.md
(plane-floor taxonomy) and/or traps.md §GRADER-GAP / read-floor note.

## Insight (transferable to any task that reads the F32 one-hot input as a spatial plane)
The NeuroGolf input is `[1,10,30,30] F32`. The FIRST materialized single-channel
30x30 spatial plane is **3,600 bytes** and is irreducible:
- `Gather(ch0)`, `ReduceSum(axes=[1])`, and `Conv(input,...)` all emit **F32** from
  an F32 input (ort_reality: Gather/ReduceSum/Conv output dtype == input dtype).
- You cannot get a UINT8/F16 plane without first materializing the F32 plane and
  then `Cast`-ing it (which adds the U8/F16 plane on TOP of the 3,600 F32 plane).
- `Conv` with F16 output on F32 input is impossible (ort_reality), so no cheap
  F16 read either.
- Casting the whole input to U8 first is 9,000 (10x900) — strictly worse.

Practical consequence: for any "read one channel -> spatial mask" task, budget
3,600 for the read and do NOT spend effort trying to make it U8/F16. The savings
must come from FUSING downstream U8 mask planes, not from the read.

## task255-specific re-confirmation (no new ship)
c13983 (mem 13,800 + p 183 = 13,983) is the LIVE/team-best and the established
clean structural floor for 265/265 + 261/261-arcgen. Independent probes this
session reproduced the documented wall:
- BOOL-`And` terminal (replace Min(U8)+Cast(BOOL)): +900 (two BOOL planes > one U8).
- drop r3/keep_r3 gate: 13,021 but fails train[2] left-edge corridor (= existing
  c13021 probe; not a clean ship).
- keep_r3=h12-only: 21/261 (v12 top/bottom edge-touch is load-bearing).
Verdict: NO clean sub-13,983 candidate. The cheaper staged probes (c13021/c13067/
c12105/c11194) each drop >=1 train pair or arc-gen and are captain gambles.

## Also: the prompt was stale
The task255 EXPERT prompt still describes a hash-MLP at the hidden zero-weight
floor with +12.91 LB upside. That was 2026-05-25 reality; the unlock was decoded,
shipped (v12 +14.90 LB), and compressed to 13,983 (real 15.45 pts). Future
task255 prompts should drop the "hash-MLP unlock" framing and ask only for a
clean sub-13,983 structural compression holding 265/265 + 261/261.
