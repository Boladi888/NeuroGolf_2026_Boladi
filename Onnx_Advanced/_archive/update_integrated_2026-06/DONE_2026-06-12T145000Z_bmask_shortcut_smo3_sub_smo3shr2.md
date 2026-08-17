# bmask shortcut: highbit|highbit>>1 = smo3 - (smo3>>2)

## What I discovered

In the task048 bitpack flood decision chain, the "box-B mask" (bmask = highbit | highbit>>1) 
can be computed as `bmask = smo3 - (smo3>>2)` after the 3-step smear that computes `smo3`.

This eliminates 2 named intermediate tensors (`smh` and `highbit`) from the decision chain,
saving 2 bytes of memory. The c2 constant (value=2) is already present for the smear step 
and is reused.

**Proof:**
smo3 is the "all-ones below MSB" pattern for lbyte: smo3 = 2^(n+1) - 1 where n is the MSB position.
Then:
  smo3 - (smo3>>2) = (2^(n+1)-1) - (2^(n-1)-1) = 2^(n+1) - 2^(n-1) = 2^n + 2^(n-1)
                   = highbit | (highbit>>1) = bmask

Edge case n=0: smo3=1, smo3>>2=0, result=1-0=1 = bmask (lbyte=1 has highbit=1, bmask=1|0=1). ✓
Edge case n=1: smo3=3, smo3>>2=0, result=3-0=3 = bmask (highbit=2, bmask=2|1=3). ✓
Verified correct for ALL 256 uint8 values.

## Where I encountered it

Task: task048 (bitpack flood for connectivity detection)
Builder: `The_400_Tasks/Task_by_Task/task048/prototypes/build_task048_v6.py`
Staged: `staged_submits/task048_v6_bmask2_cost697/`

## Why it matters

v5 k6 decision used 4 ops (smh=smo3>>1; highbit=smo3-smh; hb2=highbit>>1; bmask=highbit|hb2)
= 4 named intermediates.
v6 uses 2 ops (bmask_s=smo3>>2; bmask=smo3-bmask_s) = 2 named intermediates.
Net savings: 2 intermediate scalars = 2 bytes memory. Cost 699 -> 697.

## What's missing from the library

Section #88 (u8 flood kit) documents the 3-step smear for highbit extraction.
The shortcut "bmask = smo3 - (smo3>>2)" is NOT documented there.

It should be added to #88 as a micro-optimization bullet:
"bmask shortcut: highbit|highbit>>1 = smo3-(smo3>>2) after the 3-step smear (smo3 = all-ones-below-MSB pattern). Proof: smo3 = 2^(n+1)-1; smo3>>2 = 2^(n-1)-1 (for n>=2); smo3-(smo3>>2) = 2^n + 2^(n-1) = bmask. Saves 2 intermediate scalars vs the highbit/hb2/Or chain. Reuses the c2 param already present for the smear step."
