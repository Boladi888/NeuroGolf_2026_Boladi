# §62b/§67 FREE-OUTPUT trap: the BACKGROUND channel (ch0) also has to materialize

**Where:** task104. Builders `The_400_Tasks/Task_by_Task/task104/prototypes/build_task104_bitand_where.py` (winner, 1102) and `build_task104_bitand_freeout.py` (the ch3-only floor probe, 1020 but WRONG). Staged `staged_submits/task104_bitand_where_cost1102/`.

## What I discovered

The §62b BITAND bit-detector (techniques.md §62b) and the §67 free-output catalog (the "make the expensive expansion BE the graph output, never charged" move) have a sharp limit that the current write-ups understate: **the free-output expansion can only carry the channels its detector fires. The grader's reference one-hot ALSO sets channel-0 (background) = 1 on every in-region cell that is NOT a foreground color**, and decode is `(out > 0)`. So a free-output that emits only the foreground channel(s) and leaves ch0 empty fails the gold gate, even though it looks correct on the foreground.

Concretely for task104 (output = a 9×9 mask of color 3 on a color-0 background):
- The reviewer-verified §62b lever IS correct for the color-3 channel: `z = BitwiseAnd(x[1,1,9,1], y[1,1,1,9])` with x,y carrying block-A=bit0 / block-B=bit1 reproduces all 4 patterns EXACTLY (bit0 & bit1 = 0 kills the anti-diagonal cross terms). Verified `np.array_equal` True for P0/fliplr/flipud/rot180. Axis vectors: `x_fwd=y_fwd=[1,1,1,1,2,2,2,2,0]`, `x_rev=y_rev=[0,2,2,2,2,1,1,1,1]` (rev = reverse(fwd)).
- The prompt's literal recipe (K channel-3 = bits{1,2}, all other channels = 0, final BitwiseAnd = graph output) produces a [1,10,30,30] output whose ch0 is ALL ZERO. That scores **officials 0/3** — the grader's ch0 should be 1 on background. The "sub-700" estimate was computed from this incomplete recipe.

## Why ch0 cannot also be a free-output BITAND here

1. **Non-monotone.** ch0 = complement of the foreground = NOT(z!=0). Outer-BITAND `(x[r] & y[c] & K)!=0` is a monotone (OR-of-AND) function of the bits; the complement is non-monotone, so it can't share the foreground's bit lattice in general.
2. **Band-misalignment under offset-reversal.** The selector picks fwd/rev axis vectors; the four patterns' 4×4 blocks are offset by 1 under the reverse (P0 col-bands {0:4,4:8} vs P1 {1:5,5:9}). A band-constant background bit can't be uniform across selections. Group-level and position-level bit-partition searches for a shared `(x_fwd,x_rev,y_fwd,y_rev,K0,K3)`: **0 solutions**.
3. **`Where` can't emit all-zero.** The clean workaround "use Where(mask, CH3, CH0) to stamp both channels in one op" works for the IN-REGION cells, but `Where` always picks one branch, so the 30×30 pad region would wrongly get ch0=1. You still need a `Pad` (explicit 0 fill) as the final op → the [1,10,9,9] one-hot is CHARGED (810 B). That re-imposes the PAD-LAST floor.

## The winning architecture (still a real improvement)

`z = BitwiseAnd(x,y)` (rank-2 pattern mask, 81 B) → `zb = Cast(z, BOOL)` → **`onehot = Where(zb, CH3[1,10,1,1], CH0[1,10,1,1])`** = [1,10,9,9] uint8, ONE op that supplies BOTH the foreground channel and the background channel → `Pad → output` (#81 PAD-LAST, 810 B). This single-Where assembly is leaner than the §67 Equal-tail's {0,3}-canvas detour (1175) and than the 1209 flipbank (bank + int32 Gather perms). **Cost 1102** (memory 1024 + params 78), gold gates all pass: visible 7/7, ab-full 200007/200007 byte-equal, truth-full 200000/200000, profile==static 1102. −107 vs prior staged 1209, −282 vs team-best 1384.

Note: `Where` has NO ORT kernel for BOOL data branches (`NOT_IMPLEMENTED Where(16)`) — CH3/CH0 must be UINT8, not bool. Output uint8 decodes fine under `(out>0)`.

## What's missing from the library

- **§62b / §67 should carry an explicit BACKGROUND-CHANNEL caveat:** "the free-output expansion floors only when EVERY active channel of the reference one-hot (including ch0=background) is produced by the detector. If the rule paints foreground on a background that the grader one-hot-encodes, ch0 must be materialized too — and if ch0 is the (non-monotone) complement, the free-output collapses back to the PAD-LAST [1,10,H,W] one-hot floor." This is the same family as #87's FREE-OUTPUT SEPARABILITY TRAP but a distinct cause (channel completeness, not splitting).
- **The single-Where one-hot assembly** `Where(mask_bool, CH_fg[1,C,1,1], CH_bg[1,C,1,1]) -> [1,C,H,W]` is a clean cheaper alternative to the Equal-tail when exactly two colors (a foreground + background) are present — worth a one-liner next to §67's Equal-tail. (CH operands must be uint8, not bool — ORT.)
