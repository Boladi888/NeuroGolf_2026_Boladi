# Dilated crop-collapse Conv KILLS the "F32 input-read tax" floor (task271, 2026-06-12)

**Transferable correction to the F32 input-read-tax law (cost_floor_reference.md line ~181)
and to the #98 application notes.**

## What happened
task271 (sliding 3×3 max-blue box selector). Prior worker (2026-06-11) proved a **1942
"byte-equal floor"** and concluded Aditya's team-best **1311 was unbeatable by Path-A**,
on the premise that the rule needs **two F32 channel reads** (blue + solidity) and that
`Slice` of the canonical input preserves F32 → `[1,1,9,9]`×2 = **648 B irreducible**.

**This premise is FALSE.** The canonical graph input `[1,10,30,30]` is FREE (uncounted). A
Conv that reads it **directly** pays only its OUTPUT bytes + params — there is no F32 slice
landing at all. Result: cost **1136** (memory 990 + params 146), beating 1311 by **175**.
Gates: visible 267/267 byte-equal · 20000/20000 truth+AB vs fresh generator · score_v2
1136 · profile==static==1136 (`--arcgen 266`).

## The construction (techniques #98 + #100 combined)
- **conv1 (crop-collapse + channel-fold, reads FREE input):** kernel `[1,10,2,2]`,
  `dilations=[L−k, L−k]` (here 30−9=21), real weights on tap (0,0) only, far taps 0.
  Weights fold the two needed channels into ONE: `W[blue_ch]=+1`, `W[bg_ch]=−BIG` (BIG=16,
  > max count 9, F16-exact). Emits `comb = blue − BIG·bg` over the active 9×9 region as
  `[1,1,9,9]` (F32 324 B → Cast F16 162 B). The 648 B two-slice read is GONE.
- **conv2 (dense scan):** `[1,1,3,3]` ones over `comb` → `[1,1,7,7]` = per-window
  `blue_count − BIG·bg_count`. Solid windows (bg=0) outrank every straddling window; ArgMax
  over 49 picks the unique max-blue box (reading-order tiebreak == LIVE).
- **extraction needs NO second read:** inside the winning SOLID box every cell is blue
  (`comb=+1`) or cyan (`comb=0`), so `blue ⟺ comb>0.5`. Gather the 3×3 from `comb`,
  threshold → BOOL one-hot, Concat+Pad to `[1,10,30,30]`.

## The general law (proposed library edit)
**Whenever you would `Slice`/`Gather` a fixed sub-rectangle of the F32 one-hot input to feed
a reduce/scan, check #98 FIRST.** A 2-tap dilated valid conv (`dilation = axis_len −
crop_len`, far tap zero) reads the free input and lands only the cropped, channel-reduced
map. If you ALSO need a per-cell solidity/validity bit, fold it into the SAME conv channel
with a `−BIG·non_support` weight (#100) instead of reading a second channel. This turns the
"N-channel F32 read floor = N·h·w·4" into "one small reduced-map landing", which can be
F16-cast for half. The `cost_floor_reference.md` F32-read-tax paragraph should add: *"this
floor only applies to the materialized result of a Slice/Gather; a #98 dilated crop-collapse
Conv pays the OUTPUT, not the read — reprice before claiming a read floor."*

Also corrects task271's own ledger (1942 → 1136 staged at `staged_submits/task271_beat1311_cost1136/`).
