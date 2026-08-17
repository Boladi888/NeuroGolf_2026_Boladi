# task390 hard record: prompt's floor_est ~900 is UNREACHABLE; real floor ≈ 2360

- Updated 2026-07-12 by bo_ng_cld_worker2 (Claude lane), 2nd pass.
  (Supersedes the earlier "sub2400 unreachable" note — 2400 WAS beaten.)
- Live bar: **2514** (`ng_current_bar.py --task 390`, manifest, owner Sho, sha
  `890ed508…e064f2`, 2026-07-09) — the enumeration graph.
- **Best achieved this session: 2363** (byte-equal to live 2514: ab 2266/2266 +
  truth 2000/2000), staged `staged_candidates/task390_w4_graycode_shared_cost2363/`.
  Ladder 2514 → 2466 (gray-block) → 2432 (gray-in-code) → 2363 (shared Equal).

## Two forced components set the floor at ~2360 (NOT ~900)
1. **Free one-hot decode = 900B.** Output [1,10,30,30] one-hot with 3 colors
   {bg,red,gray}. The only cheap free-output op is a trailing 1×1 QLinearConv
   (Conv→F32 = 36 KB intermediate; Equal-tail needs a charged 30×30 pad = 1125B
   > 900; broadcast-Add can't express scattered 2-D gray). QLinearConv needs a
   2-channel feature `[c, c²]` (codes {1,2,3}; the unique encoding keeping the
   conv-pad region → sentinel while separating 3 collinear colors). That is
   feat2 450 + color4d 225 + u_sq 225 = 900. Building feat2 directly as 2-channel
   is +53 (assembly doubling). PROVEN minimal.
2. **F32 code read = ~400-500B.** Input is F32 one-hot; collapsing 10 channels →
   scalar color code needs a Conv/MatMul (F32 out, 10×10 = 400B) + Cast U8 100.
   The full 10×10 is used by the assembly (row 2 CAN carry the box bar under
   transpose — verified 766/3003 grids — so the conv crop cannot shrink).

## 2nd-pass CORRECTION to the first-pass floor proof
First pass claimed a separate ~256B F32 gray read was ALSO forced. FALSE — gray
detection folds into the code Conv (`w_code[5]=3`); one shared `Equal(Rmid,3)`
covers the interior wipe AND the reflected masks. That removed ~96B/-55 params
the first pass wrongly called forced. Lesson: before declaring a read "forced by
F32", check whether it can ride an existing reduction.

## Remaining ~860B = the two-orientation reflection (routing + assembly + wipe)
Orientation (h/v, bar row 3/4) is data-dependent, so an enumeration or Gather is
required. This lane's Gather formulation is empirically WORSE (2567, memory 2442
vs 2266). The ~28 8-byte routing masks, the 15×15 assembly rows, and the
Rmid/is_gray_rmid/Rmid_clean interior trio are each near-minimal.

## WALL
Sub-~2200 needs a fundamentally new scoreable surface that fuses code-read +
reflection + one-hot decode into fewer than the ~3 charged 15×15 planes the
QLinearConv decode requires. None known (`If`/top-level `Where` branch
materialization is scorer-rejected / 74,736B). Do not re-chase ~900.
