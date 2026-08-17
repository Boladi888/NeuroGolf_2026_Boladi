# task345 bank: plane reconstruction via Equal(iota, sum-code) + the BitwiseAnd CLASS-LUT tail (2026-06-07)

## Part 2 (added same session): BitwiseAnd class-LUT final — column-classed content with NO planes at all (v3/v4, 11,069 → 887)

When every output COLUMN (or row) falls into a small set of corpus-verifiable
classes (≤31), skip channel planes entirely:

  output = BitwiseAnd(A_lut [1,10,30,1] i32 INITIALIZER, B [1,1,1,30] i32)

- `A_lut[ch, r]` = bitmask of classes whose channel-ch fires at row r — an
  arbitrary per-class LUT, so the rank-1 limits that kill Add/GE/Equal
  broadcast tails (e.g. obstacle scatters: crossing-pair argument) DO NOT APPLY.
- `B[c] = 1 << class(c)` via a tiny u8 class-index chain + Gather(ptable).
  Width-30 Pad with value 0 → padding columns decode all-False for free.
- Decode: i32 graph output, `(out > 0)` — both AND operands non-negative.
- params side: LUT = 10·30 = 300p dense (sparse inits are scorer-fatal);
  >16 classes forces i32 (i16 saves nothing on params anyway — element count).
- Class index trick: collision-free SPARSE values beat compact renumbering —
  `k = m1 + 10·(1−b) + gate·(g+17)` (6 vec ops) with a 35-entry table beats a
  dense 17-entry mapping that needs more gating ops.
- Pair with #70 place-value packing: one free-input einsum can carry BOTH
  per-column codes (`me = m1 + 100·b`, u8-exact ≤ 107) when the second signal
  lives in rows the first signal's weights zero out; unpack Mod/Div(u8, const).
- task345: 4 staged rungs 2,751 / 1,691 / 925 / 887, ALL byte-equal vs LIVE on
  200k+264. Kaggle-novel lanes in the 887 rung: INT32 graph output decode,
  BitwiseAnd-final (task205 v31 precedent), Gather class table — one real
  submit validates the family.

## Part 1: plane RECONSTRUCTION via Equal(iota, sum-code) — skip the input crop entirely (task345 v1)

**When a needed input plane has ≤1 marker per column (or row), never Slice it.**
The [1,1,10,10] F32 ch-crop costs 400 B before you do anything. Instead:

1. `code = Einsum('bchw,c,h,d,e->bdew', input, sel_ch, ramp(r+1), one1, one1)` →
   [1,1,1,30] F32 (120 B). Sum == max because ≤1 marker/column, so `code(c)` =
   r_low+1, 0-sentinel when the column is empty. The `d,e` dummy [1] operands give
   rank-4 output directly (task246 v4 precedent) — no Unsqueeze.
2. The plane itself comes back as `Equal(R1_u8 [1,1,10,1], Cast(code) u8)` →
   [1,1,10,10] BOOL (100 B). 0-sentinel never equals r+1 ≥ 1, so empty columns
   decode all-False for free.

Net: 400 B slice + 100 B cast → 120 + 40(slice to 10) + 10(cast) + 100(Equal) =
270 B, **and** the code vector doubles as the threshold operand for any
per-column interval masks (GreaterOrEqual broadcasts against iota), which is
usually why you wanted the plane in the first place.

Used in task345 v1 (11,069 → 1,691, ab byte-equal): obstacle plane + both
line-interval masks all hang off the one einsum. Corpus-verify the ≤1-per-column
invariant first (200k sweep); the boring sibling (ReduceMax((r+1)·plane) on a real
Slice) hedges the invariant at +~500 B if there is any doubt.

u8 vector-math reminder: Mul/Sub/Add(u8) all run fine but need **opset 14**
(#74 narrow-int shape-inference trap — now also load-verified via task345 v1,
along with Pad(BOOL) at 14 and Expand(BOOL)).
