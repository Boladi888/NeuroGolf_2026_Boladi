# bit_packing_calculus.md

## Bitwise and integer operator matrix

[PROMPT] I assume your cost model exactly as stated: memory is the sum of **trace-visible intermediate tensor bytes**, while params are **initializer element counts**, with graph inputs/outputs and initializers themselves free; BOOL and U8 both cost 1 byte per element in the trace. [PROMPT] I therefore write one BOOL/U8 plane of shape `H×W` as `P = H·W` trace-bytes, one I32 plane as `4P`, one I64 plane as `8P`, and one F16 plane as `2P`. [SPECULATIVE] All break-even algebra below is derived against that metric, not wall-clock runtime.

[ORT-DOC] ORT’s `OperatorKernels.md` is auto-generated from the registered kernels, and [ORT-SRC] the CPU kernel registrations in `element_wise_ops.cc` confirm the relevant rows, so the combination of those two is the strongest internet-only evidence for ORT CPU availability short of actually running your grader binary.

| op | ONNX spec at opset ≤ 20 | ORT CPU rows at opset ≤ 20 | packing consequence |
|---|---|---|---|
| `BitShift` | [SPEC] `BitShift` is opset 11, is **elementwise**, broadcasts, and only accepts `uint8/uint16/uint32/uint64`.  | [ORT-DOC] CPU docs show `uint8/uint32/uint64`. [ORT-SRC] The CPU source explicitly registers `uint8`, `uint32`, `uint64` and comments out `uint16`.  | [SPECULATIVE] U8 is a natural packed carrier. [SPECULATIVE] Signed I32/I64 are **not** shift-friendly on ORT CPU. |
| `BitwiseAnd` / `Or` / `Xor` / `Not` | [SPEC] All four are opset 18, elementwise, broadcast, and accept integer tensors `int8/int16/int32/int64/uint8/uint16/uint32/uint64`.  | [ORT-DOC] CPU docs list the same integer set for opset 18+. [ORT-SRC] CPU registrations exist for all those integer widths for `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`.  | [SPECULATIVE] This is the key enabler: packed-word boolean logic can stay packed, so one packed op can replace `n` per-plane BOOL ops. |
| `Mod` | [SPEC] `Mod` is opset 13; with default `fmod=0` it is integer modulo, elementwise, broadcast.  | [ORT-DOC] CPU docs list opset 13+ support for `float/float16/double/int8/int16/int32/int64/uint8/uint16/uint32/uint64`.  | [SPECULATIVE] `Mod` supports unpack by `(x / 2^i) % 2`, but bitwise masking is cheaper when you only need a BOOL plane. |
| `Div` | [SPEC] `Div` is opset 14 for the widened integer set; integer division is truncating-toward-zero.  | [ORT-DOC] CPU docs list opset 14+ support for `float/float16/double/int8/int16/int32/int64/uint8/uint16/uint32/uint64`. [ORT-SRC] CPU registrations match that widened set.  | [SPECULATIVE] `Div` is usable for arithmetic unpack ladders, but it is not required for the minimal BOOL extraction path. |
| integer `Add` / `Sub` / `Mul` | [SPEC] Opset 14 widened all three to include `uint8/int8/uint16/int16` in addition to `float/float16/bfloat16/int32/int64/uint32/uint64`.  | [ORT-DOC] CPU docs list those widened rows. [ORT-SRC] CPU registrations match them.  | [SPECULATIVE] These let you pack by weighted sums (`Σ bit_i·2^i`) even where `BitShift` is unavailable on signed types. |
| `Cast` | [SPEC] `Cast` supports numeric↔numeric conversion; fixed-point to BOOL maps zero→False and nonzero→True; BOOL to fixed-point maps `{True,False}`→`{1,0}`; BOOL, U8, I32, I64, F16 are all valid input/output types.  | [ORT-DOC] CPU docs show Cast rows at opset 13+/19+/21+ covering BOOL, F16, int8/int16/int32/int64, uint8/uint16/uint32/uint64, float, double.  | [SPECULATIVE] `Cast` is enough to bridge BOOL planes into integer carriers and back out without touching floating arithmetic. |

[ORT-SRC] Standard logical `And`/`Or`/`Xor`/`Not` on ORT CPU remain BOOL-only kernels, separate from the integer `Bitwise*` family. That means packed execution must use `Bitwise*`, not logical `And`/`Or`/`Xor`.

### Carrier round-trips that actually work

[SPECULATIVE] **BOOL ↔ U8:** yes, cleanly. [SPEC] `Cast(BOOL→U8)` gives 0/1, [SPEC] `BitShift` works on U8, [SPEC] `BitwiseAnd/Or/Xor/Not` work on U8, and [SPEC] `Cast(U8→BOOL)` maps zero/nonzero back to False/True. Packing can use `Cast + left-shift + Or/Add`; unpacking can use either `(right-shift + And 1 + Cast)` or the strictly cheaper `(And mask + Cast)` path when the consumer only needs BOOL.

[SPECULATIVE] **BOOL ↔ I32:** yes, but not with signed `BitShift` on ORT CPU. [SPEC] `BitwiseAnd/Or/Xor/Not` accept I32, [SPEC] `Div` and `Mod` accept I32, and [SPEC] `Cast` bridges BOOL↔I32. [SPECULATIVE] So the integer-only round-trip is `Cast(BOOL→I32)` then `Mul/Add` or `Mul/BitwiseOr` for packing, and `BitwiseAnd(mask) + Cast(→BOOL)` or `Div/Mod + Cast` for unpacking. [SPECULATIVE] This is functionally valid, but byte-costly because every I32 intermediate is `4P`.

[SPECULATIVE] **BOOL ↔ I64:** same story as I32, just even more byte-expensive (`8P` per intermediate). [SPEC] The operators exist; [SPECULATIVE] the calculus is usually hostile unless the packed region is long and avoids unpacking almost entirely.

[SPECULATIVE] **BOOL ↔ F16 as a packed carrier:** no, not for an “entirely integer” round-trip. [SPEC] F16 can be cast to and from BOOL and ints, but [SPEC] neither `BitShift` nor `Bitwise*` accept floating-point tensors. F16 is therefore only a sink/source after counting or thresholding, not a useful boolean-packed carrier.

### Cast pairwise matrix for the requested types

| from \ to | U8 | I32 | I64 | BOOL | F16 |
|---|---:|---:|---:|---:|---:|
| U8 | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes |
| I32 | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes |
| I64 | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes |
| BOOL | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes |
| F16 | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes | [SPEC][ORT-DOC] yes |

[SPEC] The semantic details that matter here are the zero/nonzero BOOL casts and `BOOL→{1,0}` numeric casts. [ORT-DOC] ORT CPU supports all five requested types on Cast at opset ranges that include ≤20.

## Packing calculus

[PROMPT] Let `P` be the trace-bytes of one BOOL/U8 plane. [SPECULATIVE] Let `w` be carrier byte-width: `w=1` for U8, `w=4` for I32, `w=8` for I64. [SPECULATIVE] Let `n` be logical BOOL planes packed into one carrier plane, `m` the number of downstream 2-input boolean consumer sites, `α` the fraction of those sites that can execute directly on packed words, and `g` the number of distinct full-unpack stages whose outputs get reused by downstream fan-out. [PROMPT] Because one materialized plane is charged once regardless of fan-out, `g` is the right reuse variable, not raw edge count.

### Primitive costs under your trace model

[SPECULATIVE] The crucial issue is **node outputs**, not arithmetic complexity. [SPECULATIVE] Every `Cast`, `BitShift`, `BitwiseAnd`, `Add`, `Div`, and `Mod` produces another full-sized trace-visible tensor. [SPECULATIVE] Under that metric, the cheapest useful primitives are these:

| primitive | incremental trace bytes | param count |
|---|---:|---:|
| [SPECULATIVE] Pack `n` BOOL planes → one U8 plane via `Cast + Shift + Or/Add` | `(3n−2)P` | `n−1` scalar shift constants |
| [SPECULATIVE] Pack `n` BOOL planes → one I32 plane via `Cast + Mul + Add/Or` | `4(3n−2)P` | `n−1` scalar powers-of-two |
| [SPECULATIVE] Unpack one U8 bit to BOOL via `BitwiseAnd(mask) + Cast` | `2P` | `1` mask scalar |
| [SPECULATIVE] Unpack one U8 bit to BOOL via `Shift + And(1) + Cast` | `3P` | `1` shift scalar, plus shared `1` |
| [SPECULATIVE] Unpack one U8 bit to BOOL via `Div + Mod + Cast` | `3P` | `1` divisor scalar, plus shared `2` |
| [SPECULATIVE] Unpack one I32 bit to BOOL via `BitwiseAnd(mask) + Cast` | `5P` | `1` mask scalar |
| [SPECULATIVE] Unpack one I32 bit to BOOL via `Div + Mod + Cast` | `9P` | `1` divisor scalar, plus shared `2` |
| [SPECULATIVE] One packed boolean consumer on carrier width `w` | `wP` | `0` |
| [SPECULATIVE] One unpacked BOOL consumer across `n` planes | `nP` | `0` |

[SPEC] The operator legality behind those constructions is the ONNX/ORT support matrix above: U8 supports `BitShift`; signed I32 does not, but does support `BitwiseAnd` and integer `Mul/Add/Sub/Div/Mod`; BOOL casts are well-defined on zero/nonzero.

### Baseline and packed formulas

[SPECULATIVE] If you keep everything unpacked, the downstream region costs

\[
C_{\text{unpacked}} = mnP
\]

[SPECULATIVE] If you pack once, execute `αm` consumer sites on the packed carrier, and pay `g` shared full-unpack stages for the remaining sites, the region costs

\[
C_{\text{packed}} = p_w(n)P + \alpha m\,wP + (1-\alpha)mnP + gnu_wP
\]

[SPECULATIVE] where `p_w(n)` is pack cost in plane-bytes and `u_w` is per-plane unpack cost in plane-bytes. [SPECULATIVE] For the practical cases above, `p_1(n)=3n−2`, `p_4(n)=4(3n−2)`, `u_1=2` for `mask+Cast`, and `u_4=5` for I32 `mask+Cast`.

[SPECULATIVE] Subtracting the two gives the key break-even condition:

\[
\Delta = C_{\text{packed}} - C_{\text{unpacked}}
       = P\big[p_w(n) + gnu_w - \alpha m (n-w)\big]
\]

[SPECULATIVE] So packing wins exactly when

\[
\alpha m (n-w) > p_w(n) + gnu_w
\]

[SPECULATIVE] That one line is the full answer to “when packing beats plane-count” in your metric.

### What this does to the old plane-neutral claim

[SPECULATIVE] If **no** consumer can execute on packed words, so `α=0`, then

\[
\Delta = P\big[p_w(n)+gnu_w\big] > 0
\]

[SPECULATIVE] which means packing is **strictly worse**, not merely neutral, under a sum-of-intermediates cost model. [SPECULATIVE] The old claim is therefore too generous even before we use opset-18 bitwise kernels. [PROMPT] Your internal “every pack costs one Div/Mod unpack” intuition was already pessimistic about the *unpack primitive* and optimistic about *neutrality*: with opset-18 bitwise ops, the cheapest BOOL extraction is `And(mask)+Cast`, not `Div+Mod`; and when `α=0`, packing still loses because you have both the pack outputs and the unpack outputs.

[SPECULATIVE] If **all** consumers can remain packed and the payload is already packed on entry, so `p_w=0`, `g=0`, `α=1`, then the output-side savings factor is simply

\[
\frac{mnP}{mwP} = \frac{n}{w}
\]

[SPECULATIVE] For U8 that is `n×`; for I32 that is `n/4×`. [SPECULATIVE] So packed-executable consumers are the entire game.

### Concrete break-even cases

[SPECULATIVE] **U8, fully packed region, internal pack, no unpack inside the region:**
\[
m > \frac{3n-2}{n-1}
\]
[SPECULATIVE] For `n=8`, that is `m > 22/7`, so **four** packed boolean consumer sites already beat the unpacked realization.

[SPECULATIVE] **U8, fully packed region, but one shared final full-unpack stage is charged once:**
\[
m > \frac{(3n-2)+2n}{n-1} = \frac{5n-2}{n-1}
\]
[SPECULATIVE] For `n=8`, that is `m > 38/7`, so **six** packed consumer sites are enough.

[SPECULATIVE] **U8, no reuse, every unpack-needing site forces a fresh full unpack (`g=(1-\alpha)m`), using the cheapest `mask+Cast` extraction:**
For large `m`, packing can only win if
\[
\alpha(n-1) > 2n(1-\alpha)
\]
which is
\[
\alpha > \frac{2n}{3n-1}
\]
[SPECULATIVE] For `n=8`, that means **more than 69.6%** of the consumer sites must stay packed. [SPECULATIVE] If you insist on the more expensive 3-op unpack (`Shift+And+Cast` or `Div+Mod+Cast`), the asymptotic threshold rises to
\[
\alpha > \frac{3n}{4n-1}
\]
which is **77.4%** at `n=8`.

[SPECULATIVE] **I32, fully packed region, internal pack, no unpack inside:**
\[
m > \frac{4(3n-2)}{n-4}
\]
[SPECULATIVE] For `n=8`, that is `m > 22`, which is terrible compared with U8. [SPECULATIVE] So for packing up to eight BOOL planes, I32 is usually the wrong carrier under your metric even though the operators exist.

### When the total win exceeds 2×

[SPECULATIVE] The packed realization beats unpacked by **at least 2×** when `C_packed ≤ C_unpacked/2`. [SPECULATIVE] In the practically important U8 cases, that becomes:

- [SPECULATIVE] **U8, payload already packed, all consumers packed:** `n ≥ 2` is already enough for ≥2× on the consumer outputs, because `n/1 ≥ 2`.
- [SPECULATIVE] **U8, internal pack, all consumers packed, no unpack inside:**
  \[
  mn \ge 2\big((3n-2)+m\big)
  \;\Longrightarrow\;
  m(n-2) \ge 6n-4
  \]
  [SPECULATIVE] For `n=8`, this needs `m ≥ 8`.
- [SPECULATIVE] **U8, internal pack, all consumers packed, one shared final unpack:**
  \[
  mn \ge 2\big((3n-2)+m+2n\big)
  \;\Longrightarrow\;
  m(n-2) \ge 10n-4
  \]
  [SPECULATIVE] For `n=8`, this needs `m ≥ 13`.

[SPECULATIVE] For strict I32 carriers with internal packing, ≥2× is far harder; with the formula above, `n` must exceed `8` before a 2× total-fragment win is even algebraically possible.

## Reduction interactions

[SPECULATIVE] A plain `ReduceSum` over packed words is **not** a popcount; it sums byte or word *values*, not the number of 1-bits inside those values. [SPECULATIVE] So any packed counting or thresholding path needs a **local per-element popcount** first, then an optional compare or spatial reduction. [SPEC] `Gather` is legal with `int32/int64` indices, and [SPEC] `ReduceSum` exists at opset 13, but [ORT-DOC] ORT CPU’s `ReduceSum` rows are effectively `double/float/int32/int64`, not BOOL or U8.

### Popcount options that fit ONNX ≤ 20

| route | packed realization | local trace bytes | params | remarks |
|---|---|---:|---:|---|
| [SPECULATIVE] Div/Mod ladder | For `n` bits: `bit0 = x % 2`, `biti = (x / 2^i) % 2`, then sum `n` one-bit terms with `n−1` Adds | `(3n−2)wP` | `n` scalars | [SPECULATIVE] Always legal with U8/I32/I64 integer carriers, but almost never the cheapest. |
| [SPECULATIVE] U8 SWAR byte-popcount | `x = x - ((x >> 1)&0x55); x = (x&0x33)+((x>>2)&0x33); pc = (x + (x>>4))&0x0F` | `10P` | about `6` scalars | [SPECULATIVE] Very attractive for U8. [SPEC] It uses only `BitShift`, `BitwiseAnd`, `Sub`, `Add`, all supported on U8. |
| [SPECULATIVE] 32-bit multiply-shift popcount | Classical “best method”: subtract/and, pairwise-add, nibble-accumulate, multiply by `0x01010101`, shift-right to final count | `48P` on `uint32`; about `52P` if you first cast I32→U32 | about `8` scalars | [SPECULATIVE] Good for row-packed `uint32`; awkward for strict signed I32 because `BitShift` is unsigned-only. |
| [SPECULATIVE] LUT-on-byte via `Gather` | `idx = Cast(U8→I32)`; `pc = Gather(LUT256, idx)` | `5P` if LUT outputs U8 (threshold-only), `8P` if LUT outputs I32 (count/sum-friendly) | `256` LUT entries | [SPECULATIVE] Best constant-factor route for U8 when 256 params are acceptable. |

[SPEC] The Stanford “Bit Twiddling Hacks” page explicitly gives both the parallel count family and the 12-operation multiply-shift “best method.” [SPEC] The lookup-table route is also standard byte-popcount prior art.

### Pricing the major routes in your model

[SPECULATIVE] **Div/Mod ladder on U8, counting all 8 bits of a packed byte:**
`(3·8−2)P = 22P` local trace-bytes, plus roughly `8` scalar params. [SPECULATIVE] Thresholding to BOOL adds one compare output (`+P`) and one scalar threshold. [SPECULATIVE] If you also need a spatial total, cast the local count plane to I32 if it is not already I32, then `ReduceSum`.

[SPECULATIVE] **U8 SWAR byte-popcount:**
`10P` local trace-bytes, plus about `6` scalar params. [SPECULATIVE] This is already less than half the local byte cost of the U8 Div/Mod ladder for an 8-bit carrier.

[SPECULATIVE] **U8 LUT-on-byte, threshold-only:**
Mandatory `Cast(U8→I32)` for indices is `4P`; a `Gather` from a U8 LUT is `+P`; total `5P`, plus `256` params. [SPECULATIVE] This is the best byte-cost thresholding route I found for U8.

[SPECULATIVE] **U8 LUT-on-byte, count then spatial sum:**
Use an I32 LUT instead of a U8 LUT. Then local cost is `4P + 4P = 8P`, plus `256` params, and the output is already in a `ReduceSum`-friendly type. [SPECULATIVE] For any nontrivial plane size, `8P + 256` is dramatically better than unpacking all eight bits or running a `22P` Div/Mod ladder.

[SPECULATIVE] **Packed counting therefore absolutely can beat unpacking.** [SPECULATIVE] The cheapest U8 popcount routes are not “neutral”; they are meaningfully better than full unpack-and-count, especially when the packed byte is only ever compared to a threshold like “≥k” or “==k”.

### Practical consequence for thresholding

[ORT-DOC] ORT CPU supports integer `Greater`, `GreaterOrEqual`, `Less`, and `LessOrEqual` rows on U8/I32/I64-class inputs, returning BOOL.

[SPECULATIVE] So the best packed-threshold pattern is usually: **local popcount in U8 or I32, then compare to a scalar threshold, and only cast or unpack later if a later stage truly needs per-bit planes**. [SPECULATIVE] That is a direct counterexample to the “every pack costs one unpack” intuition.

## Prior art and transferability

[SPEC] Tony Finch’s Life writeup describes both a lookup-table parallel method and a **parallel bit-plane addition** method in which one register holds bit 0 of many neighbor counts, another holds bit 1, another bit 2, and boolean formulas for full/half adders build the counts in parallel; on a 32-bit machine his implementation is 32-way parallel.

[SPEC] Cameron Browne’s *Bitboard Methods for Games* abstracts the same core idea: bitboards are efficient because they support **bitwise-parallel operation across all board cells at once**, and the paper’s figures explicitly include board encodings with **separating bits** and masks to fence off row structure.

[SPEC] Chessprogramming’s set-wise bitboard literature shows the transferable neighborhood tricks very cleanly: one-step directional shifts operate on all set bits in parallel, vertical shifts simply drop out-of-range bits, horizontal/diagonal shifts need masks to prevent wraparound, and repeated shift/intersect/OR cycles implement flood-fill-like propagation (`Dumb7Fill`).

### What transfers directly to flood-fill and dilation

[SPECULATIVE] The transferable trick is simple: if a row or board state is bit-packed, then **east/west/north/south and diagonal neighborhoods are shift-derived sets**, and dilation/flood-fill becomes repeated set-wise shift-and-OR, optionally masked by blockers or borders. [SPEC] That is exactly how bitboards model movement and fill in board games, and [SPEC] Finch’s Life implementation uses bitwise-parallel adders to aggregate neighborhood information.

[SPECULATIVE] For Game-of-Life-style 3-neighborhoods on packed rows, the usual pattern is: derive left/self/right shifted versions of the current row, combine those with the previous and next rows, then run a popcount/adder rule. [SPECULATIVE] For flood-fill/dilation, the popcount is unnecessary; repeated directional expansion plus masking is enough.

### What breaks at row boundaries

[SPEC] Bitboard sources repeatedly warn about wraps from one edge to the opposite edge and fix them with masks or separators. [SPEC] Chessprogramming’s “one step only” section is explicit: east/west and diagonals need `notAFile`/`notHFile`-style masks to prevent wraparound. [SPEC] Browne’s paper likewise shows encodings with separating bits and masks.

[SPEC] ONNX `BitShift` is **elementwise** on each tensor element.

[SPECULATIVE] That means a shifted-out bit from one packed scalar **cannot** flow into the next tensor element. [SPECULATIVE] So if a single image row spans multiple packed scalars, then intra-row shifts at the ONNX level do **not** automatically propagate carries across chunk boundaries. [SPECULATIVE] You either need one packed scalar per row, or explicit seam-handling logic with extra masks and tensor movement.

[PROMPT] Because your working widths are typically `≤30`, [SPECULATIVE] a row-packed `uint32` representation is unusually attractive for row-wise neighborhood algorithms: one row fits in one 32-bit lane, so the cross-word-carry problem disappears and only left/right edge masks remain. [SPECULATIVE] That is a different packing axis from “pack multiple semantic planes into one pixel word,” but it is where the classic CA/bitboard prior art transfers most cleanly.

### Open questions and limits

[SPECULATIVE] I did **not** test your exact grader binary; this note is based on ONNX spec pages, ORT’s auto-generated CPU kernel inventory, and current ORT CPU source registrations. [SPECULATIVE] If your grader uses an older ORT than the current public `main`, a few edge rows could differ, although the opset-18 integer bitwise story is strongly evidenced by the current registry.

[SPECULATIVE] The more ambitious compositions here—especially strict-signed-I32 wider-popcount routes and row-packed CA graphs built from generic ONNX tensor plumbing—are analytic internet-only conclusions, not empirical measurements on your environment.

## Recipes table

[PROMPT] In the table below, **planes** means incremental trace-visible byte cost in units of `P` (one BOOL/U8 plane’s bytes), not peak live memory. [SPECULATIVE] Break-even formulas are against the unpacked baseline `mnP`.

| structure (n planes, consumer mix) | packed realization | planes | params | break-even vs unpacked | caveats |
|---|---|---|---:|---|---|
| [SPECULATIVE] `n≤8`, almost all consumers are `AND/OR/XOR`, packed payload can stay packed | [SPECULATIVE] Pack once to U8; run consumers with integer `BitwiseAnd/Or/Xor`; postpone any unpack to the graph boundary | [SPECULATIVE] pack `3n−2`; each packed consumer `+1` | [SPECULATIVE] `n−1` | [SPECULATIVE] if no unpack inside, `m > (3n−2)/(n−1)`; if payload arrives pre-packed, any `m≥1` wins and per-consumer saving is `n−1` planes | [SPECULATIVE] Best case; this is where packing is emphatically **not** plane-neutral |
| [SPECULATIVE] `n≤8`, mixed graph, but all unpack-needing consumers share one materialization | [SPECULATIVE] U8 pack; execute `αm` sites packed; one shared full unpack by `mask+Cast`; fan out reused BOOL planes | [SPECULATIVE] pack `3n−2`; packed sites `+αm`; shared unpack `+2n`; unpacked sites `+(1−α)mn` | [SPECULATIVE] about `2n−1` | [SPECULATIVE] `αm(n−1) > 5n−2` in the one-shared-unpack special case | [SPECULATIVE] Reuse is decisive; one shared unpack is far cheaper than unpack-per-site |
| [SPECULATIVE] `n≤8`, no consumer can execute packed | [SPECULATIVE] U8 pack followed by immediate unpack to BOOL for all sites | [SPECULATIVE] pack `3n−2`; unpack `+2gn`; sites `+mn` | [SPECULATIVE] about `(n−1)+gn` | [SPECULATIVE] never wins when `α=0` | [SPECULATIVE] This is the only regime where the old “packing doesn’t help” intuition is directionally right, but it still understates the loss |
| [SPECULATIVE] `5≤n≤31`, most consumers packed, strict signed 32-bit carrier | [SPECULATIVE] Pack to I32 with `Cast+Mul+Add/Or`; run packed sites with `Bitwise*`; unpack with `mask+Cast` only if forced | [SPECULATIVE] pack `4(3n−2)`; each packed consumer `+4`; shared unpack `+5n` | [SPECULATIVE] `n−1` plus masks if unpack | [SPECULATIVE] fully packed/no-unpack: `m > 4(3n−2)/(n−4)` | [SPECULATIVE] ORT CPU does not `BitShift` signed I32; byte cost is harsh; for ≤8 planes this is usually inferior to U8 |
| [SPECULATIVE] U8 packed counts / thresholds only | [SPECULATIVE] Use byte-popcount, preferably `Gather` LUT or U8 SWAR, then compare count to scalar | [SPECULATIVE] LUT threshold path `5`; LUT count path `8`; U8 SWAR `10`; Div/Mod ladder `22` for 8 bits | [SPECULATIVE] LUT `256`; SWAR about `6`; Div/Mod about `8` for 8 bits | [SPECULATIVE] LUT-count beats U8 Div/Mod immediately for `n=8` (`8P` vs `22P`) | [SPECULATIVE] Gather needs `I32/I64` indices; if you need `ReduceSum`, make the local count I32 |

## Blunt verdict

[SPECULATIVE] **“Packing is plane-neutral” is false in general.** [PROMPT] It is only remotely true in the degenerate regime where downstream consumers cannot execute on the packed representation, you immediately unpack, and reuse is poor; even there, under your actual “sum of trace-visible intermediates” metric, packing is usually **worse than neutral** because pack outputs and unpack outputs both count. [SPEC] Opset-18 integer `BitwiseAnd/Or/Xor/Not` on ORT CPU are exactly the feature that breaks the old premise: once a nontrivial fraction of consumers can stay packed, each packed consumer saves `n−w` plane-bytes relative to `n` separate BOOL outputs, and with reuse the one-time unpack cost is amortized rather than repeated.  [SPECULATIVE] In your metric, **U8 packing wins early and wins hard** whenever at least a moderate packed region exists: for `n=8`, an internally packed all-packed region breaks even around `m=4`, still breaks even around `m=6` with one shared final unpack, and reaches a total-fragment win of at least **2×** around `m=8` with no internal unpack or around `m=13` with one shared final unpack. [SPECULATIVE] By contrast, strict signed-I32 packing is usually a trap for ≤8 planes because every intermediate is `4P` and `BitShift` is unavailable on I32. [SPECULATIVE] So the exact corrected statement is: **packing loses when consumers cannot execute packed; packing wins when enough consumers do execute packed and any required unpacks are shared; and ≥2× wins happen readily for U8-packed regions, but not for strict I32 unless the packed region is much larger.**