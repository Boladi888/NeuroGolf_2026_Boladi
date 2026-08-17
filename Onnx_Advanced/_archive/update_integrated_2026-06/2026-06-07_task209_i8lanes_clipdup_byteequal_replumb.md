# task209 worker drop (2026-06-07): byte-equal re-plumb kit — NIBBLE-PACKED plane, i8 lanes, clip-dup phantom class, Min-trick, einsum rank-drops

Context: task209 LIVE 9,487 → byte-equal ladder v3..v6; v5 (5,826) SHIPPED +0.49 EXACT same
session; v6 (5,251) staged byte-equal on top (−45% cumulative). Six generalizable findings,
all probe- or gate-verified on ORT v1.24.x:

## 0. NIBBLE-PACKED transport plane — halve ANY color-index plane (the headline; task209 v6)
The entry channel-collapse conv can pack 2 cells/byte: kernel `[1,10,1,2]` stride (1,2),
weights `(val, 16·val)` ⇒ `packed[r][p] = color(2p) + 16·color(2p+1) ≤ 153` (F32/u8 exact).
Plane [1,1,H,W]+u8 → [1,1,H,ceil(W/2)]+u8 = **half the dominant tensor pair**.
The law that makes it pay: `packed > 0 ⟺ pair-has-content`, so every BOOLEAN consumer
(row/col-any vectors, Min(x,1)+ArgMax extents) and every ROW-GATHER runs on the packed plane
unchanged at half width; only VALUE reads unpack: `hi = BitShift(p, 4, RIGHT)` (opset 11),
`lo = p − 16·hi` (u8 Mul/Sub, opset 14) — no BitwiseAnd/opset-18 needed; cell select =
`Where(parity, hi, lo)`. task209: plane 1,620 → 810, net −575 after unpack logic.
**Two mandatory disciplines (both bit task209 before gating green):**
(a) **max-of-packed is BOOLEAN-ONLY** — nibbles of a ReduceMax'd byte are cross-row garbage
    (`max(8+16, 48) = 48` loses lo=8); for nibble checks, gather the REAL pair column and
    reduce each nibble plane separately.
(b) **clip CELL coords BEFORE pair/parity** — negative/overflow coords under Div-trunc +
    Mod(sign=divisor) route to a DIFFERENT phantom cell than a clipped-cell gather; such
    reads can be geometry-VALID (box marker at grid col 0), so validity masking does not
    cover it. Clip cells to the plane range first, then decompose.
Applicability scan hint: any graph whose dominant tensors are a color-index plane + its u8
cast, with reads that are row-gathers + point/window lookups (the #23 representation class).
Gate proof: task209 v6 (5,251) ab --full 200,266/200,266 byte-equal vs the shipped v5;
real-input profile == static EXACT. 2 cells/byte is the u8 ceiling (4 nibbles ⇒ u16, no
clean ORT path); pair this with #81 before converting — the packing pays only while the
plane (not the output stage) is the dominant tensor.

## 1. `Where(bool,i8,i8)` has NO ORT CPU kernel (probed @opset14: NOT_IMPLEMENTED)
Sub/Div/Less/GreaterOrEqual/Cast all have i8 kernels (opset 14 min for arith — #74), and
integer Div(i8) truncates toward zero like i32. **Route i8 selects through u8**: cast the
i8 value u8 BEFORE the Where (wrapped negatives land only in unselected lanes; Where(bool,
u8,u8) is kernel-backed), then one final Cast→i32 where a Gather needs indices. An index
pipeline `[N] i32 ×k tensors` becomes `[N] i8/u8 ×k + one [N] i32` — task209 render lanes:
−251 B. Candidate add to #74's lane table.

## 2. CLIP-DUP PHANTOM class — shrinking a read plane changes Clip-overflow semantics
On a full-grid plane, an overflowing clamped read (`Clip(idx,0,EDGE)`) lands on an EMPTY
row/col; after cropping the plane tighter, the same Clip DUPLICATES the last real row/col.
Phantom duplicated guide cells made wrong hypotheses pass an exact moment test (4 visible
fails). **Any plane-shrink must re-prove every Clip's overflow target is empty OR mask it.**
Two exact-emulation fixes: (a) mask gathered windows beyond the measured extent
(`Where(And(iota<gh, iota<gw), GW, 0)`), (b) add the plane-edge bound to the validity
predicate (`≤ bright AND ≤ EDGE`). Trap-worthy: this is invisible on corpora whose
geometry never reaches the edge — task209 needed h=20/t=2 (srow=17) or bright=19 pairs.

## 3. `Min(vec, 1)` replaces `Greater(vec,0)+Cast(u8)` in front of first/last ArgMax extents
Min(u8) kernel exists at opset 13 (probed). One node instead of two, u8 stays u8:
`ArgMax(Min(rowany,1))` / `select_last_index=1` gives first/last-set directly (#64 pattern).
−19 B and −1 node per extent vector.

## 4. Einsum rank-drops kill scalar plumbing
- `'bchw,c->h'` (batch axis ABSENT from output) emits a clean `[30]` — probe-verified.
  ArgMax(axis=0, keepdims=0) then yields true `[]` scalars: no Reshape/keepdims glue
  anywhere downstream. (Yellow-marker row/col marginals: count==2 exactly at the two
  marker rows ⇒ ArgMax first/last IS the box geometry, no Greater needed.)
- `'bchw,bchw->'` self-dot (same tensor twice) replaces Mul+ReduceSum for Σx² — F16 in,
  cast-inserted F32 compute (#77 safe class), probed exact. −36 B per use.

## 5. Scorer profiles on a DUMMY grid — crash-proof every data-dependent index
`score_network_v2`/`calculate_memory` run the model on a synthetic input; garbage geometry
(ArgMax of an all-zero vector → extent 29) drove ScatterND out of bounds → score None.
Clamp SEMANTIC scalars once (`gh≤3`, `gw≤5`, scatter idx clips — identity on real inputs)
instead of guarding every consumer. Cheaper than per-vector clips and keeps real-input
semantics byte-identical.

Full detail: `The_400_Tasks/Task_by_Task/task209.md` (2026-06-07 entries + don't-do log);
builders `task209/prototypes/build_task209_compress_v{3,4,5}.py`.
