# Transferable insight — a data-dependent K×K neighbourhood read is a FLAT `Gather(cell_flat, fc + static_offsets)`, NOT GatherND(2D) or a shift-bank Conv-scatter; and this broke a documented "info-flow floor"

**Source**: task080 (`bo_ng_cld_worker3`, 2026-07-12). LIVE **5605** (Sho, Einsum graph, sha `9cd05340…`)
→ staged **4965 (−640, +0.121 pts)**, byte-equal (ab 2231/2231, truth 2000/2000, profile==static).
This is a **WIN**, and it partially overturns task080's `hard_records/color_index_collapse_false_floor.md`
WALL (the front-end-collapse route stays dead; the *template-read block* was the reducible slack the
same record flagged as a low-confidence "remaining hypothesis, ~300–800 B").

## The pattern that keeps overcharging: reading a K×K patch at a DATA-DEPENDENT location
Tasks that stamp/copy a motif (Filling Regions, template-stamp, marker-dilate) must read a K×K
neighbourhood around a computed cell `fc` (e.g. the max-neighbour template centre). Three ways, cheapest last:

1. **Shift-bank Conv-scatter** (what task080's live did): `GatherND(8 nbrs) → Reshape → Mul([8,1,3,3] bank) → ReduceMax`. Costs the I64 GatherND index (below) PLUS a K²- char param bank + a K²-plane Mul intermediate.
2. **GatherND(2D coords)**: index is **I64 `[…,K²,2]`** (GatherND indices are int64-only) = `8·K²·(bytes)`; a 3×3 read = 144 B index + 2× 64 B coord-halves during Concat build ≈ 256 B.
3. **FLAT `Gather(cell_flat, fc + offsets)`** ← cheapest. `cell_flat = Reshape(cell,[H*W])`; `idx = fc + [static flat offsets]` is **I32 `[K²]`** (Gather accepts int32) = 36 B for 3×3. One Gather → `[K²]` → Reshape `[1,1,K,K]`. **~1/4 the index bytes of GatherND, no param bank at all.** For task080: this + the two cheap follow-ons below took the whole template block from ~640 B to ~85 B.

**Requirement**: the read centre must be INTERIOR so the flat index never row-wraps (`fc±1` stays in-row).
For a max-K²-neighbour centre this is STRUCTURALLY guaranteed (an edge cell can't have K²−1 nonzero
neighbours; conv `pad=0` caps edge counts). The all-background SCORING DUMMY gives `fc=0` (negative
offsets) — fine, Gather supports negative indices → valid `[K²]` shape, garbage value (unused).

## Two things that bit, worth reusing
- **Cross-correlation kernel = rotate180(patch).** ONNX `Conv`/`QLinearConv` is cross-correlation, so to
  STAMP patch T at a delta-mask you feed the **180°-rotated** T as the weight (the live shift-bank encoded
  exactly this — nbr k's colour landed at 3×3 slot `8−k`). A flat Gather reproduces it directly with
  **reversed** offsets `[+K+1,+K,+K−1, +1,0,−1, −K+1,−K,−K−1]`; then zero the centre with a `[1,1,K,K]`
  mask Mul (or reuse an existing count-kernel init — the 8-nbr count kernel `[[1,1,1],[1,0,1],[1,1,1]]` IS
  the centre-zero mask, byte-identical → free via `share_initializer`).
- **Dynamic `Slice` (computed starts/ends) is UNSCOREABLE.** Its output shape is data-VALUE-dependent →
  `shape_inference(strict)` leaves it symbolic → `calculate_memory` returns None, AND it poisons every
  downstream tensor to `[1,?,…]` (the QLinearConv/pad/upsample chain, incl. the 900 B output canvas, all
  go dynamic). Injected `value_info` does NOT survive `infer_shapes(strict_mode=True)`. Use GatherND/flat
  Gather (output shape = INDEX shape, always static) for data-dependent reads.

## Two free byte-equal simplifications that stack on any argmax-detect graph
- **`ArgMax(masked_count)` == `ArgMax(Equal(count, ReduceMax(count)))`** — both return the FIRST max index.
  If you only need the argmax LOCATION, drop the `ReduceMax→Equal→Cast` (−two [1,1,H,W] planes ≈ 200 B),
  guaranteed identical even under ties.
- **Centre value via `Gather(cell_flat, fc)`** replaces `Where(centre_mask,cell,0)→ReduceMax` (−100 B).
  CAVEAT: differs from the Where/ReduceMax version only under a max-count TIE (ReduceMax-over-ties vs
  value-at-first-tie). Verify on the full ab corpus + truth; 0 ties in task080's 2231-input corpus and
  truth stayed 100%, but keep a tie-free fallback (opt-1-only) if the class can produce K×N block motifs.

## Replace FLOAT index arithmetic with a Gather-LUT (cheaper AND dodges the U8-downcast trap)
task080's upsample built its per-output-row cell/wall index with float `Div(row,period)`/`Mod`/`Equal`/
`Where` — 5× `[1,1,30,1]` f16 ≈ 210 B. `traps.md table.12` forbids the obvious "downcast to U8 integer
arithmetic" (a FLOAT-algorithm→U8 reformulation scored **−13.32 on hidden** here even after passing
visible+450 adversarials). The SAFE win is orthogonal: if the computed index is a function of only a
small discrete key (here `period_class ∈ {0,1,2}`) and a static axis (row), **Gather it from a
`[K,N]` LUT of the EXACT f16 values the live already produces** (extract `safe_name_110` per key; verify
it's constant per key across officials). This is a table lookup — it **removes** the float arithmetic
rather than downcasting it, so it's strictly SAFER than the incumbent, and byte-equal (the LUT holds the
identical f16 bits; the final `Cast→int32` truncates identically). −88 B on task080. Guard: the live's own
`Equal(key,[0,1,2])` one-hot proves the key range, so no new hidden-input assumption. (Note the earlier
`DONE_…coloridx_collapse_falsefloor` INT64-ArgMax trap is the opposite direction — don't *introduce* an
index plane; here we *replace* float arithmetic with a tiny f16 LUT.)

## Rule of thumb
On a template-stamp / K×K-copy task, `inspect_graph` the live best's template read. If it uses
**GatherND + a shift-bank/Mul/ReduceMax** to build the stamp kernel, replace with **flat
`Gather(Reshape(cell,[HW]), argmax_fc + static_flat_offsets)` + centre-zero Mul** (offsets reversed for
the cross-correlation rotation). Byte-equal, typically saves the I64-index delta + the whole param bank.
And distrust any "info-flow floor" that bills a data-dependent neighbourhood read as *forced* — a static
K×K read via flat Gather is `K²` char + one I32 `[K²]` index, not a plane.
