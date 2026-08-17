# Transferable insight — three param-lever + scoring-mechanism facts from a memory-floor task: (1) `calculate_memory` counts EVERY `value_info` name (orphans from node-removal surgery = phantom cost), (2) equal-width bit-unpack lanes (7/7/7 not 8/8/5) share ONE pow mask at identical memory, (3) when memory is at the representation floor the credited cost STILL has a params lever that a memory-only WALL misses

**Source**: task077 (`bo_ng_cld_worker2`, Claude lane, 2026-07-11). Live team-best **6,513**
(Boladi, sha `83095145…`, refreshed 2026-07-11). Staged **v5 = 6,503**, PATH_A_BYTE_EQUAL
(ab --n 2000 2266/2266 byte_equal=True; visible 266/266; truth 2000/2000; profile static==profile==6,503;
ng_safety PASS). A **real win via the params lever only** — memory stayed pinned at the 6,424 flood floor.

Context: the 6,513 live is my own prior `v3` U8-lane-unpack (6,524) that Boladi cut by **−11 params
via initializer DEDUP** (pow_mid→pow_lo, s1/steps1→axis1, unsq_axis2→s2). My 2026-06-30 "WALL at 6,524"
was **memory-only** and missed that lever entirely. Lesson 3 below.

## 1. `calculate_memory` counts orphaned `value_info` → phantom cost from node-removal surgery (NEW TRAP)
`neurogolf_utils.calculate_memory` (L215-236) builds `tensor_names` from **node outputs ∪ every
`value_info`/input/output name**, then charges `num_elements*itemsize` for each. It does **not** check
that a `value_info` entry still has a producing node. So if you do graph surgery that DELETES nodes
(e.g. replace an unpack block) but leave the removed tensors' `value_info` entries in the graph, the
scorer charges them anyway. On task077 this added a **phantom +1,120 B** (score read 7,623 instead of
6,503) — the exact sum of the old unpack's `lo_bits/mid_b/hi_u32/…` planes.
`onnx.shape_inference.infer_shapes` (which the scorer runs first, L196) does NOT remove pre-existing
orphan `value_info` — it only adds. **Fix / rule:** any node-deleting edit MUST prune `graph.value_info`
to `{produced ∪ IO}` before save:
```python
produced = {o for n in g.node for o in n.output}
io = {i.name for i in g.input} | {o.name for o in g.output}
keep = [vi for vi in g.value_info if vi.name in produced or vi.name in io]
del g.value_info[:]; g.value_info.extend(keep)
```
Corollary (both directions of danger): a surgery that *looks* like it saved memory may just be
double-counting phantoms, and a genuine win can be *masked* by orphans. Always diff per-tensor
static-vs-runtime after surgery (see `task077/prototypes/_diag_profile.py`).

## 2. Equal-width bit-unpack lanes share ONE pow mask at identical memory (7/7/7 beats 8/8/5)
A packed-word → per-cell BOOL exit (lib #157 bit-unpack) splits an N-bit row word into lanes, ANDs each
against a `pow` mask, booleanizes, and Concats. The natural split is **byte-aligned** (21 bits → 8/8/5),
which needs TWO masks `pow_lo[8]` + `pow_hi[5]` (13 params) because the lanes differ in width.
**Split into ⌈N/8⌉ EQUAL-width lanes instead** (21 → 7/7/7): all three lanes are 7 wide, so they share
**ONE `pow7[7]`** (7 params) → **−6 params**, and the shift constants collapse too (lane B = `w>>7`,
lane C = `(w>>7)>>7` reuses one `shift7`, vs byte-aligned `shift8`+`shift16`). **Memory is identical**:
3×[1,20,7]=420 == [1,20,8]+[1,20,8]+[1,20,5]=420. Non-byte-aligned extraction is safe because
`Cast(U32→U8)` truncates to the low 8 bits and the `AND pow7` drops the one stray high bit each lane
picks up, so lanes tile `0..20` with no overlap/gap (byte-equal verified 400/400 + ab 2266/2266).
General: for an N-bit unpack prefer L=⌈N/8⌉ lanes of width ⌈N/L⌉ sharing one mask; it never costs more
memory than byte-aligned and saves `(sum of byte-aligned widths) − ⌈N/L⌉` mask params.

## 3. On a memory-floor task the credited cost STILL has a params lever — a memory-only WALL is incomplete
Credited cost = `memory + Σ math.prod(init.dims)` (`score_network`; verified: 6,424+89 = 6,513 = manifest
exactly). When every intermediate is at the representation floor (task077's row-word flood: reach 2,160
forced by visible — 2-step=258/266, ±3=217/266; unpack 1,540; Pad 900 #48-locked; V5/V5j/between each
verified-needed), **params are the only remaining lever** and they are real bytes. Byte-IDENTICAL
(provably output-preserving, not just empirical) param eliminations to sweep before declaring a WALL:
- **initializer dedup** — identical bytes+dtype → keep one (Boladi's −11 here; `ng_graph_cost_diff`
  prints DUPLICATE-INITIALIZER GROUPS).
- **shift associativity** — `x<<2`≡`(x<<1)<<1`, `x>>2`≡`(x>>1)>>1`, `x>>16`≡`(x>>8)>>8` (unsigned, values
  in range) → drop the second shift constant, reuse the base. Same op/tensor count.
- **Pad constant_value omission** — `Pad(data,pads,"",axes)` defaults to 0 (ONNX spec, stack-invariant);
  drop the explicit zero initializer.
- **`Greater(bits,0)` → `Cast(bits,BOOL)`** for unsigned `bits` (nonzero-test ≡ `>0`); drops the zero
  comparand initializer, same BOOL output.
All four are stack-invariant, so the result stays **byte-equal to the live** (Path-A LOCAL_GOLD, no
Kaggle probe). Net task077: Boladi dedup −11, then v5 shift2/zero_u32/zero_u8 folds −3 + 7/7/7 shared
mask −6 = **6,524 → 6,513 → 6,503**, memory never moved.

## Rule of thumb
Before writing "WALL at C" on a bit-packed / flood / render task whose memory is at the representation
floor: (a) split cost into memory-vs-params and run the four byte-identical param folds + dedup above —
they are free strict wins the memory analysis cannot see; (b) if the exit is a bit-unpack, try
equal-width lanes with a shared mask; (c) after ANY node-deleting graph surgery, prune `value_info` and
re-score, or the number is fiction (±phantom planes). A memory-only WALL is a partial WALL.
