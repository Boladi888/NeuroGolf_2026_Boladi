# Transferable insight — two output-identical byte shaves on GatherElements probe lanes: (1) unsqueeze BEFORE the broadcast-add, (2) zero-pad the gathered lane to delete Mod+Less+Where

**Source**: task365 batch (`bo_ng_cld_worker2`, 2026-07-12). Live team-best **2430** (Boladi, corner-detect
structure, sha `7cc43a3d…`). Shaved to **2360** (−70, +0.029 pts) by two pure byte-layout edits that are
**byte-OUTPUT-identical** to the parent (ab byte_equal 2266/2266 + 337 random grids + truth 2000/2000).
Staged `staged_candidates/task365_offsetfuse_zeropad_cost2360/`; builder
`prototypes/build_task365_offfuse_v2.py`. Both edits are generic to ANY graph that builds
`GatherElements`/`Gather` probe indices as `base + offset` then masks out-of-range probes.

## Trick 1 — OFFSET FUSION: unsqueeze the small `[K,1]` operand, not the broadcast `[K,P]` result
A very common probe pattern is: take a per-item scalar `base` `[K,1]`, add a probe-offset vector
`off` `[P]` → `idx[K,P] = base + off`, then `Unsqueeze` `idx` up to the data's rank (e.g. `[1,1,K,P]`)
so `GatherElements` accepts it. If you Unsqueeze AFTER the add, the materialised index tensor is the
FULL `[1,1,K,P]` (K·P·4 B, int32). **Unsqueeze commutes with the elementwise add under the same
broadcast**, so move it BEFORE: unsqueeze `base` `[K,1]→[1,1,K,1]` (K·4 B) and pre-shape `off` to
`[1,1,1,P]`; the broadcast-add then yields `[1,1,K,P]` directly with NO separate unsqueeze tensor.
- task365: `offu_r`/`offu_c` were `[1,1,3,3]` int32 = 36 B each; replaced by `left_col4`/`top_col4`
  `[1,1,3,1]` = 12 B each. **Net −48 B** (the `*_col4` are int32 so 12 B not 3 B — don't misprice them).
- Zero-risk: it is a pure algebraic commute; verify with an ab byte-equal pass.

## Trick 2 — ZERO-PAD OOB PROBE: pad the gathered lane so out-of-range indices read a structural 0
When probe indices can exceed the data extent and the intent is "OOB ⇒ empty/0", the naive graph is
`Mod`(wrap the index) + `Less`(build an in-bounds mask `ok`) + `Where(ok, gathered, 0)` — THREE
tensors per probe lane. If the raw index range is **statically bounded** (here corners come from an
8×8 map ⇒ `top,left ∈ [0,7]`, probe offsets `{3,4,5}` ⇒ index ∈ `[3,12]`), instead **`Pad` the
gathered occupancy lane wide enough** (10→13 cols, constant 0) and gather with the RAW (unwrapped)
index. Index 10..12 then reads a real 0 — bit-identical to the Mod+Where result — and `Mod`, `Less`,
`Where` all vanish.
- task365: dropped `offm_r/offm_c` (Mod, 36 B), `okr/okc` (Less, 9 B), `gevr/gevc` (Where, 9 B) per
  side = −54 B/side; added `rowg_pad/colgT_pad` `[1,1,3,13]` = +39 B/side; +8 params (one shared
  `pads` init). **Net −22 B** on top of Trick 1 (total 2430→2360).
- Precondition: the index upper bound must be a hard static fact (prove it from the source of the
  index — here the 8×8 corner map dimension), and the pad width must cover `max_index+1`. Pad the
  GATHERED lane (small), never the source grid if it's shared with other consumers (e.g. `nz` also
  feeds the QLinearConv — padding it there would ADD a big tensor, not save).

## Verified ORT dtype floor facts (2026-07-12, isolated minimal-model tests)
- **`TopK` on UINT8/INT8 → `NOT_IMPLEMENTED`**; INT32 and FLOAT16 work. For a `[1,N]` flatten fed to
  TopK, **F16 is the cheapest legal dtype** (N·2 B; int32 = N·4 B). So a `Cast(u8_flat → f16)` before
  TopK is NOT waste — it is the floor. (task365 `flat16` [1,64] F16 = 128 B is forced, not reducible.)
- **`Einsum` on UINT8 → `NOT_IMPLEMENTED`**; F16/F32/INT32 work. A weighted red-count
  `Einsum('ir,abrc,ic->i', selR, code, selC)` that reuses an existing F32 `code` (a Conv output on
  F32 one-hot, which MUST be F32) is cheapest with F32 masks; an F16 route needs a +200 B `code_f16`
  cast that outweighs the mask saving (net-worse). Don't try to "narrow the Einsum to u8/f16" when the
  operand it shares is already an unavoidable F32 Conv output.

## When this applies elsewhere
Any ObjectBased/crop task that localizes objects then probes fixed offsets from a corner to read
extent (width/height run-length), or any graph with a `base+offset → Unsqueeze → GatherElements →
Where(in_bounds)` probe lane. The two tricks are independent and stack. Both are output-identical —
gate them with `ab --ref <parent> --n 2000` expecting `byte_equal=True`, not with a truth re-run.
