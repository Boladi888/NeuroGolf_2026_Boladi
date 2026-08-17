# task160 — FORCED-FLOOR WALL at ~1049–1089 (the ~500 floor_est is unreachable)

- **live bar (refreshed 2026-07-11):** manifest **1094**, sha `10c3a7c017954791986b3b32b85e7b6ae908b2b2eb44e3130383e9d9cc179c50`, owner **Sho** (2026-07-09). This is a *2D INT8 QLinearConv* graph (`task160_fused_detpass_qtail_v04`: Slice→Cast→2×QLinearConv→Pad; memory 1000 + params 94), a legitimate −5 B improvement over the prior **1099** packed-row (Boladi, sha `f1dcb5ae…`, `confirmed_public`/accepted, full-truth pass).
- **local best (already staged, byte-equal to officials AND to the confirmed 1099):** `task160_objectconvnarrow4drows_cost1089` — cost **1089** (memory 726 + params 363), full truth 200000/200000, full ab byte_equal=True. **This BEATS the live 1094 by 5 B and was never promoted to a captain staging area** (no `Staged_Boladi/task160.json`). It is a byte-equal −10 B structural shave (drop `hit8` Unsqueeze + axes init) of the confirmed_public 1099 → Path-A LOCAL_GOLD, ship-ready.
- **verdict (2026-07-06, re-confirmed 2026-07-11 bo_ng_cld_worker3):** the prompt's `floor_est ~500` is a heuristic artifact. The forced components of the only viable representation sum to **> 500 before any detection logic**. No candidate strictly below ~1049 is achievable without switching to a strictly-more-expensive representation.

## 2026-07-11 refresh — false-floor DROP + fast-probe priced wall
- **The false-floor's charged target is ALREADY GONE.** The Direction attacks a "FLOAT 1×10×10×10 Conv/Concat prepad canvas" / 4000-byte prepad tensor. Neither live graph contains it: the packed-row 1089/1099 dissolved it to `[1,1,30,1]` row-codes; Sho's 1094 dissolved it to INT8 `[1,·,10,10]` planes. Per the 2026-07-10 workflow rule (charged component gone ⇒ DROP), the false-floor attack is **DROP**.
- **Fast probe (built + priced, `prototypes/freeout_tail_probe.py`):** the two minimal free-output Equal/Where decode TAILS the Direction proposes:
  - `tail_a` (one-channel index → Equal-as-output @30×30): **static=profile=13513** (the `[1,10,30,30]` bool intermediate is 9000 B; the F32 `[1,1,30,30]` slice is 3600 B). Dead — worse than the OLD 13991.
  - `tail_b` (`[1,1,10,10]` index → Equal → `[1,3,10,10]` one-hot → Pad): **static=profile=1120** — already **above the current 1089 with NO transform logic yet** (F32 `[1,1,10,10]` slice 400 + bool `eq` 300 + INT8 `onehot3` 300 = 1100). Add the hollow-square detect/render and it blows past 1089.
  - Kill rule met (prices > 500, indeed > 1089) ⇒ stop, dead formulation. Empirical corroboration: Sho's strong from-scratch 2D route priced 1094 > 1089.

## Cost model (verified from `neurogolf_utils.calculate_memory` / `calculate_params`)
- **params** = Σ `prod(dims)` over initializers — **element count, dtype-FREE**. (Sparse initializers count only nonzero `values.dims`, but sparse init is ORT-dead here — see `sparse_class_mask.md`.)
- **memory** = Σ over intermediate tensors of `elements × itemsize`, max(static, runtime); **`input` and `output` named tensors are exempt (free)**.

## Why the ~500 floor is below the FORCED components
The rule packs each grid row into a 30-bit code (3 colors × 10 cols) and decodes to the
exempt `[1,10,30,30]` output via `BitwiseAnd(rowcode30[1,1,30,1], class_col_bits[1,10,1,30])`.

1. **Decode operand = 300 forced.** The exempt output must be produced by a final broadcast
   op. To yield `[1,10,30,30]` from `rowcode[1,1,30,1]`, the *other* operand must span
   channel=10 (broadcast rule: rowcode has 1 → operand needs 10) and width=30 → **`[_,10,1,30]`
   = 300 elements minimum**. As an **initializer that is 300 params (the cheapest form, dtype-free)**.
   As *any intermediate* it is ≥300 B (uint8) … 1200 B (uint32). So 300 params is the hard floor
   for the decoder, and it is *already* what the 1089 graph uses.
2. **`rowcode30`[1,1,30,1] uint32 = 120 B forced** — 30-bit codes can't be narrower than uint32;
   30 rows are needed to produce the 30-row output. Charged (it is not the `output` tensor).
3. **`rows_f30`[1,1,30,1] F32 = 120 B forced** — the row-pack Einsum has `input` (fixed **F32**) as
   an operand, so its output is F32; 30 input rows → 30-row output. (F16 lever DEAD: casting
   `input` to F16 would materialize a `[1,10,30,30]` F16 tensor = 4500 B.)

**(1)+(2)+(3) = 300 + 120 + 120 = 540 > 500**, before ANY detection or assembly. Floor is above 500.

## Assembly is also at floor (240 B forced)
`rowcode10 = out1·1023 + out2·1048575 + 1023` (place color-1 bits at 10–19, color-2 at 20–29,
color-0 = complement). Needs `out1_u32, out2_u32, delta1, delta2, <1 add-intermediate>, rowcode10`
= **6 × uint32[1,1,10,1] = 240 B**.
- **Sum-fusion DEAD:** `onnx.checker` rejects `Sum` on `tensor(uint32)` (float-only op). The two
  binary `Add`s (any ordering) always leave exactly one intermediate. Builder kept as documented
  dead route: `prototypes/sum_fuse_builder.py`.
- uint32 is forced: the `<<10`/`<<20` placements reach 2^20…2^30, so the Mul inputs/outputs
  cannot be uint16.

## Forced-minimum tally for this family
params 363 (selector 300 + col_pow 30 + chan_sel 10 + 23 scalars/slices — all forced) +
memory { rows_f30 120 + rowcode30 120 + assembly 240 + rows_f 40 + rows 20 } = 560
→ **≈ 923 before detection**; the detection/render chain (hollow-square find + 2 ConvTranspose
renders) adds ~86 B of ≤40 B tensors → **1089 sits ~40–80 B above the true floor (~1009–1049).**

## Independent formulations evaluated (all at-floor or worse)
| # | formulation | verdict |
|---|---|---|
| 1 | packed-row + dense [1,10,1,30] AND (current, 1089) | AT FLOOR |
| 2 | sparse `class_col_bits` initializer (30 nz) | DEAD — ORT rejects sparse init (`sparse_class_mask.md`) |
| 3 | Sum-fuse the uint32 assembly | DEAD — `Sum` unsupported on uint32 |
| 4 | F16 row-pack Einsum | DEAD — F32 `input` forces F32 output |
| 5 | separable chan⊗col selector `[1,10,1,1]⊗[1,1,1,30]` | worse — materializes 300-elem uint32 intermediate (1200 B) > 270-param save |
| 6 | per-channel unpack → column-AND (selector→30 params) | worse — `rowcode_perchan[1,10,30,1]` intermediate 600–1200 B |
| 7 | comparison/Equal decode (colorgrid vs arange) | worse — 2D color plane `[1,1,30,30]` = 900 B |
| 8 | 2D Conv/morphology on planes | worse — multiple `[1,·,30,30]` planes, est. > 2 KB (scratch plan) |
| 9 | u8 BitShift detection shortcut (drop `rows_r1`) | DEAD — u8 truncation loses bit-8, breaks `adj8[7]` (col-8 adjacency) |
| 10 | **BitShift-unpack decode (BUILT + GATED 2026-07-11)** — `BitShift(rowcode30,[0,10,20,31×7])→shifted[1,10,30,1]`, `AND col_bitmask[1,1,1,30]`→output | **worse, MEASURED 2029** — visible 265/265 + truth 2000/2000 (byte-equal correct), but the `shifted[1,10,30,1]` uint32 = 1200 B charged intermediate replaces the 300-param init: params 363→103 (−260) but memory 726→1926 (+1200) = **+940**. `prototypes/decode_variants.py`. This is the concrete gated proof of formulations #5/#6. |

## One remaining hypothesis — now CLOSED by build (2026-07-11)
A decode that produces the exempt output from an operand < 300 elements would need the (channel,
column)→bit-position map to be expressible without a per-(c,w) tensor. Bit position `w+10c` is
separable (`2^w · 1024^c`) but AND does not distribute over the product; the separable/computed
reconstruction was **built and gated** (formulation #10) and priced **2029** — every reconstruction
materializes a ≥300-element uint32 intermediate (1200 B) that costs far more than the 300-param
initializer it replaces. Info-theoretic floor: the decode requires the full 10×30 = **300-element
(channel×width) map**; dense storage is forced (sparse init ORT-dead, #2). Absent a genuinely new ORT
op that fuses channel-shift + column-mask into the free output in ONE step (searched
INDEX→techniques #67/#78/#81/#109/#116/#119 + op_cost_patterns; none exists), **300 params is a true
forced floor and this task is a WALL at ~1089.**
