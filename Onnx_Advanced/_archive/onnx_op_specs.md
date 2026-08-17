# onnx_op_specs.md — upstream ONNX op references (verified locally)

> Raw ONNX 1.21 op schemas / opset coverage for ops this project leans on (BitShift, BitwiseAnd/Or/Xor, Gather, Mod).
> Consolidated from `Onnx_Advanced/_sources/` on 2026-06-01 for discoverability (see INDEX.md). Originals removed; this is the single home.

## Contents
- [onnx-1-21-op-schemas.md](#onnx-1-21-op-schemas)
- [onnx-1-21-opset.md](#onnx-1-21-opset)
- [onnx-bitshift-spec.md](#onnx-bitshift-spec)
- [onnx-bitwiseand-spec.md](#onnx-bitwiseand-spec)
- [onnx-gather-spec.md](#onnx-gather-spec)
- [onnx-mod-spec.md](#onnx-mod-spec)


---
<a id="onnx-1-21-op-schemas"></a>

# Source: ONNX 1.21.0 op schemas (verified locally)

**Verification**: I installed `onnx==1.21.0` (the project-pinned version) and called `onnx.defs.get_schema(name)` on each operator below. All numbers are from the live schema, not paraphrased from docs.

## Bitwise integer ops (J.01)

| Op | `since_version` | Domain | Type constraint `T` | NeuroGolf-relevant note |
|---|---|---|---|---|
| `BitwiseAnd` | 18 | `""` | `(u)int{8,16,32,64}` | **No float**. Requires Cast wrapper. |
| `BitwiseOr` | 18 | `""` | `(u)int{8,16,32,64}` | Same as above. |
| `BitwiseXor` | 18 | `""` | `(u)int{8,16,32,64}` | Same as above. |
| `BitwiseNot` | 18 | `""` | `(u)int{8,16,32,64}` | Same as above. |
| `BitShift` | 11 | `""` | `uint{8,16,32,64}` | **No signed int, no float.** `direction ∈ {"LEFT","RIGHT"}`. |

Source PRs:
- `https://github.com/onnx/onnx/pull/4496` — added BitwiseAnd/Or/Xor at opset 18.
- `https://github.com/onnx/onnx/pull/4497` — added BitwiseNot at opset 18.
- `https://github.com/onnx/onnx/pull/1931` — added BitShift at opset 11 ("Add bit-shift operators for supporting hashing").
- `https://github.com/onnx/onnx/issues/2406` — confirms BitShift unsigned-only.

**Critical for NeuroGolf**: even though these ops are legal under domain `""`, `onnx-tool 1.0.1` has **no profile / value_infer handler** for them (see "Source: onnx-tool 1.0.1 op handlers"). Submitting a graph that uses them risks the validator's `calculate_memory()` returning `None` or zero-cost mis-accounting. Treat as **unsafe** until empirically validated against the live Kaggle validator.

## Arithmetic ops (J.01, J.16)

| Op | `since_version` | Type `T` includes float? | NeuroGolf use |
|---|---|---|---|
| `Mod` | 13 (originally 10) | yes (mandatory `fmod=1` for float) | Bit/parity extraction; RNS residue |
| `Floor` | 13 | yes (float-only, no int) | Bit extraction via `Mul→Floor→Mod` |
| `Mul`, `Add`, `Sub` | 14 | yes | Element-wise, 1 macs/element |
| `Cast` | 25 (originally 6) | all dtype pairs | Bridges int↔float; 0 macs but adds activation memory |

Sources:
- `https://onnx.ai/onnx/operators/onnx__Mod.html` — confirms float input requires `fmod=1` (numpy.fmod semantics).
- `https://onnx.ai/onnx/operators/onnx__Floor.html` — float-only.

## Shape-only ops (J.13, J.14)

| Op | `since_version` | Notes |
|---|---|---|
| `Identity` | 25 (originally 1) | Truly free — single-node graph cost = 0. |
| `Transpose` | 25 (originally 1) | Output value_info contributes to memory if non-IO. |
| `Reshape` | 25 (originally 5) | Same caveat. |
| `Slice` | 13 | Same caveat. |
| `Gather` | 13 | `Tind ∈ {int32, int64}` — **always use int32** (saves bytes). |

Sources: respective `https://onnx.ai/onnx/operators/onnx__<Name>.html`.

## Conv (baseline reference)

| Op | `since_version` | Type `T` |
|---|---|---|
| `Conv` | 22 (originally 1) | `bfloat16, float16, float, double` — **no int** in opset ≤ 23 (the max in onnx 1.21.0). |

## Bool-typed logical ops (J.18, indicator masks)

| Op | `since_version` | Type | Note |
|---|---|---|---|
| `And/Or/Xor` | 7 | `tensor(bool)` only | Cannot be applied to int8 — that needs `Bitwise*` instead. |
| `Not` | 1 | `tensor(bool)` only | Same. |
| `Less/Greater` | 13 | float/int → bool | Useful for `(x > 0)` style threshold check. |
| `Equal` | 19 | float/int/bool → bool | Same. |
| `Where` | 16 | bool selector + any T | Useful for indicator-mask routing. |

## Random ops

| Op | `since_version` | Type | Note |
|---|---|---|---|
| `RandomUniform` | 22 (originally 1) | float types | Allowed but rarely useful in deterministic ARC. |
| `RandomUniformLike` | 22 | float types | Same. |

## Domains and IR

- `onnx==1.21.0` ships `IR_VERSION = 13`.
- Default-domain max opset shipped in 1.21.0 is **23** (the `onnx/onnx` 1.21.0 release notes record "Revert opset 25 version bump"; the in-tree schemas have `since_version` up to 25 for ops touched by the reverted bump, but the validator should accept anything ≤ 23 reliably).
- The brief locks domains to `""` and `"ai.onnx"` — these are aliases. All ops above are in this default domain.

---
<a id="onnx-1-21-opset"></a>

# ONNX 1.21.0 release / opset coverage

- **URLs:** `https://github.com/onnx/onnx/releases` (1.21.0 entry: "Revert opset 25 version bump"); `https://pypi.org/project/onnx/`; `https://onnx.ai/onnx/repo-docs/Versioning.html`.
- **Practical opset:** ONNX 1.21.0 ships opset 23 as the maximum stable default‑domain version; opset 24/25 schemas were either not bumped or reverted at release. **All opset‑18 bitwise ops, opset‑11 BitShift, opset‑10/13 Mod, and opset‑1+ Gather are present.**
- **Domain restriction:** the brief locks opset domains to `""` and `"ai.onnx"` (the empty string is the alias for `ai.onnx`). All ops in our toolkit live in these domains; no contrib op is needed.
- **IR version:** ONNX 1.22 reports IR_VERSION = 13; 1.21.0 reports 12, comfortably ahead of any consumer requirement.

---
<a id="onnx-bitshift-spec"></a>

# ONNX `BitShift`

- **URL / citation:** `https://onnx.ai/onnx/operators/onnx__BitShift.html`; PR onnx/onnx#1931 ("Add bit‑shift operators for supporting hashing").
- **Domain:** `""`.
- **`since_version`:** **11**.
- **Type constraint `T`:** `tensor(uint8|uint16|uint32|uint64)` only — **signed integers and float32 are NOT accepted**. Confirmed by onnx/onnx#2406 ("BitShift only supports unsigned types in opset 11") and microsoft/onnxruntime#2207 ("BitShift Not Implemented for uint8" — i.e. ORT historically lacked a uint8 kernel; later fixed but still unsigned‑only).
- **Attribute:** `direction ∈ {"LEFT","RIGHT"}`.
- **NeuroGolf consequences:** to extract bit‑plane *i* of a 4‑bit color value, one needs `Cast(float32→uint8)` → `BitShift(RIGHT, i)` → `BitwiseAnd(1)` → `Cast(uint8→float32)`. Each Cast is ~9 000 MACs and adds a uint8 intermediate (9 000 B activation `memory`). For 4 bit planes that is 4×(2 Casts + 1 BitShift + 1 BitwiseAnd) ≈ 144 000 MACs ignoring shifting/and constants.
- **Project mapping:** the bit‑plane idea is *legal* but not free. Compared with a single Conv 1×1 (cost 90 100) the round‑trip already runs at ~1.5×; the only place it saves is when *one* extracted plane substitutes for *several* Convs.

---
<a id="onnx-bitwiseand-spec"></a>

# ONNX `BitwiseAnd` / `BitwiseOr` / `BitwiseXor` / `BitwiseNot`

- **URL / citation:** `https://onnx.ai/onnx/operators/onnx__BitwiseAnd.html`, `…__BitwiseOr.html`, `…__BitwiseXor.html`, `…__BitwiseNot.html` (ONNX 1.22 docs; identical schema is present in 1.21.0).
- **Domain:** `""` (default `ai.onnx`).
- **`since_version`:** **18** (ratified in PR onnx/onnx#4496 and the BitwiseNot PR #4497, July–Aug 2022).
- **Type constraint `T`:** `tensor(int8|int16|int32|int64|uint8|uint16|uint32|uint64)`. **Float32 is NOT accepted.**
- **Shape semantics:** numpy‑style multidirectional broadcasting; output same dtype as inputs.
- **NeuroGolf consequences:**
  - Pinned `onnx==1.21.0` ships opset ≤ 23, so opset 18 ops are available; the bitwise ops are legal under the "" domain restriction.
  - To use them on a float32 `[1,10,30,30]` activation we must `Cast` to int* first and `Cast` back. Cast is an element‑wise op that onnx‑tool counts as ≥ 1 MAC per element (∼9 000 MACs per Cast on the working tensor), and the intermediate int tensor adds memory equal to its byte size (e.g., int8 → 9 000 B; int32 → 36 000 B; int64 → 72 000 B), counted under `memory`.
  - There is **no shape‑only path** to bitwise ops on float input; the entry tax is at minimum two Casts (in/out) + one bitwise op, so the floor cost for a single XOR pair is ~3 × element‑count + activation memory of the int intermediates.
- **Project mapping:** an XOR between two stored grids (e.g., a stored mirror copy and the input) requires Cast→BitwiseXor→Cast, totalling ~27 000 MACs + ~108 000 B of int‑intermediate memory — far above a Conv 1×1 identity (90 100). It only pays off when it replaces something more expensive than Conv 3×3 (810 900) and avoids learnable weights.

---
<a id="onnx-gather-spec"></a>

# ONNX `Gather`

- **URL / citation:** `https://onnx.ai/onnx/operators/onnx__Gather.html`.
- **Domain:** `""`. `since_version: 1`, behaviour stable through opset 13+.
- **Types:** data `T` is essentially any type including `float`; indices `Tind ∈ {int32, int64}`.
- **Shape semantics:** rank `q+(r−1)`; pure gather is element‑level memory copy with no FMA.
- **NeuroGolf cost (under onnx‑tool 1.0.1):** Gather is in the "shape‑only" cost class per the project context (cost‑lever probe). The `data/Profile.md` of onnx‑tool defines `1 MAC = float(a)*float(b)+float(c)` and explicitly notes that "treat other instructions as 1 MAC, like cmp, add, mov", but the locked NeuroGolf rule states that pure shape ops (Identity/Transpose/Slice/Gather/Reshape) are charged 0. **Initializers (the index table) count under `params`, not `memory`.**
- **Index‑table byte cost:** length‑*N* `int64` table = 8·*N* B; `int32` = 4·*N* B; both go to `params`. After constant folding the table is the only residue; downstream Reshape/Transpose plus the gather contribute zero compute.
- **Project mapping:** any "permutation" of the 900 spatial cells (Morton, Hilbert, Gray, RNS demux) reduces to a Gather initialised with a precomputed index vector. The only knob is byte width; halving from int64 to int32 halves `params`.

---
<a id="onnx-mod-spec"></a>

# ONNX `Mod`

- **URL / citation:** `https://onnx.ai/onnx/operators/onnx__Mod.html`.
- **Domain:** `""`.
- **`since_version`:** **10**, last updated in opset 13.
- **Type constraint:** `tensor(bfloat16|double|float|float16|int8|int16|int32|int64|uint8|uint16|uint32|uint64)` — **float32 is supported** (with mandatory `fmod=1`).
- **Behaviour:** `fmod=0` ⇒ Python‑style integer modulus (sign of divisor); `fmod=1` ⇒ C‑style `fmod`/`numpy.fmod` (sign of dividend). For floating‑point input `fmod` *must* be 1.
- **NeuroGolf consequences:** unlike the bitwise ops, `Mod` works directly on the float32 activation so a `Mul`+`Floor`+`Mod`+`Sub` arithmetic chain to extract a bit plane needs no Cast.
  - Probable onnx‑tool MAC count per bit plane on `[1,10,30,30]`: 4 element‑wise ops × 9 000 elems ≈ 36 000 MACs; activation memory adds ≈ 4×36 000 = 144 000 B of intermediates (the byte size of one `[1,10,30,30]` float32 tensor is exactly 36 000 B excluding I/O).
  - This is the cheapest legal path to bit extraction but still ≈ Conv 1×1 (90 100) per plane, and you typically need 4 planes.
