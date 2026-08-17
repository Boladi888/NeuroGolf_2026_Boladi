# mod_and_bitwise.md — Mod, BitShift, BitwiseAnd/Or/Xor/Not

Element-wise integer arithmetic ops that NeuroGolf rewrites often reach for arithmetic chains to emulate. These ops let you do the operation natively in 1 node.

---

## Mod **[opset 10 (integer), opset 13 added F32/F64 with `fmod=1`]**

Element-wise remainder.

```
Mod(A, B)
attributes:
  fmod: int (default 0) — 0 = integer remainder (Python style), 1 = float fmod (C style)
A, B: same dtype tensors (UINT8, INT8, INT16, INT32, INT64, F16, F32, F64)
output: same shape (broadcast), same dtype
```

### `fmod` distinction

For positive inputs, both modes give the same result.

For NEGATIVE inputs:
- `fmod=0` (Python `%`): result has same sign as `B`. `Mod(-3, 5) = 2`.
- `fmod=1` (C `fmod`): result has same sign as `A`. `Mod(-3, 5) = -2`.

**For NeuroGolf where everything is non-negative (color indices 0..9, positions 0..29), the difference doesn't matter.**

### NeuroGolf use cases

**1. Periodic indexing without CumSum chains:**

For period-3 col index from position: `pos % 3`. Stored as 30 INT64 = 240 bytes via init OR 1 Mod node + scalar `3` init = 9 bytes.

```python
positions = Range(0, 30, 1)   # [30] INT64
period3 = Mod(positions, three_scalar)   # [30] INT64 with values [0,1,2,0,1,2,...]
```

vs storing `np.arange(30) % 3` as a 240-byte init.

**Savings: ~230 bytes per period-N indexing pattern.**

**2. Wrap-around indexing:**

For `(pos + shift) % 30` (cyclic shift): one Add + one Mod, both 0 MAC, vs storing 30 different precomputed shift tables.

### Cost

- Init: scalar divisor (1 element of dtype bytes).
- 0 MACs (element-wise integer op — sometimes counted as element count by onnx-tool).
- Output tensor same size as broadcast-result of A and B.

### Gotchas

- F32/F64 requires `fmod=1` AND opset 13+.
- UINT8 Mod [NEEDS EMPIRICAL] — ORT supports per spec but verify under ORT_DISABLE_ALL.
- Division by zero in `B` — behavior unspecified, avoid.

---

## BitShift **[opset 11]**

Shifts each element of A by the corresponding amount in B (bitwise).

```
BitShift(A, B)
attributes:
  direction: "LEFT" or "RIGHT" (required, no default)
A, B: integer tensors (UINT8, UINT16, UINT32, UINT64)
output: same shape (broadcast), same dtype as A
```

### NeuroGolf use cases

**1. Bit unpacking from packed UINT8:**

```python
packed_bytes  # UINT8 tensor
for i in range(8):
    shift_i = numpy_helper.from_array(np.array([i], dtype=np.uint8), name=f"sh{i}")
    bit_i_raw = BitShift(packed_bytes, shift_i, direction="RIGHT")
    bit_i = And(Cast(bit_i_raw, INT32), one_int32)   # mask to bit 0
    # bit_i is now {0,1} per packed cell at position i
```

**2. Fast multiply/divide by powers of 2:**

`BitShift(x, 3, LEFT)` = `Mul(x, 8)` for integer x. May be faster on some kernels, but probably not relevant since both are 0-init.

### Cost

- Init: shift-amount tensor (often scalar).
- 0 MACs.
- Same memory cost as A.

### Gotchas

- **UINT-only**. INT8, INT16, INT32, INT64 are NOT supported. Cast first if needed.
- `direction` is a required attribute (no default). Forgetting it fails at load.

---

## BitwiseAnd / BitwiseOr / BitwiseXor / BitwiseNot **[opset 18]**

Element-wise bitwise ops on integer tensors. These are DIFFERENT from `And` / `Or` / `Xor` / `Not` (which are logical, BOOL-only).

```
BitwiseAnd(A, B), BitwiseOr(A, B), BitwiseXor(A, B), BitwiseNot(A)
A, B: integer tensors (UINT8/16/32/64, INT8/16/32/64)
output: same dtype as inputs
```

### NeuroGolf use cases (potential)

**1. Multi-mask combination in UINT8 space:**

If you have 8 BOOL masks, pack them as bits in a single UINT8 tensor (`np.packbits`), then use BitwiseAnd/Or to combine multiple-mask conditions without per-mask And/Or operations.

```python
# Pack 8 masks into one UINT8 per cell
combined = BitwiseAnd(packed_a, packed_b)   # all 8 masks AND'd in one op
```

vs 8 separate `And(bool_a_i, bool_b_i)` ops.

**Win**: -7 nodes, -7 intermediates per 8-mask combination.

**2. Color permutation encoding:**

Store color permutations as bit patterns; bitwise ops compose permutations.

### Cost & feasibility

- 0 init for the op itself.
- 0 MACs.
- [NEEDS EMPIRICAL] **Opset 18 may exceed NeuroGolf scorer's supported opset range.** Bump cautiously and verify load.
- ORT support for these specific ops: should be present in ORT 1.14+, but `ORT_DISABLE_ALL` might block them.

### Important distinction

**`And`, `Or`, `Xor`, `Not`** (no "Bitwise" prefix) — **BOOL ONLY**, logical operation. Opset 7. Always available.

**`BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`** — INTEGER ONLY, bit-by-bit. Opset 18. Recent, may not be available.

Sonnet often confuses these. Mnemonic: **bool → bareword (`And`); int → Bitwise prefix.**

---

## Combinable patterns

### Periodicity check via Mod + Equal

"Is position divisible by 3?"

```python
positions = Range(0, 30, 1)
mod3 = Mod(positions, three_scalar)
is_period_3 = Equal(mod3, zero_int)   # [30] BOOL with True at positions 0, 3, 6, ...
```

Cost: 4 nodes + 3 scalar inits (~32 bytes total) for a 30-element periodicity mask. vs storing the explicit mask: 30 BOOL bytes. **Init savings only emerge for multiple mask outputs from the same Mod intermediate.**

### Combined: extract bit i from packed mask, broadcast over channels

```python
# packed: UINT8 [1, 1, 30, 30] — 8 binary masks packed
# Extract bit 3 broadcast across all 10 channels
shift_3 = ConstantOfShape([1], value=Tensor(np.array([3], dtype=np.uint8)))
shifted = BitShift(packed, shift_3, direction="RIGHT")
bit_3 = And(Cast(shifted, BOOL), Cast(one_uint8, BOOL))   # or via bitwise mask
# Output [1, 1, 30, 30] BOOL — broadcasts to [1, 10, 30, 30] in downstream Where/And
```

---

## When NOT to use these

- For **single-mask** operations, the logical `And`/`Or`/`Xor`/`Not` ops are simpler and don't need opset bumps.
- For **non-periodic** integer arithmetic, plain `Sub`/`Add`/`Mul`/`Div` are fine.
- BitwiseOps at opset 18 may not work in NeuroGolf's scorer environment — verify before committing to a rewrite.

---

## Cross-references

- `Algorithm_Library/algorithm_patterns.md` — initializer / bit-packing construction tricks (former `Numpy_Advanced/`, removed 2026-05-23).
- `Onnx_basics_worth_reviewing.md` Multi-input ops — for `And`/`Or` chain collapsing.
- `exemplars/02_bool_first/` — bool-first patterns where these could plug in.

## TODO — verify empirically

1. ORT support for opset 18 BitwiseAnd/Or/Xor/Not under ORT_DISABLE_ALL.
2. Whether NeuroGolf validator rejects opset 18 (max-tested opset in this folder is opset 14 per `Ort_Advanced/ort_opset_support.md`).
3. Mod with UINT8 dtype.
4. BitShift edge cases: shift amount >= bit width is unspecified per spec — verify ORT behavior.
