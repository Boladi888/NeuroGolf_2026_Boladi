# einsum.md — multi-axis tensor contraction in one node

`Einsum` (ONNX opset 12+) replaces chains of `Transpose + MatMul + ReduceSum` (and variations) with a single op driven by Einstein notation.

For NeuroGolf, this can collapse 3–6 node sequences into 1, removing intermediate tensor memory.

---

## Signature

```
Einsum(input1, input2, ...)
attributes:
  equation: string (the Einstein notation specifier)
```

Number of inputs = number of operand labels in the equation (before `->`).

### Equation syntax

```
"ij,jk->ik"           # standard matrix multiply A @ B
"bij,bjk->bik"        # batched matmul
"ij->j"               # column sum (reduce axis 0)
"ij->ji"              # transpose
"ij->"                # full sum
"i,i->"               # dot product
"i,j->ij"             # outer product
"...ij,...jk->...ik"  # batched matmul with arbitrary leading dims (ellipsis)
```

Rules:
- Letters before `->` are operand axes (one string per input, comma-separated).
- Letters after `->` are output axes (omit `->` and ONNX infers).
- A letter appearing on the LHS but not the RHS is **summed over** (reduced).
- A letter appearing on multiple inputs and the RHS is **broadcast** across both.
- `...` (ellipsis) matches any number of leading axes — useful for variable-rank inputs.

---

## When it pays off in NeuroGolf

### Pattern 1: Per-channel weighted sum across spatial axes

You want `out[b, c] = sum_{h,w} input[b, c, h, w] * weight[h, w]`.

**Naive ONNX**:
```
Mul(input [1,10,30,30], weight [1,1,30,30]) → tmp [1,10,30,30]   (intermediate: 36k bytes F32)
ReduceSum(tmp, axes=[2,3], keepdims=0) → out [1, 10]
```

**Einsum**:
```
Einsum(input, weight, equation="bchw,hw->bc")
# 1 node, no intermediate
```

Savings: -1 node, -1 intermediate (36k bytes), same MACs.

### Pattern 2: Channel-wise dot products

You want pairwise dot products between channels: `out[c1, c2] = sum_{h,w} A[c1,h,w] * B[c2,h,w]`.

```
Einsum(A, B, equation="chw,dhw->cd")
```

No need to materialize the [c1, c2, h, w] outer product first.

### Pattern 3: Hash-decode pipelines

If you currently do `Reshape → MatMul → Reshape` for hash decoding, often this collapses to a single Einsum with appropriate axes. Spec out the pattern carefully — the Einsum form preserves dimensionality cleanly.

### Pattern 4: Batched MatMul with transpose

```
Transpose(B, perm=[0,2,1]) → BT
MatMul(A, BT) → out
```

becomes:

```
Einsum(A, B, equation="bij,bkj->bik")   # note the swapped indices on B
```

Drops the Transpose op AND its intermediate.

---

## Cost analysis

- **Init cost**: 0 (the equation is a string attribute, not a tensor).
- **MAC cost** [ASSUMED]: same as the equivalent MatMul/Reduce chain — Einsum decomposes internally.
- **Memory cost**: only the output tensor. No intermediates from chained MatMul+Reduce.

**Net savings come from eliminated intermediates, not from MAC reduction.** If the equivalent chain has 1 large intermediate, Einsum saves that intermediate's bytes.

---

## ORT support gotchas

[NEEDS EMPIRICAL] Verify each Einsum equation you use under `ORT_DISABLE_ALL`. Some quirks:

1. **Diagonal extraction (`"ii->i"`)**: takes the diagonal of a square matrix. ORT support unclear.
2. **Trace (`"ii->"`)**: full trace. May or may not be supported.
3. **Repeated indices on one operand**: technically allowed by NumPy, ORT support varies.
4. **Ellipsis with mixed-rank operands**: spec is ambiguous in places; test before using.
5. **F16 Einsum**: ORT MatMul supports F16; Einsum should inherit, but verify.

### Safe equation patterns to start with

These are common enough that ORT almost certainly handles them:

```
"bij,bjk->bik"       # batched matmul
"ij,jk->ik"          # plain matmul
"bchw,hw->bc"        # spatial reduction
"bchw,chw->b"        # full per-batch dot
"bchw->bc"           # ReduceSum over h,w
"bchw->b"            # ReduceSum over c,h,w
"ij->ji"             # transpose (probably faster than Transpose op for small mats)
"i,j->ij"            # outer product
```

### Equations to avoid until verified

- Anything with repeated indices on one operand: `"ii->i"`, `"iji->j"`, etc.
- 4+ operands in one Einsum (rare ORT-wise).
- Ellipsis chains: `"...ij,...jk->...ik"` (works but verify on your exact rank).

---

## Construction in Python

```python
from onnx import helper

einsum_node = helper.make_node(
    "Einsum",
    inputs=["A", "B"],
    outputs=["C"],
    equation="bij,bjk->bik",
    name="batched_matmul",
)
```

**The `equation` attribute is a STRING** — pass as keyword arg to `make_node`, not as part of `helper.make_attribute(...)` boilerplate.

---

## Comparison: Einsum vs explicit MatMul chain

For a single matmul `[N, K] @ [K, M] → [N, M]`:

| Approach | Nodes | Intermediates | MACs |
|---|---:|---:|---:|
| Plain `MatMul` | 1 | 0 | N·K·M |
| `Einsum("ij,jk->ik")` | 1 | 0 | N·K·M |

No win. **Einsum wins ONLY when it replaces multiple nodes.**

For a chained `Mul + ReduceSum`:

| Approach | Nodes | Intermediates | MACs |
|---|---:|---:|---:|
| `Mul → ReduceSum` | 2 | 1 (full-size) | full Mul + sum |
| `Einsum("bchw,hw->bc")` | 1 | 0 | full Mul + sum |

**Win: -1 node, -1 intermediate.** For NeuroGolf-shape intermediates (36k bytes), this matters.

---

## Diagnostic: when to suspect Einsum applies

Look for these patterns in your graph:
1. `Transpose → MatMul` (Einsum can absorb the Transpose into the index ordering).
2. `Mul (broadcast) → ReduceSum` along the broadcast axes.
3. `MatMul → ReduceSum` along an axis.
4. `Tile → Mul → ReduceSum` (the Tile is implicit broadcast).
5. Multiple `MatMul`s in a row with reshape between them.

If you see any of these, draft the Einsum equation and verify-load before committing.

---

## Empirically verified (2026-05-30, task191, ORT 1.24.x under ORT_DISABLE_ALL)

1. **ORT supports `Einsum` at opset 13** including the 3-operand chain `"ij,jk,kl->il"` with F16 operands. Loads and runs.
2. **⚠️ F16 multi-operand Einsum is NOT byte-equal to the chained MatMul it replaces.**
   `Einsum("ij,jk,kl->il", A, B, C)` ≠ `MatMul(MatMul(A,B), C)` in F16:
   the chained form rounds the intermediate `A@B` to F16 *before* the second matmul,
   whereas Einsum contracts internally with different associativity/accumulation.
   Measured: `np.array_equal == False`, max abs diff ≈ 0.042 on random [5,30]·[30,30]·[30,5].
   **Consequence for Path A (byte-equal) work**: do NOT use Einsum to "collapse" an existing
   F16 `MatMul→MatMul` chain — it silently changes values and can flip downstream
   `ArgMax`/threshold decisions on hidden inputs. Einsum collapse is a Path-B
   (rule-rebuild) tool, not a byte-equal graph-surgery tool. The task264 Einsum wins
   (`compress_v5`) were structural compressions validated by `verify_against_live`, NOT
   byte-equal substitutions — and task264 has a Kaggle-public-negative history, so that
   precedent does not license byte-equal Einsum collapse elsewhere.
3. A *single* 2-operand `Einsum("ij,jk->ik")` replacing one `MatMul` IS effectively a matmul
   (no intermediate to re-round) — but it's also a no-op win (einsum only pays when it removes
   nodes/intermediates, and a lone matmul has none). Net: Einsum only helps cost when it removes
   a materialized intermediate, and doing so in F16 forfeits byte-equality.

## TODO — verify empirically

1. ~~Does ORT support `Einsum` at opset 12 under `ORT_DISABLE_ALL`?~~ (opset 13 verified above)
2. Which equations work and which don't (diagonal, trace, repeated indices)?
3. What's the actual `onnx-tool` MAC count for Einsum nodes?
4. F16 Einsum compatibility with hash-decode pipelines (note the non-byte-equal finding above:
   if a hash decode depends on exact F16 values, Einsum substitution will break it).
