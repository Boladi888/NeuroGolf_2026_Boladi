# row_signature_rank_reduction.md

This note treats **row-signature / rank-reduction** as a reusable family for ARC-style grids \(X \in \{0,\dots,9\}^{N\times M}\), with \(1 \le N,M \le 30\). The practical ONNX-static recipe is: compute a compact per-row signature, reduce or reorder rows with `TopK` or exact pairwise comparisons, then assemble output rows with `Gather` or a dense equality/permutation mask. ARC’s official format is a grid of integers 0–9 with grid size capped at 30×30, and ARC solution practice already includes DSL primitives for sorting, deduplication, horizontal/vertical splitting, and row/column compression even though “row-signature” is not a formal benchmark taxonomy.

## Inventory of row-signature patterns

The table below is a **design taxonomy**, not an official ARC taxonomy. The names are synthesized from common array-programming patterns plus ARC/DSL practice. The important engineering point is that these signatures all collapse a row from shape `[M]` to either a small scalar or a short fixed-width vector, making static ONNX reduction feasible.

| Signature type | Output dtype | ONNX ops needed | Memory cost | Use case |
|---|---|---|---|---|
| Row-occupancy | `int32`/`int64` scalar per row | `Equal`/`Not`, `Cast`, `ReduceSum` | `[N]` | Keep rows with at least *t* active cells; dense-row crop |
| Row-color-set | `int32[N,C]` or packed bitmask `[N]` | `OneHot` or `Equal`, `Cast`, `ReduceMax`/`ReduceSum`, optional `MatMul` | `[N,C]` or `[N]` | Find rows containing color `c`; group rows by palette membership |
| Row-pattern | original row `[N,M]` or exact multi-limb key `[N,L]` | `MatMul` with positional weights, or raw row compare via `Equal` | `[N,M]` or `[N,L]` | Deduplicate identical rows; full sequence match |
| Row-rank | `int64` scalar per row or permutation index `[N]` | pairwise compare + `ReduceSum`, or `TopK` | `[N]` or `[N,N]` | Sort rows by a signature; rank-based selection |
| Row-diversity | scalar | row-key compute + pairwise equality + `ReduceSum` | `[1]` plus intermediates | Count distinct rows; trigger output templates based on diversity |

The canonical formulations are straightforward. For background color \(b\), **occupancy** is \(o_i=\sum_j [X_{ij}\neq b]\). **Color-set** is \(s_{ic}=\max_j [X_{ij}=c]\) or equivalently \([\sum_j [X_{ij}=c] > 0]\). **Pattern** is either the row itself or an exact positional code \(p_i=P(X_{i,:})\). **Rank** for scalar key \(k\) is \(r_i=\sum_j [k_j<k_i]+\sum_{j<i}[k_j=k_i]\), which is the stable order statistic. **Diversity** is \(\sum_i \text{keep}_i\), where `keep_i` is true only if no earlier row equals row `i`. These are the row analogs of the larger ARC habit of composing grid transformations from compact symbolic predicates.

Computationally, the family splits cleanly into three shapes. Occupancy-like signatures produce `[N]`; color-set signatures produce `[N,C]` then optionally repack to `[N]`; exact row-pattern signatures are `[N,M]` if kept verbatim or `[N,L]` if packed into fixed-width limbs. On ARC-sized grids, `N,M ≤ 30` and `C=10`, so even the “large” one-hot color cube is only `30×30×10 = 9000` elements before reduction.

A useful symmetry: everything here applies to **columns** after a transpose. In practice, row-signature and column-signature rules are usually the same graph pattern with the reduction axis swapped. That is a synthesis rather than a published ARC convention, but it follows directly from the grid formalism.

## ONNX-static signature computation

For opset-18 graphs, the core spec pages that matter here are `ReduceSum`, `MatMul-13`, `TopK-11`, and `Gather-11`; the same operator docs also show `OneHot`, `ArgMax`, `Trilu`, `CumSum`, `Equal`, `Cast`, and `ReduceMax`, which are enough to build the full family without loops. ONNX Runtime’s compatibility docs say prebuilt packages support all ONNX opsets/operators and that all ORT versions support opset 7+, so opset 18 is comfortably inside the supported range.

The cheapest signature is **occupancy**. Build a non-background mask with `Equal(X, bg)` then invert/cast and `ReduceSum(axis=columns)`. On ARC grids, the count is at most 30, so `int32`, `int64`, `float32`, and `float64` all represent it exactly; there is no reason to spend bits on larger types unless later nodes require integer indexing. This is the best default when the rule is “keep rows with at least *t* active cells” or “pick the densest row(s).”

For **row-color-set**, the cleanest static pattern is `OneHot(X, depth=C)` to shape `[N,M,C]`, then reduce over columns with `ReduceMax` for presence or `ReduceSum` for per-row color counts. If a scalar key is useful, multiply the `[N,C]` presence vector by constant bit weights, e.g. `[1,2,4,...]`, using `MatMul` to produce a packed color bitmask. This is exact for `C=10`, so a single 10-bit mask fits trivially in `int32`. When you only need “does color c appear in row i?”, the full cube can be avoided with `Equal(X, c)` plus `ReduceMax`; `OneHot` is usually cleaner when several colors matter.

**Correction:** the card’s “power-of-2 row packing into a `uint32`” is exact only for **binary rows** or after one-hot expansion into bit-planes. ARC rows use symbols 0–9, so an exact direct encoding needs at least 4 bits per cell. A full 30-cell ARC row therefore needs about **120 bits**, which does **not** fit in a single `uint32` or `int64`; exact scalar packing works only for short rows, while full-width exact packing needs **multiple limbs** or keeps the row itself for equality. This is a mathematical correction, inferred from ARC’s 10-color alphabet and max width 30.

That leads to the recommended exact pattern key: **multi-limb positional encoding**. For ARC colors, base 16 is convenient because each cell fits in 4 bits. Split a 30-cell row into two 15-cell halves, compute two dot-products with constant weights `[16^0,16^1,...,16^14]`, and store the result as a key tuple `(lo, hi)` of shape `[N,2]`. Each 15-cell limb uses 60 bits and fits exactly in signed `int64`. For equality and dedupe, this is excellent. For full lexicographic sort, it is more awkward because `TopK` sorts a scalar, not a tuple; exact ordering on multi-limb keys therefore wants either a comparator network or a secondary stable pass. That tradeoff is **the** major reason to prefer small scalar signatures like occupancy or color-bitmask whenever exact full-pattern ordering is unnecessary.

A lossy alternative is **hashing**, such as XOR-folding or modular weighted sums. In ONNX main domain, `Xor` operates on booleans, so a practical fold uses boolean bit-planes or repeated compare/cast logic rather than an integer XOR primitive. Hashes are fine as **prefilters** for grouping, but never as the only dedupe criterion unless collisions are provably impossible under the task’s row universe. That collision caveat is an engineering inference, not an ONNX-published guarantee.

For datatype choice, prefer the **integer path** whenever the signature is supposed to be exact. `MatMul` and `ReduceSum` both accept integer tensors, and NumPy’s datatype docs remind you that `float32` has a 23-bit mantissa while `float64` has a 52-bit mantissa, so integer exactness from float packing is bounded by mantissa width rather than range. In practice, `float32` is fine for counts and tiny masks, but not for long packed signatures; `int64` is the safe default for exact keys.

The recommended toy implementation on a 5×5 grid is below. On the ONNX side, line-by-line this maps to `Equal/Not/Cast/ReduceSum`, `OneHot/ReduceMax` or `OneHot/ReduceSum`, and constant-weight `MatMul`.

```python
import numpy as np

# 5 rows x 5 cols toy grid, ARC-style colors 0..9
X = np.array([
    [0, 2, 0, 2, 0],
    [3, 3, 3, 0, 0],
    [0, 0, 0, 0, 0],
    [4, 0, 4, 0, 4],
    [3, 3, 3, 0, 0],
], dtype=np.int64)

bg = 0

# occupancy  -> ONNX: Equal + Not + Cast + ReduceSum(axis=1)
occupancy = (X != bg).sum(axis=1)

# color-set bitmask -> ONNX: OneHot(X, depth=10) + ReduceMax(axis=1) + MatMul(weights)
colors = np.arange(10, dtype=np.int64)
present = (X[..., None] == colors).any(axis=1).astype(np.int64)   # [5,10]
color_mask = present @ (1 << colors)                              # exact 10-bit mask

# exact row-pattern key for width<=15 in base16; for width 30 use 2 limbs of 15
row_key = X @ (16 ** np.arange(X.shape[1], dtype=np.int64))

print("occupancy:", occupancy.tolist())
print("color_mask:", color_mask.tolist())
print("row_key:", row_key.tolist())
```

## Rank reduction strategies

For arbitrary scalar row keys, the best default in opset 18 is still **`TopK` with `K=N`**. `TopK` returns sorted values/indices when `sorted=1`, and the spec says ties are broken by lower original index first, which is exactly the stability rule you want for “keep first occurrence” behavior. The main-domain operator catalog gives you `TopK` but no generic `Sort` operator page, so in practice `TopK(K=N)` is the standard ONNX-native full-sort primitive.

| Strategy | Ops | Memory | Compute (FLOPs) | Static-shape-friendly | Notes |
|---|---|---|---|---|---|
| TopK with `K=N` | `TopK`, `Gather` | values `[N]` + indices `[N]` | impl.-dependent; roughly `O(N log N)` comparator work, about 150 compare-equivalents for `N=30` (inferred) | Yes | Best generic choice for scalar sortable signatures |
| ArgMax-loop-unrolled | `ArgMax`, mask update, `Gather` | stagewise `[N]` scores plus 30 outputs | about `30×30 = 900` scan-comparisons plus masking (inferred) | Yes, but graph is bulky | Good when you need custom tie rules or no `TopK` |
| Bitonic-sort network | compare/exchange blocks built from elementwise compare/select | working keys `[32]` + indices `[32]` after padding | 240 compare-exchange units for padded `N=32`, depth 15 (classical network, inferred) | Yes | Excellent for hardware-like static graphs; verbose but predictable |
| Counting sort for small integer keys | `OneHot`, `ReduceSum`, `CumSum`, `Gather` or dense bucket mask | `[N,B]` where `B` is key range | `O(NB+B)`; for occupancy `B=31`, about 930 one-hot slots plus reductions | Yes | Usually cheapest if keys are in a tiny dense range |

For `N=30`, my practical ordering is simple. If the signature is already a scalar and the domain is not tiny, use `TopK(K=30)` and stop. If the signature is a small integer like occupancy or a 10-bit color mask with only a narrow task-specific range, use **counting-sort style bucketing**. If you need lexicographic compare on multi-limb exact keys, use a **comparator network** only if the task truly depends on that exact total order. Everything else is usually over-engineering in a cost-sensitive ONNX graph.

The unrolled-`ArgMax` pattern is still useful enough to know. `ArgMax` returns the first maximal index unless `select_last_index` is set, so you can repeatedly select the next best row, then overwrite that score with a large negative sentinel, all without `Loop`. The drawback is graph size: 30 rounds means 30 `ArgMax` nodes plus 30 masking rounds. It is static-shape-correct, but `TopK` is almost always smaller and clearer.

Counting sort deserves special emphasis because it matches ARC row signatures unusually well. Occupancy lives in `[0,30]`; row-color-count lives in `[0,10]`; some palette bitmasks may also live in a tiny task-local range even if their full theoretical range is larger. In those cases, `OneHot(key, depth=B)` gives a `[N,B]` bucket matrix, `ReduceSum` gives counts, and `CumSum(exclusive=1)` gives stable bucket offsets. The remaining assembly is either scatter-based or done through a dense permutation/equality mask.

The recommended “sort rows by occupancy” toy sketch is therefore:

```python
# numpy sketch; ONNX version is TopK(largest=1, sorted=1, K=N) + Gather(axis=0)
order = np.argsort(occupancy, kind="stable")[::-1]
rows_sorted = X[order]
```

In ONNX, the stable tie rule comes from `TopK` itself rather than NumPy’s `argsort`: equal values return the lower original index first.

## Row deduplication

There are three viable static families, and they differ mainly in whether you already have an **exact key**.

If you do have an exact per-row key, the cheapest dedupe is **key equality with a lower-triangular “seen before” mask**. Compute `eq[i,j] = (key_i == key_j)`. Apply `Trilu(upper=0, k=-1)` to keep only `j < i`, then reduce across `j`; row `i` is a duplicate if any earlier key matches. This preserves first occurrence and never needs `Unique`. `Trilu` explicitly supports lower-triangular extraction with diagonal control, which is exactly what the “keep first” rule needs.

If there is no exact scalar key, use a **pairwise row equality matrix**. Broadcast rows to `[N,1,M]` and `[1,N,M]`, compare with `Equal`, reduce across columns with `ReduceSum == M` or `ReduceMin`, then apply the same lower-triangular filter. On ARC’s 30×30 limit, this is only `30×30×30 = 27,000` cellwise comparisons, plus a 900-element pairwise matrix. That is not free, but it is still very manageable, especially when you only run it once per graph and it avoids extra packing logic.

The third path is **sort-then-compare-adjacent**. When you already need sorted rows for output, sort exact row keys, compare `key[i]` with `key[i-1]`, and keep only the first key in each run. This is usually the best option when dedupe is only a substep of a larger “sort then emit unique rows” rule. The catch is that adjacent comparison is only exact if the sort key is exact; if your key is a lossy hash, adjacent hash matches must still be verified by full-row equality. That collision caveat is the critical correctness condition.

Hash-first grouping is useful, but only as a **prefilter**. XOR-folds and other fixed-width hashes can collide, especially if rows are long or colors are multi-valued. The correctness rule is simple: **drop on hash alone only when the hash is mathematically injective over the task’s row domain**; otherwise, hash → candidate group → exact compare. In practice on ARC, exact injective hashes for full 30-cell multi-color rows are uncommon unless you keep multiple limbs.

The recommended toy dedupe sketch, preserving first occurrence, is:

```python
# exact-key dedupe; ONNX version is Unsqueeze + Equal + Trilu(lower, k=-1) + ReduceMax
eq = row_key[:, None] == row_key[None, :]
dup = np.tril(eq, k=-1).any(axis=1)   # row i duplicates an earlier row
keep = ~dup
deduped = X[keep]
```

For full-width ARC rows, replace `row_key` with a 2-limb base-16 key or with the row content itself if you cannot pack exactly.

## Row-signature to output mapping

Once signatures are computed, the output stage is usually easier than the signature stage.

The cheapest path is **index-first then `Gather`**. If a deterministic rule produces output row indices `idx[t]`, `Gather(data=X, indices=idx, axis=0)` is the cleanest ONNX mapping. This is ideal for “take the top 3 densest rows” or “emit rows in sorted order.” The only hard part is producing the index tensor.

When index production is awkward, use **equality-mask assembly**. Suppose target rows are defined by target signatures `target_sig[t]`. Build `mask[t,i] = [sig_i == target_sig_t]`, cast to numeric, and use `MatMul(mask, X)` to assemble output rows. This is exact and static as long as each target row matches exactly one source row, or you intentionally want additive collapse over several matching rows. In cost-sensitive graphs this can be surprisingly attractive because it avoids index plumbing and stays purely tensorized.

The most general view is a **permutation matrix** \(P \in \{0,1\}^{T\times N}\). Then `Y = P @ X`. You can think of `Gather` as the sparse/index form and equality-mask `MatMul` as the dense form of the same operation. Dense `P` is best when the same reorder is reused later, or when you need to blend row selection and row validity in one tensor. Sparse indices are better when intermediate memory is the priority.

A lookup-table variant exists when the signature domain is a small integer range. If signature `s` directly indexes a dense table `lut[s] -> source_row_index`, then `Gather(lut, s)` is a true row-lookup. This is most useful for rules like “the row with occupancy 3 becomes output row 0.” In practice, you often still build `lut` from equality masks or a scatter-like prepass, so pure lookup is only available when the domain and mapping are especially simple. That is an engineering synthesis rather than a special ONNX feature.

My recommendation is: **`Gather` when you already have row indices; equality-mask `MatMul` when the natural object is a signature match matrix; permutation matrices only when you need to reuse or inspect the reorder explicitly.**

## Variable-size output handling

ARC task outputs can differ in size from inputs, but static ONNX graphs cannot emit runtime-varying tensor shapes without much heavier machinery. In this family, the practical solutions are all variants of **fixed-size output plus validity metadata**. ARC grids use values 0–9 only, so there is no guaranteed spare color available as a universal “invalid” sentinel inside the grid payload itself.

The safest representation is **full-size grid plus a row-valid or cell-valid mask**. Emit `grid[N,M]` and `valid_rows[N]` (or `valid_cells[N,M]`). The kept rows are packed to the top, and the consumer crops by the mask rather than by content. Semantically this is the cleanest option because a legitimate all-background row is distinguishable from padding. If your serving environment accepts auxiliary outputs, this is the best general strategy.

The cheapest single-grid representation is **background-padded output**. Pack selected rows to the top of `grid[N,M]`, fill the remainder with background rows, and let downstream consumers ignore the padded tail. This is fine when the task semantics or evaluation wrapper already know where the tail begins, or when valid all-background rows cannot occur. It is the smallest-graph option, but it is semantically ambiguous in the general case. That ambiguity is an inference from ARC’s unrestricted use of color 0, not an ONNX limitation.

A more explicit alternative is **multi-output heads**: `grid`, `height`, `width`, maybe `valid_rows`. This is elegant for a private pipeline and still static-shape-friendly, but it only works if the downstream evaluator or wrapper actually consumes those extra outputs. In standard “return one ARC grid” contexts, it often needs a post-ONNX wrapper and so may not satisfy the outer submission contract. ONNX Runtime itself is happy with multiple fixed-shape outputs; the constraint is usually at the evaluation protocol.

For this project’s cost-sensitive setting, the decision is usually: **if the wrapper can read a mask, output `grid + valid_rows`; if not, use packed-top plus background padding and only apply it to tasks where background-tail ambiguity is harmless.**

## Worked examples for task017 and task173

**Assumption for this section:** I found public evidence that NeuroGolf uses ARC public training tasks serialized as `task001`…`task400`, but I did **not** find a public authoritative writeup for the exact mnemonics `row14_densecrop_i8pad` or `keepdims active nondiagless`. The two pipelines below are therefore **explicit inferences from the labels**, not claimed official solutions.

For **task017**, the label `row14_densecrop_i8pad row-signature based` strongly suggests a **dense-row selection / crop / pad** pipeline. The ONNX sketch I would try is:
`X -> active_mask = (X != bg) -> occupancy[rows] = ReduceSum(active_mask, axis=cols) -> keep_rows = occupancy >= τ -> score = keep_rows * BIG + occupancy -> TopK(K=N) -> Gather(sorted rows) -> Slice(first H_keep or first 14 logical rows) -> Pad/Concat to fixed output shape with color 8 if "i8pad" literally means pad-with-8 -> output`.
If the phrase `row14` means “select the row whose signature rank is 14” rather than “keep 14 rows,” the only change is that `TopK/Gather` produces a single row or a fixed narrow window around its rank. I would start with occupancy because “densecrop” is far more naturally explained by `count(non-bg)` than by a full-pattern key.

For **task173**, the label `keepdims active nondiagless` reads like a **keep-dimensions masking rule** rather than a true size-reduction rule. The graph I would try is:
`X -> active = (X != bg)`; build a fixed diagonal mask `D` for the 30×30 canvas; compute `active_nondiag = active & ~D`; reduce over columns to get `row_active`, and optionally over rows to get `col_active`; broadcast `row_active` and/or `col_active` back to grid shape; `Y = X * row_active[:,None] * col_active[None,:]`, leaving dimensions unchanged because of the “keepdims” hint.
If “nondiagless” instead means “remove rows whose only activity lies on a diagonal,” the same graph still fits: the diagonal-stripped activity reduction is exactly the row signature, and the output remains full-size with inactive rows zeroed out rather than physically dropped.

The reason these two sketches belong in the same family is that neither one is primarily object-centric. In both, the row is the unit of reasoning: a row gets assigned a compact signature, the signature drives rank reduction or masking, and the final output is a reassembly of rows rather than a free-form object rewrite. That matches the design family this note is trying to formalize.

## Decision tree

The table below is the practical “which recipe should I wire into the graph?” decision rule.

| Rule shape | Signature type | Reduction strategy | Output pattern |
|---|---|---|---|
| Keep rows with at least *t* non-background cells | Row-occupancy | threshold mask, then `TopK` or keep-in-place | packed-top rows or full-size masked grid |
| Keep rows containing color `c` or palette subset `S` | Row-color-set | boolean/equality mask, optional rank by bitmask | `Gather` selected rows or equality-mask `MatMul` |
| Deduplicate identical rows, preserve first occurrence | Exact row-pattern key or full row compare | lower-triangular pairwise equality | packed unique rows or full-size with validity mask |
| Sort rows by density / palette / small integer key | Occupancy or small packed color mask | `TopK(K=N)`; counting sort if key range is tiny | reordered rows via `Gather` |
| Count how many unique row types exist | Exact row-pattern key | pairwise equality + keep-first + `ReduceSum` | scalar dispatch into later branch / template |
| Need exact full-pattern equality but not exact full-pattern ordering | Multi-limb row-pattern key | pairwise equality matrix | dedupe/select by equality, avoid lex sort |
| Need same H×W output but row-level predicate still matters | any row signature | no shape reduction; broadcast validity back | full-size masked grid |

The summary recommendation is concise. Use **occupancy** first, **color-set** second, and **exact row-pattern keys** only when the task truly depends on entire row content. Use **`TopK`** for generic scalar ordering, **counting sort** for tiny integer domains, and **pairwise equality with `Trilu`** for preserve-first dedupe. If you can afford an auxiliary output, represent variable-size outputs as **packed grid + validity mask**; otherwise use **packed-top plus background padding** and only where the semantics allow it.

### Verification checklist

- **At least 5 distinct signature types surveyed:** row-occupancy, row-color-set, row-pattern, row-rank, and row-diversity are all covered above. ARC’s grid formalism and Hodel-style DSL practice support treating rows as valid transformation units.
- **Rank reduction covered without `Sort` or `Unique`:** the note covers `TopK(K=N)`, unrolled `ArgMax`, bitonic-sort networks, and counting sort, explicitly avoiding `Unique` as an implementation dependency. `TopK` and `ArgMax` are standard ONNX ops; `TopK` provides sorted outputs and stable tie-breaking by lower index.
- **Deduplication strategy with explicit collision analysis:** hash-prefilter dedupe is discussed with an explicit “verify exact equality after hash grouping” rule; exact-key pairwise equality and sort-then-compare-adjacent are also covered.
- **Variable-size output handling covered for 2+ strategies:** the note covers packed-top plus background padding, full-size grid plus validity mask, and multi-output heads carrying shape/validity metadata.
- **task017 + task173 worked examples sketched:** both are included and clearly marked as **inferred** because I did not find a public authoritative description for the exact mnemonic labels. Public evidence for NeuroGolf task numbering and ARC provenance is cited.
- **Examples compatible with ONNX opset 18 + ORT 1.24:** every recommended pipeline is built from standard main-domain ops already defined no later than opset 18, including `ReduceSum`, `MatMul`, `TopK`, `Gather`, `OneHot`, `ArgMax`, `Equal`, `Cast`, `ReduceMax`, `Trilu`, and `CumSum`. ORT’s compatibility docs state that all versions support opset 7+ and that prebuilt packages have full support for ONNX opsets/operators, so these opset-18 patterns are within ORT 1.24’s supported envelope. 