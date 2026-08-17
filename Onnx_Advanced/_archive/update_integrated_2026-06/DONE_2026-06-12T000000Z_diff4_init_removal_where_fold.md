# Diagonal diff4 init removal + Where-fold gate trick

**Task**: 136 (coord-ray diagonal pattern)
**Builder**: `The_400_Tasks/Task_by_Task/task136/prototypes/build_task136_coord_tail_v3.py`
**Staged**: `staged_submits/task136_coord_tail_cost1492/`
**Result**: 1,670 → 1,492 (−178 bytes, −10.6%)

## What I discovered

### 1. Separable diff4 matrix → dynamic Sub (saves params, cheap intermediate)

The diagonal membership check `Equal(diff4, d_i)` where `diff4[r,c] = 4*(r-c)` can be expressed as `Equal(Sub(row4p2, d_i), col4p2)` because `row4p2[r] - col4p2[c] = 4*(r-c)`. 

The `diff4` init is a [10,10] F32 tensor (100 params). The alternative: `s_i = Sub(row4p2, d_i)` is a [10,1] F32 tensor (40B memory, 0 params) computed at runtime. Net per-color: save 100 params at a cost of 40B memory = net −60. But ALSO use s_i for the block row comparisons (see below).

### 2. Merge duplicate row/col vectors (saves 30 params)

`row30 = [0,1,...,29]` and `col30 = [0,1,...,29]` are identical initializers. In Einsums:
- Row: `Einsum("bchw,c,h->b", input, sel, row30)` — h-axis
- Col: `Einsum("bchw,c,w->b", input, sel, col30)` — w-axis

Both use the same values. Share a single `rowcol30` init for both. Saves 30 params with zero memory cost.

### 3. Eliminate b{i}_r_hi scalars by reusing b{i}_c_hi

The row upper bound `b_r_hi = r_i + 4` serves the block row check `row4p2 ≤ b_r_hi`. With the `s_i = row4p2 - d_i = row4p2 - r_i + c_i` substitution:

`s_i ≤ c_i + 4 ↔ row4p2 - r_i + c_i ≤ c_i + 4 ↔ row4p2 ≤ r_i + 4`

So `LessOrEqual(s_i, b_c_hi)` where `b_c_hi = c_i + 4` replaces `LessOrEqual(row4p2, b_r_hi)`. Eliminates the `b_r_hi = Add(r_i, plus4)` scalar node (4B each × 2 = 8B saved).

### 4. Where-gate trick: fold diagonal check + directional gate into one Equal

**Key insight**: Instead of computing `line_i = Equal(s_i, col4p2)` [100B] then `ray_i = And(line_i, b_r_le)` [100B], use:
- `s_i_masked = Where(b_r_le, s_i, sentinel)` [40B F32]
- `ray_i = Equal(s_i_masked, col4p2)` [100B BOOL]

For rows outside the gate (b_r_le=False), `s_i_masked = sentinel`. If `sentinel ∉ col4p2`, then `Equal(sentinel, col4p2[c]) = False` for all c. So `ray_i = False` for those rows. No separate And node needed.

**Sentinel value**: reuse `plus4 = 4.0`. It is NOT in `col4p2 = {2,6,10,...,38}` (all values ≡ 2 mod 4; 4.0 ≡ 0 mod 4). Also NOT equal to any `s_i` value since `s_i[r] = 4*(r - row_i + col_i) + 2 ≡ 2 (mod 4)`. Safe to reuse without adding a new param.

Net per color: save 200B (line_i + old-ray_i And) but add 40B (s_i_masked) + 0 (ray_i already counted). So net −160B... wait: the `ray_i` 100B tensor still exists (it's just produced by Equal instead of And). The saving is: remove `line_i` (100B). Also add `s_i_masked` (40B). Net per color: −60B. For 2 colors: −120B.

Wait, more carefully: the `line_i` intermediate (100B) disappears. The And node output `ray_i` is replaced by an Equal output `ray_i` — same 100B. The Where intermediate `s_i_masked` is +40B. So per color: −100B + 40B = −60B. For 2 colors: −120B.

### 5. Total accounting

| change | params delta | memory delta |
|---|---|---|
| Remove diff4 init | −100 | 0 |
| Add s1 = Sub(row4p2, d1) | 0 | +40B |
| Add s2 = Sub(row4p2, d2) | 0 | +40B |
| Merge col30 into row30 | −30 | 0 |
| Eliminate b1_r_hi | 0 | −4B |
| Eliminate b2_r_hi | 0 | −4B |
| Eliminate line1 | 0 | −100B |
| Eliminate line2 | 0 | −100B |
| Add s1_masked (Where) | 0 | +40B |
| Add s2_masked (Where) | 0 | +40B |
| **Total** | **−130** | **−48B** |

Total cost reduction: −130 − (−48) = −178 bytes. ✓

## Why it matters

This shows that separable matrix initializers (any matrix that is an outer product `f(r) - g(c)` or `f(r) + g(c)`) can be replaced by dynamic Sub/Add ops when:
- The static params cost (element count) exceeds the runtime intermediate memory cost
- The runtime intermediate can be reused for multiple downstream comparisons

Transfer pattern: any `Equal(big_2D_init, scalar)` where `big_2D_init[r,c] = f(r) - g(c)` can become `Equal(Sub(row_vec, scalar), col_vec)` saving `(HW - H - W)` params at the cost of `H` memory.

## Where to add in library

The "separable init → Sub" idea could extend techniques #82 or §39. The Where-fold trick for gating a comparison is a general pattern that could be added to the free-output catalog (techniques #67) or the comparison/BOOL tails section.

Specifically: **When you need `And(comparison, gate_mask)` where the comparison output is a `[H,W]` BOOL plane and gate_mask is a `[H,1]` or `[1,W]` BOOL strip**: instead of materializing the comparison plane then ANDing, preprocess the comparison INPUT by masking with Where using a sentinel (not in the comparison range), then do the comparison directly. Saves the `[H,W]` And input plane.

Sentinel safety rule: for `Equal(masked_vec, const_vec)` where const_vec has values in set S, pick sentinel ∉ S. If const_vec has all values ≡ k (mod m), pick sentinel ≢ k (mod m).
