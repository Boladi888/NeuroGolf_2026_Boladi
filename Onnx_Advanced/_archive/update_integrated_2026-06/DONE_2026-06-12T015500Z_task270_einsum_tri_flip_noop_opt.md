# task270 center-noop shave: upper_tri elimination + noop-direct optimizations

**Session**: worker-270, 2026-06-12 ~01:00Z
**Task**: task270 (anchor-cross scatter, 15x15 grid)
**Builder**: `The_400_Tasks/Task_by_Task/task270/prototypes/build_task270_v14_const_updates.py`
**Staged**: `staged_submits/task270_center_noop_shave_cost2645/task270.onnx`
**Cost chain**: 2978 (v10 LIVE) → 2753 (v11) → 2721 (v12) → 2657 (v13) → 2645 (v14)

---

## Insight 1: Single triangular mask for both directions via subscript swap

In multi-input ONNX Einsum, a single `lower_tri[15,15]` initializer can replace both `lower_tri` and `upper_tri` by swapping the two subscript indices.

```
lower_tri[z,v] = 1 if z < v   (subscripts: zv)
upper_tri[z,v] = 1 if z > v = lower_tri[v,z] (subscripts: vz)
```

In practice: equations ending in `...jz,zv->b` (left/up test) become `...jz,vz->b` (right/down test) with the same `lower_tri` tensor. No new node added, just equation string change.

**Measured gain**: -225 params on task270 (15×15 mask), cost 2978→2753.

**Generality**: applies to any Einsum-based triangular test where the same tensor serves both "before" and "after" directions. Saves `[N,N]` float params (here N=15 → 225 params F32 = 225 scored params).

**Proof**: `upper_tri[z,v] = lower_tri[v,z]` exactly — confirmed numerically. This is a general identity, not task-specific.

## Sorter note: techniques.md — add as new §: "Einsum triangle mirror: upper_tri = lower_tri transposed; swap subscript pair to eliminate one matrix (saves N×N params)". Also note in `cost_floor_reference.md` that triangular mask cost is N×N params but only ONE matrix needed (other direction is free via subscript swap).

---

## Insight 2: Redundant center-noop Where nodes → direct reuse

In the center-noop scatter pattern (inactive arm writes center-color at center-coord instead of sentinel-noop-coord), the "row coordinator" for row-moving arms is always the center row:
```
left7_r = Where(left7, r1_i, r1_i)  # both branches = r1_i → always r1_i
```

These Where nodes are NOPs. They can be eliminated by using `r1_i` directly in the Concat.

In a 4-direction cross with 2 color-pairs:
- Row arms (left/right): row coord never changes → 4 redundant Where nodes
- Col arms (up/down): col coord never changes → 4 redundant Where nodes
Total: 8 redundant Where nodes eliminated = 8×8 = 64 bytes INT64 memory savings.

**Measured gain**: -64 bytes memory, cost 2721→2657.

**Generality**: any center-noop scatter where the "inactive fallback" coordinate for one axis equals the already-computed center coordinate. If `Where(flag, coord, center_coord)` and the fallback IS the center coord (not a derived value), the Where is a no-op and can be removed.

## Sorter note: techniques.md §for "center-noop scatter" — add note: "Redundant Where(flag, x, x) = x: when active and noop branches both produce the same value (e.g., row coord for row-moving arm), eliminate the Where and use the input directly. Saves 8 bytes per eliminated Where on INT64 scalars."

---

## Insight 3: Constant ScatterND updates with center-writes-last

In a center-noop scatter where arm values (7 or 3) differ from center values (1 or 2), the inactive arm would need a conditional value: `Where(flag, arm_color, center_color)`. 

**Optimization**: reorder ScatterND updates so arm writes come first, center writes come last. Then arm writes can ALWAYS use the arm_color (constant), because the center write (last) will overwrite any arm_color written at the center-noop coordinate.

This eliminates:
- 8 arm_val Where nodes (dynamic value selection per arm) 
- The Concat building the updates tensor (now a constant initializer)
- The constants one_u8, two_u8, three_u8, seven_u8 (embedded in const_updates array)

Replaced by a single `const_updates [10] U8 = [7,7,7,7,3,3,3,3,1,2]` initializer (10 params).

**Net**: -18 bytes memory + 6 params = -12 cost.

**Correctness requirement**: ScatterND processes updates in order (first→last). The center must be the LAST writer to the center coordinates (slots 8,9 in this task). Arm writes go to slots 0-7.

## Sorter note: techniques.md §for "ScatterND patterns" — add: "Constant updates + reorder pattern: if inactive arms are noop'd to center-coord, make center writes last in the updates array, then all arm values can be constants (no conditional Where needed). Saves arm_val Where nodes + dynamic Concat → single const initializer."

---

## Insight 4: INT64 offset arithmetic after single Cast

When computing offset coordinates for arms (center ± 1), using:
```
r1m = Sub(r1, one_f)  # F32 intermediate
r1m_i = Cast(r1m, INT64)
```

is equivalent to:
```
r1_i = Cast(r1, INT64)  # same as before
r1m_i = Sub(r1_i, one_i64)  # INT64 arithmetic
```

The second approach eliminates the F32 intermediates (r1m, r1p, c1m, c1p, r2m, r2p, c2m, c2p = 8 × 4 = 32 bytes). `one_f` can be replaced by `one_i64` (same param count, different dtype).

**Measured gain**: -32 bytes memory, cost 2753→2721.

**Generality**: any graph computing integer coordinates from float Einsum outputs via Cast→Sub/Add pattern. Do the Cast first, then arithmetic in INT64.

## Sorter note: techniques.md §for "coordinate extraction patterns" — add: "Cast-then-offset pattern: compute all offsets in INT64 after a single Cast, not in F32 before Cast. Saves N×4 bytes (N=number of offsets). Replace one_f constant with one_i64 (same param count)."

---

## Background: truth timing

v14 (and v10-v13) use 9-input Einsums that run ~37ms/case for truth checks. Full 200k truth would take ~2 hours. Gate strategy used: algebraic proof (v14 is mathematically identical to v10 which is LIVE) + AB byte-equal 2266/2266 + truth 5000/5000.

Key lesson (already in Logs/lessons.md 2026-06-05): local truth-full timeout ≠ grader risk. The grader evaluates ~1k pairs, so the slow 9-input Einsums are fine for production.

## Sorter note: this is already in Logs/lessons.md — no new addition needed.
