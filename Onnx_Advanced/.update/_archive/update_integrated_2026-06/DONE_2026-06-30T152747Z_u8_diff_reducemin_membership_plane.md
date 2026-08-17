# UINT8 `Sub`→`ReduceMin` single-plane membership (sidesteps the no-BOOL-reduce wall)

## What I discovered
A per-axis "does value V appear in this row/column?" membership test can be done with **ONE**
16×16 (general N×M) UINT8 intermediate instead of two, when you need the answer as a projected
line. The standard pattern in this codebase is two planes per probed value:

```
match = Equal(source, V)          # BOOL  N×M  (256 B)
hit   = Where(match, V, 0)        # UINT8 N×M  (256 B)   <- second plane exists only because
line  = ReduceMax(hit, axis)      #                          ReduceMax/ReduceSum reject BOOL
```

The two-plane shape was believed forced "because `ReduceMax(BOOL)` is unsupported." That's true
(re-verified: opset 12 AND opset 14 ORT both raise a type error on `Reduce*(BOOL)`), but it's the
wrong lever. Reduce an **arithmetic** plane instead of a comparison plane:

```
diff     = Sub(source, V)          # UINT8 N×M  (256 B)   wraps mod 256; == 0 iff source == V
min_line = ReduceMin(diff, axis)   # == 0 iff that row/col contains a cell == V
present  = Equal(min_line, 0)      # BOOL line (tiny: 1×M / N×1)
hit_line = Where(present, V, 0)    # UINT8 line (tiny)
```

One 256 B plane per probed value instead of two. Output is byte-identical.

## Why it's correct
UINT8 subtraction is 0 **exactly** when the operands are equal (no false zero from wraparound:
a≠b ⇒ (a−b) mod 256 ∈ 1..255). So `ReduceMin(diff)==0` ⇔ a match exists along that axis. Safe as
long as the probed value V can never collide with a "don't-care" filler value already in `source`.
In task340 the source holds real colors 1..9, background 0, and sentinel 10; edge colors are always
1..9, so empty/masked cells never reduce to 0. Verify this disjointness for your task before using it.

## Requirement / trap
UINT8 `Sub` and `ReduceMin` have **no kernel below opset 14** (matches `ort_reality` #41: I8/U8
Mul/Add/Sub arrived at opset 14; INT32 is the floor below 14). Bumping a graph from opset 12→14 also
moves `ReduceSum`'s `axes` from an attribute to an input tensor (`ReduceMax`/`ReduceMin` keep the
attribute until opset 18). Both changes are cheap. `Reduce*(BOOL)` remains unsupported at 14 — this
trick does not reduce a BOOL, it reduces the U8 diff.

## Where / gain
- task340 (Filling-Regions projection). `The_400_Tasks/Task_by_Task/task340/prototypes/build_task340_v3.py`.
- Staged: `The_400_Tasks/Task_by_Task/task340/staged_candidates/task340_filltopologyrewrite_cost7718/`.
- **8612 → 7718 (−894 B, −10.4%), byte-equal to LIVE** (AB 200266/200266, divergences=0; truth 200000/200000).
  Dropped one 16×16 plane for each of 4 edge colors. The prior floor proof had declared 8612 the
  formulation floor — it only ever considered *merging colors*, never this 2-plane→1-plane rewrite.

## What's missing from the library
`op_cost_patterns` / `ort_reality` document that `Reduce*(BOOL)` is unsupported and that you must
`Where(match,color,0)` before `ReduceMax`. There is no entry for the cheaper inverse: when you want a
membership LINE (not the matched values themselves), reduce `Sub(source,V)` with `ReduceMin==0` and
spend only ONE plane. Worth a short technique entry + a `by_symptom` route under "membership / does-X-appear-in-row-or-col".

## Sorter note: Onnx_Advanced/op_cost_patterns.md — add as new technique "u8_diff_reducemin_membership" (1 plane vs 2 for per-axis membership; requires opset 14 U8 Sub/ReduceMin; correctness needs V disjoint from filler values). Measured −894 B on task340 (8612→7718, byte-equal). Exemplar: build_task340_v3.py.
## Sorter note: Onnx_Advanced/by_symptom.md — add route: "per-row/col membership of a known scalar → u8_diff_reducemin_membership (not Equal+Where+ReduceMax)".
## Sorter note: Onnx_Advanced/WORKER_PROTOCOL.md "Ten most-recent verdicts" — replace oldest line with: "#NNN u8-diff ReduceMin membership: per-axis 'value V present?' line costs ONE U8 Sub plane (ReduceMin==0), not Equal(BOOL)+Where(U8); opset14; V must be disjoint from source filler. task340 −894 B byte-equal (8612→7718)."
