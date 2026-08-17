# Sub-byte dtypes (UINT4/INT4) cut NEITHER memory NOR params — both metrics floor at 1 B/elem

**Where:** task091 worker1 2026-07-08. Probe: `The_400_Tasks/Task_by_Task/task091/prototypes/probe_task091_floor.py` (measures dtype itemsizes + two floor graphs). WALL floor proof in `The_400_Tasks/Task_by_Task/task091/hard_records/selector_as_index_floor.md`.

## What I measured (both are grader source-reads + a runtime probe, not guesses)

`neurogolf_utils.calculate_memory` charges each named node-output tensor `num_elements × np.dtype(np_dtype).itemsize` (graph `input`/`output` exempt). `neurogolf_utils.calculate_params` charges `math.prod(init.dims)` — a pure ELEMENT count, dtype-blind.

Measured `np.dtype(onnx.helper.tensor_dtype_to_np_dtype(T)).itemsize`:

| dtype | itemsize the grader uses |
|---|---:|
| UINT4 | **1** |
| INT4 | **1** |
| BOOL | 1 |
| UINT8 | 1 |
| F16 | 2 |
| F32 / INT32 | 4 |
| INT64 | 8 |

numpy/ml_dtypes has no true nibble storage — `uint4`/`int4` occupy a full byte, so the grader prices a UINT4 activation plane at **1 B/elem, identical to UINT8**. A 30×30 UINT4 plane is 900 B, NOT 450. There is no dtype the grader charges below 1 B/elem.

## Why it matters — resolves a live library contradiction

- `scoring_internals.md:75` (correct): "dtype-packing initializers (uint8/uint4/f16) does NOT cut params — `calculate_params` counts ELEMENTS, dtype-blind." ✓ confirmed by source read above.
- `tiny_architectures.md:109-192` (STALE / wrong for the scored metric): claims uint4 weights give "4–7× savings" ("6 × 90 × 4 = 2160 bytes float32; uint4: 270 B", "uint4: 45 B"), and line 111 flags "if opset 21 accepted — A.4 follow-up" as unresolved. The A.4 follow-up resolves to **NO**: params is an element count, so uint4 saves 0 params; and uint4 activations save 0 memory (itemsize 1 == uint8). uint4's only effect is raw file bytes, which no metric charges. Those savings columns are misleading and should be struck or explicitly relabeled "file bytes only, not scored."
- `cost_floor_reference.md:57-59` dtype×shape table lists F32/F16/U8/BOOL columns but no UINT4/INT4 — worth adding a column showing they equal the U8 column (1 B/elem), so future workers don't re-probe.

## Fresh cost-floor calibration for the canvas taxonomy

task091 (crop-to-color-bbox → 30×30 one-hot) sits in the `cost_floor_reference.md` full-canvas-plane floor class. Measured with `probe_task091_floor.py`:
- **Skeleton** (bare `Conv→Cast→STATIC Slice→Pad→Equal`, zero bbox/geometry) = **2258** (idx_f F32 15×15 = 900 + idx_u8 225 + crop_u8 196 + canvas30 U8 30×30 = 900 + params 37). This is the irreducible core BEFORE any rule logic.
- Two forced planes: the F32 channel-combine source (Conv output is F32 → 15·15·4 = 900) and the 30×30 pre-Equal output plane (the free final `Equal(plane,ids)→output` can broadcast channel but not upsample H/W, so the plane is full 30×30 = 900 at min itemsize). Confirms task224/task137's "full-canvas 900 BOOL/U8" law: keep the `Equal` as the graph output (free), but its 30×30 predecessor is charged and 900 is the dtype-independent floor.
- Deleting the 30×30 plane by expanding to a 10-channel BOOL crop then free-Pad→output measures **3317** (Equal@14×14 = 10·14·14 = 1960 > 900) — the growth-op law (#109/techniques.md:4243): the Pad predecessor is charged, so expanding channels early loses.

## Sorter note: scoring_internals.md §75 (Dead levers) — extend existing
Add the measurement that UINT4/INT4 **activation** itemsize == 1 (numpy has no nibble storage), so sub-byte packing cuts memory by 0 too — not just params. One line; the "buys file bytes only" claim is right, just make it explicit it covers memory as well as params, with the measured itemsize table.

## Sorter note: tiny_architectures.md §uint4-weights (~L109-192) — replace existing
Strike/relabel the "uint4 4–7× savings / 45 B / 270 B" claims and the open "A.4 follow-up" (L111): resolved NO. `calculate_params` is a dtype-blind element count and uint4 activations charge 1 B/elem, so uint4 saves neither scored metric — file bytes only. Cross-link scoring_internals §75.

## Sorter note: cost_floor_reference.md §dtype-shape-table (L57-59) — extend existing
Add UINT4/INT4 to the dtype×shape cost table as a column equal to the U8/BOOL column (1 B/elem). Optionally add task091 skeleton=2258 as a calibration row in the full-canvas-plane floor taxonomy (L16): 30×30 U8 output plane 900 + 15×15 F32 combine source 900 are both forced; 10-ch early-expand alternative = 3317.
