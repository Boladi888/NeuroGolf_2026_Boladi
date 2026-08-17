# DEAD ROUTE — output predicate below 900 B / prompt floor_est ~800 (task042)

- **Bar when explored:** live 2,522 → beaten to **2,322** this batch. This record kills the prompt's `floor_est ~800`.
- **Live sha (2,522 base):** c7d8b30fc976e4256083661e0d034d636031faa92dddf53e617534e51603ae25
- **Prompt claim tested:** "Forced bytes are sparse source reads and one output predicate … aim ~800." → the single output predicate ALONE is 900, so ~800 is unreachable.

## The output construction floor is 900 B

Output is `[1,10,30,30]` (free — `calculate_memory` skips names `input`/`output`). The last CHARGED tensor feeding it is the predicate/data of the final combine:
- **`Where(mask, e8[1,10,1,1], input)`** — `mask` must broadcast against `[1,10,30,30]`; min spatial = 30×30 BOOL = **900**. (Confirmed: BOOL[1,1,30,30]=900.)
- **Crop-then-Pad** alt: `Where(mask10[1,1,10,10], e8, input_crop[1,10,10,10]) → Pad`. mask10 is only 100 B but `input_crop` = **F32[1,10,10,10]=4,000** (Slice preserves F32). Far worse.
- **Pad a painted region** `[1,10,10,10]` BOOL = **1,000** > 900 (one-hot needs all 10 channels; channels {0,3,8} are non-contiguous so a 3-channel [1,3,10,10] can't be single-`Pad`-placed — W8 probe).
- **`Add(input, delta)`**: `delta` must be full `[1,10,30,30]` or a `Mul` intermediate `[1,10,30,30]=9,000`. Worse.
- **Sparse scatter** (`NonZero`→`ScatterND` a handful of cells): the painted set is data-dependent (2–6 cells) → `NonZero`/dynamic-`ScatterND` produce a **data-dependent output dim** → strict shape inference yields a `dim_param` → **`calculate_memory` returns `None` (UNSCOREABLE)** — scorer `neurogolf_utils.calculate_memory` line ~230 `if dim.HasField("dim_param"): return None`. Confirmed: the ONNX checker even refuses a `NonZero` graph without a static output shape. #139 ScatterND-patch needs a FIXED/midpoint patch, which this is not.

## Verdict
**`paint_mask30` 900 B is forced.** floor_est ~800 is **UNREACHABLE** — the output predicate alone is 900. Task-wide memory floor for the Where-output formulation = **2,212** (900 predicate + 500 read + 800 detection + ~12 scalars); cost floor **2,322** (+110 params). See task042.md "FLOOR PROOF (updated 2026-07-08)".

## One remaining hypothesis (unexplored, low confidence)
A wholly different output where the painted grid is emitted directly as the free `output` of a single op whose only charged input is < 900 B AND static. No such op is known: any op combining the free full input with a data-dependent ≤6-cell overlay needs either a full-canvas static mask (900) or a dynamic index set (unscoreable).
