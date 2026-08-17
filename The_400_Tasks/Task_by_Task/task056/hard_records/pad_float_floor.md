# task056 hard_record — cost-33 is the forced floor (Pad-float coupling)

- **Live bar:** cost **33** = memory 32 + params 1 (profile=static=33, 46 real inputs).
  - source: manifest (authoritative), owner Vamshi, updated 2026-07-02.
  - live sha256: `560206e847ad68edba82ee78a0387bc078936bf342b4f060c9e8c340a7872724`
  - graph: `task056_xor_falseinit_f16pad` — opset-9 XOR bit-decoder.

## The two rumors this batch was sent to test — both DISPROVEN

1. **"perfect-25 / cost ≤1 identity/transpose"** (prompt SPEC framing): FALSE.
   The generator (`task_27a28665.py`) paints one of **4 fixed silhouettes**;
   `output[0][0] = idx ∈ {1,2,3,6}`. The output channel index is a 4-way
   classification of the input silhouette → forced to read ≥2 input cells
   (log2(4)=2 bits) AND place a color-specific one-hot on a 30×30 canvas.
   No zero-param single op (Identity/Transpose/Slice/Gather) does 4-way
   classification. Cost ≤1 is impossible. Not a rumor-slot win.

2. **"drop the F16 Cast, Pad a BOOL/UINT8 directly → cost 21"**: FALSE at the
   floor opset. `np.array_equal` in the gate compares by VALUE not dtype, so a
   bool output WOULD match the float32 one-hot reference — but **ORT opset-9
   Pad rejects bool/uint8/int8** (`INVALID_GRAPH: Type 'tensor(bool)' of input
   ... of operator (Pad) ... is invalid`). Pad-2 (opset ≤10, attribute `pads`)
   is float-only by spec (T = f16/f32/f64).

## Why you can't buy the bool Pad — the fundamental coupling
- Attribute `pads`/`starts`/`ends`/`axes` = **0 counted params** exist ONLY in
  the opset-≤10 op versions (Pad-2, Slice-1).
- Bool/int Pad exists ONLY in Pad-13+ (opset ≥11), where `pads` is an **input
  initializer** (+8 params) AND Slice-10+ takes starts/ends/axes as **input
  initializers** (+15 params for the two single-cell slices).
- MEASURED: opset-13 bool-Pad rebuild (no Cast) = **44** = memory 20 + params 24.
  The −12 B memory saving is swamped by +23 params. `prototypes/task056_op13_bool_nocast.onnx`.

So: free pads ⟺ float Pad ⟺ the `active_channels [1,6,1,1] F16 = 12 B`
intermediate is forced. There is no opset where you get both cheap params and
a non-float Pad.

## Forced-component cost breakdown of the live 33 (all irreducible)
| component | bytes | why forced |
|---|---|---|
| `b0_f`,`b2_f` `[1,1,1,1]` F32 | 4+4=8 | Slice preserves input F32; 2 cells needed to distinguish 4 masks (2 bits) |
| decode bools h1,h3,equal_bits,b0_bool,h6,h2 | 6 | 4 minterms of 2 vars + 2 helpers = minimal (Greater trick already avoids extra casts/nots) |
| `active_bool [1,6,1,1]` BOOL | 6 | channels 1..6 span (colors {1,2,3,6}); ch4/ch5 are false-fillers because Pad can't insert INTERIOR zeros — 6 is the min for a single Pad |
| `active_channels [1,6,1,1]` F16 | 12 | Pad needs float; F16 is the smallest float (see coupling above) |
| `false_bool [1,1,1,1]` param | 1 | the only initializer; reused twice for the ch4/ch5 gap |
| **total** | **33** | |

## Routes proven worse (do NOT re-explore)
- bool/uint8/int8 output at opset 9: ORT `INVALID_GRAPH` (Pad float-only).
- opset-13 bool Pad (no cast): **44** (measured).
- Concat-with-zero-inits instead of Pad: interior/spatial zero tensors are
  billed by element count (e.g. `[1,10,30,29]` = 8700 params). Absurd.
- Upsample/Expand/Tile to canvas: replicate the 1×1 value across all 30×30
  cells → wrong output (reference is zero everywhere except (0,0)).
- ConstantOfShape/ScatterND canvas: materializes a billed `[1,10,30,30]` (9000 B) intermediate.
- Single [1,1,1,3]/[1,1,1,2] slice for both cells: 12 B / needs re-slicing → ≥ current 8 B.

## Verdict
**WALL at 33.** The live XOR decoder (Vamshi) is at the structural floor for the
ONNX/ORT op set. The one lever (kill the F16 Cast) is blocked by the
float-only Pad-2 type constraint, and buying a non-float Pad costs +23 params
to save 12 B. Nothing below 33 is legal here.
