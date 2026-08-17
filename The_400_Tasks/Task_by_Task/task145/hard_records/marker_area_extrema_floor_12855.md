# task145 — marker/area-extrema floor at 12855 (WALL)

**Session:** bo_ng_cld_worker3 (Claude lane), 2026-07-11
**Live bar:** cost **12855** = 12744 memory + 111 params · sha256 `2dbfb6f9d5ea294d2e0be0fbfd5fbb2460a1e249de794bd3bb31f95a48b4f883` · owner Sho · manifest 2026-07-09 (via `ng_current_bar.py --task 145`). Static == profile (fully static graph; scorer returns None on any dynamic dim).
**Verdict:** No strict-below-12855 correct candidate found. Every intermediate plane is load-bearing (2 shaves gated + disproved; the rest disproved by the verified cost model). **This supersedes the stale 13154 floor proof in task145.md** — Sho beat that with the color-consolidation + MaxPool-marker formulation below.

## The rule (verified from task JSON, GUI/client/public/tasks/task145.json)
Color-2 walls partition the grid into **axis-aligned rectangular** background regions.
Fill the **largest-area** region with 1, the **smallest-area** region with 8; walls stay 2;
other background stays 0; out-of-grid cells stay 0. Because regions are true rectangles,
bounding-box extent == exact area (the connected-component/L-shape trap in the don't-do log
does NOT bite here). Generator guarantees `min_area < max_area` ⇒ class-9 (both) never fires.

## The live formulation (Sho, 12855) — what makes it optimal
1. `Slice(input, ch0, 20x20)` → `z_f32` F32 **1600** (input access; F32 forced by input dtype).
2. `Cast`→`z_bool` (bg=True). 4× `Where(z_bool,0,pos_ramp)` + 4× `MaxPool` (kernel [1,20]/[20,1],
   asymmetric pad) = prefix/suffix-max of wall positions ⇒ per-cell width/height. `Add`→`lr_sum`/`ud_sum` U8.
3. `DequantizeLinear(sum, scale=1, zp=20)` → `neg_width`/`neg_height` F16 (= −width/−height);
   `Mul`→`area` F16 (=width·height); `ReduceMax`→max; `Where(z_bool,area,max_area)`=`area_for_min`,
   `ReduceMin`→min; two `Equal`→`max_mask`/`min_mask`.
4. Color cascade (4 U8 planes): `color0=Where(z_bool,0,2)`; `color1=Max(color0,w1_10,w2_10)` (out-of-grid→10);
   `color2=Where(max_mask,1,color1)`; `color_core=Where(min_mask,8,color2)`.
5. `Pad`→`color30` U8 **900** (30×30 label); **one** `Equal(color30,[0..9])` → 10-ch output. Output is FREE.

## Verified cost model (neurogolf_utils.calculate_memory / _score_v2)
`cost = Σ(every named intermediate tensor bytes, max(static,runtime)) + params`. Tensors named
`input`/`output` are **excluded** (free). Fully static required (dynamic dim ⇒ score None).
`com.microsoft` domain ⇒ score None. Banned ops: LOOP/SCAN/NONZERO/UNIQUE/SCRIPT/FUNCTION/COMPRESS.
From-scratch replica (`prototypes/build_task145_marker_variants.py --variant base`) reproduces
12744 intermediate bytes exactly, **visible 267/267**.

## Pass-2 (2026-07-11 forced second pass) — genuinely different formulations, all gated
- **base replica** (from-scratch, `--variant base`): `profile_real_cost` **static=12855 profile=12855**; `ng_graph_cost_diff` vs live Sho = **net delta 0** (byte-structure-identical, only tensor-name diffs); visible 267/267. Confirms my replica IS the floor.
- **u8area** (`--variant u8area`) — genuinely different DOMINANT DTYPE: collapse the F16 area to U8 for extrema/masks (exploiting "≤1 region ≥256 ⇒ unique max"). Result: 12742 B (saved only 2 scalar bytes; `area_u8` +400 cancels `area_for_min` U8 −400) and **visible 266/267 FAIL** — the F16→U8 Cast breaks one case ⇒ **F16 exact area is load-bearing** (U8 is a hidden-risk approximation). Dead.
- Library mining (INDEX→techniques.md): negative-MaxPool MinPool trick (§33) and CumSum→ArgMax run-length (§4h) — neither helps: Sho's pre-reversed `pos_right` initializer already gets the suffix-min for free, and CumSum forces F32 (more expensive than U8 MaxPool markers). No new lever.

## Shaves attempted and killed
| # | Idea | Predicted save | Gate result | Why dead |
|---|------|----------------|-------------|----------|
| 1 | **drop `area_for_min`** — min via `ReduceMin(area)` over ALL cells | −800 → 12055 | `visible 0/267 FAIL` | walls have marker-area=1 = global min ⇒ `min_mask` fires on walls (→ class 8) and true min region (area>1) is missed. Wall-exclusion plane is load-bearing. |
| 2 | **merge color cascade** — additive `base + Where(max,1,Where(min,8,0))` | −400 | 13144 B (+400 **regression**) | the Where-chain (each `else`=prior plane) needs no extra Add; additive fold adds 2 planes. Cascade already minimal. |
| 3 | **U8 area** (drop the 4 F16 planes to U8) | −1600 | infeasible | exact area∈[1,400] > 255; std `Mul` on U8 **wraps mod-256** (breaks order); `QuantizeLinear`-to-U8 clamp needs the F16 product first (net 0); **QLinearMul is com.microsoft ⇒ score None**. |
| 4 | **QLinearConv bucket** (the prompt's hint) | n/a | infeasible | QLinearConv **rejects F32 graph input** (ort_reality §44); narrowing costs more than it saves; cannot express connected-region area-extrema. Prompt's floor_est ~1200 is mismatched to this task. |
| 5 | **shrink `z_f32` (1600)** | up to −1200 | infeasible | Slice of F32 input is F32; 20×20 is the minimal footprint; casting whole input first = 9000 B. Prior floor proof agrees. |

## Per-plane forced-component argument (why 12855 is the floor)
- `z_f32` 1600 (F32 input access) — forced.  `color30` 900 (30×30 output label) — forced.
- Extent 4000: width+height each need **two** directional maxpools (nearest wall on both sides;
  ONNX has no MinPool, and pos_left/pos_right ramps differ) + masked src + U8 sum. Minimal.
- Area 3200: exact product needs F16 (`neg_w`+`neg_h`+`area`=2400); `area_for_min` 800 excludes
  walls from the min (shave #1 proves load-bearing). U8 product impossible (shave #3).
- `max_mask`+`min_mask` 800 (two independent extrema), color cascade 1600 (shave #2 minimal),
  out-of-grid detect ~200, `z_bool` 400, params 111.
Sum = 12855. No plane is removable without a visible miss or a larger regression.

## One remaining hypothesis (untested, low-probability)
A radically different area encoding that fuses width×height into a single ai.onnx op producing
an order-preserving key ≤ the F16 cost. None found: the product is the only order-preserving 2D
size key over h,w≤20, and the only ai.onnx ops that widen U8→(≥16-bit) are Cast/DequantizeLinear
(one factor each). If a future session finds a fused order-key op, revisit; otherwise 12855 stands.
