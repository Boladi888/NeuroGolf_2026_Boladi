# task268 — DEAD ROUTE: orientation selector prune (WALL @ 2455)

- **Live bar**: cost **2455** (memory 2399 + params 56), owner Aditya.
  - manifest.csv row: `task268,Aditya,ffc4f310eeff4185490461d25e5859ca8740fa03e4159650a36c3728dd99577f,2455,17.1941,2026-06-14T17:43:29Z`
  - Team_Best_Onnx/task268.onnx SHA-256 `ffc4f310eeff4185490461d25e5859ca8740fa03e4159650a36c3728dd99577f` (== manifest sha, verified this session).
  - Gates on incumbent this session: `visible 266/266 (100%) PASS`; `profile_real_cost --arcgen 266: static=2455 profile=2455 (memory 2399 + params 56)`; `score_v2 = (0, 2399, 56) = 2455`.
  - NOTE: supersedes the older 2456/sha `A00884889B1C...` cited in task268.md's 2026-06-12 audit; Aditya since shaved params 57→56 (one initializer element) to reach 2455.

## Attempted route
Prompt lever: "delete one orientation selector / avoid instantiating both orientation
candidates while preserving the same final decision," targeting 2350–2400.

## Why it is dead — the lever is ALREADY exploited by the incumbent
The incumbent (opset 18, 67 nodes, 8 inits) does **not** instantiate two orientation
candidates. It computes the fill geometry **once** in a canonical frame (`fill_c`, BOOL[10,10]),
where coordinates are pre-canonicalized by `Where(vertical, y*, x*)` at nodes 26–29, then:
```
fill_c_t       = Transpose(fill_c)                 # the OTHER orientation, for free
fill_xor       = fill_c XOR fill_c_t
fill_sel_delta = fill_xor AND vertical             # vertical is scalar BOOL[1,1]
fill_any       = fill_c_t XOR fill_sel_delta       # == vertical ? fill_c : fill_c_t
```
This is a BOOL-`Where`-free arithmetic mux selecting one of two orientations from a **single**
computed geometry + its transpose. There is no second geometry to delete. "Prune an orientation
selector" therefore has no target — the prune is already done.

## Exact forced-component (per-byte) argument for 2455 floor
memory 2399 breaks down as (max static/runtime per intermediate):
| bytes | tensor(s) | forced because |
|---:|---|---|
| 900 | `fill_ext` BOOL[30,30] | Final `Where(fill_ext, yellow_f32, input)` needs a condition broadcastable to the [1,10,30,30] output. A [10,10] mask cannot broadcast to [30,30] (10 ∤ 30). Building the overlay at 10×10 then `Pad`-ing the RESULT requires an F32 crop `input[:,:,:10,:10]` ≥ 4000 B and an F32 overlay ≥ 4000 B — far worse. The incumbent's genius is keeping everything BOOL/U8 and writing straight to the (uncharged) output via one `Where` on a 900-B condition. 900 is the representational floor for a 30×30 boolean fill canvas. |
| 240 | `row_sum`,`col_sum` F32[1,30] | `Einsum(input F32, nonbg_vec)` → F32 (dtype follows the F32 graph input). Narrowing to the active 10-wide needs an F32 input Slice (≥ thousands of B). Casting input to F16 first = 18 000 B. F32 marginals are forced by the F32 input. Counts (not just presence) are needed downstream by the `Gather` boundary tests, so they cannot degrade to a BOOL presence vector. |
| 900 | 9× BOOL[10,10] (`interior_c`,`beam`,`diag`,`proj_c`,`fill_c`,`fill_c_t`,`fill_xor`,`fill_sel_delta`,`fill_any`) | Geometry (5): each is a distinct binary op output; a single-`LessOrEqual` filled-wedge merge of central+diag **overfills the gap cells between the band edge (\|c−mid\|=w) and the diagonal rails (\|c−mid\|=w+2d)** → visible 53/266 FAIL (worker1 `probe_filled_wedge`). Selector (4): a 2-temp mux is impossible with binary AND/OR/XOR/NOT — `fill_any = fill_c_t XOR (vertical AND (fill_c XOR fill_c_t))` needs `fill_xor` and `fill_sel_delta` as two distinct temporaries; the `Cast→Where(U8)→Cast` alternative ties at the same byte count (worker1 `probe_u8_where_selector` = 2455). |
| 120 | `row_has`,`col_has` BOOL[1,30] + `..._u8` U8[1,30] | ORT has no BOOL `ArgMax` kernel, so the boolean presence vector must be cast to U8 before `ArgMax` (worker1 `probe_argmax_bool` opset18/20 → scorer None). Both the BOOL (from F32 `Cast`) and the U8 (for `ArgMax`) tensors are load-bearing. |
| ~239 | small U8/I8 coord & distance vectors, scalars | structural; each is a distinct op output. |

params 56 is a **hard floor**: every initializer (`row`,`col`,`col2_i8`,`nonbg_vec`,`yellow_f32`
10 each; `pad_to_external` 4; `four_u8`,`big_u8` 1 each) has N elements, and deriving it in-graph
produces an intermediate of the **same** N elements charged at the same 1-byte/elem — neutral at
best, and **strictly worse** whenever a dtype `Cast` is required (e.g. `col2_i8 = 2*col` needs
`Cast(col U8→I8)` + `Add` = +20 B to save 10 params → 2465). The one exact-tie derivation
(`col = Transpose(row)`, same dtype) saves 10 params but adds a 10-B intermediate → 2455 tie, not
a win. Closed nibbles (all documented): `Pad` attribute-form (saves 4 params but forces opset ≤10
while `Einsum` needs ≥12); `Where(BOOL,BOOL,BOOL)` (ORT/scorer reject); sparse `yellow_f32`/
`nonbg_vec` inits (scorer None); `onnxsim` (byte-identical).

## Failed/closed gate lines (this session + worker1)
- incumbent `visible 266/266 PASS`, `profile 2455`, `score_v2 2455` — bar is live & valid.
- worker1 `probe_filled_wedge`: score 2345 but `visible 53/266 FAIL` → representation dead.
- worker1 `probe_u8_where_selector`: `visible 266/266 PASS`, cost 2455 → valid TIE, never staged.
- worker1 `probe_argmax_bool` (opset18 & 20): `score_v2 None` → ORT rejects BOOL `ArgMax`.
- worker1 `probe_current_onnxsim`: byte-identical to incumbent.

## One remaining hypothesis (low confidence)
A sub-2455 win requires deleting ≥1 charged byte, which means either (a) a new final-output
family that avoids the 900-B full-canvas overlay condition without an F32 crop, or (b) a new
orientation selector that removes one 10×10 BOOL plane without BOOL `Where`, `If`/subgraphs, or
data-dependent shapes. Both were examined from first principles here and found forced; the geometry
band+rails structure and the 30∤10 broadcast gap are the hard obstacles. **Verdict: WALL.** Per
prompt ("not a floor-proving task; report the falsification and stop"), no further siege spent.
