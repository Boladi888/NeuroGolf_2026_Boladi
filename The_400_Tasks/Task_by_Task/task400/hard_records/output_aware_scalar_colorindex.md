# task400 dead route — "output-aware / scalar-color-index" sub-1000 patch lever

- **Live bar:** 1084 (Aditya). Source: `Team_Submission/Backend/manifest.csv`
  (`task400,Aditya,49f88a9d910b85ab712b0e76cf65e3b834b7606ea8c02f56f325a6dc6365fac4,1084,...`),
  byte-matches `Team_Best_Onnx/task400.onnx` (sha256 `49f88a9d...`), profile_real_cost = **1084**
  (static=1084, memory 1032 + params 52; 266 real inputs). visible 266/266 PASS.
- **Attempted route (prompt lever):** replace Aditya's `selected_5x5 [1,10,5,5] F32` = 1000-B
  patch with a "scalar / color-index / output-aware" representation to reach the 700–900 target.
- **Verdict: WALL. 1084 is the architectural floor; Aditya sits on it.**

## Why (forced-component argument)
Cost model (`neurogolf_utils.calculate_memory`): every node output must carry a static value_info
and is charged `elems × dtype_bytes` at `max(static, real-input-runtime)`; the graph output named
`output` is the ONLY free tensor (line 234 `if tensor_name in ['input','output']: continue`).

The task: extract a data-dependent 5×5 region (180°-mirror of the blue hole) from the F32 one-hot
input `[1,10,30,30]` and emit it as `encode_grid(5×5)` = `[1,10,30,30]` one-hot, top-left 5×5
populated, rest zero.

1. **The 5×5×10 F32 patch = 1000 B is forced.** Any data-dependent extractor (Slice / Gather /
   GatherND) preserves the F32 dtype and all 10 channels → `[1,10,5,5]` F32 = 250·4 = 1000 B.
2. **It cannot be the free `output`.** The answer must be zero-padded to `[1,10,30,30]`, and a valid
   one-hot input has exactly one channel=1 per spatial cell → **no all-zero cell exists to source the
   padding region** by slice/gather. A Pad (or Scatter-into-zeros) is mandatory, so the `[1,10,5,5]`
   patch is a counted intermediate feeding it, never the output.
3. **Cheaper representations cost strictly MORE:**
   - *pre-collapse to color-index* → `[1,30,30]` canvas = 3600 B. **Built + measured: F1
     `task400_worker2_f1_colorindex.onnx` score_v2 = 5155** (memory 5082 + params 73). LOSER.
   - *post-collapse* → still pays the 1000-B patch first, THEN Equal-expand (`[1,10,5,5]` BOOL 250 +
     index 100) ⇒ ≥1350. LOSER.
   - *cheaper dtype* → needs a Cast whose F32 source patch (1000) is still counted; casting the full
     input is 9000 (U8) / 18000 (F16). LOSER.
4. **No plumbing/param trim exists.** Hypothesized share of Slice+Pad `axes [2,3]` init (−2 params)
   fails: Pad-18 requires **INT64** axes and Slice couples starts/ends/steps/axes to one `Tind`, so
   sharing forces INT64 starts/ends → `sel_starts`/`sel_ends` grow +16 B > the 2 params saved (net
   worse). Dropping Pad axes → full `pads[8]` = 8 params > current `pads[4]+axes[2]` = 6. Worse.
   Params (ramp30=30 + sel10 + pads4 + step5·2 + steps·2 + slice_axes·2 + pad_axes·2 = 52) are at
   floor; ramp must be length H=30.

## Remaining hypothesis (only crack, judged closed)
A sub-1000 credited patch would need a data-dependent 5×5 extractor that outputs a **U8/BOOL** tensor
directly from the F32 input with no intervening F32 tensor and no full-canvas collapse. No such ONNX
op exists (Slice/Gather/GatherND all inherit input dtype). Absent a new ORT dtype fact, closed.

## Don't re-explore
Color-index canvas, GatherND meshgrid, post-collapse re-expand, axes-init dedup — all measured/proven
≥ 1084. The 700–900 prompt target was stale recon (superseded, same as the retired 3557 floor text).
