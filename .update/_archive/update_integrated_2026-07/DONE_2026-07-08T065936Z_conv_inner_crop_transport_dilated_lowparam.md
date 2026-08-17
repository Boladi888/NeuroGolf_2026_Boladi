# Transferable insight — Conv-inner transport breaks the "full color-index canvas is forced" crop floor; dilate a single-tap Conv to golf its params (the LEGAL alternative to dead sparse-init)

**Source**: task029 batch 54 (`bo_ng_cld_worker2`, 2026-07-08). Beat the banked bar 8,481 honestly: v26 (prior, undocumented) 8,319 → **v28 7,954**, both byte-equal to LIVE (full A/B 200266/200266), profile == static (gap 0%). Stage `The_400_Tasks/Task_by_Task/task029/staged_candidates/task029_v28_dilated_conv_lowparam_cost7954/`. Builder `.../prototypes/build_task029_v28_dilated_conv_lowparam.py`.

**Task shape**: find a data-dependent rectangular region in a `[1,10,30,30]` one-hot input and emit its interior colors as a top-left-placed `[1,10,30,30]` one-hot (the classic "zoom-box / marker-crop, place at origin" ARC pattern; 92+ Object_Based crop tasks, e.g. listed-similar task059).

## Finding 1 — Conv-inner transport deletes the full color-index canvas (the BIG one)

The task029 floor proof had been walled at **9,510** for weeks with a written "irreducible" claim: *"forced honest tensors = Einsum colour canvas F32 [1,30,30] 3,600 + U8 canvas 900 + ..."*. **That 3,600+900 is NOT forced.** To turn a `[1,C,H,W]` one-hot into a `[1,1,H',W']` color-index over a *fixed sub-window*, one Conv does channel-collapse AND crop in a single op:

```python
# read color-index of the fixed inner window rows/cols a..a+S-1, in ONE op
conv_w = np.zeros((1, C, K, K), np.float32)
for color in range(C):
    conv_w[0, color, tap_r, tap_c] = float(color)   # single spatial tap = channel index
inner_f = Conv(input, conv_w, pads=[0,0,0,0], strides=[1,1])   # -> [1,1,S,S] F32 color-index
```

Output spatial `S = 30 - eff_kernel + 1`; the nonzero tap position picks the read offset. This replaces the whole `Einsum[1,30,30] F32 (3,600) + Cast U8 (900)` canvas with just `inner_f [1,1,S,S] F32` — you never materialize the full-grid canvas. task029: **9,510 → 8,319** (−1,191), still byte-equal. Works because the interior always fits a bounded window (here ≤24×24; grids ≤25×25 so marker bbox ≤ row/col 24 — VERIFIED 0/266 violations). The dynamic Slice/Pad/Equal tail is unchanged.

## Finding 2 — golf a single-tap Conv's params by DILATION, not sparse

The single-tap kernel is almost all zeros, so it's tempting to shrink params. **Sparse init is DEAD** (see `DONE_task326_sparse_init_banned_2x2crop_floor.md`: `calculate_memory` runs `check_model(full_check=True)` which rejects a sparse tensor feeding any op → `score_network`=None → INVALID). The legal lever is **dilation**:

- Dense `[1,10,8,8]` kernel, tap at [1,1] → reads offset-1, output 23×23, **640 params** (= 10·8·8, 630 of them zeros).
- Dilated `[1,10,2,2]` kernel, `dilations=[6,6]`, tap at [0,0] → eff-kernel 7, output 24×24, **40 params**, reads offset-0.
- Trade: −600 params, +235 B memory (24×24 vs 23×23 F32+U8). **Net −365 B** → task029 **8,319 → 7,954**.

General rule: eff-kernel = `dilation*(K-1)+1`; output `= 30 - eff-kernel + 1`; taps sit at multiples of `dilation`. Pick the smallest legal window that still contains every possible target pixel — offset-0 (K=2, chosen dilation) costs `C·4` params; offset-1 minimal window needs contiguous taps (K=8, `C·64` params). The extra 1-row window almost always beats 600 params.

## Finding 3 — the banked cost for dynamic-crop tasks is a 4-input MIRAGE (source-confirmed)

Reconfirmed by source-read of `data/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py`: `verify_network` runs `verify_subset(session, examples["arc-gen"])` through the SAME profiling session **before** `score_network`, and `calculate_memory` takes `max(static, observed)` per tensor across the whole trace. So the official banked cost **includes arc-gen**. task029 LIVE (SHA `DBE79C…`) is banked 8,481 = its 4-input (train+test only) profile; its real arcgen-266 profile is 10,061 (static 8,078). Any dynamic-shape candidate must be ranked by `profile_real_cost.py --arcgen 262/266`, never the 4-input default. v28 is fully static (profile==static==7,954) so it banks honestly and beats both the mirage (−527) and LIVE's real cost (−2,107).

## What's missing from the library

I did not find the "single Conv = channel-collapse + fixed-window crop in one op" transport documented as a named technique (grep of `op_cost_patterns.md` / `techniques.md` for single-tap/conv-color-index surfaced only adjacent material). Suggest adding it alongside the crop-floor material in `op_cost_patterns.md` / `cost_floor_reference.md`, cross-linked to `DONE_task326` (sparse-init-dead) and `DONE_task262` (charge asymmetry: params=element-count, memory=bytes). The dilation-for-param-golf trick generalizes the "reduce a single-tap conv" case and is the sanctioned substitute now that sparse is closed.

**Remaining forced floor (task029 byte-equal-LIVE) = 7,954 (WALL)**: 2× per-color `[1,10,30]` F32 marginals (2,400, per-color complete-border detection is intrinsically per-color; 264/266 examples have exterior distractor pixels so all-non-bg bbox is invalid) + dilated Conv inner F32 (2,304) + inner/crop U8 (576+529) + Pad U8 [1,1,30,30] (900) + selector ints/params (~845). The two F32 marginals + the 900 Pad tail are the last irreducibles.
