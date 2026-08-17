# task265 — compacted history (moved from task265.md 2026-06-07; pre-v3 era)

## Algorithm sketch (LIVE-original, superseded by the decoded Rule + v1/v2/v3 rewrites)

```text
1. Gather inner 18x18 region of input (input[:, :, idx, idx] with idx=v0 [18]).
2. ch0 = inner[:, 0]; ch_color = inner[:, color_index].
3. Conv(ch0, 2x2 kernel) → sums of 2x2 windows.
4. (sum - target)^2 < eps → 0/1 mask of windows that match a "2-wide strip".
5. ConvTranspose(mask, 2x2 kernel, stride 1) → expand matches back to 18x18.
6. Greater 0 → bool mask of cells to recolor (= `v24` morph-open candidate).
7. Two 5x5 Convs (`v6`,`v7`, complementary hole/5 patterns at two vertical
   offsets) match EXACTLY (`Conv==25`) → EXCLUDE mask `v39` that thins 3-wide
   protrusions to the isolated 2-wide strip. fill = `v24 AND NOT v39`.
8. Compose final output via Where + Concat (10 channels) + Pad to 30x30
   (v11=0 so channels 1,3,4,6,7,8,9 are zero; ch0=holes−fill, ch2=fill, ch5=5s).
```

## v_w8a (2026-05-17) cost analysis — the 76,100 fp16-retype era

v_w8a memory split (memory=75,969; params=131):
  - Input Cast(fp32→fp16) intermediate [1,10,30,30] fp16: 18,000 bytes (the
    main fp16-retype overhead; unavoidable since the model input is fp32).
  - Per-channel Gather outputs [1, 10, 18, 30] fp16 (10,800 bytes) and
    [1, 10, 18, 18] fp16 (6,480 bytes).
  - Conv/Sub/Mul/Less chains operating on [1, 1, 17, 17] or [1, 1, 18, 18]
    fp16 tensors at ~600 bytes each.
  - Final Concat-then-Pad output assembly.

The fp16 retype halves all spatial intermediates but adds the 18,000-byte
input Cast. Net savings: ~2.4k, well under what task013 (similar approach)
achieved because task265 has far fewer and far smaller spatial intermediates
to begin with (31 nodes vs 165).

## Don't-do bullets from the 56k/76k era (superseded context, kept verbatim)

- **Do NOT** assume fp16 retype always saves cost. On a small graph with
  only 31 nodes and modest fp32 intermediates, the input fp32→fp16 Cast
  (18,000 bytes) consumes most of the savings. v_w8a still nets +0.031 pts
  but it's borderline. For dramatic wins, fp16 retype works best on
  graphs with many large fp32 intermediates (e.g. task013's 165-node
  pipeline saved ~99k of memory).

### 2026-05-20 — PR_HUGE worker (attempt failed)
- Tried: f16 conversion (both robust onnxconverter_common and direct manual converter)
- Result: NO CANDIDATE
- Reason: f16 conversion failed - either type mismatch or no benefit

## Retracted floor proofs (2026-06-01, deleted from task265.md 2026-06-07 per protocol)

Summary only (full text in git history at commit `6fa6bcdd` and earlier): the
"byte-equal floor ≈ 11,237" and "SUB-5k IS STRUCTURALLY IMPOSSIBLE" proofs rested on
(a) Conv-class ops forced ≥F16 (§8f) and (b) a mandatory Pad+Equal output tail
(900+324). Both premises fell on 2026-06-07 to QLinearConv-u8 with int32-bias
thresholding (#74/#77) and QLC-as-output free pricing (#78/task125); v3 reached
5,107 byte-equal-SAFE. The component that SURVIVED: the 6,208 morph_open-only class
is unsafe (template exclude mandatory; public-confirmed zero).

---

## WRONG-GENERATOR era (0a938d79, 5×5-template decode — HISTORICAL; applies to v1-v4; archived 2026-07-12)

**Two-component rule (= LIVE exactly, WRONG GENERATOR):**
```
fill   = morph_open(zero_set, 2x2 SE)   AND NOT   template_5x5(holes, 5s)
out    = 5 where 5 ; 2 where fill ; 0 where (hole AND NOT fill) ; (nothing at padding)
```
1. **morph_open** (`v24`): a hole is candidate iff it lies in a 2×2 all-zero window (erode by 2×2 `Conv`==4, dilate back by 2×2 `ConvTranspose`>0). Over-fills 3-wide protrusions.
2. **template exclude** (`v39`): two exact 5×5 templates (`v6`,`v7` = same pattern at two vertical offsets, `v7=shift_up(v6)`) subtract the redundant 3rd lane of a 3-wide protrusion. Each fires iff `Conv(concat(ch0,ch5),w)==25`.

**Critical finding — morph_open ALONE is WRONG and the corpus can't see it:** truth 200k/200k but visible 265/266 (hand-crafted protrusion pair, cells (8,12),(9,12)); deterministic ⇒ one visible miss = 0; public-confirmed (6,208 probe → 6428.81). Ship class = byte-equal two-component ONLY. Superseded by the correct `a8d7556c` gapfiller rule + v5-v9.

### Pre-v5 attempts (wrong-generator family)
| Date | Worker | Approach | Result | Cost | Notes |
|------|--------|----------|--------|------|-------|
| LIVE | (baseline) | Gather/Conv/ConvTranspose/Where, fp32 | 266/266 | 78,484 | LIVE==CURRENT |
| 2026-05-17 | worker-8 | v_w8a generic fp16 retype | 266/266 + adversarial 100% | 76,100 | −2,384 vs canonical |
| 2026-06-01 | DriveToFloor | morph_open-ONLY ×3 (F32/F16/MIXED) | truth 200k BUT visible 265/266; Kaggle 6428.81 | 6,208 | UNSAFE, public-confirmed zero |
| 2026-06-01 | DriveToFloor | byte-equal LIVE port (morph_open+2×5×5 templates), full-F16 | 266/266 + ab 200266 byte_equal + Kaggle 6444.48 | 11,237 | SHIPPED FLOOR (+1.62) `v1` |
| 2026-06-01 | DriveToFloor | v2 single-channel template (`Conv(ch0,Vk_ch0−Vk_ch1)==13` + rowmask) | 266/266 + ab 200266 byte_equal + 2592/2592 | 9,584 | SAFE FLOOR sub-10k, KAGGLE-CONFIRMED (+0.16) `v2` |
| 2026-06-01 | DriveToFloor | F32 port | byte-equal 200266 | 15,633 | `f32port` fallback |
| 2026-06-01 | DriveToFloor | morph-F16/template-F32 | byte-equal | 14,477 | `morphF16` middle |
| 2026-06-07 | DriveToFloor | v3 QLC-u8 + free-output conv | battery gold | 5,107 | superseded by v4 `v3` |
| 2026-06-07 | DriveToFloor | v3b safe-tail sibling | full battery + profile 6,844 | 6,844 | `v3b` |
| 2026-06-07 | DriveToFloor | v4 = v3 + template-CROP [1,1,14,14] | 266/266 + ab 200266 byte_equal + profile 4,713 | 4,713 | SAFE FLOOR (wrong-rule family) `v4` |

v4's 4,713 floor proof was CONFIRMED for the wrong-generator 5×5-template family (2026-06-12 opus floor-attack), then made obsolete when the correct `a8d7556c` generator was decoded (v5 = 3,791, pure bitwise, no templates).
