# task209 — ConvTranspose "block-expand render-dissolve" ceiling: it dissolves the WRITE side only, not the READ-plane matcher (floor confirmed ~4.5k, sub-2k unreachable in class)

## Where
- task209 (GuideBlockExpandConvT). Builder: `The_400_Tasks/Task_by_Task/task209/prototypes/build_task209_compress_v14.py`.
- Staged best (re-gated this session): `staged_submits/task209_worker209_i8constraints_cost4922/task209.onnx`, cost **4,922** (mem 4,733 + 189p), SHA `9ba9e69f70e6070da4d9050bf2b715c6758e6972e23e9e919b659e2fabb7ea5a`. Lane team_best = 4,995 (LIVE v13), so v14 already wins +73B.

## The rule (clean fresh decode, 266/266 reproduced in numpy)
4 yellow (color-4) pixels mark a crop rectangle = output size H×W. A small multi-color "key" sprite sits OUTSIDE the crop. Output = nearest-neighbor upscale of the key by N (N ∈ {2,3,4} ONLY, corpus-wide: dist 2:71, 3:162, 4:33 over 266), placed at an offset inside the crop, corners restored to 4. The in-crop "given" blocks are a SUBSET of the upscaled sprite; offset+N are recovered by sliding the upscaled key so every given block agrees (consistency match). Simple "offset = in-crop bbox top-left" FAILS (180/266) and "N = first-block run + key-firstcell align" FAILS (156/266) — because the topmost/leftmost given block need not be the topmost/leftmost key cell. That mismatch is exactly why a full moment-matcher is structurally required.

## The transferable insight (the prompt's angle was half-right)
The drive-to-floor prompt proposed: make the upscale ONE grouped `ConvTranspose(key_onehot, ones[C,1,N,N], stride=N, group=C)` to "dissolve BOTH the 648B nibble entry plane AND the 640B Gather+ScatterND render pair." **ConvTranspose (and DepthToSpace #80, the cheaper no-F32-weights twin) only dissolves the WRITE side (the render).** It cannot touch the 648B F32 entry plane, because that plane is the READ side feeding the matcher — every matcher Gather reads it to recover N/offset from the partial given blocks. ConvTranspose is a renderer, not a feature extractor. Conflating read-plane and write-render is the trap.

## Quantified floor (per-tensor, 080/254 standard)
Total 4,922 = mem 4,733 + 189p. Structurally pinned:
- **900B U8[30,30] Pad output** — #81 PAD-FLOOR, killed on 3 grounds (translated crop / multi-color fills / out-of-region leak). Do not re-attack.
- **648B F32[1,1,18,9] conv entry plane** — F32-pinned: anything read directly off the F32 graph input materializes F32 first. F16 Conv works in ORT but needs an F16 cast of the [1,10,30,30] input = 18,000B (catastrophic). Read plane, not renderable away.
- **240B = 2× F32[30] yellow marginal Einsums** — same F32-input pin; a cast to F16 ADDS bytes, the F32[30] survives.
- **Render WRITE side ~1,279B** (canv0 320 + ScatterND 320 + spcw 100 + index lanes ~400 + corners I64 ~80). An IDEAL DepthToSpace/ConvTranspose render still needs: a [16,20] U8 crop (320B, unavoidable — it's what gets Pad→Equal), data-dependent placement shift lanes, AND a non-separable corner-4 scatter (corners need r AND c at extremes simultaneously, so a separable row×col sentinel cannot isolate just the 4 corners — ScatterND is forced). Realistic net render-dissolve save ≈ 400-600B → floor ≈ **4,322-4,522**, NOT <2,000.
- **Matcher ~2,000B** (sections C-H: extents, anchor, mag-probe, 3×6 moment match, 15-cell hypothesis lattice) is irreducible: the given blocks are an arbitrary subset of the upscaled sprite, so recovering (N, irow, icol) genuinely needs the moment certificate.

## Verdict
Sub-2,000 is **not reachable in this rule class.** The <2k hope in the prompt rested on ConvTranspose dissolving the read plane, which it cannot. Formulation floor ≈ 4.3-4.5k (render-dissolve) down to the matcher floor ~4.7k as-built. v14 (4,922) is the cheapest credible gate-passing rung and already beats lane team_best 4,995.

## Library gap
`techniques_catalog.md` #80 (DepthToSpace block-stamp) and #129 (GatherND sub-shape replication) are both WRITE-side primitives. Worth a one-line caveat in #80/#129 (or by_symptom.md): "block-expand renderers dissolve the WRITE side only — if the rule needs N/offset RECOVERED from partial given content, the read-plane + matcher remains the floor; do not price the render-dissolve as if it removes the feature plane."
