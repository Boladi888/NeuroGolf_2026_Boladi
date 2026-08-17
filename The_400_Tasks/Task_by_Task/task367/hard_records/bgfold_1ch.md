# HARD RECORD — task367 bg-only 1-channel front-end fold (DEAD)

- **Date**: 2026-07-12 · worker `bo_ng_cld_worker3` (Claude lane)
- **Live bar (sha)**: cost **13,926** (manifest, `ng_current_bar`, owner Vamshi, updated 2026-07-12T15:15Z) · `Team_Best_Onnx/task367.onnx` sha256 `f719b3a457e6aec132af040e5b898329010e991e1bdb86a5b10d56bf762ecfde`.
- **Prompt target / floor_est**: ~1,500 (fantasy for the flood family — see below).

## Attempted route
Cut the dominant front-end line `v_sl` F32[1,2,20,20]=3,200 B to **1,600** by slicing ONLY the bg channel (ch0). Input is strictly {0,5} (generator `task_e73095fd.py` uses only `black()`=0 and `gray()`=5; verified 266/266), so on every REAL grid cell gray ≡ 1−bg, and a 2-ch conv on [bg,gray] folds EXACTLY (integer pre-clamp) to a 1-ch conv on [bg]: `Wm'[c,0]=Wm[c,0]−Wm[c,1]`, `Bm'[c]=Bm[c]+Σ Wm[c,1]`. Builder: `prototypes/build_task367_bgfold_1ch.py` → `task367_bgfold_1ch.onnx` (cost 11,434, −2,492 B if it worked).

## Why it is DEAD
`ng_gate visible` = **38/266 FAIL**. The encoder (`neurogolf_utils.convert_to_numpy`) sets NO channel outside the real grid, so at non-cells (cols≥W, rows≥H, and the conv zero-pad ring) bg=0 AND gray=0 — the identity gray=1−bg FAILS there (1−0=1≠0). The fold therefore reads the **empty region as a solid gray wall**. When grids are small (W or H down to 10) that empty region is up to 10 cells wide and sits right against the real content; the corner detectors (`Wm` ch2/ch3, which have real gray taps) fire spuriously there, seeding the flood, which then propagates 5 cells into the interior and corrupts fills. Diagnosis: only **48/228** failing pairs had boundary-confined diffs — the rest were pervasive (spurious seeds flood inward). The gray-only mirror fold breaks the flood instead (empty reads as bg ⇒ flood leaks through the grid edge, un-sealing edge-clipped boxes; boxes ARE edge-clipped, cols∈[−1,width−wide+1]).

**Conclusion: the empty/bg/gray 3-state distinction genuinely needs 2 channels.** The front-end 2-ch F32 slice (3,200) is FORCED; total front-end 4,400 (v_sl 3,200 + v_u8 800 + v_z 400) is forced.

## Exact failed gate
- `ng_gate visible --task 367 --cand prototypes/task367_bgfold_1ch.onnx` → **38/266 FAIL** (did NOT run truth/ab — visible-zero family, §VISIBLE-FAIL).

## One remaining hypothesis (fragile, deprioritized)
A from-scratch **re-tuned bg-only 4-ch detector** (weights searched, not folded) that seeds correctly despite the empty/gray ambiguity — would need to reach front-end ~2,000 and re-prove 200k. Unlikely to exist: the detector genuinely needs gray corner taps (drop-one visible ch2→249, ch3→230), and a bg-only conv cannot locate walls (gray and empty both read bg=0). Not a floor route.

## What WORKED instead (this session)
Byte-equal tail/seed fusion of the live graph → **12,714** (−1,212 B), staged `staged_candidates/task367_tailfuse_byteeq_cost12714/`. See `task367.md` 2026-07-12 section.
