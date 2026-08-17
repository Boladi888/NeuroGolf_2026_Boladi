# DEAD ROUTE — task205 HashMlpRuleFloor (target ~640) — 2026-07-10, bo_ng_cld_worker1

**Verdict: DROP (false floor). The charged component the attack names does not exist in the live graph.**

## Live bar (re-measured this session)
- **live_cost = 3586** (manifest, owner Aditya, updated 2026-07-03; `ng_current_bar --task 205 --json`).
- **live sha256 = 95be9d6c70cc4731bd8ae68a0a4ef72ea32c72b179e83deeeac220161721baa0** (`Team_Best_Onnx/task205.onnx`, byte-identical to OnnxlyFans backend copy).
- score_v2 recompute: memory 3316 + params 270 = **3586** (points 16.815). Graph name `task205_coarse6_rowcol_einsel`, opset-18, 101 nodes, 19 inits.

## Attempted route
Prompt = "extract generator rule and replace learned/template graph with tiny determinist;
delete charged carrier/canvas/template / dense hidden classifier weights; floor_est ~640
(= 3586/5.6 generic STRUCTURAL gap; also stale `best_ext_cost:110 kmmmvs_local_best`)."

## Why it's dead
1. **No charged component exists to delete.** The live 3586 is ALREADY the tiny-determinist
   rewrite (produced by the v18–v31 structural sessions). `largest_intermediate.py` on the
   live: biggest single tensor is `q_A30` [1,10,30,1] U8 = **300 B (9%)**; params=270; every
   remaining byte is a forced piece — output relocation (q_A30 300 + q_Ainner/q_A10 200),
   the 3-profile marginal chain (six [1,30] F32 = 720), the coarse-6 band locator
   (loc_rbands/cbands 480 + scoring). **No dense classifier, no hash-MLP, no carrier/canvas.**
2. **Fast probe killed all three independent decode formulations** (`prototypes/task205_falsefloor_probe.py`,
   fresh on the 266 corpus — numpy reference re-verified 266/266):
   - **F3 hash/LUT-gather (literal hash-MLP): DEAD** — 266/266 outputs are *distinct*
     (25 shapes, 45 color-pairs). One unique grid per input → no LUT compression possible.
   - **F1 cheap global-mode color pick: DEAD** — main_c == global input mode only 171/266
     (64%); the two rectangle colors genuinely require the 2-color largest-rectangle search.
   - **F2 single-pass theta=6 profile decode: DEAD** — 0/266 (incl. 0/4 official). Global
     row/col profiles are swamped by same-color noise cells scattered outside the rectangle;
     detection *requires* the localized coarse-6 band locator + multi-pass refinement.
3. **Forced-floor pricing (measured, not cited).** Output is rank-2 (cross = row OR col,
   non-separable) → forces the BitwiseAnd carrier ≈ **560 B** (q_A30 300 + q_Ainner/q_A10 200
   + decode). Even a single detection Einsum (~120 B) on top already exceeds 640, and
   single-Einsum/single-pass detection falsifies (F1/F2). Shippable wall = ~3,000 (three
   pins, `task205.md §<3,000 WALL`); absolute zero-robustness ≈ 1,260 and even that falsifies
   on officials. **640 sits below every measured floor.**

## Live class note
Live 3586 is PROBE-class (profile/marginal family, v30/v31 lineage: ~99.85% corpus, NOT
byte-equal — byte-equal needs the 10,132 lattice v32). Team already shipped a filtered-clean
probe here. So this is not even a byte-equal rung to shave; a `rung - epsilon` shave is
explicitly out of scope for this false-floor attack (workflow-update 2026-07-10).

## Pass-2 empirical confirmations (2026-07-10, forced second pass — build/gate push)
Mined `techniques.md` #73 (task205 IS the canonical *locate-then-detect + separable-bank +
marginal-profile* family; the marginal-profile family is CHEAPER than the locate-then-detect
LATTICE it replaced — lattice was 9,345–10,132, so reverting is a regression) and #74
(3-operand free-input Einsum verified; U8/INT8 [1,30] lanes = 30 B). Then verified, empirically:
- **F32-pin verified structurally.** Dependency scan: all six [1,30] F32 marginals
  (`p_winc_f, p_rowcnt2, p_exr0_f, p_colcnt3, p_exc2_f, p_rowcnt3`) are Einsum I/O — either an
  Einsum output (dtype forced to F32 by the F32 `input`) or an Einsum operand (must match F32).
  The #74 U8-lane lever does NOT apply; narrowing any of them needs an `input` cast ≥9,000 B.
  → the 720 B marginal chain is forced.
- **Peephole-saturated.** `peephole_rewrite.py --task 205 --check --source Team_Best_Onnx/task205.onnx`
  → "no pattern matched" (const_to_init / add_chain_to_sum / drop_cast_before_pad_bool / … all
  already applied). No byte-equal cut remains.
- **Live is a PROBE, not byte-equal gold.** `ng_gate truth --task 205 --cand <live> --n 20000`
  → **19909/20000 = 99.545%** (91 div). Team already ships a filtered-clean probe here. So there
  is no byte-equal rung to shave, and dropping robustness further hits pin 1 (2.5/single-pass →
  official pair 96 diverges → certain hidden 0).
- **No buildable candidate below 3586 exists on any route I could find.** Full-battery/profile
  was therefore NOT run on any candidate (nothing cleared the static bar — running it would be
  pure burn per gate hygiene). The one gate run was the sanctioned §CLASS-CHECK truth on the live.

**Verdict upgraded DROP → WALL.** Cheapest correct formulation found = the live probe at 3586.

## One remaining hypothesis (for a future batch, LOW value)
Between live 3586 and the ~3,000 three-pin wall there is ~586 B of *probe-family shave* room
(e.g. fold q_Ainner/q_A10 channel-select, U8-narrow the [1,30] F32 marginals if saturation-safe).
That is a `rung - epsilon` shave of a probe, NOT a floor route to 640, and the workflow-update
says stop on the priced wall rather than grind it. Do NOT re-open the 640/hash-MLP route.
