# Hard record — HashMlpRuleFloor (~900) is a FALSE FLOOR; fast-probe DROP (bo_ng_cld_worker2, 2026-07-10)

## Live bar / state at time of record
- Authoritative bar (`python tools/ng_current_bar.py --task 157 --json`): **LIVE Sho cost 6237**,
  sha `ba3c4625ec637934c21e6ed0a15ef238cab454444718810e11e012041edabcfd` (manifest, Sho, 2026-07-09).
- **Standing win already staged** (neurogolf-worker-7 run-2, earlier today): `staged_candidates/task157_cheapclamp_tail2ch_cost6008/`,
  cost **6008**, sha `085a8d5f5ac24623985183ae6f21e96bd9b689a5138e4769ceef95117835fd83`, **−229 vs bar**.
  Re-verified this session: visible 265/265; profile static=6008 profile=6008; ab-skip-crash 2252/2252
  byte-equal on all LIVE-runnable boards, 0 candidate crashes on the 13 LIVE-crash boards; ng_safety WARN-only (inherited QLinearConv x86).

## Assigned route (prompt Direction)
"hash-MLP / dense classifier decode: extract generator rule and replace learned/template graph with tiny
determinist. Delete charged carrier/canvas/template." Claimed floor ~900 (gap 6.93× from 6237).

## Fast probe — the charged component is ALREADY GONE (workflow-update kill condition met)
The prompt's charged component is the OLD Aditya/85241 dense template projection: F16 `v8` [265,75] (19,875 params)
+ F16 `v6` [14,265] (3,710 params). Fingerprint of the **current** LIVE `Team_Best_Onnx/task157.onnx`
(`inspect_graph.py`, opset 18, 385 nodes):
- **NO MatMul. NO dense F16 template.** Largest initializers: `zero_place_flat` BOOL[90], `base_active6` UINT8[6,15]=90,
  decode conv `dec_w` INT8[10,2,1,1] = **20 params**, `place_offsets_4x4` INT32[4,4]=16.
- Op signature: Gather50 / Concat43 / And40 / Cast35 / Add30 / Greater29 / … / ArgMax15 / TopK2 / ScatterElements4 / QLinearConv1.
- The current live graph is ALREADY the from-scratch structural footprint-matcher the prompt asked to build:
  TopK/ArgMax over the bottom staging region → per-object col-guarded ScatterElements into a flat [90] canvas →
  Pad → Where-overlay onto a static per-row background → **tiny 20-param QLinearConv** decodes {c,c²}→10-ch one-hot.

Per the 2026-07-10 workflow rule ("Confirm the live/team-best graph still pays the charged component named in
Direction/Fast probe; if it is already gone … return DROP"): **the dense classifier/template carrier does not
exist in LIVE. Route = DROP.**

## Why ~900 (and any sub-6008) is unreachable truth-safe
1. **The only sub-6008 numbers are a truth-unsafe learning artifact.** The lookup/template family that reaches
   4808 / 3581 (docs' scout/stretch) memorizes the 265 official templates: `task157_key_holes_tail_bool3_u64_port`
   (cost 3581) passes visible 265/265 but **truth 261/20000 = 1.305%** and ab-vs-team-best byte_equal=False.
   Collapses on the 200k ARC-GEN corpus (recorded batch42 `sub6470_vamshi_active_bool_floor.md`,
   batch46 `sub6470_direct_equal_activemask_infeasible.md`). A ~900 hash-MLP would be an even purer memorizer → hidden 0.
2. **A truth-safe matcher can't reach the target pre-render.** The key-extract + assignment scorer alone (no
   placement expansion, no Scatter, no render) already costs **4619** (`footprint_assign_only_pre_render_probe.md`,
   2026-07-07) — above even the prompt's ~3200 pre-render kill threshold, let alone 900.
3. **The truth-safe representation floors at 6008.** Per-tensor profile of the 6008 graph — every large tensor is
   ORT-forced: 300B `acg2ch` (2-ch {c,c²} conv carrier; a *linear* conv decoding 3 colors {0,1,2} needs both c and
   c² = 2 channels, provably minimal), 240B `bottom5_4d` + 120B `top0_4d` (F32-input Slices stay F32 — cast-full-input
   loses), 150B `ch1_active_pad10` placed-creature plane, 5×90B scatter/canvas BOOL planes, ~11×64B int32 scatter-index
   chain (ScatterElements needs rank-1 [64]; int32 is the index-dtype floor). Run-2 already took the two clean levers
   (cheap clamps +83 vs +275; 2-channel decode tail −310). No dominant reducible tensor remains.

## Exact gates that would fail the assigned route
- Any hash-MLP/lookup port to ~900: **truth ≪ 99%** (u64 port precedent 261/20000 = 1.305%), **ab byte_equal=False** →
  learning artifact per traps §14e; hidden 0. Do NOT run truth/ab/battery on a fresh port — the precedent already priced it dead.

## Independent from-scratch formulations considered (all worse or truth-unsafe)
| Formulation | Cost | Verdict |
|---|---|---|
| Hash-MLP / dense-classifier lookup (assigned) | ~3581 best | truth 1.3% — learning artifact, DROP |
| Assign-only matcher (no render) | 4619 pre-render | can't reach ~3200/~900, dead |
| Structural enumeration / exact-cover | 37,310 | truth-safe but 6.2× worse, dead as cost |
| Vamshi active-BOOL lineage | 6470 | worse than 6008; different output repr |
| Active-BOOL on Sho core (Path-B) | ≈6008+ | no cost win (still 2 dyn channels ≈300B + dyn wall-hole plane), breaks byte-equal, truth risk |
| **Sho-lineage byte-equal (current best)** | **6008** | forced-component floor; both clean levers taken |

## Verdict: DROP the HashMlpRuleFloor route. 6008 stands as the truth-safe floor.
The prompt's charged dense-classifier component is already deleted from LIVE; the only route below 6008 is the
truth-unsafe memorization family. Nothing new staged; the 6008 win is preserved and re-verified.

## One remaining hypothesis (low confidence, not pursued)
A ground-up graph that FUSES the placement scatter directly into the QLinearConv input (eliminating the separate
150B `ch1_active_pad10` materialization by scattering straight into a [2,10,15] carrier) *might* shave ~150B, but it
rewrites the scatter/decode core and near-certainly breaks byte-equality on the ambiguous cases (task157 is
ambiguous-class, LIVE truth 98.83%, byte-equal-vs-LIVE is the gold gate). High risk for ≤150B on an already-staged
−229 win. Not worth a siege per the workflow "one crack, stop on a priced wall" rule.
