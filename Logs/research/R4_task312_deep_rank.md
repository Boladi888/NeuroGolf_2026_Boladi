# R4 task312 deep selector-bank rank harvest

Date: 2026-06-08

## Scope

Continuation of R2 for task312 only. Goal was to harvest any exact selector-bank rank cut below the current rank-18 signed CP bank in `task312_rankcut_cost570`, using the M2 OR-Tools CP-SAT recipe.

No candidate was submitted. No sub-18 exact witness was found, so no rebuilt ONNX was staged.

## Setup

- Installed `ortools==9.15.6755` into `.venv` with `python -m pip install ortools`.
- `protobuf` is now `6.33.6` in the venv because it is constrained by OR-Tools.
- Runner added: `tools/scripts/task312_rank_cpsat.py`.
- Solver encoding:
  - exact integer CP-SAT variables;
  - table-encoded pair/triple products;
  - first-nonzero-positive normalization on A/B;
  - lexicographic support-code ordering;
  - explicit `|prefix| <= 2^24` accumulation guard;
  - zero-term padding enabled, so a rank-17 INFEASIBLE result proves no rank `<=17` solution for that exact alphabet/model.

## Bank Extraction

Source checked: `staged_submits/_shipped_archive_20260607/task312_rankcut_cost570/task312.onnx`.

The current graph stores three `[10,18]` factors, 540 factor elements, alphabet `{-1,0,1}`. Reconstructing

```text
einsum("kr,cr,dr->kcd", U, V, W)
```

matches the task312 selector bank exactly.

Effective bank:

- shape: `[10,10,10]`
- nonzero entries: `100`
- current bank factor elements: `540`
- current total rank-family cost: `570 = e0[30] + 3*10*18`

Flattening-rank kill-test:

| mode | rank |
|---|---:|
| k x (c*d) | 10 |
| c x (k*d) | 10 |
| d x (k*c) | 10 |

Verdict: flattening rank does not kill ranks 13..17. It proves only the unconditional lower bound `rank >= 10`.

## CP-SAT Results

`new_total_elements` includes the unchanged `e0[30]`. `point_gain` is `ln(570 / new_total_elements)`.

| T | alphabet | dictionary | result | time | new_bank_factor_elements | new_total_elements | point_gain | artifact |
|---:|---|---|---|---:|---:|---:|---:|---|
| <=17 | `{0,1}` | N | INFEASIBLE-in-alphabet | 2.14s | 510 | 540 | +0.054067 | `Logs/research/r4_task312_rank17_01.json` |
| <=17 | `{0,1,2}` | N | INFEASIBLE-in-alphabet | 3.61s | 510 | 540 | +0.054067 | `Logs/research/r4_task312_rank17_012.json` |
| <=17 | `{-1,0,1}` | N | OPEN-timeout | 901.53s / 900s | 510 | 540 | +0.054067 | `Logs/research/r4_task312_rank17_signed01.json` |
| <=17 | `{-1,0,1}` | Y | OPEN-timeout | 301.49s / 300s | 510 | 540 | +0.054067 | `Logs/research/r4_task312_rank17_signed01_dict.json` |
| <=17 | `{-2,-1,0,1,2}` | N | OPEN-timeout | 305.18s / 300s | 510 | 540 | +0.054067 | `Logs/research/r4_task312_rank17_signed012.json` |
| 16 | `{-1,0,1}` | N | OPEN-timeout | 300.97s / 300s | 480 | 510 | +0.111226 | `Logs/research/r4_task312_rank16_signed01.json` |
| 15 | `{-1,0,1}` | N | OPEN-timeout | 301.09s / 300s | 450 | 480 | +0.171850 | `Logs/research/r4_task312_rank15_signed01.json` |
| 14 | `{-1,0,1}` | N | OPEN-timeout | 300.86s / 300s | 420 | 450 | +0.236389 | `Logs/research/r4_task312_rank14_signed01.json` |
| 13 | `{-1,0,1}` | N | OPEN-timeout | 300.79s / 300s | 390 | 420 | +0.305382 | `Logs/research/r4_task312_rank13_signed01.json` |

Notes:

- The two nonnegative passes are exact certificates for their stated alphabets. Because the model permits zero-term padding, rank `<=17` infeasibility covers ranks 13..17 for `{0,1}` and `{0,1,2}`.
- These are not real-rank lower bounds and do not disprove signed/dyadic decompositions.
- The signed dense searches did not find a witness and did not prove infeasibility. Label remains OPEN.
- The structured dictionary used 40 A atoms, 40 B atoms, and 169 C atoms: zero, basis, ones, not-5 class indicator, with-5 pair masks, cyclic pair/triple masks, signed variants for C, and ordered basis differences for C. Its timeout is dictionary-scoped only.

## Rebuild/Gates

No FEASIBLE sub-18 witness was produced. Therefore:

- no rebuilt `task312.onnx` was created;
- no `staged_submits/task312_rankNN_cost<C>/` directory was created;
- gates were not run for a new candidate.

The existing rank-18 candidate remains the best exact bank known:

- staged archive: `staged_submits/_shipped_archive_20260607/task312_rankcut_cost570/`
- current rank: 18 signed CP
- cost: 570
- prior gates from manifest: visible PASS, truth full PASS, ab full byte-equal, `score_v2=570`, real-input profile `static=570 profile=570`.

## Final Verdict

Lowest exact rank found remains 18. R4 proved no rank `<=17` factorization over nonnegative `{0,1}` or `{0,1,2}`. Signed rank 13..17 is still OPEN under OR-Tools CP-SAT: the flattening lower bound is only 10, and the signed/dictionary/wider signed runs timed out without a certificate. There is no captured point gain and no staged candidate from this card.
