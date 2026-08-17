# task212 hard record - row-basis drop/refit route

Worker: `bo_ng_worker1`
Date: 2026-07-14

## Live bar used

- Source: `python tools/ng_current_bar.py --task 212 --json`
- Authoritative source: `Team_Submission/Backend/manifest.csv`
- Team-best owner: Vamshi
- Cost/profile cost: 412/412
- SHA-256: `f5303effda475011070aa0ce6c0db01a4b934d2d788cc012c7b520a0069a6529`
- Baseline gates:
  - `score_v2`: `(0, 0, 412)`
  - `profile_real_cost`: `task212 task212.onnx: static=412 profile=412 (memory 0 + params 412; 265 real inputs)`
  - `visible`: `[visible] 265/265 (100.00%)  PASS`

## Attempted route

The current team-best is a zero-memory one-node `Einsum`; cost is entirely dense initializer
params:

- `row_basis` F32 `[7,30]` = 210 params
- `side_coeff` F32 `[3,4,7]` = 84 params
- `chan` F32 `[4,10]` = 40 params
- `rule` F32 `[4,3,2]` = 24 params
- `ain`/`aout` F32 `[4,4]` each = 32 params total
- `feat` F32 `[2,7]` = 14 params
- `coef` F32 `[2,2,2]` = 8 params

Probe route: remove one of the 7 shared row-basis modes, then least-squares refit every affected
`side_coeff[...,7]` and `feat[...,7]` coefficient vector against the remaining six basis rows. This
would price at 368 params if it worked (`row_basis` 180, `side_coeff` 72, `feat` 12; net -44).

## Exact failed gate

In-memory visible precheck over the official 265 examples, using the official one-hot encoder and
`>0` decoder. No ONNX candidate was written because every variant missed the first visible example.

Command shape:

`drop d -> row_basis = delete(row d); side_coeff/feat refit by pinv(new_basis); ORT run from serialized model bytes`

Results:

- drop 0: prospective params 368; prefix pass `0/265`; first example had 244 tensor-cell mismatches
- drop 1: prospective params 368; prefix pass `0/265`; first example had 233 tensor-cell mismatches
- drop 2: prospective params 368; prefix pass `0/265`; first example had 245 tensor-cell mismatches
- drop 3: prospective params 368; prefix pass `0/265`; first example had 158 tensor-cell mismatches
- drop 4: prospective params 368; prefix pass `0/265`; first example had 228 tensor-cell mismatches
- drop 5: prospective params 368; prefix pass `0/265`; first example had 244 tensor-cell mismatches
- drop 6: prospective params 368; prefix pass `0/265`; first example had 42 tensor-cell mismatches

The active-grid row basis has full rank 7 over rows 0..9 and over 0..29, so this failure is structural,
not a bad least-squares target choice.

## Why this route is dead

The 7 row modes are jointly load-bearing for the separator-ray comparator. Dropping any one mode and
refitting the current coefficient family destroys visible correctness immediately. Under the worker
protocol, a visible miss is a certain hidden 0, so no `truth`, `ab`, profile, safety, or staging gate was
run for this family.

## One remaining hypothesis

A real sub-412 win must change the tensor-network family, not delete one of the current shared row
basis modes. The most plausible remaining direction is a new sign-coded comparator core that replaces
the shared `row_basis`/`side_coeff` pair with a different factorization; sparse storage, U8 CumSum,
label-tail, and simple row-mode deletion are now all ruled out locally.
