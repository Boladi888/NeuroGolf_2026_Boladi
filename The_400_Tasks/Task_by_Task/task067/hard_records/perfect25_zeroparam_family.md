# DEAD ROUTE — task067 perfect-25 zero-param (cost<=1) family

- **Live bar:** cost **47**, SHA `4e3a67d23c2347daefc176985ee779797cdb2fd159f7a652d0c866f84ac26c71`
  (manifest, owner Aditya, 2026-06-19). profile_real_cost confirms static=profile=47 (mem 39 + params 8, 266 real inputs).
- **Attempted route:** the prompt's "ZeroopsweepCeilingAudit" — a no-payload direct-output op at cost<=1
  (Identity, Transpose perms, attr-only final op) that IS the ARC answer.
- **Failed gate:** `ng_gate visible --task 67`
  - `p25_identity.onnx`      → `[visible] 0/266 (0.00%) FAIL`
  - `p25_transpose_hw.onnx`  → `[visible] 0/266 (0.00%) FAIL`
- **Why it's dead (forced-component proof):** generator `task_2dee498d.py` proves the input is three
  side-by-side `size x size` blocks (left | middle[±vflip] | right=copy-of-left), `W = 3*size`, and
  **output = the LEFT block = leftmost `W/3` columns** (`output[r][c] = grid[r][c]`, `c<size`). The output
  is a PROPER SUB-CROP that differs from the input on 2/3 of active columns, so NO zero-initializer,
  attr-only op maps input→left-third:
    - Identity emits all three blocks (wrong on middle+right).
    - Transpose(H,W) emits a transposed full grid (wrong everywhere the crop isn't).
    - hflip/vflip/rotate require a Slice(neg-step)/Gather with an INT64/index initializer PAYLOAD →
      not cost<=1 by construction (prompt's own dead-end note), and still wouldn't produce a crop.
  Any correct graph must charge at least (a) a size-recovery reduction, (b) a per-column mask tensor,
  (c) the free-output Where — all strictly above cost 1. **cost<=1 is INFEASIBLE for this task.**
- **One remaining hypothesis (NOT for perfect-25):** a sub-30-canvas true-crop contract could in
  principle price below 47, but the output shape varies per input (W/3 ∈ {2,3,4,5}) → dynamic dim →
  validator rejects `calculate_memory` (see task067.md Don't-do log). Free-output Where is the only
  viable path; the 47-rung is already a 4-node free-output crop (ReduceSum→Greater(t[5])→Pad→Where).
  Not a perfect-25 candidate and out of this prompt's cost<=1 scope.

**Verdict: WALL for the perfect-25 cost<=1 ceiling. Do not re-explore the zero-param family.**
