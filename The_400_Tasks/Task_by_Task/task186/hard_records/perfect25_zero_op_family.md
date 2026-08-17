# DEAD ROUTE — task186 perfect-25 / zero-op cost≤1 family

**Live bar at write time:** cost **62** (manifest, Vamshi, sha `3d5156bc…`, 2026-07-02).
**Current best after this session:** cost **53** (staged `task186_sub_signed_f16_cost53`).

## Attempted route
The 2026-07-06 prompt (`Prompt_..._Task186_ZeroopsweepCeilingAudit.md`) classed
task186 as a "perfect-25" slot with a forced floor of cost ≤1 via a no-payload
direct-output op (Identity / Transpose perm / attr-only op).

## Why it's dead (disproof, not "tried hard")
task186 is a **count→glyph transform, not a whole-grid relabel**. Rule: count the
color-1 cells in the top-left 3×3 window, then emit a fixed color-2 glyph whose
extent depends on that count (n=1/2/3 fill the first n top-row cells; n=4 adds the
center). Verified on every supplied example: `output != input` for all of
train[0..5] (checked via `data/competitions/neurogolf-2026/task186.json`), and the
output content (color-2 glyph on color-0) is not a permutation/flip/rotation of the
input at all. A zero-param Identity/Transpose/flip/rotate CANNOT produce a
count-dependent output. The prompt's own pre-scan already noted no whole-grid
identity/transpose/flip/rotate match — this confirms it: the cost≤1 forced-floor
premise is **false** for this task.

## One remaining hypothesis (already realised)
The real floor is the Conv-broadcast family, not a zero-op. It was driven from
LIVE 62 → 53 this session (single `Sub` signed-diff + F16 cast; see task186.md
floor proof). Do NOT re-open the perfect-25 / cost≤1 hunt.
