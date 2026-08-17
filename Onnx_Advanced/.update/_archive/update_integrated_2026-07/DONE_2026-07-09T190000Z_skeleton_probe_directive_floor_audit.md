# Stubbed-skeleton probe = cheap falsifier for directive floor claims (task365, 2026-07-09)

## What I discovered

Two coupled things, one method and one process trap.

**Method — the stubbed-skeleton floor probe.** When a prompt/workbook claims "the forced
floor of this task is only X because component Y is not forced", you can falsify it in
~30 minutes without building the real candidate: build the claim's OWN skeleton — the
real input read + the real output path + the claimed cheap mechanism — and stub every
"allegedly expensive but supposedly avoidable" quantity as free initializers, then
`score_v2` it. If the skeleton prices above the claimed floor while granting the claim
everything for free, no formulation in that family can reach the floor, no matter how
the middle is computed. On task365 the workbook claimed ~900 via #156 tiny-K scalar
selection; the skeleton (Einsum colidx read + verbatim v18 #156 chain + dynamic
Slice/Pad/QLC output, all 15 per-box scalars r0/cL/tall/wide/red stubbed) priced
**956 mem + 368 params = 1,324 > 900** → kill in one build, zero gate runs.
Probe: `The_400_Tasks/Task_by_Task/task365/prototypes/build_task365_probe_scalarsel_skeleton.py`;
record: `task365/hard_records/2026-07-09_scalarselector_900_dead.md`.

**Process trap — directives can cite a technique already harvested from the same task.**
The task365 directive "ScalarSelectorOutput 3260→900, use #156" pointed at a technique
that was DISCOVERED ON task365 (v18, −90 B, 2026-06-21) and already present in the
staged best; there were no `Concat→ArgMax→Gather` branch surfaces left to convert. The
~900 figure matched no source (BoBrain provider `cost_floor=null`, provider search for
the directive codename = 0 hits); best guess is a misapplied canvas-900B law, which
binds full-[1,10,30,30]-one-hot outputs, not free-QLC crop outputs. Also the
directive's "team_best 3260 by Boladi" was stale — the authoritative team manifest had
task365 at **2,431 (Sho, 2026-07-09T01:09Z)**. Cross-check every directive's floor and
team-best against (a) the task doc's forced-component proof, (b) the provider
consistency call, BEFORE building.

## Why it matters

- Saved a full from-scratch STRUCTURAL rewrite attempt at a floor that is provably
  unreachable (task365 forced floor stays ≈2,000–2,200: read 810 + CumSum ≈480 +
  output ≈357 + locate/bookkeeping).
- The probe pattern generalizes: it is the cheap complement to #134 floor discipline —
  #134 says which floors to trust; this says how to test a NEW floor claim in one
  skeleton build before committing a session to it.

## What's missing from the library

- techniques.md has no entry for "price the claim's own skeleton with stubbed
  scalars" as a floor-falsification move (nearest: #134 floor discipline, #68
  formulation-relative floors). Suggest adding it under floor discipline.
- WORKER_PROTOCOL/by_symptom has no row for "my prompt's floor contradicts the task
  doc's floor proof" → route: provider consistency + stubbed-skeleton probe + kill rule.
