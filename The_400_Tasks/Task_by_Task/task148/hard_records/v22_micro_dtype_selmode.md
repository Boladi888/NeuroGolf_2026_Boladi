# v22 micro dtype/selmode probes — dead

- Live bar: **1221**, SHA-256 `c38ab8981a359f6d11abeda7ad595a04cba751f3523984cd4c3eff1822a8289e`, from `python tools/ng_current_bar.py --task 148 --json` (`authoritative_source=manifest`, owner Aditya, updated 2026-06-22T06:32:53Z).
- Attempted route: micro-shave the promoted `task148_v22_reuse_B` graph by (1) replacing `GreaterOrEqual(rem, oneu)` with `Cast(rem -> BOOL)`, (2) removing `selmode` from the first code-read Einsum, and (3) changing descriptor/floating constants to F16.
- Exact failed gates: `seeded_cast` passed visible `266/266` but scored **1221** (`1039+182`) so no gain; `split_code_no_selmode` passed visible `266/266` but scored **1259** (`1039+220`); both F16 variants failed `visible` at ORT load with mixed-type `Einsum` (`tensor(float)` vs `tensor(float16)`).
- Why dead: `selmode` is cheaper than duplicating shared `B`/`basis` for the code read, all `desc_table` rows `0..21` are used in the bundled generated corpus, and the only large descriptor dtype shave is blocked by `Einsum`'s single-type constraint while the official input must remain F32.
- One remaining hypothesis: a genuinely different deep-Einsum/Boolean-synth representation that removes the INT32 key/gather or the `[1,30,2]` descriptor carrier; do not re-test descriptor F16, unused-row trimming, or no-`selmode` split without a new proof of budget.
- Probe manifest: `The_400_Tasks/Task_by_Task/task148/prototypes/task148_v22_micro_probes_drop.json`.
