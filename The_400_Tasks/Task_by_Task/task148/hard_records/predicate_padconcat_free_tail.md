# Predicate-only Pad/Concat-free tail — priced wall

- Live bar: **1221**, SHA-256 `c38ab8981a359f6d11abeda7ad595a04cba751f3523984cd4c3eff1822a8289e`, from `python tools/ng_current_bar.py --task 148 --json` (`authoritative_source=manifest`, owner Aditya).
- Attempted route: price the prompt's predicate/index-only, Pad/Concat-free output tail before building variants.
- Exact failed gate: `FAST_PROBE_PRICE` — the live `task148_v22_reuse_B` is already this representation (`Pad=0`, `Concat=0`, terminal free-output `Einsum`) and scores/profiles **1221 = 1039 memory + 182 params**, above the **720** kill threshold; visible is 266/266.
- Why dead: the named Pad/Concat/template carrier is already absent. Rebuilding variants in that representation would attack a stale component and needs at least another 501-byte reduction to reach the prompt target.
- One remaining hypothesis: compile mirror selection and row threshold descriptors directly into one repeated-input deep Einsum, but only pursue after a paper budget proves total constant bases plus runtime scalars can fit at or below 720; data-dependent row pairing is the likely priced wall.
- Probe manifest: `The_400_Tasks/Task_by_Task/task148/prototypes/task148_predicate_tail_fast_probe_drop.json`.
