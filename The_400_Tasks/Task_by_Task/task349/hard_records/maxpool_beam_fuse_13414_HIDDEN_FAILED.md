# DEAD ROUTE — MaxPool U8 beam fuse (13414) HIDDEN-FAILED on Kaggle

- **Live bar when explored:** 14854 (Vamshi, manifest sha `20657a86804d…`), 2026-07-03.
- **Route:** Replace Vamshi's beam-detector stack (ArgMax `first` → Cast `firstf` →
  ReduceMax `occu` → Cast `occ` → Where `safe` → Greater `beamb`) with a single
  **MaxPool U8 cumulative-max-down** node feeding the same final Max. Drops
  first/firstf/occu/occ/safe/beamb → -1440 B → **13414**.
- **Local gates it PASSED (all of them):**
  - visible 267/267 (100%)
  - truth-full 200000/200000 (100.000%)
  - ab-full byte_equal=True 200267/200267 divergences=0 vs LIVE
  - profile static==profile==13414 (mem 12668 + params 746), 0% gap → "exact ship" class
  - reviewer_verdict SHIP
- **EXACT FAILED GATE:** the **Kaggle hidden probe**. Packet
  `20260703T090959Z-task349.rejected_hidden-failed.json`:
  `probe_ok=false, probe_delta=-35.55, candidate_status=hidden_failed,
  kaggle_result_reason="expected-if-correct +0.102, actual -35.550", probe_kind=stack`,
  checked 20260703T144119Z. Artifact: `prototypes/w1_batch42/task349.onnx`
  (sha `4484393cc1fc…`).
- **Why it's dead:** cumulative-max-down is a *different algorithm* for the beam,
  not an algebraic identity. It agrees with LIVE on the entire 200k arc-gen corpus
  but **diverges on hidden ARC configs our synth does not generate**. This is the
  canonical GRADER-GAP / distribution-dependent-fusion trap: passing visible +
  truth-full + ab-full byte-equal + profile==static is **NOT** proof of hidden
  correctness for an algorithm change. Only pure per-cell algebraic/dtype
  substitutions on the *same* computation are hidden-safe here.
- **Do not re-explore:** any beam re-derivation via MaxPool / cumulative-max /
  prefix-max / scan that replaces the ArgMax `safe`-row semantics. The ArgMax
  beam stack is at its **safe floor**; the only cheaper form is empirically broken.
- **One remaining hypothesis (unproven, hidden-risky):** a beam formulation that
  is a *provable algebraic identity* to `Greater(positions, ArgMax_row)` (not a
  spread heuristic). None found; comparison→BOOL→Cast forces 2 planes anyway.
