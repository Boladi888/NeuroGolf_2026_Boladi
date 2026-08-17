# task134 bool_argmax hard record

- Live/team bar used: cost 891, sha256 `1913cef3e00dcebff69636d73b93cccbc15e62a5702644af02c9da7b21fb9e3e`, source `Team_Submission/Backend/manifest.csv` plus BoBrain consistency (`team_manifest_cost=891`).
- Attempted route: delete `row_any_u8` and `col_any_u8`, feeding `row_any_bool`/`col_any_bool` directly to the three bbox `ArgMax` nodes.
- Exact failed gate: `ng_gate visible --task 134 --cand prototypes/task134_bool_argmax.onnx` -> `ERROR: InvalidGraph ... Type 'tensor(bool)' ... operator (ArgMax) ... is invalid.`
- Why dead: ORT rejects `ArgMax(BOOL)`; the U8 occupancy hop is mandatory for this bbox formulation.
- One remaining hypothesis: look for a different bbox primitive that avoids `ArgMax` entirely, or a new block-color detector with fewer than the current 120 rank-2 params; do not build more BOOL-ArgMax variants unless ORT support changes.
