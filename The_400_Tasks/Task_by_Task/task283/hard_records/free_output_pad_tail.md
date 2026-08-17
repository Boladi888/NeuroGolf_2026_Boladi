# task283 — FreeOutputPadTail route is stale/dead (2026-07-11)

- Valid live/team-best bar: **745**, from `python tools/ng_current_bar.py --task 283 --json`; SHA-256 `b09a525d9464f976afab5cf0d0e2e6866f172e2bcb9d40d48400c2668a5da4e2`, `Team_Best_Onnx/task283.onnx`.
- The task-local 601-byte QLinearConv packet is not a valid lower team-best: refreshed `K-NeuroGolf-OnnxlyFans/Team_Submission/Staged_Boladi/_hidden_failed/20260710T_task283_known_reship/task283.json` marks SHA `ec9ed612c9e905645f06ce382e1305b453c57f037b7ef3de06f2eb48d1ab020a` as `hidden_failed` (`expected +1.25 if correct, landed -16.17`).
- Attempted route: prompt-mandated prerequisite and price audit for a one-channel predicate/index followed by Pad + Equal/Where output tail.
- Exact failed gate: **charged-component presence check failed**. `inspect_graph.py Team_Best_Onnx/task283.onnx` reports one `Einsum`, zero intermediates, and five initializers only: `A[10,5]=50`, `B[5,10]=50`, `K[5,3,3]=45`, shared `CR[10,30]=300`, shared `S[3,10,10]=300`. The alleged charged F16 `[1,10,10,10]` prepad/canvas/template is already absent.
- Reprice/proof: `_score_v2.score_network_v2(...) = (0, 0, 745)`; `profile_real_cost.py --task 283 --cand Team_Best_Onnx/task283.onnx --arcgen 266` => `static=745 profile=745 (memory 0 + params 745; 265 real inputs)`; visible => `265/265 (100.00%) PASS`; safety => `PASS`, 0 warnings, 0 failures.
- Why the requested family is dead: every `... -> Pad[1,1,30,30] -> Equal(ids) == output` route charges the padded U8 canvas itself (**900 B**) before producer params, already above both target 420 and valid live 745. The concrete one-channel path also pays 100 B for `idx_u8[1,1,10,10]` and 400 B for the F32 source crop, so it prices at least **1,400 B** before small params. Equal-before-Pad instead charges a 1,000 B `[1,10,10,10]` BOOL one-hot plus the index/read (at least 1,500 B). A Where tail likewise forces a 900 B full-canvas condition and one Where selects only two of the four required classes. Per the workflow update, no prototype or truth/AB/full gate was run.
- One remaining hypothesis: a new single-output algebraic factorization that replaces or shrinks the live graph's dense `CR` and `S` initializer bases (600/745 params total) while preserving its one-node free-output structure. Pad/Equal/Where and sparse-initializer variants should not be revisited.

```json
{
  "verdict": "DROP",
  "candidate_class": "NO_CANDIDATE_STALE_ROUTE",
  "cost": 745,
  "profile_cost": 745,
  "live_cost": 745,
  "status": "REJECT_LOCAL"
}
```
