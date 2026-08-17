# Submit Captain packet schema

Orchestrators hand reviewer-approved candidates to Submit Captain as JSON packets in a lane inbox:

```text
Agenters/Submit_Captain/Updates_Codex/<UTC>-taskNNN.json
Agenters/Submit_Captain/Updates_Claude/<UTC>-taskNNN.json
```

`SAFE_BATCH_QUEUE`, `RISK_SINGLE_PROBE`, and `SAFE_FALLBACK_FOR_SPREAD` are packet states/lanes, not folder names.
An inbox packet without `probe_ok: true` is `HELD`/queued unless it satisfies the strict `LOCAL_GOLD` no-probe
safety contract; queued is not terminal for any positive expected delta.

## Required keys

```json
{
  "task": "taskNNN",
  "source_lane": "Codex",
  "lane": "SAFE_BATCH_QUEUE",
  "onnx": "The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/taskNNN_desc_cost123/taskNNN.onnx",
  "cost": 123,
  "sha256": "lowercase-or-uppercase-sha256"
}
```

`source_lane` must match the inbox namespace:

```text
source_lane: Codex  -> Agenters/Submit_Captain/Updates_Codex/
source_lane: Claude -> Agenters/Submit_Captain/Updates_Claude/
```

`lane` remains the Captain queue state: `SAFE_BATCH_QUEUE`, `RISK_SINGLE_PROBE`, or `SAFE_FALLBACK_FOR_SPREAD`.

`captain_ctl.py` refuses to stage a packet unless:

- the ONNX exists,
- `source_lane` matches the inbox folder,
- the SHA256 matches case-insensitively,
- the candidate beats the team-best manifest cost,
- either `probe_ok` is `true` with `probe_kind:"stack"`, or the packet passes the strict `LOCAL_GOLD` no-probe safety contract below.

## Strongly recommended keys

```json
{
  "profile_cost": 123,
  "reviewer_verdict": "SHIP",
  "expected_delta": 0.456,
  "gates": {
    "visible": "267/267 PASS",
    "truth_n8000": "8000/8000",
    "ab_full": "byte_equal=True",
    "profile_real_cost": "static=123 profile=123"
  },
  "msg": "short team sidecar message",
  "lesson": "optional Librarian lesson text",
  "source_manifest": "The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/taskNNN_desc_cost123/manifest.json",
  "task_note": "The_400_Tasks/Task_by_Task/taskNNN.md"
}
```

## No-probe LOCAL_GOLD keys

These keys are required when a packet asks Captain to stage without a Kaggle stack probe:

```json
{
  "candidate_class": "PATH_A_BYTE_EQUAL",
  "parent_source": "K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/taskNNN.onnx",
  "parent_sha256": "sha256-of-parent",
  "candidate_sha256": "sha256-of-candidate",
  "proof_commands_run": [
    "python tools/ng_gate.py ab --task NNN --cand C.onnx --n 8000",
    "python tools/ng_safety.py model --task NNN --cand C.onnx --json"
  ],
  "proof_results": {
    "ab_full": "byte_equal=True ...",
    "profile_real_cost": "profile=123",
    "ng_safety": "PASS"
  },
  "safety": {
    "kind": "model",
    "verdict": "PASS"
  },
  "x86_required": false,
  "kaggle_probe_required": false,
  "status": "LOCAL_GOLD"
}
```

`LOCAL_GOLD` is deliberately narrow. Captain accepts it only for `PATH_A_BYTE_EQUAL` / `PATH_A_INHERITED` style packets with AB `--n 8000` byte-equal proof, matching candidate SHA, parent source/hash, and an embedded `ng_safety.py` JSON result whose verdict is `PASS` with zero warnings/failures. Path-B truth-`--n 8000`, fitted/approx, grader-gap, dynamic/profile uncertainty, x86-required, or warning-bearing packets must stay `PROBE_READY` and use `probe-stack`.

## Probe-written keys

`captain_ctl.py probe-collect --apply` writes:

```json
{
  "probe_delta": 1.234,
  "probe_ok": true
}
```

`captain_ctl.py stage --lane <lane> --apply` stages packets with `probe_ok: true` + `probe_kind:"stack"`, or strict
`LOCAL_GOLD` packets that pass the no-probe safety contract.

## Ledger writeback (captain → Orchestrator run)

`stage --apply` (and `reject --apply`) stamp the originating run's
`Agenters/Orchestrator/runs/<id>/final_report.csv` `captain_status` to a terminal value
(`staged` / `rejected`), matched by `(task, sha256)`. This closes the drift where a shipped candidate
left its run stuck at `captain_status=held` and blocked `run_audit`. Writeback is best-effort and
fail-soft — it never blocks a stage; if the run ledger is being written concurrently it skips and
leaves a note.

## Accepted key aliases (robustness)

`captain_ctl.py` normalizes older/templated packets on read so a mis-keyed packet is accepted, not
silently HELD: `candidate_path`|`onnx_path`|`path` → `onnx`; `static_cost` → `cost`; `captain_lane`
→ `lane`; and a missing `source_lane` defaults to the inbox folder. Emit the canonical keys above —
aliases are a safety net, not the contract.
