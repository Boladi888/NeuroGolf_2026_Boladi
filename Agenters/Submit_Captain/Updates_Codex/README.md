# Updates_Codex

Lane inbox for Codex-origin reviewer-approved candidate packets.

## Hard rule

Files here are **not validated team-stage artifacts**.

Do not copy anything from this folder to `Staged_Boladi/` or any team submission path just because a packet exists here.
An inbox packet is only a request for Submit Captain handling.

Submit Captain must vet/probe it first. A packet without `probe_ok: true` is `HELD` / queued unless it satisfies the
strict `LOCAL_GOLD` no-probe safety contract, even when its
`lane` is `SAFE_BATCH_QUEUE`.

Only Submit Captain stages confirmed winners, using `captain_ctl.py stage --lane codex --apply`, and that command refuses
packets unless `probe_ok: true` or strict `LOCAL_GOLD` safety passes.

Write one JSON packet per candidate using `Agenters/Submit_Captain/PACKET_SCHEMA.md`.

Captain commands:

```text
python Agenters/Submit_Captain/captain_ctl.py digest --lane codex
python Agenters/Submit_Captain/captain_ctl.py probe-batch --lane codex --apply
python Agenters/Submit_Captain/captain_ctl.py probe-collect --apply
python Agenters/Submit_Captain/captain_ctl.py stage --lane codex --apply
```

Do not put ONNX files in this inbox unless the packet's `onnx` path points here. Prefer keeping ONNX artifacts in task-local `The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/` folders and using the JSON packet as the inbox item.
