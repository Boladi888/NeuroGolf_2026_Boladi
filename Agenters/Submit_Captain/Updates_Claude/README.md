# Updates_Claude

Lane inbox for Claude-origin reviewer-approved candidate packets.

Write one JSON packet per candidate using `Agenters/Submit_Captain/PACKET_SCHEMA.md`.

Captain commands:

```text
python Agenters/Submit_Captain/captain_ctl.py digest --lane claude
python Agenters/Submit_Captain/captain_ctl.py probe-batch --lane claude --apply
python Agenters/Submit_Captain/captain_ctl.py probe-collect --apply
python Agenters/Submit_Captain/captain_ctl.py stage --lane claude --apply
```

Do not put ONNX files in this inbox unless the packet's `onnx` path points here. Prefer keeping ONNX artifacts in task-local `The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/` folders and using the JSON packet as the inbox item.
