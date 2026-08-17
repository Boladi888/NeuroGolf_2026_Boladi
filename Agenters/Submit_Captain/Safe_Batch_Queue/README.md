# Safe Batch Queue - deprecated

This folder is retained only for compatibility with notes from the first live run.

Do not add new packets here. New Captain handoffs use lane inboxes:

```text
Agenters/Submit_Captain/Updates_Codex/
Agenters/Submit_Captain/Updates_Claude/
```

`SAFE_BATCH_QUEUE` is now a packet lane/state in the JSON packet, not a folder name. See `Agenters/Submit_Captain/PACKET_SCHEMA.md`.

`submit.py` dry-run is a packaging check, not a Kaggle/publicScore delta check.
