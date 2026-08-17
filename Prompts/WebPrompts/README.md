# WebPrompts

This folder holds small, shareable prompt packs for web-only agents and teammate
handoff pages.

## Current Packs

- `Task374_Creative_WebPrompt_Pack_20260703/` - self-contained Task 374
  creative exploration pack for web-only workers.
- `_workflow/SELF_CONTAINED_WEBPROMPT_PACK_WORKFLOW.md` - reusable workflow
  for building future repo-less prompt packs.

The tracked zip bundles were removed during archive cleanup. Recreate an
attachment from the unpacked directory only if a future handoff needs one.

## House Rule

Task-specific web packs should be smaller than Aditya's pack unless there is a
clear reason to ship a larger knowledge base. The target shape is:

1. one start-here file;
2. one broad worker prompt;
3. compact task facts and visual assets;
4. prior work as evidence, not as marching orders;
5. a validation and return contract;
6. a source map so a local agent can refresh the same facts later.

Reusable workflow: `_workflow/SELF_CONTAINED_WEBPROMPT_PACK_WORKFLOW.md`.

## Local intake for returned packets (2026-07-03)

When a web/partner worker returns a packet, run it through local intake before trusting it: `python tools/external_packet_intake.py --task NNN --packet <path>` (turns it into a task-local reproducible build/probe or a rejection). See `Agenters/External_Intake/README.md` + `Agenters/External_Intake/intake_contract.md` and the HandOffz skill `Handoff/skills/handoff-workflows/references/external-worker-packets.md`.
