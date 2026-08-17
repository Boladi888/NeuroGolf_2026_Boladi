# Self-Contained WebPrompt Pack Workflow

Use this when making a zip for Claude-app, ChatGPT web, or any worker that does
not have access to this computer, repo, BoBrain, workbook, or local files.

## Pack Goal

The pack should be small enough to upload, broad enough to invite new ideas, and
concrete enough that the worker does not waste time rediscovering stale facts.
For a one-task NeuroGolf pack, stay below the Aditya knowledge pack size
baseline of 348,186 bytes.

## Build Steps

1. Pick the actual current target.
   - Use the workbook plus manifest cost when available.
   - If a staged or captain packet already beats that cost, make the staged
     rung the effective target.
   - Record conflicts explicitly. Web workers cannot refresh local truth.

2. Create a creative mandate before the narrow lever.
   - The prompt should ask for multiple formulation families.
   - Existing graphs and prior prompts are oracles and evidence, not templates.
   - A floor or wall note means "this formulation failed," not "stop thinking."

3. Bundle enough context to be repo-less.
   - Include the raw task JSON and preview image.
   - Include a compact plain-English rule summary.
   - Include current best and staged facts, with dates and hashes when known.
   - Include only source files that materially help the task.

4. Separate guardrails from constraints.
   - Guardrails: do not submit, do not stage ties, do not use banned ops, report
     exact validation evidence.
   - Non-constraints: the worker is not limited to the named lever, current graph
     family, or old prompt direction.

5. Add the return contract.
   - Ask for a candidate ONNX if the worker can create one.
   - Otherwise ask for implementable builder code or an exact construction plan.
   - Require a clear `ready`, `probe`, or `discard` decision.

6. Add a source map.
   - List bundled files first.
   - List original local paths only as provenance for a future local refresh.
   - Never make a web-only worker depend on local paths.

7. Zip and size-check.
   - Zip the pack folder beside the source folder.
   - Compare against Aditya's zip size.
   - Open the zip listing and confirm `README.md`, start-here, prompt, context,
     validation, source map, and handoff post files are present.

## Quality Bar

The best webprompt packs do not say "try to improve this." They say:

- what the rule is;
- what is currently winning;
- what was tried and why it mattered;
- which failures are real bans;
- which "walls" are only walls for one architecture;
- what a fantastic result would look like;
- how to package the worker's output so a local agent can verify it.

## Required machine-readable pack files (2026-07-03)

Every external pack (Claude-app, ChatGPT web, paid HandOff partner) must now also
carry the following machine-readable files so a local reviewer has a deterministic
intake path (see `Agenters/External_Intake/` and `tools/external_packet_intake.py`):

- **`RETURN_CONTRACT.json`** — the return contract as JSON, with the fields the
  intake tool parses: `graph_ops`, `charged_tensors`, `expected_params`,
  `expected_memory`, `current_bar_used`, `validation_command`, and
  `decision` (`ready` | `probe` | `discard`). A packet missing `graph_ops` or
  `charged_tensors` is rejected at intake.
- **`DO_NOT_DO.md`** — the forbidden actions/assumptions for this task (banned
  ops, tie-staging, submit, repo/local-path assumptions).
- **`KNOWN_STALE_SOURCES.md`** — sources the worker must NOT trust as the current
  bar (old prompt text, BoBrain memory, prior sidecars, dated notes), so the
  worker anchors on the packet's stated bar instead of stale numbers.
- **`manifest.json`** — machine-readable pack manifest with fields:
  `task`, `current_bar`, `staged_bar`, `forbidden_assumptions`,
  `local_review_owner`.

### Fixed guardrail language every external pack must carry

Include these four lines verbatim in the pack (start-here + prompt) so a web-only
worker cannot mistake candidate work for a shipping path:

- Do not submit externally
- Do not modify team best
- Return candidate artifacts only
- Submit captain decides staging after local gates
