# Task prompt authoring

This directory is now an archive-friendly prompt authoring kit, not a run
ledger. The only kept batch folder is `71/`, which serves as a representative
late-game prompt wave. Earlier prompt waves and selection summaries were removed
because their useful lessons are summarized in
`PROMPT_RESULTS_RESEARCH_2026-07-10.md`.

Read `_Before writing prompt.md` before drafting or refreshing a prompt.

For repo-access worker prompts, include a compact library lookup block near the
top of the prompt:

```powershell
python Onnx_Advanced/library_db/lib_query.py status
python Onnx_Advanced/library_db/lib_query.py context NNN
```

Use `search "<rule/op/cost keywords>"`, `op <Op> <DType>`, and `facts --task taskNNN` before opening large library files or trusting old ONNX/ORT lore. If `status` reports stale, rebuild with `python Onnx_Advanced/library_db/build_library_db.py` or fall back to cited Markdown.

For WebPrompt files, run the lookup author-side only. Do not paste repo paths or
DB commands into the browser-LLM prompt body; that model has no repo access.

External task-pack invariant: a one-task `submission.zip` contains exactly one
root file, `taskNNN.onnx`, with no folder prefix and no extras. Do not confuse it
with the team backend full zip, which is assembled elsewhere and includes the
backend invariant around `task000` plus `task001`-`task400`.

Use an existing pack bundle and its `pack_manifest.csv` when one exists; rebuild
or regenerate packs only when the baseline zip, task queue, or prompt template
has actually changed. Returned artifacts must be classified `ready`, `probe`, or
`discard` with validation counts, measured memory/params/cost, zip contents, and
first-failure details when not ready.
