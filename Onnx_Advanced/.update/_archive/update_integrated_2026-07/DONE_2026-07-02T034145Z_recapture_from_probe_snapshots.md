# RECAPTURE: recover a cleaned-but-Kaggle-confirmed ONNX byte-identically from probe snapshots

**What I discovered** — When a candidate's ONNX is "cleaned from `staged_submits/` and never banked" (the whole premise of the batch-41_special recapture prompts), you often do NOT need to re-derive it from scratch. The exact bytes usually still exist in two places, recoverable with a verifiable SHA match:

1. **The Submit_Captain probe snapshots**: `Agenters/Submit_Captain/_probe/_snapshots/iso-<Lane>-<UTC>/submission.zip` — the isolated-probe submission bundle. Every task's ONNX is a top-level entry (`taskNNN.onnx`) in that zip. If the candidate was ever probed on Kaggle, its bytes are frozen here.
2. **The consumed update manifests**: `Agenters/Submit_Captain/Updates_Codex/_consumed/<UTC>-taskNNN-<cost>.json` (and the `Updates_<Lane>` siblings) — these record `sha256`, `cost`, `profile_cost`, all gate strings, `candidate_status` (e.g. `confirmed_public`), and the snapshot path. Grep the repo for the target cost number to find them fast.
3. The local task-prototypes dir may also still hold it under a non-obvious name — always `sha256` every `prototypes/*.onnx` against the manifest's recorded SHA before assuming it's gone.

**Recipe** (60 seconds, beats hours of re-derivation):
```
grep -rl "<cost>" Agenters/Submit_Captain/ submissions/ The_400_Tasks/Task_by_Task/taskNNN/
# read the *-taskNNN-<cost>.json -> get sha256 + snapshot zip path
python -c "import zipfile,hashlib; d=zipfile.ZipFile(ZIP).read('taskNNN.onnx'); \
  print(hashlib.sha256(d).hexdigest().upper()); open('recovered.onnx','wb').write(d)"
# confirm SHA == manifest.sha256, THEN re-run all gates from scratch (do not trust old gate strings blindly)
```

**Where I hit it** — task017, recapture prompt `Prompt_2026-07-02T4102Z_Task017_PeriodicHoleFillRecapture.md`. Prompt said "re-derive it". Instead recovered `task017_v35_direct_crops_11probe` (SHA `4F7EB945…`) byte-identical from BOTH the local prototype and `iso-Codex-20260626T190519Z/submission.zip`. Re-verified all gates live: profile 4992 (static==profile), visible 266/266, truth --full 198757/200000 (99.379%), battery PASS. Staged `task017/staged_candidates/task017_periodicholefill_cost4992/`.

**Why it matters** — Batch 41_special is 10 "un-banked / onnx-gone" recapture tasks. If those ONNX were ever probed, this recovery route makes most of them a ~5-minute SHA-verify + gate-rerun instead of a from-scratch rebuild. It also GUARANTEES bit-exactness (the Kaggle-confirmed graph), which re-derivation cannot.

**A second, task-agnostic lesson (verify the bar's OWN truth on Path-B tasks)** — task017's *banked* team-best (7802, "sampled_periodic_pattern_selector", a parametric 106-pattern classifier) scores only **98.639%** arcgen truth — it passes visible 266/266 but is itself a Path-B candidate below the 99% bar. The recaptured 4992 hole-fill scores **99.379%**. So a Path-B recapture can be strictly better than the banked bar on BOTH cost AND correctness, but you only see this if you **run `truth --full` on the current banked bar too**, not just on your candidate. Don't assume the banked graph is the ground-truth reference.

**What's missing from the library** — I did not find an existing note documenting the `Agenters/Submit_Captain/_probe/_snapshots/*/submission.zip` recovery path or the `Updates_<Lane>/_consumed/*.json` manifest index as a recapture source. Recommend a short entry under the recapture/staging workflow docs (WORKER_PROTOCOL staging section or a recovery recipe in diagnostics_recipes.md).
