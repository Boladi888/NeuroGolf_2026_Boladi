# Artifact Manifest Provenance

This directory records the local-only scoring artifact archive that was restored on 2026-05-07. Use it as the first dedupe check before running any broad research or external-artifact mining sweep.

## 2026-05-07 Preserved Files Manifest

- Manifest path: `The_400_Tasks/_archive/artifact_manifests/2026-05-07-preserved-files-manifest.txt`
- Entry count: 24,856 lines.
- Git blob: `889ba8f953f97d79b00f052641a5c593cf6c4518` in commit `b2e3cd9`.
- Manifest file SHA256 as stored in the worktree, including BOM and CRLF line endings: `4BF236690914EDB1414E6E61D1D8DB8593AC410F1DC873B09CCEA6A63CD383BB`.
- LF-normalized manifest text SHA256: `724854171D8FA6D1E974F1A96B48254A172D1787BE1E41127196E6588C46BEFA`.
- Archive SHA256 verified before restore: `1947BB91253A73888D6DC30BAB43019C8F27DFFAFBC840BE5CC0A0B955F062BE`.

The `1947BB...` hash is the preserved archive hash from `Logs/sessions/2026-05-07_0450_both-score-5500-push.md` and `Logs/sessions/2026-05-07_1331_file-away-artifacts.md`. It is not the hash of the tracked manifest text file. If the manifest file hashes to `4BF236...`, that is the expected current tracked state.

## Dedupe Rules

1. Before starting a broad artifact/source sweep, search existing evidence first: `Logs/broad_*_against_coconut_2026-05-07.csv`, `Logs/all_local_external_source_sweep_2026-05-08.*`, `Logs/external_cascade_audit_2026-05-08.*`, and this manifest.
2. Treat `Logs/all_local_external_source_sweep_2026-05-08.*` as already-filed evidence for task145. Do not rerun that sweep unless testing a new source subset or adding a new metadata column without re-verifying models.
3. Future external audit and all-local sweep rows should include candidate `sha256` values so duplicate payloads can be collapsed by bytes, not just by source path.
4. Generated caches and payload directories listed in the manifest are archive inventory, not coordination state. Keep decisions in tracked CSV/Markdown ledgers.

> ⚠ **2026-05-07-preserved-files-manifest.txt is 1.6 MB (~412k tokens). NEVER Read it — grep it.**
