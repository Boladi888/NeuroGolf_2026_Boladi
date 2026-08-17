# Research/ — research-card pipeline (browse only when hunting something brand-new)

Two folders, one flow:

- **`Todo/`** — the card board + open research-card **prompts** for outside researchers (plain ChatGPT, internet-only, no repo access). `Todo/README.md` holds the card contract, authoring guidance, and the filing protocol.
- **`Done/`** — everything **completed**: `COMPLETED_CARDS_LEDGER.md` (what was researched + verdicts — check before proposing), `cards/` (retired card bodies), and process artifacts (audits, verifications, critiques, cross-references). The *actionable* content of every deliverable is extracted up to the root docs; full deliverable bodies are archived after extraction (`_archive/Research_*.md`, rule 7).

## The flow

1. Author a card in `Todo/` (follow the contract in `Todo/README.md`).
2. Owner pastes it to an outside researcher; the deliverable comes back as a Markdown note.
3. Captain files the deliverable in `Done/` under the card's suggested filename.
4. Extract durable findings into the right root doc (`techniques.md` / `traps.md` / `ort_reality.md` / …), **regenerate `techniques_catalog.md` + add the `by_symptom.md` route**, add an `../INDEX.md` row if a new doc, write a `CHANGELOG.md` entry.
5. Archive the full deliverable body to `_archive/Research_<name>.md`, retire the card body to `Done/cards/`, add the ledger row, run `lib_lint.py`.

No worker task path leads here. Durable findings get **promoted up** to the `Onnx_Advanced/` root — find them via `../INDEX.md`.

## Where older material lives

- Reference docs that lived here pre-2026-06-01 were **flattened to the `Onnx_Advanced/` root** (agents weren't reliably opening subfolders) — e.g. `arc_rule_taxonomy.md`, `synth_oracle_strategies.md`, `tiny_architectures.md`, `scoring_internals.md`.
- Pre-June deliverables, cards, and notes were **archived to `Onnx_Advanced/_archive/`** in Phase-2 (2026-06-05), prefixed `Research_*`, `TodoDone_*`, `notes_*`; the K/L/M full bodies followed on 2026-06-10. Salvage-before-archive was applied — every extraction lives in the root docs.
