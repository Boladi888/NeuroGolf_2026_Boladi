# 06_COMMUNITY.md — NG external / community-agent implementation references

> LIVE — edit freely. Tier 2 — load on demand.
>
> Catalog of competitor notebooks + external ONNX submission artifacts NG draws from or compares against. Open when porting an external pattern, composing a blend, or auditing competitive position.

---

## When to use this file

- About to port a pattern from a competitor notebook into `build_submission.py`.
- Composing a blend submission (`tools/_archive/external_blend.py`).
- Cross-checking your approach against a competitor's published solution.
- Updating the catalog after a fresh `kaggle datasets download` lands in `data/competitor_artifacts/`.

For "what's the state of MY code?" stay in `03_PROJECT.md`.

---

## Format constraint

Each entry is **name + one-paragraph summary + pointers**. Deep comparisons go to `Research_Archive/notes/<NN>_<slug>.md` — link to it from here.

---

## Competitive landscape (snapshot 2026-05-03)

From the deep-read of `Research/primary_sources/neurogolf-2026-rule-based-onnx-solver.ipynb`:

| Rank | Team | Public LB | Notes |
|---|---|---|---|
| #1 | keymoon | 6,203.22 | |
| #2 | "Claude Code or Codex?" | 5,900.41 | LLM-driven |
| #3 | "Kaggle Agent" | 5,879.54 | LLM-driven |
| Ref | Infinitesimals | 4,035.97 | Likely the clean-play ceiling (no exploits) |
| Us | Boladi888 | **4,043.11** (nectarine #14) | ~3.4 pts short of Infinitesimals; ~65% of #1; +788 vs honeydew #8 from blend pool refresh + 3 new patterns |

The 6,000+ scores all rely on the "positive-cost wrappers" exploit family (decoded in `Onnx_Advanced/scoring_internals.md#a2_cost_metric_exploits`) — out of scope per ethical-play stance.

---

## Catalog

### External-blend source pool (Apache-2.0 / CC0-1.0)

All pulled via `kaggle datasets download` into `data/competitor_artifacts/`. Verified license-compatible with our Apache-2.0 winners' license. Attribution fulfilled at win-time via `THIRD_PARTY_NOTICES.md`. Composed via `tools/_archive/external_blend.py` with `tools/onnx_repair.py` triage (only READY / REPAIRABLE-RENAME accepted).

#### ashok205/neuro-golf-2026

- **License:** Apache-2.0
- **Local:** `data/competitor_artifacts/ashok205-neuro-golf-2026/`
- **What it gives us:** 160 hand-coded task ONNXs. ~107 winners survived in grape #7 blend; **107 again in honeydew #8** (some new organics displaced naturally — the count happens to coincide).
- **Notes:** Contributing source for confirmed `[B→A]` task lists in `05_BACKLOG.md`. **PR #26 landed 5 of these** organically: 276/309 (color-replace), 108/194/214 (shape-changing-pixel-perm). 113 and 307 still pending — see `05_BACKLOG.md` Tier 2.

#### karnakbaevarthur/neurogolf-2026-task-transformation-library

- **License:** Apache-2.0
- **Local:** `data/competitor_artifacts/karnak-task-transformation/`
- **What it gives us:** Task-transformation patterns. 103 winners in grape #7; **95 in honeydew #8** (8 reduced by new organic SOLVED).

#### afr1ste/neurogolf-6285-95-open-submission-artifact

- **License:** CC0-1.0
- **Local:** `data/competitor_artifacts/afr1ste-6285-95-artifact/`
- **What it gives us:** Originally advertised 6,285 LB, but **0 winners** in grape #7 — most ONNXs use `Compress` (banned post-2026-04-21 update). Ref decay confirmed.
- **Caveat:** Always re-verify external candidates against the current local validator. See `Logs/lessons.md` 2026-05-02 "Public-LB references decay with op-ban updates" entry.

#### needless090/neurogolf-onnx-v31

- **License:** CC0-1.0
- **Local:** `data/competitor_artifacts/needless090-v31/`
- **What it gives us:** 252 task ONNXs from a DSL pipeline. **36 winners + 1 OVERRIDE** in mango #13. 1-indexed task numbering matching ours.

#### konbu17/neurogolf-2026-blended-401-v117

- **License:** CC0-1.0
- **Local:** `data/competitor_artifacts/konbu17-401-v117/`
- **What it gives us:** 401 task ONNXs (file `task000.onnx` is unused; tasks 001-400 match ours). LB 5331+ source. **46 winners + 1 OVERRIDE** in mango #13.

#### afr1ste/neurogolf-6335-19-controlled-public-artifact (REJECTED)

- **License:** CC0-1.0
- **Status:** Pulled 2026-05-04 but NOT added to `external_blend.py` `DEFAULT_SOURCES`. Spot-check confirmed Compress op present (banned post-2026-04-21). Same decay pattern as the 6285-95 artifact. Local copy retained at `data/competitor_artifacts/afr1ste-6335-19/` for reference but not used in blend.

### Competitor notebooks (catalog at `Research/primary_sources/00_INDEX.md`)

14 publicly-shared `.ipynb` notebooks pulled into `Research/primary_sources/` with full per-notebook content type / public LB / follow-up dataset slugs in `00_INDEX.md`. Highest-leverage three already deep-read:

- `neurogolf-multi-source-onnx-solver.ipynb` (the engine code behind rule-based-solver). Deep-read at `Research_Archive/notes/deep-read-multi-source-onnx-solver.md`. Surfaced 28 pattern candidates across Tier S/A/B → seeded the `[B→A]` queue in `05_BACKLOG.md`.
- `neurogolf-locally-solved.ipynb` (160 hand-coded tasks; not optimal per author). Deep-read at `Research_Archive/notes/deep-read-locally-solved.md`. 12 of 160 overlap with our solved set.
- `neurogolf-logic-driven-ensembe.ipynb` ("Safe Blend" — repairs I/O naming on third-party submissions). Deep-read at `Research_Archive/notes/deep-read-logic-driven-ensembe.md`. Repair logic dormant insurance for future external blends.

Remaining 11 notebooks not yet deep-read.

---

## Pattern integration log

| Date | Lane | Pattern | Source | Adopted in | Outcome | Source session |
|---|---|---|---|---|---|---|
| 2026-05-02 | Alpha | Translation (frame shift) | `multi-source-onnx-solver.ipynb` Tier-S | `build_submission.py:translation_network` + `_detect_translation` + 12 unit tests | Kept as primitive (0 net new hits — task053 already covered by cheaper `fixed-shift_v_1`); may pay off on shape-changing variant | `Logs/sessions/2026-05-03_0125_grape-blend-translation.md` |
| 2026-05-02 | Both | grape #7 blend submission | ashok205 + karnak (afr1ste contributed 0) | `submissions/blend.zip` → renamed `submission.zip` | Kept — scored **3,261.15** (vs projected 3,302.56, 1.3% short); +3,045.82 LB (15.1× jump) | same |
| 2026-05-03 | Alpha | Color-replace (1×1 Conv non-permutation) | ashok205 task276 / task309 confirmed list | `build_submission.py:color_replace_network` + `_detect_color_replace` + 8 unit tests | Kept — 2 hits (task276, task309); cost ~90k → ~13.6 pts each. Generalizes color-perm to merge cases (multiple input colors → same output color). | session 2026-05-03 ([PR #26](https://github.com/Boladi888/NeuroGolf_bo/pull/26)) |
| 2026-05-03 | Alpha | Pixel-perm (Slice + Reshape + Gather + Reshape + Pad) | `multi-source-onnx-solver.ipynb` Tier-A from deep-read | `build_submission.py:pixel_perm_network` + `_detect_pixel_perm` + 10 unit tests | Kept — 3 hits (task214, task373, task385); cost ~2-7k → ~16-17 pts each. | same ([PR #26](https://github.com/Boladi888/NeuroGolf_bo/pull/26)) |
| 2026-05-03 | Alpha | Shape-changing-pixel-perm (same pipeline; in/out shapes differ) | ashok205 cluster B (108/194/223 confirmed) | `build_submission.py:shape_changing_pixel_perm_network` + `_detect_shape_changing_pixel_perm` + 6 unit tests | Kept — 3 net new hits + cost-aware reclassification of 4 prior mirror-grid-2x2 / grid-pattern-color-to-grid tasks (no count regression; pts gain). | same ([PR #26](https://github.com/Boladi888/NeuroGolf_bo/pull/26)) |
| 2026-05-03 | Both | honeydew #8 blend submission | ashok205 (107) + karnak (95); afr1ste 0 | `submissions/blend.zip` → renamed `submission.zip` | Kept — scored **3,275.37** (vs projected 3,303.17, 0.84% short); +14.22 vs grape #7. Smaller jump than grape because new patterns mostly displaced external winners rather than adding to the frontier. | session 2026-05-03 |
| 2026-05-04 | Both | mango #13 — pool refresh added needless090-v31 + konbu17-401-v117 | All 5 sources (ashok 91/k 68/n 36/k17 46/+overrides) | `submissions/blend.zip` → `submission.zip` | Kept — scored **4,042.19** (+646.47 vs lemon = biggest single jump since grape #7). Coverage 311/400 (was 255). | `Logs/sessions/2026-05-04_1057_both-40hr-push.md` |
| 2026-05-04 | Alpha | kron-block-up + const-largest-bbox + const-framed-cell | mining (task307) + partial-rule refinement (task100/346) | `build_submission.py` + 28 unit tests | Kept — +3 SOLVED, +43.10 organic LB pts. nectarine #14 = **4,043.11** = +0.92 vs mango (small marginal because 100/346 were external-covered already in mango). | same |

---

## Rejected patterns

| Date | Lane | Pattern | Source | Reason rejected | Source |
|---|---|---|---|---|---|
| 2026-05-02 | Beta | `Compress`-with-fake-zero-dim exploit | `neurogolf-2026-4-27-lb-5126.ipynb` | Ethical-play stance — catalogue but don't exploit. 3 bypass hypotheses tested locally, all REJECTED (validator's `_EXCLUDED_OP_TYPES` rejects bare/derived/dead-code Compress). | `Onnx_Advanced/scoring_internals.md#a2_cost_metric_exploits` |
| 2026-05-02 | Both | "Positive-cost wrappers" exploit family | Multiple competitor notebooks at 6,000+ LB | Ethical-play; decoded as 3 layered profiler exploits (dim_param substitution, initializer→Constant, Shape→ConstantOfShape→Add wrapper) but not used. | `Onnx_Advanced/scoring_internals.md#a2_cost_metric_exploits` |
| 2026-05-02 | Both | `ConstantOfShape` with data-derived shape | "Issues in onnx-tool" Kaggle thread | Ethical-play — confirmed exploitable (scorer's `shape_infer(None)` zero-fills input → output shape collapses → 0 MACs reported despite correct runtime output) but Yiheng Wang (#3) explicitly avoided it; we follow. | `03_PROJECT.md` Open bugs |

---

## Skip lists from competitors

Four of NG's 48 SOLVED tasks (53, 129, 266, 385) appear in others' skip lists. PR #26 added task385 to this group (it was in the Russian-language `SKIP_TASKS` primary list AND now SOLVED via pixel-perm). Submitted scores matched local prediction within 0.84% on every submission to date — pattern-match robustness is competitor-specific, not universal silent-fail. Run a careful pre-submission regression check on these four before any future submission, and spot-check the others if we ever target them.

| Competitor | Skip list | Source |
|---|---|---|
| Imaad Mahmood (`rule-based-onnx-solver.ipynb`, ~6,285 LB) | `EXCLUDED_TASKS = {21, 55, 80, 184, 202, 366}` | `Research_Archive/notes/deep-read-rule-based-onnx-solver.md` |
| `acking-multiple-tasks-part-3.ipynb` (Russian-language) | Primary `SKIP_TASKS = {47, 53, 99, 127, 258, 294, 331, 385}` plus secondary lists `{100, 103, 104, 129, 167, 171, 175, 176, 186}`, `{200, 248, 266, 274, 291, 292, 305, 325, 334}`, `{357, 373, 375, 376, 399}` — ~36 tasks total | same notebook |

---

*Last updated: 2026-05-03.*
