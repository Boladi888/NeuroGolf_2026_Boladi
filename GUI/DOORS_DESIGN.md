# Doors Maze — Gamified Rule-Discovery Mode for NeuroGolf Lab

**Status: MVP BUILT (2026-06-05) — §10 MVP scope is live; Phases 2–3 are not.**
Built: `door_engine.py` (12 v1 doors, lookahead, ONNX lowering), `/api/door/*`
endpoints, the Doors UI (default mode; Lab is one click away), auto-compile +
recipe-only export, 4 tutorials + 3 warm leads (`doors_campaign.json`).
Not built yet: Tier-2/3 hints (wizard), macro-hard doors, sandbox-paint,
corpus badge. The greedy door search alone already cracked 19 tasks on all
official sets — including NotSolved task150 (one flip!), task155, task177.

---

## 1. Why / thesis

153 of 401 tasks are Solved; 239 are NotSolved (9 HiddenBad). Looking at the
attempt history in `The_400_Tasks/Task_by_Task/taskNNN.md`, the bottleneck on
NotSolved tasks is almost never ONNX golfing — it is **rule discovery**:
figuring out *what the transformation is* and finding the creative algorithm
leap. Agents grind cost reduction well once a rule/recipe exists.

Humans — including children — are exceptionally good at exactly the missing
part: looking at grid pictures and going "oh, it's a shadow / it folds / the
big one eats the small ones." The game's one job is to harvest that:

> **Let a person find a transformation and *prove it works on every training
> example at once* — then hand the proven recipe to the machines.**

The golfing stays with the agents. The human plays.

---

## 2. Core loop — the maze of doors

A **solution is an ordered path of doors** from the input grid to the target
grid. Each door is an intuitive grid move (crop, mirror, recolor, tile…).

- **Every door applies to ALL train examples simultaneously.** This is the
  non-negotiable mechanic: a door that fixes example 1 but breaks example 2
  visibly drags the meter down. The player *feels* that a rule must
  generalize — "I drew one answer" becomes "I found a rule."
- **Board** (per train example): `input · current · target`, with the current
  grid overlaid by a match heatmap — matching cells glow green, wrong cells
  stay red.
- **Distance meter** = mean over examples of (matching cells / total cells).
  Shape mismatch = 0% for that example until shapes match.
- **Win** = every train example cell-exact → the path auto-runs on test +
  arc-gen (and later a corpus sample) as the oracle → Export offered.
- **Path** = a horizontal filmstrip of door icons + post-step thumbnails.
  Click any earlier step to **rewind and branch** — the maze is literally a
  tree the player backs out of. Undo pops the last door.
- **Parameters are picked by click-on-grid, never dropdowns.** Recolor: click
  the color to change (on a grid), then click the new color (palette).
  Crop-to-color: click the color. Mirror/rotate: click an axis/arrow overlay.
  A child never sees the word "parameter."

```
flowchart (conceptual)

  Player clicks a door
    → POST /api/door/step   (stateless: re-apply whole path to all train examples)
    → door_engine.py        (numpy apply_path + 1-step lookahead)
    → Board updates         (heatmap, distance %, top-5 doors glow)

  stuck?       → POST /api/door/hint    (family priors, borrow-from-similar)
  very stuck?  → POST /api/door/wizard  (one-shot `claude -p` / `codex exec` subprocess)

  train exact  → WIN
     all doors ONNX-macro → auto-compile → existing competition gate → real points
     any macro-hard door  → recipe JSON + Rule text → spec for the agent loop
```

---

## 3. Door taxonomy

### v1 — geometric-first (12 doors, ALL auto-compilable)

Doors execute in **numpy** on scalar grids (H×W ints 0–9, ≤30×30). Every v1
door also has a known **ONNX macro lowering** onto ops already in the GUI
compiler's `SUPPORTED_OPS` (`GUI/server.py`), so an all-v1 path auto-compiles
into a real scored model with zero agent involvement.

| Group | Door (`op`) | Params (how picked) | Effect | ONNX macro lowering |
|---|---|---|---|---|
| Paint 🎨 | `recolor` | from (click cell), to (click palette) | swap one color everywhere | Equal + Where |
| Paint 🎨 | `keep_color` | color (click) | all other colors → background | Equal + Where |
| Scissors ✂️ | `crop_content` | none | crop to non-background bbox | Slice (bbox computed at compile time per-example-invariant; falls back to recipe route if bbox varies) |
| Scissors ✂️ | `crop_color` | color (click) | crop to that color's bbox | Slice (same caveat) |
| Scissors ✂️ | `pad` | amount (±), color (palette) | add N-cell border | Pad |
| Mirror 🪞 | `flip_h` | none | reflect left↔right | Slice (step −1) |
| Mirror 🪞 | `flip_v` | none | reflect top↔bottom | Slice (step −1) |
| Mirror 🪞 | `rotate` | 90/180/270 (arrow clicks) | rotate | Transpose + Slice |
| Mirror 🪞 | `transpose` | none | swap rows/cols | Transpose |
| Magnet 🧲 | `tile` | reps h/w (± buttons) | repeat grid in a grid | Tile |
| Magnet 🧲 | `shift` | dx, dy (arrow clicks) | translate, fill with bg | Pad + Slice |
| Mirror 🪞 | `mirror_concat` | axis (arrow) | original + mirror side by side | Concat |

### v2 — macro-hard doors (numpy-only, recipe-export route)

Trivial to play, hard to encode in ONNX — they still win the game and emit a
recipe; they just route to recipe-only export for agents to golf:

`gravity` (drop non-bg cells toward a side) · `flood_fill` (seed cell + color)
· `largest_object` (keep biggest blob) · `count_recolor` (recolor objects by
size rank) · `outline` (border around objects) · `symmetrize` (fill to make
symmetric) · `overlay` / `mask_by` / `diff` (binary doors taking a second
operand from the **tray** — the original `input` or any earlier `stepN`).

The tray is hidden inside the Magic ✨ group so the basic loop stays
one-click. Color-and-Logical-family tasks (45) are explicitly **routed to the
wizard** rather than forced into child-friendly doors.

---

## 4. Novelty mechanisms — how play finds what AI didn't

1. **Weird sequences for free.** Branchable paths let humans try orderings
   (`crop_content → mirror_concat → overlay(input)`) that greedy machine
   search prunes. Every winning path is captured verbatim.
2. **"Say the rule."** A persistent text box. On win (or anytime), the typed
   sentence is saved into the recipe AND appended to the task's `.md`
   `## Rule` section — directly attacking the NotSolved tasks whose Rule is
   unknown. A failed session with a good sentence is still logged value.
3. **Sandbox-paint induction (phase 3).** Show a *new* (test/arc-gen) input
   with a blank output; the player **paints the answer**. The engine searches
   the door library for a short path reproducing all demonstrated pairs — if
   found, that path *is* the induced rule; if not, the (input, painted
   output) pairs go to the wizard to induce a custom door spec. Human
   demonstrates the function; machine induces the program.

---

## 5. Assist tiers — "doors after doors"

**Tier 1 — 1-step lookahead (always on, free).** On every step the engine
tries *every* door with auto-guessed params (recolor tries each present color
pair, crop tries content + each color…) across all examples and ranks by
distance gain. The **top-5 doors glow** on the rack with projected % gain.
Returned in the same `/api/door/step` response — play stays <100 ms.

**Tier 2 — knowledge priors (cheap, contextual).** From
`The_400_Tasks/task_state.csv`: `family`, `similar_tasks`, `recipe_hint`.
Family → ordered door suggestions seeded from
`Onnx_Advanced/Algorithm_Library/per_family_starting_points.md`
(Spatial → mirror/crop/tile · Object → crop_content/largest_object ·
Pattern → crop_content/tile/symmetrize · Color_and_Logical → wizard).
If `similar_tasks` lists a **Solved** sibling with a recorded door path,
offer **"Borrow from task NNN"** — load its winning path as a starting point.

**Tier 3 — "Ask the Wizard" (on-demand CLI agent).** When the player asks —
and ONLY when the player asks — the server spawns a **one-shot headless agent
subprocess**:

- default: `claude -p "<prompt>" --output-format json` (Claude Code / Opus)
- alternative: `codex exec` (OpenAI Codex) — backend selectable in the UI
- cwd = **project root**, so the agent can itself read `Onnx_Advanced/`
  (`task_techniques_index.md`, `per_family_starting_points.md`,
  `by_symptom.md`), `Task_by_Task/taskNNN.md`, etc. — this is the "search for
  doors that looks in onnxadv when getting far"
- prompt carries: train grids (compact text), current door path with
  per-step distances, the door-library spec, the task's `## Rule` /
  `recipe_hint`
- returns structured JSON `{ruleHypothesis, doorPath, customDoors, confidence}`;
  the GUI **replays** the suggested doorPath and shows the resulting distance
  so the player can accept, trim, or branch
- ~2–5 min timeout, "Wizard is thinking…" UI state, and the subprocess is
  covered by the launcher's kill-on-close JobObject (`GUI/launch.py`) — close
  the window, the wizard dies too. No orphans.
- **No file-handoff folders, no polling loops, no API keys in the GUI.**

---

## 6. From play to competition value

A winning path is a **verified recipe** (numpy-exact on train + test +
arc-gen; corpus sample later). Export route is chosen by the path's macro
flags:

**(a) Auto-compile — every door is a v1 ONNX macro.**
The door path is translated into an `ExportPayload` (`nodes`/`edges` — the
same contract the visual editor emits) and fed through the **existing live
pipeline**: `_competition_check` → `compile_competition_model` →
`validate_competition` (official one-hot semantics) → `_score_temp_model`
(official scorer) → `save_artifact` into
`The_400_Tasks/Task_by_Task/<task>/prototypes/`. Real points, zero agents,
**no new compile/validate/score code**.

**(b) Recipe-only — any macro-hard door present.**
Write `Task_by_Task/<task>/prototypes/<task>_doorpath_<ts>.recipe.json` and
append the human's rule text to the task `.md`. The agent loop receives a
*proven, replayable* spec — far stronger than free-text `recipe_hint`.

Both routes are **additive writes only** (timestamped, never overwrite),
identical to how GUI exports already behave alongside running agents.

### Recipe schema (`neurogolf-doorpath/v1`)

```json
{
  "schema": "neurogolf-doorpath/v1",
  "taskId": "task010",
  "createdUtc": "2026-06-04T18:22:05Z",
  "ruleHypothesis": "Rank the 4 vertical bars by height; recolor each bar by rank.",
  "win": { "train": "12/12", "test": "3/3", "arcGen": "262/262", "corpusSample": null },
  "exportRoute": "recipe_only",
  "allDoorsHaveOnnxMacro": false,
  "path": [
    { "step": 1, "op": "crop_content", "params": {}, "operand": "current",
      "distanceAfter": 0.71, "onnxMacro": true },
    { "step": 2, "op": "count_recolor", "params": { "by": "size_rank" }, "operand": "current",
      "distanceAfter": 0.93, "onnxMacro": false, "macroHardNote": "needs per-object size+rank" },
    { "step": 3, "op": "overlay", "params": {}, "operand": "step1",
      "distanceAfter": 1.0, "onnxMacro": true }
  ],
  "customDoors": [],
  "provenance": { "assistUsed": ["lookahead", "wizard:claude"], "saidRule": true }
}
```

The `path` is deterministic and replayable: `door_engine.apply_path(grid,
path)` reproduces it — one file is simultaneously the spec, the verification
record, and the agent handoff.

---

## 7. Latest solutions & the current-submission standard

The GUI formalizes the standard the project already half-follows:

| Question | Source of truth | GUI access |
|---|---|---|
| What models are live? | `submissions/submission.zip` (canonical 401-model zip) | `best_zip_path` / `best_onnx_bytes` / `best_manifest` (`project_bridge.py`), `/api/best-graph/{task}` |
| Per-task live cost/points/src? | `The_400_Tasks/task_state.csv` (`current_cost`, `current_points`, `current_src` incl. `OVERRIDE-*` markers) | `task_baseline` (`project_bridge.py`), `/api/task-baseline/{task}` |
| Which candidate zip is canonical? | `The_400_Tasks/submission_candidates.csv`, role = `current_valid_best` | (doc reference; GUI reads the zip directly) |
| How does a new best get promoted? | `tools/neurogolf_sync_current_best.py` after Kaggle confirms | out of GUI scope — promotion stays with the main workflow |

**Proposed "solution card" view** (per task, read-only): live cost + points +
arch (`live_arch`) · the task's `## Rule` text · the live best graph rendered
in the editor (already works via `/api/best-graph/{task_id}`) · attempts
history table parsed from the task `.md`. One screen answering "what do we
currently know about this task?" before playing it.

---

## 8. Game feel & progression

- **Campaign queue:** default = NotSolved, sorted by `unlock_potential` desc,
  geometric-friendly families (Spatial_and_Geometric, Object_Based) first.
  Tasks whose `similar_tasks` include a Solved sibling are flagged
  **"warm leads."**
- **Tutorials:** 3 Solved geometric tasks with known door answers — wins are
  guaranteed, "ghost doors" glow along the known path. Builds the loop's
  muscle memory in two minutes.
- **Stars:** ⭐ solve (train exact) · ⭐⭐ solve in ≤ par doors (par = greedy
  lookahead path length) · ⭐⭐⭐ auto-compiled AND officially scored.
  Beating the `task_baseline` triggers **"NEW BEST!"**
- **Feel:** sub-100ms feedback, the heatmap turning green, glowing doors
  ("the maze is helping me"), a friendly wizard, playful copy ("Snip!",
  "Flip!"). Solo/family tool: no accounts, no telemetry; a localStorage play
  journal is plenty — the recipe files are the only durable assets.

---

## 9. Architecture sketch (for the implementation session)

- **`GUI/door_engine.py` (NEW)** — numpy door library + `apply_path` +
  Tier-1 lookahead. Single source of truth for play, verification, and recipe
  replay. The door→ONNX-macro lowering lives here too (emits `ExportPayload`
  node/edge dicts).
- **`GUI/server.py` (FastAPI)** — new routes, same style as existing ones:
  - `POST /api/door/step` — stateless: `{taskId, path}` → grids + per-example
    distance + lookahead top-5 + win flag
  - `GET /api/door/library` — door taxonomy + param specs (data-driven rack)
  - `POST /api/door/hint` — Tier-2 priors (family, similar_tasks, recipe_hint)
  - `POST /api/door/wizard` — spawn the one-shot CLI agent; async task id +
    poll (or long-poll) for the 2–5 min wait
  - `POST /api/door/export` — all-macro → `ExportPayload` →
    `_competition_check` → `save_artifact`; macro-hard → `save_recipe` +
    `append_rule_md`
  - `GET /api/door/campaign` — ordered task queue
  - Reuse as-is: `compile_competition_model`, `validate_competition`,
    `_competition_check`, `_grid_to_onehot`, `_score_temp_model`,
    `_official_task_examples`.
- **`GUI/project_bridge.py`** — add `save_recipe`, `append_rule_md`,
  `campaign_rows`; widen `_TASK_STATE_FIELDS` with `family`, `similar_tasks`,
  `recipe_hint`, `unlock_potential`. Reuse `save_artifact`, `task_baseline`,
  `artifact_dir` (additive-only writes, unchanged).
- **Client** — `main.jsx` gets a **Lab ↔ Doors mode toggle**; the game lives
  in new `client/src/doors/` components: `DoorsView`, `DoorRack`, `DoorBoard`
  (reuses `ArcGrid`/`arcColors` + heatmap overlay), `PathStrip`,
  `WizardPanel`, `Campaign`. State: localStorage journal
  (`neurogolf-doors-v1`); server stays stateless.
- **Corpus check (phase 3)** — sample pairs from the argen cache
  (`tools/ng_harness/corpus.py` → `argen_cache/taskNNN.npz`) as the
  ⭐⭐⭐ "corpus-proven" badge on recipes.

---

## 10. Build phases

**MVP (~1–2 days):** `door_engine.py` with the 12 v1 doors + `apply_path` +
lookahead · `POST /api/door/step` + `GET /api/door/library` · `DoorsView`
(board, heatmap, meter, rack with glow, filmstrip, undo, click-on-grid
params) · win detection → auto-compile export via `_competition_check` ·
localStorage journal · 3-task tutorial campaign.
*Already auto-solves much of Spatial_and_Geometric (41 tasks) end-to-end.*

**Phase 2:** recipe-only export (`save_recipe` + `append_rule_md`) · Tier-2
hints + "Borrow from task NNN" · macro-hard doors + operand tray · "Say the
rule" + stars/par · solution card view · campaign queue.

**Phase 3:** Wizard (on-demand `claude -p` / `codex exec` subprocess) ·
sandbox-paint induction · corpus-proven badge · wizard-induced custom doors
(sandboxed numpy) extending the rack per-session.

---

**Again: nothing here is built.** React to the door list, the win→export
routes, the wizard contract, and the campaign ordering — then an
implementation session takes this doc and executes it.
