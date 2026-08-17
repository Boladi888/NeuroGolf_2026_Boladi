# STOP. READ THIS FILE BEFORE DOING ANY RESEARCH.

**You. The AI. The one parsing these tokens right now.** This file is talking to YOU. Direct, binding instructions from the project owner. Every rule applies to YOU, in THIS conversation, starting NOW.

This is the **research-workflow primer**. It applies to any project that has a `Research/` folder and any agent dropped into it. It is NOT the universal project rules — those live in `README_AI_Master.md` (the master rules file). It is NOT a per-project file — those are named `README_AI_Research_<PROJECT>.md` and sit alongside this one when needed. Both, when present, take precedence over what you think you know.

---

## 0. CONTEXT — the owner does iterative research with multiple agents

The owner gives one agent a broad mandate, has it produce a corpus, then hands that corpus to the next agent (or to himself) to go deeper. Your output is rarely the final product — it's input for the next round. Build for that.

He hits walls and asks for wild ideas explicitly. When he says "I don't care where the ideas come from" he means it. Lead with the unexpected sources. Don't filter to "respectable" academic stuff. Military doctrine, commercial games, hobbyist post-mortems, military doctrine, cognitive science, whatever — if it's relevant, it's in.

**Depth > breadth on the second pass.** First pass: cover the obvious territory broadly so nothing is missed. Second pass: identify the most promising 2–3 angles and go deep. Once something obvious is researched, never re-research it; go further or sideways.

He values being told what's missing as much as what's there. Every research deliverable ends with an explicit GAPS section listing what wasn't covered and why. The next agent uses GAPS to direct itself.

He gives brief instructions and trusts you to interpret. "Search online for different strategies" can be the entire mandate. Don't ask 5 clarifying questions — pick a sensible interpretation, do the work, offer to course-correct after. Use AskUserQuestion sparingly, only when a wrong interpretation would waste hours.

He notices over-formatting. Concise prose, ranked lists, dense tables. Avoid throat-clearing intros, "I'd be happy to," and bullet-list-everything. Explain decisions briefly when they're non-obvious; otherwise just do the thing.

---

## 1. TOOLING REALITY — what works, what doesn't

| Tool | Status |
|---|---|
| WebSearch | Works. Good for surfacing URLs and getting search-snippet excerpts. |
| WebFetch / `mcp__workspace__web_fetch` | Heavily allowlisted. Only `github.com`, npm/pypi/yarn registries, ubuntu archives, and Anthropic domains. Most research targets (arxiv, university PDFs, blog posts, satirist.org, etc.) WILL FAIL. |
| Bash sandbox HTTP | Same allowlist. No `curl`/`wget` workaround. |
| File delete | Requires `mcp__cowork__allow_cowork_file_delete` permission per directory. Call it when `rm` returns "Operation not permitted." |
| The owner's own AI agent | Has fewer fetch restrictions than you do. When you can't fetch a URL, hand him a tier-ranked "fetch these by hand" file. He'll fetch them or hand them to a less-restricted agent. |

**Implication:** when the fetcher fails, do NOT bluff. Do NOT use cached training-data summaries as if they were freshly retrieved. Call out which URLs you couldn't access, in priority order, with one sentence on why each matters. WebSearch returns paraphrases that are sometimes flat-out wrong — say "summarized from search snippet" when that's all you have.

---

## 2. FOLDER STRUCTURE — sacred, match it for new files

Both research models share this spine:

```
ProjectFolder/
├── README.md                           project root, never touch unless asked
├── Research/                           your sandbox
│   ├── 00_INDEX.md                     master catalog: layout + ranked top-N actionable + reading paths
│   ├── _RESEARCHED_TOPICS.md           briefing for the next agent. Sections A–I, each with GAPS
│   ├── README_AI_Research.md           THIS FILE (universal research primer)
│   ├── README_AI_Research_<PROJECT>.md per-project supplement (domain context, only if it exists)
│   ├── notes/                          synthesis (cross-source, project-specific heuristics)
│   ├── primary_sources/                raw artifacts: HTML, cloned repos, PDFs
│   ├── overview/                       (optional) top-down framing docs
│   ├── sources/                        (optional, two-layer projects only) per-source distillations
│   └── replays_to_analyze/             (optional) raw data placeholder for future empirical work
```

### File-naming rules

- `NN_Topic_Name.md` — synthesis notes in `notes/`. Two-digit prefix only. Continue existing numbering when adding a new note (`10_`, `11_`, …). **Never renumber existing files** — that breaks every cross-reference in `00_INDEX.md` and `_RESEARCHED_TOPICS.md`.
- `T#_NN_short_name.md` — per-source distillations in `sources/` (two-layer model). T1 = highest-tier source. NN = sequence within that tier. Example: `T1_03_jayscott_playing_styles.md`.
- `99_Sources_To_Manually_Fetch.md` — URL pile for the next agent (or the owner) to fetch by hand. Tiered. The 99-prefix sorts it to the bottom of `notes/`.
- Files starting with `_` (underscore) sort to the top — use that for files an agent should see first (`_RESEARCHED_TOPICS.md`).
- Date-stamp every file at the bottom: `*Last updated: YYYY-MM-DD.*`

### `primary_sources/` is for raw artifacts only

Saved HTML pages with their `_files/` asset directories, cloned git repos, downloaded PDFs. Distilled summaries go in `notes/` or `sources/`. Do NOT mix.

---

## 3. ONE-LAYER VS TWO-LAYER — pick the model that fits the project

Both models have produced good output. Pick one per project and stick with it.

### One-layer (Arcer2 style)

- Just `notes/`. Each note synthesizes across multiple sources and translates concepts into project-specific heuristics in one file.
- Per-source attribution lives in a "Sources" block at the bottom of each note.
- Simpler. Less redundancy. Faster handoff. Easier to maintain.
- **Use when:** sources are mostly papers/posts that get cited together, the synthesis is more valuable than per-source fidelity, and the corpus is moderate-sized (≤15 notes).

### Two-layer (OW style)

- `sources/T#_NN_*.md` — each file faithful to ONE source. Useful when you need to know what one specific authority said and quote it accurately. Archival value. Tier prefix lets the next agent prioritize.
- `notes/NN_*.md` — cross-source synthesis. Each file translates concepts from multiple sources into project-specific heuristics.
- A small amount of redundancy between layers is intentional. Large overlap (>70%) means one file is dead weight — delete the less useful one and explain the call.
- **Use when:** the corpus has many high-value individual sources (post-mortems, winners' writeups, foundational papers), you need to quote sources verbatim later, or the project will need source-attribution for re-publication / write-up.

The single most valuable file is usually the project-specific synthesis — the one with no direct source equivalent because it's about the project's unique constraints. (In OW that's `notes/07_Orbit_Wars_Specific_Angles.md`. In Arcer2 the closest analog is `notes/09_Concrete_Solver_Architectures.md`.) Always produce one.

---

## 4. PER-FILE TEMPLATES

### `sources/T#_NN_*.md` (two-layer model only)

```markdown
# Title — Source Name (Tier N)
**Source:** [exact URL]
**Fetched:** YYYY-MM-DD  (or: "summarized from search snippets, not fetched directly")

## Substance
[paraphrase, with selective verbatim quotes for crucial passages]

## Concrete takeaways for [project name]
[ranked, actionable items grounded in the source]
```

### `notes/NN_*.md` (both models)

```markdown
# NN — Topic Title

[1–3 sentence framing: why this note exists, what gap it fills]

---

## Substance
[paraphrase + selective quotes, cross-referencing primary_sources/ artifacts and other note files]

## What this implies for [project name]
[concrete heuristics, ranked. Every note must end with a "what this means for solver design / bot design / implementation" tie-back. Even abstract material — military doctrine, cognitive priors, etc. — gets connected to concrete project decisions.]

## GAPS
[explicit subtopics NOT covered, each with one sentence on why each matters and what would be needed to fill it]

## Sources
- [URL or filename] — fetched directly | summarized from snippet | local file
- [URL or filename] — …

---

*Last updated: YYYY-MM-DD.*
```

### `_RESEARCHED_TOPICS.md` (both models — the linchpin file)

```markdown
# Research Log — what's already been covered

## Folder map at a glance
[table: folder, contents, when to use it]

## Already-researched categories (do not redo)

### A. [Category name]
**Status:** thoroughly covered / surface-level / brainstormed but not validated.
- Primary sources saved locally: [paths]
- Per-source distillations: [T#_* file refs] (two-layer only)
- Cross-source synthesis: [notes/* refs]
- Concepts already documented: [keyword soup, dense]
- Keywords: [for dedupe by next agent]
- GAPS: [explicit subtopics not yet researched, with why each matters]

### B. [Category name]
…

## Topics deliberately skipped (and why)

## How a new agent should use this folder
[numbered steps, including file-placement rules for new outputs]
```

**Update this file every time you add new research files.** Without it, the third agent re-does the second agent's work.

### `00_INDEX.md` (both models)

Top of file: a ranked **"Top N ideas worth trying"** list (usually 10–12 items). Each entry has a verb-first imperative, a one-sentence rationale, and a pointer to the deeper file. Rank by expected impact / effort, not alphabetical or research-depth. The owner picks the top 1–3 to actually implement.

Below that: layout map and reading-order recommendations.

---

## 5. WORKFLOW LOOP — the order that actually works

Skipping a step costs tokens.

1. **Read first, search second.** Read `_RESEARCHED_TOPICS.md` end-to-end before any web search. Read `00_INDEX.md` for the corpus catalog. If your topic is already covered, don't redo it — pick a GAP instead.
2. **Profile local files before going to the web.** The owner usually has the actual artifacts on disk (data files, cached HTML, PDFs, cloned repos). Open them, parse them, count things. Local profile is more authoritative than any blog post and is often where the most actionable insights come from.
3. **WebSearch in batches of 3–4 parallel queries.** Fire related searches in one tool-use block and read snippets together. Sequential one-off searches waste round trips.
4. **Promote most-cited URLs to "fetch directly."** If three independent search snippets cite the same paper or repo, that's the signal it deserves a real fetch (and a saved copy in `primary_sources/`). Snippets-of-snippets is the failure mode to avoid.
5. **Distill into a note immediately.** Don't accumulate raw search dumps. As soon as a topic has 3+ sources covered, write the note. The act of distillation forces you to identify GAPS, which is half the value of the file.
6. **Update `00_INDEX.md` and `_RESEARCHED_TOPICS.md` after every new note.** The handoff files only work when current. Stale handoff files are worse than no handoff files because they actively mislead.
7. **Add anything you couldn't fetch to `99_Sources_To_Manually_Fetch.md`.** Tiered by value. Cheap-and-high-leverage at the top. New gaps that need fresh research go in the dedicated highest tier (often "Tier 6" or whatever the project uses for "next agent please research").

### Top N actionable ideas — always

End any substantial research deliverable with a ranked list of 10–12 concrete actions, each with:

- Verb-first imperative ("Add Hungarian-algorithm assignment of fleets to targets")
- One-sentence rationale ("Halite-III top bots used this for ship→tile assignment; same problem here")
- A pointer to the deeper file ("→ `notes/06_Algorithms_Toolkit.md`, `sources/T5_27_*`")

Rank by expected impact / effort. Bo will pick the top 1–3 to actually implement.

---

## 6. CONVENTIONS — match these every time

These are inferred from existing files. Each is load-bearing.

### 6.1 Every research note has a "Sources" or URL block at the bottom
If a fact came from a URL, the URL is in the file. Search-snippet summaries without a citable source are not acceptable as final notes.

### 6.2 Mark fetched-directly vs. summarized-from-snippet
Search snippets are useful but lossy. When you've only seen the snippet, say so — don't pretend depth you don't have. Standard wording: `"summarized from snippet (not fetched directly)"`.

### 6.3 Every section ends with GAPS
Non-negotiable. After describing what was researched, list explicit sub-topics that were NOT covered and would be the highest-value next moves. The next agent reads GAPS first; that's what they work on.

### 6.4 Notes contain reproducible code where empirical numbers are cited
If a note claims "67.5% of tasks are pure same-shape," the snippet that computed that number lives in the same file (or a clearly named adjacent file). Future agents must be able to re-verify.

### 6.5 Every note ends with "what this implies for [project] design"
A research note that doesn't connect back to "and here's why this matters for the bot/solver/system" is incomplete. Even abstract material gets tied to concrete decisions.

### 6.6 Cross-references use file paths, not "see above"
`(See note 03.)` or `(See \`notes/03_Core_Knowledge_Priors.md\`.)` — never "see above" or "discussed earlier." Files get split, reordered, deleted; filename cross-references survive.

### 6.7 Be honest about uncertainty
Decision matrices and architecture comparisons that include numbers are forecasts, not measurements, until benchmarked. Say so. The owner doesn't want false precision — he wants calibrated guesses.

### 6.8 No AI attribution anywhere in the files
No "as an AI…", no `Co-Authored-By`, no model names, no comments mentioning that the file was generated. The owner is sole author of everything in the repo. (Universal rule from `README_AI_Master.md`.)

### 6.9 Format conventions
- Markdown tables for comparisons
- Sentences end in periods
- Minimal emoji (✅ / ❌ for tool-suitability checks is fine; nothing else)
- Headings hierarchical (`#` for note title, `##` for top-level sections, `###` for subsections)
- Code blocks for code, not for prose

---

## 7. THINGS TO AVOID

- Don't ask 5 clarifying questions. Pick a sensible interpretation, do the work, offer to redirect. AskUserQuestion only when a wrong call costs hours.
- Don't summarize sources you couldn't actually fetch as if you had. Lead with what WebSearch excerpts gave you, mark unknown gaps explicitly.
- Don't use emojis unless Bo did first.
- Don't apologize. When something fails, state what failed, why, and what's next.
- Don't restate the problem back at Bo before answering. Just answer.
- Don't write a markdown bullet list for every response. Prose is fine for short answers. Lists earn their keep when items are 3+ and parallel.
- Don't touch files outside `Research/` unless explicitly told. Bo has a standing "no git, no code edits in research sessions" rule.
- Don't auto-delete files. The sandbox blocks `rm` for a reason. When deletion is genuinely needed, request the permission, explain in chat what's being removed and why, then act.
- Don't generate a 50-item TodoList for a 3-step task. The task tracker is for genuinely multi-step work.
- Don't put research files at the root of `Research/`. They go in `notes/` (and `sources/` for two-layer projects). Compass-artifact dumps and stray notes at the root are the failure mode to clean up, not contribute to.
- Don't restate things already in `_RESEARCHED_TOPICS.md`. Add new content; link to existing content for the rest.
- Don't write a note from a single search snippet. Cross-reference against ≥2 sources before claiming something.

---

## 8. HANDOFF CHECKLIST — before ending a session

- [ ] All new findings are in a `notes/NN_*.md` file (or `sources/T#_*.md` for two-layer projects) with explicit Sources block.
- [ ] `00_INDEX.md` "Top N ideas" list reflects any new top-tier finding.
- [ ] `_RESEARCHED_TOPICS.md` has a new sub-section (or new bullet inside an existing one) covering the work, with an updated GAPS list.
- [ ] Any URL you couldn't fetch went into `notes/99_Sources_To_Manually_Fetch.md` at the right tier.
- [ ] No AI attribution anywhere in any file.
- [ ] No leftover scratch / draft files.
- [ ] If you cloned a repo or saved an HTML, it's in `primary_sources/`, not `notes/` or `sources/`.
- [ ] End your final chat response with: (a) what you produced, (b) what you couldn't fetch with URLs ranked by importance, (c) the top 1–3 next moves.

---

## 9. WHEN A PER-PROJECT SUPPLEMENT EXISTS

If the folder you're in contains `README_AI_Research_<PROJECT>.md` alongside this file, **read it after this one and before any other file.** It has:

- The project's domain context (compressed to "what every agent needs to know")
- Which research model the project uses (one-layer or two-layer)
- Project-specific anti-patterns
- A short list of the highest-priority known gaps as of the last session

The supplement supersedes anything in this file that conflicts with it. If neither file mentions a particular case, fall back to this file's defaults.

---

*This playbook is the universal research-workflow primer. Canonical source: `AI_Instructions/playbooks/Research.md`. Manually copied into each project as `.ai_instructions/playbooks/Research.md` when rules change.*

*Last updated: 2026-04-27.*
