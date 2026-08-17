# Playbook: Research Handoff

## When to use

The owner wants research done on a topic and the doing-research agent does NOT have access to this project folder. They will receive ONE file (`_HANDOFF.md`) and respond with ONE markdown document.

If the receiving agent has full project access, skip this playbook — they read the project files directly.

## What you produce

A single self-contained file at `<project>/Research/_HANDOFF.md`. Overwrite any prior handoff. The file is gitignored.

## Refusals — when NOT to generate _HANDOFF.md

- **No `Research/` folder exists.** Tell the owner to scaffold the Research/ structure first per `playbooks/Research.md`. There is nothing to hand off if no research has been organized yet.
- **No `_RESEARCHED_TOPICS.md` AND no specific topic the owner asked about.** Without either a list of GAPS or a focused ask, the receiving agent has nothing to focus on. Tell the owner to either populate `_RESEARCHED_TOPICS.md` with at least the categories of interest and known GAPS, or to specify a topic in their request.

If neither refusal applies, proceed.

## What goes in `_HANDOFF.md` — 5-section structure

### 1. Header

```markdown
# Research Handoff Brief — <project key> — <YYYY-MM-DD>

You have ONLY this file. Your goal: research the GAPS in section 5 below
in the style described in section 3, and deliver findings as a single
markdown document with the structure shown in section 6. Do NOT redo any
topic listed in section 4 — those are already covered.

If you cannot fetch a URL, mark it explicitly. Do not bluff. Do not
substitute training-data summaries for fresh retrieval without saying so.

The owner will paste your response into a project session; an agent there
will file it into the project's Research_Archive/notes/ folder per the ingestion
rules at the bottom of this file.
```

If the owner specified a focus topic, add a paragraph: `**Focus topic for this handoff:** <topic>. Concentrate effort here; cover other GAPS only if time permits.`

### 2. Project context

Copy these sections verbatim from `<project>/.ai_instructions/03_PROJECT.md`:
- Project description
- GitHub URL, branch, license
- Tech stack
- Architecture and key design decisions
- (If there is a Pipeline / CLI / other architectural section, include it.)

SKIP these sections (noise for a researcher):
- TODO
- Open bugs / known issues
- Fixed bugs
- Dismissed critiques (unless they are RESEARCH-related — then include with a note)
- Where I left off
- File structure (usually irrelevant for research)

### 3. Research workflow primer

Copy `playbooks/Research.md` content in full. This is how the owner wants research delivered: concise prose, ranked actionables, GAPS sections, fetched-vs-snippet-marked sources, no AI attribution, etc.

If a per-project research supplement exists at `<project>/Research/README_AI_Research_<KEY>.md`, append its content under a `## Per-project research notes` subsection.

### 4. What's already been researched

Copy `<project>/Research/_RESEARCHED_TOPICS.md` in full.

If `_RESEARCHED_TOPICS.md` does not exist, state explicitly:

> No prior research log exists for this project. Treat all topics as open.
> Owner is asking for first-pass coverage on the focus topic in section 1.

### 5. Top-N actionable + open GAPS

Compose this section from:
- The "Top N actionable ideas" from `<project>/Research/00_INDEX.md` if it exists.
- The GAPS subsections from each category in `_RESEARCHED_TOPICS.md`, consolidated into a single ordered list. Drop duplicates. Preserve the rationale ("why this matters") that GAPS entries usually carry.

If the owner specified a focus topic in section 1, put related GAPS at the top of this list.

### 6. How to deliver findings

```markdown
Return ONE markdown document. Structure:

## <Topic 1>
[paraphrase + selective quotes; cross-reference any sources you used]

### Sources
- [URL] — fetched directly | summarized from search snippet | local file

### What this implies for <project>
[concrete heuristics, ranked by impact / effort]

### GAPS
[explicit subtopics not covered, each with one sentence on why each matters
and what would be needed to fill it]

(repeat per topic researched)

## Top-N actionable
[verb-first imperatives, ranked by impact/effort, one-sentence rationale,
pointer to whichever Topic block contains supporting material]

## URLs I could not fetch
[ranked by importance, one sentence each on why they would have mattered;
 the owner has fewer fetch restrictions and can retrieve these by hand]
```

## Ingestion — when the response comes back

The owner pastes the response into a project session. The agent there reads the markdown and:

1. **For each `## <Topic>` block in the response:** create a new file at `<project>/Research_Archive/notes/NN_<topic_slug>.md` using the next available `NN` two-digit prefix. NEVER renumber existing notes — that breaks every cross-reference in `00_INDEX.md` and `_RESEARCHED_TOPICS.md`.
2. **Update `_RESEARCHED_TOPICS.md`:** add the topic to the right category (or add a new category), list the new note file path, update the GAPS subsection (remove items the response addressed; add any new GAPS the response surfaced).
3. **Update `00_INDEX.md`:** if any new finding deserves promotion to the top-N actionable list, insert it. Re-rank if needed.
4. **Append unfetched URLs:** add the contents of the response's "URLs I could not fetch" section to `<project>/Research_Archive/notes/99_Sources_To_Manually_Fetch.md`. Tier them per the existing convention.
5. **Update `<project>/Logs/lessons.md`** if any durable lesson came out of the research that applies beyond this single topic.

This is the only ingestion step. There is no automation for it. The project agent does it during the next session, or the owner does it by hand.

## Notes on size

`_HANDOFF.md` will often be 30-100 KB depending on how much research has accumulated. That is fine — receiving agents can handle it. If it is approaching 200 KB, that is a signal to scope down: the owner probably wants research on ONE category, not the whole corpus. Ask before generating in that case.

---

*This playbook is universal across projects. Source: `AI_Instructions/playbooks/handoffs/ResearchHandoff.md`.*

*Last updated: 2026-04-27.*
