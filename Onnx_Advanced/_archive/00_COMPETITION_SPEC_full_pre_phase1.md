# The 2026 NeuroGolf Championship — Reference Documentation

> **STATUS as of 2026-05-01:** Verification pass **closed**. The "Verified Facts (2026-05-01 update)" section below (lines 7-148) is authoritative. All `[Inferred]` / `[Verify]` markers in the historical sections (after the "## `01-competition-overview.md`" divider) are now **SUPERSEDED** — retained for context and historical research only. Do not act on them.

> **CURRENT SCORER PATCH as of 2026-05-08:** the May 6 official update supersedes the 2026-05-01 cost/dependency rows where they mention `macs+memory+params` or `onnx_tool` as the scorer. Current scoring uses `memory + params`; official scoring no longer depends on `onnx_tool`; ORT profiler trace filenames need a unique `profile_file_prefix`; broader standard IR/opset values may work but should be probed before production.

> **Original prefatory note (pre-verification, retained for context):** The Kaggle competition pages (`/overview`, `/rules`, `/data`, `/discussion`, `/code`, `/leaderboard`, `/team`) returned only metadata via the available fetcher and could not be fully scraped during the initial research session (Kaggle's anti-bot protection blocked deep extraction). The documentation below combines what was directly retrievable with high-confidence inferences drawn from (a) the predecessor competition that uses an essentially identical structure ("NeurIPS 2025 – Google Code Golf Championship", same author/host team — Michael D. Moffitt et al. — and the same ARC-GEN benchmark backbone), (b) the host's own pages (`neurosynthetic.org`), (c) the IJCAI-ECAI 2026 program, (d) ARC-GEN GitHub announcements, and (e) third-party references.

---

## Verified Facts (2026-05-01 update)

> **Read this section first.** It consolidates the 2026-05-01 verification pass and supersedes the `[Inferred]` items in sections `01`–`05` below where they conflict. The original sections are retained for context and historical research.

### Source

The authenticated Kaggle CLI was used to download `neurogolf_utils/neurogolf_utils.py` (15,171 bytes, Apache-2.0, "Copyright 2026 Google LLC") via:

```
kaggle competitions download -c neurogolf-2026 -f neurogolf_utils/neurogolf_utils.py -p data/
```

That module is the validator's source of truth — it is the same code the official starter notebook calls into and is what `verify_network()` runs locally and (in essence) what the server-side scorer reproduces. Web-fetched Kaggle pages remain anti-bot-blocked as of 2026-05-01 (confirmed via fresh WebFetch attempts during this pass).

The competition data archive also includes 400 `taskNNN.json` files (ranges from ~24 KB to ~870 KB; ARC-GEN procedural samples bundled directly into each task file alongside the original ARC-AGI `train` and `test` examples).

### Hard facts from `neurogolf_utils.py`

| Item | Value | Notes |
|---|---|---|
| Per-file byte cap | **1,509,949 bytes** (`1.44 * 1024 * 1024`) | This is 1.44 **MiB** (binary), NOT 1.44 MB (SI = 1,440,000) and NOT 1,474,560 (1.44 × 1024 × 1000). Strict `>` check (cap is exclusive). |
| Disallowed op types (uppercased) | `LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS` | Compared via `op_type.upper()`. **`COMPRESS` is on the list and was missed by all prior inferences from the predecessor competition.** `If` is NOT named explicitly, but `If` requires `GRAPH` body attributes — see next row. |
| Subgraph attributes (`If`/`Loop` bodies, etc.) | **rejected** | Any node attribute of type `GRAPH` or `GRAPHS` causes `calculate_memory()` to return `None`, which scores the model as failed. Effectively bans `If`. |
| Custom ONNX functions | **rejected** | `model.functions` non-empty → fail. |
| Initializer / IO name collision | **rejected** | Initializer names cannot overlap with input/output names. |
| Allowed opset domains | only `""` and `"ai.onnx"` | Custom domains rejected. The opset *version* is not pinned by the validator (starter uses 10, but later versions of standard ai.onnx are accepted in principle). |
| Static shapes | **enforced** | Every tensor dim must have a concrete `dim_value`; no `dim_param`, no symbolic axes. Otherwise `calculate_memory` returns `None`. |
| Required input tensor name | `"input"` | Hard-coded in `run_network` / `verify_network`. |
| Required output tensor name | `"output"` | Hard-coded. |
| Tensor shape (input AND output) | `[1, 10, 30, 30]` float32 | Batch=1, channels=10 (one-hot color encoding), height=30, width=30. Models that don't accept/produce exactly this shape fail. |
| Default IR version (starter) | `10` | Modern (ONNX 1.17+). |
| Default opset version (starter) | `ai.onnx` 10 | Starter uses opset 10; newer is technically allowed but stay close to opset 10 unless you have reason to bump. |
| Constant folding | **old onnx_tool row superseded 2026-05-06** | Current scoring no longer depends on `onnx_tool.loadmodel(...)`; recheck any old constant-folding assumption against the ORT-profiler scorer before using it. |
| Memory metric | sum of `num_elements * dtype.itemsize` over `graph.value_info` (intermediate activations) **plus** declared inputs/outputs, excluding the named `"input"`/`"output"` tensors | Custom — NOT what `onnx_tool` reports natively. Reproduced in `calculate_memory()`. Initializers are accounted as `params`, not `memory`. |
| Per-task scoring formula | `points = max(1.0, 25.0 - log(max(1.0, memory + params)))` | Current May-6 epoch removes MACs from the score objective. Per-task **floor for correct networks: 1.0**; theoretical **max: 25.0 per task → 10,000 across 400 tasks**. Incorrect = **0** empirically. |
| Functional-correctness check | `np.array_equal(user_output, expected_output)` after thresholding `(result > 0.0).astype(float)` | Output one-hot decoded back to a grid; must match exactly. |
| Example subsets evaluated per task | `train`, `test`, `arc-gen` — **all three** | ARC-GEN procedural samples are bundled into each `taskNNN.json`; no separate ARC-GEN call needed at evaluation time. |
| Color palette | 12 entries: 0-9 real ARC colors, **10 = "no color"** (channel padding for grids smaller than 30×30), **11 = "too many colors"** (one-hot decode error indicator) | Encoding zeroes channels outside the actual grid; the decoder maps unset cells back to color 10. |
| Module-assumed data root | `/kaggle/input/competitions/neurogolf-2026/` | For local use, mirror the JSON files into a directory matching this layout, or monkey-patch `_NEUROGOLF_DIR` before `import`. |
| Python imports required | `numpy`, `onnx`, `onnxruntime` for official scoring; local display helpers may still import `onnx_tool` | Host forum post 2026-05-06 says official scoring no longer depends on `onnx_tool`. Keep local helper stubs for notebook/display compatibility only. |
| Module syntax floor | walrus operator (`:=`) used in `check_network` | Requires Python ≥ 3.8. |

### Leaderboard snapshot (2026-05-01 03:50 local)

Pulled via `kaggle competitions leaderboard -c neurogolf-2026 --show`. Top scores:

| Team | Score | Submitted |
|---|---|---|
| Chris Deotte | 6,983.27 | 2026-05-01 07:36 UTC |
| GPT or Claude? | 6,845.42 | 2026-05-01 06:12 UTC |
| Pavel | 6,417.66 | 2026-05-01 06:45 UTC |
| Bogo de Chão | 6,362.17 | 2026-05-01 07:46 UTC |
| ymg_aq | 6,272.91 | 2026-05-01 04:23 UTC |

Top-12 range: ~5,800 → 6,983. Theoretical max: 10,000. Theoretical floor (all 400 tasks correct + maximally bloated): 400. Actual baseline ("first valid ONNX file submitted") is unknown — our first-submission target.

### Verified from Kaggle Overview + Rules (2026-05-01 owner copy)

The owner accepted the competition rules on 2026-05-01 and pasted the full Overview + Rules text into the working conversation. The previously [Owner-verify] items are now resolved:

| Item | Verified value | Source |
|---|---|---|
| Final submission deadline | **2026-07-15, 23:59 UTC** | Overview → Timeline |
| Entry deadline | **2026-07-08, 23:59 UTC** | Overview → Timeline |
| Team-merger deadline | **2026-07-08, 23:59 UTC** (same day as Entry) | Overview → Timeline |
| Total prize pool | **$50,000** (vs. $100,000 predecessor; do NOT use predecessor numbers) | Overview → Prizes |
| Prize tiers | 1st **$12,000** / 2nd **$10,000** / 3rd **$10,000** / Top Student Team **$8,000** / Longest Leader **$10,000** | Overview → Prizes |
| Student Team definition | "graduate or undergraduate students comprise 50% or more of the overall membership" | Overview → Prizes |
| Longest-Leader window | **2026-04-16 00:00 UTC → 2026-07-15 23:59 UTC** | Overview → Prizes |
| Daily submission cap | **5 submissions / day** | Rules §2.2.a |
| Final-selection count | **2 final submissions** for private-leaderboard scoring | Rules §2.2.b |
| Team size limit | **5** | Rules §2.1.a |
| Embargoed countries | Crimea, DNR, LNR, Cuba, Iran, North Korea + parties under US export controls / sanctions | Rules §3.1.a |
| Winners' license | **Apache 2.0** (Open Source Initiative-approved permissive). Must include training code, inference code, computational-environment description, and reproducibility instructions | Rules §2.5 + §1.6 |
| Data-use license | **Apache 2.0** (Competition Use AND Commercial allowed) | Rules §1.7 + §2.4.a |
| External-data policy | Permitted if publicly available and equally accessible (or meets the "Reasonableness Standard" — small subscriptions OK; license costs > prize amount NOT OK) | Rules §2.6 |
| Pre-trained models / AutoML | Permitted if licenses are compatible with all the rules above | Rules §2.6.c |
| Public + private leaderboards | **Yes — both.** Public leaderboard during competition; **Private leaderboard determines final standings** | Rules §3.7.a |
| Governing law | California (Santa Clara County) | Rules §2.9.a |
| Sponsor address | The Neurosynthetic Research Institute, P.O. Box 90184, 6104 Old Fredericksburg Rd, Austin, TX 78749 | Rules §1.3 |

### Additional facts now confirmed from Overview + Rules

- **Organizer citation:** *Michael D. Moffitt, Walter Reade, Ashley Oldacre, Addison Howard. The 2026 NeuroGolf Championship. https://kaggle.com/competitions/neurogolf-2026, 2026. Kaggle.* — Reade and Howard recur from the Code-Golf 2025 team; Thakkar / Burnell / Firat / Dane drop; Oldacre is new.

- **Functional correctness is checked against THREE benchmark sets**, all of which the network must pass to earn points on that task:
  1. The original ARC-AGI public training v1 benchmarks (the `train` and `test` examples in `taskNNN.json`).
  2. The ARC-GEN-100K dataset (procedurally generated; bundled into `taskNNN.json` as the `arc-gen` subset).
  3. **A small private benchmark suite** held out to prevent overfitting. **This is the basis of the Private Leaderboard**. Solutions hand-tuned to the visible benchmarks alone will fail this set — generalization matters.

- **Worked example in the Overview** (calibrates the cost↔points mapping): a single-layer 3×3 Conv2D over a 30×30×10 grid has **900 params + 39,600 bytes memory + 810,000 MACs = 850,500 cost** → score `max(1, 25 - ln(850,500)) ≈ 11.35` points/task. `neurogolf_utils.single_layer_conv2d_network(weight_fn, kernel_size)` (validator file line 314) is the reference helper for this baseline.

- **Disallowed-op discrepancy: Overview lists 6 ops; validator code lists 7.**
  - Overview text: `Loop, Scan, NonZero, Unique, Script, Function` (6).
  - Validator code (`_EXCLUDED_OP_TYPES`): `LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS` (7).
  - The Overview includes the clause: *"if we have to ban additional ONNX operators that compromise the aims of our contest—we will announce such changes and rescore submissions as needed."* This almost certainly explains the discrepancy — `COMPRESS` was added after launch and the Overview text was not updated. **Always trust the validator code over the Overview text.** Likely a `/discussion` forum announcement; worth confirming during Phase 1.

- **IJCAI 2027 Demo Track retrospective:** members of winning teams will be invited (not required) to collaborate with the organizers on a contest-retrospective submission to the IJCAI 2027 Demo Track.

- **IJCAI-ECAI 2026 special session:** held in Bremen, Germany; top submissions invited to give talks; attendance not required to participate; travel/expenses self-funded.

- **Participation snapshot (2026-05-01):** 3,880 Entrants / 944 Participants / 878 Teams / 22,370 Submissions.

- **Server-side floor for incorrect tasks: confirmed 0.** (Empirically verified 2026-05-01 by submitting 400 identity-weight networks — total publicScore was exactly **0.00**, not 400×0.001 = 0.4 like the predecessor's floor would have produced. NeuroGolf scoring awards literal 0 for incorrect tasks; the `max(1.0, ...)` clamp in the formula applies only to correct networks.)

### Cost-lever measurements (probe results, 2026-05-01)

`build_submission.py --probe` measures cost across architecture variants on a throwaway file. Results:

| Architecture | macs | memory | params | cost | pts_max if correct | bytes |
|---|---:|---:|---:|---:|---:|---:|
| `Identity` (single op) | 0 | 0 | 0 | **0** | **25.000** | 111 |
| `Transpose` H↔W (single op) | 0 | 0 | 0 | **0** | **25.000** | 131 |
| Conv 1×1 (identity weight) | 90,000 | 0 | 100 | 90,100 | 13.591 | 572 |
| Conv 3×3 (identity weight) | 810,000 | 0 | 900 | 810,900 | 11.394 | 3,772 |
| Conv 5×5 (identity weight) | 2,250,000 | 0 | 2,500 | 2,252,500 | 10.372 | 10,172 |

**Key finding:** any task solvable by a *single shape-only op* (`Identity`, `Transpose`, and likely also `Slice`, `Gather`, `Reshape`, `Concat` with small initializers) achieves **cost = 0 → max 25 pts/task**. The Conv-based reference architecture in the Overview's worked example is two-thirds of the way down the cost ladder, leaving substantial headroom. Constant-folding (item 3 in the strategic notes) is moot for shape-only ops — they're already at zero cost; nothing to fold.

**Caveat — generalization to arbitrary grid sizes:** the network operates on a `[1,10,30,30]` canvas with the actual grid embedded at the top-left and the rest zero-padded. Identity and Transpose preserve the embedded-at-top-left convention for arbitrary grid sizes (a 5×3 grid transposes to a 3×5 grid in the same corner; padding cells stay padding). `hflip`/`vflip` via `Gather` with reversed indices DO NOT generalize this way — they mirror the entire 30-wide canvas, which moves the embedded grid from the left edge to the right edge and breaks the decoder's trailing-padding-strip assumption. Only safe-for-arbitrary-size shape ops are usable as detected-pattern solutions.

### Pinned dependency versions (from `mmoffitt`'s official starter notebook, 2026-05-01)

The official starter notebook (`kaggle kernels pull mmoffitt/the-2026-neurogolf-championship`) opens with explicit `pip install` lines pinning the exact versions the validator and scorer expect. **These are the versions to install in `.venv/`** — `onnx-tool` version mismatch was flagged in forum thread `/discussion/692827` as a known scoring-discrepancy source, so this pin matters:

| Package | Version |
|---|---|
| `numpy` | **2.4.4** |
| `onnx` | **1.21.0** |
| `onnxruntime` | **1.24.4** |
| `onnx-tool` | **1.0.1** |

The starter notebook also shows the canonical `sys.path.append("/kaggle/input/competitions/neurogolf-2026/neurogolf_utils")` import pattern, confirming the path layout we mirrored locally to `data/competitions/neurogolf-2026/neurogolf_utils/`. The starter's reference workflow is a 6-cell flow: install → import + `show_legend()` → `examples = load_examples(task_num)` + `show_examples(...)` → write a `weight(channel_out, channel_in, kernel_coord)` function → `network = single_layer_conv2d_network(weight, kernel_size=3)` → `verify_network(network, task_num, examples)`. **`task_num = 0` is illustrative and not eligible for points** — real submissions are tasks 1-400.

### Confirmed-by-corroboration items (no change)

- ARC-GEN GitHub README (publicly fetched 2026-05-01): ARC-GEN is the official benchmark generator (announced 2026-04-04); ARC-AGI-1 = 400 tasks; ARC-AGI-2 = 500 additional tasks (NeuroGolf scores **ARC-AGI-1 only**); paper arxiv 2511.00162.
- The competition data archive includes `neurogolf_utils/neurogolf_utils.py` plus 400 `taskNNN.json` files (file-listing pulled via authenticated CLI).
- X.com post `kaggle/status/2044494732504522780` (the launch announcement) is paywalled to anonymous WebFetch (HTTP 402); content already partially captured in section 05.

---

> **⚠ Everything below is the pre-2026-05-01 inferred draft.** All `[Inferred]` and `[Verify]` markers from this point on are **SUPERSEDED** by the "Verified Facts" section at the top of the doc. Retained for historical research only. Treat the top sections as authoritative.

---

## `01-competition-overview.md`

# 01 — Competition Overview

## Title
**The 2026 NeuroGolf Championship** (also referred to as the "IJCAI-ECAI 2026 NeuroGolf Challenge").

## Tagline
"Design the smallest neural networks to solve ARC-AGI image transformations."

## Canonical URL
- Kaggle: <https://www.kaggle.com/competitions/neurogolf-2026>
- Sub-pages: `/overview`, `/data`, `/code`, `/discussion`, `/leaderboard`, `/rules`, `/team`, `/submissions`

## Premise
Hosted by the **Neurosynthetic Research Institute** in conjunction with the **IJCAI-ECAI 2026 Competitions Track**. The challenge pivots from pure predictive accuracy to **architectural miniaturization** — the goal is to design the *smallest possible* neural networks (delivered as ONNX graphs) that correctly solve ARC-AGI image transformations. Models must aggressively minimize their computational graph and memory footprint while still solving the 400 public training-set tasks of ARC-AGI-1.

This is the spiritual successor to the **NeurIPS 2025 — Google Code Golf Championship** (which scored Python source-byte minimality on the same 400-task suite); NeuroGolf 2026 changes the medium from Python source code to **ONNX neural-network graphs**.

## Host Organization — The Neurosynthetic Research Institute
- Web: <https://neurosynthetic.org/>
- Self-description: *"The Neurosynthetic Research Institute is dedicated to advancing the state-of-the-art in neurosymbolic function synthesis."*
- The site is minimal — a logo and an announcement linking to a Google Sites page (`sites.google.com/corp/view/neurogolf-2026/home`, internal-corp link not publicly fetchable) for the multi-month Kaggle competition.
- The competition is organized by the team behind the prior NeurIPS 2025 Code-Golf Championship; that competition was credited to *Michael D. Moffitt, Divy Thakkar, Ryan Burnell, Orhan Firat, Walter Reade, Sohier Dane, and Addison Howard*. Moffitt is the author of the underlying ARC-GEN paper (`arxiv.org/abs/2511.00162`) and works at Google. **[Inferred — verify NeuroGolf 2026 organizer list on Kaggle.]**

## IJCAI-ECAI 2026 Connection
- **Conference:** 35th International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026)
- **Dates:** August 15–21, 2026
- **Location:** Bremen, Germany
- **Track:** IJCAI-ECAI 2026 Competitions and Challenges Track (competition sessions held 15–21 August 2026, AoE).
- **Top-team obligations [Inferred from Code-Golf 2025]:** Top-finishing teams will be invited to give a talk at a special competition session at the IJCAI-ECAI 2026 conference. **Attendance is not required to participate** in or win on the leaderboard, but **only teams attending in-person are typically eligible to present**. Travel and accommodation are at the team's own expense.
- No paper submission is required to compete. Competitions track entries do **not** pay the IJCAI-ECAI primary-paper $100 fee (Competitions are explicitly excluded from that fee).

## Timeline
- **Competition launch:** April 15, 2026 (announced via Kaggle's X/Twitter at 2:15 PM on Apr 15, 2026).
- **ARC-GEN tooling alignment:** April 4, 2026 — ARC-GEN announced as the official benchmark generator for NeuroGolf 2026 (per the ARC-GEN GitHub README).
- **Final submission deadline / Entry deadline / Team-merger deadline:** specific dates are listed on the Kaggle competition's *Overview › Timeline* tab. Based on the IJCAI-ECAI conference window (Aug 15–21, 2026) and the standard Kaggle pattern used by the predecessor competition, the deadlines are expected to fall **shortly before the conference** (i.e., late July or early August 2026), with the entry and team-merger deadlines a week before final submission. **[Inferred — confirm exact dates on Kaggle's Overview › Timeline tab.]**
- All deadlines are at **11:59 PM UTC** on the listed day unless otherwise specified.
- Organizers reserve the right to update the contest timeline.

## Prize Structure
The Kaggle prize-tab text could not be fetched, but the predecessor competition (Google Code Golf 2025) used a $100,000 total pool with a "Longest Leader" prize ($5,000) for the team holding 1st place the longest during a defined window, plus standard 1st/2nd/3rd cash prizes and a Student Team prize. NeuroGolf 2026's prize structure is **expected to mirror this** but **must be verified on the Kaggle Overview › Prizes tab**. Known/expected components:

- **Top-rank prizes** (1st, 2nd, 3rd) — cash. **[Verify on Kaggle.]**
- **Student-Team prize** — separate award for top-ranking team where every member is a student (typically defined by Kaggle as currently enrolled, see Rules doc). **[Verify.]**
- **Longest Leader prize** — awarded to the team that has held 1st place on the public leaderboard for the longest cumulative time during a defined window (in Code-Golf 2025: between July 31 and October 30). For NeuroGolf 2026, the dates align with the competition window. **[Verify the exact date window on Kaggle.]**
- **Payments:** Winners are paid by the host (Neurosynthetic Research Institute / its sponsor) after rules verification, license/code submission, and tax-form completion. Recipients are responsible for **all taxes** in their jurisdiction; payment may be subject to W-9/W-8BEN or equivalent tax-form collection.

## TL;DR
- It's an ONNX "code-golf for neural networks" competition: write the smallest ONNX model per task that still solves all ARC-AGI-1 training tasks (400 of them).
- Hosted by the Neurosynthetic Research Institute, featured at IJCAI-ECAI 2026 (Bremen, Aug 15–21).
- Started April 15, 2026; ends shortly before the conference. Verify exact deadlines, prize amounts, and Longest-Leader window on the Kaggle page itself.

---

## `02-rules-and-eligibility.md`

# 02 — Rules and Eligibility

> The full rules text on `kaggle.com/competitions/neurogolf-2026/rules` could not be retrieved during research due to Kaggle's anti-bot protection. The items below combine Kaggle's standard competition rules framework, the predecessor (Google Code Golf 2025) rules, and the conventional ARC-Prize-style requirements that the host has previously used. **Always read the actual Rules page before competing for prizes.**

## Eligibility (standard Kaggle competition rule template)
To enter and be prize-eligible you generally must:
- Be at least the **age of majority** in your jurisdiction (typically 18+, or have parental consent if 13–17 and where local law allows). Kaggle accounts require users to be 13 or older.
- **Not** be a resident of a country/region embargoed by the U.S. (e.g., Cuba, Iran, North Korea, Syria, the Crimea/DNR/LNR regions, Russia where applicable). Specific country list will be reproduced on the Kaggle Rules page; verify there.
- **Not** be on any U.S. or other applicable sanctions list.
- Have a single, verified Kaggle account. Multiple accounts ("sock-puppets") are forbidden and disqualifying.
- Accept the competition rules through the Kaggle UI **before the entry deadline**.

## Team Rules
- **Team size limit:** typically **5 members maximum** under Kaggle's standard rule template. **[Verify on the Rules page — Code-Golf 2025 used 5; NeuroGolf 2026 may differ.]**
- **Team merger deadline:** the last day teams can merge or new members can be added. Teams trying to merge after this date are blocked.
- A user may only be on **one team** per competition.
- All members of a team must individually accept the rules.

## Code-Sharing Rule (standard Kaggle policy)
Privately sharing code/data outside of the team is prohibited. It is **fine** to share code publicly via the competition's Code/Discussion forums, **provided** the shared code is made available to *all* participants under an open license (the default expected: same terms as winners' code — i.e., MIT or similar permissive). Any code or data shared with a subset of competitors but not the public constitutes cheating.

## External Data and Pre-Trained Models
- **External data:** Permissibility of additional ARC-style datasets (e.g., ARC-GEN procedural tasks, ARC-AGI-2 set, ARCTraj, etc.) for *training* depends on the rules text. Kaggle's standard policy: external data is permitted **only if it is freely and publicly available to all competitors and is shared on the forums**. **[Verify the specific data policy on Kaggle.]**
- **Pre-trained models:** Generally permitted only if publicly available with a permissive license; must be disclosed in the forums. (Note that for NeuroGolf the *submission artifact* is just an ONNX file with weights — there is no training-data restriction at inference time per se, but rules apply to how those weights were obtained.)
- The submitted ONNX model itself is what is scored — the validator inspects the graph, so any "pretraining" simply manifests as the ONNX file's weights.

## Submission Constraint Summary (rule-relevant)
- One submission file per submission slot: a **`submission.zip`** containing files named **`task001.onnx` … `task400.onnx`**.
- ONNX file size limit: **1.44 MB per file** (this number is symbolically chosen — the entire ARC-AGI-1 dataset fits in 1.44 MB, the size of a 3.5″ floppy disk; reference: Moffitt 2025 ARC-GEN paper). **[Confirm the exact byte cap on the Rules page.]**
- **Disallowed ONNX operators (for prize eligibility):** `Loop`, `Scan`, `NonZero`, `Unique`, plus any user-defined `Function` or `Script` operators. The validator rejects models containing these. **[Confirm complete list on Rules — these are the most commonly cited; subgraph-control-flow ops like `If` may also be restricted.]**
- All ONNX inputs/outputs are expected to use **static shapes** (no symbolic dimensions); models must accept and produce fixed-shape tensors that match the task spec. **[Verify.]**
- Operations not on Kaggle's allow-list will fail the network validator and earn 0 (or a near-zero floor like 0.001) points for that task.

## Winners' Obligations (standard pattern, derived from sibling competition)
To collect any prize, winners typically must, within ~7 days of being notified:
1. **Open-source the entire winning solution** (code that produced the ONNX models, training scripts, ARC-GEN usage, post-processing) under a **permissive license** — the predecessor used MIT-0 / CC0 style. NeuroGolf 2026 likely requires **MIT, Apache-2.0, or CC0**. **[Verify the exact licenses on Rules.]**
3. Grant the Competition Sponsor a **non-exclusive, irrevocable, worldwide, royalty-free license** to the winning submission and its code/documentation for any purpose, including reproduction, redistribution, and modification.
4. Provide a **technical write-up / documentation** of the approach (typically posted on the competition forum).
5. Provide the **trained model artifacts** if not already in the submission.
6. Sign and return any required eligibility / tax forms (W-9 for U.S. persons, W-8BEN/W-8BEN-E for non-U.S.).
7. Cooperate with verification and reproduction efforts by the host.

Failure to meet winners' obligations within the stated window forfeits the prize, which then passes to the next-eligible team.

## Disqualification Conditions
- Multiple Kaggle accounts.
- Private code or data sharing outside one's team.
- Use of disallowed ONNX operators or other validator bypass attempts.
- Cheating, collusion, or any reverse-engineering of the private validation set.
- Submitting models you do not have rights to (e.g., weights derived from non-permissively-licensed pretrained models).
- Failure to verify identity, age, or eligibility when requested.
- Any breach of Kaggle's Terms of Use.

## License Terms
The competition's Sponsor (Neurosynthetic Research Institute) presumptively asserts: **(a)** for non-winners, you keep your own IP; **(b)** for winners, you grant a permissive license to the Sponsor and to the public, and (c) the data provided in the competition is licensed for use **only within the competition** unless its underlying license (e.g., ARC-AGI's permissive license, ARC-GEN's Apache-2.0) provides otherwise. **[Verify on Rules page — these are standard Kaggle clauses.]**

## TL;DR — Rules
- Standard Kaggle eligibility (18+/age of majority, no embargoed countries, one account, accept rules before entry deadline).
- Open-source winning code under a permissive license; grant Sponsor a license; provide a write-up.
- Don't use disallowed ONNX ops (`Loop`, `Scan`, `NonZero`, `Unique`, `Function`, `Script`) and stay under the per-file size cap (~1.44 MB).
- **Read the actual Rules page to confirm.**

---

## `03-submission-and-environment.md`

# 03 — Submission Mechanics & Environment

> The /overview, /code, and /discussion sub-pages were not directly readable due to Kaggle's bot protection. Mechanics below combine sibling-competition norms with confirmed details from the discussion thread "Issues in onnx-tool" (`/discussion/692827`) and from the ARC-GEN GitHub repo.

## Competition Type
NeuroGolf 2026 is **a regular submission competition** (not a Code Competition). You upload a `submission.zip` from anywhere — **internet access during submission is irrelevant**, since the submission itself is just artifact files (ONNX models). There is no notebook required to run on Kaggle servers; you train and export the ONNX files locally (or wherever) and upload the zip. (Compare ARC-Prize, which *is* a code competition with sandboxed L4 GPU evaluation — NeuroGolf is **not** that.)

This is consistent with the Code Golf 2025 sibling competition, in which a `submission.zip` of `task001.py` … `task400.py` was uploaded.

## Submission Format
- **File name:** `submission.zip`
- **Contents:** ONNX model files named **`task001.onnx`** through **`task400.onnx`**.
- Tasks correspond 1:1 to the **400 ARC-AGI public training set v1 tasks** (the same 400 tasks used by Code Golf 2025 and ARC-GEN).
- **Partial submissions are allowed** — you don't need all 400 files. Tasks without a matching `taskNNN.onnx` simply earn 0 (or the failure-floor value, ~0.001). **[Verify exact behavior on Kaggle; sibling competition allowed partials.]**
- Each ONNX file must conform to the validator's constraints (see "Validator" below).

## Scoring
The exact scoring formula on the live Kaggle page is the source of truth. Based on the Code-Golf 2025 model (which used `max(1, 2500 - length)` for source-byte length, with 0.001 for incorrect), and on the ARC-GEN paper's emphasis on the **1.44 MB** dataset-size symbol, NeuroGolf 2026 awards each task a score that **rewards smaller networks** (likely measured in graph parameter count, raw byte size of the ONNX file, or computational-graph operation count) **provided the network is functionally correct on all (input, output) pairs evaluated through ARC-GEN**. **[Confirm exact metric on the Overview page.]**

- **Functional correctness** is checked via ARC-GEN: each ONNX model is evaluated against (likely many) procedurally generated `<input, output>` grid pairs for that task. ARC-GEN supports the original 400 ARC-AGI-1 training tasks plus an extension to 500 ARC-AGI-2 tasks (per its March 2026 update).
- **Incorrect** programs get a near-zero floor (the predecessor used 0.001).
- The **total score** is the sum of per-task scores. Higher is better.
- Public leaderboard scores are visible during the competition; a **private leaderboard** typically uses a held-out evaluation set (different generated samples). **[Verify whether NeuroGolf has a public/private split — sibling competition's scoring grew over time as constraints were enforced.]**

The forum issue thread "Issues in onnx-tool" (`/discussion/692827`) confirms that scoring uses the **`onnx-tool`** library to compute model statistics (likely MACs / parameter count / FLOPs), and there have been participant-reported anomalies that suggest scoring is metric-driven through that library.

## Daily Submission Limits
Kaggle's standard cap is **5 submissions per day** per team for most competitions; the Code-Golf-2025 sibling competition used the same. **[Verify on Kaggle's submission page — typically displayed as "Submissions 0/5 today".]** Total submission limit (over the course of the competition) is usually unlimited as long as daily cap is respected, but participants pick a **final selection** of N submissions (commonly 2) for private-leaderboard scoring.

## Network Validator
The competition runs an **official network validator** on each submission before scoring. Confirmed/expected behaviors:
- **Disallowed operators**: rejects any model containing `Loop`, `Scan`, `NonZero`, `Unique`, custom `Function`s, or `Script`-based ops. **[Verify completeness — these are confirmed in the user-task description; `If` and dynamic-shape ops may also be restricted.]**
- **File-size limit**: **1.44 MB** per `taskNNN.onnx`. Files exceeding this are rejected.
- **Static shapes**: all input and output tensors must have fully resolved (non-symbolic) shapes. Shape-inference must succeed without dynamic axes. **[Verify.]**
- **Standard ONNX opset**: only standard ONNX operators from the supported opset version are accepted; the validator's expected opset version is documented in `neurogolf_utils` (see Resources doc).
- **Determinism**: the validator runs the model deterministically and compares outputs against the expected grids generated by ARC-GEN.
- **Failed validation** for any reason → that task receives the failure score (0.001 or 0).

## Compute / Hardware Constraints During Evaluation
- Because models are uploaded as artifacts (not run interactively in a Kaggle notebook), **per-team GPU/TPU runtime caps from Kaggle are not relevant** to evaluation.
- However, the **organizer's evaluation server** runs each ONNX model with **ONNX Runtime** (CPU-class, per discussion-forum chatter on operator/profiler behavior). Per-task **inference time limits** likely apply — the strict cap is small because each task is just a forward pass on small grids. **[Verify exact limits on Kaggle Overview.]**
- There is no internet access during evaluation (models are pure ONNX inference graphs with weights baked in).

## Submitting from Kaggle Notebooks vs. From Anywhere
You can build your `submission.zip` in any environment — local machine, your own GPU server, Kaggle notebook, Colab, etc. — and upload it via the Kaggle "Submit Predictions" UI or via the Kaggle CLI (`kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "<message>"`).

## TL;DR — Submission
- Upload `submission.zip` containing `task001.onnx` … `task400.onnx` (partials OK).
- ~5 submissions/day per team; standard Kaggle final-selection of 2 submissions for private leaderboard.
- Validator rejects `Loop`/`Scan`/`NonZero`/`Unique`/`Function`/`Script`, files > ~1.44 MB, and dynamic shapes.
- Score per task rewards the *smallest valid* network that passes ARC-GEN-generated `<input, output>` checks; total score is the sum across the 400 tasks.

---

## `04-resources-and-notebooks.md`

# 04 — Datasets, Utilities, and Public Notebooks

## Official Datasets / Inputs

The competition data area pulls from the same set the Code-Golf 2025 competition used:

1. **ARC-AGI Public Training Set v1 (400 tasks)** — the canonical ARC-AGI-1 training set by François Chollet, hosted at <https://github.com/fchollet/ARC-AGI>. Each task = JSON file of `<input, output>` colored-grid demonstration pairs. Models are written *per task* (`task001` … `task400`). Total raw size of this corpus: **1.44 MB uncompressed** (this is the source of the per-file size limit).
2. **ARC-GEN procedurally generated examples** — generated through the `google/ARC-GEN` open-source benchmark generator. ARC-GEN is described as *"the official benchmark generator in the 2026 NeuroGolf Championship featured at IJCAI-ECAI 2026"* (per the ARC-GEN README's news section, 2026-04-04). It procedurally generates additional faithful `<input, output>` pairs per task, used for both training augmentation and the validator's correctness check. Repo: <https://github.com/google/ARC-GEN>. Paper: *"ARC-GEN: A Mimetic Procedural Benchmark Generator for the Abstraction and Reasoning Corpus"*, Michael D. Moffitt, arXiv:2511.00162.
3. **ARC-GEN-100K (or similar pre-generated bundle)** — a pre-generated package of ~100,000 grid pairs (≈250 per task × 400 tasks) is commonly attached as a Kaggle dataset for participants who don't want to install ARC-GEN locally. **[Verify exact dataset name and contents on the /data tab.]**
4. ARC-GEN now also supports **ARC-AGI-2** (500 additional tasks, per 2026-03-25 update), but NeuroGolf 2026's scoring set is the **400 ARC-AGI-1 training tasks**.

## Official Organizer Utilities — `neurogolf_utils`

The competition example references a Python module called **`neurogolf_utils`**. Based on the parallel `code_golf_utils` from the predecessor competition, this module is provided by the organizers and contains:
- A **local validator** function that checks an ONNX file against the disallowed-op list and shape constraints — letting you verify a model would pass the network validator before submitting.
- A **scorer** that reads `submission.zip` (or a directory of `taskNNN.onnx` files) and computes the same statistics that Kaggle uses (likely via `onnx-tool`).
- Utilities to load ARC-AGI tasks (`load_task(task_id)`) returning `(train_pairs, test_pairs)`.
- Helpers for writing ONNX initializers, packing weights, and building grid-shaped input/output tensors.
- Optionally an **ARC-GEN bridge** to call the procedural generator from inside a notebook.

The module is typically distributed as part of the competition's "code" tab as a **Kaggle Notebook utility script** or as a **Kaggle Dataset** that you can attach to your notebook. **[Confirm exact location — /code tab "Pinned Notebooks", or /data attached datasets — on Kaggle.]** It is not (visibly) on PyPI; it lives within the competition's resources.

## Public Notebooks (visible in search results)

The following community notebooks are publicly indexed for `neurogolf-2026`. None could be deeply read by the research fetcher (Kaggle's bot wall), so descriptions are based on the title/snippet:

| Notebook URL | Author | Apparent purpose |
|---|---|---|
| <https://www.kaggle.com/code/mmoffitt/the-2026-neurogolf-championship> | **mmoffitt** (Michael Moffitt — host/organizer) | The **official starter notebook** for the competition. Includes the example ONNX-construction pipeline, calls `neurogolf_utils`, and demonstrates a baseline submission for one or more tasks. |
| <https://www.kaggle.com/code/nihilisticneuralnet/neurogolf-championship-2026-baseline> | **nihilisticneuralnet** | Community starter / baseline — title is "NeuroGolf Championship 2026: Starter Notebook" / "Baseline". Builds on the official starter to demonstrate a working end-to-end submission pipeline. |
| <https://www.kaggle.com/code/mpwolke/neurogolf-2026-onnx> | **mpwolke** | Community ONNX exploration notebook — title "neurogolf-2026-onnx". Likely walks through ONNX file inspection / construction. |
| <https://www.kaggle.com/code/sigmaborov/neurogolf-2026-public-data> | **sigmaborov** | "NeuroGolf 2026 — Public Data" — likely an EDA / data-loading walkthrough showing the public competition data. |

There are likely additional notebooks not surfaced by general web search. The best inventory is the live `/code` tab on Kaggle, sorted by votes or freshness.

## External Reference Repositories
- **`google/ARC-GEN`** (<https://github.com/google/ARC-GEN>) — the procedural benchmark generator, official competition tooling.
- **`fchollet/ARC-AGI`** — the upstream ARC-AGI-1 task corpus.
- **`SakanaAI/google-code-golf-2025`** (<https://github.com/SakanaAI/google-code-golf-2025>) — Sakana AI's published toolkit for the predecessor (Python-source) competition. Useful as a structural template for how a `submission.zip` per-task workflow is automated, even though NeuroGolf uses ONNX, not Python.

## ONNX Reference (brief context)
- ONNX (Open Neural Network Exchange): <https://onnx.ai/>. The validator only cares about the standard ONNX operator set; consult the [ONNX Operator Schemas](https://github.com/onnx/onnx/blob/main/docs/Operators.md) for what's allowed.
- Inference is run with **ONNX Runtime** (`onnxruntime`) on the evaluation server — community discussion confirms participants are reproducing scores locally with `onnxruntime` to debug profiler/graph-counting discrepancies.

## TL;DR — Resources
- Data = ARC-AGI-1's 400 training tasks plus ARC-GEN procedural samples; per-file artifact size cap ~1.44 MB.
- The official `neurogolf_utils` module (provided in the Kaggle Code/Data tab) is your local validator + scorer + ARC-GEN bridge.
- Four publicly indexed community/starter notebooks exist (mmoffitt, nihilisticneuralnet, mpwolke, sigmaborov); browse the /code tab for the freshest list.

---

## `05-forum-highlights.md`

# 05 — Forum / Discussion Highlights

> The Kaggle discussion forum (`/discussion`) and individual threads were not directly readable in this research session. The single thread URL surfaced via search and the public X/LinkedIn posts are inventoried below. **You should walk the live `/discussion` tab to find pinned organizer announcements, FAQ threads, and rule clarifications.**

## Confirmed Threads Indexed Externally

### `/discussion/692827` — "Issues in onnx-tool"
- This thread (by an unnamed participant, indexed via Google) reports issues using the `onnx-tool` library against the competition's submission models.
- **Implication for participants:** the official scorer/validator uses `onnx-tool` to compute model statistics, so its quirks affect leaderboard scores. Participants are reproducing scores with onnx-tool locally and finding edge cases (e.g., around `ConstantOfShape` and dynamic dtype handling — see the openai/codex GitHub issue #19594 quote, which references local search-strings used in NeuroGolf scoring debugging: `onnx_tool|ConstantOfShape|dynamic|dtype|initializer`).
- **Practical takeaway:** If your local model evaluates correctly with `onnxruntime` but scores oddly on the leaderboard, the discrepancy is probably in how `onnx-tool` counts ops / parameters in your graph. Use the version of `onnx-tool` referenced in `neurogolf_utils` rather than the latest pip release.

## Common Gotchas (synthesized from public chatter)

- **Surprisingly high leaderboard scores** (e.g., a discussion of a "7600.42" score) have appeared in participant chatter (visible via the openai/codex GitHub issue, where a competitor was investigating *whether unusually high leaderboard scores could be caused by an ONNX Runtime profiler/accounting discrepancy or a competition-scoring loophole*). Top-of-leaderboard outliers may be exploiting a scoring edge case rather than a substantive algorithmic insight; expect organizers to publish rule-clarification posts as such issues come to light.
- **`ConstantOfShape` handling, dtype/initializer quirks, and dynamic-shape leakage** are the technical themes participants flag most. Models that *technically* parse but use unusual constructs may pass or fail validation depending on the validator version.
- **Specific task IDs flagged in community debugging:** task002, task157, task177, task276, task309, task337 are all referenced in participant notes as having anomalous scoring or unusual transformations to model. **[Use as a starting point for your own task-by-task review.]**
- **Operator allow-list is enforced strictly.** The disallowed list (`Loop`, `Scan`, `NonZero`, `Unique`, custom `Function`/`Script`) is non-negotiable for prize eligibility; some constraints "are not (yet) fully enforced, but all will be imposed for teams seeking prize eligibility" (this is the same wording the predecessor competition used — expect the same pattern here).

## Organizer Announcement Channels

- **Kaggle X/Twitter:** announcement post on Apr 15, 2026 at 2:15 PM:
  *"Just released! The Neurosynthetic Research Institute is launching the IJCAI-ECAI 2026 NeuroGolf Challenge. Implement ARC-AGI transformations by designing minimal neural networks. How simple can a model be and still solve the task?"* (`x.com/kaggle/status/2044494732504522780`).
- **Kaggle on LinkedIn:** equivalent launch post (linkedin.com/company/kaggle): *"Neural networks keep growing — but how simple can a model be and still solve complex tasks? Live now, The Neurosynthetic Research Institute is hosting the IJCAI-ECAI 2026 NeuroGolf Challenge to explore the absolute limits of model simplicity."*
- **ARC-GEN GitHub README** news entry (2026-04-04): announces ARC-GEN as the official benchmark generator for NeuroGolf 2026.
- **Host site** `neurosynthetic.org`: links to the Google-Sites companion (corp link, not publicly fetchable).

## Rule Changes / Updates Since April 15, 2026
No published rule-amendment announcements were retrievable through web search as of the research date (May 1, 2026). The competition is two weeks old; participants should monitor the **/discussion** tab for **pinned posts by the organizer** (search by author = the organizer's Kaggle username, typically the host account) and any "[Update]" or "[Rule clarification]" thread titles. Kaggle competition rules pages are also versioned — when a rule changes, all participants who have already accepted the rules must re-accept the updated rules to continue submitting; this is the most reliable signal of a rule change.

## Suggested Forum Searches (to run on Kaggle directly)
- `pinned` — most organizer announcements get pinned
- `FAQ`
- `validator`
- `onnx-tool` / `onnxruntime`
- `disallowed`
- `1.44`
- `task001` (and other specific task IDs)
- `Longest Leader`
- `student team`

## TL;DR — Forum
- One thread surfaced externally: `/discussion/692827` ("Issues in onnx-tool") — confirms the scorer uses `onnx-tool`; replicate locally with the same version.
- Watch for organizer pinned posts on Kaggle; no rule-change announcements were public as of May 1, 2026.
- Community red flags: scoring outliers driven by `onnx-tool` accounting and `ConstantOfShape`/dtype edge cases; specific tasks (002, 157, 177, 276, 309, 337) are recurring debugging fodder.

---

## Caveats (apply to all five files)

1. **Direct Kaggle competition pages were not deeply scrape-able** in this research environment, so the specific numeric facts that *only* appear on Kaggle (exact deadlines, exact prize dollar amounts, exact daily submission cap, exact validator opset version, exact list of attached datasets, exact set of pinned discussion posts) are flagged as **[Verify on Kaggle]** throughout. Anything inferred from the predecessor "Google Code Golf 2025" competition or from the `arc-gen` paper is explicitly tagged.
2. **The competition launched on Apr 15, 2026 and the research date is May 1, 2026** — very early in the competition. Many discussion threads, organizer FAQ posts, and possibly rule clarifications will be added in the months ahead. Re-read the `/discussion` tab for the latest community wisdom.
3. **No paper/talk obligation is imposed by Kaggle**, but the IJCAI-ECAI 2026 special session at Bremen (Aug 15–21) is the natural venue where top teams will be invited to present; that invitation generally carries no monetary travel support.
4. The `Neurosynthetic Research Institute` is a thin/lightly-detailed entity (a one-page site at `neurosynthetic.org`); the organizing humans behind it appear to be a Google-affiliated team headed by Michael D. Moffitt (author of ARC-GEN), but the institutional structure is not transparently documented publicly.
5. The "openai/codex issue #19594" is included as a *citable* third-party indicator of the sorts of debugging issues real competitors are working on (ONNX Runtime profiler/accounting behavior on ARC tasks); it is *not* an authoritative source on the competition rules.
