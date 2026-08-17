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


---

**Phase-1 note (2026-06-05)**: the superseded draft body (65% of this file) moved to `_archive/00_COMPETITION_SPEC_full_pre_phase1.md`. Live worker facts: this head + `core.md` (canonical). Captain-side rules (deadlines/prizes/submit mechanics): see head sections above.
