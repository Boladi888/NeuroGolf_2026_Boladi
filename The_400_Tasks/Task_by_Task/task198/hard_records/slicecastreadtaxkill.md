# DEAD ROUTE — task198 "SliceCastReadTaxKill" (2898 → 900 false-floor attack)

**Recorded:** 2026-07-10 by `bo_ng_cld_worker2` (Claude lane). Verdict: **DROP / WALL**.

## Live bar (authoritative, refreshed this session)
- `ng_current_bar.py --task 198 --json` → **live_cost = 2898** (authoritative_source = manifest, owner **Sho**, updated 2026-07-10T09:55Z).
- Champion ONNX: `Team_Best_Onnx/task198.onnx`, **sha256 `ce164130c94a15ec0e7e357dbe5cb80f2ad1ed096f8b84ad51654b3368a46619`**, graph name `task198_v52_opset18_padmaxpool`, opset 18.
- Confirmed real: `ng_gate visible --task 198` → **266/266 PASS**; `profile_real_cost --arcgen 266` → **static = profile = 2899** (memory 1259 + params 1640). No dynamic inflation.
- Stale sources ignored: `task_state.csv` = 9324, context-pack `live_costs.json` = 9324, task198.md body = 89301/9324 — ALL predate Sho's v52 rewrite. The 9273 staged candidate is the obsolete direct-render lineage (team_best was 9324 then).

## Attempted route (from the prompt)
"SliceCastReadTaxKill": avoid a Slice+Cast read double-charge by using the smallest legal input read and pushing the cast/narrowing to the first real consumer; from-scratch STRUCTURAL rewrite to a forced floor **~900**.

## Why it's dead

### 1. The named charged component is GONE (workflow-update DROP trigger)
The live champion (v52) has **0 Slice ops** (`grep Slice` → no matches). It reads the input via 4 ring-Convs (`ring4/ring5/ring6_h/ring6_v`) directly on `input`, classifies into 3 structural templates (sp0/sp1/always_true via MaxPool+Less), and decodes with one Einsum + `membership_diff`. There is **no Slice+Cast read tax to kill** — Sho already killed it (and the whole flood formulation) in the 9324→2898 rewrite. Per the 2026-07-10 workflow update ("if [the charged component named in Direction] is already gone… return DROP"), this alone is a DROP.

### 2. Fast probe — the forced read AND forced decode each individually exceed 900
Champion cost 2899 = memory 1259 + params 1640, decomposed exactly (`inspect_graph` + `largest_intermediate` + `_score_v2`):
- **Read/detect = 1350**: ring kernels `ring4`(250)+`ring5`(360)+`ring6_h`(100)+`ring6_v`(100) = **810 params**; conv outputs `cell_sum4`[1,1,7,7](196)+`cell_sum5`[1,1,6,6](144)+`cell_sum6_h`[1,1,5,5](100)+`cell_sum6_v`[1,1,5,5](100) = **540 mem**. → **1350 > 900**.
- **Decode = 976**: `membership_diff`[3,30,8] = **720 params** + `score_bucket`[1,1,8,8] F32 = **256 mem**. → **976 > 900**.
- `params` counts **element count** (dtype-independent), so U8/F16 narrowing does NOT reduce the 810+720 table params — the classic dtype lever is inert here.

For 900 to be reachable a formulation must have read **and** decode both < 900, but the two cheapest KNOWN components (Sho's downsampled ring read; Sho's Einsum decode) **each already exceed 900**. Flood-reachability is data-dependent/iterative — it is NOT a fixed multilinear map, so it cannot collapse to a single fused Einsum that shares read+decode structure.

### 3. Both independent formulation families are priced walls
- **Formulation A — direct-render / per-cell flood** (task-local v1–v17 + `regionfilllineagerewrite_cost9273`, my own 2026-06-30 work): manifest `next_lever` records a **proven forced floor of 7200 B** = `bg_f32` 3600 (full-res F32 read; the random non-periodic gap pattern cannot be recovered from line-only/coarse reads) + 4×900 planes. **8× over 900, 2.5× over 2898.** Dead.
- **Formulation B — template / ring-conv (Sho v52 = live)**: 2899, read 1350 > 900, decode 976 > 900 (above). Beating 2899 at all would be a `rung − epsilon` shave, which the false-floor prompt explicitly forbids ("This is a false-floor attack, not a rung − epsilon shave").

The prompt's 900 target was estimated against the obsolete ~9324 flood graph; even the old task198.md floor analysis debunked the sub-5k estimate ("10-channel-map floor, NOT a single-channel crop floor"). Sho's v52 already performed the false-floor rewrite (direct-render 7200 B → template 2898). There is no third, cheaper family in evidence, and 900 sits below the forced read of BOTH known families.

## One remaining hypothesis (for the next batch, low confidence)
A template scheme that simultaneously (a) uses fewer/smaller ring scales than Sho's 3 (dropping read below ~900) **and** (b) replaces the 720-param `membership_diff` Einsum with a sub-900 decode — would have to pass visible 266/266. No evidence such a joint reduction exists: v52 (52 iterations) carries 3 ring scales and a 3-template decoder precisely because fewer lose corpus coverage. If a future worker still probes, do it as a **single fast probe** (build the minimal 1-ring + [10,10]-remap decode, gate visible) and stop on the first visible miss — do NOT rebuild the flood family (proven 7200 B) or re-derive Sho's decoder.

**Gates run this session:** `ng_current_bar`, `inspect_graph`, `largest_intermediate`, `_score_v2`, `ng_gate visible` (champion 266/266), `profile_real_cost` (champion static=profile=2899), `grep Slice` (0). No sub-2898 candidate built → no truth/ab/battery/ng_safety needed (nothing to ship).

---

## PASS 2 (forced second pass, 2026-07-10) — deeper decode + fresh gated build

**True rule re-derived from examples (task198.json train/arc-gen):** periodic wall lattice (period P∈{5,6,7}); each background cell → **4 if its wall-ring has a gap** (reachable), **3 if the ring is intact** (enclosed); walls pass through; ch0 emptied. This is a **LOCAL ring-gap test, NOT a global flood** — the champion has no iterative dilation (its two MaxPools are 7×7/6×6 *alignment*, not flooding). The old task198.md "flood K=11" rule was an over-complication; Sho's local formulation is why 2898 is so cheap.

**Champion fully decoded** (`recon198.py`, `dump_membership.py`):
- Einsum `brij,salm,tden,shi,twj,kc,bkhw->bchw` over (`score_bucket`, `sp_onehot`×2, `membership_diff`×2, `transform_s`, `input`): period-classify → per-cell reach (`score_bucket`) → **expand cells→pixels** via `membership_diff[3,30,8]` (block one-hot, used for BOTH row `shi` and col `twj`) → color-remap `transform_s[10,10]` + wall passthrough `input`, all **folded into the free `output`** (zero expansion memory).
- **Read-tax-kill already applied:** `ring4/5/6` are `[1,10,K,K]` but **nonzero only on channel 0** — Sho embeds the bg-channel selection into the (mostly-zero) Conv kernel (810 params) to avoid `Slice(ch0)` = a 3600-B full-res plane. Slicing/pooling to get bg costs the 3600 plane → strictly worse. This IS the "SliceCastReadTaxKill" lever, already spent.
- **All 3 period cases are load-bearing:** measured case distribution over the 262 arc-gen inputs = sp0 35.9% / sp1 30.9% / else 33.2% (`case_dist.py`). None droppable → `membership_diff` 720 and the 3 ring detectors 810 are forced for truth-exactness. Champion truth-exact (3000/3000 capped).

**Fresh genuinely-different candidate built + gated THIS pass** — pixel-resolution / slice-concat architecture (no downsample, no membership expansion): `prototypes/task198_p2_pixelres.onnx` (`build_task198_p2_pixelres.py`).
- `_score_v2`: **cost 43238** (memory 42300 + params 938).
- `ng_gate visible`: **17/266 (6.39%) FAIL** → §VISIBLE-FAIL: dead family, no truth/ab/profile run.
- Why it explodes: the 7 per-channel passthrough `Slice` planes = 7×3600 = 25200 B, plus full-res reach planes — the exact full-res read/passthrough tax the champion's Einsum-fold avoids. Empirically confirms the pixel-res architecture cannot approach 2898.

**Architecture trichotomy (all priced, none beats 2898):**
| architecture | forced dominant cost | total | gate | verdict |
|---|---|---|---|---|
| pixel-res / slice-concat | 7× full-res slice planes 25200 | **43238** | visible 17/266 FAIL (pass-2 build) | dead |
| direct-render / per-cell flood | `bg_f32` 3600 + 4×900 = 7200 | **9273** | full battery PASS (prior staged) | 3.2× over bar |
| downsample→expand→Einsum (Sho v52) | ring 810 + membership 720 | **2899** | visible 266/266, truth 3000/3000 | **champion** |

**Information-theoretic wall:** the two axes are in tension — avoiding the full-res plane forces downsampling, which forces a cell→pixel **expansion matrix** (≥720, folded into free output = optimal; runtime-generating it just moves 720 params → ≥720 memory + gen ops = loss); avoiding the expansion forces **pixel resolution**, which forces ≥3600-B full-res planes. No formulation escapes both. Within Sho's optimal architecture, forced params alone = ring 810 + expand 720 = **1530 > 900**, and forced read/score memory ≈ 540+256 = **796**. The 900 target sits **below a single forced component** (the 720 expansion, or the 810 read). 900 is a false floor; 2898 is the genuine architecture floor. **WALL confirmed with a fresh gated build.**

---

## PASS 3 (2026-07-12, `bo_ng_cld_worker2`) — tensor-level shave attack on the 2898 champion itself

New mandate framing this session: beat 2898 **by any means, even a rung−ε shave** (not just the 900 false-floor). So I attacked Sho's v52 champion graph directly, tensor-by-tensor, rather than the 900 target. **Result: no sub-2898 candidate exists. WALL re-confirmed with stronger, quantified evidence.**

### Bar (fresh, this session)
- `ng_current_bar --task 198 --json` → **live_cost 2898** (authoritative = manifest, owner Sho, `2026-07-10T09:55Z`, sha `ce164130…`). Non-stale. `team_best` source (2895) flagged stale, `local task_state.csv`/context-pack `live_costs.json` (9324) flagged stale.
- Champion `_score_v2(Team_Best_Onnx/task198.onnx)` = **2899** (macs 0, mem 1259, params 1640). **The manifest 2898 is 1 B BELOW the champion's own local score** — even the champion does not score `<2898` locally, so a same-arch candidate ties at 2899 at best. A win needs a robust **≥2 B structural shave**; none exists.

### Exact decomposition (`largest_intermediate` + `dump_inits`), sums to 2899
- **Read (1350):** ring4 250 + ring5 360 + ring6_h 100 + ring6_v 100 = **810 params** (kernels are `[1,10,K,K]` but nonzero ONLY on ch0 — channel-selection embedded in the mostly-zero Conv kernel to avoid the 3600-B `Slice(ch0)`); F32 conv outputs cell_sum4 196 + cell_sum5 144 + cell_sum6_h 100 + cell_sum6_v 100 = **540 mem** (Conv on F32 input ⇒ F32 output, forced; immediately Cast→U8).
- **Expand 720:** `membership_diff[3,30,8]` — folded into the free `output`.
- **Remap 100:** `transform_s[10,10]`.
- **Score bucket 256:** `score_bucket[1,1,8,8]` F32 (Einsum operand).
- **Forced subtotal = 2426.** Minimal cell-selection tail = 463 mem + 10 params = **473**. Total **2899**.

### Decode fully understood (outer-product fold)
`output[c,h,w] = P[h,w] · W[c,h,w]`, one Einsum `brij,salm,tden,shi,twj,kc,bkhw->bchw`, where P = score/membership expansion (per-pixel scalar) and W = `transform_s·input` (per-pixel remapped color). Both fold into the FREE output → zero full-res intermediates. **Splitting P and W would materialize P `[1,1,30,30]` (≥1800) and W `[1,10,30,30]` (≥18000) — catastrophic. The single fused Einsum is uniquely optimal.**

### Every lever priced DEAD (new this pass)
1. **F16 Einsum (biggest prize: score_bucket 256→128 + sp_onehot 12→6 = 134 B → would give 2765).** BLOCKED. Empirical test (`einsum_dtype_test.py`): ORT `Einsum` rejects mixed F16/F32 — `Type parameter (T) bound to different types (tensor(float16) and tensor(float))`. The Einsum must be uniform-dtype; `input` is F32 (declared graph input; grader feeds F32); casting input→F16 = `[1,10,30,30]` F16 = **18000 B** ≫ 134. Dead.
2. **Narrow param tables** (membership 720, rings 810, transform 100): inert — `_score_v2` counts params by **element count**, dtype-independent.
3. **Shrink 8×8 coarse grid → 7×7** (would cut membership −90, score_bucket −60, paint_bucket −15): DEAD. `membership_diff` slice 0 routes wall rows/cols → coarse index **7** (the always-True border bucket); the 8th cell is the load-bearing wall-line bucket.
4. **onnxsim / polish on v52:** DEAD. 24→24 nodes, 12→12 inits, cost 2899→2899 (unchanged). (The 2026-06-24 don't-do note only tested onnxsim on the OLD 9324 graph; now confirmed inert on v52 too.)
5. **Cheaper cell-selection tail:** DEAD. The 2-Where / 2-Pad cascade with **incremental padding** (6→7→8, keeps intermediates small) + a **single late U8→BOOL cast** is minimal. Gather/Concat select (`Concat[3,8,8]` 192 + Gather 64 = 256) or feeding all 3 period buckets into the Einsum (`score_bucket[3,8,8]` 768) both cost MORE than the 134-B cascade they'd replace.
6. **Conv outputs F32→narrower:** DEAD. Conv on F32 input ⇒ F32 (forced); already Cast→U8 at the next node.

### No build shipped — and why that's correct, not lazy
Every lever above is blocked by exact byte-counting or a decisive empirical test; any constructible same-arch candidate is provably ≥2899 (tie-or-worse). The three architecture families are exhaustively priced (pixel-res 43238 visible-FAIL / direct-render 9273 / template 2899), and PASS 2 already shipped a **fresh gated pixel-res build** (visible 17/266). Building another provably-losing graph would burn budget for zero information (resource-safety: "A ship gate on a losing candidate is pure burn"). No gates re-run on the champion (bar already fresh from `ng_current_bar`); no `ng_safety` (no candidate beats bar).

**VERDICT: WALL. 2898 is the genuine floor.** Forced components alone (read 1350 + expand 720 + remap 100 + score 256 = 2426) leave only a 473-B tail that is itself minimal and F16-locked. One remaining (very low confidence) hypothesis for a future batch: a *joint* reduction that simultaneously uses <3 ring scales AND a sub-720 decode while holding visible 266/266 — no evidence it exists (all 3 periods measured load-bearing, case dist 35.9/30.9/33.2%). Probe as a SINGLE fast visible gate and stop on first miss; do NOT rebuild flood (7200 B) or re-derive Sho's decoder.

---

## PASS 4 (2026-07-12, forced 2nd pass, `bo_ng_cld_worker2`) — library/provider mining + GLOBAL floor across all 3 assemblies + fresh canvas gate

Runner forced a genuinely-different formulation + re-gate. Did the mining the prompt asked for; built nothing new that beats 2898. **WALL holds — now as a GLOBAL (all-assembly) floor, not just an Einsum-family floor.**

### Library / BoBrain mining — nothing new
- `lib_query context/search` (DB STALE @ best_cost 9324, predates v52), BoBrain provider `context/consistency/search` → authoritative bar **team_manifest 2898** (points 17.0282); all other sources stale. No cheaper assembly.
- Cited hits are non-levers here: **#58** = period-detection by *color count* (task198's period is grid *spacing* 4/5/6, not color count — N/A). **traps §INPUT-DTYPE-LOCK explicitly lists "task198 Slice+Cast double-charge"** and prescribes "#98-style Conv read/fuse only when it avoids materializing the F32 plane" — which is **exactly** Sho's channel-embedded `[1,10,K,K]` ring-Convs. The library endorses the champion's read as optimal.

### GLOBAL forced-component floor (spans EVERY assembly, via the sign-only `out>0` grader + `output=P[h,w]·W[c,h,w]` outer-product decode)
Any correct graph must, at minimum:
1. **Read wall-ring gaps** (3-vs-4 per cell): downsample-ring `= 810 params + 540 F32 conv-out = 1350` **OR** full-res `bg_f32[1,1,30,30] = 3600`. min **1350**.
2. **Expand coarse paint→pixels**: `membership_diff[3,30,8] = 720` folded into free output **OR** full-res paint planes (≥900 each). Runtime-generating membership is WORSE: the selected `[30,8]` **F32** one-hot (Einsum-locked to F32) = 960 alone > 720 stored ×3. min **720**.
3. **Carry data-dependent wall colours**: the free `input` as an Einsum operand `= 0` **OR** a materialised colour canvas (≥3600). min **0** — but it F32-locks the fold (ORT rejects mixed-dtype Einsum, PASS-3 empirical), forcing `score_bucket[1,1,8,8]` **F32 = 256** and `transform_s[10,10] = 100`.
4. The P·W combine that writes the FREE output **must** be Einsum/MatMul — the only ops that form the outer product WITHOUT materialising P `[1,1,30,30]` (≥1800) or W `[1,10,30,30]` (≥18000). An explicit `Mul`/`Where`/`Add` combine materialises them → dead.
5. **Cell-select tail** (pick 1 of 3 period maps + build the 8×8 sign bucket, incl. the load-bearing wall-line bucket) = **473**, each intermediate min-typed (U8 select, single late BOOL cast, incremental 6→7→8 padding).

**Σ minima = 1350 + 720 + 256 + 100 + 473 = 2899.** Deviating from ANY line jumps families. **This is a GLOBAL minimum, not a per-arch one.**

### Empirical confirmation of the deviation penalty (re-gated THIS pass)
- **Canvas-Equal family** (`staged_candidates/…cost9273/task198.onnx`, sha `5d59c92a…`; different dominant intermediate = full-res U8 planes, different assembly = Equal free-output): `ng_gate visible` **266/266 PASS**; `profile_real_cost --arcgen 266` **static=profile=9273** (mem 8928 + params 345). `ng_graph_cost_diff` vs champion: **+6365**, dominated by `bg_f32[1,1,30,30]` **+3600 B** — the exact full-res read Sho's ring-Convs avoid (line 1 penalty, empirically).
- Pixel-res family (PASS 2): 43238, visible 17/266 FAIL.
- Three assemblies now priced: Einsum-fold **2899** ‹ canvas-Equal **9273** ‹ pixel-res **43238**. The champion sits at the global floor.

**No sub-2898 candidate exists in any assembly.** No ship gate / `ng_safety` run (nothing beats the bar — pure burn). WALL re-confirmed with a fresh this-pass gated build of the next-best family.
