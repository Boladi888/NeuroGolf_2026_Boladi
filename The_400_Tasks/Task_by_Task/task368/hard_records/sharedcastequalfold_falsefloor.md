# task368 — SharedCastEqualFold structural target is a FALSE FLOOR → DROP

**Recorded:** 2026-07-11 by `bo_ng_cld_worker3` (Claude lane).
**Verdict:** DROP. The prompt's 4532→1400 structural premise is stale on every axis; the live graph is already a near-floor Conv/render formulation whose real cost is ~1558, not 4532.

## Live bar (as of this session)
- **Manifest (authoritative per tools):** `task368,Sho,902959d0299713acc6eb0438f80f8bbc4a6f5ac3a6b3280683c943bd7de99341,4532,16.5811,2026-07-09T01:09:17Z`.
- **MEASURED real cost of that exact sha: ~1558** (three independent tools agree):
  - `ng_graph_cost_diff` (cost_predictor): memory 1486 + params 72 = **1558**.
  - `profile_real_cost --arcgen 266`: **1566** (memory 1486 + params 80, 269 real inputs).
  - Direct ORT runtime-shape probe over **all 765 official grids** (train+test+762 arc-gen): intermediates sum to **1486**, zero exceptions, no tensor > 400 B. Output [1,10,30,30] UINT8 = 9000 B is **not charged**.
- **Visible gate on live:** 765/765 (100%) PASS.
- **`_score_v2` static on live: returns None** (graph has a dynamic Slice surface) — so static scoring cannot yield 4532 either. No local tool reproduces 4532.

## Why the SharedCastEqualFold route (and the 3.24x prize) is dead
1. **Named charged component is GONE.** Direction targeted "duplicate Cast/Equal carriers across parallel boolean chains." The live graph (`bianfu3`, opset 18, 19 nodes) has exactly **one Cast pair and one Equal**, no parallel boolean chains. It already did the from-scratch structural rewrite (techniques #35 asym-conv free crop + #36 rank-Where; ledger 2026-06-04 shipped v9 at 6294, then bianfu3 to 4532/really-1558). SharedCastEqualFold (the old 50,484→staged route) is two generations superseded.
2. **The 4532 bar is a false floor.** The real effective-best cost is ~1558. A candidate in (1558, 4532) would be a **regression** dressed as a win. Only < 1558 is a genuine improvement.
3. **Cost is structurally content-independent and capped at 10×10.** Node 0 `Conv(input,[0..9], kernel [1,1], pads [0,0,-20,-20])` decodes color (channel-index weighted sum) AND crops the 30×30 canvas to 10×10 via a **negative pad** — one free op. Node 18 `QLinearConv(..., pads [0,0,20,20])` pads back to 30×30 for the output. The hard 10×10 crop means the graph can only serve 10×10 grids (all 765 official grids are exactly 10×10), and every intermediate shape is fixed regardless of input content → the cost is ALWAYS ~1558. There is no hidden-grid path to 4532.
4. **Near-floor + forced Conv/render.** The rule (copy the unique non-0/non-5 template block into every gray-5 region) is **data-dependent placement** → a Conv/render floor (traps §132: variable placement counts are render floors, not free-output separable). The live 1486 B decomposes into forced components: F32 [1,1,10,10]=400 color-decode (Conv reads F32 one-hot; F32 output forced), ~6×UINT8/BOOL [1,1,10,10]=100 masks (is5 / foreground / template-only / corner-map / stamp-map), UINT8 [1,3,10,10]=300 color re-encode, UINT8 [1,1,11,11]=121 padded-grid-for-slice, plus small scalars. Floor est 1400 is ~90% of the live 1558; the forced F32 decode + render planes sit right at it.

## Independent from-scratch formulations considered (all ≥ 1558)
- **Gather-based stamp** (per-cell source index from block-relative offset): needs a data-dependent INT32 index plane [1,1,10,10]=400 on top of the 400 decode → ≥ live. LOSE.
- **Separable / free-output Einsum**: forbidden — region positions vary per input (§132/§109 rank gate), no static separable decomposition. LOSE.
- **"Boring static sibling" (§GRADER-GAP)**: does not apply — the live graph is already static-shaped and honestly priced (visible 100%, runtime==static), so there is no dynamic-shape fiction to convert into a win.

## Remaining hypothesis (low-confidence, NOT pursued)
A ~100–150 B micro-shave might exist: (a) collapse the UINT8 [1,3,10,10]=300 re-encode to [1,2,10,10]=200 if 2 channels can carry the ≤2-color template mapping; (b) fuse one of the redundant 100-B mask planes (Xor-trick per Logs/lessons 2026-06-03). This is an epsilon shave, not the scoped structural win; it is high-risk on a hand-tuned 19-node graph and the "win" is ambiguous while the manifest disagrees with measurement. Do not grind it.

## PASS 2 (2026-07-11, forced second pass) — empirical floor probe, WALL reconfirmed

Rule locked with a numpy reference (`scratchpad/ref_solve.py`): **765/765** official pairs (template = unique non-0/non-5 block; regions = solid 5-rects of matching HxW; copy template into each region). Oracle for any candidate.

**Decode semantics found:** the harness decodes output via `run_raw = (output > 0)` (`tools/ng_harness/session.py:62`) then `np.array_equal` vs the one-hot expected. So each output channel must be `>0` **iff** it is the true color. The stamp plane `v_t17` already holds `color+1`; the re-encode's whole job is `v∈{1..10} → (channel j>0 iff j==v-1)`.

**Every bianfu3 plane proven load-bearing (real ~1486 B; measured, not manifest 4532):**
| plane | bytes | why forced |
|---|---|---|
| F32 decode [1,1,10,10] | 400 | reading F32 one-hot needs F32-output op; 10×10 = active grid; F16-Conv impossible; ArgMax route = INT64 7200 (no fused crop). Negative-pad crop = 400 (beats documented v9's 1200). |
| colu8 / is5 / fg / templonly [1,1,10,10]×~4 | ~400 | single-channel color + masks to locate template & regions; any 10-ch one-hot intermediate = 10× and only free as the (uncharged) output. |
| pad-extract [1,1,11,11] | 121 | zero-border so variable template (3×3..4×4) → fixed 4×4 conv kernel + edge safety; Gather-clamp would edge-repeat and corrupt. |
| corner-map + stamp [1,1,10,10]×2 | 200 | runtime-weight Conv stamp (#54); data-dependent placement = Conv/render floor (§132). |
| **3ch re-encode [1,3,10,10]** | **300** | **EMPIRICALLY the floor.** `>0`-one-hot needs each color linearly isolable ⇒ code points in convex position ⇒ ≥3D. 2ch scan: **0 / 4,161,600** combos convex (incl. negative weights AND the color-5 don't-care that bianfu3 already exploits — color 5 never appears in output, so v=6 is free). |

**Genuinely-different assemblies evaluated, all ≥ floor:** (1) Gather-extract patch → adds a data-dependent INT32 index plane (400) or still needs the zero-pad; (2) ConvTranspose one-hot **paint** → forces a 10-channel [1,10,10,10]=1000 intermediate (10×); (3) ConvTranspose single-channel stamp → identical plane count to bianfu3's Conv; (4) parabolic 1→10 readout `W[j]=2(j+1),B[j]=-(j+1)²` → argmax-correct but fails the `>0` decode (many channels >0); (5) 2ch re-encode → proven infeasible above.

**Conclusion:** bianfu3 (~1486) sits at the forced Conv/render + decode + 3ch-reencode floor; the `floor_est 1400` is ~6% optimistic (ignores the forced 3rd re-encode channel + pad border). The only residue is a possible ~1×100 B mask-fusion epsilon — high-risk on a hand-tuned graph, an explicit rung-ε shave the prompt deprioritized, and an ambiguous "win" while manifest (4532) ≠ measured (~1558). **No structural win exists on this route. WALL.**

## ACTION FLAG for captain/orchestrator (not a worker action)
The manifest records task368=4532 but the live graph measures ~1558 on the full official corpus and is structurally capped there. If 4532 is stale, task368 is **under-credited by ~1 pt** (points 25−ln(1558)≈17.65 vs recorded 16.58). Recommend the captain **re-verify/re-score the manifest cost for sha 902959d0…** — a no-op re-score may reclaim ~1 pt with zero new graph work. (Worker cannot submit or edit team files.)
