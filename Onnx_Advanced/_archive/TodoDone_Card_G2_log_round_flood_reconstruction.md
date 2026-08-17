## Card G2 — Exact morphological reconstruction in O(log K) parallel rounds

**Researcher contract (read first)**: You are doing INTERNET research only. You have NO access to our repository, code, files, or machine — everything you need is in this prompt; do not ask for files and do not assume any project detail not stated here. Produce ONE copyable Markdown research note as your entire answer (no code execution, no file writes). Follow the required answer order and table schemas exactly, and tag every claim with its source class.

**Deliverable**: a copyable Markdown research note (`log_round_flood_reconstruction.md`). Research note only.

**Context (project-verified)**: Several of our models compute **masked flood-fill / morphological reconstruction by dilation**: `reach <- seed; repeat K times: reach <- dilate3x3(reach) AND mask`. K is set by the worst-case geodesic depth measured over a 200k corpus (observed K = 5, 14, 25; one task's depth histogram maxed at 11). Each round materializes one HxW plane (cost proportional to K). We want exact results in fewer rounds.

**Research questions (in order)**:
1. Is **geodesic dilation with doubling structuring elements** (radii 1,2,4,8...) EXACT for masked reconstruction, or does masking break the semigroup property? Give the standard result or a counterexample (narrow concave corridors). Cite morphological-reconstruction literature (Vincent 1993, parallel / "fast hybrid" algorithms) and parallel-rounds complexity results.
2. What IS provably correct in o(K) rounds for masked reconstruction on a grid (e.g., repeated squaring of the reachability relation as min-plus matrix powers; jump-flooding with validity re-checks; alternating big-dilate + r-step re-mask schedules)? For each: rounds needed, per-round tensor cost on an HxW grid, and exactness conditions.
3. **Grid-specific bounds**: for 4-connectivity on an NxN grid with arbitrary masks, what is the minimum number of constant-size-kernel masked-dilation rounds (lower bound = geodesic diameter); can composite rounds with kxk kernels do better, and under what local-convexity-at-scale-k condition is a kxk masked dilation exact? Formalize the condition.
4. **Recommended schedules** (table): given measured corpus depth D and grid N, the cheapest exact schedule mixing kernel sizes (3x3 vs 5x5 vs strided tricks), with the validity condition each step requires.

**Constraints/cost model**: each dilation = MaxPool(kernel k, pad) on a single-channel plane = H*W bytes (U8/F16 carrier); masking = And/Mul at the same plane size; banned: Loop/Scan; everything unrolled.

**Required table schema** (Q4): | schedule | rounds | total plane-bytes (N=30, D=11 example) | exactness condition | failure example if any |

**Source policy**: image-processing and parallel-algorithms literature; mark unproven constructions SPECULATIVE.

**Verification checklist**: Q1 has an explicit corridor counterexample or proof sketch; Q2 lists >= 3 candidate schemes with rounds+cost; Q4 includes at least one schedule strictly cheaper than K unit rounds for D >= 8 with stated conditions.
