# forced_floor_memory_model hard record (batch50, bo_ng_cld_worker3)

- **Live/team-best bar:** cost/profile **635**, sha256 `bb14c2aac494c18167f7eab7cc1ac124e94634011c4d2e81da68a8acb1321c61`, source `Team_Submission/Backend/manifest.csv` + workbook row `task333` (authoritative via `ng_current_bar.py`, refreshed 2026-07-06). No `Staged_Boladi/task333.json` pending. Local task_state 2775/2783 is STALE (flagged by bar tool).
- **LIVE gates (this batch):** visible **265/265 (100%) PASS**; `profile_real_cost --arcgen 266` → `static=635 profile=635 (memory 0 + params 635; 265 real inputs)`.

## The prompt's ~300 floor_est is INVALID for this representation
The prompt assumed LIVE was the legacy 2775 `paint30_u8[1,1,30,30]` canvas model and that an "Equal-tail before the 30-Pad" would drop a 900B canvas to reach ~300. **LIVE is NOT that model.** LIVE (`task333_quadratic_relation_pairroute_renderer`) is a **single Einsum**, `input[1,10,30,30] → output[1,10,30,30]`, with `memory 0` (the only node output is the graph `output`, which `calculate_memory` excludes). Its entire cost is initializer params = 635.

## Grader cost model (read from neurogolf_utils, this batch)
- `calculate_memory` = **sum** over every intermediate node-output tensor of `max(static, runtime)` bytes. Only tensors literally named `input`/`output` are excluded.
- `calculate_params` = sum of all initializer element counts (dense dims). Sparse initializers count nonzeros — **but** `onnx.checker.check_model(full_check=True)` rejects sparse Einsum operands (rank-0 shape-inference), so `score_v2`→None (see `sparse_einsum.md`).

**Consequence:** the "drop the 900B canvas → 300" argument double-counts a saving that does not exist. To emit the required `[1,10,30,30]` one-hot output you must either (a) produce it inside the output-op (a single Einsum → needs the 30↔10 map, see below), or (b) materialize a `[1,1,30,30]` U8 canvas (900B) or a `[1,10,10,10]`/`[1,10,30,30]` one-hot intermediate (1000–9000B) — every option (b) charges **more memory than the 300 params it tries to save**. The prior scalar/U8 rewrite profiled **6203** (Family C, scratch plan). So multi-op / U8-canvas routes are all strictly above 635.

## Forced-component floor for the single-Einsum family (≈530 hard, before decoder)
Einsum index→operand map (verified this batch; total 635):
- **active_map [30,10] = 300 — FORCED.** Any single Einsum with `[1,10,30,30]` I/O must contract the 30-axis to a 10-axis (input crop `uU,vV,pP,qQ`) and expand 10→30 (output pad `rR,cC`). A dense 30×10 selection is the minimum; sparse is checker-rejected; a Slice/Pad split materializes a ≥1000B F32 intermediate ≫ 300. Shared across all 6 uses (one tensor).
- **relation_feature [2,10,10] = 200 — FORCED.** plane0 = upper-tri (`≥`), plane1 = ones−eye (`≠`); together they synthesize `eq`, `<`, `>` for the ray betweenness. One-plane (`le_only`, `transposed_le`) both gate **visible 0/265** (see those hard_records). Neither plane derivable from the other inside one Einsum without an added operand (≥100 params) or a charged intermediate.
- **color_atoms [3,10] = 30 — FORCED.** atoms = [all-ones, one-hot(0)=bg, one-hot(3)=block]; shared by 6 indices (a,A,B,d,k,o); all three rows used across `source_atom_coeff`/`output_atom_coeff`/`green_atom_coeff`. Zero-columns in individual coeffs (green B=1, output A=2) cannot shrink the *shared* `color_atoms` without adding a per-consumer slice init that costs more than saved.

Forced subtotal = **530**. The remaining **105** is the decoder (route_core 27, pair_basis 27, relation_decoder 12, channel_core 12, green 9, source 6, output 6, channel_relation 6). Every decoder latent index (z,w,s,t,x,y,h,l,m,n,E..N) is **shared** between ≥2 operands; collapsing any requires aligned zero-slices across *all* its consumers — none exist (verified by index→operand map + per-slice zero scan this batch). Cores are already sparse/minimal.

## Verdict: WALL at 635
635 sits ~105 above the 530 forced floor, entirely inside an already-minimal, fully-interlocked decoder. The only theoretical lever is a cleverer decoder factorization saving a few of the 105 core params — a sub-epsilon grind the prompt explicitly deprioritizes, and prior batches' factorization probes (le_only, transposed_le) all failed visible. ~300 is unreachable in any representation admissible to this grader. No stageable candidate.
- **One remaining hypothesis (low EV):** a decoder refactor that expresses the block-detect + ray-render with a smaller shared core (<105) while keeping both relation planes and one direct-output Einsum. Would yield at most a handful of params, not the ~300 target.
