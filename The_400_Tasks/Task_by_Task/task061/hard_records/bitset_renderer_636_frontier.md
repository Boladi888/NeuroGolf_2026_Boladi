# task061 bitset renderer frontier at 636

**Live bar:** 636 from `python tools/ng_current_bar.py --task 61 --json`
(`manifest`, owner Sho, updated `2026-07-09T01:09:17Z`).
**Live sha256:** `32a5b76ea9d734c26721e622269013863efedfb7953db16fc82b82a05959f5b6`
(`Team_Best_Onnx/task061.onnx`).

## Route 1: sparse bit constants

Attempt: `prototypes/build_task061_v19_sparse_bits.py` converts the three dense
INT64 bit constants to sanitizer-stable sparse initializers.

Exact gates/results:
- `ng_gate visible --task 61 --cand .../task061_v19_sparse_bits.onnx --json`:
  `correct=267 total=267 pct=100.0 pass=true`
- `score_network_v2(.../task061_v19_sparse_bits.onnx)`: `None`
- `onnx.checker.check_model(full_check=True)`:
  `[ShapeInferenceError] (op_type:BitwiseAnd, node name: safe_name_5): A typestr: T, has unsupported type: sparse_tensor(int64)`
- `onnx.shape_inference.infer_shapes(strict_mode=True)`:
  `Gather ... data tensor must have rank >= 1`; final `BitwiseAnd` infers sparse/tensor type mismatch

Why dead: raw ORT can execute it, but the official scorer/checker rejects sparse
operands for both `Gather` and `BitwiseAnd`. This cannot be staged despite the
visible pass.

## Route 2: UINT32 / INT32 bit cover

Attempt: `prototypes/search_task061_bitcover32.py` models every BitwiseAnd bit as
a legal row x `(channel,col)` rectangle enabled for a subset of `n in {4..9}`.
It enumerates all maximal valid shared rectangles and solves exact set cover.

Exact result:
- `candidate_rectangles=735`
- `true_elements=1944`
- `milp_status=0 success=True`
- `optimum_bits=39.0`
- `selected_bits=39`
- `uint32_possible=False`

Why dead: a UINT32 renderer has at most 32 positive bit positions; the exact
rectangle-cover optimum is 39, matching the live INT64 design. Plain dtype
narrowing of the bitset carrier is therefore impossible.

## Route 3: mechanical polish

Attempt: task-local copy of the live parent plus `tools/polish_candidate.py`.

Exact results:
- `polish_candidate.py`: `original cost: 636`; `polished cost: 636 (-0 bytes)`;
  no-op patterns `onnxsim, safe_opt, const_to_init, clip_mul_to_min, scalar_broadcast_ones, slice_axis_collapse`
- `profile_real_cost.py --task 61 --cand .../task061_v20_parent_polished.onnx --arcgen 266`:
  `static=636 profile=636 (memory 296 + params 340; 267 real inputs)`
- `ng_graph_cost_diff.py`: `NET COST DELTA 0 -> tie (no charged-cost change)`

Why dead: no charged structure changes versus live.

## Route 4: signed INT32 inhibitor bitset

Reviewer-locked rework (2026-07-13): test whether lane 31 can be used as an
INT32 sign/inhibitor bit so the current 39 positive INT64 bitsets collapse into
31 positive lanes + 1 sign lane, preserving the exact 5-node free-output
`BitwiseAnd` family and adding no final `Greater`/`Cast`/`Where`/`Equal`.

Exact command:
- `python The_400_Tasks/Task_by_Task/task061/prototypes/search_task061_bitcover32.py --signed-inhibitor`

Exact result:
- `signed_inhibitor_pair_model=true`
- `true_elements=1944`
- `incompatible_pairs=1522632`
- `compatible_pairs=365964`
- `certificate_clique_size=38`
- `positive_lanes_available=31`
- `signed_int32_possible=False`

Why dead: for any pair in the 38-witness certificate, one positive lane covering
both true logits necessarily creates false positive cross-logits. The minimal
sign rectangle that would suppress those false positives also hits at least one
true logit, making that true channel negative and violating the `>0` decoder.
Therefore each witness requires a distinct positive lane. Since signed INT32 has
only 31 non-sign positive lanes, no <=32-lane signed-inhibitor renderer exists
under this exact 5-node family.

## One remaining hypothesis

Only reopen for a genuinely different non-bitwise renderer that avoids the
300-element `col_bits` initializer without materialising a full
`[1,10,30,30]` intermediate. Plain sparse constants, standard polish, positive
UINT32 packing, and signed INT32 inhibitor packing are closed.
