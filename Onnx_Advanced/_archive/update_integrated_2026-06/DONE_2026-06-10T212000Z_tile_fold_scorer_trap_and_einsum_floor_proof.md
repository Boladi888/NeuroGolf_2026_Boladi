# Tile Constant-Fold Trap + Single-Einsum Phase-Repeat Floor Proof (task215, 2026-06-10)

## What I discovered

### Trap 1: Tile with all-init inputs does NOT escape memory charges

**Symptom**: Building `Tile(eye3[3,3], reps[2])` to regenerate `phase_map[30,3]` from a 9-param
base + 2-param reps. `score_network_v2` showed cost=31 (vs LIVE 110). But `profile_real_cost`
showed cost=1111 — the Tile output was charged as a 360-byte intermediate.

**Root cause**: The `scoring_internals.md` note "constant folding runs first" refers to an older
version of `neurogolf_utils.py` (pre-2026-05-06). The **current** `calculate_memory` (as of
2026-06-10) uses `onnx.shape_inference.infer_shapes()` NOT constant folding. Shape inference
adds Tile outputs to `value_info` with their full shape ([30,3] = 360B each). The ORT profiler
(with `ORT_DISABLE_ALL`) also sees them as runtime intermediates. Both static + profiler charge
them.

**Measured values** (task215):
- 3 × Tile([30,3]) intermediates = 3 × 360B = 1080B memory
- Total cost with Tile approach: 1080 + 31 = 1111 (vs LIVE 110)
- Tile approach is 10× WORSE

**General rule**: Any node whose inputs are ALL initializers STILL produces a charged
intermediate if its output appears in graph `value_info`. Constant folding does NOT occur in
the current neurogolf_utils.py scoring path. The only way to have zero memory is to have a
single node whose output IS the graph output (named "output").

**Affected pattern**: Any attempt to use `Tile`, `Gather`, `OneHot`, `Range+Cast`, or any
op to "generate" a small base initializer into a larger one. They ALL charge the output.

### Trap 2: Tuple interpretation in score_network_v2

`score_network_v2` returns `(macs, memory, params)` — MACs first, then memory, then params.
Cost = memory + params (NOT macs + params). Easy to misread as (memory, macs, params).

### Finding: Single-Einsum floor for phase-3 tiling = 110 params

**Task structure**: "tile a 3-row sprite to all rows of matching phase mod 3". Single Einsum
with equation `nchw,nksw,rp,hp,sp,k,c->ncrw` requires:
- phase_map[30,3]: 90 params — minimum for row-to-phase one-hot. No smaller stored repr.
- fg_weight[10]: 10 params — cross-channel foreground detection. Cannot fuse with out_sign
  (k=c equation fails for background: background detection requires sum over ALL k, not just k=c).
- out_sign[10]: 10 params — background sign correction. Cannot be eliminated or merged.

**Probes attempted and rejected**:
1. Tile-fold phase_map: cost=1111 (Tile intermediates charged)
2. k=c fusion (fg_weight*out_sign → combined[10]): fails on all TRAIN examples (background
   detection incorrect: gives 81 at sprite positions instead of negative)
3. Drop sp term (2 instead of 3 phase_map usages): fails on all TRAIN examples (cross-phase
   foreground contamination: total FG count across all rows ≠ per-phase FG count)
4. Two-step template approach (template[1,10,3,30] intermediate): min 3600B memory >> 110 params

**Conclusion**: 110 is the mathematical floor for the current single-Einsum algorithm.

### Structural observation about task215 inputs

ARC-GEN always places the 3 sprite rows at rows 3-4 and 4-6 (first sprite row is 3 or 4).
Sprite is ALWAYS exactly 3 rows in the input (one per mod-3 phase). But this observation
doesn't help reduce params because the OUTPUT still spans all 30 rows and phase_map[30,3]
is needed for the output dimension.

## Where I encountered it

- Task 215, `The_400_Tasks/Task_by_Task/task215.md`
- Prototype: `The_400_Tasks/Task_by_Task/task215/prototypes/build_task215_v2.py`
  (the Tile-fold version that scores 31 statically but 1111 in profile)
- Test models: `task215_vA_tile3.onnx`, `task215_vF_fused.onnx`, `task215_vG_nosp.onnx`
  (all in project root, created during this session for testing)

## Why it matters

1. The "constant folding runs first" claim in scoring_internals.md is STALE/WRONG for the
   current neurogolf_utils.py. Workers building "Tile-then-Einsum" or "Range-then-Expand"
   patterns will be tricked by score_network_v2 showing low memory while profile_real_cost
   shows the actual charged cost.
   
2. The floor proof prevents future workers from spending time on task215 without a fundamentally
   different algorithm.

## What's missing from the library

- `scoring_internals.md` needs a correction: "constant folding runs first" is stale. The
  current `calculate_memory` uses `infer_shapes()` not constant folding. Any node with
  all-init inputs still charges its output. Workers must use `profile_real_cost.py` to verify;
  `score_network_v2` can show a falsely low memory score when Tile/etc. ops are present.
  
- `zero_param_generators.md` should add a note to Range/Tile/ConstantOfShape sections:
  "Even with all-init inputs, the node output is charged as a value_info intermediate. This
  CANNOT be avoided unless the output is the graph output (named 'output')."

- `traps.md` could add: "Tile-with-init trap: Tile(init1, init2) does NOT produce free output.
  shape_inference adds it to value_info. Measured: 3×Tile([30,3]) = 1080B extra vs 90 params
  for the stored initializer. ALWAYS check profile_real_cost when any non-Identity/non-Einsum
  op feeds the main Einsum."

## Sorter note: scoring_internals.md — extend "Constant folding runs first" with correction

The claim that constant folding eliminates all-init node outputs is wrong for the current
neurogolf_utils.py. Add a trap/correction note in the constant folding section.

## Sorter note: zero_param_generators.md §Range — add warning

Add to the Range and Tile sections: "The output of Range/Tile/ConstantOfShape is ALWAYS charged
as a value_info intermediate even when all inputs are initializers. The old constant-fold claim
was wrong. Use profile_real_cost.py to verify any multi-node graph."

## Sorter note: traps.md — add "Tile-fold trap" 

Add a trap entry for: Tile(init1, init2) appears free statically but is charged by infer_shapes.
Measured: 3 Tiles([30,3] F32) = 1080B. Workers building "compact base + Tile" architectures
to save params are losing on the memory side instead.
