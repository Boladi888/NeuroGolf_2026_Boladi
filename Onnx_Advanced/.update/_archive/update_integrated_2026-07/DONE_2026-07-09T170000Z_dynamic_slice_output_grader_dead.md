# Dynamic-shape OUTPUT is grader-dead even when it scores cheaply (variable-crop tasks)

## What I discovered
For a variable-output-size task (crop the bbox → emit HxW, H,W vary), a **bare data-dependent `Slice` whose result is the graph `output`** prices *extremely* cheap — the scorer skips `input`/`output` in `calculate_memory` (`neurogolf_utils.py:234`), so a bare-Slice crop with a baked static `output` value_info scored **469** (memory 464 + params 5) on task310, vs the 1485 live radix. It is also 262/262 arc-gen + 4/4 train/test **content-correct** when you decode ORT's actual runtime output.

**But it can never ship.** The official grader is `np.array_equal(user_output, benchmark["output"])` (`neurogolf_utils.py:546`), and `benchmark["output"] = convert_to_numpy(example)["output"]` is **always [1,10,30,30]** (padded). `np.array_equal([1,10,8,8], [1,10,30,30]) → False`. ORT even warns `VerifyOutputSizes {1,10,8,8}≠{1,10,30,30}`. So a graph whose runtime output shape ≠ [1,10,30,30] fails **every** example (0/266 visible), regardless of content. `ng_gate visible` reproduces the official behavior exactly.

**Corollary / sharpening of the existing dynamic-Slice note.** `cost_floor_reference.md` (task203/task036) says "re-test fixed-shape dynamic Slice with baked value_info." My finding splits that into two regimes:
- **Baked value_info on an INTERMEDIATE** (e.g. Slice→Pad, crop is internal) → scores fine; the runtime output stays [1,10,30,30]. ✅ (task310 Slice→Pad: 266/266, cost 3133.)
- **Baked value_info on the OUTPUT itself** (Slice IS the output) → scorer accepts it (output skipped) but the **grader's `np.array_equal` rejects the shape**. ❌ Cheap-but-dead. The [1,10,30,30] output contract is *grader-level*, not just scorer-level — you cannot buy the "output is free" saving by shrinking the output tensor.

## Where
task310 (crop-bbox). Probes: `The_400_Tasks/Task_by_Task/task310/prototypes/build_task310_direct_slice/` (469, grader-dead) and `build_task310_slice_pad/` (3133, 266/266). Full floor analysis: `The_400_Tasks/Task_by_Task/task310/hard_records/external_crop_direct_tail.md`.

## Why it matters
Kills a whole tempting family for every variable-output task: "just Slice/Gather the crop and output it, memory ≈ 0." The output-tensor-is-free lever only pays off when the output is *already* the forced [1,10,30,30] shape (free-output Einsum / final Pad). Saves future batches from re-deriving the 469-and-dead result.

## What's missing from the library
`cost_floor_reference.md` documents dynamic-Slice-scores-None and baked-value_info-fixes-scoring, but does NOT state the grader-level [1,10,30,30] `np.array_equal` output contract as the reason a *dynamic output* is dead even after it scores. Add that distinction.

## Sorter note: cost_floor_reference.md — extend the dynamic-Slice/baked-value_info note with the "OUTPUT tensor must be exactly [1,10,30,30] (grader np.array_equal, neurogolf_utils.py:546); baked-value_info buys scoring but NOT a shrunk output" distinction; cite task310 probes (469 grader-dead vs 3133 correct).
