## TopK over full palette can remove color-id shifts

**What I discovered:** If a graph slices non-background channel presence `[1,9]`, ranks it with `TopK`, then adds `+1` to recover ARC color ids, it may be cheaper to keep a full `[1,10]` score vector instead. Set channel 0's score weight to 0 and non-background weights positive. When the generator guarantees exactly K non-background colors, `TopK(k=K)` returns true ARC color ids directly, so the non-bg `Slice` and downstream `+1` color-id `Add` tensors disappear.

**Where:** task174, builder `The_400_Tasks/Task_by_Task/task174/prototypes/build_task174_score10_boolreduce.py`, staged candidate `staged_submits/task174_score10_boolreduce_cost2225/`.

**Why it matters:** On task174 this changed `Slice(presence10)->presence [1,9]` + `Mul -> score [1,9]` + four scalar color-id Adds into one `Mul(presence10, present_weight[1,10]) -> score [1,10]`. It saved 48B on top of the prior 2273 rung, producing a gate-clean/profile-static cost 2225 candidate. Safety condition: all selected non-background channels must have strictly positive scores while channel 0 and absent colors have score 0, so `TopK` never selects channel 0.

**What's missing from the library:** I found TopK stream-compaction and ranking routes, but not this palette-index idiom for avoiding non-bg slicing and `+1` repairs.

## Sorter note: techniques.md — extend the TopK/ranking technique area or add a micro-idiom near #99/#123 for "full-palette TopK gives real color ids"
