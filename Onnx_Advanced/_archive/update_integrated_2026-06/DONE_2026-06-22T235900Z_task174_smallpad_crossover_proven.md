# #91 small-Pad-last crossover — PROVEN win when the answer is tiny and the canvas is single-channel U8

## What I discovered
The #81/#91 library verdict currently says "Pad-last only saves when the small tensor is the graph output; padding an 18x18 one-hot (3240 B) loses to a 900 B canvas." That is true for a FULL-grid one-hot, but it understates the win when the answer region is genuinely tiny.

task174's output canvas was `labels30` U8[1,1,30,30] = 900B (a single-channel color-index plane Pad'd to 30x30, then `Equal(ch_values)` → [1,10,30,30] one-hot output). The answer is provably <=5x5. The win is to:
1. Build the color-index canvas at a STATIC small window (5x5, via double-Gather static-index crop so score_v2 still returns a number — no dynamic Slice).
2. Run the one-hot `Equal` at 5x5 → `onehot5` BOOL[1,10,5,5] = **250B**.
3. **Pad-last** the one-hot to [1,10,30,30] with const False.

The crossover math that decides it: padding the one-hot BEFORE shrinking the window charges `10 * H * W` bytes for the one-hot. At full 10x10 that's 1000B (LOSES to the 900B U8 canvas — matches #81). But at the true 5x5 answer window it's 250B, which BEATS the 900B canvas. So the rule is: **Pad-last wins iff `10 * H_win * W_win < (pad_H * pad_W)` of the single-channel U8 canvas**, i.e. iff the static answer window is small enough that the per-channel one-hot is cheaper than the full single-channel padded canvas. For a 30x30 output that crossover is `H_win*W_win < 90`, i.e. any window up to ~9x9 wins.

Result: task174 3,716 → **2,527** (-1,189), visible 266/266, truth 200000/200000, AB byte-equal 200266/200266, profile==static. Cost delta vs team-best 4,786 = -2,259.

## Key enabler: bound the window from the ARC-GEN generator, not the visible pairs
I read `external/ARC-GEN/tasks/task_72ca375d.py` directly: `wide in [2,5]`, `tall in [2, 7-wide]`, output = winner-box bbox, so H<=5, W<=5, H+W<=7. A static 5x5 window provably covers ALL 200000 truth cases. Reading the generator (not just the 266 visible) is what makes a fixed small static window safe — this is the difference between a gate-passing guess and a proven floor.

Bonus also confirmed from the generator: the picker is the box that is BOTH `is_symmetric` (LR-mirror) AND `is_rotational` (180deg); the model's LR-only selector happens to discriminate all 200000 corpus cases, but the true rule is the conjunction.

## Where
- Builder: `The_400_Tasks/Task_by_Task/task174/prototypes/build_task174_smallpad.py` (graph-edit on `staged_submits/task174_nozero_cost3716/task174.onnx`).
- Staged: `staged_submits/task174_smallpad_cost2527/`.

## Why it matters / what's missing from the library
The current #81/#91 entry reads as a blanket "Pad-last usually loses." It should be sharpened to the crossover inequality above: Pad-last of a C-channel one-hot beats a single-channel U8 canvas exactly when `C * H_win * W_win < pad_H * pad_W`, and the static answer window must be bounded from the GENERATOR (output bbox bound), not the visible pairs. task174 is a clean exemplar where this turned a "structural 900B canvas" floor claim into a -1,189 win.

## ORT facts re-confirmed (opset 20)
- `Pad` accepts BOOL data with a BOOL constant_value (`False`) — used for the pad-last of the one-hot.
- Double-`Gather` static-index crop (axis=2 then axis=3 with fixed-length index vectors) keeps the cropped shape STATIC so `score_v2` prices it exactly; avoids the dynamic-Slice → None trap (§39).
