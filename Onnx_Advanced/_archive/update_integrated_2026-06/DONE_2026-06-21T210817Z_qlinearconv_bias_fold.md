## QLinearConv Detector Bias Fold

What I discovered:

For a QLinearConv detector whose UINT8 output clamps negative scores to zero, a small negative INT32 bias can sometimes be removed by rescaling the positive witness weights and anti-background weights so that true cases remain positive and every false case is zero or negative before clamping.

In task034, the polished graph used:

```text
score = 6 * red_at_corner - 2 * bg_count - 5
```

That produces seed 1 only when the target 2x2 corner is red and the 2x2 window has no background. The equivalent no-bias detector is:

```text
score = 1 * red_at_corner - 1 * bg_count
```

Truth table under the task034 generator:

- aligned red square corner: `1 - 0 = 1`
- red corner in a background-touching false window: `1 - bg_count <= 0`
- non-red square corner: `0 - 0 = 0`
- background window: `0 - bg_count < 0`

Because QLinearConv emits UINT8, the zero/negative cases clamp to 0. This removed the 4-element `seed_b_q` INT32 initializer without changing the seed map.

Where I encountered it:

- Task: task034
- Builder: `The_400_Tasks/Task_by_Task/task034/prototypes/build_task034_nobias_v1.py`
- Staged candidate: `staged_submits/task034_worker034a_nobias_cost3492_shaB3190D10/`
- Result: cost 3496 -> 3492, full gates clean (`visible 267/267`, `truth 200000/200000`, `ab 200267/200267`, `profile_real_cost --arcgen 266` static/profile 3492).

Why it matters:

The gain is small on task034, but the pattern is reusable for QLinearConv template detectors where a bias only exists to require all non-target cells to be non-background or to enforce an exact-match threshold. If the false cases can be made non-positive by weight scaling alone, deleting the bias saves one initializer element per output channel without adding any charged tensor.

Rejected sibling:

I also tested omitting QLinearConv zero-point inputs with empty optional slots. The checker rejects it (`input 2 is marked single`). The existing library already mentions this in `by_symptom.md`; this note is not asking to add that fact again.

What is missing from the library:

I found QLinearConv coverage in `techniques.md` (#72/#85/#120/#137) and `by_symptom.md`, but not this bias-to-weight fold for clamped detector banks.

## Sorter note: techniques.md QLinearConv detector sections (#72/#85/#120 or nearby) - extend existing with this micro-rule, not a new broad QLinearConv section.
