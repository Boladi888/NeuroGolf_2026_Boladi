# task330 QLinearConv optional q_zero shave = DEAD (2026-07-12)

**Live bar at attack time:** manifest/profile cost **2121** (memory 1980 +
params 141), owner Vamshi, sha256
`b9e27000233af9777ab5b300fd967c78cc5bc24214c7b31eb3c2656f5157308e`.

## Attempted route
Delete the scalar `q_zero` initializer from the current 2121 graph and omit the
three QLinearConv zero-point inputs, relying on zero as the default. Prototype
builder: `prototypes/build_task330_omit_qzero_from_teambest.py`.

## Exact failed gate
`python The_400_Tasks/Task_by_Task/task330/prototypes/build_task330_omit_qzero_from_teambest.py`

Fails before ONNX save:

`ValidationError: Node ()'s input 2 is marked single but has an empty string in the graph`

## Why it is dead
For this pinned ONNX checker/schema, QLinearConv input 2 is not optional in the
way Pad/Resize optional placeholders are. The zero point scalar is therefore a
required charged initializer; omitting it cannot produce a validator-clean
candidate.

## One remaining hypothesis
Only non-schema label/layout changes remain. A bounded 9-bucket relabel search
over the current 5x5 block-label counter reached 262/266 at numpy level but no
266/266 pattern, so a real win likely needs a new component-counting idea rather
than QLinearConv optional-input cleanup.
