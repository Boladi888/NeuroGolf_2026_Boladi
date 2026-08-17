## Pad default constant_value can save one param

**What I discovered:** For constant-zero/False Pad, the third `constant_value` input can be omitted. ONNX Pad defaults it to zero; ORT 1.24.4 and the NeuroGolf scorer accept this for BOOL Pad at opset 13. This saves the scalar initializer param that many hand-built graphs currently pay as `padval=False` or `zero`.

**Where encountered:** task083, `The_400_Tasks/Task_by_Task/task083/prototypes/build_task083_freeout_padlast_nopadval.py`, staged at `staged_submits/task083_freeout_padlast_nopadval_cost680/`.

**Why it matters:** It reduced the byte-equal task083 Pad-last graph from cost 681 to 680 with no correctness or profile gap:
- visible 266/266
- truth 200000/200000
- AB 200266/200266 byte-equal
- score/profile 680 = memory 648 + params 32

**What's missing from the library:** `techniques.md` and `ort_reality.md` already mention default/optional Pad constant handling, but they do not make the scoring implication explicit: when the pad fill is zero/False, omit `constant_value` to save the scalar initializer unless an opset/axes form requires occupying the input slot.

## Sorter note: techniques.md / ort_reality.md — extend existing Pad default/optional-input notes with the scorer-param shave and task083 exemplar.
