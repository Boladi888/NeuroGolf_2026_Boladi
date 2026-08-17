## Signed paint/compute initializer sharing

**What I discovered.** In tasks where the final output is decoded by the official
`>0` threshold, a single signed channel tensor can sometimes serve two roles:

- as a compute weight tensor with negative values on channels that should not be
  painted, and
- as the final paint branch for `Where(mask, paint, input)`, because negative
  channels decode as off while the target channel remains positive.

For task335, v19 had separate `code_weight[10]` (`ch8=32`, `ch2=1`) and
`color4[10,1,1]` (`ch4=1`) initializers = 20 params. I replaced them with one
`shared_weight[10,1,1]` (`ch8=-32`, `ch2=-1`, `ch4=1`) used by both endpoint
coordinate `Einsum`s and the final paint `Where`. Two scalar `Neg` nodes restore
positive packed codes before `UINT16 Div/Mod`.

**Where.**

- Build script:
  `The_400_Tasks/Task_by_Task/task335/prototypes/build_task335_w3_signed_weight_share.py`
- Staged candidate:
  `staged_submits/task335_w3_signed_weight_share_cost1344/task335.onnx`

**Why it matters.** This saved 2 bytes on a task that had been documented as a
1346 floor: params drop by 10, scalar memory rises by 8, static/profile cost is
1344. Gates passed: visible 266/266, truth full 200000/200000, AB full
200266/200266, profile_real_cost static=profile=1344.

**What is missing from the library.** Existing docs mention signed weights and
the official `>0` decoder separately, but I did not find this exact initializer
sharing idiom: use negative compute-only channels so the same tensor is safe as
a paint branch under threshold decoding.

## Sorter note: techniques.md — add as a micro-idiom or extend the free-output/Where-output catalog with "signed paint/compute initializer sharing"
