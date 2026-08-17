## Tiny-K scalar selector can beat ArgMax plus attribute stacks

**What I discovered:** For a tiny fixed number of scalar candidates, a chained
`Greater`/`Where` update can be cheaper than `Concat` -> `ArgMax` -> `Gather`
from parallel attribute stacks. In task365, the prior v16 winner tail stacked
three red-count scalars, `ArgMax`ed the length-3 vector, then gathered
`r0/cL/tall/wide` from four length-3 INT32 stacks. Replacing that with two
scalar comparison/update steps was byte-equal and saved 90 bytes of profiled
memory.

Pattern:

```text
gt01 = Greater(score1, score0)
score01 = Where(gt01, score1, score0)
attr01 = Where(gt01, attr1, attr0)
gt2 = Greater(score2, score01)
attrw = Where(gt2, attr2, attr01)
```

This works best when the selected attributes are small scalars/vectors and
ties are impossible or the `>`/`>=` tie behavior is explicitly matched. It is
not a replacement for selecting large output branches; `Where` would
materialize both branches in that case.

**Where encountered:** task365, builder
`The_400_Tasks/Task_by_Task/task365/prototypes/build_task365_u8_v18_scalarwhere_selector.py`,
staged candidate `staged_submits/task365_scalarwhere_v18_cost3262`.

**Why it matters:** The prior polished v16 was 3352/profile 3352. This selector
rewrite preserved visible 266/266, truth 200000/200000, and A/B byte equality
against v16 over 200266 pairs, while dropping profile cost to 3262.

**What's missing from the library:** I found related selector notes and a
general warning that `Where` over full output branches is expensive, but no
small-K scalar-attribute selector cost rule.

## Sorter note: techniques.md — add as new section or extend a selector/mux section with this tiny-K scalar-selector rule
