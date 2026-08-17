# Transferable insight: the `--arcgen 0` profile mirage + single-channel-crop law

## The trap (from task029, 2026-06-18)

`tools/agent_helpers/profile_real_cost.py` defaults to `--arcgen 0`, which replays
ONLY the 4 train+test inputs. For any graph with a **data-dependent dynamic crop**
(Slice whose extent depends on the input), the 4 train+test inputs do not contain
the largest crop the generator produces, so the observed runtime shape is small and
the profile looks cheap. The OFFICIAL grader (`calculate_memory`, §39) profiles the
**arc-gen split too** and charges `max(static, runtime)` per tensor — so the real
credited cost is the `--arcgen 262` number, which can be 2-3x the 4-input number.

task029 concrete evidence (same graphs, two profiles):

| candidate | `--arcgen 0` (4 inputs) | `--arcgen 262` (266 inputs) |
|---|---|---|
| v18 direct one-hot slice | 6,796 | **19,436** |
| v17 colour-canvas F32 crop | 8,481 | **10,061** |
| v22 complete-border | 8,656 | **21,296** |
| v11 fixed K=23 (static) | 11,556 | 11,556 (gap 0) |

The whole task029 "ladder" (6,796 / 8,481 / 8,656) was a measurement artifact. The
team-best "8,481" never survived the real grader. **RULE: always profile dynamic-crop
graphs with `--arcgen 262`. A candidate is only honest (safe ship) when
arcgen-profile == static, gap <= 5%.** Consider making `--arcgen 262` the default in
profile_real_cost.py, or at least loudly warning when `--arcgen 0` is used on a graph
containing a dynamic Slice/Pad.

## The win: single-channel U8-canvas crop beats both F32 variants

For a "crop an interior window and re-emit one-hot" rule, the dynamic crop dtype/shape
is the dominant honest cost. Three ways to crop, cheapest last:

1. Slice the F32 **one-hot input** directly -> crop `[1,10,23,23]` F32 = **21,160 B**
   (10 channels!). This is why v18/v22 are credit-probes.
2. Slice an **F32 colour-index canvas** -> crop `[1,23,23]` F32 = 2,116 B, then Cast->U8.
   (v17.)
3. **Cast the colour-index canvas to U8 ONCE** (`[1,30,30]` U8 = 900 B), then Slice the
   U8 canvas -> crop `[1,23,23]` U8 = **529 B**. Pad(fill=sentinel) -> final Equal ->
   free BOOL one-hot output. (task029 v23: honest 9,510, gap 0%.)

The colour-index Einsum `bchw,c->bhw` `[1,30,30]` F32 (3,600 B) is the cheapest single-
channel colour map (ArgMax->INT64 is 7,200; integer Einsum needs a 36,000 B INT cast of
the one-hot). Annotate every dynamic tensor at the TRUE generator max (verified by
replaying all official inputs through the numpy selector) so static == profile.

Builder: `The_400_Tasks/Task_by_Task/task029/prototypes/build_task029_v23_u8canvas_honest_slice.py`
