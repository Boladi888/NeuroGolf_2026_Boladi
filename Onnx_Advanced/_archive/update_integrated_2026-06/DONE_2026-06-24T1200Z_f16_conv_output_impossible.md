# Dead-end: you CANNOT get a smaller-dtype Conv output from an F32 official input read

**Where:** task017 (`The_400_Tasks/Task_by_Task/task017.md`), re-pricing the staged
team-best `staged_submits/task017_v31_drop_p2_cost9892/`. Prompt asked to "do the
label-extraction Conv in F16 to halve the 1764 B F32 read → 882 B".

**What I discovered (reusable for any full-plane F32-input read):**

The grader does NOT charge the official graph input (it's the input, free). It charges
INTERMEDIATES. When the cheapest formulation reads the full F32 official input with a
single op (here a 1×1 negpad channel-collapse Conv producing an F32 [1,1,21,21] = 1764 B
label plane), that F32 Conv OUTPUT is the forced full-plane read cost.

The intuitive "just make the Conv emit F16" lever is **structurally impossible**:
1. ONNX `Conv` requires X, W, Y to share one dtype (ort_reality dtype matrix already
   notes `Conv(F32 in, F16 w)` FAILS load; cost_floor task400 already records "ORT forces
   Conv→F32"). So an F16 Conv output requires an F16 Conv INPUT.
2. The only F16 input you can give it is a `Cast(input → f16)`, which materializes the
   FULL [1,10,30,30] F16 plane (18000 B) — charged. **I built and scored the variant:
   COST 26128 (mem 25936) vs 9892 baseline = +16236 WORSE.** (probe builder:
   `The_400_Tasks/Task_by_Task/task017/prototypes/_task017_v32_f16labelconv_probe.py`)
3. Slice-then-cast-then-F16-Conv is also worse: Slice input→[1,10,21,21] F32 (15876) +
   Cast→F16 (7938) + F16 Conv out (882) = 24696 B new intermediates vs 1764 baseline.
4. ArgMax (the other full-input label read) outputs INT64 only → 21×21×8 = 3528 B, worse
   than the 1764 F32 Conv.

**General rule:** for a one-op full-plane read of an F32 official input, the F32 Conv
output (4 B/elem over the active grid) is the floor; you cannot shrink it via dtype
because Conv inherits input dtype and any pre-cast/pre-slice materializes a larger
charged intermediate than you save. Only a formulation that NEVER reads the full F32
plane (e.g. a downstream op that consumes a region directly, or a different op class)
can beat it — and that changes the algorithm, not just the dtype.

**What's missing from the library:** ort_reality and cost_floor each note pieces of this
(Conv dtype constraint; "ORT forces Conv→F32"), but there's no single "F16-Conv-output
on an F32 input is impossible AND pre-casting always loses — here's the measured 26128"
entry workers can find when a prompt suggests this lever. The task017 Don't-do log now
has the measured falsification; consider promoting a one-liner to cost_floor's full-plane
read section so it isn't re-walked per-task.
