route: ArgMax + UINT8 OneHot for background/fill selector
live_bar: 353, sha256 d80a6a7add9e2aeacf687bf4839439867f3959ceed8430604ed14a6c735ab1f7, source `python tools/ng_current_bar.py --task 65 --json`
attempt: `prototypes/build_task065_w2_argmax_u8_bg.py`, intended to replace `ReduceMax -> Equal -> Where(bg,0,1)` with `ArgMax -> OneHot(values=[1,0])`
failed_gate: build/shape-inference failed: `OneHot` requires indices rank >= 1 when ArgMax over `[10]` uses `keepdims=0`
why_dead: making indices rank-1 requires keeping/squeezing a `[1,10]` or `[10,1]` OneHot result; that adds at least one extra 10-byte reshape/squeeze surface plus params, so it cannot beat the current 353-byte parent.
remaining_hypothesis: only revisit if a legal scalar-index OneHot-to-vector form is discovered; current ONNX shape rules make the direct U8 selector shave unavailable.
