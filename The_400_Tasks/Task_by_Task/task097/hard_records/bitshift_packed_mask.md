# task097 bitshift_packed_mask dead route

- live bar: 910, SHA e11f658842029acf14bc7f02675371d143c017e0709af40a14889bf685d7340e, source `python tools/ng_current_bar.py --task 97 --json` / team manifest.
- attempted route: row-packed UINT32 mask lanes using `BitShift`/`Bitwise*` for the 8-neighbour non-bg keep predicate, followed by output-boundary decode.
- exact failed gate: price probe before build; `python The_400_Tasks/Task_by_Task/task097/prototypes/price_bitshift_packed_mask.py` gives `LOWER_BOUND_PACKED_CORE_BEFORE_FINAL_REPAIR=2640`, already above the 910 bar before ch0 repair, final 10-channel decode, constants, row indices, pow masks, or any data-dependent color-selection helper.
- why dead: the current best is a one-node free-output Conv with zero intermediate memory; BitShift requires integer carriers, so the F32 one-hot input must first be packed/cast into charged row lanes, and even the non-bg packed body exceeds the whole live graph before producing a valid 10-channel full-canvas output.
- one remaining hypothesis: only a new single-node/free-output formulation can beat 910; any multi-node packed representation must first show a sub-910 analytic bound with all adapter/decode tensors included.
