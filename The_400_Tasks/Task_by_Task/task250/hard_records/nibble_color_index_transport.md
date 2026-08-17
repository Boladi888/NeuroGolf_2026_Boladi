# task250 hard record: nibble/color-index transport

- Live bar used: 2054 from `python tools/ng_current_bar.py --task 250 --json`, manifest source, SHA `ad105c42ffbb6f5f9dd67517f01d79cdc04657cacdbee9527a7012db34facdcd`.
- Attempted route: replace the dominant carrier with #83 nibble-packed/color-index transport and delay one-hot expansion to the output.
- Failed gate: prompt fast probe before build. Packed-executable consumer count is below four in the current 2054 slot-pool graph: the gray carrier feeds the 12 boolean slot pools as one consumer family, then the final painter must expand to NeuroGolf one-hot output. The ABI does not permit leaving output packed.
- Why dead now: #83 only pays while the packed plane remains the dominant reusable plane. Here the graph already runs on 5x5 U8 slot maps and a 10x10 U8 QLinearConv painter; packing would immediately unpack/expand for the final red/gray/background separation and cannot approach the prompt's sub-750 target.
- One remaining hypothesis: revive only if a new free-output construction consumes packed row/bitset lanes directly and eliminates the `planes [1,3,10,10]` QLinearConv tail; otherwise work scalar dtype/initializer shaves around the slot-pool formulation.
