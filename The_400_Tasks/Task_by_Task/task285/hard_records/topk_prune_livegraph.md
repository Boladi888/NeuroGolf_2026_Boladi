# task285 TopK-Prune Live-Graph Probe

- live bar: 13,361 profile cost, SHA `9D6EB4B8D922251A97AC2FA9AF1314A329C6AF5D31AC1A5F6ECF866E6C9FECFF`, source `python tools/ng_current_bar.py --task 285 --json` manifest row updated `2026-07-04T05:54:11Z`.
- attempted route: current live graph with `share_axes3_c3`, then first-stage key TopK `10 -> 9` and selected-offset TopK `8 -> 7`.
- exact failed gates: `stage_k9` scored 13,305 then `[visible] 263/265 (99.25%)  FAIL`; `off_k7` scored 13,185 then `[visible] 264/265 (99.62%)  FAIL`.
- why dead: both prunes remove a required visible key/offset case; by the visible-fail axiom, lower-K variants in this representation are hidden-zero probes, not candidates.
- one remaining hypothesis: only a different representation that removes the full scalar-label/canvas family could beat the current floor materially; simple K-pruning of the current live graph should not be retried.
