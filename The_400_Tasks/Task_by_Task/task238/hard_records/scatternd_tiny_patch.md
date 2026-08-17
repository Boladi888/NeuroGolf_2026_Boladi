# task238 hard record: ScatterND tiny-patch route

- Live bar used: 1,999 bytes, SHA `14e56fdc4e64dacd34b96212744bdf3aca2c865da71421ff3c95077a27c78813`, source `python tools/ng_current_bar.py --task 238 --json` (team manifest).
- Attempted route: replace the dense slot/Where interior overlay with a sparse ScatterND patch driven by active scatter cells.
- Probe: `python The_400_Tasks/Task_by_Task/task238/prototypes/probe_task238_patch_vs_where.py --n 3000`.
- Failed gate: prompt fast-probe price gate, before ONNX build.
- Exact result: fresh ARC-GEN active scatter cells ranged 3..16; dense condition mask was cheaper than active ScatterND indices on 3000/3000 pairs. ScatterND color-index active-patch lower bound was min 638, p50 770, max 1067 bytes before detector/palette/base-map costs; the graph must provision for the max case, so it cannot be a sub-700 candidate.
- Why dead: dynamic active indices would require banned/expensive index extraction; fixed ScatterND writes are even worse. The current v6 Where/condition family is the cheaper final assembly, but its measured full graph remains 1,999 bytes.
- Remaining hypothesis: only a different output-tail formulation that avoids the charged `[1,10,7,7]` BOOL tile could move the floor materially; cropped outputs are invalid because official `verify_subset` compares full `[1,10,30,30]` one-hot tensors.
