# 08_gather_lookup — worked case studies

Lookup-table compression and Gather optimizations.

## Files

- `task266_scalar_broadcast.md` — Uniform-value init → scalar broadcast. Mechanical.
- `task167_opset9_attr_shrink.md` — Drop redundant attributes on Gather/Slice nodes.
- `task129_tile_to_edge_pad.md` — Replace Tile with Pad(mode="edge") in narrow cases.
- `task118_bool_scatter.md` — ScatterND with BOOL inputs (opset 11+).

## When applicable

- Large lookup tables with structure → compress via separable / scalar / packing.
- Tile ops where the goal is edge replication → switch to Pad(mode="edge").
- ScatterND ops where the output is 0/1 → switch to BOOL.
- Gather ops with redundant attributes → cleanup via attr-shrink.

## Cross-refs

- `task118_bool_scatter.md` (this category) is the ScatterND home.
- (Numpy_Advanced/ folder removed 2026-05-23; init-construction tricks are inline in `Algorithm_Library/algorithm_patterns.md` and `traps.md §17`)
