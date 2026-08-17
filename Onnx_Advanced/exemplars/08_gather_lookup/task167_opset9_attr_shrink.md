# task167 — opset9 attribute shrink

| Field | Value |
|---|---|
| Task | task167 (also task103, task056) |
| Cost reduction | -22 each |
| Pattern | opset-9 attribute-form Gather → minimal-attr representation |

## Pattern

Some Gather/Slice nodes have redundant attributes (axes specified when default works, axis specified when it's already 0, etc.). Removing the redundant attributes shrinks the node's serialized representation slightly.

This is a small per-op optimization — not high-leverage individually, but adds up across many nodes in dense graphs.

## Reproducibility

`onnxsim` typically catches this. If `onnxsim` no-ops on your graph but you have many Gather/Slice with explicit default attributes, a manual pass may find a few bytes.

## Cross-ref

- `08_gather_lookup/task266_scalar_broadcast.md` — different init compression.
- `techniques.md` §18 (const_to_init — moving Constant nodes to initializer reduces overhead similarly).
