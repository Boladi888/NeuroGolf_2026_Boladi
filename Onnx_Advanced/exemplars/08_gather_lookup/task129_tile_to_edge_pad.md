# task129 — Tile → Edge-Pad

| Field | Value |
|---|---|
| Task | task129 |
| Pattern | Replace Tile with Pad(mode="edge") |
| Cost reduction | -4 (small, but structural cleanup) |

## Before

```
Tile(input, repeats=[1, 1, K, 1])   # repeat K times along H axis
```

## After

```
Pad(input, pads=[..., 0, 0, K-1, 0, ...], mode="edge")
```

If the goal is to extend the input along one axis by replicating the edge values, Pad with mode="edge" achieves the same result as Tile in some cases — and avoids materializing the full tiled tensor.

## Why it works (when it does)

Tile produces a full repetition. Edge-Pad fills with the edge value. These coincide when:
- The "tiled" content is just the edge values of the original input.
- OR you're effectively extending the boundary.

For genuine multi-copy tiling (where the content is a non-edge block), Tile is required. The substitution only applies in the narrow case.

## Cross-ref

- `Spatial_and_Geometric` family — common pattern for boundary extensions.
- `techniques.md` Anti-patterns: Tile is rarely the best op for the cost.

## Reproducibility

Identify Tile ops in your graph. Check if the desired output is just edge replication. If yes, replace with Pad(mode="edge"). Verify byte-exact.
