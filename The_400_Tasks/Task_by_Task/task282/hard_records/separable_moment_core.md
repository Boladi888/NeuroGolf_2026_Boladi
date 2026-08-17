# DEAD ROUTE - separable row/column moment core - task282

- **Live bar:** 345 (manifest authoritative), sha256 `c1d1471a4566e8315d6901754864eb540ae28baece27ec97f1f6ff3760673ffb`, owner Vamshi, updated 2026-07-02T02:35:15Z.
- **Date:** 2026-07-14 · worker `bo_ng_worker1`

## Attempted route

Test the prompt's proposed separable/moment rewrite: reduce the input to row/column/color
marginals, then render the 3x3 color-5 stamps with a free-output separable Einsum instead
of the live `Slice -> QuantizeLinear -> ConvInteger` graph.

## Exact failed gate / probe

Prototype probe: `The_400_Tasks/Task_by_Task/task282/prototypes/moment_collision_probe.py`

Output:

```text
fresh_arcgen_pairs=200 max_seeds=4 max_seed_matrix_rank=4
fresh_row_col_signature_collisions=3
constructed_pairing_counterexample a=((1, 1), (4, 4)) b=((1, 4), (4, 1)) valid_a=True valid_b=True same_row_col_marginals=True output_diff_cells=32
direct_separable_stamp_param_floor placement=1260 channel=90 color_selector=10 total>=1360 live=345
```

## Why it is dead

Row/column marginals are not sufficient: valid seed sets can share the same row and
column counts while requiring different stamp pairings. Example:
`[(1,1),(4,4)]` and `[(1,4),(4,1)]` both satisfy the task spacing invariant and have
identical row/column marginals, but their stamped outputs differ in 32 cells.

The direct free-output separable-stamp fallback is also priced out before build. Even
ignoring all plumbing, shifted row/column placement maps for 3 offsets over 7 possible
seed positions to 30 output slots cost at least `2 * 3 * 7 * 30 = 1260` initializer
elements. Adding 3x3 channel routing (`90`) and the color selector (`10`) gives
`>=1360`, already 1015 bytes worse than the live 345. This does not solve the cheaper
live graph's boundary distinction either; the live zero-point-padded `ConvInteger` is
the compact mechanism that lights in-grid background while keeping beyond-grid cells all
zero.

## Remaining hypothesis

Only a compression inside the existing `Slice -> QuantizeLinear -> ConvInteger` graph
could plausibly tie or improve the bar. The current forced components are already
minimal in the task note: `gray7` f32[1,1,7,7] (196), `code7` u8[1,1,7,7] (49),
3x3 10-channel weights (90), Slice starts/ends (8), scale/zp (2).
