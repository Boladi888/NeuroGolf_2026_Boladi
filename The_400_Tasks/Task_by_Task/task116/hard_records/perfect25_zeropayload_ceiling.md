# task116 — hard record: perfect-25 (cost≤1) zero-payload ceiling route is DEAD

- **Date:** 2026-07-07 (worker `bo_ng_cld_worker2`, Claude lane, batch53 ZeroopsweepCeilingAudit)
- **Live bar:** cost **30** (manifest authoritative, `ng_current_bar.py`), sha256 `3f87aab73b8c538752ffb070a34dd07ef95ccba2d299bd7526da67fcf956e1e5`, owner Vamshi.
- **Prompt hypothesis tested:** a "perfect-25" route paying only a no-payload direct-output op → cost ≤ 1 (Identity / all Transpose perms / attr-only flip/reshape).

## Route attempted
Zero-payload direct-output single-op sweep, scored via real pipeline + byte-checked
vs LIVE on the 267 bundled visible+arc-gen pairs. Repro: `prototypes/ceiling_probe.py`.

| op | cost | visible vs LIVE |
|---|---|---|
| Identity (output=input) | 0 | **0/267** |
| Transpose — all 24 rank-4 perms | 0 | **0/267** (every perm) |
| Reshape [1,10,30,30] | 4 (shape init counts) | **0/267** |

## Exact failed gate
`ng_gate visible` equivalent (in-probe byte-equality) = **0/267** for every
zero-payload candidate. Per §VISIBLE-FAIL AXIOM a visible miss is a certain
hidden 0 → truth/ab/battery NOT run (would be pure burn). No candidate is both
visible-correct AND cost ≤ 1, so the perfect-25 kill rule fires.

## Why it's dead (forced-component argument)
task116 is **not** an identity/transpose/flip/rotate task. The output is a
vertical **symmetric mirror-stack fold** of the 3 active rows:
`out_rows = [2,1,0, 0,1,2, pad×24]` (numpy `symmetric` pad-before, which ONNX
`Pad` does not implement). A fold **reorders rows WITH duplication** of the
boundary row (`…,1,0,0,1,…`). No zero-payload attribute-only op reorders rows
that way — Identity keeps order, Transpose only swaps axes, Reshape preserves
element order. Producing the 30 output rows in this order requires *specifying*
the row index, i.e. a 30-element `Gather` index → **30 params minimum**. That is
exactly LIVE (1× `Gather(input, idx[30] INT64, axis=2)`, memory 0 + params 30 =
cost 30). Perfect-25 (cost ≤ 1) is structurally impossible here; the forced
floor is **30**, matching the live bar. Independently re-confirmed floor on
2026-05-19 / 05-31 / 06-01 (see `task116.md`), now again from the cost≤1 angle.

## One remaining hypothesis (theoretical only)
The sole escape is an **official-scorer behavior change** — if the scorer ever
stopped counting a `Constant`-node output as memory, or accepted a sparse/`<INT32`
Gather index, sub-30 could open (currently: Const-node idx = 270; sparse idx →
`None`; INT16/INT8/UINT8 idx → `None`). None of these reach ≤1 regardless; the
30-param Gather index is irreducible. No worker action — re-check only if the
scorer contract changes.

**VERDICT: WALL @ cost 30. Nothing to stage (no win, no tie-stage). Perfect-25 family disproved.**
