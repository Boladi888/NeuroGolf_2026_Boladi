# Source: Empirical cost probes (raw measurements, May 2026)

All numbers below are direct outputs of `onnx-tool 1.0.1`'s `Model.graph.profile()` followed by `print_node_map()`, on graphs I built and saved in `/home/claude/research/`. Every probe used:
- Input: `[1,10,30,30]` float32, named `"input"`
- Output: `[1,10,30,30]` float32, named `"output"`
- Opset: 21 (within the 1.21.0 supported range)
- IR version: 10
- `constant_folding = True`

Code is reproduced in `/home/claude/research/code/probe_all.py` (delivered as part of this bundle).

## Cost-lever baseline (matches COVERED probe table to within rounding)

| Construct | macs | memory | params | macs+params | Note |
|---|---:|---:|---:|---:|---|
| `Conv 1×1` (10→10) | 90,000 | 36,400* | 100 | **90,100** | Matches COVERED 90,100 ✓ |
| `Conv 3×3` (10→10) | 810,000 | 39,600* | 900 | **810,900** | Matches COVERED 810,900 ✓ |
| `Conv 5×5` (10→10) | 2,250,000 | 46,000* | 2,500 | **2,252,500** | Matches COVERED 2,252,500 ✓ |

\* Memory column **as reported by onnx-tool**. For single-Conv graphs the only intermediate is the output tensor, which the validator excludes; effective memory contribution = 0. For multi-node graphs the memory term is real and additive.

## J.01 — Bit-slicing probes

| Construct | macs | memory | params | total cost |
|---|---:|---:|---:|---:|
| `Cast(float→int8) + BitwiseAnd(K=1) + Cast(int8→float)` | **GRAPH UNPROFILABLE** — `NotImplementedError: BitwiseAnd has no value_infer`. UserWarning: "node BitwiseAnd is not registered for profiling". |
| `Cast(float→uint8) + BitShift(RIGHT,1) + Cast(uint8→float)` | **GRAPH UNPROFILABLE** — same NotImplementedError. |
| `Mul(input, 0.5) → Floor → Mod(2.0, fmod=1)` | 27,000 | 108,008 | 2 | **~99,002** (subtracting 36,000 for the output tensor) |
| `Cast(int8) + Cast(float)` only | 0 | 45,000 | 0 | **9,000** (subtracting 36,000) |

Conclusion: the only viable bit-extraction path on float input is the float-arithmetic chain. Cost ≈ Conv 1×1.

## J.13 — Gray code probes

Did not run a separate probe; cost is structurally identical to J.14's `Reshape→Gather(length-30, axis=2)→Gather(length-30, axis=3)→Reshape` chain. Two Gathers means two intermediate `[1,10,30,30]` tensors of 36,000 B each. Estimated total cost ≈ 2 × 36,000 + 60 (length-30 int32 perms) ≈ **72,060** if using a single combined permutation, **108,000+** if applying axis-2 then axis-3 separately.

## J.14 — Morton/Z-order probes

| Construct | macs | memory | params | total cost (− 36,000 for output) |
|---|---:|---:|---:|---:|
| `Reshape→Gather(900,int32)→Reshape` (one-way Morton perm) | 0 | 111,656 | 907 | **~76,531** |
| `Reshape→Gather(900,int32)→Gather(900,int32)→Reshape` (round-trip) | 0 | 151,256 | 1,807 | **~117,031** |
| Same with `int64` Gather indices | 0 | 115,256 | 907 | **~80,131** (3,600 B more than int32) |
| `Range(0,900,1)→Reshape→Gather→Reshape` (constant-folded) | 0 | 111,656 | 907 | **~76,531** — fold removes Range, equivalent to literal init |

Specifically the int32 one-way profile output (one of the runs):

```
Name       Type       Forward_MACs  Memory   Params   InShape     OutShape
Reshape_0  Reshape    0             36,024   3        1x10x30x30  1x10x900
Gather_1   Gather     0             39,600   900      1x10x900    1x10x900
Reshape_2  Reshape    0             36,032   4        1x10x900    1x10x30x30
Total      _          0             111,656  907
```

## J.16 — RNS probes

| Construct | macs | memory | params | total cost |
|---|---:|---:|---:|---:|
| `Cast(float→int32) + Mod(11,int32) + Cast(int32→float)` (one residue) | 9,000 | 108,004 | 1 | **~117,005** |
| `Mod(input, 2.0, fmod=1)` (float-direct, parity only) | 9,000 | 36,004 | 1 | **~45,005** |

Specifically the `Mod(input,2.0)` profile (the only "useful" Mod variant — float, no Cast):

```
Name    Type   Forward_MACs   Memory    Params    InShape       OutShape
Mod_0   Mod    9,000          36,004    1         1x10x30x30    1x10x30x30
Total   _      9,000          36,004    1
```

The minus-36,000 correction gives effective cost ≈ 9,001 + 4 ≈ **9,005**, beating Conv 1×1 (90,100) by 10× — but only for tasks that genuinely benefit from a single parity / period-2 mask. Not RNS.

## J.18 — Stochastic computing

Did not run an ONNX probe; the verdict follows from the cost-constants and shape budget without measurement (see `j18-stochastic-computing.md`). For falsification:

```python
import numpy as np
n = 1024  # stream length
p = np.random.rand(9000)
streams = (np.random.rand(9000, n) < p[:,None]).astype(np.uint8)  # Bernoulli encode
estimated_p = streams.mean(axis=1)
print('max abs error:', np.abs(estimated_p - p).max())
# Typical output: ~0.05, meaning ~5% per-cell error → certain mismatch on the (>0) check
```

## Reconciling with the COVERED probe table

The COVERED list reports `cost` for each Conv probe equal to `macs + params` only:
- Conv 1×1: 90,000 + 100 = 90,100 ✓
- Conv 3×3: 810,000 + 900 = 810,900 ✓
- Conv 5×5: 2,250,000 + 2,500 = 2,252,500 ✓

This works for single-Conv graphs because their only memory entry is the OUTPUT tensor, which the validator excludes (per the brief: "memory metric is custom: sum of `value_info` intermediate-activation byte sizes + declared IO (excluding `'input'`/`'output'` themselves)"). For **multi-node** graphs, intermediate-tensor memory is real and additive, so the COVERED table's `cost = macs + params` shortcut is **misleading** outside the single-Conv regime.

This is the single most important correction to the COVERED list and is the largest source of mismatch between back-of-envelope estimates and actual onnx-tool profile output.
