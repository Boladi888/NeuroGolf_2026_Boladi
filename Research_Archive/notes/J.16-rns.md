# J.16 — Residue Number System (RNS)

## Verdict
**Skip.** RNS's wins are *hardware bit‑width* wins; onnx‑tool measures MACs in float32 regardless. For ARC's tiny dynamic range (colors 0–9, max counts 900) RNS only multiplies the operator count without buying any reduction in the cost formula.

## Actionable ONNX design rules
- **Op availability:**
  - `Mod` exists since opset 10 (updated 13), supports float32 with `fmod=1`. So an RNS chain `r_i = x mod m_i` is implementable directly without Cast.
- **Cost of one residue extraction on `[1,10,30,30]` float32:** `Mod(input, m_i)` ≈ 9 000 element‑wise ops + 36 000 B activation. Compare with Conv 1×1 = 90 100. So **one** Mod ≈ 0.4× Conv 1×1; **three** Mods (a triple {7,11,13}) ≈ 1.2× Conv 1×1, and you still owe a (non‑trivial) reconstruction step before the threshold.
- **Reconstruction:** Chinese Remainder Theorem (CRT) requires a multiply‑add per modulus *per element* plus a final Mod by ∏mᵢ. For three moduli on 9 000 elements: ≥ 4 × 9 000 = 36 000 MACs and ≥ 3 × 36 000 = 108 000 B intermediate memory. **Cumulative RNS round‑trip on the activation tensor is ≈ 1.5× Conv 1×1 with no functional benefit on ARC's range.**
- **Range coverage:**
  - Color values 0–9: any single modulus ≥ 11 is exact (e.g., m=11). RNS isn't needed.
  - Cell counts on 30×30: max 900 = 5²·6². Triple {7,11,13} gives range 1001 ≥ 900; pair {31,29} gives 899 < 900 (fails); so **{7,11,13} is the minimal viable triple** if anyone needed it — but the ARC color range never makes this binding.
- **No ARC family materially benefits.** Counting tasks ("how many objects?") top out at counts < 100; a single int8 (if we ever cast that way) covers them.

## Literature survey & project‑mapping commentary
- **Demirkiran et al. 2023** (arXiv 2306.09481): RNS lets analog DNNs use 6‑bit ADCs while keeping FP32 accuracy. The win is *physical precision per bit of converter*, not algorithmic op count. NeuroGolf has no analogue of bit‑width.
- **RNSnet (UCSD 2018):** 35.4× speedup, 145.5× energy efficiency vs GPU — these are wall‑clock and energy, not ONNX MAC counts.
- **Valueva et al. 2020:** 7.86–37.78 % HW resource reduction for RNS Conv layers in FPGA — same caveat.
- **Symbolic regression** is a stronger neighbour: when an ARC task admits a closed‑form arithmetic rule, symbolic regression discovers it; you then encode the formula as a few Conv 1×1 / Add / Mul nodes. RNS adds zero discovery power.
- **Crystallographic groups / sparse coding / lossless compression** offer better dynamic‑range parsimony than RNS for small ranges (e.g., a single int4 channel covers ARC's 12‑value palette including 10 and 11).

## Discrepancies with COVERED list
None. The COVERED list reminds us that RNS likely has small benefit; this analysis confirms ≈ zero benefit and recommends skipping.

## Recommended next experiment
Skip the experiment; burn the budget on J.01 and J.14 instead. If sanity‑check is required, build `Mod(input, 11.0, fmod=1)` and confirm onnx_tool charges ≥ 9 000 MACs and ≥ 36 000 B activation memory — establishing that one Mod alone exceeds 0.4× Conv 1×1.


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from j16-rns.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.16 — Residue Number System (RNS)

## Summary

The Residue Number System represents an integer *r* by its tuple of residues `(r mod m₁, r mod m₂, ..., r mod mₖ)` for pairwise-coprime moduli; addition and multiplication factor through the channels independently, so the underlying multipliers and adders need only ⌈log₂(max mᵢ)⌉ bits each instead of ⌈log₂(∏mᵢ)⌉ bits. Reconstruction of *r* uses the Chinese Remainder Theorem (CRT). The technique was developed by Svoboda and Valach in the 1950s, brought into hardware DSP in the 1970s, and revived for neural networks in 2018+ for area/energy efficiency on FPGAs and analog accelerators (Demirkiran 2023, Valueva 2020, Salamat "RNSnet" 2018).

## Project mapping — verdict: **skip**

RNS is a hardware bit-width optimization. NeuroGolf measures cost as `macs + memory + params` where `macs` is element-op count regardless of integer width. **There is no analogue to bit-width savings in the scoring metric.** Splitting one float32 multiply into three modular multiplies *triples* the macs count without buying anything back.

Empirical confirmation from my probes against `onnx-tool 1.0.1` on `[1,10,30,30]`:

| Construct | macs | memory (B) | params | total cost | pts |
|---|---:|---:|---:|---:|---:|
| `Cast(int32) → Mod(11) → Cast(float)` (one residue extraction) | 9,000 | 108,004 | 1 | **~117,005** | 13.330 |
| Conv 1×1 baseline | 90,000 | 0 | 100 | **90,100** | 13.591 |

A *single* residue extraction already costs **30% more than a full Conv 1×1**. Three residues `{7, 11, 13}` plus a CRT reconstruction would total ≥ 4×9,000 macs + 5+ intermediate float32 tensors at 36,000 B each = at least **216,000 macs + 180,000 B memory ≈ 396,000 cost**, ~4.4× worse than Conv 1×1. There is no ARC task family where this trade goes in our favor.

**Why the technique exists but doesn't help here:**

- Color values 0–9 fit in 4 bits; pad-color 10 and decode-error 11 fit in another 1 bit. A single int8 (8 bits) covers the entire palette losslessly with no need to decompose. ARC's "dynamic range" is tiny.
- Cell counts on a 30×30 grid max out at 900 (= ~10 bits). Every reachable count fits in int16 with massive headroom; counts of distinct objects per task rarely exceed 30. RNS-style overflow protection has no use case here.
- Position arithmetic (e.g., "draw a line from (a,b) to (c,d)") needs values ≤ 30. int8 is fine.
- The CRT reconstruction itself requires multiplications by moduli-specific Bezout coefficients — these are float32 muls, fully counted by onnx-tool.

**One valid use of `Mod` that is *not* RNS but worth noting**: `Mod` with a small constant divisor (like 2 or 3) is a cheap way to derive a parity / period mask. Cost ~9,000 macs + 36,000 B memory + 1 param ≈ **45,001** for a one-shot "mark cells where (something) is odd". This is < Conv 1×1 by ~50% and is the only place `Mod` justifies its inclusion in the model. But that's J.01 territory (parity / bit extraction), not RNS proper.

**Failure modes specific to RNS attempts:**

1. The `Mod` op with float input requires `fmod=1` (numpy.fmod semantics, sign of dividend) per the ONNX spec; using `fmod=0` with float will fail validation.
2. Casting to int32 to use Python-style `mod` and back to float costs **two extra Casts** at ~9,000 B memory each, on top of the Mod itself.
3. CRT reconstruction's Bezout coefficients depend on the specific moduli and must be hard-coded as float32 initializers; they're not mathematically clean small numbers.

## Sources

- **Demirkiran et al., "Leveraging Residue Number System for Designing High-Precision Analog DNN Accelerators"** — `https://arxiv.org/pdf/2306.09481`. Quote (page 1): "RNS allows us to use 6-bit ADCs while maintaining FP32 precision." Demonstrates that the win is in *physical converter resolution*, not in op count.
- **Valueva et al., "Application of the residue number system to reduce hardware costs of the convolutional neural network"** — `https://www.sciencedirect.com/science/article/abs/pii/S0378475420301580`. Reports 7.86–37.78% hardware area reduction for INT8 RNS Conv vs binary Conv. Hardware area, not algorithmic op count.
- **Salamat et al. "RNSnet: In-Memory Neural Network Acceleration Using Residue Number System"** — ICRC 2018, `https://acsweb.ucsd.edu/~sag076/papers/icrc18_rnsnet.pdf`. Reports 35× speedup, 145× energy efficiency vs GPU. Speed and energy, not MAC counts.
- **CRT reconstruction & Bezout coefficient construction** — Knuth *TAOCP Vol 2 §4.3.2*, formula (24).
- **ONNX `Mod` spec** — `since_version=10`, updated to opset 13, supports float (with `fmod=1` mandatory), domain `""`. `https://onnx.ai/onnx/operators/onnx__Mod.html`.
- **onnx-tool 1.0.1 source** — `/usr/local/lib/python3.12/dist-packages/onnx_tool/node.py` line 332: `class ModNode(NpMathBase): self.op_mac = ADD_MACS` (1 op/element). Confirms 9,000 macs for our `[1,10,30,30]` tensor.
- **My probe** — `Cast(int32) → Mod(11) → Cast(float)` measured cost 9,000 macs + 108,004 B memory + 1 param. See `/home/claude/research/code/probe_j16.py` in delivery.

## Contradictions / corrections

- The COVERED list does not contain RNS-specific facts. No contradictions.
- **Subtle correction**: per the brief's text "ONNX `Mod` op availability/cost", I confirm `Mod` is available, but its cost is **not** zero — it counts as 1 op/element via `NpMathBase`. A common mistake is to think Mod is "just a fancy reshape"; it is not.

## Confidence

**High**. The negative verdict follows mechanically from onnx-tool's MAC accounting (ADD_MACS=MUL_MACS=1 regardless of operand width). I cannot construct any RNS variant that costs less than a Conv 1×1, and the Conv 1×1 itself is dominated by Morton-style Gather permutations (J.14) for nearly every ARC task family.
