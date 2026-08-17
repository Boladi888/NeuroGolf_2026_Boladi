# J.18 — Stochastic computing emulation

## Verdict
**Skip.** Stochastic computing's value proposition is "lower hardware area at the cost of variance"; NeuroGolf's exact‑match scoring rule and onnx‑tool's static MAC counting eliminate both halves of that trade in the wrong direction.

## Actionable ONNX design rules — why nothing materialises
- **Stream length vs accuracy:** Vahapoglu & Altun 2018 (arXiv 1805.06262) measure that an SC AND‑gate multiplier needs streams of n ≈ 100 bits for 10 % error and ≈ 1 000 bits for 1 % error; *zero* error requires infinite streams. ARC requires zero error on every cell.
- **Tensor budget:** the working tensor `[1,10,30,30]` has only 10 channels available as a "bit dimension". 10 bits represent at most 1 024 distinct probability levels but with binomial std dev ~ 0.16 of true value — incompatible with the 0/threshold equality test. Using 10 channels as a stream also collides with the channel‑per‑color encoding the brief specifies (12‑color palette across the 10 channels for 0–9 plus padding).
- **MAC accounting:** in onnx‑tool, an `And`/`Or` is one element‑wise op of cost ≈ 1 MAC per element — exactly the same as `Mul`. Therefore replacing a single `Mul` with an SC chain *increases* MACs; it never decreases them.
- **The thresholding rule (`>0`) does not save SC.** The check is `np.array_equal((user_output > 0.0).astype(float), expected)`; this is a binary comparison after thresholding, but the threshold is at the *very end*. Stochastic noise inside the network propagates into the pre‑threshold values; getting all 9 000 cells to land on the correct side of zero with a 10‑bit stream is statistically infeasible (P[any cell wrong] ≫ 1 in realistic noise budgets).

## Literature survey & project‑mapping commentary
- **Liu et al. 2021** (PMC 7924419) and Sim et al. 2021 (PMC 7793640) confirm SC‑CNNs achieve < 1 % accuracy degradation only for *image classification* under top‑1 or top‑5 metrics; they have never targeted exact pixel reconstruction.
- **Biological neural efficiency** (the project lens neighbour): the brain solves tasks like ARC with finite, noisy spikes — but it does not require exact‑match scoring. The analogy breaks at the scoring rule.
- **Sparse coding / sparse distributed memory** tolerate noise by spreading information; same problem — they don't reproduce exact targets.
- **Hash‑based dimensionality reduction** (locality‑sensitive hashing): produces approximate matches, not equalities; not applicable.
- **Where SC *might* be reconsidered:** if NeuroGolf scoring ever permitted `np.allclose` instead of `np.array_equal`, SC would re‑enter the conversation, with stream lengths chosen per‑task to meet tolerance. As specified, do not invest.

## Discrepancies with COVERED list
None. The COVERED list explicitly accepts a "skip" verdict here; this analysis honours that.

## Recommended next experiment
Do not run an SC experiment. Instead, run a 1‑hour falsification test: build a graph that copies an input through a 10‑channel stochastic multiplier `Mul(input_stream, weight_stream)` driven by `RandomUniform`/`Less` (note: `RandomUniform` is allowed, but `(in > rand)` likely conflicts with the static‑shape rule and may be flagged by the banned‑ops list once seeded — confirm before building). Verify in numpy first that even with stream length 1 024, ≥ 1 % of 9 000 cells flip side relative to the deterministic answer; declare topic dead and recover the budget for J.01 and J.14.


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from j18-stochastic-computing.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.18 — Stochastic computing emulation

## Summary

Stochastic Computing (SC), introduced by John von Neumann (1956 lectures, "Probabilistic Logics") and developed by Brian Gaines (1969), represents a real number `p ∈ [0,1]` by a Bernoulli bit-stream of length *n* with `p·n` ones in expectation. An AND gate then implements multiplication of two SC streams (`p·q`); a multiplexer with selector probability 0.5 implements the average `(p+q)/2`. The technique re-emerged for energy-efficient neural networks in 2014–2020 (Yuan, Hwang, Sim, et al.), achieving < 1% accuracy loss on image classification with stream lengths of 128–2048 bits and CMOS area savings of 5–20× over binary multipliers.

## Project mapping — verdict: **skip** (recommended in the brief, confirmed)

Stochastic computing is fundamentally incompatible with NeuroGolf's scoring rule for **two independent reasons**, either of which is sufficient:

1. **Variance vs. exact-match**. The functional check is `np.array_equal((output > 0.0).astype(float), expected_output)`. This is a per-cell, exact, all-or-nothing comparison after thresholding. SC's output is a Bernoulli sample with variance `p(1-p)/n` per cell. To reach a per-cell error rate ε such that the probability of *any* of 9,000 cells being wrong is < 1%, you need ε ≪ 1/9,000, which by Chebyshev requires `n ≫ 9,000² = 81 million bits per cell`. The activation tensor `[1,10,30,30]` has total capacity 9,000 float32 values; there is no shape-compatible way to store streams that long.

2. **No MAC-count benefit in the cost model**. SC's value proposition is hardware area: an AND gate is ~1/100 the transistor count of a binary multiplier. In `onnx-tool 1.0.1`'s scoring, `Mul` and `And`/`Or` would both count as `ADD_MACS=1` per element (verified: `MUL_MACS=1, ADD_MACS=1, CMP_MACS=1` at the top of `node.py`). Replacing one Mul with an AND-stream of length *n* multiplies the macs count by *n*. Even at the most aggressive *n=8* this is 8× more expensive than the Mul it replaces.

Empirical reinforcement: I verified `onnx-tool 1.0.1` has registered handlers for `And`, `Or`, `Xor`, `Not` (these are bool-type Boolean ops, distinct from the `Bitwise*` integer ops which are unhandled, see J.01). They count as 1 op/element via `NpMathBase`-style accounting. So even if you accept the variance, the MAC count goes up.

**Tensor-shape constraint, in detail.** The `[1,10,30,30]` activation has 9,000 cells × 4 bytes = 36,000 B. To represent each cell's value as a stream of length *n*, you need either:

- An additional dimension of size *n* — but the tensor shape is hard-coded `[1,10,30,30]`. No room.
- Repurposing the channel dimension (size 10) as bits — gives 10-bit streams, variance ~0.025, P[any of 9,000 cells wrong] ≈ 1.0. Useless.
- Repurposing the spatial dimension (30 or 900) — destroys the actual ARC grid layout.

**Where SC is *correctly* recommended for skip.** The brief's pre-judgment ("likely too lossy for 30×30 ARC; failure modes — accept 'skip' if MAC savings don't materialize") is exactly right. Both halves of the analysis confirm: variance is fatal, and MAC savings are negative.

**Adjacent techniques worth considering instead** (project lens):
- **Sparse coding** — represent sparse-grid inputs with very few active cells; not directly Bernoulli but uses a similar "few non-zero positions encode the answer" idea. Maps to `Where` (`since_version=16`, supports float) for indicator-mask routing.
- **Bit-stream convolution via depthwise Conv1D** — if you ever wanted multi-bit-precision arithmetic, depthwise Conv along a synthetic time axis is far cheaper in onnx-tool than SC because it batches MACs.
- **Hybrid stochastic-binary networks** (Sim et al., NIPS 2017) — keep binary precision in critical layers, use SC only where variance is OK. ARC has no "where variance is OK" because of exact match.

## Sources

- **Original SC theory** — von Neumann, "Probabilistic Logics and the Synthesis of Reliable Organisms from Unreliable Components" (1956). B.R. Gaines, "Stochastic Computing", AFIPS 1967, `https://doi.org/10.1145/1465482.1465508`.
- **SC for CNNs** — Liu, Han, et al., "A Survey of Stochastic Computing Neural Networks" (2021), `https://pmc.ncbi.nlm.nih.gov/articles/PMC7924419/`. Sim & Lee, "Cost-Effective Stochastic MAC Circuits for Deep Neural Networks" (2017), reports 1% accuracy loss on CIFAR with n=1024.
- **Hybrid SC-binary networks** — Lee, Wang, Lin, "Energy-Efficient Hybrid Stochastic-Binary Neural Networks for Near-Sensor Computing", arXiv:1706.02344, NIPS 2017.
- **From-stochastic-to-bit-stream comparison of arithmetic accuracy** — Vahapoglu & Altun, "From Stochastic to Bit Stream Computing", arXiv:1805.06262, reports `error ≈ 1/sqrt(n)` for AND-gate multiplier, confirming the n ≫ 81M lower bound for our use case.
- **ONNX bool ops** — verified on `onnx==1.21.0`: `And/Or/Xor` `since_version=7`, `Not` `since_version=1`, all bool-only. `Less/Greater` `since_version=13`, `Equal` `since_version=19`. `RandomUniform/RandomUniformLike` `since_version=22`, only float types.
- **onnx-tool 1.0.1 cost constants** — `/usr/local/lib/python3.12/dist-packages/onnx_tool/node.py` lines 1–80: `ADD_MACS=1, MUL_MACS=1, CMP_MACS=1, DIV_MACS=4, EXP_MACS=32`. Confirms there is no operator-class discount for AND vs Mul.
- **Empirical**: I did not run a full SC ONNX graph because the verdict is determinable from first principles (MAC count goes up, variance is non-zero) without measurement. If desired, a 1-hour falsification probe is described in the next section.

## Contradictions / corrections

- The COVERED list does not contain SC-specific facts. No contradictions.
- **Subtle**: the COVERED line "Incorrect = literal 0 — empirically confirmed" is the dispositive fact for J.18. It means there is no graceful degradation; one bit wrong = entire task scores 0, regardless of how close the rest of the output is to correct. This single line refutes any approximate-arithmetic technique.

## Confidence

**High** for the negative verdict. The MAC-count argument (replacing one Mul with *n* ANDs costs *n*× more) is independent of any ARC-task taxonomy. The variance argument (n ≫ 81M for 9,000 cells) is independent of the network architecture. Both arguments converge on "do not use SC".

**Recommended falsification probe (1 hour, optional):** Build a numpy-only simulation: pick stream length n ∈ {64, 256, 1024, 4096}; encode a known 30×30 grid as Bernoulli streams; apply an AND-gate multiply with another encoded grid; threshold and count mismatches against the deterministic product. Confirm that at every reasonable n, at least one of the 9,000 cells flips. Conclude topic dead and reallocate budget to J.14 (Morton/Hilbert).
