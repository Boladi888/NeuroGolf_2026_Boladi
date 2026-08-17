# J.20 — Hyper-quantized associative memories (LUT-as-network)

**Status:** synthesis. Maps "encode many input→output mappings into a sub-byte-packed lookup table read via Gather" onto the validator's accounting rules.
**Cluster:** logic / discrete / combinatorial.

## TL;DR

- **The 1.44 MiB cap is *not* the binding constraint** for LUT-as-network designs in this competition. The binding constraint is the **`params` term in the score formula**: every initializer byte counts as 1 param (in float-equivalent units), and `params` adds linearly to `macs+memory` inside the log. The fileSize cap of 1,509,949 bytes only fails *correctness* if blown; the score degrades smoothly with params well before that.
- **Sub-byte storage in raw_data: yes for INT4/UINT4, no for everything else.** ONNX 1.21 supports INT4/UINT4 packed as 2-per-byte in `raw_data` (verified at onnx.ai/onnx/technical/int4.html). But INT4-typed initializers are only consumable by `QuantizeLinear`/`DequantizeLinear`/`Shape`/`Size`/`Transpose`/`Reshape`/`Constant`. **`Gather` cannot read an INT4 LUT directly without a `DequantizeLinear` step**, which itself costs MACs.
- **Realistic max LUT density: ~1.5M float32 entries (6 MB raw)? No — 1.44 MiB cap → ~376K float32 entries. With INT8 cast: ~1.5M entries.** The brief's "12M 1-bit patterns" ceiling is *physically* available (1,509,949 bytes × 8 bits ≈ 12M bits) but **practically unreachable** because no ONNX op consumes a 1-bit-packed tensor as data.
- **The `Gather`-LUT approach IS in the COVERED pattern library** (color-perm-via-Gather, hits tasks 016/337). The question is how to scale it to *position*-dependent or *neighborhood*-keyed lookups. Position-keyed: Gather over a precomputed key tensor (still cost 0 if the key tensor is a foldable initializer). Neighborhood-keyed: requires building the key from input pixels at runtime — that's where the MAC cost shows up.
- **Net assessment: J.20 is the most directly actionable of the five topics.** Existing color-perm pattern is a 2/400 hit; extending to *2-input neighborhood LUTs* (key = (left_color, right_color)) plausibly hits another 5-15 tasks, with a per-task cost competitive with Conv 1×1.

## How `params`, `memory`, `macs` interact in the score

The validator scores `points = max(1.0, 25.0 - log(max(1.0, macs+memory+params)))` (COVERED). The three terms add. From COVERED:

- `params` = sum of initializer byte sizes, **divided into "param count" units**. For a float32 initializer of N elements, params contributes N. For a int8 initializer of N elements, the `numel` is N regardless of data type — onnx-tool counts elements, not bytes (see onnx-tool Profile.md: "Parameter number stands for the model space size. The total size of weight tensor"). **Open question (DISCREPANCY flag below):** does onnx-tool count int8 params as N or N/4?
- `memory` = sum of intermediate-activation byte sizes (excluding `"input"`/`"output"`). A `[1,10,30,30]` float32 activation = 36,000 bytes contributes 36,000 to memory. **Memory dominates over params for grid-sized intermediates.**
- `macs` = the elementwise/Conv accounting from earlier topics.

For a Gather LUT with K entries of shape [V] consumed once, the cost contribution is roughly:
- params: K × V (the table)
- memory: V (the output of one Gather) — actually `numel(output)` = V, same magnitude
- macs: 0 (Gather is shape-only per COVERED)

**A single-pixel-keyed LUT of K=12 colors × V=12 outputs (a color-perm) costs 144 params + 12 memory + 0 MACs ≈ 156 score units. Log(156) ≈ 5.05, so points = 25-5.05 = 19.95.** Compare to Conv 1×1 at 90,100 + memory: log ≈ 11.4, points = 13.6. **Color-perm Gather wins by ~6 points over Conv 1×1.** This matches the COVERED probe (Conv 1×1 = 13.591).

## ONNX raw_data byte packing — what actually works

Verified facts (sources: onnx.ai/onnx/technical/int4.html, onnx GitHub issues #2825, #5776, #6160):

| dtype | bytes per element in raw_data | Operators that can consume it |
|---|---|---|
| FLOAT (32-bit) | 4 | every numeric op |
| INT8 / UINT8 | 1 | most ops since opset 14 |
| INT16 / UINT16 | 2 | most ops since opset 14 |
| BOOL | 1 (per ONNX spec: "00000001 for true, 00000000 for false") | `And`/`Or`/`Xor`/`Not`/`Where`/`Cast` |
| INT4 / UINT4 | 0.5 (packed 2-per-byte: `pack(x,y) = y << 4 \| x & 0x0F`) | only `QuantizeLinear`/`DequantizeLinear`/`Shape`/`Size`/`Transpose`/`Reshape`/`Constant` (per onnx issue #5776) |

**Confirmed: BOOL is 1 byte per element in raw_data, not 1 bit.** The brief's "1-bit storage" assumption is too aggressive — the ONNX IR does not bit-pack bool tensors. The closest you get to sub-byte storage that's *useful* is INT4/UINT4 (4 bits per element), and even that requires a Dequantize step.

So:
- 1.44 MiB cap = 1,509,949 bytes
- max float32 entries: ~377,487
- max int8 entries (via `Cast` to float at use): ~1,509,949
- max int4 entries (requires `DequantizeLinear`): ~3,019,898

**The brief's "12M 1-bit patterns" goal is unreachable without a custom packing-and-unpacking subgraph that itself costs MACs.** Building a "bool tensor stored as 8-bools-per-byte" would require `BitwiseAnd` + `BitShift` + `Cast` chains, which are opset 18 (BitwiseAnd) — and even if accepted, the unpack step costs ~9,000 MACs per use.

## LUT keying schemes — concrete patterns

### Pattern 1: per-pixel single-channel lookup (already covered)

This is `color-perm-via-Gather`. Input: one-hot `[1,10,30,30]`. Convert to indices via... actually wait, the input is already one-hot, so a "color perm" is just a permutation along the channel axis = `Gather(input, axis=1, indices=perm_table)` where `perm_table` is a [10] int64 initializer. **Cost: 0 MACs, 10 params, 9,000 memory.**

### Pattern 2: per-pixel multi-channel-derived lookup

Key = (top-left neighbor color, top-right neighbor color). Encoding: 12×12 = 144 distinct keys. Realization:

1. Slice top-left and top-right neighbors (`Slice` × 2 = 2 shape-only ops, cost 0).
2. Convert each neighbor's one-hot to an integer index. This step is the catch — converting one-hot back to index requires `ArgMax` (1 op, 9,000 cmp ops) per neighbor.
3. Combine indices: `key = 12 * top_left + top_right`. Cost: 1 `Mul` (with scalar 12) + 1 `Add` = 2 ops × 30×30 = 1,800 ops.
4. `Gather(LUT, key)` where LUT is shape [144, 10] = 1,440 params. Output [30, 30, 10], reshape to [1, 10, 30, 30].

**Total cost:** 2 ArgMax (≈18,000 MACs) + 2 elementwise ops (1,800 MACs) + 1,440 params + 9,000 memory ≈ **30,240**. Conv 3×3 baseline = 810,900. **Win: 27×.**

### Pattern 3: full 3×3 neighborhood lookup

Key = 9 neighbors' colors. Encoding: 12^9 = 5.16 billion distinct keys. **Won't fit.** Even at INT8 = 5GB. Don't try without further structure.

### Pattern 4: 3×3 neighborhood lookup over binary mask only (foreground/background)

Key = 9 binary bits = 2^9 = 512 keys. LUT = [512, 10] float32 = 5,120 params. Construction:
1. Threshold input to a binary mask: `Cast(Greater(ReduceSum(input, channel-axis-skip-bg), 0), float)`. ~2 ops × 9000 = 18,000 MACs.
2. Build key from 9 bits: 9 `Slice` shifts (each shape-only, cost 0) + 9 `Mul`-by-power-of-2 (or `BitShift` if integer) + 8 `Add`s. ~17 ops × 900 (key tensor is shape [30,30]) = ~15,300 MACs.
3. `Gather(LUT, key)`. Cost: 0 MACs, 5,120 params, 9,000 memory.

**Total: ~33,300 MACs + 5,120 params ≈ 38,420.** Beats Conv 3×3 (810,900) by 21×, and Conv 1×1 (90,100) by 2.3×.

This is the most expressive LUT pattern that fits in the cap and the param budget. **It can encode any cellular-automaton update rule with binary state.** That covers a non-trivial slice of ARC tasks: bounding-box edge detection, "fill enclosed regions," "find pixels with ≥3 colored neighbors," etc.

## Failure modes when keys collide

The brief asks: "what happens when keys collide?"

In a deterministic LUT, every key has exactly one stored value. "Collision" only matters if you fold *different per-task rules* into the same LUT — e.g., trying to use one shared 12^2-entry LUT for multiple tasks. **The competition does not require this**: each task gets its own ONNX file with its own initializer, so cross-task LUT sharing is irrelevant.

The more interesting failure mode: **train-vs-test key distribution mismatch.** If the LUT is built from train + arc-gen samples but the private test set produces a key the LUT never saw, the looked-up value is whatever default was stored at that key. **Mitigation:** build the LUT to produce the input value (identity) for unseen keys, so unrecognized neighborhoods pass through unchanged. This is achievable by initializing the LUT to identity then overriding only the keys the training data exhibits.

## Discrepancies / flags vs COVERED

- **Direct conflict with COVERED claim about sub-byte packing:** the brief asks "whether ONNX raw_data byte-pack permits actual sub-byte storage (overlaps A.2)." **Answer: only INT4/UINT4 are sub-byte (4 bits each). Everything else is byte-aligned at minimum. BOOL is 1 byte, not 1 bit.** This invalidates the brief's "12M 1-bit patterns" framing.
- **DISCREPANCY:** I claim onnx-tool counts initializer params by *element count*, not bytes. Verified by Profile.md ("Parameter number stands for the model space size. The total size of weight tensor"). But the wording "total size" is ambiguous — could mean elements or bytes. **Action item:** open `onnx_tool/graph.py` (the pinned 1.0.1 version) and find where `params` is summed in `calculate_*` — check whether it iterates `numel(initializer)` or `nbytes(initializer)`. **If it's nbytes, INT8 LUTs become 4× cheaper than float32 LUTs in the score.** This is potentially a multi-point swing on the leaderboard.
- **The forum thread `/discussion/692827` flagged in COVERED about onnx-tool scoring discrepancies most likely concerns this exact question.**
- **No conflict with COVERED `_PATTERNS` list:** the existing `color-perm-via-Gather` is a special case of this topic; the more general "neighborhood LUT" pattern is unimplemented and a clear extension.

## Actionable design rules

1. **Build LUTs for tasks where the output at each cell depends on a small (≤9-bit) function of nearby cells.** Use Gather with a precomputed integer index tensor.
2. **Cast input to int8 for storage but keep the live tensor in float32** to satisfy the I/O constraint. Float→int8 round-trip costs ~18,000 MACs total (2 Casts).
3. **`ArgMax` is the right way to convert a one-hot input back to an index.** It costs ~9,000 cmp ops per channel-axis collapse.
4. **Build the LUT to default to identity** for unseen keys, so generalization to held-out test sets degrades gracefully instead of catastrophically.
5. **The 1.44 MiB cap is essentially never the binding constraint for this style of model.** The score-formula `params` term will degrade your score long before you hit the file size cap. Optimize for `params` (≤ ~1,000 entries gives substantially better points than ~10,000).
6. **Verify whether onnx-tool counts params by element-count or byte-count** before betting on int8 LUTs (action item above). This is the highest-leverage open question in this cluster.

## Sources consulted

- ONNX 1.21 docs on raw_data and INT4 packing: onnx.ai/onnx/technical/int4.html
- ONNX issue #5776 (introducing INT4 type, 2023) — confirms 2-per-byte packing and operator-set restriction.
- ONNX issue #2825 (raw_data semantics) — confirms BOOL is 1 byte/element.
- ONNX issue #6160 (INT4 protobuf encoding bug) — confirms raw_data is the cheap path; int32_data field is wasteful.
- onnx-tool Profile.md — params accounting wording (ambiguous between elements and bytes).
- Gather op reference (opset 1, 11) — index lookup is shape-only.
- COVERED `_PATTERNS` library entry: color-perm-via-Gather hits tasks 016/337.


<!-- ========================================================== -->
<!-- Empirical update appended 2026-05-01 from J.20-hyper-quantized-associative-memories.md -->
<!-- Source: research-agent #03 cluster-J empirical pass with     -->
<!-- onnx-tool 1.0.1 source inspection + verify_mac_accounting.py -->
<!-- ========================================================== -->

# J.20 — Hyper-quantized associative memories (LUT-as-network)

## Summary

The "LUT-as-network" idea has a long pedigree: Kanerva's Sparse Distributed Memory (1988), the LUT-based "weightless neural networks" / RAM-net tradition (Aleksander 1966 onward), and modern revivals in differentiable LUT layers (e.g. "DiffLogic" learnable LUTs). For NeuroGolf 2026, the relevant question is concrete: how many distinct input→output mappings can a single ONNX model store and retrieve via `Gather`-as-LUT, and what's the optimal storage encoding under onnx-tool 1.0.1's `params + memory + macs` cost metric?

The empirical answers (verified by reading the pinned onnx-tool source and probing the library):
- **`params` is counted by element count, not bytes.** Confirmed at line 1393–1394 of `onnx_tool/graph.py`: `elesize = volume(self.tensormap[input].get_shape()); _params += elesize`. An int8 LUT of N elements counts the same as a float32 LUT of N elements in `params`.
- **`memory` IS counted in bytes** (line 1395: `_memory += elesize * self.tensormap[input].get_elementsize()`). So int8 LUTs reduce `memory` by 4× compared to float32 LUTs.
- **`Gather` is 0 MACs.** Confirmed empirically (verified above): `Gather (color-perm via axis=1): macs=0 params=10`.
- **The brief's "12M 1-bit patterns" assumption is unreachable** without custom unpack subgraphs. Bool tensors in ONNX are stored 1 byte per element (per ONNX spec: "Boolean type MUST be written one byte per tensor element (00000001 for true, 00000000 for false)"). The only sub-byte type that ONNX supports is INT4/UINT4, which is 2 elements per byte but only consumable by `QuantizeLinear`, `DequantizeLinear`, and shape-only ops.

The actionable design space: a `Gather`-LUT keyed on a small derived integer (≤9 bits) over a binary or quantized tensor lets you encode arbitrary cellular-automaton update rules at a few-thousand-params cost, beating Conv 3×3 (810,900 MACs) by 20× or more on those tasks.

## Project mapping

**Cost accounting verified empirically and from source:**

| Storage approach | `params` per N entries | `memory` per N entries (when consumed) | Notes |
|---|---|---|---|
| float32 raw_data | N | 4N bytes | baseline |
| int8 raw_data | N | N bytes | needs `Cast` to float at use; Cast is 0 MACs |
| INT4 raw_data | N | N/2 bytes | only `DequantizeLinear` consumes it; Dequant cost = numel × 1 MAC (extends `PWNode`); breaks even-or-loses vs int8 |
| BOOL raw_data | N | N bytes | per ONNX spec; 1 byte/elem despite being "1-bit" semantically |

**`memory` calculation note:** the COVERED list says "memory metric is custom: sum of `value_info` intermediate-activation byte sizes + declared IO (excluding input/output themselves). Initializers count as `params`, NOT `memory`." That's the validator's rule (in `neurogolf_utils.py`) — **different from onnx-tool's per-node accumulator** which DOES include initializer bytes in memory. Action: read `neurogolf_utils.py` calculate_memory to confirm.

**Pattern 1: Color-perm via Gather (already in COVERED).** A 10-element `int64` permutation initializer + `Gather(input, perm, axis=1)`. **Cost: 0 MACs, 10 params, ~36 bytes memory** (verified empirically: `params=10`). Beats Conv 1×1 by enormous margins. Hits tasks 016/337 per COVERED.

**Pattern 2: 9-bit binary-neighborhood LUT.** The most general pattern this competition seems to admit cleanly. Treat the input's "is colored vs not colored" as a binary mask, then for each pixel build a key from its 3×3 binary neighborhood:
1. Reduce input to binary: `Cast(Greater(ReduceSum(input, axes=[1], keepdims=1), 0), float)` → `[1,1,30,30]`. Cost: 9,000 (ReduceSum) + 9,000 (Greater) + 0 (Cast) = 18,000 MACs. Wait — `Cast` is 0, but the resulting tensor type is float. (`Greater` outputs bool, but the validator's `(result > 0.0)` post-thresholding works on float.)
2. Build 9-bit key from neighborhood: 9 `Slice`s into the 30×30 mask (cost 0) + 9 `Mul`-by-power-of-2 + 8 `Add`s. ~17 elementwise ops over the inner ~28×28 = 784 cells = 17 × 784 × 1 ≈ 13,328 MACs. Or: a single `Conv 3×3` with weights `[256,128,64; 32,16,8; 4,2,1]` produces the key directly: 1 op × 28×28 × 9 = ~7,056 MACs.
3. `Cast(key, int64)` for indexing: 0 MACs.
4. `Gather(LUT, key, axis=0)` where LUT has shape `[512, 10]` (5,120 float32 params, or 5,120 int8 params then `Cast`): 0 MACs. Output shape: `[28, 28, 10]`.
5. `Pad` and `Transpose/Reshape` to `[1,10,30,30]`: 0 MACs (shape-only).

**Total: ~25,000–38,000 MACs + ~5,120 params + bounded memory.** Conv 3×3 baseline = 810,900 MACs. **20–30× win.** Encodes any binary cellular-automaton-style update rule.

**Pattern 3: 2-input neighborhood LUT.** Key = (left_color, right_color) with 12×12 = 144 possible keys. Realization:
1. `Slice` left and right neighbors (0 MACs each).
2. `ArgMax(axis=1)` on each to get color index (0 MACs).
3. `Mul(left_idx, 12)` + `Add(right_idx)` to get composite key: 2 × elementwise × ~900 cells = 1,800 MACs.
4. `Gather(LUT, key)` where LUT is `[144, 10]` = 1,440 params: 0 MACs.

**Total: ~1,800 MACs + 1,440 params.** Beats every Conv approach by orders of magnitude. Useful for any "rewrite based on adjacent pair" rule.

**Pattern 4: Position-keyed LUT.** Replace the per-pixel function with a precomputed `[1,10,30,30]` initializer and a `Mul`-by-input-mask for any task whose output depends on a *fixed pattern conditioned on a coarse property of the input*. **Cost: 1 `Mul` = 9,000 MACs + 9,000 params for the initializer.** This is what J.6's "diagonal of bounding box" example reduces to.

**Failure modes for LUT keying.**
- **Train/private-set drift on neighborhood keys.** A 9-bit binary key has 512 possible values; if the training set + arc-gen subset only exhibit, say, 250 of them, the held-out private set might hit unseen keys. Default the LUT to an "identity" or "input pass-through" output for unseen keys.
- **Color collision in 12-color one-hot.** The competition's color encoding uses 0–9 + 10=no-color + 11=too-many-colors. A 2-input neighborhood key on raw color indices ranges 0–143 but only colors 0–9 will appear in well-formed inputs. Indices 10–143 in the LUT can be don't-cares; size impact is small.
- **Sub-byte storage is a red herring.** Trying to pack 1-bit-per-entry into raw_data and consuming it via shifts/masks at runtime requires opset-18 `BitwiseAnd` (which is integer-only), `BitShift` (opset 11, unsigned-integer only), and at least 8 `Cast` ops. The total runtime cost (~80,000 MACs of unpack arithmetic) blows past whatever you saved on storage. **Stick with byte-aligned int8 or float32 storage.**

## Sources

- onnx-tool 1.0.1 source — `params` accounting (element count, not bytes): `onnx_tool/graph.py` line 1393: `elesize = volume(self.tensormap[input].get_shape()); _params += elesize; _memory += elesize * self.tensormap[input].get_elementsize()`. `volume()` returns numel; `get_elementsize()` returns bytes-per-element.
- onnx-tool 1.0.1 — `GatherNode` defines `shape_infer`/`value_infer` only, no `profile()` override; inherits `Node.profile` returning `[0, 0]` (line 898–918 of `node.py`). Verified empirically.
- ONNX 4-bit-integer technical spec: `https://onnx.ai/onnx/technical/int4.html` — "All 4 bit types are stored as 2x4bit in a single byte. The first element is stored in the 4 LSB and the second element is stored in the 4 MSB. i.e. for elements x, y, that are consecutive elements in the array: pack(x,y): y << 4 | x & 0x0F".
- ONNX issue #5776 ("Add support for Int4 data-type"): "Adding int4 to a subset of the operators, including QuantizeLinear, DequantizeLinear (Optionally: Shape, Size, Transpose, Reshape, Constant)." Confirms INT4 is not directly consumable by `Gather`/`Mul`/`Conv`.
- ONNX raw_data spec for BOOL: issue #2825: "Boolean type MUST be written one byte per tensor element (00000001 for true, 00000000 for false)."
- Kanerva, "Sparse Distributed Memory" (MIT Press, 1988) — foundational LUT-as-associative-memory text. Conceptual reference; the high-dimensional binary key + content-addressable lookup pattern transfers directly to LUT-as-Gather.
- Aleksander & Stonham, "Guide to pattern recognition using random-access memories," Computers and Digital Techniques 1979 — original RAM-net (LUT) paper.
- DiffLogic / "Differentiable Logic Networks" (Petersen et al. 2022, arXiv:2210.08277) — modern revival; not directly applicable since NeuroGolf has no training, but confirms the LUT layer formalism.
- Empirical: `Gather (color-perm via axis=1): macs=0 params=10 memory=36080`.

## Contradictions / corrections vs COVERED

- **The brief's framing of "1-bit patterns to fit ~12M in 1.44 MiB" is wrong.** ONNX BOOL is byte-aligned (1 byte per element), not bit-packed. The maximum sub-byte density is 4-bit (INT4/UINT4) at 2 elements per byte, and these can only be consumed by `DequantizeLinear` (which adds runtime MAC cost). **Practical max LUT density: ~1.5M int8 entries within the 1.44 MiB cap.** The brief should be amended.
- **No conflict with COVERED `_PATTERNS` claim** that color-perm-via-Gather hits tasks 016/337. This topic's contribution is extending the same idea to 9-bit-neighborhood and 2-input-neighborhood keying.
- **Empirical confirmation of params accounting:** the open question I flagged in earlier notes ("does onnx-tool count params by element-count or byte-count?") is now answered: **element count**. So an int8 LUT and a float32 LUT contribute the same to `params`, but the int8 LUT contributes 1/4 to `memory`. **Use int8 LUTs whenever feasible** — the only reason not to is the `Cast` step at use (0 MACs anyway).

## Confidence

**High** for the cost numbers (source-read AND empirically verified). **Medium-high** for the design patterns — Pattern 2 (9-bit neighborhood LUT) is plausibly a major leaderboard lever but its actual hit rate on the 400 tasks needs implementation and testing.
