# Card J1 — Row-packed uint32 morphology: exact ONNX constructions for pack / shift-OR flood / unpack

**Researcher contract (read first)**: You are doing INTERNET research only. You have NO access to our repository, code, files, or machine — everything you need is in this prompt; do not ask for files and do not assume any project detail not stated here. Produce ONE copyable Markdown research note as your entire answer (no code execution, no file writes). Follow the required answer order and table schemas exactly. Tag every claim `[PROMPT]`, `[SPEC]`, `[ORT-DOC]`, `[ORT-SRC]`, `[DERIVED]`, or `[SPECULATIVE]`. Cite human-readably (op-version pages, file::function, author-year). End with a one-paragraph blunt verdict.

**Deliverable**: a research note (`row_packed_uint32_morphology.md`).

**Context (project facts — all [PROMPT])**: ONNX graphs on onnxruntime CPU, `ORT_DISABLE_ALL`, opset ≤ 20 grader-verified. cost = memory + params; memory = Σ trace-visible intermediate bytes (graph input/outputs + initializers free); params = initializer element count (dtype-blind). Banned: Loop/Scan/NonZero/Unique/Compress. Grids ≤ 30×30; masked flood-fill (`reach ← dilate3x3(reach) ∧ mask`, unrolled K times, K = corpus-measured depth 5–25) currently costs one 900 B U8 plane per round and dominates several task floors. Prior research established: doubling schedules are NON-exact (corridor counterexample); a 5×5 round = two 3×3 rounds only under a corpus-certified convexity condition; opset-18 `BitwiseAnd/Or/Xor/Not` CPU-confirmed at u8/i32/u32, `BitShift` CPU-confirmed at u8/u32/u64 (UNSIGNED only, elementwise — bits never cross tensor elements); bitboard literature (chessprogramming Dumb7Fill, Finch's Life) shows shift-OR flood on packed words. **The promise**: a [30]-element UINT32 vector holds a whole 30×30 grid (one row per lane, ≤30 bits used); a flood round on that vector costs ~120 B instead of 900 B — IF exact ONNX pack/step/unpack constructions exist.

**Known traps to respect ([PROMPT])**: F32 arithmetic is exact only to 2^24 — packing 30 bits via float MatMul with powers-of-two weights is NOT exact; `MatMulInteger`/`QLinearMatMul` accept only u8/i8 inputs; `Gather` indices ≥ INT32; our graph INPUT is fixed F32 [1,10,30,30] (free to read, any cast of it is a charged intermediate).

**Research questions (answer in this order)**:
1. **Pack**: cheapest exact construction from a [1,1,30,30] BOOL/U8 plane to a [30]-lane UINT32 row-pack under the cost model. Candidate families to price: (a) log-tree of `BitShift`+`BitwiseOr` on successively halved column groups (what intermediate shapes/bytes?); (b) per-column weighted sums in I64 then Cast→U32 (I64 exact to 2^53 — but 8 B/elem intermediates); (c) `Slice` column-extracts + shift-or accumulation (30 steps of [30]-vectors = tiny). Give exact byte totals for each at 30×30.
2. **Flood round on the packed vector**: east/west = `BitShift(x,1)` ± edge masks (price the masks); north/south = lane shifts via `Slice`+`Pad` (or `Concat`) on the [30] axis; round = `Or` of 5 terms ∧ packed mask. Exact byte cost per round; compare to 900 B unpacked. Verify every op's u32 CPU legality at opset ≤ 20.
3. **Unpack**: cheapest exact [30]-lane U32 → [1,1,30,30] BOOL: (a) broadcast `BitwiseAnd` against a [30]-element power-of-two initializer + `Greater(,0)` (what's the intermediate? [30,30] I32 = 3,600 B?); (b) per-bit shift+mask accumulation; (c) `Mod`/`Div` ladders. Give totals; identify the cheapest that lands in BOOL/U8 at 900 B final.
4. **End-to-end pricing**: for K ∈ {5, 11, 14, 25}, total bytes for pack + K packed rounds + unpack vs K × (900+900) unpacked rounds (dilate+mask outputs). Where is the crossover K? Include params.
5. **Exactness audit**: confirm the packed 4-neighbor round is EXACTLY one geodesic step (no doubling shortcuts smuggled in); enumerate edge conditions (bit-30/31 hygiene, lane-0/29 boundaries) and the masks each needs.
6. **Blunt verdict**: adopt / adopt-conditionally / reject, with the K-crossover number.

**Required table schema (Q4)**: | K | packed total bytes | unpacked total bytes | ratio | params |

**Source policy**: ONNX op-version pages for legality; ORT kernel registry/source for CPU dtype rows; bitboard literature for the shift algebra; everything cost-numeric is [DERIVED] from the stated model.

**Verification checklist**: Q1 prices ≥3 pack families; Q2 verifies every op-dtype pair; Q4 has the crossover; Q5 lists every boundary mask; blunt verdict present.
