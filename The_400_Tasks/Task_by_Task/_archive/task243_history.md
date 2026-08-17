# task243 — archived history

Provenance cold-storage for `task243.md`. Live facts stay in `task243.md`
(CURRENT BEST 6014 + 2nd-pass WALL block), `staged_candidates/.../manifest.md`,
and `task243/hard_records/`. This file holds superseded prose only.

---

## RETRACTED floor proof — 11,315 (Aditya `task243_insert_pruned`), w13 2026-06-24

**Status: SUPERSEDED / RETRACTED.** 11,315 was broken by Vamshi's combined Einsum
(6985), then by the single-axis Einsum family (4718 probe, 6014 SAFE floor). The
"11,315 is the floor" verdict is WRONG. The 4-conn / round-count / dtype / MaxPool
facts below remain true and are the reason the CURRENT floor (6014) also holds; they
are re-stated live in the 6014 manifest and `hard_records/`.

Team-best ONNX `../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task243.onnx`
SHA256 `2761FFEC56E1D94FF68619E27D6A8C0608619E60BF673B3047FB37F32EFEF871`,
score_network_v2 = (macs 0, memory 10908, params 407) ⇒ **cost 11,315**, profile==static (fully static), visible 265/265.
It is a **U8-byte-lane bit-packed alternating-axis flood** (NOT the 21,328 bitpack28): each 18-wide row packed into 3 UINT8 lanes
(bits 0-7 / 8-15 / 16-17) via one Conv→u32-as-F32 then byte-split (Cast(u8)=low, Mul(1/256)+Cast=mid/hi). 36 single-axis rounds = **20 vertical + 16 horizontal**, hand-tuned interleave `vhhvvhvhvvhvvhvvhvhvhhvvhvhvhvhvhvvh`. V round = Pad(up/down)+OR per lane (~149 B/round); H round = within/cross-byte BitShift carry + OR (~250 B/round). Unpack: per-lane BitwiseAnd(2^c masks)→Cast(BOOL)[1,1,18,8]→Concat[1,1,18,18]→Pad[1,1,30,30]→Where(color1,input).

**w13 floor proof that 11,315 is at/near the formulation floor (so verdict = NO-CANDIDATE):**
1. **Round count is a hard diameter floor (§37 corpus-measured).** Dropping ANY single round from the 36-round schedule fails the 265-pair official corpus (drop-1-H → 1 fail; drop-1-V → 1 fail). Combined 4-conn step needs 28 rounds, alternating needs 39 pure / 36 tuned. Round-cost = 6,984 B (V 2,988 + H 3,996) — minimal for this schedule.
2. **Alternating split beats every alternative.** Combined-step at U8 pays the expensive cross-byte H carry every round (28×) vs 16× → loses. Orientation-swap (pack rows instead of cols) is symmetric (14H vs 15V floor) → no gain. My Bresenham schedule search = 41 rounds (worse); cost-aware greedy = 36 @ worse V/H ratio. Aditya's 20V/16H is near-optimal.
3. **dtype is at floor.** U8 lanes = 1 B/elem (cheapest scoreable; UINT16 has no ORT kernel; the 21,328 bitpack28 used U32 = 4 B/elem). F16 was tried 2026-05-20 → catastrophic adversarial fail.
4. **8-conn MaxPool flood is WRONG** (differs 264/265 — rule is strictly 4-conn; diagonals blocked); 4-conn MaxPool = unpacked [1,1,18,18] planes (324 B/plane) ≫ packed lanes.
5. **Setup (540 B) and unpack (1,872 B) are information-required.** Setup is dominated by F32-locked Conv bridge outputs (6×72 B). Unpack is dominated by `fill_full` BOOL[1,1,30,30]=900 B — the irreducible non-separable 30×30 Where condition. Unpack fusion (keep masked-u8, single Cast) saves only ~36 B.
6. **No closed-form alternative** — region reachability/connectivity is intrinsically iterative; CC-labeling (#33/#126) propagates labels = more expensive than binary bit-flood.

Best-case extractable slack across the whole graph ≈ 100-300 B (micro-fusions only); a hypothetical 10,600 yields ln-gain +0.065 — below a confident strict beat and not worth the risk of perturbing the exactly-tight 36-round schedule. **Verdict: NO-CANDIDATE; 11,315 is the floor for the iterative-bit-packed-flood family.**
