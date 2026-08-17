# task158 crash-fix via TF table-extension — DEAD (2026-07-12, bo_ng_cld_worker1)

**Bar:** LIVE v130 = 23,187 (manifest, sha `84bf2db2…`, crash-free clamp).

## Idea (rejected)
v130's crash fix is `CPc=Min(CP,76)` → +2,601 B (INT32 [1,1,26,25] plane + c76).
Alternative: drop the clamp, extend the paint table `TF` from 77→N entries with
entries 77..N-1 replicating `TF[76]` (so byte-equal to v130's clamp for all CP≤N),
at only +6 B/entry (TM/TA/TS/TF +1 B each = +4 mem, IDXM/IDXA +1 each = +2 params).
This beats the clamp iff N < ~510.

## Why DEAD — max CP grows unboundedly with corpus size
Probe `prototypes/_probe_cp_max_w1.py` (exposes CP as a graph output, scans
officials + N corpus boards, reports per-board max CP):
- 266 officials: **max CP = 76** (NO overflow — the crash is a synthetic-corpus
  artifact; real/official boards never overflow).
- 8,266 boards: max CP = 273 (4 overflow boards).
- 60,266 boards: **max CP = 1,254** (44 overflow boards), buckets
  {≤76: 60222, 77-128: 26, 129-256: 11, 257-512: 4, >512: 3}.

Max CP keeps climbing with sample size (273→1254). To cover the corpus a
byte-equal table needs N ≥ 1,255 → +6×1,178 = **+7,068 B ≫ the +2,601 clamp**.
And it would still crash on any unseen board with CP>N (provable max CP = 339,150,
per prior worker-7). The `Min(CP,76)` clamp is the correct BOUNDED, provable fix;
no cheaper crash-free-on-all-realizable-boards substitute exists via table size.

## Verdict
Clamp is forced for provable crash-free. CP (2,600) + CPc (2,600) INT32 double is
the price of a crash-free data-dependent Gather render. Do not re-explore table
extension. (Render restructure to drop the separate `pos` plane is a *different*,
live lead — see the v131 paint-cells-mask candidate.)
