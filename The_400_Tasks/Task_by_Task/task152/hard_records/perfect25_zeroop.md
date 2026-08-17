# DEAD ROUTE — task152 perfect-25 zero-cost (<=1) family

- **Live bar at test:** cost **90**, sha256 `8b22336696657efc20493b8a5f9b79ef00364bda26ce4e542600d0a7fae5a151` (manifest, owner Vamshi, 2026-07-02).
- **Attempted route:** perfect-25 "ZeroopsweepCeilingAudit" hypothesis — a no-payload direct-output op (Identity / Transpose perms / attr-only) pricing <=1, per prompt `Prompt_2026-07-06T195908Z_5648_Task152`.
- **Exact failed gate:** `ng_gate visible --task 152`:
  - `task152_identity.onnx` → **0/267 (0.00%) FAIL**
  - `task152_transpose_hw.onnx` (perm [0,1,3,2]) → **0/267 (0.00%) FAIL**
- **Why it is dead (structural, not just empirical):** ARC-space transform is a **3×3 → 6×6 expansion** (mirror-tile: TL=identity, TR=fliplr, BL=flipud, BR=rot180) then pad to 30×30. Output element count (36 active) **exceeds** input (9 active) and positions are remapped by the mirror map `[0,1,2,2,1,0]`. No zero-payload op (Identity/Transpose/flip/rotate — all element-count-preserving whole-grid permutations) can synthesize this expansion. The "perfect 25" in the task name = 25 solved examples, **not** a trivial identity task. A visible miss on the zero-op family = certain hidden 0 (traps §VISIBLE-FAIL).
- **One remaining hypothesis (also dead — see floor proof in task152.md):** shrink the LIVE free-output Einsum factor below `[30,3]`=90. Ruled out: the 30-dim is forced by the un-cropped 30-wide grid axes, the 3-dim by three distinct source rows, and the single shared `[30,3]` already covers all four einsum legs (read-h, read-w, write-r, write-s). Any split/crop/gather introduces a counted intermediate (crop 360 B, gather 9000 B, built-map 360 B) or a second initializer (>90). **90 is the forced floor.**

Do not re-explore the zero-op family or sub-90 einsum-factor shrink for task152.
