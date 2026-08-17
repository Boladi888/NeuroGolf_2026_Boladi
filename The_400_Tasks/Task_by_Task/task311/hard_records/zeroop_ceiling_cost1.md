# DEAD ROUTE — task311 zero-param cost≤1 ceiling (ZeroopsweepCeilingAudit)

- **Live bar:** cost **30** — manifest authoritative, sha256 `250771b94d3db4658d4ac04e3bd9a0740f2af0f3aba0bb9b5df4d6181bb470d4`, owner Vamshi, updated 2026-07-02 (not stale; ng_current_bar.py).
- **Attempted route:** the prompt's SPEC "perfect-25 forced floor ≤1" hypothesis — a no-payload direct-output op (Identity / Transpose perms / attr-only) that reproduces the horizontal mirror-stack.
- **Exact failed gate:** `ng_gate visible --task 311`:
  - `z_identity` (Identity, cost 0) → **0/266 FAIL**
  - `z_transpose_wh` (Transpose perm=[0,1,3,2], cost 0) → **0/266 FAIL**
  - Control: LIVE `Team_Best_Onnx/task311.onnx` visible **266/266 PASS**, profile `static=30 profile=30`.
- **Why it's dead (forced-component argument):** the rule places input column 2 at output columns 2 **and** 3 (`out[r]=[c0,c1,c2,c2,c1,c0,bg×24]`) — a strict one-to-many column mapping. No attr-only / zero-payload ONNX op performs column duplication: Identity copies (col3=bg≠c2), Transpose permutes axes (no intra-row duplication), Slice/reverse gives `[29..0]` not the mirror-stack, and Pad has no `symmetric` mode (`reflect`=[0,1,2,1,0]≠[0,1,2,2,1,0]; documented +3611). The minimum-payload op that duplicates a column is **Gather with a length-30 index = 30 params** (memory 0, since the Gather output IS the graph output). Therefore cost **30 is the forced floor** and the ≤1 ceiling is provably unreachable for this rule.
- **One remaining hypothesis:** none for ≤1. Sub-30 territory is separately closed in the byte-identical twin `task164.md` (idx[30] INT8→None; sparse idx→None; computed index=343; Slice+Concat+Pad→None). **30 is the hard floor; do not re-hunt.**
