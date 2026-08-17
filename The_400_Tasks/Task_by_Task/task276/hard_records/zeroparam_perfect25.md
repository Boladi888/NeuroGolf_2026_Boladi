# task276 DEAD ROUTE — zero-param "perfect-25" identity/transpose (cost <=1)

- **Live bar at check:** cost **10**, sha256 `2a3e881ae92099c36f1955bdf96ca1e9f77a4ef3d1d3eb58c63eccc80a655c7e` (manifest, owner Vamshi, 2026-07-02). `ng_current_bar --task 276`.
- **Attempted route:** the batch53 ZeroopsweepCeilingAudit prompt hypothesised 276 is a "perfect 25" priced <=1 via a no-payload direct-output op — Identity, Transpose perms, or any attr-only final op.
- **Exact failed gate:**
  - `Identity` (input->output, cost~0):        `ng_gate visible --task 276` = **0/266 FAIL**.
  - `Transpose perm=[0,1,3,2]` (H<->W swap):    `ng_gate visible --task 276` = **0/266 FAIL**.
  - (Per §VISIBLE-FAIL a visible miss = certain hidden 0 — no truth/ab/battery run on these.)
- **Why it's dead:** 276 is a **recolor**, not a copy. ARC-GEN ground truth (2000 pairs, seed 7777, `prove_floor.py`): input colors `{6,7}` (pink bg + orange) -> output `{2,7}` (red bg + orange). So **input != output** -> no zero-param direct-output op can be visible-correct. The required 2<->6 channel swap is NOT reshape-transpose-expressible (brute force over factorizations {(2,5),(5,2),(10,)} finds no perm with idx[2]=6 AND idx[7]=7) and NOT a full-axis reversal (reversal gives out[2]=in[7], we need out[2]=in[6]). The only op that reindexes channels arbitrarily is `Gather(axis=1)`, which structurally needs `idx.shape==[10]` => 10 params. Sub-10 requires `sparse_initializer` => `score_network_v2 -> None` (UNSCOREABLE).
- **Floor:** cost **10** is the proven GLOBAL floor (forced component: one `[10]` INT64 Gather index, memory 0). Not just no sub-10 — cost<=1 is structurally impossible.
- **One remaining hypothesis:** none under the current scorer. Only a change in scorer behavior (e.g. sparse-initializer becoming scoreable, or a new attr-only channel-permute op) could reopen this. Do not re-explore the zero-param family.
