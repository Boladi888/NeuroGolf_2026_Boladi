# Dead route: score-grid row/col trim (bg_score17 < 16×16)

- **Live bar:** 3759 (manifest+workbook, Aditya, sha256 `460d7e10b614fb8cd16a82fd7567f63d166c287b56bfc0b5a336f7fb356aa73b`, updated 2026-06-14). LIVE truth = 800/800 (100%).
- **Attempted route:** shrink `bg_score17` below 16×16 to shave U8 score bytes (16 B/row, 16 B/col), compensating the lost row/col with extra pad in the second (paint) QLinearConv so `mask19_u8` stays 19×19. Candidates left by `bo_ng_cdx2` in `prototypes/`.
- **Why it looks tempting:** static == profile cost strictly below 3759 (clean static graph, no fiction):
  - `task208_bo_ng_cdx2_score15x16.onnx` (15 rows) → 3743 (mem 3652 + params 91)
  - `task208_bo_ng_cdx2_score16x15.onnx` (15 cols) → 3743
  - `task208_bo_ng_cdx2_score15x15.onnx` → 3728
  - `task208_bo_ng_cdx2_u8_geometry.onnx` → 3755
- **Exact failed gates:**
  - `score15x15`  → `[visible] 259/266 (97.37%) FAIL`  (drops row15 AND col15 placements)
  - `score16x15`  → `[visible] 259/266 (97.37%) FAIL`  (drops col15 placements)
  - `u8_geometry` → `[visible] 0/266 (0.00%) FAIL`
  - `score15x16`  → `[visible] 266/266 PASS` (smoke only) then **HIDDEN-BAD**:
    - `ab --n 766` → `match 762/766 | divergences=4 | byte_equal=False (first_div=602)`
    - `truth --n 800` → **793/800 (99.125%)** vs LIVE 800/800 (100%). Divergences are candidate ERRORS, not more-correct-than-LIVE (LIVE is perfect on the same sample). traps §14e: <99% truth + not byte-equal = learning artifact, never "best".
    - `first_div=602` is BEYOND the 266 visible corpus → classic visible-pass / hidden-fail signature. This is the documented "row trim hidden-bad" trap (task208 don't-do log).
- **Why it's dead (forced-component evidence):** placement anchors in `bg_score17` span the **full 16×16** across the real corpus — hot-cell row range `[0,15]`, col range `[0,15]` (measured over train+test+266 arc-gen on the LIVE graph). Trimming any row or col of the score grid necessarily drops the placement anchors that live in that row/col, so it can only ever be a lossy approximation, not a reparametrization. The 15×16 pad-compensation trick masks this on the small visible set but not on ground truth.
- **Remaining hypothesis:** none for score-grid trimming — it is provably lossy. Any real win must remove/hide a whole charged surface (`ch0_inner17` F32 1156, `mask30` BOOL 900, or the `mask19_u8`+`mask19` = 722 pair), and those are floored for the 2-placement conv-stamp algorithm (see task208.md wall note + `bo_ng_cdx2_worker1_packet.md`). WALL at 3759.
