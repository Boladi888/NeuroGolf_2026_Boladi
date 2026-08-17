# task323 direct marker-to-class-masks hard record

- Live/promoted bar: 1020 from `python tools/ng_current_bar.py --task 323 --json`, authoritative source `<TEAM_REPO>/Team_Submission/Backend/manifest.csv`, sha256 `e68850f364c4b2e265a7dcc6aa26b9faddfccf9da073f7b4a19a21f59628e70a`.
- Task-local reviewer bar: 1018 from `The_400_Tasks/Task_by_Task/task323/staged_candidates/task323_recenter_zero_reuse_cost1018/manifest.json`, sha256 `cbc8f9235f224c3c95c913be628ab67af8df375de0e18ad98f12c50f425f72b7`.
- Locked route: replace `compact_motif_code` + `saturating_exact_masks` with one legal direct `QLinearConv(marker -> class_masks)`, preserving the existing anchor extraction and final `ConvInteger` decoder.
- Failed gate: one pre-build feasibility audit, `python The_400_Tasks/Task_by_Task/task323/prototypes/audit_direct_marker_to_masks.py`.
- Exact audit result:
  - `required gray offsets: 48`
  - `required cyan offsets: 1`
  - `required union offsets: 49`
  - `row offset span: -12..12`
  - `col offset span: -12..12`
  - `best legal dilation lattice: dilation=(1,1) kernel=(25,25) area=625`
  - `winning max area without direct bias: 189`
  - `winning max area with direct bias: 188`
  - `cost floor without direct bias: 1888 = memory 571 + params 1317`
  - `cost floor with direct bias: 1890 = memory 571 + params 1319`
  - `verdict: INFEASIBLE_BELOW_BAR`
- Why dead: the marker has one nonzero centered tap, so a direct QLinearConv can only distinguish a true gray/cyan cell if its kernel receptive field covers that anchor-relative offset. The required offsets span `[-12,12]` on both axes and include consecutive residues, forcing dilation `(1,1)` and a dense `25x25` kernel. Deleting the 210-byte `code` tensor and 170 code/mask params cannot offset the required `2*625` direct kernel params.
- One remaining hypothesis: only a different procedural motif/code basis can attack the `code_kernel` cost; do not retry the one-shot `marker -> class_masks` QLinearConv replacement under the current marker semantics.
