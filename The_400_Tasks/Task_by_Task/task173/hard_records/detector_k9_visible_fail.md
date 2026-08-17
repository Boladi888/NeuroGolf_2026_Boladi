# task173 hard record - detector k9 crop

- **bar used**: team-best/profile cost `8710`, sha256 `27f31830cf75f60bdcf0273b403dcb804b8f4b6950817091f50e761be3a57940`.
- **bar source**: refreshed workbook and manifest; task-local staged manifest agrees.
- **attempted route**: reuse existing task-local prototype `prototypes/km_k9.onnx` to shrink the detector map below the k=8 23x23 detector.
- **exact failed gate**: visible gate:
  `[visible] 262/266 (98.50%)  FAIL`
- **why dead**: one visible miss is a certain hidden zero under the protocol. This confirms the task-note floor proof: k=9's 22x22 detector misses complete-template centers at the active-grid edge.
- **one remaining hypothesis**: only a detector that keeps row/col coverage equivalent to k=8 while reducing dtype or avoiding F32 Conv output could help; F32 input lock currently makes that more expensive than the 2116 B `detect_map`.
