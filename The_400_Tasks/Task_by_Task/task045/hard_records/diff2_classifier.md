# task045 hard record: diff2 classifier

- Live bar: 512 bytes, sha256 `4230502505d6e36e8aab07c4039c8af5a6aaf9b53c6217b209b9bc677aad7fcf`, source `python tools/ng_current_bar.py --task 45 --json` / manifest owner Sho.
- Attempted route: replace `Equal -> Not -> Cast` nonmatch flag with `Sub(left_i8,right_i8)` and `Mul(diff,diff)`; retune `q_shift_w`, `q_cls_w`, and `q_cls_b` so the charged graph drops the bool/nonmatch tensors.
- Exact failed gate: visible gate on `prototypes/task045_diff2_classifier.onnx` returned `correct=0 total=265 pct=0.0 pass=false`.
- Why dead: visible-zero means certain hidden zero under the worker protocol. Local decode inspection of one train example matched the active 10x10 grid, so the remaining mismatch is likely exact one-hot/padding/channel behavior rather than the visible row-fill rule itself.
- Remaining hypothesis: a different rewrite would need to prove exact padding and channel-0 semantics while eliminating the nonmatch flag; do not continue the diff2 proxy family without a new representation proof.
