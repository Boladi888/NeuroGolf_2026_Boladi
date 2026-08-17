# task045 hard record: nonmatch Where rewrites

- Live bar: 512 bytes, sha256 `4230502505d6e36e8aab07c4039c8af5a6aaf9b53c6217b209b9bc677aad7fcf`, source `python tools/ng_current_bar.py --task 45 --json` / manifest owner Sho.
- Attempted route: replace `Not -> Cast` nonmatch flag with `Where(same_color, zero, one)` to remove one charged 10-byte intermediate.
- Exact failed gate: profile/load gate on `prototypes/task045_where_nonmatch_i8.onnx` failed with `NotImplemented: Could not find an implementation for Where(16) node`.
- Exact failed gate: profile/load gate on `prototypes/task045_where_nonmatch_u8.onnx` failed with `NotImplemented: Could not find an implementation for QLinearConv(10) node`.
- Why dead: ORT does not support the needed int8 `Where`, and the uint8 row-feature variant crosses into an unsupported `QLinearConv` dtype combination before visibility can be tested.
- Remaining hypothesis: only revisit if the graph avoids `Where` entirely or keeps the final `QLinearConv` input/output dtype pairing identical to the live int8-supported path.
