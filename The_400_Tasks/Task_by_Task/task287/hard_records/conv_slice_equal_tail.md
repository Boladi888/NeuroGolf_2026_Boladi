# conv_slice_equal_tail dead route

Live bar used: cost 190, sha256 `b3dda820c5b93cb43f60e655a297068590e5a584d36466d5b0201784d0002861` from `python tools/ng_current_bar.py --task 287 --json` / manifest.

Attempted route: the prompt's Conv2x2 + reverse-Slice + Equal-tail family. Existing task-local prototypes from that family were repriced against the refreshed 190 bar.

Exact failed gate: score/pricing.
- `task287_organic_v4.onnx`: `score_network_v2 -> (0, 60980, 42)`, cost 61,022.
- `tmp_f16_task287_f16_v2.onnx`: `score_network_v2 -> (0, 79600, 110)`, cost 79,710.
- `task287_organic_v3.onnx`: `score_network_v2 -> (0, 71220, 42)`, cost 71,262.

Why dead: the refreshed team-best is a zero-intermediate, one-node `Einsum` costing only 190 params. Any Conv/Slice/Equal-tail graph materializes at least one canvas/tail intermediate; the Equal-tail canvas alone is about 900 bytes before Conv/Slice plumbing, so it cannot beat 190.

One remaining hypothesis: a single terminal op that computes the full one-hot output without materialized intermediates can still compete; that is the current dense bit-factor `Einsum` family, not the Conv/Slice/Equal-tail family.
