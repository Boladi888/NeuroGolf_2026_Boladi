# task012 stale free-output Pad/Equal-tail route

- Live bar: **858**, SHA-256 `e936055201ecc0182097c6faf68bae80c1dad3a95a9e7772c2b601aaddfbb87e` (authoritative team manifest, Sho confirmed-public model).
- Attempted route: refresh the prompt's one-channel predicate/index plus Pad/Where/Equal output tail against the claimed charged 2880-byte prepad.
- Exact failed gate: **structure/price fast-probe before build**. `inspect_graph.py Team_Best_Onnx/task012.onnx` shows the live graph has no Pad, Equal, or 2880-byte carrier; it already uses a two-channel U8 role feature and terminal free `QLinearConv`. A one-channel padded value canvas/Equal tail forces roughly 900 B for the canvas (about 1044 B including the 12x12 source), already above both the 858 live bar and the prompt's 420 kill threshold.
- Why dead: one scalar followed by the current affine positivity decoder also cannot represent background, A, and B as three mutually exclusive classes; one of the three scalar codes is necessarily a non-separable middle class. The prior 666 estimate was retracted after full one-hot background review.
- One remaining hypothesis: a genuinely non-affine one-channel output tail that deletes `feat[1,2,12,12]` without materializing a 900 B canvas; no dense, scorer-safe construction is currently known. The exact same-topology floor is the staged 821 candidate.
