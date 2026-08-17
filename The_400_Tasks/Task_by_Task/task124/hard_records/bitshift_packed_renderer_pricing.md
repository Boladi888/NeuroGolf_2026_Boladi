# bitshift_packed_renderer_pricing

- Live bar: manifest cost 920, SHA `219bccc70d68bf8ea5505adad4aebafb0b5a57775ff83cf6f8545e19d2fa362b`.
- Attempted route: price the prompt's packed `BitShift`/`BitwiseAnd` renderer as a replacement for the current 10x10 `QLinearConv` final renderer.
- Failed gate: pre-build pricing gate; no ONNX variant built.
- Why dead: `BitwiseAnd` cannot provide the current free 10x10-to-30x30 spatial padding that `QLinearConv(pads=[0,0,20,20])` gives at the graph output. A packed final renderer needs either a `[1,10,30,1]` per-channel row-code input (10*30*4 = 1200 B for U32) or a `[1,10,10,30]`/Pad input of the same order, before row-bit construction and color/background separation. That already exceeds the current charged renderer surface (`feature_grid` 200 + `out_fg` 100, with output skipped).
- Remaining hypothesis: packed bits may still help only if the bottom-row foreground synthesis can stay packed and feed a final op with built-in padding; plain final-output bit unpack is not competitive for this graph.
