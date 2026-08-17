live bar: 4074 profile_cost, sha256 234c8a210458b062aada6f4a1a9ea75c15a20501af71852b12c148838d23eda3.

Attempted route: downgrade the graph to opset 10 so Pad can use free `pads`/`value` attributes, deleting `pad_spatial` and `sentinel10_u8` params.

Failed gate: `score_network_v2` returned `None`; `ng_gate visible --json` failed to load with `InvalidGraph: Type 'tensor(uint8)' of input parameter ... of operator (Equal) ... is invalid.`

Why dead: Pad attributes require opset <=10, but the current graph relies on U8 Equal/Add/Sub legality from later opsets. Rewriting those predicates to opset-10-safe forms would add full planes and exceed the 5-param Pad saving.

Remaining hypothesis: only revisit Pad params if ONNX/ORT gains per-node opset mixing, otherwise keep the opset-20 Pad-with-axes tail.
