# Sparse initializer probe dead route

Live/team-best bar: cost 122, SHA-256 `61c58f8fbb942e5a13ed755b3fdc0f7e68c58c405f091240a235d4b0a85359be`.

Attempted route: convert the zero-heavy `Einsum` initializers from the 122-param team-best graph into ONNX sparse initializers to reduce charged parameter count without changing the dense math.

Exact failed gate:
- `prototypes/task261_sparse_inits.onnx`: `score_network_v2 -> None`; no visible/truth/profile gates run.

Why dead: scorer/ORT validation rejects sparse initializer inputs for this `Einsum` graph, matching the project trap that sparse op inputs do not survive the official scorer path.

One remaining hypothesis: only a dense encoding change, not sparse accounting, can improve below 122.
