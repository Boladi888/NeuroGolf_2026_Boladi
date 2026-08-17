# score-coded first-index route

- Live bar used: manifest/team-best task153 cost 576, sha256 `09aa50d3395919a0efaeb2d0763086401a6ddda1677bc91bce6d04e6253284ef`; local staged rung to beat is `task153_twocolorextraction_cost575` at profile cost 575.
- Attempted route: task-local score-coded row/col first-index extractor in `prototypes/extracted_submission_task153.onnx`, replacing MaxPool/TopK index extraction with `Einsum` row/col scores and log/floor decode.
- Exact failed gate: `_score_v2` returned `None`; `profile_real_cost.py --task 153 --cand prototypes/extracted_submission_task153.onnx --arcgen 266` reported `static=None profile=655 (memory 559 + params 96; 265 real inputs)` and ORT raised `Einsum operands could not be broadcast together` on the scorer dummy path with a zero-sized dimension.
- Why dead: visible passed (`[visible] 265/265 (100.00%) PASS`), but the route is scorer-invalid as built and its real-input profile cost is already +80 bytes versus the 575 local staged rung, so heavy truth/AB/full gates would be burn.
- One remaining hypothesis: a selected-score clamp after the row/col `Gather` nodes could make the dummy path valid, but it adds charged tensors and only merits revisiting if paired with a separate tail rewrite that removes more than 80 bytes.
