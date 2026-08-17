# task059 w1_sparse_base4 dead route

- live bar: 470, SHA256 `105da16b59be13e9d797c5d0121e3cc1fb749eb4799b254155921b686bf0900d` (`python tools/ng_current_bar.py --task 59 --json`, manifest source, owner Sho, updated 2026-07-09T01:09:17Z).
- attempted route: store the Concat-only `base4 [1,1,4,4]` initializer as a sparse initializer with 10 stored F16 values instead of 16 dense elements, targeting a 6-param shave.
- exact failed gate: `score_network_v2` returned `None`; visible failed at ORT load after scorer sanitization with `Invalid model. Node input 'safe_name_11' is not a graph input, initializer, or output of a previous node.`
- why dead: the official sanitizer does not carry sparse initializer names through as ordinary initializers, leaving the Concat input dangling. This matches the project sparse-initializer trap even though `base4` is not an Einsum operand.
- one remaining hypothesis: only revisit if scorer sanitization gains sparse-initializer support; a Constant sparse node would materialize `base4` as a charged 32B intermediate and is already dominated by the dense 16-param initializer.

