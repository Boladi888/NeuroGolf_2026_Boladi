# task392 dead route — invariant-bias signed shift to L62

- Live/task-local bar: live **1001** SHA-256 `4dfb666721285b099201df853ccf962b8598bf6cacc66a2c1b6c282358d8f283`; task-local staged bar **955** SHA-256 `8e6460e4534e5c3290e82d5a02e140a052cfa3f2aa2fd41ce6cdab19df7d7d1a`.
- Attempted route: add an invariant `ch0(9,9)=1` probe with weight `-5` to shift the signed67 raw range from `[-57,66]` to `[-62,61]`, then shrink the packed row tables to length 62.
- Exact failed gate: `python The_400_Tasks/Task_by_Task/task392/prototypes/build_signed62_bias_probe.py` raised `RuntimeError: conflicting row-table slot 38`.
- Why dead: signed `Gather` aliases by `index mod table_length`; changing length from 67 to 62 changes alias classes, so a constant shift does not preserve the collision-free table mapping.
- Remaining hypothesis: search real lower-modulus hash weights/subsets directly; the successful follow-up solved same-probe length 66 at cost 953, while L65 or alternate lower tables remain open.
