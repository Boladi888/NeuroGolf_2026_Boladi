# task138 - DEAD ROUTE: rank-4 shared row/col basis probe

**Recorded:** 2026-07-07 by `bo_ng_cdx_worker2`, task138 only.

## Live / bar at time of probe

- Authoritative live from `python tools/ng_current_bar.py --task 138 --json`: **5403**, owner Vamshi, sha256 `c077e34a5e6c47f2818e70d5337d672fd4d8962b36cd2de8d1be32d0eb67c97a`, manifest updated `2026-07-02T02:35:15Z`.
- Lower task-local staged bar validated this pass: **5278**, `staged_candidates/task138_dedupinit_cost5278/task138.onnx`, sha256 `1a53cbd15e46c93932ed7dd9675b48774256d7e426870cfa3c6cef7cbca909db`.
- Current staged win after this pass: **5276**, `staged_candidates/task138_squeezevcol_cost5276/task138.onnx`, sha256 `bdda4d889ce179f6fd9fdf363ba3ad639494ce1fdd72c06a51ba65b9ccd05f80`.

## Attempted route

Prompt route: reduce the shared free-output row/col basis below the current rank-5 family. I exposed `rowfeat_h` and `colfeat_h` from the 5276 graph as additional graph outputs in-memory, then measured their numerical matrix ranks over cached ARC-GEN inputs. No ONNX file was written for this probe.

## Exact failed probe

```text
rank probe n=5000:
row_rank_hist={3: 45, 4: 253, 5: 4702}
col_rank_hist={4: 44, 5: 4956}
joint_rank_hist={(3, 4): 19, (3, 5): 26, (4, 4): 25, (4, 5): 228, (5, 5): 4702}
first_nonfinite=None
```

## Why this route is dead

A rank-4 shared basis cannot represent the active family generically: both row and column bases are simultaneously rank 5 on **4702/5000** fresh generated inputs, and the column basis alone is rank 5 on **4956/5000**. This hits the prompt kill rule: if rank needs `>=5`, stop rather than building rank-4 variants. No visible/truth gate was run because the rank precondition already fails.

## One remaining hypothesis

Do not try to drop the basis rank directly. The only plausible structural route is a different bounded positional code or a different feature factorization that changes the basis itself before final rendering; it must also preserve raw `nan`/`inf` cleanliness because the old exp-ramp F16 route is already documented dead.
