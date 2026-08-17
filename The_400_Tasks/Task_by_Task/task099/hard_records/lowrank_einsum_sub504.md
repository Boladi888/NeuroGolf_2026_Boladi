# task099 hard record: sub-504 low-rank Einsum factor sweeps

- Live bar used: 504 bytes, SHA-256 `6c419ff35ddc8651209c1ac5f32b0f2aecce69501cd274348e10c97a469bcfed`, from `python tools/ng_current_bar.py --task 99 --json` authoritative manifest source.
- Attempted route: preserve the current free-output one-node `Einsum` family while reducing charged initializer count below 504 by sweeping `(terms,basis)` shapes: `6x5` cost 482, `7x4` cost 446, `10x4` cost 500, and `15x3` cost 500.
- Probe builder/search script: `The_400_Tasks/Task_by_Task/task099/prototypes/fit_lowrank_einsum.py`; trained against the full finite generator enumeration of 496 unique inputs from ARC-GEN `task_444801d8.py`.
- Exact failed gate: in-memory official visible check after fitting stayed visible-zero for every tested shape (`official ok 0/265`); representative results: `6x5` best runs had >149k official cell sign diffs, `10x4` best run had 147,568 official cell sign diffs, `15x3` best run had 127,558 official cell sign diffs.
- Why dead: the 504-byte parent's 7-term/5-basis dense color factorization is not carrying obvious redundancy. Dropping a term, dropping a basis row, or trading basis rows for extra CP terms leaves many false positives/negatives after L-BFGS optimization and never reaches a visible-passing neighborhood.
- One remaining hypothesis: a different one-node equation with a semantic color/geometry decomposition, rather than this parent-preserving CP/Tucker form, could still lower params; start from the generator rule, not from truncating these factors.
