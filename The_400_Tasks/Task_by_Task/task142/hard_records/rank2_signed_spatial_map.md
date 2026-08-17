# task142 hard_record — rank-2 signed spatial map probe

**Live bar at audit:** cost **90** · sha
`93e24c57db3e11645f669d852569898ad47ed9aaded90e8ef4c95a2dbec99f76`
(`python tools/ng_current_bar.py --task 142 --json`, manifest, owner Sho,
updated 2026-07-09T01:09:17Z).

**Attempted route:** replace LIVE's exact shared `[30,3]` F32 map with a shared
rank-2 signed embedding `[30,2]` and keep the same free-output `Einsum` family:
`hp,rp,bcrs,wq,sq->bchw`. Hypothetical static cost is **60** params, memory 0.

**Prototype:** `The_400_Tasks/Task_by_Task/task142/prototypes/build_task142_rank2_spatial_probe.py`
builds `build_task142_rank2_spatial_probe.onnx`.

**Candidate sha256:** `1b1a5a261fb0a1af9216f8bdb765a5b49b5527cd12561bb12dcc596d69bec92e`.

**Failed gate:** visible.

```
python The_400_Tasks/Task_by_Task/task142/prototypes/build_task142_rank2_spatial_probe.py
saved <REPO_ROOT>\The_400_Tasks\Task_by_Task\task142\prototypes\build_task142_rank2_spatial_probe.onnx
score: macs=0 memory=0 params=60 cost=60

python tools/ng_gate.py visible --task 142 --cand The_400_Tasks/Task_by_Task/task142/prototypes/build_task142_rank2_spatial_probe.onnx
[visible] 1/266 (0.38%)  FAIL

python tools/ng_graph_cost_diff.py --task 142 --live Team_Best_Onnx/task142.onnx --candidate The_400_Tasks/Task_by_Task/task142/prototypes/build_task142_rank2_spatial_probe.onnx
NET COST DELTA -30; removed initializer A [float32 [30,3]] (-90 params), added initializer A2 [float32 [30,2]] (+60 params)
```

**Why dead:** with a shared rank-2 spatial embedding, the row/col selection
matrix is a Gram factor. Three source rows cannot be mutually orthogonal in
rank 2. Any nonzero off-diagonal row factor creates positive double-wrong
row/col products, so arbitrary color assignments can light channels from
diagonal non-selected cells. The visible corpus already exposes this; per
`traps.md §VISIBLE-FAIL`, do not run truth/AB/profile/safety and do not build
rank-2 variants in this family.

**One remaining hypothesis:** only a different scoreable sub-dense encoding of
the `[30,3]` map could beat 90. Sparse initializer nnz=6 remains unscoreable
under scorer sanitize/check/strict shape inference, documented in
`sparse_init_undercount_and_reindex.md`; no scoreable sub-90 route is known.
