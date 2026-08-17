# ngs long-run harness — operator guide

Resumable, watchable, checkpointed search over the task bucket, with a solver cascade that grows from
the fast DSL beam search → the zero-param Einsum cost-0 search → the GPU-batched discrete search. Built
for "get it ready for a long run I can steer." Everything writes to `ml/ngs/_runs/<name>/`.

## TL;DR — launch, watch, resume

```powershell
# fast passes over all 400 (DSL beam + zero-param Einsum). ~10 min. resumable.
.venv\Scripts\python.exe -m ml.ngs.longrun --tasks 1-400 --einsum --name baseline

# the hard bucket (team cost > 100 = 330 tasks) with a per-task time budget (the long run):
.venv\Scripts\python.exe -m ml.ngs.longrun --tasks hard --einsum --budget 90 --name hardrun

# resume after a kill / window-close (skips finished tasks): re-run the SAME command (default resumes).
# watch a running job from a second window:
.venv\Scripts\python.exe -m ml.ngs.longrun --watch --name hardrun
```

Visible-console launchers (VizTerms pattern — a window you watch and CLOSE to stop; resumable):
`ml\ngs\longrun.bat --tasks hard --einsum --budget 90 --name hardrun` and `ml\ngs\watch.bat hardrun`.

## Task selectors (`--tasks`)
`1-400` / `all` · `hard` (team cost > 100, 330 tasks) · `cheap` (≤ 100, 71 tasks) · `12-40,55,60` lists.

## What it writes (a monitor watches ARTIFACTS, never a PID)
- `status.json` — rewritten every task: `state` (running|complete|interrupted|crashed), `heartbeat_utc`,
  `done/total`, `solved`, `wins` (cheaper than team best), `per_task_sec`, `eta_human`. A stale heartbeat
  (> a few min) with `state:running` means the process was hard-killed — resume is still safe.
- `results.jsonl` — one line per finished task (cost, points, label, method, `team_cost`, `ngs_win`,
  argen p/t). This is the resume ledger: a restart skips every tid already here.
- `done.json` — terminal sentinel written on complete/interrupt/crash.
- live console dashboard (progress bar + ETA) streamed to stdout (the visible window).

## Solver cascade (cheapest-first, per task, under `--budget` seconds)
1. **beam** (`search.beam_search`) — the DSL program search (ColorRemap, CoordRemap, Transpose, …),
   arguments inferred from ~48 ARC-GEN pairs, each solve confirmed on 512 held-out pairs (the truth gate).
2. **einsum** (`einsum_search.solve_einsum`, enabled by `--einsum`) — the **cost-0 method**: enumerate a
   grammar of zero-parameter Einsum equations (operands are copies of the input one-hot → no params →
   cost 0), numpy-prefilter on ARC-GEN, confirm the winner. Recovers task067's `abcd,aedc->abcd` at cost 0.
3. **batched** (`batched.py`, engine ready — see below) — the GPU-batched discrete search for the hard
   bucket: sweep a large mixed-radix space of small operand tensors on the B70, keep the top-k, ONNX-verify
   survivors. This is the stage that makes the run *long*.

The driver keeps the **lowest-cost argen-confirmed** result across stages and flags `ngs_win` when it
beats the team's current best cost (already true for task179/241: cost 0 vs the team's 1).

## The cost-0 / cost-1 picture (measured, honest)
`points = max(1, 25 − ln(cost))`, so **cost 0 and cost 1 both score the 25-pt ceiling.** From the team's
current bests: only **3** tasks sit at cost ≤ 1 (task067=0 zero-param Einsum, task179/241 Transpose). Strict
cost-0/1 is a *small* set by construction — a cost-0 graph must have zero initializers and no intermediate
memory, i.e. a single op writing `output`: `Identity` / `Transpose` / a zero-param `Einsum`. The v1
zero-param Einsum grammar (36 equations) solves 3 tasks at cost 0; **growing that grammar (more operands /
label patterns) is the direct lever for more cost-0 tasks** and is a natural long-run job. The larger prize
is *cheap* (cost ≤ ~30–100) via the batched factor search over the 330-task hard bucket.

## Batched discrete search — status & the template interface
`batched.py` is the **engine + plug interface**, verified on the B70 (torch-xpu) at **22–65 M candidates/s**
(finds the exact-fit candidate in a synthetic low-rank selftest). Enable the stage with `--batched`; it
runs every object in `batched.TEMPLATES` and is a safe no-op while that list is empty.

A **template** is three parts (register one by appending to `TEMPLATES`):
```python
class LowRankEinsum:
    name = "lowrank_einsum"
    def propose(self, tid, X, Y):        # X,Y: [n,10,30,30] argen input/expected one-hot
        # return (DiscreteDomain(cardinalities), score_fn, meta) or None
        # score_fn(choices)->residual[chunk] (lower=better); e.g. batched ridge-fit + sign-mismatch
        ...
    def emit(self, choice, meta):        # winning candidate -> onnx.ModelProto (a cheap graph)
        ...
```
`solve_batched` sweeps the domain on the B70, keeps the top-k, and **argen-verifies** each survivor
(only exact ones count) — GPU proposes, the truth gate disposes. The first template to build is a
**low-rank Einsum factor fit** (domain = discretized factor entries; score = ridge-fit residual;
emit = `Einsum` + factor inits) targeting the team's task380=66 / task108=300 shape. `demo_lowrank_scorer`
is the batched-kernel shape to start from. This is the piece to steer (which cheap-op family first).

## Time estimate for a full long run
- fast passes (beam + einsum) over all 400: **~10 min**, dominated by ARC-GEN corpus loads.
- batched over the 330 hard tasks: `330 × per-task budget`. At the B70's throughput a per-task budget of
  60–120 s (tens to hundreds of millions of candidates + survivor ONNX verification) puts a full pass at
  **~5–11 hours** — resumable, so it runs overnight and survives interruption. Knob: `--budget`.

Run in the SD venv (`Q:\AIz\SD\venv`) when the batched stage is active — that's the env with torch-xpu on
the B70. The beam+einsum stages run in the project `.venv` (CPU ONNX); the batched engine falls back to
numpy there, so everything imports and runs anywhere — only batched *throughput* needs the GPU.
