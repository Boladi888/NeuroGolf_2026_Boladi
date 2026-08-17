# ng_gate — shared NeuroGolf verification harness

One tool replaces the hand-rolled per-task `ab_gate` / `synth_gate` / `final_battery` /
`argen_verify` scripts. Orphan-safe, CPU-throttled, corpus-cached.

```
py tools/ng_gate.py <mode> --task NNN --cand cand.onnx [opts]
```
(Prefer the pinned venv: `${NEUROGOLF_PYTHON} tools/ng_gate.py ...` — see `../../WHERE_DATA_LIVES.md`.)

## Modes
| Mode | What it does |
|---|---|
| `visible` | candidate vs the 266 official `taskNNN.json` pairs (vs expected) |
| `truth` | candidate vs fresh ARC-GEN **ground truth** (cached corpus). `--ref V.onnx` also reports cand-vs-variant agreement |
| `ab` | candidate **byte-equal** vs a reference onnx. Auto-resolves the parent (`models/taskNNN.onnx` → `submission.zip` → newest staged); override with `--ref`. `--early-exit` stops at first divergence |
| `synth` | candidate vs an independent numpy ref: `--synth-ref ref.py` exposing `solve(grid)->grid` |
| `adversarial` | candidate vs LIVE on perturbed inputs (wraps `tools/adversarial_test.py`) |
| `battery` | visible + truth + ab + score → `PASS`/`REVIEW` + manifest-ready JSON (`--json`/`--out`) |
| `fill-corpus` | pre-generate the corpus: `--task N` or `--tasks 1-400` / `--tasks 18,110,324`, `--n 200000` |
| `corpus-info` | show cached corpus size per task |

## Common options
`--n N` corpus pairs (default 2000) · `--full` use the 200k cap · `--ref PATH` · `--synth-ref PATH` ·
`--early-exit` · `--no-sanitize` · `--no-throttle` · `--new-window` · `--json` · `--out file.json` · `--verbose`

## Examples
```
py tools/ng_gate.py battery --task 324 --cand cand.onnx --n 5000 --json
py tools/ng_gate.py ab      --task 324 --cand cand.onnx --early-exit         # byte-equal vs live?
py tools/ng_gate.py truth   --task 324 --cand cand.onnx --ref old.onnx --n 20000
py tools/ng_gate.py fill-corpus --tasks 1-400 --n 200000 --new-window        # one-time full corpus
```

## Why it's efficient / safe (vs the old per-task scripts)
- **Single-threaded ORT sessions** (`session.py`) — no all-core oversubscription when many agents run.
- **Self-throttle to ~80% CPU** across all agents (`throttle.py`; crash-safe PID-keyed slots).
- **Cached deterministic corpus** on the Q: drive (`corpus.py`) — generate once, every agent reuses; no per-run regeneration.
- **Kill-on-close JobObject** (`killjob.py`) — closing the `--new-window` console (or any launcher death) kills the whole worker tree. No orphans (the task018 incident can't recur).

## Modules
`killjob` · `session` · `encode` · `registry` · `corpus` · `throttle` · `modes` — see `PLAN.md`.

## Tests
`tests/test_ng_harness.py` (units) and `tests/test_ng_killjob_orphan.py` (orphan-kill contract).
Run: `${NEUROGOLF_PYTHON} -m pytest tests/test_ng_harness.py tests/test_ng_killjob_orphan.py -q`
(also pass under `python -m unittest`).
