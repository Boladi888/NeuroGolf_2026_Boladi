# NeuroGolf

[![tests](https://github.com/Boladi888/NeuroGolf_2026_Boladi/actions/workflows/tests.yml/badge.svg)](https://github.com/Boladi888/NeuroGolf_2026_Boladi/actions/workflows/tests.yml)

Solving all 400 [ARC-AGI-1](https://arcprize.org/arc-agi) tasks as hand-optimized **ONNX graphs**, scored on how small the graph is.

This is the completed archive of my entry in the [2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) on Kaggle. The submission medium is not Python — it is a set of ONNX neural network graphs (opset 10) that must reproduce each task's output exactly. Points come from *shrinking* a correct graph, so the work is closer to code golf against a cost model than to conventional ML.

**Final Kaggle score: 7311.74** (confirmed 2026-06-21). The climb from the initial zero-score state is recorded row by row in [`Logs/submissions_ledger.csv`](Logs/submissions_ledger.csv).

## The problem, precisely

Each task gives input/output grid pairs. A submission must contain an ONNX graph per task that maps input to output **byte-exactly** — no tolerance, no partial credit for "close". The score rewards minimizing:

```
cost = memory (intermediate tensors) + params (initializers)
```

That combination is what makes the problem interesting. A solution that is obviously correct is usually not cheap, and the cheapest formulation is rarely the obvious one:

- A `Pad` to a 30×30 boolean canvas costs 900 B of intermediate memory, which puts a floor under *any* multi-node approach for that task — so sub-160 solutions have to be single-node, where cost collapses to params alone.
- `Where(bool, i8, i8)` has no ONNX Runtime kernel, so int8 lanes have to be routed through `uint8` with the cast placed before the select.
- Nibble-packing two grid cells per byte (kernel `[1,10,1,2]`, stride 2) halved a convolution that two written floor analyses had called pinned.

Findings like these accumulate in [`Onnx_Advanced/`](Onnx_Advanced/INDEX.md) as a routing index of reusable techniques, and in [`Logs/lessons.md`](Logs/lessons.md) as cross-session lessons.

The competition's cost metric also had exploitable gaps — wrapper families and data-derived-shape tricks that price differently at scoring time than they behave at runtime. This entry did not use them, by a stance recorded up front in [`.ai_instructions/03_PROJECT.md`](.ai_instructions/03_PROJECT.md) and held for the whole run: only mechanisms that produce the same answer at runtime as at scoring time.

## How correctness was enforced

Byte-exactness is the whole game, so verification is the load-bearing part of the toolchain.

`tools/ng_gate.py` is a shared verification harness that checks a candidate graph against a cached ARC-GEN ground-truth corpus — deterministic (`seed_offset=2026`), generated once and reused, up to 200,000 pairs per task. That gate is what makes aggressive rewriting safe: with a perfect oracle, a risky structural re-plumb becomes a mechanical refactor. One task went through four successive full rewrites (−45 % cost) in a single session, each one gated byte-equal in about two minutes.

The harness self-throttles to ~80 % CPU across concurrent agents, since several were running verification at once.

## What's in here

| Path | What it is |
|---|---|
| [`tools/`](tools/README.md) | The working toolchain — verification gate, cost profiler, ONNX optimizers, submission builder. `tools/build_submission.py` is a 19k-line, 311-function ARC detector library, exercised by `tests/test_detect.py`. |
| [`The_400_Tasks/`](The_400_Tasks/current_strategy.md) | One record per task: what was tried, what the cost floor turned out to be, why an approach was rejected. The bulk of the repo. |
| [`Onnx_Advanced/`](Onnx_Advanced/INDEX.md) | Durable ONNX technique library and routing index, plus the worker protocol for safe candidate development. |
| [`ml/ngs/`](ml/ngs/README.md) | Neural-Guided ONNX program Search — a program-synthesis engine that solved 23 tasks byte-exact against 512 held-out ARC-GEN pairs. |
| [`tests/`](tests/) | 1,253 tests across 37 modules — 1,247 pass locally, 6 skip without the archived team repo. On a bare checkout (no competition data) 615 still run, which is what CI executes. |
| [`Logs/`](Logs/README.md) | Result ledger and the durable cross-session lessons. |
| [`Agenters/`, `Handoff/`, `Prompts/`, `.ai_instructions/`](docs/METHODOLOGY.md) | The coordination layer this project actually ran on — see below. |
| [`submissions/`](submissions/README.md) | The final upload payload. |
| [`GUI/`](GUI/README.md) | A local viewer for inspecting task grids and candidate outputs. |

## How the work was organized

A 400-task search problem with a byte-exact oracle parallelizes well, and this project was run by multiple AI agent lanes working concurrently against a shared board, with a written protocol for how candidates get proposed, gated, and shipped. That coordination layer is documented in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and is a real part of the engineering, not an afterthought — the protocol exists because the failure modes it prevents (duplicated work, ungated ships, silently stale state) all happened first.

**On authorship, since the commit log makes it obvious anyway.** `git shortlog -sne` shows 691 of 1,182 commits (58 %) authored by `Hermes neurogolf-worker-7`, an agent worker identity, and 491 by me (under two spellings of the same account). That split is the point rather than a disclaimer: the deliverable here is a pipeline that could run that many gated changes without shipping a wrong answer, and the parts I own are the ones that determine whether such a pipeline produces anything trustworthy — the task batching, the gate design (including the real-input profiling gate that caught local cost being off by 78×), the record-keeping rules that make a floor claim falsifiable, the ethical-play stance, and every ship decision. Agents that build candidates never ship them.

## Running it

The repo is an archive, not a live workspace. Most tooling is here for study.

```bash
pip install -r requirements.txt   # numpy 2.4.4, onnx 1.21.0, onnxruntime 1.24.4, onnx-tool 1.0.1
python -m pytest tests -q
```

The competition data is not vendored (it is ~670 MB and re-pullable):

```bash
kaggle competitions download -c neurogolf-2026 -p data/competitions/neurogolf-2026
```

Layout details, including the machine-local stores this ran against, are in [`WHERE_DATA_LIVES.md`](WHERE_DATA_LIVES.md). Tooling carries absolute Windows paths from the machine it ran on; they are left as-is, since this is a record of how the project actually ran rather than a portable distribution.

## Reading order

1. [`Onnx_Advanced/INDEX.md`](Onnx_Advanced/INDEX.md) — the technique library, and the best single view of the technical work.
2. [`Logs/lessons.md`](Logs/lessons.md) — what transferred across sessions.
3. [`The_400_Tasks/current_strategy.md`](The_400_Tasks/current_strategy.md) — strategy and ship-decision reference.
4. [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — how the multi-agent process worked.
5. [`AGENT_TASK_START.md`](AGENT_TASK_START.md) — the 20-line router a single-task session started from.

## License

Apache-2.0 for this repository's original content — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Third-party Kaggle notebooks with unknown license metadata were removed during archive cleanup; the retained analytical survey is in [`Competitor_Notebooks/README.md`](Competitor_Notebooks/README.md). `NOTICE` carries onnx-tool's MIT attribution for the vendored reference under `tools/onnx_tool_1.0.1_reference/`.
