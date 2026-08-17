# ngs — Neural-Guided ONNX program Search for NeuroGolf

A program-synthesis engine that **composes ONNX ops into tiny, exact, low-cost graphs**,
guided by a learned policy and scored by the *real* NeuroGolf gates + `memory+params` cost.

## Why this can work where "train a net and ship it" can't

Shipping a trained network loses NeuroGolf head-on: `points = max(1, 25 − ln(memory+params))`,
so a net's parameter count is the enemy (a toy 3-layer MLP already costs ~27k → ~14.8 pts;
real nets sit near the 1-pt floor), and gradient descent yields *approximate float* outputs that
fail the byte-exact gates. **NGS flips it:** the ML is the *search policy*, and the *artifact* is a
hand-crafted-style exact graph (the same thing the human cost-golfers ship). ML **discovers**; the
search **embodies** — so the output is exact + tiny by construction, and the policy just makes the
search tractable.

This is the "option 2 / neural-guided program search" bet. It is a **research bet**: the existing
hand-craft + spread + multi-agent pipeline already wins, so this earns its keep only if guided
search finds graphs faster/cheaper than humans on some task families. The point of this package is
to make that testable, incrementally.

## Architecture (data flow)

```
tasks.py      load task example pairs (+ synthetic identity/transpose/rot180/mirror2x2 targets)
argen.py      read BoBrain's ARC-GEN corpus (≤200k pairs/task) — robust arg-inference + the TRUTH GATE
   │
ops.py        the DSL: op vocabulary (each op -> ONNX node(s), shape rules)   ── widen this to grow reach
              incl. data-aware ops: ColorRemap (recolor Gather) + CoordRemap (separable pixel remap)
builder.py    program (op list) -> valid NeuroGolf ONNX (named input/output, static, no banned ops)
score.py      cost (canonical tools/_score_v2, fallback static) + correctness (official encode→ORT→>0→verify)
              + argen_gate(): batched re-verify a solve on N held-out ARC-GEN pairs (the truth gate)
   │
search.py     beam search over programs, ranked by (solved-fraction, then cost); data-aware ops infer
   │            from argen pairs, and a solve is recorded only if it passes the argen truth gate
   │            └─ policy hook: orders/prunes op expansions per node  ◄────────┐
policy.py     HeuristicPolicy (no training) │ PolicyModel (torch MLP)  ─────────┘  the "neural" layer
mine.py       401 winning ONNXes → op-frequency map + imitation corpus  (what to add to the DSL next)
train.py      search-bootstrap imitation → trains PolicyModel (auto-targets the B70 torch-xpu env)
run.py        CLI: selftest | task <id> | sweep <a> <b>   [--policy heuristic --top-k K --beam --depth]
```

Everything lives under `ml/` only. The search substrate runs in the project `.venv`
(onnx/onnxruntime/numpy); `PolicyModel`/`train.py` need torch (`.venv-ml`, see `ml/README.md`).

## Status (proven vs scaffolded) — honest

**Headline (2026-07-10): 1 → 23 argen-confirmed real solves across tasks 1–400.**
Every solve is re-verified byte-exact on **512 held-out ARC-GEN pairs** (0 overfits rejected). By op:
`colorremap` ×4 (t16/276/309/337, cost 10), `transpose` ×2 (t179/241, cost 0), a zero-param `einsum`
×1 (t67, cost 0), and `coordremap` ×16 (t83/87/108/116/135/140/142/152/164/172/210/211/223/311/380/385).
The engine as-shipped had this reach latent — it was never swept past ~task 60 and had no truth gate.
Reproduce: `python -m ml.ngs.run sweep 1 400 --depth 2 --policy heuristic` (add `--einsum` via
`ml.ngs.longrun` for the zero-param Einsum stage).

`coordremap` now emits the **cost-optimal** graph per case (2026-07-10): single-axis mirror/shift →
one canvas `Gather(idx[30])` = **cost 30** (ties the team's task116/164/172/210/385); flip/rot180/crop →
one 2-axis `Slice` = **376** (was 3974); mirror-tile → crop+`Gather` = **~3390** (was ~9142). See the
three emission cases in `ops.py::CoordRemap`.

**Proven (runs today in `.venv`, tested):**
- **ARC-GEN truth gate + robust arg-inference (`argen.py`, `score.argen_gate`)** — the big lever. BoBrain
  caches ≤200k deterministic ARC-GEN pairs per task (all 400); ml/ngs now (1) infers data-aware op
  arguments from ~48 of them (not the ~5 visible → no fragile coincidences) and (2) records a solve only
  if it also reproduces **512 held-out pairs** (batched single-session ORT run). This is the "arc-gen /
  truth-full gate" this README used to only ask for. Read-only (`np.load`, never extends the cache).
- **Data-aware DSL — the two ops that carry the reach:**
  - `ColorRemap`: infers the channel permutation → one `Gather(axis=1)` (**cost 10**). Size-agnostic
    (works for any grid H×W) → solves t16/276/309/337.
  - `CoordRemap` (NEW): infers a **separable per-axis coordinate map** from the pairs (fingerprint each
    cell by its value-vector across pairs; unique cells pin `idx_r`/`idx_c`, background wildcards are
    skipped) and emits `Gather(axis=2)→Gather(axis=3)→Pad-to-canvas` (+ a leading `Transpose` for
    axis-swaps). One op covers the whole flip / rot180 / rot90 / crop / mirror-tile family for
    constant-geometry tasks — it found **16 solves, 9 more than a hand-coded transform catalog predicted**
    (it generalizes to separable rearrangements the catalog didn't enumerate). Output is re-embedded
    top-left in `[1,10,30,30]` (a `Slice`-as-output is grader-dead — Onnx_Advanced task310).
- **Policy fix**: `HeuristicPolicy` now ranks `colorremap`/`coordremap` by task features. Previously it gave
  `colorremap` a **zero** prior, so `--top-k` pruning dropped the one op that solves real tasks (a sweep
  with `--top-k 4` scored 0 on task016 while a bare `task 16` solved it). Fixed + covered by selftest.
- Substrate end-to-end: synthetic identity/transpose/colorswap/**rot180**/**mirror2x2** all solve
  (selftest green under both plain and `--policy heuristic --top-k 4`), cost reported **exact**.
- **DAG search + heavy ops**: DSL is branching-graph-capable (GState + dead-node elim) with `Conv`,
  `Add/Sub/Mul/Max/Min/Equal`, `Where`. Explores DAGs but the reach lever remains **arg-inference**.
- **B70 GPU (`backends.py`, `train.py --auto`)**: `train.py --auto` re-execs into the discovered B70 env
  (`Q:/AIz/SD/venv`, torch 2.11+xpu, ort 1.25) and trains the PolicyModel over the enlarged vocab —
  verified. Honest caveat: today's solves are depth-1, so the policy-learning problem (and the argen gate)
  are trivial on CPU; the GPU's real payoff is future (see Roadmap: batched discrete arg-search).

**Scaffolded (real code, needs more to be strong):**
- `PolicyModel` + `train.py`: correct torch loop, auto-targets the B70; but today's real solves are all
  depth-1 single ops, so imitation demos are near-trivial and the learned policy adds little over the
  heuristic yet. Its value scales with program depth (see Roadmap: spatial/conditional ops → deeper programs).
- The DSL is 19 ops but only **two** carry the real-task reach (`ColorRemap`, `CoordRemap`); the rest are
  substrate/structure. Reach now grows by adding more *arg-inferring* ops (Roadmap §1), not more structure.

## Coverage + the data-driven roadmap (from `mine.py`)

Only **14/401** winning graphs are expressible with the current op-set. Highest-value ops to add
next (each appears in this many winning graphs — implement in `ops.py`, ranked):

| add op | unlocks up to | | add op | unlocks up to |
|---|---|---|---|---|
| `Pad`   | 323 | | `Mul`       | 199 |
| `Slice` | 272 | | `Add`       | 183 |
| `Where` | 227 | | `ReduceMax` | 173 |
| `Sub`   | 222 | | `Concat`    | 164 |
| `Equal` | 210 | | `Conv`      | 161 |

(Most are cheap shape/elementwise ops — the sub-1k-cost sweet spot. `Conv`/`Einsum` are the
higher-reach, higher-care additions.)

## Run it

```bash
# substrate proof (project venv)
.venv/Scripts/python.exe -m ml.ngs.run selftest
.venv/Scripts/python.exe -m ml.ngs.run selftest --policy heuristic --top-k 2   # guided

# search a real task / sweep a range — real-task runs auto-infer op args from the ARC-GEN corpus and
# confirm each solve on 512 held-out ARC-GEN pairs (reported as "[argen p/t]"). Point the reader at a
# different corpus with the NEUROGOLF_ARGEN_CACHE env var; if the corpus is absent it degrades to visible.
.venv/Scripts/python.exe -m ml.ngs.run task 87
.venv/Scripts/python.exe -m ml.ngs.run sweep 1 400 --depth 2 --policy heuristic   # 22 argen-confirmed solves

# mine the winning corpus (op-frequency + roadmap + imitation.jsonl)
.venv/Scripts/python.exe -m ml.ngs.mine

# discover which scattered env runs ONNX/torch on the B70 (writes _env/backends.json)
.venv/Scripts/python.exe -m ml.ngs.backends

# train the policy ON THE INTEL ARC B70 PRO — auto-targets the discovered torch-xpu env
.venv/Scripts/python.exe -m ml.ngs.train --auto

# search-throughput bench (project-venv CPU baseline)
.venv/Scripts/python.exe -m ml.ngs.bench --n 200

# OpenVINO B70 inference bench (builds graphs here, scores them on the B70 dGPU via the OpenVINO env)
.venv/Scripts/python.exe -m ml.ngs.bench --ov --n 16
```

## Intel Arc B70 Pro + scattered envs (`backends.py`)

`backends.py` discovers the machine's scattered Python envs (`Q:/AIz/*`, the project venv, …) and
probes each — in its own subprocess (different GPU runtimes can't share a process) — for ONNX/torch
GPU capability, writing `_env/backends.json`. On this host it found (B70 = `Intel(R) Arc(TM) Pro B70
Graphics`, Xe2 Battlemage, 32 GB):

| env | stack | role |
|---|---|---|
| `Q:/AIz/SD/venv` | onnx + onnxruntime 1.25 + **torch 2.11+xpu (B70)** | **one-stop ngs env** — search AND policy training on the B70 |
| `Q:/AIz/ComfyUI/venv` | torch 2.11+xpu (B70) | torch-xpu only (no onnxruntime) |
| `Q:/AIz/LLM/OV/venvs/openvino` | OpenVINO 2026.2 sees the **B70 dGPU** | future ONNX-on-GPU via OpenVINO |
| project `.venv` | onnxruntime 1.24 (CPU EP only) | where the search runs today (CPU) |

**What the B70 actually buys (honest, measured):**
- **Policy training — real win, works now.** `train.py --auto` re-execs in the SD venv and trains the
  PolicyModel on the B70 (torch-xpu). *Verified: trains + saves `_corpus/policy.pt`.*
- **ONNX inference on the B70 — works, but NOT worth it for today's graphs.** `ov_eval.py` runs the
  search's candidate graphs on the B70 via OpenVINO (`bench.py --ov` auto-bridges: build here, score on
  the GPU there; it resolves the **B70 dGPU by name** because the default `GPU` is `GPU.0` = the iGPU).
  Measured on tiny shape-only graphs: **onnxruntime-CPU ~570/s · OpenVINO-CPU ~77/s · OpenVINO-B70 ~4/s.**
  Per-model GPU *compile* overhead dominates (each tiny graph = a fresh kernel build), so the GPU is
  ~140× slower here. It flips only when graphs get heavy (Conv/Einsum) or a compiled model is reused
  across many inputs (batched eval). The path is built and verified (0 compile/run errors on the B70);
  `score.py` is also ready for an `onnxruntime-openvino` EP if one is ever installed. **Don't GPU the
  search until the DSL adds heavy ops.**

(This supersedes the stale A770 / `.venv-ml` assumptions in `ml/README.md` for ngs — `backends.py`
discovers the real envs instead of hardcoding them.)

## Key finding (2026-07-10): the arg-inference lever is CONFIRMED — and it wants MANY pairs + a truth gate

The 2026-06-21 diagnosis ("more *arg-inference*, not more structure") is now demonstrated: adding **one**
arg-inferring op (`CoordRemap`) took real solves from 6 → 22, while the structural heavy ops still add none.
Two refinements this session sharpened:
- **Size-agnostic beats geometry-dependent.** The ops that generalize are the ones whose operation is
  independent of the (variable) grid H×W — channel `Gather` (recolor) and canvas `Transpose` work for any
  size. Spatial rearrangements (flip/crop/mirror) are only expressible when the geometry is *constant*
  across the task's pairs (a static index vector can't track a grid whose size varies) — which is exactly
  the const-geometry regime `CoordRemap` targets.
- **Inference needs many pairs; solves need a truth gate.** Inferring an argument from ~5 visible pairs is
  fragile and a visible-pair "solve" can be a coincidence. The ARC-GEN corpus fixes both: infer from ~48
  pairs, then gate the solve on 512 held-out pairs. (This session: 0 of 22 were overfits — but the DSL is
  still only global transforms; once spatial/conditional ops land, the gate is what keeps them honest.)

## Roadmap (re-prioritized)

1. **Spatial / conditional arg-inference — the next reach frontier.** The big unsolved buckets are
   `same_other` (~251 tasks: same-shape non-global rules) and `smaller_other` (~101). These need ops that
   infer a *local* rule from the pairs: neighbour-conditioned recolor (trigger colour + offset set → write
   colour, à la the teammate's `arc_einsum_search`), bounding-box crop (runtime bbox via `ReduceMax`/
   `ArgMax` → dynamic `Slice→Pad`), and small free-output `Einsum` equations. Each tends to unlock a family.
2. **GPU-batched discrete arg-search (the teammate's `gpu_batch_domain`).** When an op's argument lives in a
   large discrete space (Einsum labels, Conv kernels, factor columns), enumerate mixed-radix candidate IDs,
   decode+score whole chunks as tensors on the B70, keep the global top-k, then ONNX-verify survivors. This
   is where the free B70 finally earns its keep (today's depth-1 solves don't need it).
3. **Decision-equivalence cost golf (also from the teammate).** `score.evaluate` already decides on `>0`, so
   a solve may keep its cost down by accepting logit-sloppy graphs that stay decision-exact (certify with a
   positive winner-margin). Lets dtype-shrink / factor-tie rewrites lower cost without breaking the gate.
4. **Train `PolicyModel` on the argen solve-pool** once programs get deeper than one op; move beam → MCTS as
   the branching factor grows. Honest bake-off vs the human/spread pipeline on a held-out slice.

## Caveats baked in
- Correctness uses the **official** encode→`>0`→`array_equal` path. A "solve" is byte-equal on the visible
  pairs **and** on 512 held-out ARC-GEN pairs (the `argen_gate`). That is a strong signal, not a proof:
  ARC-GEN samples may still not cover a rare hidden config, and the gate needs BoBrain's corpus present —
  `sweep` output without an `[argen p/t]` tag means the corpus was missing and only the visible pairs gated.
- `CoordRemap` is **const-geometry only** — it returns None when a task's grid size varies pair-to-pair
  (a static index vector can't track it). Variable-geometry flips/tiles (e.g. t150/t155/t249) are correctly
  left unsolved rather than solved wrongly; they need runtime-index ops (a Roadmap item).
- `GatherReverse` (full-canvas flip) intentionally fails on sub-canvas grids — the env catches it,
  which is the correct behavior (the top-left-embedding trap from core.md).
