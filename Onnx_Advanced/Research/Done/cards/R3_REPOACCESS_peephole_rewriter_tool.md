# Research Card R3 (REPO-ACCESS) — Build the sanctioned ONNX peephole rewriter (auto-harvest byte-equal cost cuts)

**Type**: repo-access research + tooling. **Deliverable**: a tool `tools/peephole_rewrite.py` + a report `Logs/research/R3_peephole_sweep.md` listing every task it can cut, with gated candidates staged for captain review. This is the K1-sanctioned fallback (`../Done/k1_superoptimization_survey.md`: "skip the superoptimizer ports; the sanctioned path is a tiny ONNX-native peephole over the exact-rule shortlist").

## Why this is high-value

We have ~91 library techniques and ~219 boladi-workable tasks. Many cost cuts are MECHANICAL and repeat across tasks: free-output Equal/Pad tail conversion (#67/#81), negpad-Conv crop fusion (#41), groups-halving (#84), the no-Mul scaled-conv decode (#90), einsum-separable-mask (#89), dtype-narrowing of intermediates where exact. A tool that applies these byte-equal-gated rewrites automatically can harvest wins across the whole workable set WITHOUT a per-task worker each — turning a 100-task grind into one supervised sweep.

## Read first
`../Done/k1_superoptimization_survey.md` (the mandate + the exact-rule shortlist), `techniques.md` (the rewrite rules to encode — start with the highest-frequency, lowest-risk ones), `tools/swap_and_submit.py` + `tools/agent_helpers/profile_real_cost.py` + the `ng_gate` interface (for gating), `traps.md` §FREE-OUTPUT BOUNDARY LEAK + §FREE-OUTPUT ACTIVE-MASK + INPUT-DTYPE-LOCK (the rewrites that need guards).

## Method (build incrementally, safest rewrites first)
1. **Pick the 3–5 highest-frequency, provably-safe rewrites** to encode first. Candidates, each as a pattern-match → rewrite → re-gate:
   - **Equal-tail / Pad-as-output conversion**: a graph ending `...->Equal/decode->output` where the canvas can be Pad-as-output (#67/#81); apply #81's pad-floor crossover (only when it pays) + the §ACTIVE-MASK / §BOUNDARY-LEAK guards.
   - **Intermediate dtype-narrowing**: f32 intermediate → u8/bool where values provably fit and no downstream op needs f32 (respect INPUT-DTYPE-LOCK: Conv/MatMul force f32 output — don't narrow those).
   - **negpad-Conv crop fusion** (#41): Slice+Cast → one negpad Conv.
   - **groups-halving** (#84) on a final placer/decode Conv where the channel partition holds.
   - **variadic Sum collapse** (#75): Add-chain → one Sum node.
2. **Every rewrite is byte-equal-gated**: the tool must run `ng_gate ab --full` (or equivalent byte-equal-vs-LIVE on the corpus) AND gate-5 real-input profile on the rewritten graph, and REJECT any rewrite that isn't byte-equal or doesn't actually lower the credited (profile) cost. No rewrite ships without passing.
3. **Sweep** the `boladi_workable=Y` set, apply the safe rewrites, stage every passing cut to `staged_submits/taskNNN_peephole_cost<C>/` + manifest, and report the full before/after table.

## Output
- `tools/peephole_rewrite.py` (idempotent; `--task N` single, `--sweep` all-workable, `--check` dry-run reporting potential cuts without staging).
- `Logs/research/R3_peephole_sweep.md`: per-task `task, rewrite(s) applied, old_cost, new_cost, byte_equal(Y/N), profile_gap, staged(Y/N)`.

## Discipline (this is a SAFETY-CRITICAL tool — it touches many graphs)
- **Byte-equal-vs-LIVE on the FULL corpus is mandatory before staging any rewrite** — a peephole that changes output on even one hidden-distribution pair is a zero-risk. No exceptions.
- **profile==credit**: re-profile every rewrite on real inputs; reject if the credited cost doesn't actually drop (a "static-cheaper" rewrite that profiles higher is a credit-probe, not a win).
- Start with the 3 safest rewrites; do NOT encode exotic/structure-dependent ones (einsum-separable-mask needs a separability check — leave for a v2).
- Captain ships from the staged dirs; the tool NEVER submits.
- End with a verdict: how many of the workable set got a mechanical cut, total point-gain harvested, and which rewrites paid off most.
