# Methodology — how a 400-task search was actually run

The technical problem (shrink an ONNX graph without breaking byte-exactness) is described in the [README](../README.md). This document covers the process layer: how many concurrent workers were kept from stepping on each other, and how a claim like *"this is the cost floor"* was made trustworthy.

It is written up because the controls below are the interesting part. Every one of them exists because the failure it prevents happened first.

## Why a process layer at all

400 independent tasks with a perfect oracle is an embarrassingly parallel search. The bottleneck is not compute, it is **trust**: a candidate that looks like a win but is subtly overfit, or priced with a cost model that does not match the grader, is worse than no candidate — it burns a submission slot and corrupts the record of what has already been ruled out.

So the pipeline is built around a single rule: **the agent that builds a candidate never ships it.**

## Roles

```
owner
  │
  ├─▶ Supervisor      scans project state, starts one Orchestrator per assigned task batch
  ├─▶ Prompter        picks high-expected-value tasks, writes <=2 KB worker prompts
  ├─▶ Orchestrator    assigns prompts, routes output, enforces parallel-write safety
  │     ├─▶ workers        build candidates (task-local files only)
  │     └─▶ Tough Reviewer attacks the result -> SHIP / KEEP-WORKING / DROP
  ├─▶ Submit Captain  vets SHIP candidates, batch-validates on its own stack, stages
  ├─▶ Janitor         clears scratch so workers never carry repo logistics
  └─▶ Librarian       integrates durable lessons into the shared technique library
```

Two Orchestrators may run concurrently, but only over **disjoint task sets** assigned by the owner, and the downstream writers (Submit Captain, Librarian) stay serialized — they are the points where independent lanes would otherwise collide. Coordination happens through role-local inbox folders, deliberately not through shared mutable state files.

Full playbooks: [`Agenters/README.md`](../Agenters/README.md).

## The gates

[`Onnx_Advanced/WORKER_PROTOCOL.md`](../Onnx_Advanced/WORKER_PROTOCOL.md) is the canonical worker contract — prompts point at it rather than restating it, and where a prompt and the protocol disagree, the protocol wins. Five gates must pass before a candidate can even be staged:

1. **Visible** — 100 % on the task's visible pairs. One visible miss implies a hidden zero; the candidate is abandoned, not tuned.
2. **Truth** — against the cached ARC-GEN corpus. Below 99 % and not byte-equal to the live solution, a candidate is a learning artifact, never a "best".
3. **A/B byte-equal vs. live** — the gold ship class. Divergences get attributed rather than waved away: if the candidate's misses are a strict subset of live's, it is *more* correct than live and is shippable.
4. **Static cost** — priced with the local scorer.
5. **Real-input profile** — the candidate is run over the real official inputs under ONNX Runtime profiling, and credited cost is taken as `max(static, runtime)` per tensor.

Gate 5 exists because of a specific, expensive discovery: **static cost is fiction the moment a graph has a dynamic shape, dummy branch, or optional input** — a 78× discrepancy between local price and credited price was observed. Any gap over 5 % between static and profiled cost demotes a candidate from "exact ship" to "credit probe".

Gates run cheapest-first and in the foreground, throttled so that concurrent heavy verification stays bounded. A gate that pauses is waiting its turn, not hung.

## Making "floor" mean something

The most common way to waste a day on this problem is to accept a floor claim that was never proven. The record-keeping rules push back:

- A floor proof must give **exact margins or a source read** — "tried hard" is not a floor.
- Every improvement rung is laddered, but each superseded rung compresses to **one table row** (`version | cost | delta | verdict`). Full prose survives only for the current best and the current floor proof.
- A retracted floor proof is **deleted, not stacked**, so the file cannot accumulate contradictory claims.
- Dead routes get a one-line record under `hard_records/` so the next batch does not re-run them.

Floors turned out to be formulation-relative even when the person claiming them had written the proof: one component described as "pinned at 1,620 B" in two separate written analyses fell by half on a third forced pass, to a transport-packing idea nobody had tried. Anti-quit prompts that force re-deriving a pinned component from scratch earned their cost.

## Moving knowledge between sessions

Agents do not edit the shared library directly — that is how a library drifts. A worker that finds something transferable drops a `.update/DONE_*.md` packet at session close; the Librarian integrates it into [`Onnx_Advanced/`](../Onnx_Advanced/INDEX.md) and assigns section numbers.

The library is queryable rather than readable end to end (`Onnx_Advanced/library_db/lib_query.py`: `search`, `op`, `facts --task`), because the technique catalog alone is 420 KB and no agent should be linear-reading it.

Context discipline is treated as a first-class constraint: `tools/agent_helpers/context_pack.py --task N` assembles a ~14 KB curated pack (state row, status, rule, don't-dos, prompt, staged candidates, prototype inventory, cited sections) so a worker starts from a bounded, current view instead of grepping a 1 MB CSV.

## Lessons about the process itself

Recorded in [`Logs/lessons.md`](../Logs/lessons.md), the durable ones:

- **A convention doc is a claim about behavior, not proof of behavior.** One directory's README described a live logging convention that had been dead for weeks while 413 files accumulated in direct violation of its own rule. Rules without enforcement decay, and they decay fastest under parallel agent pressure.
- **An empty load-bearing file is worse than an absent one.** A 0-byte telemetry file that a README promised was populated made agents trust a non-source. Keep the data with an honest freshness note, or delete the file and fix the pointer.
- **Auto-generated status snapshots go stale and lie.** The repo's `CURRENT_STATE.md` was a generated snapshot that went stale three times and was flagged by three separate audits. It is now a deliberate stub that routes to sources which are current by construction — a ledger, a per-task query tool — rather than a fourth attempt at a snapshot.
