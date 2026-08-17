# synth_oracle_strategies.md — evicted content: executive summary, attachment-scope notes, generator survey prose, coverage-driven + dimension-stress sections (snippets S2/S3/S4/S6), adversarial prose, diagnostic prose, decision matrix, checklist, methodology/assumptions, and the verbatim A3 assignment card.
Evicted 2026-06-10 (refactor pass 4). Live distillate: `../synth_oracle_strategies.md`; full pre-refactor snapshot: `synth_oracle_strategies_pre_refactor_2026-06-10.md`.
Original headings and order preserved below; content kept live (4-axis fidelity rubric, self-consistency trap + mitigation, generator comparison table, snippets S1/S5, diagnostic flowchart + symptom table) is omitted here.

## Executive Summary

The attachment does **not** ask for a broad literature review. It narrows the deliverable to a concrete, implementation-grade research note that answers six specific questions in a fixed order, with emphasis on a **generator capability matrix**, **coverage-driven input generation**, **dimension-stress recipes**, **adversarial search algorithms**, and a **reformatted treatment** of oracle independence and postmortem diagnostics. It explicitly says to stop doing meta-critique, state assumptions if needed, and proceed.

The highest-confidence conclusion is that **there is no single best synthetic generator for ARC-style validation**. Instead, the strongest practical stack is layered. For **task-local fidelity**, use **RE-ARC** when you want high-confidence resampling of the original 400 ARC-AGI-1 training tasks; for **benchmark-mimetic breadth**, use **ARC-GEN**, which validates all 400 ARC-AGI-1 generators and has since added support for 500 ARC-AGI-2 tasks; for **cross-example consistency**, borrow **ARC-TGI’s** task-level-constraint design; for **large pretraining corpora**, use **BARC** or **NVARC** artifacts. For search, **Hypothesis** is the most practical structured generator under a CPU budget, **coverage.py** is the most practical branch-measurement tool, **Atheris** is useful only after you define a grid-aware serializer and preferably a custom mutator, and **mutmut** is best used to harden property bundles and oracle logic rather than to generate grids.

“Passed synth, failed test” is best understood as a **distribution mismatch** across at least one of four reconstructed fidelity axes: **rule fidelity**, **dimension coverage**, **distribution realism**, and **oracle independence**. That diagnosis is strongly supported by official ARC work showing that static task sets are vulnerable to overfitting and leakage, and by the sharp drop of strong ARC-AGI-1 systems on ARC-AGI-2. ARC-TGI explicitly argues that independent per-example sampling can fail to expose the underlying rule, while ARC-AGI-2’s official results show that methods tuned to ARC-AGI-1 transferred poorly to the harder benchmark.

The most useful operational recommendation is therefore a **three-layer validation protocol**: first, generate structurally valid and branch-diverse inputs; second, validate with an **independent oracle bundle** made of at least one external/public derivation plus metamorphic/property checks; third, run **adversarial counterexample mining** with disagreement fitness and boundary pressure. That stack is more robust than “1000 random samples and 1000/1000 matches,” because it directly targets hidden dimensions, ambiguity across examples, and circular oracles. This three-layer protocol is a reconstruction from the attachment’s summary rather than a verbatim copy from the missing partial audit, and I flag that assumption explicitly below.

## Attachment Scope and Extracted Research Request

The attachment asks for a **single copyable Markdown research note** that answers, in order: the failure mode, a survey of published generators, coverage-driven approaches, a dimension-stress framework, adversarial techniques, oracle-independence techniques, a diagnostic sequence for “passed synth, failed test,” a decision matrix, and a verification checklist. It also narrows fresh research to **Q1–Q4**, while saying **Q5–Q6 should be reformatted rather than re-researched**, and it requires specific table schemas for the generator survey, dimension-stress families, and diagnostics.

The attachment also corrects an important source-list error: there is **no “Hodel ARC-GEN.”** Michael Hodel’s public generator is **RE-ARC**, while **ARC-GEN** is Michael Moffitt’s separate Google project. It further names **BARC**, **ARC-TGI**, **ARC Prize 2024/2025 official reports**, **ARChitects**, **NVARC**, **Hypothesis**, **Atheris**, **coverage.py**, and **mutmut** as the key sources to use.

## Findings

**Problem statement and the synth-distribution-mismatch failure mode.**
The core problem is not “how to generate many valid inputs,” but “how to generate a synthetic **input distribution** that stresses the same latent dimensions as the held-out task family.” ARC-GEN frames the dataset constraint as one of **modest cardinality** for methods that need more within-task examples. ARC-TGI goes further and argues that progress is hard to measure on static hand-authored collections because of **overfitting, leakage, and memorization**, and that **task-level constraints** are needed because independently sampling examples often fails to expose the underlying rule. The official ARC-AGI-2 launch also demonstrates how apparent progress on ARC-AGI-1 did not transfer cleanly: the 2024 ARChitects solution posted about **53.5% on ARC-AGI-1** but only about **2.5–3% on ARC-AGI-2**, which is exactly what a benchmark-mismatch failure looks like in practice.

My working reconstruction of the attachment’s implied fidelity rubric is a four-axis check, stated here as an assumption because the named partial-audit file was not attached. The four axes are: **rule fidelity** (samples really preserve the same latent transformation), **dimension coverage** (all causally relevant axes are stressed), **distribution realism** (sizes/layouts/object statistics resemble the intended held-out distribution), and **oracle independence** (the pass/fail test is not merely echoing the same mistaken interpretation as the candidate). This reconstruction is consistent with the attachment’s emphasis on fidelity, the self-consistency trap, and the “passed synth, failed test” diagnostic.

**Survey of published generators.**
The practical generator landscape separates into three categories. **RE-ARC** and **ARC-GEN** are the strongest public choices when the goal is **validation-grade resampling** of known ARC families. **ARC-TGI** is the strongest public choice when the failure mode involves **cross-example ambiguity** and you want generators that preserve episode-level inferability. **BARC** and **NVARC** are strongest when the goal is **large-scale synthetic corpora** for training or heavy augmentation, not when the goal is a lightweight, benchmark-faithful local validator. Winner systems such as **ARChitects** and **NVARC** did release solution repos and writeups, but those are better viewed as **competition pipelines** than as polished general-purpose generator libraries. Augmentation-oriented projects such as **AugARC** are useful but are materially weaker for generalization validation because they often perturb the presentation rather than expanding the semantic support of the task family.

RE-ARC’s public repo states that it provides a generator for **each of the 400 training tasks**, includes **1000 verified generated examples per task**, and uses corresponding **verifier programs** to keep only valid generated examples; the repo is marked **MIT**, and the latest captured commit is **2025-02-24**. ARC-GEN’s repo describes itself as an **exhaustive and mimetic** procedural generator, shows that `validate` passes all **400 ARC-AGI-1 generators**, is marked **Apache-2.0**, and the captured commit history shows updates through **2026-04-18**, including the addition of **500 ARC-AGI-2 tasks** on **2026-03-25**.

BARC’s public repo says it contains **162 manually written seed solutions** and a three-stage pipeline where the system generates natural-language descriptions, generates solution and input-generator code with LLM assistance, and finally executes those programs to produce concrete ARC problems. The accessible repo page did **not** surface a declared license, and its captured commit history runs through **2024-11-20**. Public first-party Hugging Face assets clearly expose a **200k “heavy” dataset** and multiple **100k synthetic datasets**; later survey sources describe BARC more broadly as **ARC-Heavy plus ARC-Potpourri with 200k and 400k tasks**, but that exact 400k figure was not directly exposed on the first-party pages I captured, so I treat it as a plausible but not fully confirmed summary rather than a primary-source certainty.

ARC-TGI’s paper and public repo are unusually important for local validation design because they make **cross-example constraints** an explicit first-class concern. The authors release **461 generators** covering **180 ARC-Mini**, **215 ARC-AGI-1**, and **66 ARC-AGI-2** tasks, say the generators are **human refined and locally verified**, and release the project under **Apache-2.0**; the captured repo history reaches **2026-02-16**. That combination makes ARC-TGI the best public reference for preventing the specific failure mode where independently valid train examples are jointly too weak or too ambiguous to force the intended rule.

The Kaggle winners matter mostly as **methodological signals**. ARChitects’ 2025 technical report says their new system trained on **ReARC, ARC Gen 100k, ARC 1 and 2 train/eval, ARC-Heavy, and ConceptARC**, but also says their best synthetic generation effort did **not scale**, with only about **1 in 50 to 100** generations looking novel/useful enough and a curated set of **150 additional tasks** not giving enough gain to justify pursuing that direction. NVARC’s public repo describes a **multi-stage synthetic data generation pipeline**, releases **103k synthetic puzzles** and **3.2M augmented puzzles**, and the official ARC Prize 2025 report says the winning entry leaned heavily on synthetic data. The takeaway is that winner pipelines are ideal for studying **what worked in competition systems**, but they are not yet the cleanest public templates for lightweight CPU-side validation.

For completeness, not every ARC-adjacent dataset is a generator. **ConceptARC** reorganizes tasks into concept groups, and **1D-ARC** simplifies ARC into a one-dimensional benchmark more conducive to LLM reasoning; both are useful diagnostically, but neither is a general 2D task-family generator of the kind needed for local synthetic validation.

**Coverage-driven approaches.**
For structured grid inputs, the most practical ranking is: **Hypothesis first**, **coverage.py second**, **manual branch tags plus greedy corpus selection third**, **Atheris fourth**, and **mutmut as a validator hardening tool rather than a generator**. Hypothesis is practical because it already speaks in terms of strategies, examples, shrinking, and targeted testing; coverage.py is practical because it tells you which branches were missed; Atheris becomes practical only after you serialize grids and, ideally, add a **custom mutator** so the fuzzer preserves grid structure; mutmut is practical for checking whether your property bundle is too weak, especially because it can rerun only affected tests and browse surviving mutants.

Coverage.py’s branch mode measures **source→destination line transitions** and reports missing branches, which is exactly the right notion if your candidate/reference function contains rule-dispatch logic or edge-case conditionals. The most practical local pattern is **not** to run coverage.py continuously inside the hot loop, but to use it intermittently to calibrate a lighter-weight in-process proxy, such as manual `hits.add("tag")` markers on branch families. Atheris’ own docs likewise note that using coverage.py for the fuzzer is useful for visualization but slows the run substantially.

*(Snippet S1 — manual branch-hit instrumentation with greedy corpus selection — kept live.)*

The reason this works well for ARC-like rules is that grids are small, discrete, and usually branch on a modest number of semantic cases: empty versus non-empty, one object versus many, edge-touching versus interior, square versus rectangular, symmetric versus asymmetric, and so on. Those tags become cheap coverage signals for later optimization. This is a design recommendation, but it is directly motivated by coverage.py’s branch model and by the need to stay under the CPU budget stated in the attachment.

Hypothesis is the best structured-search tool in this setting because it combines **generation**, **explicit seeded examples**, **shrinking to minimal failures**, and a **targeted testing** mode that mutates examples in pursuit of a quantitative objective. The docs recommend using `target()` when you have a natural metric such as an error term, load factor, or similar score; for grids, useful targets include **density**, **number of connected components**, **bounding-box aspect ratio**, or **branch-hit novelty count**. Hypothesis also reports the **simplest failing example**, which is extremely valuable for debugging latent dimension gaps.

**Snippet S2 — boundary-aware Hypothesis strategy for grids**

```python
import numpy as np
from hypothesis import given, example, settings, target, strategies as st

@st.composite
def grids(draw):
    h = draw(st.sampled_from([1, 2, 3, 5, 10, 15, 30]))
    w = draw(st.sampled_from([1, 2, 3, 5, 10, 15, 30]))
    vals = draw(st.lists(st.integers(0, 9), min_size=h * w, max_size=h * w))
    return np.array(vals, dtype=np.uint8).reshape(h, w)

@given(grids())
@example(np.zeros((1, 1), dtype=np.uint8))
@settings(max_examples=2000)
def test_rule(g):
    target(np.count_nonzero(g) / g.size, label="density")
    out = rule(g)
    assert out.shape[0] <= 30 and out.shape[1] <= 30
```

Atheris is only worth the setup cost when you already know how to serialize grids and can define structure-aware mutation. Its docs explicitly say that plain mutation-based fuzzing struggles on complex structured inputs because most mutations are rejected early, but that **custom mutators** can make it efficient by preserving the grammar of valid inputs. That makes byte-level Atheris reasonable for stressing decoders, parsers, serializers, or highly branchy rule code once you have a good grid codec; it is materially less attractive than Hypothesis for everyday local validator design.

Mutation testing belongs in the same stack, but in a different role. Mutmut’s docs emphasize incremental reruns, mutant browsing, and focused test execution. In this problem, that is exactly what you want for the **oracle bundle**: mutate the property checks and auxiliary validators, and if obviously wrong mutants survive, your “independent oracle” is not strong enough. On Windows, mutmut requires a fork-capable environment such as WSL, which is another reason to treat it as a strengthening pass rather than the main generation engine.

**Dimension-stress framework and per-family checklists.**
A usable dimension-stress method for ARC-style rules has six steps. First, classify the candidate into a **rule family**. Second, list the explicitly controllable axes the family depends on. Third, add the attachment’s extra axes that are often forgotten but crucial: **distractor presence**, **object overlap/adjacency**, **output-shape inference difficulty**, and **cross-example consistency under augmentation**. Fourth, define **anchor values** on each axis, especially minima, maxima, and near-boundary values. Fifth, sample with a **pairwise-coverage mindset** rather than a full Cartesian product, to stay inside the CPU budget. Sixth, store axis tags beside every synthetic input so that later diagnostics can slice failures by dimension. ARC-TGI’s core insight that episode-level constraints matter is central here: for some ARC rules, the unit of stress is not a single grid but the **joint training set**.

A practical per-family checklist is: **Did I hit min/max grid sizes? Did I hit single-object and many-object cases? Did I hit corner, edge, and center anchors? Did I force asymmetric forms to break ambiguous rotations/reflections? Did I add irrelevant colors/objects? Did I stress output-shape ambiguity? Did I ensure the train examples jointly disambiguate the rule?** That checklist is the shortest reliable guardrail against the common “all my synthetic examples looked valid, but none forced the hidden edge case” failure.

| Rule family | Critical dimensions | Per-dimension stress recipe | Sample numpy snippet ref |
|---|---|---|---|
| Color remap / filter / subset | Palette size, rare colors, equal-count ties, distractors, parity of counts | Sweep palette cardinality; inject irrelevant colors; create equal-count and one-off-count bins | S4 |
| Object selection / counting | Object count, size, adjacency, overlap, tie-breaks, border contact | Build one-object, two-object, many-object bins; vary bbox sizes; force equal-size ties and edge-touching | S3 |
| Geometric transform / symmetry | Rotation, reflection, asymmetry, non-square shapes, anchor position | Use asymmetric seeds; test 0/90/180/270 and mirror variants; include rectangular grids | S3 |
| Relational placement / interaction | Relative distance, containment, collision, order, occlusion | Move objects across corner/edge/center anchors; sweep nearest/farthest separations | S5 |
| Growth / fill / tiling / completion | Sparsity, density, holes, stop conditions, boundary saturation | Alternate sparse and nearly full grids; create cavities; force path-to-border and no-path cases | S4 |
| Multi-example induction | Cross-example variation, ambiguity breaking, example order, augmentation consistency | Sample train examples jointly; require each train set to eliminate at least one simpler alternative rule | S1 / S2 |

The strongest practical design choice here is to keep **family-specific generators** small and composable. A good generator is usually not “sample a random 30×30 grid with random colors.” It is “sample an object-count case, then a size interaction, then a position anchor, then a distractor regime, then a cross-example disambiguation pattern.” That is much closer to how ARC-GEN, RE-ARC, and ARC-TGI think about data.

**Snippet S3 — shape and anchoring sweep**

```python
import numpy as np

def anchored_variants(seed):
    for h, w in [(1, 1), (1, 30), (30, 1), (2, 2), (3, 5), (30, 30)]:
        g = np.zeros((h, w), dtype=np.uint8)
        sh, sw = min(seed.shape[0], h), min(seed.shape[1], w)
        g[:sh, :sw] = seed[:sh, :sw]          # corner anchor
        yield g
        g2 = np.zeros((h, w), dtype=np.uint8)
        g2[h-sh:h, w-sw:w] = seed[:sh, :sw]   # opposite-corner anchor
        yield g2
```

**Snippet S4 — distractor injection and density sweep**

```python
import numpy as np

def inject_distractors(g, rng, k=3):
    out = g.copy()
    h, w = out.shape
    for _ in range(k):
        r, c = rng.integers(0, h), rng.integers(0, w)
        if out[r, c] == 0:
            out[r, c] = int(rng.integers(1, 10))
    return out

def density_bin(h, w, fill_prob, rng):
    return (rng.random((h, w)) < fill_prob).astype(np.uint8) * rng.integers(1, 10)
```

**Adversarial techniques.**
For discrete grid rules, the best edge-case finding methods are **smart-seeded random generation**, **mutation-plus-disagreement evolutionary search**, **boundary-aware Hypothesis strategies**, and **byte-level fuzzing only after a good serializer exists**. The attachment explicitly asks for executable recipes for all four, and that is the right emphasis: random generation without a seed bank or a fitness signal is too weak for this setting.

Smart-seeded random generation should start from the visible examples or from public generators such as RE-ARC and ARC-GEN, then apply small semantics-preserving or semantics-perturbing mutations: add/remove a distractor, move one object from center to edge, resize a selected object by one cell, replace one color with another, pad the border, transpose rectangular grids, or create ambiguous near-ties. This is much more sample-efficient than uniform random grids because it stays near the actual support of interest. That strategy is also consistent with how competition systems used augmentation rather than pure random data, and with ARChitects’ observation that naive synthetic generation had low yield.

Mutation-plus-disagreement evolutionary search is the single most important adversarial tool. The fitness should not just be “candidate differs from reference.” It should combine at least four signals: **candidate/reference disagreement**, **branch-hit novelty**, **boundary bonus**, and **distance from already-seen bins**. If you do not yet trust the reference, the disagreement term can instead be candidate versus an **alternative interpretation set**, or versus a **property bundle**. This lets you mine precisely the inputs most likely to reveal a hidden unstressed dimension.

*(Snippet S5 — mutation-plus-disagreement evolutionary search — kept live.)*

Boundary-aware Hypothesis strategies should bias toward the **small, large, degenerate, and tie-rich** parts of the space, not just random interior values. In practice that means `sampled_from([1,2,3,5,10,15,30])` for dimensions, explicit singleton and full-grid objects, and seeded examples via `@example`. Because Hypothesis also shrinks failures, it is an ideal complement to evolutionary search: evolution finds a disagreement-rich region, and Hypothesis then minimizes within that region.

Byte-level Atheris fuzzing is worthwhile mainly in two cases: when your candidate/reference pipeline has a fragile **decoder/serializer boundary**, or when you already have a compact binary encoding of grids and a domain-specific mutator that preserves shape and color validity. Without that, early rejection wastes most mutations. With it, Atheris can discover unexpected branch combinations in programs that are hard to model manually.

**Snippet S6 — grid serialization wrapper for Atheris**

```python
import numpy as np

def decode_grid(data: bytes) -> np.ndarray:
    h = 1 + (data[0] % 30 if len(data) > 0 else 0)
    w = 1 + (data[1] % 30 if len(data) > 1 else 0)
    raw = (data[2:] + b"\x00" * (h * w))[:h * w]
    return (np.frombuffer(raw, dtype=np.uint8) % 10).reshape(h, w)

def custom_mutator(data, max_size, seed):
    g = decode_grid(data).copy()
    r, c = seed % g.shape[0], (seed // 31) % g.shape[1]
    g[r, c] = (g[r, c] + 1) % 10
    return bytes([g.shape[0] - 1, g.shape[1] - 1]) + g.astype(np.uint8).tobytes()[:max_size - 2]
```

**Oracle-independence techniques.**
Kept live in full, except these two provenance hedges trimmed from the live text: “This protocol is a reconstruction based on the attachment’s summary of the missing partial audit, so I label it as a design assumption rather than a verbatim extraction.” (after the three-layer protocol) and “This is a design recommendation grounded in the attachment’s stated concern about circular testing.” (after the property-bundle families).

**Diagnostic sequence for “passed synth, failed test.”**
The first question is not “why did the candidate fail,” but “what information do I have about the synthetic corpus that can localize the unstressed dimension?” If every synthetic sample is stored with metadata such as rule family, shape bin, palette bin, object-count bin, branch-hit set, and mutation provenance, diagnosis becomes much faster. The best diagnostic sequence is: **tag**, **slice**, **compare**, **mine**, **rebuild**, **retest**. ARC-TGI’s emphasis on task-level constraints and the official ARC reports’ emphasis on overfitting make this metadata discipline especially important.

*(Mermaid flowchart — kept live.)*

The most actionable numbered diagnostic sequence is this. **Step one:** verify that the held-out failure can be mapped to a candidate rule family. **Step two:** bisect the synthetic corpus by stored dimension tags and branch tags, not just by random halves. **Step three:** run differential testing against a small library of **simpler alternative interpretations**, especially those that fit the visible examples with fewer conditions. **Step four:** launch disagreement-guided evolutionary search centered on the highest-risk bins. **Step five:** check oracle independence by mutating the property bundle and auxiliary validators. **Step six:** promote every discovered failure into a mandatory generator bin or cross-example constraint, and rebuild the local benchmark. This is the shortest loop that actually changes the synthetic distribution rather than merely adding more random samples.

**Decision matrix.**
The right recommendation depends less on the abstract task and more on the **failure you are trying to prevent**. If the risk is family drift, pick a mimetic generator. If the risk is ambiguity across train examples, pick a joint sampler with constraints. If the risk is hidden branch logic, pick Hypothesis plus branch tags. If the risk is circular testing, pick an oracle-hardening stack. That is the central selection principle.

| Situation | Recommended approach | Why this is the best fit |
|---|---|---|
| Validate one candidate against a known ARC-AGI-1 family under a tight CPU budget | RE-ARC or ARC-GEN family sampler + manual branch tags + Hypothesis boundary probes | Best fidelity-to-effort ratio for local validation |
| Need held-out-like breadth and benchmark realism | ARC-GEN first, then family-specific stress overlays | Publicly validated and explicitly mimetic |
| Need train-set examples that jointly force the rule | ARC-TGI-style episode constraints | Explicitly designed to prevent independent-sampling ambiguity |
| Need lots of synthetic training data rather than validator-grade probing | BARC or NVARC artifacts | Best public scale and augmentation volume |
| Need to find hidden edge cases quickly | Evolutionary disagreement search, seeded from visible examples | Most sample-efficient route to counterexamples |
| Need parser/serializer robustness or very branchy binary pipelines | Atheris with grid serializer and custom mutator | Byte-level fuzzing only becomes useful after structure-aware mutation |
| Need to harden the oracle bundle | Property bundle + mutmut + independent public verifier | Best defense against the self-consistency trap |

**Verification checklist.**
The attachment’s required checklist can be satisfied with the following gate. A note is ready only if: the six research questions are answered; generator rows include source links and license status; the dimension framework names the extra axes the attachment called out; the self-consistency trap is explicitly mitigated with independent oracles and property checks; the postmortem sequence is actionable enough to run immediately; and the recommended local generation loop can hit roughly **1000 examples in under 30 seconds** on CPU by preferring **NumPy family generators + manual branch tags + smart-seeded mutation**, while reserving Atheris and large-scale winner pipelines for more targeted roles.

## Methodology and Sources

This report extracted the attachment text, identified the actual reduced scope, and then prioritized **primary sources** in this order: public generator repositories and commit histories, arXiv papers from the generator authors, official ARC Prize reports and competition pages, winner technical writeups, and official tool documentation for Hypothesis, Atheris, coverage.py, and mutmut. That source strategy is the best fit for the attachment because it asks specifically for methodology overviews, public availability, licenses, fidelity claims, and practical testing-tool behavior.

The main limitations are also clear. The attachment references a **partial-audit file** containing a four-axis rubric and a three-layer validation protocol, but that file was not attached, so those pieces were **reconstructed** from the attachment summary rather than quoted. BARC’s exact “200k–400k” public footprint is also somewhat messy in captured first-party pages: the public assets clearly expose a 200k heavy dataset and multiple 100k datasets, while later survey material summarizes BARC as 200k ARC-Heavy plus 400k ARC-Potpourri. I have therefore separated **high-confidence first-party facts** from **plausible but less directly surfaced summary figures**. Finally, the winner writeups from ARChitects and NVARC are valuable for methodology, but they do not amount to polished public general-purpose generator libraries in the same way that RE-ARC, ARC-GEN, and ARC-TGI do.

## Assumptions and Privacy Handling

The attachment contains no obvious personal, financial, medical, or regulated data. It does, however, include **internal filenames, process history, and prior-attempt notes**, which I treat as potentially proprietary workflow context rather than public facts. I therefore use the attachment only as the task specification for the report and reproduce it in the appendix solely because the user explicitly requested extracted text.

My most important assumption is explicit: because the cited partial-audit note was not attached, I reconstructed the **four-axis fidelity rubric** as rule fidelity, dimension coverage, distribution realism, and oracle independence, and reconstructed the **three-layer validation protocol** as structural validity plus branch diversity, independent semantic validation, and adversarial counterexample mining. Those reconstructions are consistent with the attachment’s stated goals, but they are still reconstructions, not verbatim extractions.

## Appendix

The following is the extracted text of the attached Markdown file, reproduced verbatim as the task specification appendix.

```markdown
# Research Card A3: Synthetic test-data generation methodologies for ARC-style puzzles

> **History**:
> - **First attempt (2026-05-25, researcher #1)**: declined the card and produced a 5-option scoping report. See `Todo/Declined/a3_scoping_attempt_2026-05-25.md`. Useful only for the source list seed.
> - **Second attempt (2026-05-25, researcher #2)**: produced a partial deliverable — source audit + methodology critique + 4-axis rubric proposal, but NOT the requested research note format. See `_archive/notes_synth_oracle_partial_audit_2026-05-25.md`. Substantial value: caught the card's "Hodel's arc-gen" error, identified BARC + ARC-TGI as missing references, proposed the 4-axis fidelity rubric and 3-layer validation protocol.
>
> **CORRECTED SOURCE LIST** (was wrong in the original card; fixed 2026-05-25):
> - `michaelhodel/re-arc` (MIT, GitHub) — Hodel's procedural generators for all 400 ARC training tasks. arXiv:2404.07353.
> - `google/ARC-GEN` (Apache-2.0, GitHub) — Moffitt 2025 mimetic benchmark generator. arXiv:2511.00162. Also covers 500 ARC-AGI-2 tasks as of Mar 2026.
> - `xu3kev/BARC` (no declared license, GitHub) — Ellis-group LLM seed-remix generator, 200k-400k synthetic tasks. ICLR 2025.
> - ARC-TGI (2026) — cross-example-constraint-aware generator design from TU Dresden / TIB.
> - ARC Prize 2024 + 2025 technical reports (official).
> - ARChitects 2024 paper + 2025 technical report (winners).
> - NVARC 2025 (winner) — synthetic-data-driven pipeline per ARC Prize official summary.
> - `Hypothesis`, `Atheris` (google/atheris), `coverage.py`, `mutmut` — official docs for property-based testing, coverage-guided fuzzing, branch coverage, mutation testing respectively.
>
> **REDUCED SCOPE — focus on these unanswered pieces only**:
> - **Q1**: per-generator capability comparison (the source list above is mapped; need per-tool feature/cost/strength/weakness analysis in a comparison table). Skip generic survey — go straight to the capability matrix.
> - **Q2**: specific coverage-driven generation techniques for grid inputs with concrete numpy snippets (5-15 lines each). The partial-audit doc covers methodology; needs the concrete techniques.
> - **Q3**: dimension-stress design with numpy implementations (boundary sweep, mutation, distractor injection, output-shape variation). 4 ADDITIONAL axes have been proposed beyond the card's original list: distractor presence, object overlap/adjacency, output-shape inference difficulty, cross-example consistency under augmentation. Include these.
> - **Q4**: adversarial search algorithms as executable recipes (not just descriptions). Include: smart-seeded random, mutation+disagreement-fitness evolutionary, boundary-aware Hypothesis strategies, byte-level Atheris fuzzing wrapped around serialized grids.
>
> **Q5 + Q6 reformat-only**: the partial-audit doc already has substantial content on oracle independence (self-consistency trap), the 4-axis fidelity rubric, the 3-layer validation protocol, and the diagnostic Mermaid flowchart. **Do not re-research these — reformat into the canonical research-note structure** specified below in "Required answer order".
>
> **DIRECTIVE**: do NOT produce another meta-critique. The first two attempts have audited the problem space; the work remaining is concrete implementation-grade research. Produce numpy snippets, comparison tables, and algorithm recipes. If a question genuinely needs assumption, make a stated assumption and proceed — do not punt back with options.

## Deliverable

Produce a single **copyable Markdown research note**. Do not write files.
Target length: ~250 lines. Suggested filename: `synth_oracle_strategies.md`.

Research/design note, not implementation. Include short numpy snippets only for the recommended generator patterns.

## Goal

Given a small set (e.g., 4–10) of input/output puzzle pair examples and a candidate transformation rule (as a numpy function), determine systematically how to generate a synthetic input distribution that meaningfully validates whether the candidate rule generalizes to a held-out test distribution.

## Background problem statement

A developer derives a candidate rule, builds a synthetic input generator, runs the rule on 1000 synthetic inputs, and gets 1000/1000 matches with a reference oracle. Yet on the held-out evaluation, the candidate fails. This indicates the synthetic distribution did not stress the same dimensions as the held-out distribution. The card asks: how do we systematically prevent this?

## Environment

- Tools available: plain numpy, optionally Hypothesis (Python property-based testing). No ML frameworks at generation time.
- Target throughput: 1000+ inputs in ≤ 30 seconds on CPU.
- Output format: numpy arrays compatible with the rule under test.

## Required research questions

### Q1. Published ARC-AGI synthetic data generators
Survey (see corrected source list above; do NOT search for "Hodel's arc-gen" — that's a card error, doesn't exist):
- `michaelhodel/re-arc` — Hodel's procedural generator
- `google/ARC-GEN` — Moffitt 2025 mimetic benchmark generator (NOT Hodel)
- `xu3kev/BARC` — Ellis-group LLM seed-remix generator
- ARC-TGI 2026 — cross-example-constraint-aware
- Any other generators released by ARC-AGI 2024 / 2025 Kaggle winners
- Academic generators for ARC-style puzzles (search arxiv "ARC-AGI synthetic" 2023–2025)

For each: methodology overview, public availability, license, fidelity claims.

### Q2. Coverage-driven generation
Approaches to instrument a numpy reference and generate inputs that maximize branch coverage. Look at:
- Hypothesis (Python property-based testing) `@given` / `@example` strategies
- coverage-guided fuzzing (atheris, etc.)
- Mutation testing tools

State which are practical for puzzle-grid inputs (discrete, small, structured).

### Q3. Dimension-stress methodology
Systematic way to enumerate the dimensions a transformation rule depends on. Example dimension axes for grid puzzles:
- Color set (which colors present, count of each)
- Object size (single-cell, small, large)
- Object count (1, 2, many)
- Object position (corner, edge, center)
- Rotation/reflection presence
- Scale variations
- Sparsity / density
- Parity-related properties (cell count mod 2, mod 3, etc.)

Propose a checklist + per-axis stress-generation recipe in numpy.

### Q4. Adversarial / fuzzing approaches
Techniques specifically for finding edge cases in discrete-grid rules:
- Boundary testing (min/max grid size, single-cell objects, full-grid objects)
- Random walks in input space
- Evolutionary search with the candidate-vs-reference disagreement as fitness
- Smart-seeded random generation (mutate from visible examples)

### Q5. The self-consistency trap
When the synthetic oracle is derived from the same source as the candidate (e.g., both implement the same rule interpretation), the test is circular: synth oracle says "yes" because it implements the same wrong rule. How to ensure oracle independence:
- Use multiple oracles from independent derivations
- Use external oracles (published implementations)
- Use property tests that don't require a full oracle (e.g., "output must contain colors that are a subset of input colors plus X")

### Q6. The "passed synth, failed test" diagnostic
When this occurs, what's the recommended diagnostic sequence to identify which dimension was unstressed? Propose:
- Bisection: split synth into halves, find which half fails
- Differential testing: compare candidate output to alternative interpretations
- Counterexample mining: search for inputs where candidate disagrees with simpler rules

## Required answer order

1. Problem statement and the synth-distribution-mismatch failure mode
2. Survey of published generators (Q1)
3. Coverage-driven approaches (Q2)
4. Dimension-stress framework + per-family checklists (Q3)
5. Adversarial techniques (Q4)
6. Oracle-independence techniques (Q5)
7. Diagnostic sequence for "passed synth, failed test" (Q6)
8. Decision matrix: situation → recommended approach
9. Verification checklist

## Required table schemas

For Q1 (generator survey):

| Generator | URL | Methodology | Public/private | Last updated | Notable for |

For Q3 (dimension-stress per family):

| Rule family | Critical dimensions | Per-dimension stress recipe | Sample numpy snippet ref |

For Q6 (diagnostic):

| Symptom | First hypothesis | Test to perform | Next step |

## Code snippet policy

- One numpy snippet per recommended generator/recipe.
- Plain numpy + optionally Hypothesis.
- No heavy dependencies; no GPU/ML frameworks.

## Source policy

- Cite RE-ARC (Hodel), ARC-GEN (Google/Moffitt), and BARC (Ellis-group) repos by URL. NOTE: there is no "Hodel arc-gen" — original card erred; Hodel = RE-ARC.
- Cite Hypothesis docs for property-based testing claims.
- Cite Kaggle write-ups by URL for winner methodology claims.
- Mark any unverified inferences.

## Verification checklist

- All 6 questions addressed.
- Generators have URLs + license info.
- Dimension framework includes a per-rule-family checklist.
- Self-consistency trap explicitly addressed with mitigation.
- Diagnostic sequence is actionable (numbered steps).
- Recommended generators run in ≤ 30s for 1000 inputs.
```


---

## [completeness sweep 2026-06-10] snapshot lines not carried into live or the sections above

> Restored verbatim from the pre-refactor snapshot after a containment audit found the
> compression pass had dropped them from both live and this file. Mostly MAC-era points math
> and dtype-packing claims (dead under current scoring — see scoring_internals.md) plus
> stray prose; preserved here so nothing exists only in the snapshot. Lines appear in
> snapshot order, context-free; the snapshot has the surrounding text.

> **STATUS: Card A3 deliverable, researcher #3, 2026-05-25T17:43Z**. This is the actual research note A3 requested — implementation-grade, with code snippets (numpy/Hypothesis/coverage.py/Atheris/mutmut), a generator capability matrix, 3-layer validation protocol, and 4-axis fidelity rubric. Card A3 body + the researcher-#1 scoping attempt were deleted 2026-05-27 in the Done/Declined cull (commit eb74a378 — git history only). The researcher-#2 partial audit survives at `_archive/notes_synth_oracle_partial_audit_2026-05-25.md`.
The self-consistency trap is real: if the candidate rule and the synthetic oracle are both derived from the same mistaken interpretation, “1000/1000 matches” is nearly meaningless. The most reliable mitigation is to require **three independent semantic lenses** before trusting the result. First, use at least one **independent implementation** or public verifier whenever possible; RE-ARC and ARC-GEN are especially useful here because both expose explicit generators and validation logic, and RE-ARC additionally exposes corresponding solver/verifier programs. Second, add a **property bundle** that does not require full output equality, such as color-subset constraints, count-preservation bounds, equivariance under recoloring bijections, or monotonicity/containment constraints. Third, run **mutation testing** on those properties and helpers; if obvious mutants survive, the property bundle is too weak to serve as an independent oracle layer.
The cleanest practical three-layer validation protocol is therefore: **Layer one**, structural validity and branch diversity; **layer two**, semantic checks from an external/public implementation plus metamorphic or invariant properties; **layer three**, adversarial disagreement search against alternative interpretations. This protocol is a reconstruction based on the attachment’s summary of the missing partial audit, so I label it as a design assumption rather than a verbatim extraction.
A strong property bundle for ARC-like grids usually includes at least one property from each of these families: **shape constraints** such as “output dims remain within 1..30”; **palette constraints** such as “output colors are a subset of input colors plus the known marker color set”; **object constraints** such as “the number of selected objects in output equals the number of target objects in input”; and **symmetry/metamorphic constraints** such as “relabeling colors by a bijection relabels the output correspondingly” when the task is color-agnostic. Those are not universal truths; they are **task-family-specific auxiliary oracles**. Their job is not to solve the puzzle in full, but to make it harder for a wrong shared interpretation to pass. This is a design recommendation grounded in the attachment’s stated concern about circular testing.
