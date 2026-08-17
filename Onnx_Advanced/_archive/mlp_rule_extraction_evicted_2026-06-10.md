# mlp_rule_extraction.md — evicted content: executive summary, technique-inventory taxonomy, PLNN theory, encoder/decoder probing prose, sample-complexity prose, verification checklist + caveats, full URL dump.
Evicted 2026-06-10 (refactor pass 4). Live distillate: `../mlp_rule_extraction.md`; full pre-refactor snapshot: `mlp_rule_extraction_pre_refactor_2026-06-10.md`.
Original headings and order preserved below; sections kept live (NeuroGolf empirical addendum, practical pipelines with code, adversarial validation + decision table, sample-complexity table) are omitted here.

## Executive summary

This note follows the user-supplied research card specification and treats missing operational details as unspecified rather than invented. In particular, the card specifies a small trained MLP setting, an example shape such as `21 → 264 → 52 → 150`, discrete grid-style inputs and outputs, and a desire for an explicit symbolic rule in a fixed operation vocabulary, without an inference-time LLM.

For the specific problem class in the card, the cheapest *published* family of approaches is usually **pedagogical rule extraction by querying the trained MLP and distilling it into a small decision tree or rule list**, optionally using TREPAN-style query synthesis or soft-tree distillation. These methods are easy to implement with standard tooling, work well when arbitrary teacher queries are cheap, and directly output symbolic objects that can often be compiled into a fixed operator vocabulary. Their main weakness is that they optimize **fidelity by approximation**, not exact equivalence.

If exactness matters, the strongest family is **piecewise-linear analysis of ReLU networks**, especially OpenBox-style exact interpretation for piecewise-linear neural networks and verifier-backed reasoning with Marabou, ReLUplex, or MILP encodings. ReLU MLPs compute piecewise-linear or piecewise-affine functions over activation regions, so in principle one can express network behavior as a partition of the input space into polytopes with affine output rules. In practice, this is excellent for proving or refuting candidate rules and extracting exact *local* rules, but often expensive for producing one short global symbolic program.

Pure attribution methods such as **LIME, Anchors, and SHAP** are useful support tools but are rarely sufficient by themselves for global rule recovery. LIME fits local surrogate models, Anchors returns local if-then clauses with precision guarantees, and SHAP returns additive feature attributions. These are valuable for naming features, stress-testing hypotheses, and guiding probe construction, but they do not natively return a faithful global symbolic rule for a whole MLP.

For a **small ReLU MLP with discrete categorical outputs**, the problem is substantially more tractable than generic “deep interpretability.” On each fixed ReLU activation pattern, the network reduces to an affine map; then argmax or rounding partitions that affine region into linear inequality cells. This means candidate rules can be represented as trees, rule lists, or polyhedral clauses. The two hidden layers also make a practical two-stage workflow possible: infer `input → hidden_1` primitive semantics, then `hidden_1 → hidden_2 → output` composition and decoding. That is a synthesis from decompositional rule extraction, probing, and PLNN theory rather than a theorem proved specifically for the 30×30-grid / 21-dimensional compressed-latent case.

The strongest practical recommendation is therefore a **hybrid workflow**: first, build interpretable probes for hidden units and decoder blocks; second, distill the teacher into a compact tree or rule list on valid synthetic queries; third, use Marabou or MILP to search for counterexamples separating the extracted rule from the MLP; and only if fidelity remains poor, escalate to OpenBox- or ECLAIRE-style exact/decompositional extraction. This gives the best cost-to-faithfulness tradeoff for the problem described in the card.

## Inventory of techniques

The rule-extraction literature still largely organizes methods into **pedagogical** approaches that treat the network as a black box, **decompositional** approaches that exploit internal structure layer by layer, and **eclectic** approaches that mix both. That taxonomy originates in classic surveys and is still used in recent reviews. A recurring theme across the literature is the tradeoff between **fidelity**, **accuracy**, and **comprehensibility**: the most exact methods are usually the least concise, while the most concise methods are usually approximate.

| Technique | Reference | Output type | Sample complexity | Code/tool available | Applicability to small discrete-output MLP |
|---|---|---|---|---|---|
| TREPAN / query-induced decision tree extraction | Craven & Shavlik 1995; later TREPAN case study  | Global decision tree | Moderate to high query budget; specifically benefits from teacher queries outside the training set because it “uses queries to induce a decision tree.”  | Unofficial Python implementations exist.  | **High** if arbitrary valid teacher queries are available and the desired rule can be tree-like. |
| Soft decision tree distillation | Frosst & Hinton 2017  | Soft or hardenable decision tree | Moderate; needs a distillation set, often thousands of teacher-labeled samples or logits.  | Community PyTorch implementations exist.  | **Medium to high** for a first symbolic approximation, but soft splits still need hardening for fixed-op deployment. |
| LIME | Ribeiro et al. 2016  | Local sparse linear surrogate | High per explained point; common implementation default is `num_samples=5000`.  | Official repo and docs.  | **Low as a standalone extractor**, **useful as support** for discovering which latent directions matter locally. |
| Anchors | Ribeiro et al. 2018  | Local if-then rules with precision guarantee | Adaptive and potentially high; multi-armed-bandit search continues until a precision threshold is met.  | Official repo.  | **Medium** for extracting local clauses around specific examples or regions; **low** for whole-network global recovery by itself. |
| SHAP / KernelSHAP | Lundberg & Lee 2017  | Local additive feature attributions | Exact model-agnostic Shapley is exponential, `O(2^M)`; sample-based explainers use approximation, and common auto settings use `nsamples ≈ 2M + 2048`.  | Official repo and docs.  | **Low as a final rule extractor**, **good** for ranking latent features and checking whether a surrogate ignored important directions. |
| DeepRED / layerwise decompositional extraction | Zilke et al. 2016  | Rule set / rule tree extracted layerwise | Can become expensive with depth and data volume; original work improved on shallow-only methods but still suffers scaling issues.  | Available in research packages such as `ruleex`; community repos also exist.  | **High** for small feedforward MLPs when layerwise semantic decomposition is wanted. |
| ECLAIRE / REMIX | Espinosa Zarlenga et al. 2021  | Global rule set | Claimed polynomial-time extraction and better scaling than DeepRED/REM-D.  | Official REMIX library.  | **High** when the model is still feedforward but larger, or when DeepRED-style layerwise extraction becomes clumsy. |
| OpenBox exact PLNN interpretation | Chu et al. 2018  | Equivalent local linear classifiers / exact piecewise explanation | No stochastic neighborhood sampling required, but region complexity can still grow.  | Official code repo.  | **Very high** for ReLU MLPs when exact local linear semantics matter more than short global summaries. |
| ReLUplex / Marabou / MILP verification | Katz et al. 2017; Marabou 2019 and 2024; Tjeng et al. 2019  | SAT/UNSAT certificates, counterexamples, feasible region witnesses | Not sample-driven; complexity is solver/search complexity over piecewise-linear constraints.  | Official repo, docs, and tools.  | **Very high for validation and exact region testing**, **moderate** for full global extraction because enumeration can explode. |
| TabNet-style attentive interpretability | Arik & Pfister 2021  | Instancewise masks and global feature importances | Retraining cost roughly comparable to training a new tabular model.  | Official paper and maintained implementation.  | **Low as a post-hoc extractor for an existing vanilla MLP**; **useful as a surrogate baseline** if retraining is allowed. |
| Linear probes / TCAV / mechanistic feature analysis | Alain & Bengio 2016; Kim et al. 2018; Olah et al. 2020; Anthropic 2022  | Concept detectors, probe accuracies, feature subspaces | Moderate; requires concept datasets or synthetic labels rather than exhaustive querying.  | Paper code varies; the methodology is easy to reproduce with standard ML tooling.  | **High as a bridge technique** for decoding hidden-layer semantics, but **not sufficient alone** for final global rule extraction. |

The conflict in the literature is not really “which method is best,” but **which objective is primary**. If the objective is a compact symbolic object that behaves like the MLP on most valid inputs, pedagogical distillation is usually best. If the objective is a mathematically exact account of the ReLU network, exact PLNN interpretation plus verification is stronger. If the objective is understanding what hidden units encode, probes and concept methods are more informative. Recent surveys and eclectic methods explicitly frame modern work as attempts to balance those goals rather than to eliminate the tradeoff.

## Small MLPs with discrete outputs

For the card’s main setting, a three-layer MLP with ReLU activations can be written as

\[
h_1 = \mathrm{ReLU}(W_1x+b_1), \quad
h_2 = \mathrm{ReLU}(W_2h_1+b_2), \quad
y = W_3h_2+b_3.
\]

The example architecture in the card, `21 → 264 → 52 → 150`, is still tiny by modern standards and well within the regime where teacher-query distillation, exhaustive activation tracing, and solver-backed counterexample search are all realistic engineering choices.

The key structural fact is that a ReLU network computes a **piecewise-linear** or **continuous piecewise-affine** function. Each fixed active/inactive pattern of the hidden ReLUs defines one linear region, and on that region the network acts like a single affine map. This is the theoretical basis of both region-counting work and exact interpretation methods such as OpenBox, and it is exactly why rule extraction is more plausible for small ReLU MLPs than for arbitrary nonlinear nets.

For a **10-color × 15-cell categorical output**, two common decoders matter. If each cell selects one color by **argmax**, then on any fixed ReLU region the class assignment for a cell is determined by pairwise linear inequalities comparing the chosen logit with the nine alternatives. If the output is instead **rounded** to integers, then each rounded output pattern corresponds to half-open interval constraints on affine coordinates. This means that within each activation region, the final symbolic rule is a conjunction of linear inequalities. That conclusion is an inference from PLNN theory plus the formal semantics of ReLU, ArgMax, and Round; it does **not** imply the global rule will be short.

This polyhedral view is exactly why **Marabou-style reasoning** is relevant. Marabou represents ONNX networks as piecewise-linear constraints, answers SAT/UNSAT queries, and can return a concrete satisfying assignment when a property fails. The Marabou papers also state that termination is guaranteed only for piecewise-linear activations, which aligns well with a small fully connected ReLU MLP. In practice, that makes Marabou more useful for **checking candidate rules and finding counterexamples** than for globally enumerating every region.

The small depth also supports a practical **two-stage extraction** strategy. First, interpret `input → hidden_1` as primitive feature detectors. Second, interpret `hidden_1 → hidden_2 → output` as composition and decoding. That is consistent with decompositional rule extraction, which explicitly extracts rules layer by layer, and with probe methodologies that ask what information is linearly recoverable from each intermediate representation. The literature does not directly benchmark this exact grid-encoder setting, so this recommendation should be read as a technically justified inference rather than a published turnkey recipe.

```mermaid
flowchart LR
    A[Raw grid or valid 21-d latent code] --> B[Layer 1 21→264 + ReLU]
    B --> C[Layer 2 264→52 + ReLU]
    C --> D[Layer 3 52→150 logits]
    D --> E[ArgMax or Round decoder]
    B -. probe / sparse rule fit .-> F[Primitive detectors]
    C -. probe / sparse rule fit .-> G[Composed concepts]
    D -. block grouping / calibration .-> H[15 cells × 10 colors]
    F --> I[Candidate symbolic rule]
    G --> I
    H --> I
```

For **discrete latent codes**, solver-based extraction becomes even more attractive. Marabou’s examples include disjunction constraints, which can be used to encode discrete cases such as “this latent coordinate must be 0 or 1” or small finite-choice structures. If the 21-dimensional input is not arbitrary continuous data but a compact code with combinatorial validity constraints, bringing those constraints into the solver is one of the best ways to avoid learning spurious rules on impossible inputs.

## Input and output encoding decoding

A crucial caveat from the card is that the first and last layers may partly “memorize” unknown encoder/decoder behavior. If the raw 30×30 grid encoder is unavailable, then any recovered symbolic description is identifiable only **up to the latent code actually presented to the MLP**. In other words, one may recover the MLP’s rule over the 21-dimensional latent domain without uniquely recovering the original human-designed encoder unless there is additional calibration data tying latent coordinates back to grid primitives. That caveat matters for every method in this report.

The most defensible way to reverse-engineer the **input encoder** is to treat `hidden_1` as a bank of candidate primitive detectors and use **concept probing**. Build a representative synthetic set of raw inputs or valid latent codes that vary one grid primitive at a time, record `hidden_1` activations, and train simple probes or sparse rule models to predict each neuron’s activation state or activation bin from candidate interpretable features. Linear probes tell you whether a concept is recoverable from a hidden representation, and TCAV-style concept vectors give a way to test whether a chosen human concept aligns with a direction in latent space.

For this specific setting, the cheapest practical procedure is usually: standardize the 21 latent inputs, rank each `hidden_1` neuron’s incoming weights by absolute standardized effect, generate activation-triggering examples near the ReLU threshold, and then fit a **very small** decision stump, sparse logistic regressor, or anchor clause for “neuron `j` is on.” Because the neuron pre-activation is affine in the latent input before ReLU, large standardized coefficients are suggestive; because real networks can be polysemantic, they are not conclusive. Probe fitting and contrastive testing are what make the interpretation credible.

The **output decoder** is usually easier, because the 150 outputs already live close to the symbolic target. If the intended structure is really 15 cells × 10 colors, then either the output order is known or it can often be recovered with a small calibration set by mapping output vectors to known cell-color labels. After that, one can train one-vs-rest probes from `hidden_2` to each cell-color category, or fit sparse trees from `hidden_2` to each cell’s color choice. OpenBox’s perspective is useful here: on each fixed activation region the network is already just a linear classifier over inputs, so the last layer often behaves like a sparse decoder over a handful of hidden concepts.

A practical heuristic for mapping `hidden_2 → output` is to search for **block structure** in the last-layer weights. If outputs are ordered by cell, the last-layer weight matrix often exhibits rows or columns that cluster by cell or by color family. Even when that structure is not obvious by inspection, sparse per-cell probes and mutual-exclusivity checks generally expose it quickly. This is an engineering inference, but it is exactly the kind of layerwise decomposition that decompositional rule extraction methods operationalize.

One strong warning from modern mechanistic work is that **single-neuron explanations can be misleading**. Toy-model studies of sparse-feature learning show that networks can represent more features than dimensions via superposition, producing polysemantic units. In a small MLP, that means several hidden neurons may each mix multiple grid concepts, or a human concept may live in a small subspace rather than a single neuron. When that happens, the right move is to switch from “name the neuron” to “name the sparse set of neurons or direction” and then distill rules on those grouped features.

## Sample complexity — surrounding prose (table + condensed takeaways kept live)

There is no single sample-complexity answer for “how many auxiliary inputs are needed,” because the answer depends on the **hypothesis family of the target symbolic rule**, whether **arbitrary membership queries** are allowed, and whether the goal is **approximate fidelity** or **exact equivalence**. The literature therefore gives useful bounds for target families and query models rather than one universal number.

Two sharper takeaways follow from that table. First, **active querying matters**. Balcan, Hanneke, and Wortman Vaughan show that active learning can provide exponential gains in the limit in settings where passive sampling appears stubbornly expensive. That is directly relevant here because the MLP can serve as a cheap labeling oracle on constructed inputs.

Second, the right “query budget” is almost always much smaller than people fear *if* the extracted rule family is simple. If the final rule is a shallow tree or a small rule list over 21 interpreted features, then tens of thousands of valid teacher queries is already a generous offline budget on commodity hardware. If the network really requires a large number of piecewise-affine regions to explain, no such budget will rescue a short symbolic rule, and the problem shifts from sample complexity to **description complexity**. That is exactly the fidelity-versus-comprehensibility dilemma emphasized in the rule-extraction literature.

For the card’s example dimensionality, a good practical order of operations is: use a few thousand synthetic queries to decode hidden semantics, then 10k–50k valid teacher queries for surrogate distillation, and only then pay solver costs for exact counterexample search. That recommendation is a synthesis of active-learning bounds, query-based extraction results, and the engineering behavior of modern verification tools.

## Practical pipelines with code references — commentary dropped from live

That snippet is deliberately simple. A published TREPAN-style system would query adaptively and may use richer split tests, but the core idea is the same: treat the MLP as an oracle, train small symbolic surrogates, then refine them with disagreement search.

## Adversarial validation, failure modes, decision guidance, and checklist — parts dropped from live

**Verification checklist.**

- [x] At least five distinct technique families are covered.
- [x] The small-MLP, discrete-output ReLU case is addressed explicitly.
- [x] A concrete encoder/decoder decomposition strategy is provided.
- [x] More than two published extraction pipelines are included, with code references.
- [x] An adversarial / contrast-set validation method is described.
- [x] Failure modes are stated explicitly.
- [x] The decision table covers more than four common MLP situations.

**Caveats and recommended next steps.** Several card-critical details remain unspecified: whether the 21-dimensional input is a fully valid latent code for all query points, whether output decoding is by argmax or rounding, whether any normalization precedes the first affine layer, and whether raw grid inputs can be paired with latent encodings during analysis. Those details materially change what counts as a “valid” synthetic query and therefore which extracted rules are trustworthy.

## Source URLs

The inline citations above are the authoritative references. The primary URLs used most heavily are listed below for convenience.

```text
https://arxiv.org/abs/1711.09784
https://proceedings.neurips.cc/paper_files/paper/1995/file/45f31d16b1058d586fc3be7207b58053-Paper.pdf
https://arxiv.org/abs/1602.04938
https://homes.cs.washington.edu/~marcotcr/aaai18.pdf
https://arxiv.org/abs/1705.07874
https://www.sciencedirect.com/science/article/pii/0950705196819204
https://www.lamda.nju.edu.cn/publication/jcst04.pdf
https://arxiv.org/abs/2312.12878
https://arxiv.org/abs/1908.07442
https://openreview.net/forum?id=HJ4-rAVtl
https://proceedings.mlr.press/v80/kim18d/kim18d.pdf
https://distill.pub/2020/circuits/zoom-in
https://www.anthropic.com/research/toy-models-of-superposition
https://arxiv.org/abs/1802.06259
https://github.com/lingyangchu/OpenBox
https://proceedings.mlr.press/v119/rolnick20a.html
https://arxiv.org/abs/1702.01135
https://theory.stanford.edu/~barrett/pubs/KHI+19.pdf
https://github.com/NeuralNetworkVerification/Marabou
https://neuralnetworkverification.github.io/Marabou/
https://arxiv.org/abs/1711.07356
https://github.com/vtjeng/MIPVerify.jl
https://arxiv.org/abs/2111.12628
https://github.com/mateoespinosa/remix
https://github.com/marcotcr/lime
https://github.com/marcotcr/anchor
https://github.com/shap/shap
https://github.com/dreamquark-ai/tabnet
https://onnx.ai/onnx/operators/onnx__Relu.html
https://onnx.ai/onnx/operators/onnx__ArgMax.html
https://onnx.ai/onnx/operators/onnx__Round.html
```


---

## [completeness sweep 2026-06-10] snapshot lines not carried into live or the sections above

> Restored verbatim from the pre-refactor snapshot after a containment audit found the
> compression pass had dropped them from both live and this file. Mostly MAC-era points math
> and dtype-packing claims (dead under current scoring — see scoring_internals.md) plus
> stray prose; preserved here so nothing exists only in the snapshot. Lines appear in
> snapshot order, context-free; the snapshot has the surrounding text.

Marabou’s official ONNX example shows the same core operations: load an ONNX model, read input and output variables, set bounds, and call the solver. Its lower-level MarabouCore example shows how to add explicit equations and ReLU constraints when a higher-level ONNX query is not expressive enough.
The third pipeline is especially valuable in the card’s “memorized encoder / decoder” scenario, because the hard part is not immediately “find the whole rule,” but first “discover what the hidden layers mean.” Linear probes are the cheapest reliable starting point, and TCAV-style concept tests are useful if you already have candidate human concepts such as translation, border color, object count, symmetry, or connected-component presence.
**Adversarial validation.** The right standard is not “does the extracted rule look sensible,” but “can one find a valid input where the rule and the MLP disagree?” Formally, if `R(x)` is the extracted rule and `f(x)` is the MLP, then the ideal check is to search the valid domain `D` for any `x` such that `x ∈ D` and `R(x) ≠ f(x)`. If a verifier proves that set is empty on a region, the rule is faithful there. If a solver finds one witness, the rule is wrong or incomplete. This is exactly the use case where Marabou and MILP-style verifiers shine.
**Contrast-set construction.** The most effective contrast sets are usually near the surrogate’s boundaries: examples near tree thresholds, near ReLU switching surfaces, or near output-tie surfaces where two colors compete for the same cell. Active learning suggests querying exactly such uncertain or boundary-adjacent examples, and model-extraction work confirms that carefully selected queries recover target behavior faster than passive sampling. For grid tasks, this usually means perturbing just one hypothesized primitive at a time while preserving all others.
**When extraction is hopeless.** There are three major failure modes. First, the MLP may have memorized idiosyncratic associations with no short symbolic summary, in which case any high-fidelity tree or rule list will become large and unstable. Second, the learned computation may be naturally continuous or geometric rather than expressible in the user’s fixed op vocabulary, so an exact descriptor exists but is not human-compact. Third, the representation may be polysemantic, meaning the relevant concepts are superposed rather than neuron-localized, which defeats simple per-neuron explanations. All three reflect the central tension between fidelity and comprehensibility emphasized in rule-extraction surveys and recent mechanistic-interpretability work.
**Decision guidance.** The table below is the shortest practical answer to “given MLP shape, data, and target rule expressiveness, which technique should be used?”
A sensible concrete plan is therefore:
