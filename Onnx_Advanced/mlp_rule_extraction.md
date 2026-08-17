# Extracting symbolic rules from small trained MLPs

Distilled 2026-06-10 (refactor pass 4). NeuroGolf-tested findings first, then the extraction pipelines with code, adversarial validation + decision table, and the sample-complexity numbers. Generic survey/taxonomy prose, PLNN background, encoder/decoder probing prose, and the full URL list were evicted to `_archive/mlp_rule_extraction_evicted_2026-06-10.md` (full original: `_archive/mlp_rule_extraction_pre_refactor_2026-06-10.md`).

## NeuroGolf empirical addendum (2026-05-25)

The first real-world application of this report — task219 LIVE — surfaced one important assumption gap. Documenting here for future workers who apply the pipeline to other NeuroGolf MLP LIVEs. *("E1" below = this report; its three pipelines are the table in §Practical pipelines with code references.)*

### Gap: not all NeuroGolf MLPs use ReLU

The report's pipelines (especially OpenBox/Marabou PLNN analysis) assume **ReLU activations** so the network computes a piecewise-affine function. In practice, the task219 LIVE used **RBF-style activation**: `(W₁·x − b₁)² < 0.01`. This is NOT piecewise-affine — it's a narrow-band radial detector that fires only when the pre-activation lands in a small interval around `b₁`. The piecewise-linear theory and Marabou's termination guarantees do not apply.

**How to recognize the RBF/hash-MLP variant**:
- First hidden layer is `Squared → Less → Cast` rather than `Add → ReLU`.
- Threshold is small (~0.01) so detectors are 1-hot per input (exactly one detector fires).
- `min == max == mean == 1.0` for "detectors active per input" across the visible set is the smoking gun.

**Implication**: for this variant, **pedagogical distillation still works** (TREPAN, decision trees on probe outputs) because it treats the MLP as a black box. But **decompositional methods** (DeepRED, OpenBox region enumeration) and **PLNN verification** (Marabou termination on piecewise-linear) lose their guarantees. The right tool reduces to: probe-first to identify the hash structure, then distillation, then empirical counterexample search rather than provable verification.

### Hash-MLP architectural fingerprint

*(This fingerprint was also promoted to `techniques.md` **#56** on 2026-06-05 — the quick-check version lives there; below is the full empirical signature.)*

When the first hidden layer behaves as 1-hot-per-input detectors AND the count of detectors ≈ count of visible inputs (e.g., 264 detectors for 266 visible pairs), the MLP is functioning as a **perfect hash table**, not a learned rule. Empirical signature on task219:

| Property | Value | Interpretation |
|---|---|---|
| Hidden1 width | 264 | Matches 266 visible (minus duplicates) |
| Activation | `(x − b)² < 0.01` | RBF, NOT ReLU |
| Detectors active per input | exactly 1 | Perfect hash, no superposition |
| Single-bit perturbation → detectors active | 0 | Zero generalization |
| Distillable as small tree? | 1/150 outputs at depth 4 | Rule is global, not local |

**Decision tree distillation on this variant gives a TRUE NEGATIVE for "there is a simpler rule"** — the small-tree distillation explicitly fails because the rule IS inherently global. This is a useful signal: if probes reveal hash-table structure, the true rule MUST exist outside the MLP (in the training-data construction process), and the MLP is not a useful proxy for finding it.

### Updated decision table — when to apply E1's three pipelines to NeuroGolf

| LIVE architecture | Probe-first finding | Recommended path |
|---|---|---|
| ReLU MLP, polysemantic hidden, distillable small tree exists | Probe reveals interpretable features per neuron | E1 pipeline 1 (TREPAN distillation) + E1 pipeline 2 (Marabou verification) |
| ReLU MLP, polysemantic hidden, no small tree | Probe shows superposition; deep distillation needed | E1 pipeline 3 (probe + concept decomposition) |
| **Hash-MLP (RBF or argmax-narrow), 1-hot detectors, 1:1 with visible** | Probe shows perfect hash with zero generalization | **NOT extractable from the MLP — the rule lives outside it**. Skip E1 entirely. Look for the underlying generator (row-grammar, template, geometric construction) by analyzing the visible input→output pairs directly with numpy/sklearn. The MLP is a memoization layer, not a model. |
| QLinearConv stack or Conv-distilled architecture | Probe shows the rule lives in Conv kernels | Pipeline 3 with concept probing on Conv outputs; refer to `arc_rule_taxonomy*.md` for family-matching |

**Worker advisory**: before spending budget on E1 distillation, run a 30-line "is this a hash-MLP?" probe. Count detectors per input + check single-bit-flip generalization. If hash signature confirmed, do NOT distill — go back to the visible pairs and look for the generator. Empirical case-in-point: task219 LIVE wasted 0 worker hours on E1 after this probe established the rule lives outside the MLP.

## Practical pipelines with code references

The literature supports three realistic pipelines for this problem class, each with a different cost profile. **Query-distillation** is the cheapest to build, **polytope/exact extraction** is the strongest on faithfulness, and **probe-first extraction** is the best bridge when the major unknown is what the encoder and decoder are doing internally.

| Pipeline | Source | Steps | Output format | Known limitations |
|---|---|---|---|---|
| Distill into a decision tree or rule list | TREPAN; soft-tree distillation; classic pedagogical extraction  | Query the teacher MLP on many **valid** synthetic inputs, fit a shallow tree or rule list, prune for simplicity, then search for counterexamples and refine. | Global decision tree, per-cell rule list, or a fixed-op symbolic program compiled from the tree. | Approximate by default; can learn rules on impossible latent states if synthetic queries leave the true input manifold.  |
| Polytope / exact PLNN extraction | OpenBox; Marabou; MILP verification  | Export to ONNX, analyze local linear regions, encode candidate rule properties as linear constraints, and use exact solving to prove or refute them. | Polytope partition, local linear classifier family, or formally checked region-wise rule clauses. | Region explosion is real; exact local explanations do not automatically compress into a short global symbolic rule.  |
| Probe-network extraction | Linear probes; TCAV; concept-guided distillation  | Dump intermediate activations, train probes for candidate concepts and outputs, group neurons into interpretable concept sets, then distill rules on probe outputs or selected hidden units. | Concept map, sparse neuron sets, and rule lists over hidden concepts or proxy concepts. | Requires a concept library or engineered probes; can miss polysemantic subspaces or nonhuman concepts.  |

The first pipeline is usually the right starting point when the end product must be a **symbolic rule executable without an LLM**. In spirit it is the TREPAN idea updated with modern tooling: use the MLP as a teacher oracle, but only query on valid latent points or raw-grid-derived encodings.

```python
# Query-distillation on valid latent inputs
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

# X_valid_query: shape [n, 21], sampled from the VALID latent manifold
# y_teacher: shape [n, 15], one color index per output cell
y_teacher = teacher_predict_colors(X_valid_query)  # e.g. argmax over 15x10 logits

trees = []
for cell in range(15):
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=0)
    clf.fit(X_valid_query, y_teacher[:, cell])
    trees.append(clf)
    print(f"\nCell {cell}")
    print(export_text(clf, feature_names=feature_names))
```

The second pipeline is the one to use when you already have a candidate rule and want to know whether it is **actually faithful**. OpenBox is attractive because it is designed for piecewise-linear neural networks, while Marabou and MILP tools are attractive because they can search for exact counterexamples in constrained regions.

```python
# Minimal Marabou ONNX query skeleton
from maraboupy import Marabou

net = Marabou.read_onnx("mlp.onnx")
options = Marabou.createOptions(verbosity=0)

input_vars = net.inputVars[0][0]   # batch dim stripped
output_vars = net.outputVars[0]

# candidate_rule_input_box should encode the region where your extracted rule claims to apply
for i, (lo, hi) in enumerate(candidate_rule_input_box):
    net.setLowerBound(input_vars[i], lo)
    net.setUpperBound(input_vars[i], hi)

# Example property: the target output coordinate must remain above a chosen floor.
# For pairwise logit comparisons, add custom equations via MarabouCore.
target = output_vars[target_index]
net.setLowerBound(target, target_floor)

vals, stats = net.solve(options=options)
print("SAT" if vals else "UNSAT")
```

Marabou's official ONNX example shows the same core operations: load an ONNX model, read input and output variables, set bounds, and call the solver. Its lower-level MarabouCore example shows how to add explicit equations and ReLU constraints when a higher-level ONNX query is not expressive enough.

The third pipeline is especially valuable in the card's "memorized encoder / decoder" scenario, because the hard part is not immediately "find the whole rule," but first "discover what the hidden layers mean." Linear probes are the cheapest reliable starting point, and TCAV-style concept tests are useful if you already have candidate human concepts such as translation, border color, object count, symmetry, or connected-component presence.

```python
# Probe hidden activations, then induce a tiny rule on selected hidden units
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

H1 = collect_hidden_activations(model, X_probe, layer_name="hidden1")  # [n, 264]
concept = concept_has_red_border.astype(int)                           # engineered labels

probe = LogisticRegression(penalty="l1", solver="liblinear", C=0.1)
probe.fit(H1, concept)

selected = np.flatnonzero(probe.coef_[0])
cell0_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0)
cell0_tree.fit(H1[:, selected], teacher_predict_colors(X_probe)[:, 0])

print("Selected hidden units:", selected.tolist())
print(export_text(cell0_tree))
```

A strong pattern in practice is to run **probe-first, distill-second, verify-third**. Probe-first tells you whether the network has learned clean hidden concepts at all. Distillation then turns those concepts into short rules. Verification finally tells you whether the resulting rule is faithful or just plausible. That ordering tends to minimize wasted solver time.

## Adversarial validation, failure modes, and decision guidance

**Adversarial validation.** The right standard is not "does the extracted rule look sensible," but "can one find a valid input where the rule and the MLP disagree?" Formally, if `R(x)` is the extracted rule and `f(x)` is the MLP, then the ideal check is to search the valid domain `D` for any `x` such that `x ∈ D` and `R(x) ≠ f(x)`. If a verifier proves that set is empty on a region, the rule is faithful there. If a solver finds one witness, the rule is wrong or incomplete. This is exactly the use case where Marabou and MILP-style verifiers shine.

**Contrast-set construction.** The most effective contrast sets are usually near the surrogate's boundaries: examples near tree thresholds, near ReLU switching surfaces, or near output-tie surfaces where two colors compete for the same cell. Active learning suggests querying exactly such uncertain or boundary-adjacent examples, and model-extraction work confirms that carefully selected queries recover target behavior faster than passive sampling. For grid tasks, this usually means perturbing just one hypothesized primitive at a time while preserving all others.

**When extraction is hopeless.** There are three major failure modes. First, the MLP may have memorized idiosyncratic associations with no short symbolic summary, in which case any high-fidelity tree or rule list will become large and unstable. Second, the learned computation may be naturally continuous or geometric rather than expressible in the user's fixed op vocabulary, so an exact descriptor exists but is not human-compact. Third, the representation may be polysemantic, meaning the relevant concepts are superposed rather than neuron-localized, which defeats simple per-neuron explanations. All three reflect the central tension between fidelity and comprehensibility emphasized in rule-extraction surveys and recent mechanistic-interpretability work.

**Decision guidance.** The table below is the shortest practical answer to "given MLP shape, data, and target rule expressiveness, which technique should be used?"

| Situation | MLP shape | Training data | Recommended technique | Why |
|---|---|---|---|---|
| Small ReLU MLP, shallow rule expected, cheap query access | One or two hidden layers, under ~100k parameters | Visible pairs plus ability to synthesize many valid teacher queries | TREPAN-style decision-tree distillation, then counterexample search  | Lowest engineering cost, directly yields symbolic objects, and query complexity is favorable when the target rule is shallow. |
| Small ReLU MLP, exactness matters more than compactness | Two or three hidden layers, piecewise-linear activations, ONNX export possible | Valid input domain can be bounded or constrained | OpenBox plus Marabou or MILP verification  | Best route to exact local rules and proof-grade counterexamples; strongest faithfulness story. |
| Unknown encoder or decoder is the real bottleneck | Small or medium MLP with hidden layers much wider than input | Some labeled concepts or engineered grid features are available | Probe-network pipeline, then distill on probe outputs  | Cheapest way to expose whether hidden states correspond to interpretable primitives before attempting global extraction. |
| Feedforward net is deeper or larger, but explicit rules are still required | More layers, more data, still feedforward and piecewise-linear | Sufficient training or query data | ECLAIRE / REMIX or similar eclectic/decompositional extractor  | More scalable than classic DeepRED while still aiming at global rule sets. |
| Query access is limited, and only local understanding is immediately needed | Any small MLP | Mostly fixed visible pairs, few synthetic queries | Anchors or SHAP or LIME for local hypotheses, not final global extraction  | Good for debugging and hypothesis generation, but usually insufficient for a deployable global symbolic rule. |

**Concrete order of operations.**
1. Export the network to ONNX and dump `hidden_1`, `hidden_2`, and output activations on all visible pairs plus a few thousand **valid** synthetic queries.
2. Build a compact grid-concept library and run probe tests on hidden layers to see whether primitives are actually recoverable.
3. Distill the teacher into per-cell trees or a short global rule list on those concepts or on validated latent features.
4. Search for disagreement cases with Marabou or MILP, and refine only where counterexamples exist.
5. If the tree explodes or probes stay uninterpretable, treat that as evidence that the MLP did not learn a short symbolic rule in the available vocabulary.

## Sample complexity

No single answer exists — it depends on the **hypothesis family of the target symbolic rule**, whether **arbitrary membership queries** are allowed, and whether the goal is **approximate fidelity** or **exact equivalence**. The numbers that matter:

| Target family | Best cited guidance | Practical implication for this problem |
|---|---|---|
| Depth-`d` decision tree over `n` features | Adaptive exact learning with membership queries can use roughly `Õ(2^(2d)) + 2^d log n` randomized queries for depth-limited trees.  | If the true rule is a shallow tree over interpretable latent or probed features, exact disambiguation can be query-efficient relative to brute force. |
| Linear separator / sparse threshold rule | Under favorable distributional assumptions, active learning can reach error `ε` using about `Õ(d log(1/ε))` labels rather than passive rates that can be much worse.  | If a hidden unit or decoder block behaves like a sparse linear threshold, a few thousand well-chosen probes can be enough. |
| Local additive explanations | Exact Shapley is `O(2^M)` in the number of features `M`; approximate explainers reduce this with sampling. For `M=21`, exact coalition enumeration would mean `2,097,152` coalitions per explanation.  | SHAP is feasible as a diagnostic on a few outputs, but not as the primary route to a full 15-cell symbolic rule. |
| Local perturbation explainers | The common LIME tabular implementation uses `num_samples=5000` by default, and Anchors keeps sampling until a precision guarantee is established.  | LIME and Anchors are practical for a handful of examples or neurons, but expensive to run indiscriminately over many outputs and regions. |
| Black-box teacher extraction | Model-extraction work shows >99% matching on target input space for several model classes, but neural networks can require far more queries than linear equation-solving attacks; ACTIVETHIEF reports useful stolen models with only 10–30% of public samples.  | Query-efficient imitation of a small MLP is realistic, but exact equivalence still depends on the rule family and the validity of queried inputs. |

Takeaways:
1. **Active querying matters** — active learning can give exponential label savings over passive sampling (Balcan, Hanneke & Wortman Vaughan), and the MLP is a free labeling oracle on constructed inputs.
2. If the extracted rule family is simple (shallow tree / small rule list over ~21 interpreted features), **tens of thousands of valid teacher queries is already a generous offline budget** on commodity hardware.
3. Practical order: a few thousand synthetic queries to decode hidden semantics → 10k–50k valid teacher queries for surrogate distillation → pay solver costs for exact counterexample search only at the end.
4. If the network needs a large number of piecewise-affine regions to explain, no query budget rescues a short rule — the problem shifts from sample complexity to **description complexity** (the fidelity-vs-comprehensibility dilemma).

## Canonical sources

- Craven & Shavlik 1995 — "Extracting Tree-Structured Representations of Trained Networks" (TREPAN)
- Frosst & Hinton 2017 — "Distilling a Neural Network Into a Soft Decision Tree"
- Ribeiro, Singh & Guestrin 2016 / 2018 — "Why Should I Trust You?" (LIME) / "Anchors: High-Precision Model-Agnostic Explanations"
- Lundberg & Lee 2017 — "A Unified Approach to Interpreting Model Predictions" (SHAP)
- Zilke, Loza Mencía & Janssen 2016 — "DeepRED: Rule Extraction from Deep Neural Networks"
- Espinosa Zarlenga, Shams & Jamnik 2021 — "Efficient Decompositional Rule Extraction for Deep Neural Networks" (ECLAIRE / REMIX)
- Chu et al. 2018 — "Exact and Consistent Interpretation for Piecewise Linear Neural Networks" (OpenBox)
- Katz et al. 2017 / 2019 — "Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks" / "The Marabou Framework for Verification and Analysis of Deep Neural Networks"
- Elhage et al. (Anthropic) 2022 — "Toy Models of Superposition"

---
*Evicted 2026-06-10 (refactor pass 4): executive summary, technique-inventory taxonomy, PLNN theory, encoder/decoder probing prose, sample-complexity prose, checklist + caveats, full URL dump → `_archive/mlp_rule_extraction_evicted_2026-06-10.md` (pre-refactor snapshot: `_archive/mlp_rule_extraction_pre_refactor_2026-06-10.md`).*
