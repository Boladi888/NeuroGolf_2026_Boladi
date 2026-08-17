# function_proto_subgraphs.md

> **STATUS: DEAD-END for NeuroGolf — 2026-05-26**. Research is technically accurate but the NeuroGolf validator (`data/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py:202,204,211`) rejects every construct discussed here:
> - `if model.functions: return None` → kills `FunctionProto` (any function definition)
> - `if opset.domain not in {"", "ai.onnx"}: return None` → kills the custom domain that `FunctionProto` registration requires
> - `if attr.type in [AttributeProto.GRAPH, AttributeProto.GRAPHS]: return None` → kills `If` / `Loop` / `Scan` / `SequenceMap` (every subgraph-bearing op)
>
> See `Onnx_Advanced/Research/notes/b3_function_proto_subgraphs_critique_2026-05-26.md` for the full disposition and `Onnx_Advanced/core.md` "Validator rejection criteria" lines 49-65 for the project-level documentation of these gates. Card B3 moved to `Todo/Declined/` to avoid future re-investigation.
>
> The content below is preserved as a general ONNX/ORT reference for non-NeuroGolf contexts.

## FunctionProto structure and registration

`FunctionProto` is a model-local function definition in ONNX IR. The IR spec says a function has `name`, `domain`, optional `overload`, `doc_string`, `attribute` / `attribute_proto`, `input`, `output`, `node`, `opset_import`, optional `value_info`, and optional `metadata_props`. The same IR section also says model-local functions serve as a default or fallback implementation of the corresponding operator, while a runtime may choose an alternative optimized kernel implementation. Importantly for this card, the serialized `FunctionProto` structure has **no `initializer` field**.

To register a reusable function end-to-end in a model, the pattern is:

1. pick a non-empty custom domain, such as `"custom"`;
2. create `OperatorSetIdProto` entries for both the default ONNX domain `""` and the custom domain;
3. build the function with `onnx.helper.make_function(...)`;
4. add that function to `make_model(..., functions=[...])`;
5. import the custom domain in `opset_import`;
6. call the function from the main graph by creating a normal node whose `op_type` is the function name and whose `domain` is the custom domain. The ONNX Python docs show this exact pattern with `LinearRegression` in a `custom` domain. The IR spec separately requires that each operator used in a graph be declared by one of the imported operator sets.

The helper API also exposes `overload` on both `make_function` and `make_node`. That matters only if you need multiple model-local functions with the same `(domain, name)` and distinct overload identifiers in IR v10+, but for most ARC-style hand-built graphs a single `(domain, name)` pair is the simplest path.

A minimal verified pattern from the ONNX helper docs is:

```python
new_domain = "custom"
opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

body = [
    make_node("MatMul", ["X", "A"], ["XA"]),
    make_node("Add", ["XA", "B"], ["Y"]),
]

linear_regression = make_function(
    new_domain, "LinearRegression",
    ["X", "A", "B"], ["Y"],
    body, opset_imports, []
)

call = make_node(
    "LinearRegression", ["X", "A", "B"], ["Y1"],
    domain=new_domain
)

model = make_model(graph, opset_imports=opset_imports, functions=[linear_regression])
```

This is the official helper-doc pattern: define the function, import the custom domain, then call it as a node in the main graph. The same ONNX docs note that `make_function` “does not include initializers either,” which matches the IR structure above.

## ORT execution behavior

The highest-confidence answer for ONNX Runtime is **some hybrid**, not pure eager inlining and not a guaranteed shared callable. ORT stores model-local functions once at the model level as `FunctionTemplate`s; the model source keeps a map of model-local functions, creates a `FunctionTemplate` whose `onnx_func_proto_` points at the stored `FunctionProto`, and exposes them via `GetModelLocalFunctionTemplates()`. Nodes, in turn, can hold a pointer to a function template via `SetFunctionTemplate`, which assigns `func_template_ = &func_template`. That means the model representation is shared at load time.

At the same time, ORT clearly has an explicit call-site inlining path. The graph API exposes `Status InlineFunction(Node& node);`, and the comment snippet visible in the ORT headers says the graph must be `Resolve()`d after this call. ORT’s own IR tests load a model with model-local functions, iterate nodes, check `node.GetFunctionBody() != nullptr`, then call `graph.InlineFunction(node)` in a loop until no function bodies remain, and finally call `graph.Resolve()`. That is strong evidence that ORT does **not** eagerly inline every function at initial load, because the tests explicitly perform inlining after load.

ORT also contains a specialization/inliner in `function_utils.cc` that rewrites a callee for a **particular call site** by binding the formals to the call node’s actual inputs and outputs and then renaming internal values with a unique prefix. That is textbook call-site duplication semantics. Once you use that path, body-local constants and intermediates become call-site-specific, which defeats any “single shared callable body” assumption for scoring.

The practical conclusion is that ORT 1.24 should be treated as **representation-sharing at the model/template level plus optional call-site inlining**, not as a runtime contract that guarantees amortized callable sharing across all `FunctionProto` invocations. I did **not** find a public ORT doc that says the executor keeps one callable body and dispatches to it for all call sites, so that stronger claim would be an inference too far. For engineering decisions around your scorer, assume there is **no guaranteed score-saving runtime sharing** unless you verify your own pipeline.

| Behavior | Source | Implication for sharing |
|---|---|---|
| Model-local functions are stored once in the `Model` as `FunctionTemplate`s, and nodes can point at that template | ORT `model.cc` / `model.h` / `graph.cc` snippets showing `GetModelLocalFunctionTemplates`, `onnx_func_proto_ = &func`, and `func_template_ = &func_template`  | There is **some** representation-level sharing at model load |
| ORT exposes `Graph::InlineFunction` and its tests manually inline after load | ORT `graph.h` snippet and `onnx_model_test.cc` inlining loop  | ORT does **not** appear to eagerly inline every call at load time |
| ORT has a call-site specializer/inliner that renames values with a unique prefix | ORT `function_utils.cc` `Inliner::specialize` and transform logic  | Once inlined/specialized, body contents behave like duplicated inline chains, not shared code |

If you want a definitive answer for **your** ORT 1.24 session path, the clean experiment is to save the optimized model and inspect it. ORT documents `optimized_model_filepath` / `OptimizedModelFilePath` as the mechanism to serialize the optimized graph after graph-level transformations. A minimal test is: build one model with a function called once and another with the same function called `N` times; set optimized-model serialization; compare whether the saved graph still contains function-call nodes or expanded per-call-site node copies, and compare counts of `Constant` nodes and internal value names.

## Scorer counting policies

For the exact scorer you described — `param_elements = sum(initializer elements + Constant tensor elements)` — the first spec-level fact is simple: **serialized `FunctionProto` bodies do not have initializers**. The IR definition for `FunctionProto` omits any `initializer` field, and the ONNX Python helper docs explicitly say `make_function` does not include initializers. So a generic scorer that traverses ONNX protobufs literally should count **zero** “function-body initializers,” because there are none in the serialized model format.

`Constant` is different. A `FunctionProto` body is a node list, so it may contain `Constant` nodes, and those constant payloads are stored once in the function definition. If your scorer counts constants by walking the serialized model **definitions** without semantic expansion, then a function-local constant tensor should be counted **once per function definition**, not `N` times per call site. That policy measures compressed/deduplicated model structure. This is an inference from the protobuf storage model plus the IR function structure; ONNX itself does not define a canonical “score” for this.

A second, equally defensible policy is **expansion-style counting**: count function-body constants `N` times when the function is called `N` times. This policy is consistent with the IR’s semantic framing that a function’s meaning is given by expansion into a subgraph, and it is also consistent with ORT’s explicit specialization/inlining machinery, which rewrites bodies for particular call sites. Under that policy, `FunctionProto` provides readability and authorship reuse, but **no parameter-count savings versus fully inlined op chains**.

One subtle scorer blind spot is worth calling out. The IR allows function attributes and `attribute_proto` default values, and attribute-parameters used inside the function body are replaced by actual call-site values or defaults. If you allow tensor-valued default attributes and your scorer only counts initializers plus `Constant` nodes, then you may miss tensor payloads routed through attributes. That is a scorer-definition issue, not a `FunctionProto` sharing guarantee.

| Behavior | Source | Implication for sharing |
|---|---|---|
| `FunctionProto` has no `initializer` field; helper API also says `make_function` has no initializers | ONNX IR spec and helper docs  | No serialized function-internal initializer payload exists to “share” or count |
| Count function-body `Constant` tensors once per function definition | Inferred from serialized storage of a single `FunctionProto` in the model and model-level template storage; no published scorer spec  | Function reuse can reduce `param_elements` **only if** your scorer deduplicates by definition |
| Count function-body `Constant` tensors once per call site after semantic expansion | ONNX IR says function semantics are expressed via expansion; ORT has specialization/inlining per call site  | No `param_elements` savings versus explicit inline chains |

My recommendation for this specific project is to decide **up front** which scoring semantics you want:

- **Definition-level scoring** rewards reusable structure in the protobuf and can make `FunctionProto` worthwhile.
- **Expansion-level scoring** better matches worst-case inlined execution graphs and removes almost all scoring benefit from `FunctionProto`.

Without that choice, engineers will reach different conclusions from the same model. This recommendation is inferred from the spec/runtime split above.

## If op semantics

The ONNX `If` operator takes a scalar boolean input `cond` and two required graph attributes, `then_branch` and `else_branch`. The spec says `then_branch` is the graph to run if the condition is true, and `else_branch` is the graph to run if the condition is false. The branches must return the same number of outputs, with matching types; later versions of `If` allow branch output shapes to differ, with the enclosing `If` output shape representing the union/compatibility of both possibilities.

For **runtime** cost, the unused branch is not executed, so its intermediates are not materialized for that invocation. In other words, the branch you did not take has zero actual runtime intermediate-memory cost for that run. For a **static scorer**, however, branch counting is a policy choice. A conservative scorer may sum both branches; a tighter per-run upper-bound scorer may take `max(cost(then), cost(else))`; and an input-conditional scorer may count only the branch proven active. The spec tells you execution semantics, but it does not define your custom score model.

`If` is ORT-supported in your environment in the normal prebuilt package path. ONNX’s own Python tutorial builds an `If` node and executes it successfully with `onnxruntime.InferenceSession`. Separately, ORT’s install and compatibility docs say prebuilt packages have full support for ONNX opsets and operators, and that all ORT versions support ONNX opsets 7 and higher. Since your environment is opset `<= 18`, `If` is available in ORT 1.24 unless you are using a reduced/custom build that intentionally strips operators.

A minimal verified `If` shape looks like this:

```python
then_body = make_graph(
    [make_node("Constant", [], ["then_out"], value=then_cst)],
    "then_body", [], [then_out]
)

else_body = make_graph(
    [make_node("Constant", [], ["else_out"], value=else_cst)],
    "else_body", [], [else_out]
)

if_node = make_node(
    "If", ["cond"], ["Y"],
    then_branch=then_body,
    else_branch=else_body
)
```

That is the same pattern used in the ONNX Python docs, which then run the resulting model with ONNX Runtime.

In a no-`Loop`/no-`Scan` environment, `If` is still useful for **runtime branching without repetition**. Practical examples are: choose between a cheap path and an expensive path based on a scalar predicate computed from the input; gate a cleanup/normalization subgraph only when needed; or select among tensor-shape-compatible construction strategies without paying both branches at runtime. The key caveat is that these savings are real only if your scorer understands control flow instead of flattening both branches into one static total.

## Other subgraph-bearing ops

In the standard ONNX main domain, the verified operators with graph-valued attributes relevant here are `If`, `Loop`, `Scan`, and `SequenceMap`. `If` has `then_branch` and `else_branch`; `Loop`, `Scan`, and `SequenceMap` each have a required `body` graph. `Loop` and `Scan` are banned by your environment, so the only additional standard graph-attribute operator left to consider is `SequenceMap`.

`SequenceMap` is not a good substitute for ordinary tensor-graph reuse in this project. Its spec says the `body` graph is run for each sample in the input sequence(s), which is semantically much closer to a map/iteration primitive than to a reusable tensor subroutine. Even though it is not explicitly on your banned list, it is generally a poor fit for hand-built ARC-style graphs unless you already represent your data as sequences and want per-element mapping semantics.

I did **not** find an official ONNX page that globally enumerates “all operators with graph attributes,” so the statement that these are the full standard main-domain set up to opset 18 is an **inference from the operator-spec sweep above**, not a quoted ONNX master list.

## Practical use cases and net-savings analysis

For the formulas below, let `N` be the number of repeated call sites, `K` the total embedded constant-element count inside the reusable body, and `I` the additive intermediate-memory score of one body instance. These formulas are **inferred** from your stated score model, and if your `intermediate_memory_bytes` metric is peak-live memory rather than additive per-intermediate accounting, replace the `I` arithmetic with your liveness model.

| Pattern | Inline cost | FunctionProto/If cost | Net savings | Depends on ORT behavior? | Recommendation |
|---|---|---|---|---|---|
| Per-channel repeated predicate chain for `N=10` channels, with a small fixed threshold/mask constant inside the body | `10·K + 10·I` | **Definition-level scorer:** `K + 10·I`. **Expansion-level scorer:** `10·K + 10·I` | Only saves `9·K` under definition-level counting; otherwise none | Yes. If ORT/save-to-optimized-model expansion is your effective scoring view, savings vanish | Use `FunctionProto` only if your scorer deliberately deduplicates function-body constants; otherwise inline directly |
| Per-quadrant repeated transform for `N=4` quadrants, body has no hidden constants | `4·I` | `4·I` under either policy if there are no body-local constants to deduplicate | Usually zero | Weakly. ORT behavior hardly matters because the score comes from intermediates, not reusable parameters | Do **not** use `FunctionProto` as a scoring trick here; use it only for readability/maintainability |
| Per-quadrant repeated transform for `N=4` quadrants, body has axis/shape/constants embedded as `Constant` nodes | `4·K + 4·I` | **Definition-level scorer:** `K + 4·I`. **Expansion-level scorer:** `4·K + 4·I` | Saves `3·K` only under definition-level counting | Yes | Same conclusion as the per-channel case: possible structural-score win, no reliable runtime/share win |
| Branch-on-input-property using `If`, where one branch is cheap (`I_small`) and the other is expensive (`I_big`) | No branch: pay the pessimistic always-on chain, effectively `I_big` or a fused superset | **Runtime actual:** only the chosen branch runs. **Branch-aware scorer:** `max(I_small, I_big)` or path-conditional. **Flattening scorer:** `I_small + I_big` | Can be substantial if the scorer counts only the executed or max branch; can be negative if it sums both branches | Not mainly ORT behavior; mainly scorer policy, because the `If` runtime semantics are clear | `If` is the strongest remaining composition tool in this environment, but only when your scorer is control-flow aware |

The recurring result is that `FunctionProto` only buys score when the repeated body contains **counted payloads** — for your scorer, that means `Constant` tensors, not just op structure — and when the scorer counts the function definition once rather than the expanded call sites. If the repeated body is purely arithmetic on inputs, `FunctionProto` yields essentially **no** `param_elements + intermediate_memory_bytes` advantage over inline implementation.

`If` is different. It can produce real runtime savings because the untaken branch is not executed. That makes it the only remaining sanctioned mechanism here that can change the executed intermediate-memory footprint without using loops. But the win is entirely contingent on your scorer matching control-flow semantics instead of flattening both branches structurally.

## Recommendations

Use `FunctionProto` in this project primarily as an **authoring abstraction**, not as a guaranteed optimization. It is good for keeping repeated micro-pipelines readable, giving them a name, and centralizing a verified op chain. It is **not** a reliable way to obtain score savings unless you have explicitly chosen a deduplicated definition-level scorer for function-body constants.

Treat the ORT answer as: **shared template storage, optional later inlining, no guaranteed callable amortization that you should bake into scoring assumptions**. If score matters more than source brevity, assume expansion-style counting until your own optimized-model dump proves otherwise. That is the safer engineering default.

Use `If` when you have a genuine mutually exclusive cheap/expensive path and you can make your scorer branch-aware. In this banned-op environment, `If` is the composition feature most likely to translate into real benefit, because only one branch runs at inference time. A good internal policy is: count `If` as `max(then, else)` for intermediate-memory score unless you can prove the predicate statically, and count branch-local parameters according to the same policy you adopt for `FunctionProto` constants. This scoring recommendation is an inference, but it aligns cleanly with ONNX execution semantics.

Avoid trying to use `FunctionProto` as a loophole for hidden parameters. There are no function-body initializers in the serialized IR, and any advantage you get from body-local `Constant` nodes is scorer-defined rather than runtime-guaranteed. Engineers should regard a “FunctionProto saves parameters” claim as valid **only** after answering: “Are we scoring stored definitions, or expanded call sites?”

`SequenceMap` is technically another subgraph-bearing operator, but I would avoid it for this project. It reintroduces map/iteration-like semantics over sequences and is unlikely to fit ARC-style tensor constructions better than explicit tensor ops or `If`.

## Verification checklist

- [x] FunctionProto registration steps are documented end-to-end: define function, choose domain, import domain in `opset_import`, add to `functions=[...]`, and call it as a node in the main graph.
- [x] The ORT execution-behavior question is explicitly answered as **some hybrid**, with source-backed evidence for model-level template storage and explicit post-load inlining, plus an experiment proposal using optimized-model serialization where the docs are not definitive.
- [x] Scorer counting policies are described both ways: deduplicated definition-level counting and expansion/call-site counting.
- [x] `If` cost semantics are stated: unused branch is not executed at runtime; static scoring depends on branch policy. `If` support in ORT is also stated.
- [x] At least two practical use cases are evaluated; three are included, with a required-schema table.
- [x] The final recommendation distinguishes what is useful **in principle** from what is useful **given ORT/scorer behavior**. 