# N2 - Sequence/container ops as cost reducers

## 1. Operator inventory

Scope: default-domain ONNX sequence, optional, sparse-ish, and string operators that can touch container or non-numeric tensor values. Excluded: `Loop`, `Scan`, `NonZero`, `Unique`, `Compress`, custom domains, and `ai.onnx.ml` map operators per [PROMPT].

Relevant default-domain inventory:

- Sequence: `SequenceConstruct`, `SequenceEmpty`, `SequenceInsert`, `SequenceErase`, `SequenceLength`, `SequenceAt`, `SplitToSequence`, `ConcatFromSequence`, `SequenceMap`. The non-`SequenceMap` sequence surface starts at opset 11; `SplitToSequence` has versions 11 and 24; `SequenceMap` starts at opset 17 and contains a required subgraph body. [PRIMARY: ONNX sequence specs]
- Optional: `Optional` starts at opset 15; `OptionalGetElement` and `OptionalHasElement` start at opset 15 with current version 18. ONNX optional values can appear as inputs, outputs, and intermediates and may reference tensor/sequence/map/sparse elements in the type system, but the default-domain Optional operator schema constrains practical values here to tensor and sequence element types. [PRIMARY: ONNX optional specs and ONNX types]
- String-ish: `StringNormalizer` starts at opset 10; `TfIdfVectorizer` starts at opset 9; `StringConcat`, `StringSplit`, and `RegexFullMatch` start at opset 20. [PRIMARY: ONNX string specs]
- Sparse path: no default-domain sparse algebra or sparse-to-dense bridge surfaced in the ORT CPU kernel table. The relevant path is `Constant` with `sparse_value` attribute, available in modern `Constant` schemas back to opset 13, producing a sparse-format constant value. [PRIMARY: ONNX Constant spec; ORT kernels]

Primary source links used:

- ONNX sequence specs: [`SequenceEmpty`](https://onnx.ai/onnx/operators/onnx__SequenceEmpty.html), [`SequenceConstruct`](https://onnx.ai/onnx/operators/onnx__SequenceConstruct.html), [`SequenceInsert`](https://onnx.ai/onnx/operators/onnx__SequenceInsert.html), [`SequenceErase`](https://onnx.ai/onnx/operators/onnx__SequenceErase.html), [`SequenceLength`](https://onnx.ai/onnx/operators/onnx__SequenceLength.html), [`SequenceAt`](https://onnx.ai/onnx/operators/onnx__SequenceAt.html), [`SplitToSequence`](https://onnx.ai/onnx/operators/onnx__SplitToSequence.html), [`ConcatFromSequence`](https://onnx.ai/onnx/operators/onnx__ConcatFromSequence.html), [`SequenceMap`](https://onnx.ai/onnx/operators/onnx__SequenceMap.html).
- ONNX optional/type specs: [`Optional`](https://onnx.ai/onnx/operators/onnx__Optional.html), [`OptionalGetElement`](https://onnx.ai/onnx/operators/onnx__OptionalGetElement.html), [`OptionalHasElement`](https://onnx.ai/onnx/operators/onnx__OptionalHasElement.html), [ONNX Types](https://onnx.ai/onnx/repo-docs/ONNXTypes.html).
- ONNX string/sparse specs: [`StringNormalizer`](https://onnx.ai/onnx/operators/onnx__StringNormalizer.html), [`RegexFullMatch`](https://onnx.ai/onnx/operators/onnx__RegexFullMatch.html), [`StringConcat`](https://onnx.ai/onnx/operators/onnx__StringConcat.html), [`StringSplit`](https://onnx.ai/onnx/operators/onnx__StringSplit.html), [`TfIdfVectorizer`](https://onnx.ai/onnx/operators/onnx__TfIdfVectorizer.html), [`Constant`](https://onnx.ai/onnx/operators/onnx__Constant.html).
- ORT kernel source: [`OperatorKernels.md` for `rel-1.24.0`](https://raw.githubusercontent.com/microsoft/onnxruntime/rel-1.24.0/docs/OperatorKernels.md), generated from registered kernels by ORT's own script. The installed local patch runtime was `onnxruntime==1.24.4`; local CPU smoke tests matched this table for the tested ops. [PRIMARY][DERIVED]

## 2. ORT CPU support

Table A:

| op | since-opset | input/output type | ORT 1.24.4 CPU kernel? | tensor bridge available? | validator risk | source tags |
|---|---:|---|---|---|---|---|
| `SequenceConstruct` | 11 | variadic tensors -> `seq(tensor(T))` | Yes, 11+ in ORT kernel table; local 1.24.4 returned Python `list` for sequence output | `SequenceAt` for one tensor or `ConcatFromSequence` for all tensors | High: non-tensor intermediate/output unless immediately bridged | [PRIMARY][DERIVED] |
| `SequenceEmpty` | 11 | none -> empty `seq(tensor(T))` | Yes, 11+; local 1.24.4 returned empty `list` | Only after `SequenceInsert`, then `SequenceAt`/`ConcatFromSequence`; `SequenceLength` gives scalar count | High: empty sequence itself is non-tensor and useless alone | [PRIMARY][DERIVED] |
| `SequenceInsert` | 11 | `seq(tensor(T))`, tensor, optional scalar int position -> `seq(tensor(T))` | Yes, 11+; local bridge test passed | `SequenceAt` or `ConcatFromSequence` | High: output remains sequence | [PRIMARY][DERIVED] |
| `SequenceErase` | 11 | `seq(tensor(T))`, optional scalar int position -> `seq(tensor(T))` | Yes, 11+; local bridge test passed | `SequenceAt`, `ConcatFromSequence`, or `SequenceLength` | High: output remains sequence | [PRIMARY][DERIVED] |
| `SequenceLength` | 11 | `seq(tensor(T))` -> scalar `tensor(int64)` | Yes, 11+; local 1.24.4 produced scalar ndarray | Output already tensor | Medium: sequence input still present; scalar tensor is normally acceptable | [PRIMARY][DERIVED] |
| `SequenceAt` | 11 | sequence plus scalar int position -> `tensor(T)` | Yes, 11+; local 1.24.4 produced ndarray | It is the one-element tensor bridge | Medium: prior sequence is risky, output is normal tensor | [PRIMARY][DERIVED] |
| `SplitToSequence` | 11, 24 | tensor plus optional split -> `seq(tensor(T))`; ORT input `T` is limited to bool/double/float/float16/int32/int64/string in 1.24 table, despite broader sequence type list | Yes: `[11,23]` and `24+`; local float split/concat passed | `ConcatFromSequence` for full tensor; `SequenceAt` for one piece; `SequenceLength` for count | High: output is sequence; important dtype caveat for int8/uint8 grid tensors | [PRIMARY][DERIVED] |
| `ConcatFromSequence` | 11 | `seq(tensor(T))` -> `tensor(T)` | Yes, 11+; local 1.24.4 bridge passed | It is the full tensor bridge | Medium: output is tensor, but bridge materializes the same aggregate tensor | [PRIMARY][DERIVED] |
| `SequenceMap` | 17 | sequence(s)/tensor(s) plus required graph body -> sequence output(s) | No registered CPU kernel found in ORT 1.24 kernel table | If runtime existed, still needs `SequenceAt`/`ConcatFromSequence` | Fatal: subgraph plus sequence output; also no ORT CPU kernel table entry | [PRIMARY][DERIVED] |
| `Optional` | 15 | tensor or sequence, or no input with `type` attr -> optional(tensor/sequence) | Yes, 15+; local wrap/unwrap passed | `OptionalGetElement` for contained tensor/sequence; `OptionalHasElement` for scalar bool | High: optional output is non-tensor unless unwrapped | [PRIMARY][DERIVED] |
| `OptionalGetElement` | 15, 18 | optional(tensor/sequence) or plain tensor/sequence -> contained tensor/sequence | Yes: `[15,17]` and `18+`; local optional tensor unwrap produced ndarray | It is the optional bridge if contained value is a tensor | Medium: output can be tensor, but empty optional is undefined/error per spec | [PRIMARY][DERIVED] |
| `OptionalHasElement` | 15, 18 | optional/tensor/sequence or omitted input -> scalar `tensor(bool)` | Yes: `[15,17]` and `18+`; local 1.24.4 produced scalar bool ndarray | Output already tensor | Medium: non-tensor optional input, scalar output rarely useful | [PRIMARY][DERIVED] |
| `StringNormalizer` | 10 | `tensor(string)` -> `tensor(string)` | Yes, 10+; local 1.24.4 passed | No numeric bridge except unsupported/irrelevant string conversion chains | Fatal for numeric grid scoring | [PRIMARY][DERIVED] |
| `RegexFullMatch` | 20 | `tensor(string)` -> `tensor(bool)` | Yes, 20+; local 1.24.4 passed | Output is bool tensor; input still must be string | Fatal unless the whole problem is already string-valued, which NeuroGolf is not | [PRIMARY][DERIVED] |
| `StringConcat` | 20 | `tensor(string)`, `tensor(string)` -> `tensor(string)` | Yes, 20+; local 1.24.4 passed | No numeric bridge | Fatal for numeric grid scoring | [PRIMARY][DERIVED] |
| `StringSplit` | 20 | `tensor(string)` -> `tensor(string)` substrings plus `tensor(int64)` counts | Yes, 20+; local 1.24.4 passed | Counts are tensor; substrings remain string | Fatal: string input/output surface, dynamic padded split tensor | [PRIMARY][DERIVED] |
| `TfIdfVectorizer` | 9 | `tensor(int32/int64/string)` -> 1-D `tensor(float)` | Yes, 9+; local int64 smoke passed | Output already tensor | High/fatal for grids: fixed vocabulary vector, attribute-heavy n-gram pool, wrong shape/domain | [PRIMARY][DERIVED] |
| `Constant` with `sparse_value` | 13 for sparse attr in current docs lineage | no input -> sparse-format constant value / sparse tensor path | `Constant` is supported; local 1.24.4 returned ORT `SparseTensor` for `sparse_value` | No default-domain dense bridge found in ORT kernel table | Fatal unless validator accepts sparse outputs, which [PROMPT] says likely not | [PRIMARY][DERIVED] |

## 3. Validator/scorer risk analysis

Sequence family:

- Sequence-producing ops (`SequenceConstruct`, `SequenceEmpty`, `SequenceInsert`, `SequenceErase`, `SplitToSequence`, `SequenceMap`) produce non-tensor values. A tensor-only scorer/validator is likely to reject graph outputs of these types and may reject them even as intermediate values. [PROMPT][PRIMARY]
- `SequenceAt`, `ConcatFromSequence`, and `SequenceLength` are bridges back to tensors. They are the only useful sequence exits here. `SequenceAt` materializes one member tensor; `ConcatFromSequence` materializes the combined tensor; `SequenceLength` materializes a scalar `int64`. [PRIMARY][DERIVED]
- Under the cost model, if a sequence is used only to delay materializing a tensor that must later be bridged back, the bridge tensor is still charged if it is intermediate, or final-output-free only if it is the required final output. That usually saves nothing compared with direct tensor ops such as `Slice`, `Concat`, `Reshape`, or producing the final tensor directly. [PROMPT][DERIVED]
- `SplitToSequence` has a hard dtype wrinkle: ORT 1.24's CPU kernel table does not list `uint8`/`int8` as input `T` for the split input, even though the sequence type list contains many element types. Typical grid tensors often use compact integer dtypes, so this blocks a common use case. [PRIMARY][DERIVED]

Optional family:

- `Optional` produces an optional value, not a normal tensor. `OptionalGetElement` can recover a tensor if the element exists. `OptionalHasElement` gives a scalar bool tensor. [PRIMARY]
- Wrapping then immediately unwrapping a tensor is semantically equivalent to an identity-like transfer with extra non-tensor surface. It cannot hide the unwrapped tensor if later numeric ops need it. [DERIVED]
- Empty optional values are dangerous: `OptionalGetElement` is undefined/error on an empty optional, so any conditional use would need `If`, which introduces subgraphs and validator risk. [PRIMARY][PROMPT][DERIVED]

String family:

- `StringNormalizer`, `StringConcat`, and `StringSplit` operate on string tensors; `RegexFullMatch` converts strings to bools; `TfIdfVectorizer` can consume int/string tokens and emits a 1-D float vector. [PRIMARY]
- NeuroGolf grids are numeric tensors, so these ops require entering string space or accepting a vocabulary-vector output shape. There is no cheap exact bridge from numeric grids to string tensors and back in the relevant default-domain CPU surface. [PROMPT][PRIMARY][DERIVED]
- `TfIdfVectorizer` is the only string-ish op that can consume integer tensors, but it is an n-gram counter with pool/index attributes. Those attributes are parameter-like payloads, and the output is a vocabulary vector, not `[1,C,H,W]`. [PRIMARY][PROMPT][DERIVED]

Sparse path:

- `Constant.sparse_value` exists in the ONNX spec, but local ORT 1.24.4 returned a sparse runtime value for that path. The ORT CPU kernel table does not expose a default-domain sparse-to-dense bridge. [PRIMARY][DERIVED]
- A sparse graph output or intermediate is a validator risk under the prompt. If a dense tensor is needed, this path is dead without a supported densify bridge. [PROMPT][DERIVED]

## 4. Bridge patterns and pricing

Pricing assumptions from [PROMPT]:

- Final graph output tensor is free.
- Intermediate tensor outputs are charged by byte size.
- Initializers and Constant-node tensors are charged by element count.
- Non-tensor sequence/optional/sparse values are not clearly priced, but are high validator risk. Treat any "saved" bytes as unusable unless the graph is proven accepted.

Legal tensor bridges:

- `SequenceAt(seq, pos) -> tensor(T)`: materializes exactly one sequence element. Price: `sizeof(T) * numel(element)` if intermediate; free only if it is the final required output. Requires scalar int32/int64 position; a Constant position costs 1 param if an initializer/Constant tensor is used. [PRIMARY][PROMPT][DERIVED]
- `ConcatFromSequence(seq) -> tensor(T)`: materializes the full concatenated/stacked tensor. Price: `sizeof(T) * numel(concat_result)` if intermediate; free only if final output. It usually recreates the same tensor a tensor-only `Concat`/`Reshape` would create. [PRIMARY][PROMPT][DERIVED]
- `SequenceLength(seq) -> scalar tensor(int64)`: price 8 bytes if intermediate. It can avoid a 1-element initializer only when used as a final scalar output, which is not the normal grid target. [PRIMARY][PROMPT][DERIVED]
- `OptionalGetElement(optional(tensor(T))) -> tensor(T)`: materializes the contained tensor. Price: `sizeof(T) * numel(tensor)` if intermediate; free only if final output. Empty optional is invalid/undefined for this bridge. [PRIMARY][PROMPT][DERIVED]
- `OptionalHasElement(optional) -> scalar tensor(bool)`: price 1 byte if intermediate. It provides control information, not numeric grid data. [PRIMARY][PROMPT][DERIVED]
- String bridges: `RegexFullMatch` and `StringSplit` counts can emit bool/int64 tensors, but they require string input. No cheap numeric-grid-to-string-to-grid bridge exists in the supported surface. [PRIMARY][DERIVED]
- Sparse bridge: none found in default-domain ORT CPU table. [PRIMARY][DERIVED]

Table B:

| candidate pattern | before tensor-only chain | after container chain | bytes/params saved or lost | fatal caveat | verdict |
|---|---|---|---|---|---|
| Full tensor split and rejoin | `Split`/`Slice` tensors then `Concat`/direct final output | `SplitToSequence` -> `ConcatFromSequence` | Saves no final tensor bytes; `ConcatFromSequence` materializes the same tensor. If intermediate, loses full output bytes. | Sequence intermediate risk; `SplitToSequence` lacks ORT CPU `uint8/int8` input support in 1.24 table | dead |
| Select one slice from a split | `Slice` directly to required tensor | `SplitToSequence` -> `SequenceAt` | At best equal when `SequenceAt` is final; if intermediate, pays selected slice bytes. Position constant costs 1 param. | `Slice` already avoids unused split outputs without containers; sequence intermediate risk | only-if-needed |
| Hide multiple branch tensors in a sequence | `Split` or many separate tensors used later | `SequenceConstruct`/`SplitToSequence`, retrieve pieces with `SequenceAt` | Any retrieved tensor is charged when intermediate. Hidden un-retrieved tensors help only if the tensor-only baseline foolishly materializes unused outputs. | A direct tensor-only rewrite with `Slice`/single-output ops should dominate | only-if-needed |
| Build concat from many constants | `Concat` constants -> tensor | `SequenceEmpty`/`SequenceInsert` or `SequenceConstruct` -> `ConcatFromSequence` | Constant payload elements remain charged; concat result still materializes if intermediate. | Non-tensor sequence surface with no payload savings | dead |
| Generate scalar constants via length | scalar initializer/Constant of 0/1/N | `SequenceEmpty`/`SequenceConstruct` -> `SequenceLength` | Replaces 1 param with an 8-byte intermediate scalar if used downstream; worse under score. | Sequence intermediate risk; final scalar output not a grid | dead |
| Optional identity transfer | direct tensor or `Identity` | `Optional` -> `OptionalGetElement` | Saves 0; unwrapped tensor has same size and same downstream charge | Optional intermediate risk; empty optional is invalid to unwrap | dead |
| Optional boolean guard | scalar bool Constant or existing condition | `OptionalHasElement` | Replaces 1 bool param with 1-byte intermediate in normal use; no grid data | Useful guards need `If`/subgraphs, risky/fatal per prompt | dead |
| String regex/string split features | numeric compare/count chain | cast/encode to string -> `RegexFullMatch`/`StringSplit` | No cheap numeric-to-string bridge; string tensors are not grid outputs | Fatal string surface and wrong data domain | dead |
| `TfIdfVectorizer` as integer-token lookup/count | numeric gather/one-hot/count tensors | int64 grid flattened as tokens -> 1-D float vocabulary vector | Attribute pools/indexes/weights add params; output shape is vocabulary vector, not grid | N-gram semantics, float output, no spatial reconstruction bridge | dead |
| Sparse constant payload | dense Constant/initializer | `Constant(sparse_value)` | Could reduce serialized payload only if scorer undercounts sparse values, but local ORT returns sparse value and no dense bridge exists | Sparse output/intermediate likely rejected; no default dense bridge | dead |

## 5. Ranked verdict

`probe-now`: none.

No sequence/container/string/sparse family has a safe, source-supported path to lower NeuroGolf score under the stated tensor-only final-output requirement. Every apparent win depends on the scorer ignoring non-tensor intermediates while the validator still accepts them, and every useful bridge back to tensors materializes the same bytes later. [PROMPT][PRIMARY][DERIVED]

`only-if-needed`:

- `SplitToSequence` + `SequenceAt`, and `SequenceConstruct`/`SequenceAt`, only for a very specific future case where the current tensor-only graph unavoidably creates multiple charged tensor outputs and a direct `Slice`/single-output rewrite is impossible. Minimal test shape if this ever becomes necessary: opset 11, `float` input `[2,2]`, `SplitToSequence(axis=0, keepdims=1)` -> `SequenceAt(pos=0)` -> final tensor `[1,2]`. Do not try this for `uint8/int8` split input without first retesting ORT support because ORT 1.24's kernel table omits those input types for `SplitToSequence`. [PRIMARY][DERIVED]
- `ConcatFromSequence`, only as the required bridge for an already-existing accepted sequence. It should not be introduced as a cost reducer by itself. Minimal test shape: opset 11, two `float[1,2]` tensors -> `SequenceConstruct` -> `ConcatFromSequence(axis=0)` -> final `float[2,2]`. [PRIMARY][DERIVED]

`dead`:

- `SequenceEmpty`, `SequenceInsert`, `SequenceErase`, and `SequenceLength` as cost reducers. They either leave sequence values exposed or produce only scalar metadata. [PRIMARY][DERIVED]
- `SequenceMap`. It has a required subgraph body and no ORT CPU kernel table entry for 1.24. [PRIMARY][DERIVED]
- All optional wrappers as cost reducers. `OptionalGetElement` is just a bridge back to the same tensor, and `OptionalHasElement` only yields scalar control information. [PRIMARY][DERIVED]
- All listed string ops for numeric grid scoring: `StringNormalizer`, `RegexFullMatch`, `StringConcat`, `StringSplit`, `TfIdfVectorizer`. They operate in string/token-vector space and do not provide a cheap exact grid bridge. [PRIMARY][DERIVED]
- Sparse constants/paths. `Constant(sparse_value)` is not a normal dense tensor bridge in local ORT 1.24.4 behavior, and no default-domain sparse-to-dense CPU bridge was found. [PRIMARY][DERIVED]

Final blunt verdict: keep sequence, optional, sparse, and string/container ops declined for NeuroGolf cost work. The only narrow exception is a future manual experiment with `SplitToSequence -> SequenceAt` when it replaces an otherwise unavoidable multi-output tensor split, but there is no `probe-now` candidate from this card.

## Verification checklist

- [x] All default-domain sequence/container/optional/string families covered.
- [x] Every row distinguishes schema support from ORT CPU kernel support.
- [x] Every non-tensor output names its tensor bridge and prices it.
- [x] Every `probe-now` candidate includes a minimal graph shape/dtype/opset to test locally. There are no `probe-now` candidates.
- [x] Final blunt verdict: sequence/container ops should stay declined.

## Local smoke checks

Environment check:

- `onnxruntime 1.24.4`
- available providers: `AzureExecutionProvider`, `CPUExecutionProvider`
- `onnx 1.21.0`

Temporary ORT CPU smoke results:

- Passed: `SequenceConstruct`, `SequenceEmpty`, `SequenceInsert` + `SequenceAt`, `SequenceErase` + `SequenceLength`, `SequenceLength`, `SequenceAt`, `SplitToSequence` + `ConcatFromSequence`, `Optional` + `OptionalGetElement`, `Optional` + `OptionalHasElement`, `StringConcat`, `RegexFullMatch`, `StringSplit`, `StringNormalizer`, `TfIdfVectorizer`.
- Sparse check: `Constant(sparse_value)` loaded and ran, but returned an ORT sparse runtime value rather than a normal ndarray tensor, reinforcing the sparse-path rejection.
