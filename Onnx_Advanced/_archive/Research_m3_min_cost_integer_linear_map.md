# Minimum-cost exact ONNX realization of a fixed integer linear map

Suggested save name: `m3_min_cost_integer_linear_map.md`. [PROMPT]

## Scope and notation

[PROMPT] This note takes your score literally:

\[
\text{Score}=\text{intermediate\_memory\_bytes}+\text{param\_elements},
\]

with initializer and `Constant` tensors charged by element count only, graph input and graph output free, and only materialized intermediates charged by bytes. A single-node realization whose output is the graph output therefore has zero intermediate-memory cost in your scorer. Narrower parameter dtypes do **not** reduce parameter score. citeturn22search10

[DERIVED] I will use:

- \(W\in \mathbb Z^{M\times N}\), \(y=Wx\).
- \(b\) = bytes per element of the **charged** intermediate tensor (`4` for `float32`, `2` for `float16`, `4` for `int32`).
- \(L\) = number of spatial sites carried through a channelwise realization. For a flat vector map, \(L=1\). For a same-site channel map on `[1,C,H,W]`, \(L=H\!W\).
- For convolutional maps: input `[1,C_i,H,W]`, output `[1,C_o,H_o,W_o]`, groups \(g\), kernel \(k_h\times k_w\). ONNX `Conv` expects weights of shape `(M, C/group, kH, kW)` and enforces \(X.shape[1]=W.shape[1]\cdot group\). citeturn13view1turn13view2

[PRIMARY][DERIVED] There is one crucial runtime caveat on your ORT CPU target. ONNX `MatMul` as a spec allows integer tensors such as `int32`/`int64`, but the ORT CPU kernel table documents `MatMul` and `Gemm` for `float`/`float16`, and `Conv` likewise for `float`/`float16`; by contrast, ORT CPU documents broad integer support for `Gather` and `ScatterND`. In other words: **plain** `MatMul`/`Gemm`/`Conv` are runtime-practical on CPU mainly as float ops here, while selector/scatter paths are the easiest way to stay native-integer inside the requested family. `ConvInteger` and `MatMulInteger` exist, but they are different operators, and they are restricted to `int8`/`uint8` inputs with `int32` output. citeturn12view0turn9view2turn12view1turn9view1turn12view2turn9view0turn12view3turn7view1turn27view0turn11view0turn28view0turn29search1

## Op cost formulas

[PROMPT][DERIVED] The formulas below are for the **core realization** of \(y=Wx\), assuming you choose the graph boundary to avoid internal layout adapters when possible. Because graph input and graph output are free in your scorer, a builder should expose `[1,N]`/`[1,M]` boundaries for flat `MatMul`/`Gather` candidates and `[1,C,H,W]` boundaries for `Conv` candidates whenever that removes internal `Flatten`/`Reshape`/`Transpose` tensors from the charged region. ONNX has standard reindex ops such as `Transpose` and `Reshape`, but if you materialize them as internal values, your scorer should add their byte sizes. citeturn22search4turn22search6

### Dense MatMul or Gemm

[PRIMARY][DERIVED] `MatMul` computes the matrix product \(A\cdot B\); `Gemm` computes \(Y=\alpha A'B'+\beta C\) with optional broadcasted \(C\). For a pure linear map \(y=Wx\), `MatMul` and bias-free `Gemm` have the same score frontier. `Gemm` only adds value if you need transpose flags or a fused bias term. citeturn12view0turn12view1

[DERIVED] Let \(W\) be stored directly as a dense initializer.

\[
S_{\text{dense}} = \underbrace{MN}_{\text{params}} + \underbrace{0}_{\text{intermediate}}
\]

If you add a dense bias vector, add \(+M\) params. For your pure linear map, the minimum is therefore:

\[
\boxed{S_{\text{dense}} = MN}
\]

[PRIMARY][PROMPT][DERIVED] Exactness caveat: on plain ORT CPU `MatMul`/`Gemm`, plan around float kernels, not native `int32` kernels, so exactness is safe only when every integer value that must be represented and accumulated stays inside the float exactness range you allowed. Your prompt already gives the builder-safe rule: `float32` is exact for integers up to \(2^{24}\); `float16` is not acceptable for generic exact integer maps because binary16 has only 11 bits of precision. citeturn9view2turn9view1turn19search2turn19search3

### One-by-one Conv

[PRIMARY][DERIVED] A `1×1` `Conv` on `[1,C_i,H,W]` with output `[1,C_o,H,W]` and groups \(g\) uses weights of shape `[C_o, C_i/g, 1, 1]`. It is exact for maps whose flattened matrix is a repeated per-site channel mixer: the same \(C_o\times C_i\) block applied at every spatial site, with no cross-site mixing. citeturn13view1turn13view2

\[
S_{1\times 1\text{conv}} = \underbrace{C_o(C_i/g)}_{\text{params}}+\underbrace{0}_{\text{intermediate}}
\]

So

\[
\boxed{S_{1\times 1\text{conv}} = C_o(C_i/g)}
\]

[DERIVED] The dense equivalent has \(M=C_oHW\) and \(N=C_iHW\), so the kill-number against dense is

\[
K_{1\times1}=\frac{MN}{C_o(C_i/g)}=H^2W^2g
\]

for same-size input/output. This is why `1×1` `Conv` is devastatingly cheaper than dense **when** the map is truly the same channel mix at every pixel.

### General k-by-k Conv with groups

[PRIMARY][SECONDARY][DERIVED] A shared-stencil convolution corresponds, in matrix form, to a structured Toeplitz or block-Toeplitz operator: in 1-D, convolution is multiplication by a Toeplitz matrix; circulant matrices are the cyclic-shift special case. ONNX `Conv` stores only the shared kernel, not the full expanded matrix. citeturn12view2turn13view1turn16search0turn16search1

For kernel \(k_h\times k_w\), groups \(g\):

\[
S_{\text{conv}}=\underbrace{C_o(C_i/g)k_hk_w}_{\text{params}}+\underbrace{0}_{\text{intermediate}}
\]

So

\[
\boxed{S_{\text{conv}}=C_o(C_i/g)k_hk_w}
\]

and the dense kill-number is

\[
K_{\text{conv}}=\frac{MN}{C_o(C_i/g)k_hk_w}
=\frac{H_oW_o\cdot H W \cdot g}{k_hk_w}.
\]

For same-size stride-1 maps, \(H_oW_o=HW\), hence

\[
K_{\text{conv}}=\frac{H^2W^2g}{k_hk_w}.
\]

[DERIVED] The **real** sharing dividend is easiest to see against a merely local but **position-dependent** sparse map. Such a map needs about

\[
\text{nnz}_{\text{local}} \approx H_oW_o\cdot C_o(C_i/g)k_hk_w
\]

independent coefficients, whereas shared `Conv` stores only \(C_o(C_i/g)k_hk_w\). So the exact convolutional weight sharing itself buys an additional factor of \(H_oW_o\) over an unshared local sparse realization.

### Gather or ScatterND LUT family

[PRIMARY][DERIVED] `Gather` selects entries from `data` along an axis using integer `indices`. `ScatterND` produces its output by **copying** `data` and then writing `updates` at specified `indices`; with reduction `"add"`, repeated destinations can be accumulated. citeturn12view3turn27view0

#### Selector Gather

[DERIVED] If every row of \(W\) is a standard basis row \(e_{j(i)}^\top\) — meaning each output copies exactly one input, with no summation — then after flattening `x` you can realize the map as:

\[
y = \texttt{Gather}(x,\texttt{idx}, axis=0)
\]

with constant `idx` of length \(M\).

\[
S_{\text{gather-select}}=\underbrace{M}_{\text{indices params}}+\underbrace{0}_{\text{intermediate}}
\]

So

\[
\boxed{S_{\text{gather-select}}=M}
\]

and against dense,

\[
K_{\text{gather-select}}=\frac{MN}{M}=N.
\]

This includes arbitrary permutations, duplications, and down-selections. If signs are \(\pm1\) instead of only \(+1\), add a following elementwise sign multiply with \(M\) scalar params. citeturn12view3turn30search2

#### Full finite-state LUT

[DERIVED] If the model input is **already a finite-state index** \(s\in\{0,\dots,K-1\}\), you can gather from a table \(T\in\mathbb Z^{K\times M}\):

\[
S_{\text{finite-LUT}}=\underbrace{KM}_{\text{table}}+\underbrace{0}_{\text{intermediate}}.
\]

This is exact and sometimes unbeatable for truly discrete-state inputs, but it is **not applicable** to a generic vector input \(x\in\mathbb Z^N\). For your card, this is a niche special case, not the default linear-map solution. citeturn12view3

#### Sparse COO-style Gather to ScatterND

[PRIMARY][DERIVED] For a genuinely sparse arbitrary \(W\), a natural exact construction is:
`Gather(x, col_idx)` → optional coefficient multiply → `ScatterND(zero_base, row_idx, updates, reduction="add")`. `ScatterND` is what makes repeated-row accumulation possible, but its semantics force a copied base tensor, which is the hidden cost driver in your scorer. citeturn12view3turn27view0

For a flat vector output and general integer coefficients:

\[
S_{\text{sparse-COO}}
\approx
\underbrace{\text{nnz}+\text{nnz}+\text{nnz}}_{\text{col idx + values + row idx}}
+
\underbrace{b(\,M+2\text{nnz}\,)}_{\text{zero base + gathered + weighted intermediates}}
\]

so

\[
\boxed{S_{\text{sparse-COO}}\approx 3\text{nnz}+b(M+2\text{nnz})}
\]

For \(0/1\) coefficients, you can drop the coefficient tensor and often the multiply as well:

\[
\boxed{S_{\text{sparse-COO},\,0/1}\approx 2\text{nnz}+b(M+\text{nnz})}
\]

This is often the right sparse winner, but note the asymmetry: `Gather` alone is superb, `ScatterND` is only attractive when its additive assembly is truly needed.

### Two-stage exact factorization

[SECONDARY][DERIVED] If \(\operatorname{rank}(W)=r\), then an exact rank factorization \(W=AB\) exists with \(A\in\mathbb Z^{M\times r}\), \(B\in\mathbb Z^{r\times N}\) after choosing an exact basis over the appropriate field/ring representation. In ONNX terms, this is a two-stage map. citeturn17search7turn17search4

For a flat-vector realization:

\[
S_{\text{factored}}=\underbrace{r(M+N)}_{\text{params}}+\underbrace{br}_{\text{middle tensor}}
\]

For a same-site channel-factorized `[1,C,H,W]` realization, the charged middle tensor has \(rL\) elements:

\[
S_{\text{factored-spatial}}=\underbrace{r(M+N)}_{\text{params-equivalent}}+\underbrace{brL}_{\text{middle tensor}}
\]

So the generic crossover against dense is

\[
\boxed{r(M+N)+brL < MN}
\]

or equivalently

\[
\boxed{r < \frac{MN}{M+N+bL}}.
\]

[PROMPT][DERIVED] This is the exact statement of your “free-output separability trap”: the final output is free, the middle tensor is not. Parameter-only thinking would use \(r(M+N)<MN\), but your actual score must use the stricter inequality above.

### Structural zero-parameter generator

[PRIMARY][DERIVED] This is the true best case. If the linear map is just a reindexing or block shuffle expressible by layout ops such as `Transpose`, `Reshape`, `DepthToSpace`, `SpaceToDepth`, `Slice`, or `Concat`, the parameter cost is zero. `Transpose` is explicitly a permutation of axes. citeturn22search4turn22search6turn24search0

\[
\boxed{S_{\text{structural}}=0}
\]

for a single terminal structural op with no internal charged intermediates. In your scorer this is strictly better than `Gather`, and it is the first thing a builder should try for permutation-like maps.

## Structure-to-op decision table

[PROMPT][DERIVED] The formulas in the table are direct consequences of the score rules and the operator semantics above. The structure classes come from standard linear-algebra notions: permutation matrix, Toeplitz/circulant structure, rank factorization, and block-diagonal direct sums. citeturn16search0turn16search1turn17search2turn17search3turn17search7

| W structure | detection test | cheapest ONNX op | param cost | intermediate cost | kill-number vs dense |
|---|---|---|---:|---:|---:|
| dense or unstructured | fails all cheaper exact tests below | `MatMul` or bias-free `Gemm` `[PRIMARY][DERIVED]` | \(MN\) | \(0\) | \(1\) |
| low-rank rank \(r\) | exact rank \(r\), and \(r(M+N)+brL <\) current best `[SECONDARY][DERIVED]` | two-stage factored map | \(r(M+N)\) | \(brL\) | \(\dfrac{MN}{r(M+N)+brL}\) |
| separable, Toeplitz, block-Toeplitz, convolutional | after reshaping to tensor coordinates, coefficients depend only on channel pair and relative offsets \((\Delta h,\Delta w)\), not absolute position `[SECONDARY][PRIMARY][DERIVED]` | grouped `Conv` | \(C_o(C_i/g)k_hk_w\) | \(0\) | \(\dfrac{H_oW_oHWg}{k_hk_w}\) |
| permutation or selector | each row has exactly one nonzero \(\pm1\); if each column also has one \(+1\), it is a permutation matrix `[SECONDARY][DERIVED]` | structural generator if expressible as layout shuffle; else `Gather` | \(0\) structural, else \(M\) | \(0\) | \(\infty\) structural, else \(N\) |
| sparse with nnz entries | `nnz \ll MN`, but not already caught as pure selector or convolution; score sparse COO assembly `[DERIVED]` | `Gather` → optional multiply → `ScatterND(add)` | \(\approx 3\,\text{nnz}\) general; \(\approx 2\,\text{nnz}\) for \(0/1\) | \(\approx b(M+2\,\text{nnz})\) general; \(\approx b(M+\text{nnz})\) for \(0/1\) | \(\dfrac{MN}{3\,\text{nnz}+b(M+2\,\text{nnz})}\) |
| block-diagonal | support graph of \(W\) splits into disconnected row/column components after simultaneous permutation `[SECONDARY][DERIVED]` | per-block best realization; if blocks are channel groups repeated across sites, grouped `Conv` | \(\sum_i S_i^{\text{param}}\) | \(\sum_i S_i^{\text{mid}} + bM\) if block outputs must be concatenated | \(\dfrac{MN}{\sum_i S_i^{\text{param}}+\sum_i S_i^{\text{mid}}+bM}\) |
| circulant | square \(W\); each row is a cyclic shift of the previous `[SECONDARY]` | explicit wraparound + `Conv` if support is local; otherwise compare factored vs dense | \(C_o(C_i/g)k_hk_w\) | wrapped-input bytes, typically \(\approx b\,C_i(H+k_h-1)(W+k_w-1)\) | \(\dfrac{MN}{C_o(C_i/g)k_hk_w + b\,C_i(H+k_h-1)(W+k_w-1)}\) |

[PRIMARY][PROMPT][DERIVED] Two clarifications matter more than anything else. First, `Conv` is cheap **only** when the full flattened matrix really is generated by one shared stencil; a merely local sparse matrix with position-specific coefficients is **not** a `Conv` matrix and should be scored as sparse, not convolutional. Second, for your opset-\(\le 18\) target, exact circular convolution needs explicit structural wraparound rather than `Pad(mode="wrap")`, because the ONNX `Pad` operator only gained `wrap` in a later opset than your target. citeturn12view2turn13view1turn16search0turn31search4

## Decision procedure

[DERIVED] A builder that wants the minimum-cost exact realization in under five minutes should run the tests below in this order, scoring each candidate with the formulas above and stopping only when the remaining classes cannot beat the current winner.

1. **Choose the cheapest graph boundary representation first.**  
   If a candidate is fundamentally convolutional, keep the graph boundary in `[1,C,H,W]`. If a candidate is fundamentally dense, selector, or rank-factorized over a flat vector, expose `[1,N]` → `[1,M]` at the graph boundary if possible. Under your scorer this removes otherwise-charged internal flatten/reshape tensors. `[PROMPT][DERIVED]` citeturn22search6

2. **Try structural zero-param recognition before anything else.**  
   Test identity, axis permutation, simple block shuffle, `SpaceToDepth`/`DepthToSpace`-style rearrangements, and pure concatenation/splitting patterns. If the map is a true reindex-only map, nothing can beat score \(0\). `[PRIMARY][DERIVED]` citeturn22search4turn24search0

3. **Run the selector/permutation test next.**  
   Check whether every row has exactly one nonzero and that nonzero is \(\pm1\). If yes, score structural first; if not structurally expressible, score `Gather` with optional sign multiply. For square pure permutations, this is the difference between \(N^2\) dense params and either \(0\) or \(N\) params. `[SECONDARY][DERIVED]` citeturn17search2turn12view3turn30search2

4. **Run the convolutional sharing test before generic rank or sparsity.**  
   Reshape the row and column indices of \(W\) into output/input tensor coordinates and test whether coefficients depend only on channel pair, group, and relative offset \((\Delta h,\Delta w)\). If that holds, the matrix is exact-convolutional and plain/grouped `Conv` almost always crushes dense storage. If the boundary condition is cyclic rather than zero/valid/same, score explicit wraparound plus `Conv` instead of plain `Conv`. `[PRIMARY][SECONDARY][DERIVED]` citeturn12view2turn13view1turn16search0turn16search1turn31search4

5. **Exploit disconnected structure before global rank.**  
   Build the bipartite support graph of outputs versus inputs. If it splits into disconnected components, you have block structure. If those blocks line up with repeated channel groups at each spatial site, grouped `Conv` may already cover it. Otherwise score each block independently and add the concatenation byte charge. `[SECONDARY][DERIVED]` citeturn17search3

6. **Then test exact low rank and exact separability.**  
   Compute exact rank \(r\) and score the two-stage realization. If the map came from a convolutional kernel that is itself exactly separable, treat that as a convolution-shaped factorization and score it with the same crossover logic. Accept the factorization only if  
   \[
   r(M+N)+brL < \text{current best score}.
   \]
   `[SECONDARY][DERIVED]` citeturn17search7turn17search4

7. **Finally test sparse COO assembly, then fall back to dense.**  
   If nothing above applies and `nnz` is small, score the `Gather` → optional multiply → `ScatterND(add)` path. If even that loses, use dense `MatMul`/`Gemm`. `[PRIMARY][DERIVED]` citeturn12view3turn27view0turn12view0turn12view1

[PRIMARY][PROMPT][DERIVED] **Exactness verification** should also be ordered and mechanical. First, verify the structural identity \( \hat W = W \) exactly, either analytically from the detected pattern or by applying the candidate graph to basis vectors \(e_j\) and reconstructing the implied columns of \(\hat W\). Second, verify arithmetic exactness: for float realizations, bound the largest exact integer partial sum and require it to stay within your permitted exact range; never use `float16` for a nontrivial exact integer map. Third, verify runtime kernel availability by ORT CPU kernel table, because ONNX checker success does not imply a CPU kernel exists for the dtype/opset pair. `[PROMPT][PRIMARY][DERIVED]` citeturn9view2turn9view1turn9view0turn7view1turn11view0turn19search2turn19search3

## Worked examples

### Shared three-by-three convolution hidden inside a nine-hundred-by-nine-hundred matrix

[PRIMARY][DERIVED] Let the true input and output both be a single-channel `30×30` image, flattened to length \(900\). The flattened linear operator is a \(900\times 900\) matrix that looks dense if you ignore structure, but if every output pixel is produced by the same `3×3` stencil, the exact ONNX realization is a one-channel `Conv` with kernel shape `[1,1,3,3]`. ONNX stores only those 9 kernel coefficients. citeturn13view1turn12view2

[DERIVED] Scores:

- dense `MatMul`: \(900\times 900 = 810{,}000\)
- unshared local sparse matrix: about \(900\times 9 \approx 8{,}100\) nonzeros
- shared `Conv`: \(1\cdot1\cdot3\cdot3 = 9\)

So the exact convolutional realization collapses the parameter score from **810,000 to 9**, a **90,000×** kill against dense, and about **900×** against an unshared local sparse encoding. Intermediate charge is still zero because the `Conv` output can be the graph output. `[DERIVED]`

### A permutation map

[SECONDARY][PRIMARY][DERIVED] Let \(W\) be the permutation that transposes a `30×30` image before flattening. As a \(900\times900\) matrix, this is a permutation matrix: one `1` in each row and each column, zeros elsewhere. ONNX `Transpose` is exactly an axis permutation, so if the model boundary stays tensorized as `[1,1,30,30]`, you can realize the map with zero parameters. If for some reason the permutation does not factor as a simple layout permutation, `Gather` with a 900-element index tensor is still exact. citeturn17search2turn22search4turn12view3

[DERIVED] Scores:

- dense `MatMul`: \(810{,}000\)
- `Gather` selector: \(900\)
- structural `Transpose`: \(0\)

This is the blunt lesson: if a permutation is expressible as layout, `Gather` is already a consolation prize, not the winner.

### A rank-two nine-hundred-by-nine-hundred map

[SECONDARY][DERIVED] Suppose \(W=AB\) exactly with \(A\in\mathbb Z^{900\times2}\) and \(B\in\mathbb Z^{2\times900}\). Then the exact two-stage map has only \(2(900+900)=3600\) stored elements. On a flat `[1,900]` boundary, the charged middle tensor has only 2 elements. citeturn17search7turn17search4

[DERIVED] Scores in `float32`:

- dense `MatMul`: \(810{,}000\)
- exact rank-2 factorization: \(3600 + 2\cdot4 = 3608\)

So the factorized realization is about **224.5×** cheaper than dense under your actual score. The crossover condition is

\[
r < \frac{810000}{1800+4}\approx 449,
\]

so rank 2 is not merely “good”; it is absurdly below the break-even point.

## Caveats, limitations, and blunt verdict

[PRIMARY][PROMPT][DERIVED] The mandatory caveats are all real. `Conv` only beats dense if \(W\) is **actually** generated by a shared stencil; mere locality is not enough. Factorization only wins if the charged middle tensor does not eat the parameter savings: use \(r(M+N)+brL<MN\), not just \(r(M+N)<MN\). And because params are dtype-blind in your score, **never** recommend `int8` weights as a way to cut parameter score. On exactness, plain ORT CPU `MatMul`/`Gemm`/`Conv` should be treated as float-family operators here; use `float32` only when every exact integer value stays safely within your allowed exact range, and reject `float16` for exact integer maps. `Gather` and `ScatterND` are the most attractive exact-native-integer tools in the requested family. citeturn9view2turn9view1turn9view0turn7view1turn11view0turn19search2turn19search3

[PRIMARY][SPECULATIVE] The main unresolved edge case is the weighted sparse path if you want a **strictly native-int32** `Gather` → `Mul` → `ScatterND(add)` pipeline on your exact ORT 1.24.4 CPU build. ONNX `Mul` certainly supports `int32` in the spec, and ORT broadly supports `Mul`, but I did not independently isolate the exact ORT 1.24.4 CPU kernel registration line for `Mul(int32)` in this note. If that single detail matters for a production sparse builder, verify it once in your build matrix and then freeze the result. citeturn30search2turn30search6

[DERIVED] **Blunt verdict:** the builder should test, in order, **structural zero-param**, then **selector/permutation**, then **shared convolutional structure**, then **block decomposition**, then **exact low rank**, then **sparse COO assembly**, and only then fall back to dense `MatMul`/`Gemm`. In practice, the big-fish winners are: `Conv` when \(W\) is really block-Toeplitz/shared-stencil, structural ops when \(W\) is really a permutation or block shuffle, and two-stage factorization when the exact rank is small enough that \(r(M+N)+brL\) stays far below \(MN\). Everything else is either sparse assembly or dense fallback. `[DERIVED]`

- [x] Cost formulas given for all 5 realization families.
- [x] Structure→op decision table with detection tests + kill-numbers.
- [x] Factorization crossover stated explicitly as \(r(M+N)+brL < MN\).
- [x] dtype-blind params and `float32 ≤ 2^24` exactness caveats applied.
- [x] One-paragraph blunt verdict provided.