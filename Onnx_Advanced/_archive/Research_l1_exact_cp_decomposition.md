# Exact CP and Boolean rank decompositions for tiny selector banks

[PROMPT] This note is written for the exact-reconstruction regime you described: tiny static selector-bank tensors used inside F32 `Einsum`, where the cost driver is initializer size, and where two semirings matter operationally. The first is **real CP feeding an F32 einsum**, which allows mixed signs and cancellation as long as every contraction remains exactly representable in binary32. The second is **Boolean OR/AND reconstruction**, relevant when banks are consumed by `BitwiseAnd`-style logic and exact OR-semiring recovery is acceptable.

[PROMPT] One terminology warning matters immediately: classical tensor rank is defined over a **field**, while “rank over ℤ” is not a standard tensor-rank notion. For your use case, “integer rank” is best interpreted as **real CP rank with the added restriction that factor entries (or term coefficients after normalization) lie in a bounded integer or dyadic set**. That restriction is algorithmically useful even though it is not the classical algebraic notion. Schaefer and Stefanković explicitly frame tensor rank over fields such as ℚ, ℝ, and ℂ, and also note that tensor rank can vary with the underlying field. citeturn35view0

## Exact rank of small structured families

[PRIMARY] The strongest clean exact family in your selector-bank regime is the **cyclic/group permutation stack**  
\[
T(k,i,j)=\mathbf 1[j-i\equiv k \pmod n]
\]
which is the order-3 group tensor of the cyclic group \( \mathbb Z_n \). Alexeev, Forbes, and Tsimerman show that for finite abelian groups, over fields admitting the relevant representation-theoretic decomposition, the tensor rank is exactly \(|G|\); in particular, over such fields the cyclic family has rank exactly \(n\). The same paper also gives an interpolation-based upper bound that works over any sufficiently large field, yielding \( \operatorname{rank}(T_{\mathbb Z_n}^3)\le 3(n-1)+1=3n-2 \), and proves a monotone rank of \(n^{d-1}\), hence \(n^2\) in the order-3 cyclic case. citeturn9view2turn8view1turn39view0

[DERIVED] For the **superdiagonal / Kronecker delta tensor**  
\[
\Delta_n(i,j,k)=\mathbf 1[i=j=k],
\]
the exact rank over any field is \(n\): the explicit decomposition is
\[
\Delta_n=\sum_{r=1}^n e_r\otimes e_r\otimes e_r,
\]
and the mode-1 flattening has matrix rank \(n\), so no shorter CP decomposition exists. The same construction is integral and Boolean. On the Boolean side, every rank-1 block contained in the support of \(\Delta_n\) must be a singleton, so the Boolean rank is also \(n\). This is elementary from the rank-1 definition used by Schaefer–Stefanković and the support/flattening arguments just stated. citeturn35view0

[SECONDARY] For **small structured 0/1 matrices**, there is now a tight low-rank picture that is directly relevant to your 30×30 structured-matrix cases. Parnas and Shraibman prove that if a 0/1 matrix has real rank \(d\le 4\), then its Boolean and binary ranks differ from real rank by at most a small constant; for \(d=3,4\), they show
\[
d \le \operatorname{rank}_{\mathrm{bin}}(M)\le 2d-2,\qquad
d-1\le \operatorname{rank}_{B}(M)\le 2d-2,
\]
and the extremal case is achieved by a specific circulant matrix \(C_{2d-2}\). They also show that for \(d=3,4\), that circulant matrix is the unique extremal example of size \((2d-2)\times(2d-2)\). citeturn40view0

[DERIVED] For a **stack of circulant matrices** \(C_k\in\mathbb C^{n\times n}\), simultaneous diagonalization by the DFT gives a constructive CP upper bound of at most \(n\) over \(\mathbb C\): each slice can be written as
\[
C_k = F^{-1}\operatorname{diag}(\lambda_{k,0},\dots,\lambda_{k,n-1})F,
\]
so the 3-tensor of slices is
\[
\mathcal C=\sum_{r=0}^{n-1}\lambda_{:,r}\otimes u_r\otimes v_r.
\]
That is a useful **construction**, but I did **not** locate a general tight real- or rational-rank formula for arbitrary circulant stacks in the public sources I found. The reliable exact statements I did find are the cyclic-group tensor results above and the order-2 circulant matrix results of Parnas–Shraibman. The Fourier-diagonalization fact itself is standard in the circulant-matrix literature. citeturn40view0turn11search7turn10search0

[DERIVED] For **outer-product-plus-corrections** families, the safe exact statement is subadditivity, not a universal closed formula. If
\[
T = \sum_{r=1}^{s} u_r\otimes v_r\otimes w_r + \sum_{\ell=1}^{m} E_{i_\ell j_\ell k_\ell},
\]
then an explicit exact decomposition with \(s+m\) terms always exists, and every flattening gives a certified lower bound. In special incompatible-support situations, each correction singleton is forced to occupy its own Boolean or nonnegative term, but I did not locate a general theorem sharper than those two generic bounds. citeturn35view0

| family / method | exact over which semiring | rank bound / size limit | construction or tool | citation |
|---|---|---:|---|---|
| superdiagonal \( \Delta_n(i,j,k)=\mathbf 1[i=j=k] \) | ℝ / ℚ / integral-restricted / Boolean OR | exact rank \(n\) [DERIVED] | explicit sum \( \sum_r e_r\otimes e_r\otimes e_r \); flattening lower bound; singleton-support Boolean argument | Schaefer–Stefanković 2024 *The Complexity of Tensor Rank* + [DERIVED] citeturn35view0 |
| cyclic shift / group tensor \(T(k,i,j)=\mathbf 1[j-i\equiv k\!\!\!\pmod n]\) over splitting fields | field with full abelian character decomposition [PRIMARY] | exact rank \(n\) | representation/Fourier decomposition of finite abelian group tensors | Alexeev–Forbes–Tsimerman 2011 *Tensor Rank: Some Lower and Upper Bounds* citeturn9view2 |
| same cyclic shift stack over arbitrary sufficiently large fields, including ℝ | ℝ / general fields [PRIMARY] | \(n \le \operatorname{rank}\le 3n-2\) for order 3; exact real/rational value not located | flattening gives lower bound \(n\); interpolation gives upper bound \(3n-2\) | Alexeev–Forbes–Tsimerman 2011 citeturn8view1turn39view0 |
| same cyclic shift stack without cancellation | nonnegative / monotone [PRIMARY]; Boolean OR [DERIVED] | exact rank \(n^2\) in order 3 | AFT prove monotone rank \(n^{d-1}\); same support argument yields Boolean rank \(n^2\) for permutation support | Alexeev–Forbes–Tsimerman 2011 + [DERIVED] citeturn8view1turn8view2 |
| circulant 0/1 matrices of real rank \(d\le 4\) | real vs Boolean / binary matrix ranks [PRIMARY] | for \(d=3,4\): \(d\le \operatorname{rank}_{bin}\le 2d-2\), \(d-1\le \operatorname{rank}_B\le 2d-2\); extremal \(C_{2d-2}\) achieves \(2d-2\) | structural classification plus computer-assisted proof | Parnas–Shraibman 2025 *A Study of the Binary and Boolean Rank of Matrices with Small Constant Real Rank* citeturn40view0 |
| outer-product-plus-corrections | ℝ / ℚ / Boolean OR [DERIVED] | always \( \le s+m \); lower bounded by maximum flattening rank; no general tight formula located | explicit correction-by-singletons; flattening lower bounds | Schaefer–Stefanković 2024 + [DERIVED] citeturn35view0 |

[DERIVED] The direct answer to the “field difference” part is therefore this: **yes, tensor ranks can differ by field in general**, but in the small structured families most relevant here, the only fully tight field-sensitive family I found was the cyclic/group tensor story: exact \(n\) over splitting fields, only upper bounds over arbitrary small fields and over ℚ in the cited sources. For the diagonal tensor there is no gap. For arbitrary permutation stacks beyond the cyclic/group case, I did not locate a general tight real-vs-rational theorem. citeturn35view0turn9view2turn8view1

## Cancellation gains

[PRIMARY] The cleanest published separation answering your “can cancellation beat Boolean/nonnegative rank on 0/1 tensors?” question is again Alexeev–Forbes–Tsimerman’s cyclic group tensor. They prove that for the explicit 0/1 tensor \(T_{\mathbb Z_n}^d\), the unrestricted rank is at most \(d(n-1)+1\), while the monotone rank is exactly \(n^{d-1}\). For order \(d=3\), that is
\[
\operatorname{rank}(T_{\mathbb Z_n}^3)\le 3n-2,\qquad \operatorname{rank}_{+}(T_{\mathbb Z_n}^3)=n^2.
\]
That is a **quadratic-vs-linear** separation. citeturn8view1turn8view2

[DERIVED] Instantiating the theorem at \(n=3\) produces a tiny **3×3×3** 0/1 tensor with nonnegative rank exactly \(9\) and unrestricted rank at most \(7\). So a small tensor can already gain at least **two CP terms** from cancellation, and possibly more depending on the exact real rank. Instantiating at \(n=10\) gives a family with nonnegative rank \(100\) and unrestricted rank at most \(28\). That does **not** prove your specific 10×10×10 bank has rank below its known construction, but it does prove that mixed-sign exact CP can beat nonnegative/Boolean rank by a very large margin even on explicit 0/1 permutation tensors. citeturn8view1turn8view2

[SECONDARY] On the matrix side the gap is much smaller for low real rank. Parnas–Shraibman show that when a 0/1 matrix has real rank \(d\le 4\), the maximal Boolean/binary gap is only additive-constant sized, with extremal value \(2d-2\) for the relevant circulant examples. So if your bank is effectively order-2 after flattening into a structured matrix, there is much less room for spectacular cancellation gains than in genuine order-3 cyclic/permutation tensors. citeturn40view0

[DERIVED] For your concrete “slots \(T=10\ldots18\) are open” question, the literature says **yes, that kind of gap is mathematically plausible in principle**. It does **not** say it is likely for a generic tiny bank, and it does **not** hand you a universal certificate method for unrestricted real CP. The lesson is narrower and still valuable: if your tensor has a permutation/group/circulant skeleton, cancellation can be real and large; if it is closer to a low-real-rank matrix or a diagonal-plus-few-corrections object, the room for dramatic improvement is much smaller. citeturn8view1turn8view2turn40view0

## Exact computation tooling

[PRIMARY] Exact tensor-rank computation is fundamentally hard. Håstad proved NP-hardness over ℚ, and Schaefer–Stefanković sharpened the picture by showing that tensor rank over a field \(F\) is polynomial-time equivalent to the existential theory of that field: \(\exists\mathbb Q\)-complete over ℚ, \(\exists\mathbb R\)-complete over ℝ, and \(\exists\mathbb C\)-complete over ℂ. They also stress that solutions may require algebraic numbers of high complexity. So the right practical question is not “is there a general exact solver?” but “what exact **restricted** search class can public tooling still certify?” citeturn32search3turn35view0

[SECONDARY] For exact Boolean matrix factorization, the public tooling is much better. Günlük, Hauser, and Kovács give compact and exponential integer-programming formulations for rank-\(k\) Boolean matrix factorization under Boolean arithmetic and report a column-generation approach that is competitive on synthetic and real data. Kovács’ thesis gives concrete exact rank-1 IP behavior: on binary matrices classed as “medium” (up to about \(100\times 50\) with Boolean rank at most 20), CPLEX solved the exact 1-BMF IP to optimality in an average of 12.47 seconds, while “large” cases (up to about \(150\times 70\), Boolean rank at most 50) still had only a 3.9% gap after a 3600-second budget. citeturn20view0turn24view0

[SECONDARY] Their later synthetic rank-\(k\) experiments are also useful for scale-setting: for matrices with 20, 35, or 50 rows and 20 columns, and \(k\in\{2,5,10\}\), column-generation plus IP runs under budgets of 600 seconds for LPs and 300 seconds for master IPs, with the best method often solving in seconds to a few minutes. That is matrix-sized, not tensor-sized, but it is the closest public exact benchmark I found for bank-like Boolean problems. citeturn26view0

[SECONDARY] AO-BMF with IP subproblems is useful as an **initializer**, not as an exact certificate. Kolomvakis, Vandaele, and Gillis describe Gurobi-based integer subproblems that handle medium-scale matrices efficiently, reporting, for example, that combining 800 rank-one factors of a \(101\times101\) matrix took about 30 seconds, and that a \(361\times2429\) CBCL data matrix at rank 20 converged in about one hour. That is evidence that modern MIP solvers are strong on exact Boolean subproblems, but it is not a proof method for unrestricted real CP rank. citeturn17view1turn17view0

[SECONDARY] For symbolic exactness over ℚ or ℝ, the main public algebra tools are **Sage + Singular** and **Macaulay2**. Sage’s documentation states that it relies on Singular for Gröbner-basis computation and describes Singular as the best open-source system for these calculations; Macaulay2 is explicitly a system for algebraic geometry and commutative algebra. Those are the right tools when you already have a **small polynomial system** or a **fixed structural ansatz**. I did **not** locate a public benchmark showing them certifying unrestricted CP rank for a free-form 10×10×10 tensor at ranks near 10–20, so any size claim beyond “tiny or strongly templated” should be treated as speculative. citeturn27search1turn27search18turn27search0

[PRIMARY] SMT over nonlinear reals is not the right exact certificate engine for unrestricted CP search. Microsoft’s Z3 guide says nonlinear real arithmetic is “very expensive” and **not complete**, with `check-sat` possibly returning `unknown` or looping. dReal is stronger on nonlinear continuous constraints, but its guarantee is **δ-completeness**, not exact symbolic certification. So SMT is reasonable for bounded rational/integer templates, but not for proving unrestricted real CP infeasibility at a target rank. citeturn36search10turn36search1turn36search12

[SECONDARY] Standard alternating least squares or NMF-style routines are not exact-certification tools. MATLAB’s own `nnmf` documentation explicitly says the factorization is not exact and minimizes an approximation residual. The same caution applies to Tensorlab-style CP routines unless you later perform exact rational reconstruction and symbolic verification. citeturn13search16

| family / method | exact over which semiring | rank bound / size limit | construction or tool | citation |
|---|---|---:|---|---|
| unrestricted tensor-rank decision | ℚ / ℝ / ℂ | exact in principle; complexity is \(\exists\mathbb Q/\exists\mathbb R/\exists\mathbb C\)-complete | existential-theory reduction | Schaefer–Stefanković 2024 *The Complexity of Tensor Rank*; Håstad 1990 *Tensor Rank is NP-Complete* citeturn35view0turn32search3 |
| bounded-integer or bounded-dyadic CP search | real CP with coefficient restrictions [DERIVED] | practical only for tiny/structured tensors; no public 10×10×10 unrestricted benchmark located | MIQCP/MILP in Gurobi/CPLEX; exact only within searched alphabet | supported by exact-optimization literature above + [SPECULATIVE] citeturn35view0turn16view0 |
| exact 1-BMF compact IP | Boolean OR | medium cases up to about \(100\times50\) solved exactly in ~12.47 s avg; large \(150\times70\) still open at 3600 s with 3.9% gap | CPLEX exact IP using McCormick/binary formulation | Kovács 2023 *Binary Matrix Factorisations Under Boolean Arithmetic* citeturn24view0 |
| exact / globally optimized rank-\(k\) BMF | Boolean OR | synthetic matrices of size 20–50 by 20, with \(k\le 10\), handled under 600 s / 300 s budgets | compact IP, exponential IP, column generation | Günlük–Hauser–Kovács 2023 *Binary Matrix Factorization and Completion via Integer Programming*; Kovács thesis citeturn20view0turn26view0 |
| AO-BMF with integer-programmed subproblems | Boolean OR, but not global exactness | \(101\times101\) factor-combination in ~30 s; \(361\times2429\), \(r=20\), ~1 hour | Gurobi-based integer subproblems | Kolomvakis–Vandaele–Gillis 2023 *Algorithms for Boolean Matrix Factorization using Integer Programming* citeturn17view1turn17view0 |
| Gröbner-basis workflow | ℚ / algebraic extensions / ℝ via symbolic elimination | no public tensor-rank size benchmark located; realistically tiny or strongly templated [SPECULATIVE] | Sage + Singular; Macaulay2 | Sage docs; Macaulay2 site citeturn27search1turn27search18turn27search0 |
| SMT on nonlinear arithmetic | bounded real templates only | not a reliable exact CP-rank certificate engine | Z3, dReal | Z3 Guide; dReal docs citeturn36search10turn36search1 |

## Boolean matrix and tensor factorization

[PRIMARY] For 0/1 **matrices**, Boolean rank is the minimum number of all-ones rectangles needed to cover the one-entries, equivalently the biclique-cover number of the associated bipartite graph. Parnas–Shraibman restate these equivalences explicitly, and they are the cleanest exact formulation for bank masks that are really matrices or unfoldings. citeturn40view0

[PRIMARY] For exact matrix BMF, there is a mature line of integer-programming formulations. Lu, Vaidya, and Atluri already modeled optimal Boolean matrix decomposition and its variants as binary integer programs in the role-engineering literature. Kovács’ thesis then substantially sharpened the polyhedral and computational understanding, including exact compact IPs and stronger exponential formulations with column generation. citeturn30view0turn21view0turn20view0

[SECONDARY] For ≤8-class membership masks, the matrix case is especially favorable. If one factor dimension is at most 8, then the possible class-membership patterns per column or fiber are only \(2^8=256\). [DERIVED] That limited pattern space makes exact Boolean searches much easier to warm-start and symmetry-break than generic tensor CP searches. Operationally, this is the best place to exploit exact OR-semiring factorization rather than mixed-sign real CP. The theory above tells you that exact BMF is the right formalism; the small pattern count is the practical bonus. citeturn40view0turn20view0

[SECONDARY] For **Boolean tensor** decomposition, the approximate-tool ecosystem exists, but the exact-certification ecosystem is thin. `TensOrMachine` is a probabilistic Boolean tensor decomposition method; `GETF` is a geometric Boolean tensor decomposition algorithm whose abstract explicitly states that Boolean tensor decomposition is NP-hard and focuses on computational efficiency and reconstruction quality, not exact global certification. I did **not** find a public exact Boolean tensor-rank package with benchmarked certification behavior comparable to the matrix-IP literature. citeturn31academia32turn31academia33

[DERIVED] The practical consequence is simple: for exact OR-semiring bank synthesis, **reduce to matrices whenever you legitimately can**. Use exact BMF on a suitable unfolding if the semantics genuinely factor by a class/membership mode, or use a direct 0/1 MILP over tensor rectangles only when the tensor structure really needs 3-way blocks. For your tiny mask banks, matrix exact-BMF is the mature public path; exact Boolean tensor factorization is still mostly a custom IP exercise.

| family / method | exact over which semiring | rank bound / size limit | construction or tool | citation |
|---|---|---:|---|---|
| rectangle cover / biclique cover | Boolean OR for matrices | exact Boolean rank of matrix | cover 1-entries by all-ones rectangles / bicliques | Parnas–Shraibman 2025 citeturn40view0 |
| optimal Boolean matrix decomposition | Boolean OR for matrices | exact BIP model; size depends on solver budget | binary integer programming for exact decomposition and variants | Lu–Vaidya–Atluri 2008 *Optimal Boolean Matrix Decomposition* citeturn30view0 |
| compact IP for 1-BMF | Boolean OR for matrices | exact; up to about \(100\times50\) comfortable, \(150\times70\) challenging | CPLEX exact rank-1 formulation | Kovács 2023 thesis citeturn24view0 |
| rank-\(k\) BMF with column generation | Boolean OR for matrices | synthetic \(20\!-\!50 \times 20\), \(k\le 10\), minutes-scale budgets | compact/exponential IP + column generation | Günlük–Hauser–Kovács 2023; Kovács thesis citeturn20view0turn26view0 |
| low-real-rank 0/1 matrices | Boolean OR / binary / real | for \(d=3,4\), tight ranks within \(2d-2\) | structural classification + computer search | Parnas–Shraibman 2025 citeturn40view0 |
| Boolean tensor decomposition packages | Boolean OR tensors, approximate only | scalable, but not exact certification | TensOrMachine, GETF | Rukat–Holmes–Yau 2018; Wan et al. 2020 citeturn31academia32turn31academia33 |

## Integer-exact coefficients for F32 einsum

[PRIMARY] IEEE binary32 has a **24-bit significand precision** (23 explicit bits plus the hidden leading bit). The standard practical consequence is that all integers with magnitude up to \(2^{24}\) are exactly representable, and more generally any number with a finite binary expansion within exponent range is exactly representable. Oracle’s IEEE-arithmetic notes and Goldberg’s floating-point survey are the standard references; the binary32 summary agrees with that interpretation. citeturn37search2turn37search3turn37search1

[DERIVED] For your search formulation, the safest sufficient constraint is therefore:
\[
\max_{i,j,k,t}|a_{it}b_{jt}c_{kt}| < 2^{24},
\qquad
\max_{i,j,k}\left|\sum_{t=1}^{T} a_{it}b_{jt}c_{kt}\right| < 2^{24}.
\]
If every multiplicand and every accumulated partial sum stays in that exact-integer window, then reassociation and FMA details do not matter: every intermediate remains exactly representable as an integer in F32.

[DERIVED] If you need rationals, use **dyadic rationals** \(p/2^s\), not arbitrary rationals. Dyadics have finite binary expansions, so they are exactly representable in binary floating-point as long as exponents stay in range. In practice that means searching over a bounded alphabet such as
\[
S=\{0,\pm1,\pm2,\pm\tfrac12,\pm\tfrac14\},
\]
and then clearing shared powers of two across modes of each rank-1 term. Because
\[
(\alpha u)\otimes(\beta v)\otimes(\gamma w)=(\alpha\beta\gamma)\,u\otimes v\otimes w,
\]
you can shift powers of two between factors without changing the represented tensor, which is a useful balancing trick for keeping all factor entries and products small.

[DERIVED] The clean operational rule is therefore: **search over bounded integers first; only then enlarge to bounded dyadics**. Integer alphabets give you the strongest exactness guarantee and the smallest branch-and-bound tree. Dyadics are the right second step when you have evidence that cancellation is real but integral coefficients are too restrictive.

## Worked search recipe for a 10×10×10 bank

[PROMPT] The target case here is exactly the one you described: a 10×10×10 0/1 tensor with a known exact \(T=19\) construction and a flattening rank of 10.

[DERIVED] **First pass: compute every cheap lower bound you can certify.** Record all three matrix flattening ranks, because they are unconditional lower bounds on CP rank. If the support is permutation-like, also test the “one rectangle can cover at most one 1” argument on the Boolean/nonnegative side, since it can instantly rule out OR-semiring compression even when real cancellation remains open. Keep the known \(T=19\) construction as a warm start and as an incumbent upper bound.

[DERIVED] **Second pass: attack the bounded-integer search class before the unrestricted-real class.** For each target \(T=18,17,\dots,10\), solve a feasibility model with
\[
a_{it},b_{jt},c_{kt}\in S,\qquad S=\{0,\pm1\}
\]
first, then \(S=\{0,\pm1,\pm2\}\), then a small dyadic alphabet if needed. Normalize each term so its first nonzero entry in one chosen mode is positive and equal to 1 whenever possible, and lexicographically order term supports to kill the \(T!\) permutation symmetry. Those two symmetry breaks help more than any solver tweak on problems this small.

[DERIVED] **A usable MIQCP sketch for fixed \(T\)** is:

```text
variables a_it, b_jt, c_kt in bounded alphabet S
p_ijt = a_it * b_jt
q_ijkt = p_ijt * c_kt
for all (i,j,k):  sum_t q_ijkt = X_ijk
for all (i,j,k,t): |q_ijkt| <= B^3
for all (i,j,k):   |sum_t q_ijkt| <= 2^24
normalize each term: first_nonzero(a_:t) = 1 and positive
symmetry break: support(a_:1) <=_lex ... <=_lex support(a_:T)
```

[DERIVED] In Gurobi this is a **nonconvex MIQCP** if you keep the bilinear equalities directly. If you want pure MILP, encode each scalar entry as a one-hot choice over the finite alphabet \(S\), then linearize the products with assignment variables. For your sizes, the tensor itself only contributes \(10^3\times T\) product variables, so the model size is large but still in the range where a modern desktop commercial solver is worth trying.

[DERIVED] **Third pass: restrict the dictionary before expanding the alphabet.** Your banks are highly structured, so do not start with free dense factors unless you have to. Build a dictionary of candidate mode vectors from the structures you actually use: standard basis vectors, all-ones vectors, class-indicator vectors, cyclic shifts, diagonal selectors, and any row/column motifs already present in the known \(T=19\) construction. Then solve a set-partitioning or coefficient-selection model over that dictionary. For these selector-bank problems, a good dictionary often beats a much richer coefficient alphabet.

[DERIVED] **Fourth pass: use deletion-and-reoptimize around the known expansion.** Try removing one known term from the \(T=19\) construction and permitting all remaining coefficients/factors inside the chosen integer or dyadic class to reoptimize. Then try removing two, and so on. This is not globally complete, but it is much more likely to find the kinds of cancellation identities that matter for your “slots 10..18” question than a blind free-form search.

[DERIVED] **Fifth pass: interpret certificates honestly.** If the bounded-alphabet MIQCP proves infeasible at \(T=18\), you have certified **no exact decomposition exists in that alphabet class**. You have **not** certified infeasibility over all reals. Schaefer–Stefanković’s complexity result is the reason: unrestricted real-rank infeasibility is an existential-theory-of-reals problem, not a routine MIP proof obligation. A full unrestricted-real certificate only becomes realistic if you already have a fixed support/sign template and can hand the resulting polynomial system to Gröbner-basis elimination. citeturn35view0turn27search1turn27search0

[DERIVED] **A realistic time budget** for a public-tooling run is: about 10–20 minutes for invariants and dictionary extraction; about 30–60 minutes for \(T=18\) with \(S=\{0,\pm1\}\); another 1–2 hours for \(S=\{0,\pm1,\pm2\}\) and one small dyadic alphabet; then stop unless the solver is already finding near-feasible incumbents. That last stopping rule is practical, not theoretical: if a bounded MIP does not show promise quickly on a 10×10×10 structured bank, the unrestricted-real problem is almost never the place where public tooling suddenly becomes easy. [SPECULATIVE] This timing guidance extrapolates from the exact Boolean-matrix benchmarks above rather than from a published tensor-rank benchmark. citeturn24view0turn26view0

## Overall verdict

[PRIMARY][DERIVED] **Blunt verdict:** for arbitrary tiny 10×10×10 0/1 selector banks, **do not expect public tooling to close the gap from a known exact construction down to unrestricted real CP rank on demand**. Tensor-rank exactness over ℚ/ℝ is inherently \(\exists\mathbb Q/\exists\mathbb R\)-hard, public exact software is much better for Boolean **matrices** than for unrestricted **tensors**, and nonlinear SMT is not a complete rescue path. The one regime where it is still practically worth computing is the regime you actually care about operationally: **bounded integer or bounded dyadic coefficients, strong symmetry breaking, and a hand-built structured dictionary**, solved as a **Gurobi bounded-dyadic MIQCP/MILP**. That is the single best method. If that search class fails to beat your known construction quickly, treat the known \(T\) as final for engineering purposes. citeturn35view0turn36search10turn36search1turn16view0turn24view0

## Open questions and limitations

[SECONDARY] I did **not** locate a general tight real- or rational-rank classification for **arbitrary** permutation stacks \(T[k,:,:]=P_k\); the strongest exact published family I found is the cyclic/group case. citeturn9view2turn8view1

[SECONDARY] I did **not** locate a public benchmark showing Gröbner-basis or secant-variety tooling certifying unrestricted CP rank for a free-form **10×10×10** tensor at ranks in the low teens. Sage/Singular and Macaulay2 are the right symbolic tools, but size claims beyond tiny strongly templated systems would be speculative. citeturn27search1turn27search0

[SECONDARY] Exact **Boolean tensor** factorization appears substantially less mature in public tooling than exact Boolean **matrix** factorization. The scalable tensor packages I found are approximate or probabilistic, not certificate-producing global exact solvers. citeturn31academia32turn31academia33

[DERIVED] The most actionable split, therefore, is this: use exact BMF/IP methods aggressively for OR-semiring matrix-like masks; use bounded-integer/dyadic MIQCP only for genuinely 3-way real CP searches; and do not spend serious time chasing unrestricted-real certificates unless a very rigid algebraic template is already visible.