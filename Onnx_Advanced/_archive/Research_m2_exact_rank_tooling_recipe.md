# Exact Rank Tooling Recipe for Selector-Bank CP Decomposition

Suggested save name: `m2_exact_rank_tooling_recipe.md`.

This note is the execution follow-through to L1. It takes L1’s theory as given and turns it into a concrete, runnable recipe for exact bounded-alphabet CP feasibility on small selector-bank tensors, with a frank solver verdict for the public tools you named. The guiding assumption is the one from your prompt: what matters operationally is **exact** bounded-integer search first, then wider bounded alphabets, with symmetry breaking and a structured dictionary before free dense factors. [PROMPT]

## Solver shortlist and verdict

| solver | expresses bounded-int CP? | exact arith? | est. scaling @10x10x10 | license | verdict |
|---|---|---|---|---|---|
| Gurobi | Yes, either as non-convex bilinear MIQCP or as a stronger linearized MILP over small alphabets | **Qualified only**: a found integer witness can be re-checked exactly, but solver feasibility/infeasibility is tolerance-based | Usually the best **incumbent finder** of the four on this model family; strongest if you care about speed more than strict certificates | Commercial; free academic license with full features and no model-size limit for academics | Excellent scout, **not** the exact certificate engine |
| OR-Tools CP-SAT | Yes, naturally, using bounded `IntVar`s plus table/product constraints and exact linear equalities | **Yes for the bounded int64 model**; solver semantics are integer-exact, and CP-SAT’s LP machinery is only used to derive exact integer propagation/cuts | Best practical strict-exact choice at `10×10×10` with `R≈10…18`; tens of thousands of small-domain vars are normal territory | Apache 2.0, free | **Winner** under your exactness contract |
| Z3 | Yes, but only sanely if you encode the bounded alphabet explicitly; native nonlinear integer arithmetic is the wrong default | **Yes** for `Int`/bit-vector semantics, but native nonlinear arithmetic is incomplete/fragile and can return `unknown` or fail to terminate | Good for tiny or highly support-fixed cases; poor as the main engine for free `10×10×10×R` search | MIT, free | Tiny-instance oracle / debugging tool, not the workhorse |
| Sage + Singular or Macaulay2 | Yes for **fixed-support algebraic certificates** after you freeze supports/sign patterns; not good for open support search | **Yes**, symbolic exact algebra over `QQ`/polynomial rings | Useful only after combinatorial support search has collapsed the problem size; global free search is not realistic | GPL, free | Use only for fixed-support elimination/certificates |

The feature basis for that table is straightforward. Gurobi explicitly supports bilinear and other non-convex quadratic constraints, and its non-convex MIQCP machinery aims at global optimality, but always **subject to feasibility and integrality tolerances**; its own numeric guide is explicit that slight infeasibility can sit inside a tolerance gray zone. Gurobi’s academic program is free and unrestricted in model size for qualifying academics. [PRIMARY: Gurobi Optimizer Reference Manual, “Constraints,” “Tolerances and User-Scaling,” “Parameter Reference,” “Downloads / Academic Licenses”] citeturn8view0turn6view0turn8view1turn6view2

OR-Tools CP-SAT, by contrast, is an **integer-only** solver. The docs say CP-SAT works over integers only; the Python API exposes `add_multiplication_equality`, `add_allowed_assignments`, `add_hint`, and exact status naming; and Google’s CP-SAT architecture slides state that propagation is kept exact with int64 arithmetic and “no epsilon,” with LP output used only as a hint that is converted back into exact integer reasoning. CP-SAT is free under Apache 2.0. [PRIMARY: OR-Tools CP-SAT docs, `cp_model` API, `sat_parameters.proto`; SECONDARY: Perron et al. CP-SAT slides] citeturn2view3turn3view0turn31view0turn31view1turn31view3turn33view0turn34view0turn28view0

Z3 is MIT-licensed and exact at the logic level, but the official Z3 arithmetic guide is blunt: nonlinear real arithmetic is very expensive and incomplete, while nonlinear integer arithmetic is undecidable in general; native `QF_NIA` is therefore the wrong top-level encoding for your free bounded CP search. [PRIMARY: Z3 GitHub README; Z3 Guide, Arithmetic / Non-linear arithmetic] citeturn11view2turn11view0

For the algebraic option, SageMath is GPL-licensed, includes interfaces to Singular, and Singular is a GPL computer algebra system centered on polynomial computation and Gröbner bases; Macaulay2 is likewise an open-source algebraic-geometry system with Gröbner-basis support. Those are exact symbolic tools, but they become attractive only after the support/sign combinatorics have already been fixed or crushed. [PRIMARY: SageMath site and docs; Sage interface to Singular; Singular site; Macaulay2 site and GitHub] citeturn21view1turn21view0turn24search1turn24search10turn19view1turn19view2turn19view0turn19view3

**Verdict.** If you relax your exact-certification requirement, Gurobi is the speed-biased scout I would also run. But under your stated rules — exact witnesses, exact bounded-alphabet infeasibility interpretation, and no “float LP then round” certificates — **OR-Tools CP-SAT is the single solver I would actually recommend first**. That is a derived engineering judgment from the documented solver semantics above, not a claim that a published paper benchmarked this exact tensor-rank search head-to-head. [DERIVED]

## Recommended CP-SAT recipe

The right model for your real/signed F32-Einsum case is a **fixed-rank feasibility problem** over a small symmetric alphabet `S`, solved descending in rank `R`. For a target tensor `T[k,i,j]`, you search for factor matrices

\[
A \in S^{K \times R}, \quad B \in S^{I \times R}, \quad C \in S^{J \times R}
\]

such that

\[
T[k,i,j] = \sum_{r=0}^{R-1} A[k,r]\,B[i,r]\,C[j,r]
\]

for every entry, **exactly**. That is your exact bounded-integer CP model. The symmetry-breaking package from L1 carries over directly: normalize the first nonzero in each `A[:,r]` and `B[:,r]` column to be positive, and order rank-1 terms by lexicographically ordered support patterns. Because your alphabets are small and discrete, CP-SAT should not use floating quadratic constraints here; it should use **explicit bounded integer domains plus exact table/product constraints**. [PROMPT] [PRIMARY: OR-Tools integer-only CP-SAT model; table and multiplication API] citeturn2view3turn3view0turn33view0

For the F32 consumer, the conservative exactness guard is: every accumulated partial sum along the rank index must remain an exactly representable binary32 integer. Since binary32 exactly represents all consecutive integers through \(2^{24}\), it is enough to enforce prefix sums inside `[-2^24, 2^24]`. With `S={0,±1}` or `S={0,±1,±2}`, pairwise and triple products are tiny, so the real issue is accumulation, not the multiplicative primitives. [PROMPT] [SECONDARY: MathWorks `flintmax("single")`; binary32 precision references] citeturn32search4turn32search0

### Fully runnable winner script

Copy this into a file such as `exact_cp_cpsat.py`, paste your actual tensor into `T`, and run it with Python 3. It does four things in one pass: exact flattening-rank lower bound, bounded-alphabet CP-SAT model build, F32-safe prefix guards, and exact witness re-check in Python integers.

```python
#!/usr/bin/env python3
"""
Exact bounded-alphabet CP feasibility for a 3-D tensor T[k][i][j].

Winner recipe:
- exact integer domains
- small-alphabet table constraints for products
- first-nonzero-positive normalization on A and B columns
- lexicographic support ordering by packed support code
- exact binary32-safe prefix-sum guards |prefix| <= 2^24
- exact post-check in Python integers

Requires:
    python -m pip install ortools sympy
"""

from ortools.sat.python import cp_model
from sympy import Matrix
import sys

# =========================
# User input
# =========================

# Replace this with your actual selector bank as nested lists T[k][i][j].
# Example placeholder: 10x10x10 all-zero tensor.
T = [[[0 for _ in range(10)] for _ in range(10)] for _ in range(10)]

RANK_TARGET = 18              # try 18, then 17, ..., down to 10
ALPHABET = [-1, 0, 1]        # first pass; then widen to [-2, -1, 0, 1, 2]
TIME_LIMIT_S = 1800          # 30 min per run is a good starting point
NUM_WORKERS = 8              # set explicitly
B32_INT_EXACT = 2**24        # exact consecutive integer range for binary32
LOG_PROGRESS = True

# =========================
# Exact lower bound
# =========================

def shape3(t):
    K = len(t)
    I = len(t[0])
    J = len(t[0][0])
    return K, I, J

def flatten_mode(t, mode):
    K, I, J = shape3(t)
    if mode == 0:   # K x (I*J)
        return [[t[k][i][j] for i in range(I) for j in range(J)] for k in range(K)]
    if mode == 1:   # I x (K*J)
        return [[t[k][i][j] for k in range(K) for j in range(J)] for i in range(I)]
    if mode == 2:   # J x (K*I)
        return [[t[k][i][j] for k in range(K) for i in range(I)] for j in range(J)]
    raise ValueError("mode must be 0, 1, or 2")

def flattening_lower_bound(t):
    r0 = Matrix(flatten_mode(t, 0)).rank()
    r1 = Matrix(flatten_mode(t, 1)).rank()
    r2 = Matrix(flatten_mode(t, 2)).rank()
    return max(r0, r1, r2), (r0, r1, r2)

lb, flat_ranks = flattening_lower_bound(T)
print(f"flattening ranks = {flat_ranks}, lower bound = {lb}")
if RANK_TARGET < lb:
    print(f"Abort: target rank {RANK_TARGET} is below unconditional flattening lower bound {lb}.")
    sys.exit(0)

# =========================
# Model build
# =========================

K, I, J = shape3(T)
R = RANK_TARGET
S = sorted(ALPHABET)
if 0 not in S:
    raise ValueError("Alphabet must contain 0.")
if any(-s not in S for s in S):
    print("Warning: alphabet is not sign-symmetric; A/B sign normalization may not be safe.")

max_abs_s = max(abs(s) for s in S)
force_first_one = (max_abs_s == 1)

model = cp_model.CpModel()
dom_S = cp_model.Domain.from_values(S)

# Free dense factors
A = [[model.new_int_var_from_domain(dom_S, f"A_{k}_{r}") for r in range(R)] for k in range(K)]
B = [[model.new_int_var_from_domain(dom_S, f"B_{i}_{r}") for r in range(R)] for i in range(I)]
C = [[model.new_int_var_from_domain(dom_S, f"C_{j}_{r}") for r in range(R)] for j in range(J)]

def channel_nz_and_pos(X, prefix):
    rows = len(X)
    cols = len(X[0])
    nz = [[model.new_bool_var(f"{prefix}_nz_{u}_{r}") for r in range(cols)] for u in range(rows)]
    pos = [[model.new_bool_var(f"{prefix}_pos_{u}_{r}") for r in range(cols)] for u in range(rows)]
    tuples = [(s, int(s != 0), int(s > 0)) for s in S]
    for u in range(rows):
        for r in range(cols):
            model.add_allowed_assignments([X[u][r], nz[u][r], pos[u][r]], tuples)
    return nz, pos

A_nz, A_pos = channel_nz_and_pos(A, "A")
B_nz, B_pos = channel_nz_and_pos(B, "B")
C_nz, C_pos = channel_nz_and_pos(C, "C")

def normalize_first_nonzero_positive(X, prefix, force_one=False):
    rows = len(X)
    cols = len(X[0])
    for r in range(cols):
        first = [model.new_bool_var(f"{prefix}_first_{u}_{r}") for u in range(rows)]
        model.add_exactly_one(first)
        for u in range(rows):
            if force_one:
                model.add(X[u][r] == 1).only_enforce_if(first[u])
            else:
                model.add(X[u][r] >= 1).only_enforce_if(first[u])
            for v in range(u):
                model.add(X[v][r] == 0).only_enforce_if(first[u])

# Safe symmetry breaking for sign-symmetric alphabets:
# normalize A and B; leave C free to absorb residual sign.
normalize_first_nonzero_positive(A, "A", force_one=force_first_one)
normalize_first_nonzero_positive(B, "B", force_one=force_first_one)

# Forbid zero C columns (otherwise an entire rank-1 term is wasted).
for r in range(R):
    model.add(sum(C_nz[j][r] for j in range(J)) >= 1)

# Lexicographic support ordering, packed into one int64 key when K+I+J <= 62.
L = K + I + J
if L > 62:
    raise ValueError("K+I+J > 62: split support key into chunks before using this shortcut.")
weights = [1 << (L - 1 - p) for p in range(L)]
support_code = [model.new_int_var(0, (1 << L) - 1, f"support_code_{r}") for r in range(R)]
for r in range(R):
    bits = [A_nz[k][r] for k in range(K)] + [B_nz[i][r] for i in range(I)] + [C_nz[j][r] for j in range(J)]
    model.add(support_code[r] == sum(weights[p] * bits[p] for p in range(L)))
for r in range(R - 1):
    model.add(support_code[r] >= support_code[r + 1])

# Exact product tables over the small alphabet
pair_vals = sorted({a * b for a in S for b in S})
term_vals = sorted({ab * c for ab in pair_vals for c in S})
dom_pair = cp_model.Domain.from_values(pair_vals)
dom_term = cp_model.Domain.from_values(term_vals)

pair_tuples = [(a, b, a * b) for a in S for b in S]
term_tuples = [(ab, c, ab * c) for ab in pair_vals for c in S]

AB = [[[model.new_int_var_from_domain(dom_pair, f"AB_{k}_{i}_{r}")
        for r in range(R)] for i in range(I)] for k in range(K)]

W = [[[[model.new_int_var_from_domain(dom_term, f"W_{k}_{i}_{j}_{r}")
         for r in range(R)] for j in range(J)] for i in range(I)] for k in range(K)]

for k in range(K):
    for i in range(I):
        for r in range(R):
            model.add_allowed_assignments([A[k][r], B[i][r], AB[k][i][r]], pair_tuples)

for k in range(K):
    for i in range(I):
        for j in range(J):
            for r in range(R):
                model.add_allowed_assignments([AB[k][i][r], C[j][r], W[k][i][j][r]], term_tuples)

# Exact binary32-safe accumulation guard on the rank summation
PREFIX = [[[[None for _ in range(R)] for _ in range(J)] for _ in range(I)] for _ in range(K)]

for k in range(K):
    for i in range(I):
        for j in range(J):
            p0 = model.new_int_var(-B32_INT_EXACT, B32_INT_EXACT, f"P_{k}_{i}_{j}_0")
            model.add(p0 == W[k][i][j][0])
            PREFIX[k][i][j][0] = p0
            for r in range(1, R):
                p = model.new_int_var(-B32_INT_EXACT, B32_INT_EXACT, f"P_{k}_{i}_{j}_{r}")
                model.add(p == PREFIX[k][i][j][r - 1] + W[k][i][j][r])
                PREFIX[k][i][j][r] = p
            model.add(PREFIX[k][i][j][R - 1] == T[k][i][j])

# =========================
# Solve
# =========================

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = TIME_LIMIT_S
solver.parameters.num_workers = NUM_WORKERS
solver.parameters.linearization_level = 2
solver.parameters.log_search_progress = LOG_PROGRESS

status = solver.solve(model)
print("status:", solver.status_name(status))

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    A_sol = [[solver.value(A[k][r]) for r in range(R)] for k in range(K)]
    B_sol = [[solver.value(B[i][r]) for r in range(R)] for i in range(I)]
    C_sol = [[solver.value(C[j][r]) for r in range(R)] for j in range(J)]

    # Exact integer re-check (do not skip this).
    T_rec = [[[sum(A_sol[k][r] * B_sol[i][r] * C_sol[j][r] for r in range(R))
               for j in range(J)] for i in range(I)] for k in range(K)]
    assert T_rec == T, "Internal error: solver witness failed exact integer re-check."

    print(f"FOUND exact rank-{R} witness over alphabet {S}")
    print("A =")
    for row in A_sol:
        print(row)
    print("B =")
    for row in B_sol:
        print(row)
    print("C =")
    for row in C_sol:
        print(row)

elif status == cp_model.INFEASIBLE:
    print(f"PROVED infeasible for rank {R} over alphabet {S}")
else:
    print(f"No certificate within {TIME_LIMIT_S}s for rank {R} over alphabet {S}")
```

**How to run it.** First, verify the encoding by solving the known `R=19` case. Then run descending `R=18,17,...,10` over `S={0,±1}`; only after that widen to `S={0,±1,±2}`. Set `num_workers` explicitly, because CP-SAT’s own parameter docs recommend doing so. [PRIMARY: OR-Tools `sat_parameters.proto`, `cp_model` API] citeturn31view0turn31view3turn34view0

### Dictionary-first and Boolean variants

The structured-dictionary variant should come **before** free dense columns, exactly as L1 recommended. The easiest exact encoding in CP-SAT is one-hot atom selection. If `DICT_A`, `DICT_B`, and `DICT_C` are lists of candidate columns — basis vectors, all-ones, class indicators, cyclic shifts, diagonal-like patterns, and any bank-specific selectors you know are plausible — replace the free factor creation with this pattern:

```python
def factor_from_dictionary(model, D, name, R):
    # D = list of candidate columns, each a list of length n_rows
    n_rows = len(D[0])
    M = len(D)
    X = [[None for _ in range(R)] for _ in range(n_rows)]
    pick = [[model.new_bool_var(f"{name}_pick_{r}_{m}") for m in range(M)] for r in range(R)]
    for r in range(R):
        model.add_exactly_one(pick[r])
        for u in range(n_rows):
            vals = sorted({D[m][u] for m in range(M)})
            X[u][r] = model.new_int_var_from_domain(cp_model.Domain.from_values(vals), f"{name}_{u}_{r}")
            model.add(X[u][r] == sum(D[m][u] * pick[r][m] for m in range(M)))
    return X, pick
```

The practical sequence is: **dictionary-only**, then **dictionary plus a small free tail**, then **fully free dense**. That ordering is a direct translation of the L1 strategy and is where you are most likely to win time on selector banks with real structure. [PROMPT] [DERIVED]

For the **Boolean OR/AND** semiring, use the same CP-SAT scaffolding but drop signed alphabets entirely. Let `A,B,C ∈ {0,1}` and create Boolean `W[k,i,j,r]` for the rank-1 cubes. The exact OR/AND encoding is:

\[
W_{kijr} \le A_{kr},\quad
W_{kijr} \le B_{ir},\quad
W_{kijr} \le C_{jr},\quad
W_{kijr} \ge A_{kr}+B_{ir}+C_{jr}-2
\]

and then

- if `T[k,i,j] = 1`, require \(\sum_r W_{kijr} \ge 1\);
- if `T[k,i,j] = 0`, require \(\sum_r W_{kijr} = 0\).

That is exact Boolean tensor factorization in CP-SAT. The same support-ordering symmetry break applies. [DERIVED]

## Lower bounds and certificates

The cheap kill-test remains the same one from L1: compute the rank of the three mode flattenings, and take their maximum. Each flattening rank is an unconditional lower bound on the real CP rank, so any target `R` below that number is dead without touching the solver. For integer tensors, computing those matrix ranks over `QQ` is exact and matches the real rank because the entries are integers and the determinants/minors live in characteristic zero. [PROMPT] [DERIVED] [PRIMARY: SymPy matrix rank / exact-domain matrix docs] citeturn29view0turn16view1turn16view2

If you want a tiny standalone lower-bound snippet separate from the main script, this is enough:

```python
from sympy import Matrix

def flatten_mode(T, mode):
    K, I, J = len(T), len(T[0]), len(T[0][0])
    if mode == 0:
        return [[T[k][i][j] for i in range(I) for j in range(J)] for k in range(K)]
    if mode == 1:
        return [[T[k][i][j] for k in range(K) for j in range(J)] for i in range(I)]
    if mode == 2:
        return [[T[k][i][j] for k in range(K) for i in range(I)] for j in range(J)]
    raise ValueError

ranks = [Matrix(flatten_mode(T, m)).rank() for m in range(3)]
lb = max(ranks)
print("flattening ranks:", ranks, "lower bound:", lb)
```

For the Boolean problem, be more careful: this flattening-rank test is the right unconditional kill-test for the real/signed CP search, but Boolean OR-rank needs its own structural lower bounds. Your prompt already gave the main one you care about: **permutation-support selector banks have Boolean rank \(n\)**, so below that threshold they are OR-incompressible. Use that theorem directly when it applies. [PROMPT]

The honest interpretation of solver outcomes should be:

- **Found a shorter decomposition.** You have an exact bounded-alphabet witness. For CP-SAT and Z3 that witness is exact by model semantics; for Gurobi it becomes exact only after an explicit integer re-check in Python or another exact arithmetic environment. [PRIMARY: Gurobi numeric tolerances; OR-Tools integer-only semantics; Z3 arithmetic semantics] citeturn6view0turn2view3turn28view0turn11view0
- **Proved infeasible at fixed `R` and fixed alphabet `S`.** You have ruled out that bounded-alphabet model, and nothing stronger. This is an honest bounded certificate. If the model was dictionary-restricted, it rules out only the dictionary-restricted family. [DERIVED]
- **Did not solve within time.** You learned essentially nothing dispositive about that `R` except that your current encoding + symmetry-breaking + time budget did not close it. [DERIVED]

What would a **real-rank** certificate require? In principle, you would need an unrestricted-real nonexistence proof: something like a secant-variety membership obstruction, exact real quantifier elimination, or a symbolic certificate for a fully continuous polynomial system. L1 already settled the meta-point: unrestricted-real rank is \(\exists\mathbb{R}\)-hard, so that path is usually the wrong one operationally at your problem size. In practice, the only algebraic route I would seriously use is **fixed-support elimination**: once a support/sign pattern is frozen by combinatorial search, pass the remaining polynomial equations to Sage+Singular or Macaulay2. [PROMPT] [PRIMARY: Z3 non-linear arithmetic warning; Sage/Singular/Macaulay2 exact algebra docs] citeturn11view0turn24search1turn24search10turn19view1turn19view3

A minimal Sage skeleton for that fixed-support phase looks like this:

```python
# Skeleton only: use after supports/sign-patterns are fixed
# so that only the active coefficients remain symbolic.

from sageall import *

# Example: active coefficient symbols after support freezing
R = PolynomialRing(QQ, ['a0','a1','b0','b1','c0','c1'], order='lex')
a0,a1,b0,b1,c0,c1 = R.gens()

eqs = [
    # fill with exact coefficient-matching equations
    # e.g. a0*b0*c0 - 1,
    #      a1*b0*c1 + a0*b1*c1,
]

I = ideal(eqs)
gb = I.groebner_basis()
print(gb)
# If gb == [1], this fixed support pattern is impossible over QQ, hence over R.
```

The Macaulay2 analogue is the same idea:

```macaulay2
R = QQ[a0,a1,b0,b1,c0,c1, MonomialOrder => Lex];
I = ideal(
  -- fill with exact coefficient-matching equations
);
gb I
```

These symbolic steps are excellent for **one frozen support family** and usually a bad use of time for the full free search. [PRIMARY: Sage interface/Singular docs; Macaulay2 docs] citeturn24search1turn24search10turn19view3turn19view0

## Time budget and stopping rule

A useful mental model for the free dense `10×10×10` case is this: with `R=18`, the CP-SAT model above has about `30R = 540` factor variables, `100R = 1,800` pair-product auxiliaries, `1000R = 18,000` triple-product variables, and another roughly `18,000` prefix-sum variables, plus support/sign channeling bits. That is a **nontrivial but absolutely reasonable** CP-SAT instance class. It is still exponentially hard in the worst case, but it is the right order of magnitude for a modern multithreaded SAT/CP engine. [DERIVED]

My practical wall-clock guidance for a current desktop with 8–16 cores is:

- **Flattening ranks first:** effectively instantaneous to a few seconds at these sizes. [DERIVED]
- **Dictionary-only, `S={0,±1}`:** seconds to a few minutes for `R=18…16`, sometimes much longer if the dictionary is weak. [SPECULATIVE]
- **Free dense, `S={0,±1}`:** minutes to tens of minutes around `R=18…16`; often the best band to close first. [SPECULATIVE]
- **Free dense, `S={0,±1}` below that:** `R=15…13` is often “overnight if the bank is structured, otherwise sticky.” [SPECULATIVE]
- **Free dense, `S={0,±1,±2}`:** usually materially slower than the ternary case because the value domains widen and propagation gets weaker; think a multiplicative slowdown, not a constant-factor one. [SPECULATIVE]
- **`R=12…10`:** low confidence that you will routinely close these on a generic `10×10×10` selector bank without either a very informative dictionary or additional structure-specific valid inequalities. [SPECULATIVE]

That runtime guidance is not a published benchmark; it is an engineering estimate derived from the documented solver architecture and the size of the resulting model. [DERIVED]

The stopping rule I would actually use is simple:

- Start with `R=19` once, just to validate the encoding against your known construction.
- Then run descending `R=18,17,16,...`.
- At each `R`, do **three passes in order**: dictionary-only `S={0,±1}`, then free dense `S={0,±1}`, then free dense `S={0,±1,±2}`.
- Give each pass a hard wall-clock budget, for example 10 min / 45 min / 90 min.
- Treat `INFEASIBLE` as a real bounded-alphabet certificate; treat `UNKNOWN` only as “not closed yet.”
- If nothing below 19 appears after a full evening on `18…16`, do **not** spend your next hours grinding `12…10` with the same weak dictionary. Improve the dictionary first. [DERIVED] [PRIMARY: OR-Tools time and worker parameters] citeturn31view0turn31view3

If you also have Gurobi available, the best use of it is **parallel scouting**: let Gurobi search a one-hot linearized MILP for fast incumbents while CP-SAT remains the exact bounded-model certifier. That hybrid usually makes sense, but because you asked for a single public solver recipe you can honestly rely on, CP-SAT remains the recommendation. [DERIVED] [PRIMARY: Gurobi bilinear/nonconvex support and tolerance caveat] citeturn8view0turn6view0

## Open questions and limitations

No public source I found benchmarks **this exact selector-bank bounded-rank feasibility search** head-to-head across CP-SAT, Gurobi, Z3, and Gröbner tools. The “winner” call here is therefore a reasoned tooling verdict from solver semantics and model structure, not a quoted benchmark table. [DERIVED]

CP-SAT gives the right **exact integer semantics** for your bounded model, but as of the current OR-Tools parameter docs, its DRAT/LRAT proof machinery is only for restricted pure-SAT settings, not for the arithmetic CP model you actually need here. So a CP-SAT `INFEASIBLE` result is an honest solver result for the exact model, but not a portable independently-checkable proof artifact in the SAT-proof sense. [PRIMARY: OR-Tools `sat_parameters.proto`] citeturn31view0turn31view1turn31view2

Finally, the script above cleanly covers the **bounded-integer** part of your L1 regime. A true **dyadic** extension is easy algebraically — multiply all factors by a common power of two and solve an equivalent integer-scaled system — but the F32 safety guard then becomes a **significand-width** condition, not just an integer-magnitude bound. For M2’s practical benchmark question, I would keep the mainline recipe on integer alphabets first and only introduce dyadics later with a fixed common denominator. [DERIVED]

## Verification checklist and blunt verdict

- [x] At least 4 public solvers compared in the table with an exactness column.
- [x] One fully runnable recommended recipe (copy-pasteable) for the winner.
- [x] Flattening-rank kill-test included as step 0.
- [x] Honest infeasibility-interpretation paragraph (alphabet vs real).
- [x] **One-paragraph blunt verdict**: name the single solver+recipe I should run on a 10×10×10 bank, and the realistic chance it beats a known T=19 construction down toward rank-10.

**Blunt verdict.** Run **OR-Tools CP-SAT** first, using the exact bounded-integer recipe above: small alphabet `S={0,±1}` first, table-encoded products, first-nonzero-positive normalization on `A` and `B`, support-code ordering of rank-1 terms, and explicit `|prefix| ≤ 2^24` guards on the rank accumulation. Under your rules, that is the one solver/recipe pair that is both practically credible and semantically honest. My realistic expectation is: a good chance to close or improve the known `R=19` construction down into the `18…16` band if the bank has selector structure, a materially lower but still real chance in the `15…13` band if the dictionary is informative, and a **low** chance of genuinely closing `10…12` on a generic `10×10×10` bank without stronger structure discovery. If you want extra speed, Gurobi is worth running as a scout — but not as the proof engine for the claims you care about. [DERIVED] [SPECULATIVE] [PRIMARY: Gurobi tolerance caveat; OR-Tools exact integer semantics] citeturn6view0turn8view0turn2view3turn28view0