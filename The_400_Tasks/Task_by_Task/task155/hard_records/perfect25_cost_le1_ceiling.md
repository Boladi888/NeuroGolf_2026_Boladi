# DEAD ROUTE — task155 perfect-25 / cost<=1 vflip ceiling (2026-07-07, Claude bo_ng_cld_worker1)

**Live bar at audit:** cost **158** (manifest, sha `81969433909ac8f8a591262db6256652ed49bebbc3380a0cfed52ec22751c1d1`,
owner Aditya). static=158, profile=158 (266 real inputs). Model = ReduceL2 -> Cast ->
Sub(N,range1[30]) -> Gather(axis=2); 120 B of that is the idx[30] INT32 Gather index.

**Audit question:** can vflip (`output[r,c]=input[N-1-r,c]`, top-anchored square N in {4..8},
30x30 canvas) be a cost<=1 no-payload direct-output graph (the "perfect-25" ceiling)?

**Verdict: NO — WALL. cost<=1 is impossible; 158 is the forced floor.**

## Families swept and exact failed gate

| # | route | cost | gate result | why dead |
|---|---|---:|---|---|
| P0 | `Identity(input)` — the literal cost<=1 no-payload graph | **0** | `visible 0/266 FAIL` | reverses nothing |
| P1 | `Slice(steps=-1, axis=2)` full-canvas row reverse (data-independent) | **4** | `visible 0/266 FAIL` | reverses ALL 30 rows -> top-anchored data lands at the bottom; a fixed permutation cannot satisfy `pi(0)=N-1` for N=4 AND N=8 |
| P2 | `ReverseSequence` — only ONNX op that reverses `input[0:N]` with a *scalar* length (would drop the 120 B idx[30]) | **None** | `score_v2: "Op type ReverseSequence is not permitted"` | official scorer allowlist rejects the op -> unscoreable/ineligible. Independently, ORT restricts `time_axis`/`batch_axis` to {0,1}; the row axis is 2, so reaching it needs a Reshape/Transpose that materializes a `[10,30,30]` F32 = **36,000 B** intermediate (228x worse than 158) |

Per §VISIBLE-FAIL a visible 0 is a certain hidden 0, so no truth/ab/battery was run on P0/P1
(pure burn). P2 is unscoreable so it never reaches a gate.

## Forced-component argument (why cost<=1 can never hold)

The output map depends on input CONTENT: the reversal anchor `P[0]=N-1` varies with the
active-row count N in {4,5,6,7,8}. A cost<=1 no-payload graph applies ONE fixed
position-permutation independent of input values (Identity/Transpose/Reshape/fixed-Slice).
No fixed permutation pi satisfies `out[r]=in[pi(r)]` for all N at once (pi(0) is not constant)
— proven empirically by P0/P1 = 0/266. Therefore any correct graph MUST read N from the input
(a reduction, ~4-8 B) AND value-dependently reindex 30 output rows. That reindex forces payload:
- **Gather**: needs a full `[30]` index, data-dependent (static idx = 0/266 wrong — floor proof),
  min legal dtype INT32 (`Tind in {int32,int64}`; scorer = None for INT8/16/UINT8) = 120 B. This IS E@158.
- **ReverseSequence** (scalar length): scorer-rejected + ORT axis {0,1} only (see P2).
- **GatherElements/ScatterND**: full-rank `[1,10,30,30]` index >=9000 elems (floor proof: 9000).
- **Einsum**: reversal permutation matrix `[30,30]`, data-dependent on N — worse.

Minimum credited cost for a correct vflip = idx[30] INT32 (120 B) + range1 (30 par) + N-scalars
(8 B) = **158**. This is the E@158 construction. **cost<=1 perfect-25 is unreachable.**

## Remaining hypothesis
None that beats 158. The 120 B Gather-index term is forced (30 rows x INT32); the only
scalar-length reversal op (ReverseSequence) is both scorer-illegal and layout-blocked. Do not
re-open the perfect-25 ceiling for task155.
