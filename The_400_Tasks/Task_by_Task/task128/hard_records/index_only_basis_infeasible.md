# task128 index_only_basis_infeasible

- Live bar: cost 104, profile cost 104, SHA256 `ccdb0c525b1f0ba110466e0c36a9f4eedb963c098794a264c725a71595205798`, source `python tools/ng_current_bar.py --task 128 --json` manifest authority. Local 300-byte task notes and library_db/BoBrain crop route are stale for the current bar.
- Locked reviewer hypothesis: single-node #89 free-output `Einsum`, replacing live `basis[2,30] = [1,row]` with an index-only coordinate vector `[row]`, while taking count terms directly from raw `input` summations. No multi-op core.
- Screen command: `python The_400_Tasks/Task_by_Task/task128/prototypes/screen_index_only_basis.py`
- Screen result: `verdict=INFEASIBLE_FOR_LOCKED_HYPOTHESIS`
- Why dead: after row summation, each live factor has form `F(r)=a00*N + a01*rN + a10*S + a11*rS`, where `N=sum(input)` and `S=sum(row_index*input)`. Raw input summation can supply `N`; an index vector can supply `rN`/`rS`. But the output label `r` must be present in any single-node `Einsum` term that emits `[b,c,r,w]`; broadcasting `N` or `S` across output rows requires an all-ones row carrier, exactly the removed `basis[0,r]` or an equivalent dense selector. The live margins contain r-independent terms in all six channel/factor cases:
  - `coef1 nonbg: +43.5*N -4*S`
  - `coef1 bg: -14.5*N`
  - `coef2 nonbg: -14.5*N +2*S`
  - `coef2 bg: +13.5*N -4*S`
  - `coef3 nonbg: +1*N`
  - `coef3 bg: +0.5*N +2*S`
- Stop condition: no builder emitted. Any one-node builder that stays below 104 and uses only `[row]` must drop the listed r-independent terms; any builder that restores them reintroduces an all-ones row carrier or equivalent selector and loses the intended basis collapse.
- Relevant library citations used: `Onnx_Advanced/INDEX.md` routes to `techniques.md #67/#89`; #67 says make the expensive op the free output; #89 says final `Einsum` in-kernel temporaries are uncharged but operands are still charged; `traps.md §FAMILY-LABEL` says generator/rule beats stale labels/routes; `traps.md §FREE-OUTPUT` requires decode-positive boundary safety for output-writing formulations.
- One remaining hypothesis: a different non-sparse algebraic factorization with fewer than 104 dense operand elements, not an index-only replacement of the live `[1,row]` basis.
