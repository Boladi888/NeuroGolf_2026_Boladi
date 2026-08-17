# task071 dead route: F-basis row drop / dense factor probes

- Live bar used: authoritative live cost 843, SHA `82a0d5108c509ef5bf46191bcb953420f9fd32560cb87f8c5f04e17446f83da5`, Vamshi team-best from manifest via `python tools/ng_current_bar.py --task 71 --json`; task-local staged bar was 799 from `task071_activegridcrop_direct_zero_trig_cost799/manifest.json`.
- Attempted route: reduce or factor the dominant `F [7,2,30]` final Fourier basis in the DFT/Einsum graph.
- Exact failed gate/probe: algebraic visible-corpus simulation over 265 officials. Plain row drops failed immediately: best single-row drop was `drop k=5 -> 21/265`, with first miss at pair 0. Least-squares `R` refits for row-dropped bases could not reproduce the incumbent kernel on `axis_local` 7..18 and still failed officials.
- Why dead: the basis is the real Fourier reconstruction of `(c + d - 2) mod 13 == axis_local`; deleting rows leaves side lobes that cross the decoder threshold. Exact dense matrix factorization is also not a win: `F.reshape(14,30)` has rank 14, so a direct `H*C` factor would cost at least `14*(14+30)=616` params versus dense `F` at 420.
- One remaining hypothesis: find a non-materialized cyclic anti-diagonal factorization inside one final free-output `Einsum`, or continue small descriptor-basis reuse around the graph; do not retry plain row drop or dense matrix factorization.
