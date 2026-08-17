# multistage_key_first_compositional.md

Multi-stage ARC-style rules are best written as explicit tensor compositions of the form `K = f(X)`, `S = g(X, K)`, `Y = h(X, K, S)`: a key is extracted first, later scoring or propagation depends on that key, and only then is the final grid assembled. ARC-AGI-2 explicitly frames this family of difficulty as multi-rule composition, multi-step composition, and contextual rule application, while Hodel’s ARC-DSL shows the same idea in symbolic-program form via compositions such as object extraction → filtering → painting.

**ARCHIVED (Phase-2)** — extractions live in techniques.md (Wins integrated); full text: `_archive/multistage_key_first_compositional.md`.
