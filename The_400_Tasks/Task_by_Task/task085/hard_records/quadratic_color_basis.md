# quadratic_color_basis route

- Live/team-best bar used: cost 136, sha256 c2d0c80df4932d3b3c782185af15b9d2749d2f6caee9c1fbfe9a5e74895873b7, source workbook + team manifest refreshed 2026-07-03.
- Attempted route: keep the current one-Einsum structure, `parity_basis`, and `row_weight`, but replace the cubic `[1,k,k^2,k^3]` shared `color_basis` with a lower-degree shared basis to reduce params below 136.
- Exact failed gate: linear feasibility over train+test+262 bundled `arc-gen` active-cell/channel constraints. Unique signed constraints: n=1 had 1034 constraints/2 vars infeasible; n=2 had 8530 constraints/8 vars infeasible; n=3 had 8530 constraints/18 vars infeasible. n=4 was feasible, matching the current team-best basis size.
- Why dead: under the current parity/row features, a quadratic color basis cannot separate keep-color, delete-to-background, and active-background logits across all colors and output channels. Building an ONNX from it would be a visible/truth miss.
- One remaining hypothesis: a different spatial feature basis could change the color separability, but generating that basis at runtime costs memory and a denser spatial initializer likely exceeds the 136-param free-output bar.
