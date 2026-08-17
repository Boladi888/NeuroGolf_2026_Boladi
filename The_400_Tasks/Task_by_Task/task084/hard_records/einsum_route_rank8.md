# task084 hard record: einsum_route_rank8

Live bar: 594 bytes, sha256 `b25020cf1be3f0d28187254d13e99642312461153567f22039e53714d7fa09ac`, source `python tools/ng_current_bar.py --task 84 --json` / manifest, owner Vamshi.

Attempted route: reduce the final free-output Einsum route axis from T=12 toward T=8 by rank-cutting the `[12,10]` `route_out` and `route_in` pair plus `route_kind`.

Failed gate: fast rank probe before build. Results on `Team_Best_Onnx/task084.onnx`: `rank(route_out)=10`, `rank(route_in)=10`, `rank(concat(route_out, route_in, route_kind))=12`; channel-kind tensor `einsum('jo,ji,jh->oih')` unfold ranks are output=10, input=10, kind=4.

Why dead: T=8 is below the unconditional output/input unfolding lower bound of 10, and the simple route-row basis has full row rank 12. The local ARC-GEN JSON uses all non-background colors 1..9 across sizes 3..21, so a six-color-only route would be a hidden-bad/color-pruned formulation.

One remaining hypothesis: a different channel-routing formulation might beat 504 by replacing the CP route with a structured identity-plus-three-specials factor, but it is not the documented T=8 row-rank cut and needs a fresh algebraic proof before build.
