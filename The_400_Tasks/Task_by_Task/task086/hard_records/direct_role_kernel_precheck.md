# task086 direct-role-kernel precheck

- Live bar used: manifest live cost 1759, sha256 `bb030e0b8ad5d60d2d4d91eaf882bdf470c9c0663ade6e83b72b92a6d94bc3d0`; stricter task-local staged bar 1750, sha256 `a7f360732697b59f338487593e5b515eece550dcb35ca024a7e5158fdda8cfbd`.
- Attempted route: replace `background_u8 -> occ_u8 -> outer_score_u8 -> Concat(role_code)` with one 3-channel 11x11 QLinearConv over `background_u8`.
- Failed gate/pre-gate: no ONNX visible gate fired. The support-channel precheck over the full cached 200000 ARC-GEN corpus found 93856 unique 11x11 background patches with 0 label conflicts, but high-C LinearSVC still missed 3203 unique patches, so the single linear support channel is not a credible visible-pass candidate.
- Why dead for this batch: the route would add about 238 params and only wins if the outer/support role is exactly one int8 linear threshold over the 11x11 background patch; the float linear probe already fails.
- Remaining hypothesis: a two-feature nonlinear direct detector, or a CP-SAT proof with engineered features, could still be explored, but it must beat the new 1726 bar and should visible-gate the support mask before color rendering.
