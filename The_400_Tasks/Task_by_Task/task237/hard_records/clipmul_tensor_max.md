# task237 clipmul tensor-max dead route

- Live bar: team manifest/workbook cost 705, sha256 a59a2fa610fa58f67f8026812f7a160bf82f9fb1b65955c50fb91dde71bf538e, updated 2026-07-02T20:17:39Z.
- Attempted route: `prototypes/build_task237_clipmul.py` tried to delete `selected_t_pre` by replacing `Max(latest_grid, one_u8)` + `Min(packed_u8, selected_t_pre)` with `Clip(latest_grid, one_u8, packed_u8)`.
- Exact failed gate: static score/profile load via `score_network_v2` returned `None`; ORT error tail: `Clip::ComputeImpl<unsigned char>::operator () max->Shape().IsScalar() was false. max should be a scalar.`
- Why dead: ORT 1.24.4's UINT8 Clip kernel requires scalar min/max, so a tensor-valued `packed_u8` upper bound cannot encode the pointwise min.
- Remaining hypothesis: a non-Clip `Min`/`Where` algebraic identity may still remove one 9-byte tensor; `task237_rankone_cost696` uses one such identity by changing the false marker sentinel and deleting `selected_t_pre`.
