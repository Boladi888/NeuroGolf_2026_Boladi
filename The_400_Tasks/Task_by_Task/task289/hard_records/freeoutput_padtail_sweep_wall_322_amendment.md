# task289 amendment — 324 wall narrowed to the structural free-output routes

**Bar rechecked 2026-07-11:** official static/profile cost **324**, owner Sho, SHA-256
`ffbb8ecbbd332816e69ab20d1539bda1c778bc7372b01a37ba061d95766de62e`
(`python tools/ng_current_bar.py --task 289 --json`; authoritative team manifest).

The earlier `freeoutput_padtail_sweep_wall.md` remains correct about the prompt's ~129
representation families, but its broader statement that no sub-324 graph exists is superseded.
A parent-preserving tail micro-fold reaches official static/profile **322**:

```text
old: roi_end = Div(9.8, present_sum)
     roi = Concat(0, 0, roi_end, roi_end)
new: roi = Div([0, 0, 9.8, 9.8], present_sum)
```

Candidate SHA-256:
`ebefa45e3ef20fcae3a8e004fc67445be413917101cfa230b6a66f6eaef8a65b`.
Named charge delta: delete `roi_end` F32 `[1,1,1,1]` (**-4 B memory**), replace two
one-element initializers with one four-element vector (**+2 official params**), net **-2**.

Exact proof:

- visible: `268/268 (100.00%) PASS`
- profile: `static=322 profile=322 (memory 263 + params 59; 268 real inputs)`
- truth full: `200000/200000 (100.000%)`
- A/B full vs the 324 parent: `match 200268/200268 | divergences=0 | byte_equal=True`
- safety: `WARN`, 0 failures, one inherited `WARN_X86_QUANTIZED_OP_FAMILY`; therefore
  `PROBE_READY`, not `LOCAL_GOLD` (`x86_required=true`, `kaggle_probe_required=true`).

The prompt's structural routes are still dead and were not rebuilt:

- direct same-size 30→30 dynamic-ROI Resize: exact failed gate `visible 0/268`; ORT
  shape-short-circuits the Resize;
- 3×3 pre-Resize input: exact failed gate `visible 0/268`; it has no zero border;
- low-rank CP expansion bank: exact failed gate `visible 0/268`; positive off-band
  cancellation residue survives the decoder's `>0` threshold;
- stored exact free-output bank remains ~519 or more, above both 324 and 322.

**One remaining hypothesis:** delete or compress the `present_u8[1,10,1,1]` plus
presence-sum QLinearConv chain. U8/INT32 ReduceSum is rejected by ORT, so this needs a
different exact kernel; otherwise 322 is the current parent-preserving floor. A future ORT
same-size Resize that honors dynamic crop ROI would reopen the prompt's ~129 route.

Reviewer packet:
`staged_candidates/task289_roi_broadcast_fold_cost322/manifest.json`.
