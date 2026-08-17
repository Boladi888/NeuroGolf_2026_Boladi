# Research Card O4_REPOACCESS - ConstantOfShape scorer-confirmation probe package

**Deliverable**: repo-access research note plus a captain-ready A/B probe package if feasible. Save outputs under `Logs/research` with basename `O4_constantofshape_scorer_confirmation`.

## Worker contract

You have full repo and tool access. Follow `Onnx_Advanced/WORKER_PROTOCOL.md`. Do not submit to Kaggle. Do not edit root knowledge docs directly. Build only tiny, isolated probe assets and a report.

## Why this card exists

R6 locally measured final-output `ConstantOfShape` accounting as shape-init params only, with the scalar `value` attribute apparently uncharged. The library correctly keeps a Kaggle-confirmation caveat before treating this as doctrine.

This card prepares a minimal local and captain-ready confirmation package.

## Required work

1. Identify the smallest task/probe context where a uniform final-output fill is valid and can be A/B compared:
   - dense `Constant` final output
   - `ConstantOfShape` final output with equivalent values
   - optional variants using default zero and explicit scalar value attribute
2. Build tiny ONNX models or task-specific candidates that differ only in dense constant vs `ConstantOfShape` construction.
3. Run local checker, shape inference, ORT CPU load/run, and local scorer.
4. Package the exact files and instructions a captain would need for external Kaggle confirmation, but do not submit.
5. Record risks: validator shape handling, attr accounting, graph output dtype, task validity, and cross-task cascade concerns.

## Required report sections

1. **Blunt verdict**: is the A/B package ready for captain submission?
2. **Model inventory**:

`| file | construction | output dtype/shape | local params | local memory | checker | ORT run | task validity |`

3. **Delta explanation**: why the two graphs are semantically identical and what cost difference is being tested.
4. **Captain instructions**: exact next step if external confirmation is desired.
5. **Do-not-infer list**: what this probe would and would not prove.

## Guardrails

- Do not touch submission pipelines unless you only create isolated artifacts.
- Do not claim Kaggle confirmation; this card can only prepare and locally verify.
- If no valid public task admits a uniform final output, write a negative report and propose a safer probe shape.

## Final checklist

- [ ] Local A/B models built or infeasibility explained.
- [ ] Local scorer outputs recorded.
- [ ] Captain-ready instructions written.
- [ ] No external submission made.
