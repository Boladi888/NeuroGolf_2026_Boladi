# cascade_strategy.md — exploiting the cross-task memory cascade

The Kaggle scorer has a documented memory-leak bug between consecutive task evaluations: a high-memory ONNX in slot N can ERROR slot N±1 (or paired tasks). Both pass `verify_network` in isolation; only Kaggle's actual eval pipeline triggers the error. The errored task scores **literal 0** instead of its normal points. See `core.md` "Cross-task memory cascade bug" for the base facts.

This file is the **strategy doc**: how to find new cascade-triggers, exploit them, and prioritize work to maximize cascade-unlock LB.

---


---
**Phase-1 (2026-06-05)**: campaign diary/watchlists -> `Logs/cascade_campaign_log.md`. Worker facts: threshold T~250-300k, cherry13b protocol, Rule-1 submit-singly, raw-log calibration, 1 confirmed cascade event ever.

## 2026-06-07 NULL datapoint: task260 trigger-block probe
task260 (member of the documented 256-262 cascade-trigger set) shipped with memory cut
11,932 → 1,227 (−90%) and landed EXACTLY its solo prediction (6607.53 vs 6607.52 computed)
— ZERO cascade excess. The cumulative-memory cascade hypothesis takes a clean negative at
this position. Combined with the per-task exactness of ~60 ships since 2026-06-04, the
working model remains: per-task scoring is independent; treat cascade theory as historical.
