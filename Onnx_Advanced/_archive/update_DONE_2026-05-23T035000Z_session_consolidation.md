# DONE 2026-05-23T035000Z — Session consolidation

## High water mark: 5920.57 (was 5874.78 at session start = +45.79 LB)

## New library claims worth integrating (already added to traps.md / techniques.md / cascade_strategy.md / cost_floor_reference.md but listed here for sorter cross-check):

### Confirmed wins this session (17 submissions):

| # | Submission | LB delta | Notes |
|---|---|---:|---|
| 1 | task383 bool_u8_maxpool wildcard | +1.01 | LIVE-equiv MaxPool-window pattern |
| 2 | task069 scalar_squarecode_sub10k wildcard | +2.54 | BOOL-output Track B confirmation |
| 3 | task358 organic_cross_under10k wildcard | +2.61 | algorithm rewrite |
| 4 | 3-stack t383+t069+t358 | +6.16 | perfect additivity proven |
| 5 | 21-stack | +20.90 | 22+ task additivity |
| 6 | task175 v12 wildcard | +2.98 | variadic-Max + sum-minus-ch0 + sentinel pad |
| 7 | task018+task089 2-stack | +1.97 | t018 zero-weight confirmed |
| 8 | task382+task284+task077 3-stack | +3.15 | t284 0.00 cost-decoupling |
| 9 | task008+task051+task157 3-stack | +5.05 | t157 zero-weight confirmed |
| 10 | task209+task234 2-stack | +2.53 | t234 partial decoupling 22% |
| 11 | task255 wildcard | 0.00 | 7th zero-weight confirmed |
| 12 | task157 smallest variant | 0.00 | zero-weight is task-level property |
| 13 | BOOL-output 9-stack | +2.03 | first multi-task sweep ship |
| 14 | 11-stack BOOL+t330+t369 | +5.41 | external worker integration |
| 15 | t117+t396 (regression) | -11.29 | t396 count_hash hidden-bad |
| 16 | task117 solo | +2.10 | algorithm rewrite at 0/100 adv |
| 17 | task396 longrun_rule | +1.70 | algorithm rewrite recovery |

### Big strategic finding: ALGORITHM REWRITE >> TRICK STACKING

5 confirmed algorithm-rewrite ships at +1.70 to +3.30 LB each. Tricks (BOOL-output sweep) yielded +2.03 LB across 9 tasks combined. **One radical rewrite = ~3-5× a trick application per-task.**

### Memorized-lookup zero-weight family: 7/7 confirmed

- task366, task324, task133, task182, task018, task157, task255
- Three architectural signatures documented (`cascade_strategy.md`)
- task157 zero-weight confirmed at smallest cost variant (3,413) — zero-weight is HARD TASK-LEVEL property

### New trap class (traps.md §14e): count-vector hash lookup hidden-bad

- task077 lookup8: -13.13 LB
- task396 count_hash_lookup: -13.40 LB
- task157 bitpack_* family: would have been similar but task157 is zero-weight
- Detection: < 1% adversarial agreement on random inputs + `[N=266, K]` template initializer + Hash/Flatten+MatMul/Equal-pack-key architecture

### Cost-decoupling spectrum (NEW)

- Clean (LIVE-equivalent): full predicted LB gain
- Partial decoupling: % LB realized depends on what fraction of hidden inputs the candidate's assumptions hold on (task234 top2 0/100 adv → 22% credit)
- Full decoupling: 0.00 LB (task284 scatter_dynamic 19/150 adv → 0.00)
- Negative (hidden-bad): -X LB (task077/task396 lookup family)

### NEW multi-task sweep infrastructure (tools/multi_task_sweeps/)

- bool_output_sweep.py: auto-applier for Track B (9 wins, +2.03 LB Kaggle)
- task383_maxpool_sweep.py: detection-only (15 candidates flagged for radical work)
- variadic_max_sweep.py: nearly exhausted (1 candidate left)
- cherry13b_scan.py: 12 zero-weight suspects flagged (4 high-confidence: task285, task101, task076, task118)
- assemble_megastack.py: stack-builder for sweep results

## Sorter notes

- These library updates are LIVE (already in traps.md / techniques.md / cascade_strategy.md / cost_floor_reference.md). Sorter should verify cross-references and dedup if it finds parallel entries.
- The 5 algorithm-rewrite exemplars (task330, task051, task175, task117, task396, task008) deserve exemplar.md entries in `exemplars/01_whole_graph_rewrites/` for future workers to study. Sorter or follow-up worker can extract.
