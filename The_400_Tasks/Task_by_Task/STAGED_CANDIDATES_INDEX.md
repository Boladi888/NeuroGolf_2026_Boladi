# Staged Candidates Archive Ledger

Cleanup refresh: 2026-07-15. The competition is over, and the task-local `staged_candidates/` directories described manifests for payloads that are not part of this repository. The durable summary now lives here; representative proof records were moved under `_archive/staged_candidate_examples/`.

## Cleanup Decision

- Pre-cleanup tracked staged-candidate files: **1,741**.
- Representative exemplar files moved to `_archive/staged_candidate_examples/`: **9 files in 8 candidate dirs**.
- Remaining tracked staged-candidate files slated for deletion: **1,732** (957 active-manifest files plus 775 archived-manifest files).
- Gitignored staged-candidate ONNX payloads present in this worktree: **0**.
- After cleanup, `Task_by_Task/taskNNN/staged_candidates/` is intentionally absent from the archive checkout.

## Preserved Exemplars

| task | preserved dir | why kept |
|---|---|---|
| `task021` | `_archive/staged_candidate_examples/task021/task021_no_candidate_floor_20260624_w9/` | No-candidate/floor packet with explicit NO_CANDIDATE note. |
| `task029` | `_archive/staged_candidate_examples/task029/task029_v18_direct_input_slice_profile6796_creditprobe/` | Profile-gap credit-probe example: static 3636 vs profile 6796. |
| `task029` | `_archive/staged_candidate_examples/task029/task029_v26_inner23_conv_u8slice_cost8319_profile8319/` | Fully profiled exact-style candidate manifest with static==profile. |
| `task048` | `_archive/staged_candidate_examples/task048/task048_no_candidate_hiddenfail_floor_20260624_w048/` | Hidden-fail/no-candidate floor example preserved as manifest.md. |
| `task176` | `_archive/staged_candidate_examples/task176/task176_shared_spatial_basis84_cost84/` | Tiny exact profiled manifest example. |
| `task185` | `_archive/staged_candidate_examples/task185/task185_u8_step_chain_cost1093/` | Late low-cost proof manifest, static/profile exact. |
| `task246` | `_archive/staged_candidate_examples/task246/task246_direct_scalar_extract_h1_cost388/` | Rescue-style direct scalar extraction manifest. |
| `task398` | `_archive/staged_candidate_examples/task398/task398_unscaled_shared_pow_cost2264/` | Late active proof manifest for shared-power basis route. |

## Removed Directory Counts

Counts below are the historical inventory after the exemplar move and before deleting the remaining `staged_candidates/` tree.

- Active candidate dirs removed: **664**
- Archived candidate dirs removed: **617**
- Tasks represented in removed staged-candidate dirs: **319**

| task | active dirs removed | archived dirs removed |
|---|---:|---:|
| `task001` | 0 | 1 |
| `task002` | 1 | 0 |
| `task004` | 2 | 0 |
| `task005` | 5 | 0 |
| `task006` | 0 | 2 |
| `task007` | 0 | 5 |
| `task008` | 3 | 0 |
| `task009` | 2 | 0 |
| `task010` | 4 | 2 |
| `task011` | 4 | 4 |
| `task012` | 2 | 2 |
| `task013` | 1 | 0 |
| `task014` | 2 | 3 |
| `task015` | 1 | 0 |
| `task017` | 4 | 1 |
| `task018` | 3 | 0 |
| `task019` | 1 | 5 |
| `task020` | 2 | 1 |
| `task021` | 1 | 0 |
| `task022` | 1 | 5 |
| `task025` | 1 | 0 |
| `task026` | 0 | 1 |
| `task027` | 2 | 1 |
| `task029` | 5 | 2 |
| `task030` | 1 | 8 |
| `task031` | 1 | 14 |
| `task032` | 3 | 2 |
| `task033` | 1 | 4 |
| `task034` | 5 | 1 |
| `task035` | 7 | 2 |
| `task036` | 1 | 1 |
| `task037` | 1 | 0 |
| `task038` | 0 | 2 |
| `task039` | 1 | 4 |
| `task040` | 1 | 1 |
| `task041` | 2 | 5 |
| `task042` | 4 | 1 |
| `task044` | 2 | 2 |
| `task045` | 1 | 0 |
| `task046` | 3 | 3 |
| `task047` | 2 | 2 |
| `task048` | 3 | 6 |
| `task049` | 1 | 0 |
| `task050` | 4 | 0 |
| `task051` | 2 | 0 |
| `task052` | 1 | 0 |
| `task054` | 8 | 0 |
| `task055` | 0 | 6 |
| `task056` | 0 | 2 |
| `task057` | 1 | 1 |
| `task059` | 1 | 2 |
| `task061` | 0 | 8 |
| `task063` | 1 | 0 |
| `task064` | 5 | 3 |
| `task065` | 2 | 1 |
| `task066` | 2 | 0 |
| `task067` | 0 | 1 |
| `task068` | 2 | 2 |
| `task069` | 1 | 0 |
| `task070` | 1 | 0 |
| `task071` | 3 | 0 |
| `task072` | 1 | 1 |
| `task074` | 2 | 6 |
| `task075` | 1 | 0 |
| `task076` | 6 | 0 |
| `task077` | 4 | 2 |
| `task078` | 0 | 3 |
| `task079` | 2 | 12 |
| `task080` | 3 | 0 |
| `task081` | 1 | 2 |
| `task083` | 2 | 2 |
| `task084` | 2 | 0 |
| `task086` | 3 | 7 |
| `task089` | 1 | 0 |
| `task090` | 2 | 0 |
| `task091` | 1 | 11 |
| `task092` | 3 | 2 |
| `task093` | 1 | 2 |
| `task094` | 3 | 2 |
| `task096` | 7 | 3 |
| `task100` | 1 | 6 |
| `task101` | 8 | 1 |
| `task102` | 2 | 5 |
| `task104` | 1 | 3 |
| `task105` | 4 | 0 |
| `task106` | 3 | 1 |
| `task107` | 2 | 0 |
| `task108` | 0 | 1 |
| `task109` | 5 | 0 |
| `task110` | 8 | 3 |
| `task111` | 7 | 1 |
| `task112` | 1 | 0 |
| `task114` | 2 | 1 |
| `task117` | 3 | 2 |
| `task118` | 1 | 0 |
| `task119` | 1 | 1 |
| `task121` | 2 | 0 |
| `task122` | 1 | 2 |
| `task123` | 2 | 5 |
| `task124` | 5 | 2 |
| `task125` | 0 | 6 |
| `task127` | 1 | 1 |
| `task128` | 1 | 0 |
| `task129` | 0 | 2 |
| `task130` | 0 | 3 |
| `task131` | 3 | 0 |
| `task132` | 3 | 5 |
| `task133` | 5 | 0 |
| `task134` | 2 | 0 |
| `task136` | 1 | 2 |
| `task137` | 2 | 0 |
| `task138` | 6 | 0 |
| `task139` | 2 | 0 |
| `task140` | 0 | 1 |
| `task141` | 0 | 10 |
| `task142` | 1 | 3 |
| `task143` | 1 | 2 |
| `task144` | 0 | 1 |
| `task145` | 1 | 0 |
| `task146` | 3 | 1 |
| `task147` | 0 | 1 |
| `task148` | 0 | 8 |
| `task151` | 0 | 2 |
| `task152` | 1 | 1 |
| `task153` | 5 | 0 |
| `task154` | 5 | 2 |
| `task156` | 3 | 0 |
| `task157` | 5 | 0 |
| `task158` | 10 | 3 |
| `task160` | 2 | 0 |
| `task161` | 2 | 0 |
| `task162` | 1 | 0 |
| `task163` | 1 | 3 |
| `task166` | 0 | 3 |
| `task167` | 1 | 0 |
| `task169` | 4 | 9 |
| `task170` | 6 | 2 |
| `task173` | 3 | 0 |
| `task174` | 7 | 3 |
| `task175` | 1 | 1 |
| `task176` | 3 | 1 |
| `task177` | 4 | 1 |
| `task178` | 4 | 11 |
| `task180` | 0 | 2 |
| `task181` | 2 | 2 |
| `task182` | 2 | 0 |
| `task183` | 0 | 7 |
| `task184` | 5 | 2 |
| `task185` | 0 | 2 |
| `task186` | 1 | 1 |
| `task187` | 2 | 0 |
| `task188` | 1 | 2 |
| `task190` | 2 | 1 |
| `task191` | 2 | 0 |
| `task192` | 4 | 5 |
| `task193` | 1 | 0 |
| `task194` | 2 | 4 |
| `task195` | 3 | 2 |
| `task196` | 2 | 0 |
| `task197` | 1 | 2 |
| `task198` | 1 | 0 |
| `task199` | 3 | 3 |
| `task200` | 1 | 0 |
| `task201` | 3 | 0 |
| `task203` | 1 | 0 |
| `task205` | 4 | 9 |
| `task206` | 3 | 0 |
| `task207` | 1 | 1 |
| `task208` | 1 | 1 |
| `task209` | 4 | 5 |
| `task211` | 1 | 0 |
| `task213` | 5 | 0 |
| `task214` | 2 | 0 |
| `task215` | 0 | 1 |
| `task216` | 3 | 0 |
| `task217` | 2 | 4 |
| `task218` | 1 | 0 |
| `task219` | 2 | 0 |
| `task220` | 2 | 0 |
| `task221` | 1 | 0 |
| `task222` | 4 | 3 |
| `task224` | 1 | 0 |
| `task225` | 2 | 10 |
| `task226` | 1 | 3 |
| `task228` | 2 | 15 |
| `task229` | 1 | 0 |
| `task230` | 1 | 1 |
| `task233` | 8 | 7 |
| `task234` | 4 | 3 |
| `task235` | 1 | 1 |
| `task236` | 2 | 1 |
| `task237` | 3 | 0 |
| `task238` | 3 | 0 |
| `task239` | 2 | 0 |
| `task240` | 0 | 1 |
| `task242` | 1 | 2 |
| `task243` | 3 | 0 |
| `task245` | 1 | 0 |
| `task246` | 1 | 7 |
| `task247` | 1 | 2 |
| `task249` | 1 | 0 |
| `task250` | 4 | 3 |
| `task251` | 2 | 0 |
| `task253` | 2 | 4 |
| `task255` | 2 | 0 |
| `task256` | 1 | 10 |
| `task257` | 0 | 1 |
| `task258` | 0 | 1 |
| `task259` | 3 | 0 |
| `task260` | 2 | 6 |
| `task263` | 1 | 0 |
| `task264` | 3 | 11 |
| `task265` | 5 | 3 |
| `task266` | 1 | 3 |
| `task267` | 4 | 3 |
| `task268` | 0 | 5 |
| `task269` | 1 | 0 |
| `task270` | 3 | 2 |
| `task271` | 2 | 1 |
| `task272` | 0 | 1 |
| `task273` | 3 | 0 |
| `task274` | 1 | 0 |
| `task275` | 2 | 0 |
| `task277` | 4 | 3 |
| `task278` | 3 | 6 |
| `task279` | 2 | 0 |
| `task280` | 1 | 0 |
| `task281` | 0 | 2 |
| `task283` | 3 | 1 |
| `task284` | 2 | 1 |
| `task285` | 2 | 0 |
| `task286` | 8 | 0 |
| `task288` | 0 | 4 |
| `task289` | 3 | 2 |
| `task290` | 2 | 0 |
| `task291` | 1 | 4 |
| `task292` | 0 | 1 |
| `task293` | 1 | 1 |
| `task294` | 3 | 0 |
| `task295` | 1 | 1 |
| `task296` | 1 | 0 |
| `task297` | 2 | 0 |
| `task298` | 0 | 1 |
| `task299` | 0 | 2 |
| `task300` | 3 | 6 |
| `task301` | 1 | 0 |
| `task302` | 1 | 0 |
| `task303` | 1 | 2 |
| `task304` | 0 | 5 |
| `task306` | 0 | 1 |
| `task307` | 0 | 1 |
| `task308` | 3 | 1 |
| `task310` | 1 | 0 |
| `task312` | 0 | 8 |
| `task313` | 1 | 2 |
| `task316` | 1 | 0 |
| `task317` | 0 | 2 |
| `task318` | 0 | 2 |
| `task319` | 3 | 0 |
| `task321` | 1 | 5 |
| `task323` | 3 | 0 |
| `task324` | 0 | 2 |
| `task325` | 1 | 0 |
| `task327` | 2 | 2 |
| `task328` | 1 | 0 |
| `task329` | 2 | 1 |
| `task330` | 1 | 0 |
| `task331` | 1 | 0 |
| `task333` | 1 | 0 |
| `task335` | 2 | 0 |
| `task336` | 1 | 0 |
| `task338` | 2 | 0 |
| `task340` | 2 | 0 |
| `task341` | 2 | 2 |
| `task342` | 1 | 1 |
| `task343` | 3 | 0 |
| `task345` | 2 | 4 |
| `task346` | 1 | 4 |
| `task348` | 2 | 5 |
| `task349` | 3 | 0 |
| `task350` | 2 | 2 |
| `task351` | 0 | 3 |
| `task353` | 1 | 2 |
| `task354` | 1 | 4 |
| `task355` | 1 | 0 |
| `task357` | 0 | 5 |
| `task358` | 1 | 0 |
| `task359` | 1 | 2 |
| `task360` | 1 | 10 |
| `task361` | 4 | 0 |
| `task362` | 0 | 2 |
| `task363` | 3 | 0 |
| `task365` | 5 | 8 |
| `task366` | 4 | 0 |
| `task367` | 4 | 0 |
| `task368` | 1 | 0 |
| `task370` | 1 | 0 |
| `task371` | 0 | 3 |
| `task374` | 1 | 0 |
| `task376` | 2 | 1 |
| `task377` | 7 | 2 |
| `task378` | 3 | 0 |
| `task379` | 5 | 0 |
| `task381` | 5 | 0 |
| `task382` | 3 | 3 |
| `task383` | 1 | 1 |
| `task384` | 3 | 0 |
| `task386` | 1 | 1 |
| `task387` | 3 | 0 |
| `task388` | 3 | 1 |
| `task389` | 1 | 2 |
| `task390` | 7 | 8 |
| `task391` | 1 | 1 |
| `task392` | 3 | 0 |
| `task394` | 3 | 0 |
| `task395` | 2 | 1 |
| `task396` | 3 | 0 |
| `task397` | 6 | 7 |
| `task398` | 2 | 1 |

## Archive Root Counts

| archive root | tracked files removed |
|---|---:|
| `dominated_archive_2026-06-21` | 200 |
| `incorporated_archive_20260620_152315` | 205 |
| `shipped_archive_20260607` | 370 |

## Notes For Future Agents

- Use `hard_records/` and the task-level `taskNNN.md` conclusions first; they are the durable proof layer.
- Use the exemplars above to understand manifest shapes and proof vocabulary, not as runnable payloads.
- If a future project needs active staging again, recreate `staged_candidates/` from the worker protocol in the new active repo rather than reviving this archived run history.
