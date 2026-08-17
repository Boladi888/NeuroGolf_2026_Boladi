# task118 — archived history

Pointer target for `The_400_Tasks/Task_by_Task/task118.md` `## COMPRESSED HISTORY`.
Moved 2026-07-08 by bo_ng_cld_worker3 to keep the live task doc under the 20 KB manual-content lint cap. Nothing here is a live route — the lookup family below is the Cherry13b zero-public-weight memorization branch that was abandoned in favour of the structural cross-cover rule (the shipped LIVE graph and the current best).

## Lookup-family cost analysis (SUPERSEDED — Cherry13b memorization suspect, do not revive as a ship)

Best LIVE-gate/LB candidate: `task118_packedrc_twostage_ordered_tiergroups_i32scalar_lb.onnx`, official score `(macs=0, memory=1959, params=1890)`, cost 3,849.

Best robust no-match/adversarial candidate remains `task118_packedrc_twostage_ordered_tiergroups_i32scalar.onnx`, official score `(macs=0, memory=2385, params=1925)`, cost 4,310.

The current LB branch intentionally drops the no-match dummy fallback and therefore should be treated as LIVE-gate-only. Owner direction on 2026-05-23: adversarial does not matter; LIVE equivalence matters.

Main savings vs live 20,762 (lookup family):
- Replaced `[1,267,11]` Equal/Cast/ReduceMin slab and `[1,267]` F32 weighted row-id recovery with a two-stage bucket lookup: primary 5-bit bucket, secondary 6-bit compare inside an 11-slot bucket, and one packed INT32 slot carrying both secondary code and row id.
- Replaced padded `[267,22]` position table with packed base-30 row/col groups: four INT64 groups per matched row, each carrying six row/col pairs.
- Removed the 901st sentinel cell, Slice, and final mask reshape by scattering into a `[30,30]` mask and relying on `Where` broadcasting.
- Removed the false-update pad machinery by padding each row with duplicate true target cells.
- Remapped the internal row ids so rows needing packed groups 2/3/4 are contiguous. This allows threshold-gather fallbacks instead of a full three-group table for every row. The packed-bucket dummy slots now point to the new id for original row 0, preserving LIVE's no-match fallback.
- Kept the bucket-selected row id in INT32 and squeezed the `[1,1]` gather result to a scalar. That removes the INT64 cast/reshape lane and trims the robust candidate by 36 cost.
- For the LB branch, removed the no-match dummy slot, reducing the bucket table to `[32,10]`.
- Fed `ScatterND` indices as `[4,6,2]` directly. `ScatterND` accepts the final dimension as the index tuple, so the `[24,2]` Reshape and its 384-byte intermediate are unnecessary.
- Reshaped packed group tables to `[N,1,1,1]`; concatenating four selected group rows now directly produces `[4,1,1]`, avoiding the unpack `Unsqueeze`.

Remaining LIVE-gate cost is dominated by the 900-cell BOOL mask tail, the 900-param zero mask initializer, and two INT64 unpack intermediates for the 24 ScatterND row/col pairs. A further major drop probably needs a scoreable way to avoid the dense zero canvas/mask pair, or a completely different exact rule implementation that stays out of MAC-heavy compare/reduce ops.

(The lookup-family Attempts rows 2026-05-19..2026-05-23 remain in the live task doc's Attempts table for continuity; the structural branch superseded all of them.)
