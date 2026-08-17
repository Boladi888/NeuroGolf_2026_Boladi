# no_candidate_reshape

- Live bar: manifest cost 920, SHA `219bccc70d68bf8ea5505adad4aebafb0b5a57775ff83cf6f8545e19d2fa362b`.
- Attempted route: remove `Reshape(candidate_3d -> candidate[1])` and feed `[1,1,1]` `candidate_3d` directly into `Gather` start lookups and downstream `Slice` starts.
- Failed gate: cheap score/load, `python The_400_Tasks/Task_by_Task/task124/prototypes/build_task124_worker2_no_candidate_reshape.py` returned `score=None`.
- Why dead: ORT runtime rejected the first dynamic `Slice` with `Starts must be a 1-D array`; `candidate_3d` rank cannot feed `Slice` starts.
- Remaining hypothesis: a one-dimensional candidate still needs an explicit reshape/squeeze-equivalent 4-byte tensor unless the upstream ArgMax path is rebuilt to emit `[1]` directly without adding larger row squeezes.
