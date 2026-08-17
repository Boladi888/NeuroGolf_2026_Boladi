live_bar: cost 1161, sha256 ca47e6f02338f613e92d1a6e9ab87ea4cfffca5192ac7360a322e462aa41d65d, source `python tools/ng_current_bar.py --task 281 --json` manifest/Sho 2026-07-09T01:09:17Z

attempted_route: Deleted the two charged `Sign` nodes from the Sho 1161 team-best graph and fed pre-sign `F14r`/`F14c` carriers directly into the final free `Einsum`. Builder/probe: `The_400_Tasks/Task_by_Task/task281/prototypes/build_task281_raw_f14_probe.py` -> `task281_raw_f14_probe.onnx`.

failed_gate: `python tools/ng_gate.py visible --task 281 --cand The_400_Tasks/Task_by_Task/task281/prototypes/task281_raw_f14_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.

why_dead: The pre-sign carriers are not binary masks; official examples show positive counts ranging from 6 to 256. The final `Einsum` needs the post-`Sign` 0/1 masks, so raw count magnitudes corrupt every visible output.

remaining_hypothesis: A sub-1161 win needs either a legal dense parameter factorization of the Sho graph or a genuinely different output family. Do not retry raw `F14r`/`F14c` without a replacement binarizer that is cheaper than the existing F32 `Sign` outputs.
