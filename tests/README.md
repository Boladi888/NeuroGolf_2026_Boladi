# tests/

Regression tests for NeuroGolf tooling, submit safety checks, orchestration helpers, and detector/builder parity.

## Running

```bash
pytest tests/ -q
```

`pytest.ini` pins the test root to this directory. Do not use `python -m unittest discover`
as the main runner: most modules are pytest-style functions with fixtures, so unittest imports
them but silently collects only the `unittest.TestCase` subset.

From any worktree without `.venv` in-tree, use `${NEUROGOLF_PYTHON}` (see `../WHERE_DATA_LIVES.md`).

## Files

- `test_detect.py` — large file covering most detector/builder pairs. Per-task and per-pattern.
- `test_detector_builder_parity.py` — strict-parity gate. Verifies that every `tools.inspect_tasks.SOLVED_TASKS` entry has both a working detector AND a Kaggle-faithful `convert_from_numpy` ONNX. **This is the gate for promoting any task to SOLVED.**

## Conventions

- Use `NEUROGOLF_DATA_DIR` env var to point at the task JSONs; default falls through to `../data/competitions/neurogolf-2026/` (which is a junction — see `../WHERE_DATA_LIVES.md`).
- The detector/parity suite is not portable to a fresh clone without `data/` and the project import path. In the cleanup audit, forcing `NEUROGOLF_DATA_DIR` to a nonexistent path left only a small subset collectable and hard-crashed modules including `test_detect.py`, `test_detector_builder_parity.py`, `test_neurogolf_pool_b_autoforeman.py`, `test_rvb.py`, and `test_verify_against_live.py`.
- Tests must pass before any future tooling or archive-fix PR opens.
- Adding a new pattern: add detector tests + builder tests + add to `SOLVED_TASKS` only after the strict-parity gate passes.

## What to do when a test fails

1. Run just that test with `--verbose` to see the failure.
2. Check `tools/inspect_tasks.py:SOLVED_TASKS` — is the failing task supposed to be SOLVED at all?
3. If you broke a passing test with an unrelated edit: roll back that edit and verify isolation.
4. If a Kaggle-faithful parity test failed: the ONNX output and the numpy-verifier output disagree. The numpy verifier is the spec; fix the ONNX.
