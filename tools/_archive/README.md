# tools/_archive

Stale tool scripts moved out of `tools/` on 2026-05-17. None are imported by
active modules. They are kept here (rather than deleted) so future agents can
still grep for historical research recipes.

Do not import from `tools._archive`. If a script here turns out to still be
needed, move it back to `tools/` and re-cite it in the docs.

Contents:

- `peek_tasks.py` — one-off task dump utility
- `try_dispatcher.py` — exploratory dispatcher trial
- `summarize_orange5_ablation_queue.py` — historical ablation summary
- `apply_safe_opt_to_fig13b.py` — Figure-13b-specific safe-opt sweep
- `true_zeros_prototype.py` — early "true zeros" prototype probe
- `verify_new_tasks.py` — superseded by `verify_against_live.py`
- `verify_patterns.py` — superseded by `silent_fail_check.py`
- `task014_synthetic_guard.py` — task-specific guard, in Logs/lessons only
- `task153_synthetic_guard.py` — task-specific guard, in Logs/lessons only
- `build_task366_compressed_lookup.py` — task366 compressed-lookup recipe builder
- `verify_beta_recipes.py` — beta-recipe verification, see Research/notes/recipe-*
