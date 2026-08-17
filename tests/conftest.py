"""Degrade gracefully when the competition data is not present.

`tools/build_submission.py` imports `neurogolf_utils`, the module Kaggle ships
inside the competition dataset, by putting `<data>/neurogolf-2026/neurogolf_utils`
on `sys.path` at import time (see its `DATA_DIR` resolution). That dataset is not
vendored here -- it is ~670 MB and re-pullable with the Kaggle CLI -- so on a bare
checkout the modules below cannot even be collected, and pytest aborts the whole
run with a collection error.

With the data present (the owner's checkout, or after a `kaggle competitions
download`) every module runs. Without it, the seven data-dependent modules are
skipped and the remaining 30 still give a real signal, which is what CI runs.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Modules that transitively import neurogolf_utils.
DATA_DEPENDENT_MODULES = [
    "test_detect.py",
    "test_detector_builder_parity.py",
    "test_handoff_process_hardening.py",
    "test_neurogolf_pool_b_autoforeman.py",
    "test_neurogolf_sync_task_state.py",
    "test_rvb.py",
    "test_verify_against_live.py",
]


def _competition_data_present() -> bool:
    """True when neurogolf_utils can be found the same way build_submission finds it."""
    data_dir = os.environ.get("NEUROGOLF_DATA_DIR") or (
        REPO_ROOT / "data" / "competitions" / "neurogolf-2026"
    )
    return (Path(data_dir) / "neurogolf_utils").is_dir()


collect_ignore = [] if _competition_data_present() else list(DATA_DEPENDENT_MODULES)
