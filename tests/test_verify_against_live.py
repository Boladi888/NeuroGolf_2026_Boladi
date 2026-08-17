from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

NG_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NEUROGOLF_DATA_DIR", NG_ROOT / "data" / "competitions" / "neurogolf-2026"))

sys.path.insert(0, str(NG_ROOT))
sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))

import neurogolf_utils as nu  # noqa: E402
from tools import verify_against_live as vpl  # noqa: E402

nu._NEUROGOLF_DIR = str(DATA_DIR) + "/"


def test_full_tensor_compare_catches_padding_only_difference():
    example = {"input": [[7]], "output": [[3]]}
    benchmark = nu.convert_to_numpy(example)
    assert benchmark is not None

    proto_out = benchmark["output"]
    live_out = benchmark["output"].copy()
    live_out[0, 0, 1, 0] = 1.0  # padding-only drift that legacy crop mode would ignore

    full = vpl.compare_full_tensor_outputs(proto_out, live_out, benchmark["output"])
    assert full["correct_proto"] is True
    assert full["correct_live"] is False
    assert full["match_live"] is False


def test_legacy_cropped_compare_ignores_padding_only_difference():
    example = {"input": [[7]], "output": [[3]]}
    benchmark = nu.convert_to_numpy(example)
    assert benchmark is not None

    proto_out = benchmark["output"]
    live_out = benchmark["output"].copy()
    live_out[0, 0, 1, 0] = 1.0  # still outside the visible 1x1 crop

    legacy = vpl.compare_cropped_outputs(proto_out, live_out, example["output"])
    assert legacy["correct_proto"] is True
    assert legacy["correct_live"] is True
    assert legacy["match_live"] is True


def test_legacy_encode_uses_ch0_padding_but_official_benchmark_does_not():
    example = {"input": [[7, 8], [9, 0]], "output": [[3, 4], [5, 6]]}
    benchmark = nu.convert_to_numpy(example)
    assert benchmark is not None

    legacy_in = vpl.legacy_encode(example["input"])
    official_in = benchmark["input"]

    assert np.array_equal(official_in, legacy_in) is False
    assert np.all(official_in[0, 0, 2:, :] == 0.0)
    assert np.all(official_in[0, 0, :2, 2:] == 0.0)
    assert np.all(legacy_in[0, 0, 2:, :] == 1.0)
    assert np.all(legacy_in[0, 0, :2, 2:] == 1.0)
