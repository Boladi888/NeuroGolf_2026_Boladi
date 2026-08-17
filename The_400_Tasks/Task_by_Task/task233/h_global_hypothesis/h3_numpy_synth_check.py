"""Validate the H3 numpy design at scale against the existing reference + synth gate."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / ".tmp_h3"))
sys.path.insert(0, str(REPO / "The_400_Tasks/Task_by_Task/task233/prototypes"))

from h3_numpy_design import signed_score_h3
from task233_numpy_reference import predict_task233
from task233_synthetic_gate import synthetic_case


def check(n_dihedral, n_legacy):
    print(f"--- {n_dihedral} stress_dihedral seeds + {n_legacy} legacy seeds ---")
    h3_eq_ref = 0
    h3_eq_constructed = 0
    ref_eq_constructed = 0
    first_h3_fail = None
    first_ref_fail = None

    # Dihedral-stress
    for i in range(n_dihedral):
        seed = 233233 + i
        ex = synthetic_case(seed, stress_dihedral=True)
        g = np.asarray(ex["input"])
        out_c = np.asarray(ex["output"])
        ref = np.asarray(predict_task233(g.tolist()))
        h3 = np.asarray(signed_score_h3(g))
        if (h3 == ref).all():
            h3_eq_ref += 1
        if (h3 == out_c).all():
            h3_eq_constructed += 1
        else:
            if first_h3_fail is None:
                first_h3_fail = ("dihedral", i, seed)
        if (ref == out_c).all():
            ref_eq_constructed += 1
        else:
            if first_ref_fail is None:
                first_ref_fail = ("dihedral", i, seed)
    print(f"  dihedral: h3==ref {h3_eq_ref}/{n_dihedral}; h3==constructed {h3_eq_constructed}/{n_dihedral}; ref==constructed {ref_eq_constructed}/{n_dihedral}")
    if first_h3_fail:
        print(f"    first h3 fail: {first_h3_fail}")
    if first_ref_fail:
        print(f"    first ref fail: {first_ref_fail}")

    # Legacy random (uses predict_task233 as oracle, so this checks h3 against ref)
    h3_eq_ref = 0
    first_h3_fail = None
    for i in range(n_legacy):
        seed = 233233 + i
        ex = synthetic_case(seed, stress_dihedral=False)
        g = np.asarray(ex["input"])
        ref = np.asarray(ex["output"])  # equals predict_task233 by synth gate code
        h3 = np.asarray(signed_score_h3(g))
        if (h3 == ref).all():
            h3_eq_ref += 1
        else:
            if first_h3_fail is None:
                first_h3_fail = ("legacy", i, seed)
    print(f"  legacy:   h3==ref {h3_eq_ref}/{n_legacy}")
    if first_h3_fail:
        print(f"    first h3 fail: {first_h3_fail}")


if __name__ == "__main__":
    check(n_dihedral=1000, n_legacy=1000)
