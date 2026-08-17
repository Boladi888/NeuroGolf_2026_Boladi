"""The 5 gate modes + battery. Each returns a structured dict.

Inference uses single-threaded sessions (session.py) over the cached corpus
(corpus.py); the caller (ng_gate.py) holds one throttle slot for the whole run.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np

from . import corpus, encode, registry, session

_NG_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_NG_ROOT / "tools"))


def _score(cand):
    try:
        from _score_v2 import score_network_v2
        r = score_network_v2(str(cand))
        if r:
            mem, par = r[1], r[2]
            return {"memory": mem, "params": par, "cost": mem + par,
                    "points": max(1.0, 25.0 - math.log(max(1.0, mem + par)))}
    except Exception:  # noqa: BLE001
        pass
    return None


# ── visible ───────────────────────────────────────────────────────────────────
def mode_visible(task_num, cand, sanitize=True, **_):
    s, name = session.make_session(cand, sanitize=sanitize)
    ok = tot = 0
    first_fail = None
    for p in registry.official_pairs(task_num):
        xi = encode.encode_grid(p["input"]); ei = encode.encode_grid(p["output"])
        if xi is None or ei is None:
            continue
        tot += 1
        if np.array_equal(session.run_raw(s, name, xi), ei):
            ok += 1
        elif first_fail is None:
            first_fail = p["input"]
    return {"mode": "visible", "correct": ok, "total": tot,
            "pct": 100.0 * ok / max(1, tot), "pass": tot > 0 and ok == tot,
            "first_fail": first_fail}


# ── truth (vs ARC-GEN ground truth; optional --ref variant agreement) ───────────
def mode_truth(task_num, cand, n=2000, ref=None, sanitize=True, verbose=False, **_):
    s, name = session.make_session(cand, sanitize=sanitize)
    sr = nr = None
    if ref:
        sr, nr = session.make_session(ref, sanitize=sanitize)
    ok = ref_agree = tot = 0
    first_fail = None
    for X, Y, s0 in corpus.iter_tensors(task_num, n, verbose=verbose):  # streamed: bounds memory
        for j in range(X.shape[0]):
            out = session.run_raw(s, name, X[j:j+1])
            if np.array_equal(out, Y[j:j+1]):
                ok += 1
            elif first_fail is None:
                first_fail = s0 + j
            if sr is not None and np.array_equal(out, session.run_raw(sr, nr, X[j:j+1])):
                ref_agree += 1
            tot += 1
    n = tot
    res = {"mode": "truth", "correct": ok, "total": n, "pct": 100.0 * ok / max(1, n),
           "first_fail_idx": first_fail}
    if sr is not None:
        res["ref_agree"] = ref_agree
        res["ref_agree_pct"] = 100.0 * ref_agree / max(1, n)
    return res


# ── ab (byte-equal vs a reference onnx: parent/live/variant) ────────────────────
def mode_ab(task_num, cand, ref=None, n=2000, early_exit=False, sanitize=True, verbose=False, **_):
    if ref:
        src = "explicit"
    else:
        ref, src = registry.parent_onnx(task_num)
        if ref is None:
            return {"mode": "ab", "error": src}
    sc, nc = session.make_session(cand, sanitize=sanitize)
    sr, nr = session.make_session(ref, sanitize=sanitize)
    match = tot = 0
    first_div = None
    stop = False

    def _cmp(x):
        nonlocal match, tot, first_div, stop
        tot += 1
        if np.array_equal(session.run_raw(sc, nc, x), session.run_raw(sr, nr, x)):
            match += 1
        else:
            if first_div is None:
                first_div = tot - 1
            if early_exit:
                stop = True

    # official inputs first (small)
    official = [encode.encode_grid(p["input"]) for p in registry.official_pairs(task_num)]
    official = [x for x in official if x is not None]
    n_official = len(official)
    for x in official:
        _cmp(x)
        if stop:
            break
    # then streamed corpus inputs (memory-bounded)
    if not stop:
        for X, _Y, _s0 in corpus.iter_tensors(task_num, n, verbose=verbose):
            for j in range(X.shape[0]):
                _cmp(X[j:j+1])
                if stop:
                    break
            if stop:
                break
    return {"mode": "ab", "ref_source": src, "match": match, "total": tot,
            "divergences": tot - match, "n_official": n_official,
            "first_divergence": first_div, "byte_equal": match == tot}


# ── synth (vs an agent-supplied numpy reference: module with solve(grid)->grid) ──
def mode_synth(task_num, cand, synth_ref=None, n=2000, sanitize=True, **_):
    if not synth_ref:
        return {"mode": "synth", "error": "pass --synth-ref path/to/ref.py exposing solve(grid)->grid"}
    spec = importlib.util.spec_from_file_location("ng_synth_ref", synth_ref)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "solve"):
        return {"mode": "synth", "error": f"{synth_ref} has no solve(grid)->grid"}
    s, name = session.make_session(cand, sanitize=sanitize)
    d = corpus.build(task_num, n)
    nn = min(n, len(d["in_idx"]))
    agree = tot = 0
    first_fail = None
    for i in range(nn):
        h, w = int(d["in_shapes"][i][0]), int(d["in_shapes"][i][1])
        grid = d["in_idx"][i][:h, :w].astype(int).tolist()
        try:
            ref_grid = mod.solve(grid)
        except Exception:  # noqa: BLE001
            continue
        ref_t = encode.encode_grid(ref_grid)
        if ref_t is None:
            continue
        x = encode.encode_batch_index(d["in_idx"][i:i+1], d["in_shapes"][i:i+1])
        tot += 1
        if np.array_equal(session.run_raw(s, name, x), ref_t):
            agree += 1
        elif first_fail is None:
            first_fail = i
    return {"mode": "synth", "agree": agree, "total": tot,
            "pct": 100.0 * agree / max(1, tot), "first_fail_idx": first_fail}


# ── adversarial (wrap tools/adversarial_test.py: cand vs LIVE on perturbations) ──
def mode_adversarial(task_num, cand, sanitize=True, **_):
    try:
        import adversarial_test as adv
        live = registry.SUBMISSION_ZIP
        res = adv.adversarial_test(str(cand), task_num, live, registry.DATA_DIR)
        res["mode"] = "adversarial"
        return res
    except Exception as e:  # noqa: BLE001
        return {"mode": "adversarial", "error": f"{type(e).__name__}: {e}"}


# ── battery (the standard pre-ship combo) ───────────────────────────────────────
def mode_battery(task_num, cand, n=5000, ref=None, sanitize=True, verbose=False, **_):
    vis = mode_visible(task_num, cand, sanitize=sanitize)
    tru = mode_truth(task_num, cand, n=n, ref=ref, sanitize=sanitize, verbose=verbose)
    ab = mode_ab(task_num, cand, ref=ref, n=min(n, 3000), sanitize=sanitize)
    sc = _score(cand)
    verdict = "PASS" if (vis["pass"] and tru["pct"] >= 99.0) else "REVIEW"
    return {"mode": "battery", "task": task_num, "verdict": verdict,
            "visible": vis, "truth": tru, "ab": ab, "score": sc}


DISPATCH = {
    "visible": mode_visible, "truth": mode_truth, "ab": mode_ab,
    "synth": mode_synth, "adversarial": mode_adversarial, "battery": mode_battery,
}
