"""score.py — cost + correctness for a candidate model.

cost: prefers the canonical tools/_score_v2.py (byte-exact vs Kaggle, per core.md);
falls back to a self-contained STATIC estimate (params = sum of initializer elements;
memory = sum of intermediate value_info bytes after shape inference, skipping input/output)
so ml/ngs runs even if the repo scorer is unavailable.

correctness: the official encode -> ORT run -> (>0) threshold -> compare against the
encoded expected output. Returns (n_correct, n_total). One miss = a real miss (the
§VISIBLE-FAIL axiom: a single visible miss is a certain hidden 0).
"""
import importlib.util
import pathlib
import tempfile
import numpy as np
import onnx
import onnxruntime as ort

from . import spec

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ort.set_default_logger_severity(3)   # quiet

_NP = {spec.FLOAT: np.float32, spec.FLOAT16: np.float16, spec.BOOL: np.bool_,
       spec.INT32: np.int32, spec.INT64: np.int64, spec.UINT8: np.uint8, spec.INT8: np.int8}


def _load_repo_scorer():
    p = REPO_ROOT / "tools" / "_score_v2.py"
    if not p.exists():
        return None
    try:
        s = importlib.util.spec_from_file_location("_ngs_score_v2", p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return getattr(m, "score_network_v2", None)
    except Exception:
        return None


_REPO_SCORER = _load_repo_scorer()


def static_cost(model):
    """Self-contained STATIC cost = params + intermediate-memory bytes (excl. input/output)."""
    params = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        params += n if init.dims else 1
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        inferred = model
    mem = 0
    for vi in list(inferred.graph.value_info):
        if vi.name in (spec.IN_NAME, spec.OUT_NAME):
            continue
        tt = vi.type.tensor_type
        n = 1
        for d in tt.shape.dim:
            n *= (d.dim_value or 1)
        mem += n * spec.ITEMSIZE.get(tt.elem_type, 4)
    return int(mem + params)


def exact_cost(model):
    """Canonical cost via tools/_score_v2.score_network_v2 (writes a temp .onnx); fallback static."""
    if _REPO_SCORER is None:
        return static_cost(model), "static"
    try:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        onnx.save(model, path)
        res = _REPO_SCORER(path)              # -> (macs, memory, params) or None
        pathlib.Path(path).unlink(missing_ok=True)
        if not res:
            return static_cost(model), "static(scorer-None)"
        _, memory, params = res
        return int(memory + params), "exact"
    except Exception:
        return static_cost(model), "static(scorer-err)"


GPU_EPS = ["OpenVINOExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"]


def pick_providers():
    """Best ONNX execution provider available in THIS interpreter (GPU first, CPU last).
    On the project CPU-only venv this is just CPU; in an Arc-B70 ONNX env (OpenVINO/DML EP)
    it accelerates the ORT-bound search. Use ml.ngs.backends to find such an env."""
    avail = set(ort.get_available_providers())
    return [p for p in GPU_EPS if p in avail] + ["CPUExecutionProvider"]


def _run(mb, pairs, providers):
    sess = ort.InferenceSession(mb, providers=providers)
    passed = []
    for ig, og in pairs:
        try:
            res = sess.run([spec.OUT_NAME], {spec.IN_NAME: spec.encode(ig)})[0]
            pred = (np.asarray(res) > 0.0).astype(np.float32)
            passed.append(bool(pred.shape == spec.IN_SHAPE and np.array_equal(pred, spec.encode(og))))
        except Exception:
            passed.append(False)
    return passed


def evaluate(model, pairs, providers=None):
    """pairs: list[(in_grid,out_grid)] -> (n_correct, n_total) via the official encode->ORT->>0->verify path.
    A full solve found on a GPU EP is auto re-confirmed on CPU (the competition is CPU-scored; GPU float
    ops can differ at boundaries) — only full-solvers pay that recheck, so search stays fast and exact."""
    providers = providers or pick_providers()
    mb = model.SerializeToString()
    try:
        passed = _run(mb, pairs, providers)
    except Exception:
        return 0, len(pairs)
    if pairs and all(passed) and providers[0] != "CPUExecutionProvider":
        try:
            passed = _run(mb, pairs, ["CPUExecutionProvider"])
        except Exception:
            return 0, len(pairs)
    return sum(passed), len(pairs)


# ---- ARC-GEN truth gate ----------------------------------------------------------------
# A program that reproduces the ~5 VISIBLE pairs but is the wrong rule is a certain hidden 0
# (§VISIBLE-FAIL). Before trusting a solve, re-verify it on a large held-out ARC-GEN sample
# from BoBrain's argen corpus (ml.ngs.argen). This is the "arc-gen truth gate" the README asks
# for. Runs the whole sample in ONE batched ORT call (the ops are batch-polymorphic), which is
# also the reused-model / many-inputs regime where a GPU EP finally pays off.

def _batch_model(model, n):
    """Clone `model` with input/output batch dim set to n (ops here are all batch-polymorphic).
    value_info is cleared so ORT re-infers intermediate shapes at the new batch size."""
    import copy
    m = copy.deepcopy(model)
    for vi in list(m.graph.input) + list(m.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if dims:
            dims[0].ClearField("dim_param")
            dims[0].dim_value = n
    m.graph.ClearField("value_info")
    return m


def argen_gate(model, tid, n=256, seed=0, providers=None):
    """Verify `model` reproduces ARC-GEN outputs for task `tid` on up to n sampled pairs.
    Returns (n_pass, n_total); (0, 0) when the corpus is unavailable so callers can no-op.
    Batched single-session run, with a per-sample fallback if the batched run errors."""
    from . import argen
    s = argen.sample(tid, n, seed)
    if s is None:
        return 0, 0
    m = s["n"]
    X = s["X"]                                            # [m,10,30,30] f32
    Y = argen.encode_batch(s["out_idx"], s["out_shapes"])  # expected one-hot [m,10,30,30]
    providers = providers or pick_providers()
    try:
        sess = ort.InferenceSession(_batch_model(model, m).SerializeToString(), providers=providers)
        res = np.asarray(sess.run([spec.OUT_NAME], {spec.IN_NAME: X})[0])
        if res.shape != Y.shape:
            raise ValueError("batched output shape mismatch")
        pred = (res > 0.0).astype(np.float32)
        per = np.all(pred == Y, axis=(1, 2, 3))          # [m] bool, per-pair exact match
        return int(per.sum()), m
    except Exception:
        pass
    # fallback: per-sample (batch dim stays 1)
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception:
        return 0, m
    npass = 0
    for i in range(m):
        try:
            r = sess.run([spec.OUT_NAME], {spec.IN_NAME: X[i:i + 1]})[0]
            pred = (np.asarray(r) > 0.0).astype(np.float32)
            if pred.shape == spec.IN_SHAPE and np.array_equal(pred, Y[i:i + 1]):
                npass += 1
        except Exception:
            pass
    return npass, m
