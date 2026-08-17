"""Repo-native Sho-style safety checks for NeuroGolf ONNX candidates.

This module is intentionally separate from the heavyweight correctness gates in
``ng_gate``. It answers a different question: "is this model or bundle safe
enough to promote without obvious submit-killer risks?"
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import pathlib
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from . import registry

INT_TYPES = {2, 3, 4, 5, 6, 7, 12, 13}  # UINT8, INT8, UINT16, INT16, INT32, INT64, UINT32, UINT64
DEFAULT_FILESIZE_LIMIT = int(1.44 * 1024 * 1024)

RISK_FLAG_OPS = {
    "Div",
    "Mod",
    "Cast",
    "QLinearConv",
    "ConvInteger",
}

X86_WARN_OPS = {
    "QLinearConv",
    "ConvInteger",
}

GRADER_GAP_RISK_OPS = {
    "If",
    "Loop",
    "Scan",
    "NonZero",
    "Unique",
    "Compress",
    "SequenceConstruct",
    "SequenceAt",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
}


@dataclass
class ModelBlob:
    task: int | None
    name: str
    blob: bytes
    source: str


def _data_dir() -> pathlib.Path:
    return registry.DATA_DIR


def _neurogolf_utils():
    data_dir = _data_dir()
    sys.path.insert(0, str(data_dir / "neurogolf_utils"))
    import neurogolf_utils as ngu  # type: ignore

    ngu._NEUROGOLF_DIR = str(data_dir).replace("\\", "/") + "/"
    return ngu


def _file_limit() -> int:
    try:
        return int(getattr(_neurogolf_utils(), "_FILESIZE_LIMIT_IN_BYTES"))
    except Exception:  # noqa: BLE001
        return DEFAULT_FILESIZE_LIMIT


def _banned_ops() -> set[str]:
    try:
        return {str(op).upper() for op in getattr(_neurogolf_utils(), "_EXCLUDED_OP_TYPES", [])}
    except Exception:  # noqa: BLE001
        return set()


def _sanitize(model: onnx.ModelProto) -> onnx.ModelProto | None:
    try:
        ngu = _neurogolf_utils()
        if hasattr(ngu, "sanitize_model"):
            san = ngu.sanitize_model(model)
            if not san:
                return None
            return san
    except Exception:  # noqa: BLE001
        return model
    return model


def _make_ort_session(model: onnx.ModelProto):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 4
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    try:
        opts.add_session_config_entry("mlas.disable_kleidiai", "1")
    except Exception:  # noqa: BLE001
        pass
    return ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])


def _parse_task_from_name(name: str) -> int | None:
    stem = pathlib.Path(name).stem.lower()
    if stem.startswith("task"):
        stem = stem[4:]
    if stem.isdigit():
        n = int(stem)
        if 1 <= n <= 400:
            return n
    return None


def _value_type_map(model: onnx.ModelProto) -> dict[str, int]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:  # noqa: BLE001
        inferred = model
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    out: dict[str, int] = {}
    for vi in values:
        elem = vi.type.tensor_type.elem_type
        if elem:
            out[vi.name] = int(elem)
    return out


def _op_inventory(model: onnx.ModelProto) -> dict[str, int]:
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def _official_io_contract(model: onnx.ModelProto) -> dict[str, Any]:
    initializer_names = {init.name for init in model.graph.initializer}
    data_inputs = [inp.name for inp in model.graph.input if inp.name not in initializer_names]
    outputs = [out.name for out in model.graph.output]
    failures: list[dict[str, Any]] = []

    if "input" not in data_inputs:
        failures.append({
            "code": "FAIL_OFFICIAL_INPUT_CONTRACT",
            "inputs": data_inputs,
            "expected": "input",
        })
    extra_inputs = sorted(name for name in data_inputs if name != "input")
    if extra_inputs:
        failures.append({
            "code": "FAIL_OFFICIAL_EXTRA_INPUTS",
            "inputs": data_inputs,
            "extra_inputs": extra_inputs,
        })
    if "output" not in outputs:
        failures.append({
            "code": "FAIL_OFFICIAL_OUTPUT_CONTRACT",
            "outputs": outputs,
            "expected": "output",
        })

    return {
        "data_inputs": data_inputs,
        "outputs": outputs,
        "ok": not failures,
        "failures": failures,
    }


def _runtime_denominators(model: onnx.ModelProto, include_input_denominator: bool = True) -> tuple[list[dict], list[dict]]:
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    constant_names = {node.output[0] for node in graph.node if node.op_type == "Constant" and node.output}
    dtype_by_name = _value_type_map(model)
    integer_hits: list[dict] = []
    unknown_hits: list[dict] = []

    for idx, node in enumerate(graph.node):
        if node.op_type not in {"Div", "Mod"} or len(node.input) < 2:
            continue
        denom = node.input[1]
        if denom in initializer_names or denom in constant_names:
            continue
        if denom == "input" and not include_input_denominator:
            continue
        elem_type = dtype_by_name.get(denom)
        rec = {
            "node_index": idx,
            "op_type": node.op_type,
            "node_name": node.name or "",
            "denominator": denom,
            "elem_type": elem_type,
        }
        if elem_type in INT_TYPES:
            integer_hits.append(rec)
        elif elem_type is None:
            unknown_hits.append(rec)
    return integer_hits, unknown_hits


def _dynamic_official_denominator_probe(blob: ModelBlob, model: onnx.ModelProto, denoms: list[str]) -> dict:
    if blob.task is None:
        return {"reason": "no_task_number", "n_boards": 0, "zero_hits": None, "verdict": "SKIP"}

    pairs = registry.official_pairs(blob.task)
    if not pairs:
        return {"reason": "no_official_pairs", "n_boards": 0, "zero_hits": None, "verdict": "SKIP"}

    dtype_by_name = _value_type_map(model)
    probe_model = copy.deepcopy(model)
    existing_outputs = {out.name for out in probe_model.graph.output}
    for denom in denoms:
        if denom not in existing_outputs:
            elem_type = dtype_by_name.get(denom, onnx.TensorProto.INT64)
            probe_model.graph.output.append(onnx.helper.make_tensor_value_info(denom, elem_type, None))

    try:
        sess = _make_ort_session(probe_model)
    except Exception:
        san = _sanitize(probe_model)
        if san is None:
            return {"reason": "probe_sanitize_failed", "n_boards": 0, "zero_hits": None, "verdict": "ERROR"}
        try:
            sess = _make_ort_session(san)
        except Exception as exc:  # noqa: BLE001
            return {"reason": f"probe_load_error:{type(exc).__name__}:{str(exc)[:120]}",
                    "n_boards": 0, "zero_hits": None, "verdict": "ERROR"}

    ngu = _neurogolf_utils()
    zero_hits = 0
    n_boards = 0
    for ex in pairs:
        bm = ngu.convert_to_numpy(ex)
        if not bm:
            continue
        try:
            outs = sess.run(denoms, {"input": bm["input"]})
        except Exception as exc:  # noqa: BLE001
            return {"reason": f"probe_run_error:{type(exc).__name__}:{str(exc)[:120]}",
                    "n_boards": n_boards, "zero_hits": None, "verdict": "ERROR"}
        n_boards += 1
        for arr in outs:
            zero_hits += int((np.asarray(arr) == 0).sum())

    return {
        "reason": "ok",
        "n_boards": n_boards,
        "zero_hits": zero_hits,
        "verdict": "FAIL_DIV0" if zero_hits else "OK",
    }


def scan_model(blob: ModelBlob, *, load: bool = True, dynamic_official: bool = False,
               include_input_denominator: bool = True) -> dict[str, Any]:
    limit = _file_limit()
    row: dict[str, Any] = {
        "task": None if blob.task is None else f"task{blob.task:03d}",
        "name": blob.name,
        "source": blob.source,
        "sha256": hashlib.sha256(blob.blob).hexdigest(),
        "size_bytes": len(blob.blob),
        "failures": [],
        "warnings": [],
    }
    if len(blob.blob) > limit:
        row["failures"].append({
            "code": "FAIL_FILESIZE",
            "size_bytes": len(blob.blob),
            "limit_bytes": limit,
        })

    try:
        model = onnx.load_model_from_string(blob.blob)
    except Exception as exc:  # noqa: BLE001
        row["failures"].append({"code": "FAIL_ONNX_PARSE", "detail": f"{type(exc).__name__}: {str(exc)[:160]}"})
        row["op_inventory"] = {}
        row["risk_op_flags"] = []
        row["runtime_integer_denominators"] = []
        row["runtime_denominators_unknown_dtype"] = []
        row["verdict"] = "FAIL"
        return row

    row["op_inventory"] = _op_inventory(model)
    banned_upper = _banned_ops()
    banned = sorted(
        op for op in row["op_inventory"]
        if op.upper() in banned_upper or "Sequence" in op
    )
    if banned:
        row["failures"].append({
            "code": "FAIL_BANNED_OP",
            "ops": banned,
            "official_ops": sorted({op.upper() for op in banned}),
        })

    row["risk_op_flags"] = sorted(set(row["op_inventory"]) & RISK_FLAG_OPS)
    risky_x86 = sorted(set(row["op_inventory"]) & X86_WARN_OPS)
    if risky_x86:
        row["warnings"].append({"code": "WARN_X86_QUANTIZED_OP_FAMILY", "ops": risky_x86})

    risky_gap = sorted(set(row["op_inventory"]) & GRADER_GAP_RISK_OPS)
    if risky_gap:
        row["warnings"].append({"code": "WARN_GRADER_GAP_OP_FAMILY", "ops": risky_gap})

    integer_denoms, unknown_denoms = _runtime_denominators(model, include_input_denominator=include_input_denominator)
    row["runtime_integer_denominators"] = integer_denoms
    row["runtime_denominators_unknown_dtype"] = unknown_denoms
    if integer_denoms:
        row["warnings"].append({
            "code": "WARN_X86_REQUIRED_RUNTIME_INT_DIVMOD",
            "count": len(integer_denoms),
            "denominators": sorted({rec["denominator"] for rec in integer_denoms}),
        })
    if unknown_denoms:
        row["warnings"].append({
            "code": "WARN_RUNTIME_DIVMOD_UNKNOWN_DTYPE",
            "count": len(unknown_denoms),
            "denominators": sorted({rec["denominator"] for rec in unknown_denoms}),
        })

    if dynamic_official and integer_denoms:
        denoms = sorted({rec["denominator"] for rec in integer_denoms})
        dyn = _dynamic_official_denominator_probe(blob, model, denoms)
        row["dynamic_official_denominator_probe"] = dyn
        if dyn.get("verdict") == "FAIL_DIV0":
            row["failures"].append({"code": "FAIL_DYNAMIC_OFFICIAL_DIV0", **dyn})
        elif dyn.get("verdict") == "ERROR":
            row["warnings"].append({"code": "WARN_DYNAMIC_OFFICIAL_PROBE_ERROR", **dyn})

    san = _sanitize(copy.deepcopy(model))
    if san is None:
        row["official_io_contract"] = {"ok": False, "failures": [{"code": "FAIL_SANITIZE"}]}
        row["failures"].append({"code": "FAIL_SANITIZE"})
    else:
        io_contract = _official_io_contract(san)
        row["official_io_contract"] = {
            key: value for key, value in io_contract.items() if key != "failures"
        }
        row["failures"].extend(io_contract["failures"])
        if load:
            try:
                _make_ort_session(san)
                row["load_after_sanitize"] = "ok"
            except Exception as exc:  # noqa: BLE001
                row["load_after_sanitize"] = "fail"
                row["failures"].append({
                    "code": "FAIL_ORT_LOAD_AFTER_SANITIZE",
                    "detail": f"{type(exc).__name__}: {str(exc)[:160]}",
                })

    row["verdict"] = "FAIL" if row["failures"] else ("WARN" if row["warnings"] else "PASS")
    return row


def _bundle_entries(zpath: pathlib.Path) -> tuple[list[ModelBlob], dict[str, Any]]:
    models: list[ModelBlob] = []
    with zipfile.ZipFile(zpath) as zf:
        onnx_names = sorted(name for name in zf.namelist() if name.lower().endswith(".onnx"))
        base_names = [pathlib.PurePosixPath(name.replace("\\", "/")).name for name in onnx_names]
        counts = Counter(base_names)
        expected = {f"task{i:03d}.onnx" for i in range(1, 401)}
        present = set(base_names)
        manifest = {
            "n_onnx_entries": len(onnx_names),
            "n_unique_basenames": len(present),
            "missing": sorted(expected - present),
            "extra": sorted(present - expected),
            "duplicates": sorted(name for name, count in counts.items() if count > 1),
        }
        for name in onnx_names:
            base_name = pathlib.PurePosixPath(name.replace("\\", "/")).name
            models.append(ModelBlob(
                task=_parse_task_from_name(name),
                name=base_name,
                blob=zf.read(name),
                source=f"{zpath}::{name}",
            ))
    return models, manifest


def scan_bundle(zpath: pathlib.Path, *, load: bool = True, dynamic_official: bool = False,
                include_input_denominator: bool = True) -> dict[str, Any]:
    try:
        models, manifest = _bundle_entries(zpath)
    except FileNotFoundError as exc:
        return {
            "kind": "bundle",
            "source": str(zpath),
            "manifest": None,
            "failures": [{"code": "FAIL_BUNDLE_READ", "detail": f"{type(exc).__name__}: {exc}"}],
            "warnings": [],
            "models": [],
            "summary": {"n_models": 0, "n_failures": 1, "n_warnings": 0, "n_runtime_integer_denominator_models": 0},
            "verdict": "FAIL",
        }
    except zipfile.BadZipFile as exc:
        return {
            "kind": "bundle",
            "source": str(zpath),
            "manifest": None,
            "failures": [{"code": "FAIL_BUNDLE_ZIP_PARSE", "detail": f"{type(exc).__name__}: {exc}"}],
            "warnings": [],
            "models": [],
            "summary": {"n_models": 0, "n_failures": 1, "n_warnings": 0, "n_runtime_integer_denominator_models": 0},
            "verdict": "FAIL",
        }
    result: dict[str, Any] = {
        "kind": "bundle",
        "source": str(zpath),
        "manifest": manifest,
        "failures": [],
        "warnings": [],
        "models": [],
    }
    if manifest["missing"]:
        result["failures"].append({"code": "FAIL_BUNDLE_MISSING_TASKS", "tasks": manifest["missing"]})
    if manifest["extra"]:
        result["failures"].append({"code": "FAIL_BUNDLE_EXTRA_TASKS", "tasks": manifest["extra"]})
    if manifest["duplicates"]:
        result["failures"].append({"code": "FAIL_BUNDLE_DUPLICATE_TASKS", "tasks": manifest["duplicates"]})

    for blob in models:
        row = scan_model(
            blob,
            load=load,
            dynamic_official=dynamic_official,
            include_input_denominator=include_input_denominator,
        )
        result["models"].append(row)
        for failure in row["failures"]:
            result["failures"].append({"model": row["name"], **failure})
        for warning in row["warnings"]:
            result["warnings"].append({"model": row["name"], **warning})

    result["summary"] = {
        "n_models": len(models),
        "n_failures": len(result["failures"]),
        "n_warnings": len(result["warnings"]),
        "n_runtime_integer_denominator_models": sum(
            1 for row in result["models"] if row.get("runtime_integer_denominators")
        ),
    }
    result["verdict"] = "FAIL" if result["failures"] else ("WARN" if result["warnings"] else "PASS")
    return result


def scan_model_path(path: pathlib.Path, *, task: int | None = None, load: bool = True,
                    dynamic_official: bool = False, include_input_denominator: bool = True) -> dict[str, Any]:
    try:
        blob_bytes = path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        row = {
            "task": None if task is None else f"task{task:03d}",
            "name": path.name,
            "source": str(path),
            "sha256": None,
            "size_bytes": None,
            "failures": [{"code": "FAIL_MODEL_READ", "detail": f"{type(exc).__name__}: {exc}"}],
            "warnings": [],
            "op_inventory": {},
            "risk_op_flags": [],
            "runtime_integer_denominators": [],
            "runtime_denominators_unknown_dtype": [],
            "verdict": "FAIL",
        }
        return {
            "kind": "model",
            "source": str(path),
            "verdict": "FAIL",
            "summary": {"n_failures": 1, "n_warnings": 0, "runtime_integer_denominator_count": 0},
            "model": row,
        }
    blob = ModelBlob(
        task=task if task is not None else _parse_task_from_name(path.name),
        name=path.name,
        blob=blob_bytes,
        source=str(path),
    )
    row = scan_model(
        blob,
        load=load,
        dynamic_official=dynamic_official,
        include_input_denominator=include_input_denominator,
    )
    return {
        "kind": "model",
        "source": str(path),
        "verdict": row["verdict"],
        "summary": {
            "n_failures": len(row["failures"]),
            "n_warnings": len(row["warnings"]),
            "runtime_integer_denominator_count": len(row.get("runtime_integer_denominators", [])),
        },
        "model": row,
    }


def exit_code(result: dict[str, Any]) -> int:
    verdict = result.get("verdict")
    if verdict == "FAIL":
        return 2
    if verdict == "WARN":
        return 1
    return 0


def dumps(result: dict[str, Any], *, compact: bool = False) -> str:
    if compact:
        return json.dumps(result, separators=(",", ":"), default=str)
    return json.dumps(result, indent=2, default=str)
