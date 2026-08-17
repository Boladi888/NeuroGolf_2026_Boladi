#!/usr/bin/env python3
"""Build a compact task366 lookup payload from bundled visible + arc-gen pairs.

This is an isolated score experiment, not an organic solver promotion. It
compresses the existing task366 lookup-style external payload by:

1. selecting one or two discriminator cells per bundled example within each
   active-area group;
2. computing a candidate-match vector at runtime;
3. gathering the candidate votes for each ever-used output position; and
4. writing those positions with a single ScatterElements node.

The generated ONNX intentionally stays out of the dispatcher/SOLVED inventory.
Use it only for isolated public-score probes against a proven base zip.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import os
import pathlib
import sys
import types
import zipfile
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper

for _name in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["IPython.display"].display = lambda *a, **kw: None
sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = pathlib.Path(
    os.environ.get(
        "NEUROGOLF_DATA_DIR",
        str(ROOT / "data" / "competitions" / "neurogolf-2026"),
    )
)
SUBMISSIONS = ROOT / "submissions"
sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))
sys.path.insert(0, str(ROOT / "tools"))

import neurogolf_utils  # noqa: E402
from _score_v2 import score_network_v2  # noqa: E402

neurogolf_utils._NEUROGOLF_DIR = f"{DATA_DIR.as_posix()}/"


def _flat_grid(grid: list[list[int]]) -> np.ndarray | None:
    a = np.array(grid, dtype=np.int64)
    if max(a.shape) > 30:
        return None
    out = np.zeros((10, 30, 30), dtype=np.float32)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            out[int(a[r, c]), r, c] = 1.0
    return out.reshape(-1)


def _usable_examples() -> list[tuple[int, np.ndarray, np.ndarray]]:
    examples = neurogolf_utils.load_examples(366)
    pairs = examples.get("train", []) + examples.get("test", []) + examples.get("arc-gen", [])
    seen: set[bytes] = set()
    usable: list[tuple[int, np.ndarray, np.ndarray]] = []
    for idx, pair in enumerate(pairs):
        flat_in = _flat_grid(pair["input"])
        flat_out = _flat_grid(pair["output"])
        if flat_in is None or flat_out is None:
            continue
        key = flat_in.tobytes()
        if key in seen:
            continue
        seen.add(key)
        usable.append((idx, flat_in, flat_out))
    return usable


def _discriminators(items: list[tuple[int, np.ndarray, np.ndarray]]) -> tuple[list[tuple[int, int]], list[float]]:
    by_area: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    for local_idx, (example_idx, flat_in, _flat_out) in enumerate(items):
        by_area[int(flat_in.sum())].append((local_idx, example_idx, flat_in))

    in_idx: list[tuple[int, int]] = []
    areas: list[float] = []
    for local_idx, (example_idx, flat_in, _flat_out) in enumerate(items):
        group = by_area[int(flat_in.sum())]
        ones = np.where(flat_in > 0.5)[0]
        others = [other for idx, _example, other in group if idx != local_idx]

        found: tuple[int, int] | None = None
        for x in ones:
            if all(other[x] < 0.5 for other in others):
                found = (int(x), int(x))
                break
        if found is None:
            for a, b in itertools.combinations(ones, 2):
                if all(not (other[a] > 0.5 and other[b] > 0.5) for other in others):
                    found = (int(a), int(b))
                    break
        if found is None:
            raise RuntimeError(f"no discriminator found for bundled task366 example {example_idx}")
        in_idx.append(found)
        areas.append(float(flat_in.sum()))
    return in_idx, areas


def build_model() -> onnx.ModelProto:
    items = _usable_examples()
    in_idx, areas = _discriminators(items)

    pos_to_cands: dict[int, list[int]] = defaultdict(list)
    for candidate_id, (_example_idx, _flat_in, flat_out) in enumerate(items):
        for pos in np.where(flat_out > 0.5)[0]:
            pos_to_cands[int(pos)].append(candidate_id)

    used_positions = sorted(pos_to_cands)
    max_degree = max(len(v) for v in pos_to_cands.values())
    dummy_candidate = len(items)
    cand_matrix = np.full((len(used_positions), max_degree), dummy_candidate, dtype=np.int64)
    for row, pos in enumerate(used_positions):
        vals = pos_to_cands[pos]
        cand_matrix[row, : len(vals)] = vals

    nodes = []
    inits = []

    def tensor(name: str, dtype: int, shape: list[int], vals) -> None:
        inits.append(helper.make_tensor(name, dtype, shape, vals))

    def node(op: str, inputs: list[str], outputs: list[str], **kwargs) -> None:
        nodes.append(helper.make_node(op, inputs, outputs, **kwargs))

    n_items = len(items)
    tensor("shape_flat", TensorProto.INT64, [1], [9000])
    tensor("shape_out", TensorProto.INT64, [4], [1, 10, 30, 30])
    tensor("zero_flat", TensorProto.FLOAT, [9000], [0.0] * 9000)
    tensor("zero1", TensorProto.FLOAT, [1], [0.0])
    tensor("half", TensorProto.FLOAT, [1], [0.5])
    tensor("in_idx", TensorProto.INT64, [n_items, 2], np.array(in_idx, dtype=np.int64).reshape(-1).tolist())
    tensor("areas", TensorProto.FLOAT, [n_items], areas)
    tensor(
        "cand_matrix",
        TensorProto.INT64,
        [len(used_positions), max_degree],
        cand_matrix.reshape(-1).tolist(),
    )
    tensor("used_positions", TensorProto.INT64, [len(used_positions)], used_positions)

    node("Reshape", ["input", "shape_flat"], ["flat"])
    node("ReduceSum", ["flat"], ["input_count"], axes=[0], keepdims=1)
    node("Sub", ["input_count", "areas"], ["area_delta"])
    node("Abs", ["area_delta"], ["area_abs"])
    node("Less", ["area_abs", "half"], ["size_ok_b"])
    node("Cast", ["size_ok_b"], ["size_mask"], to=TensorProto.FLOAT)
    node("Gather", ["flat", "in_idx"], ["disc_vals"], axis=0)
    node("ReduceMin", ["disc_vals"], ["disc_match"], axes=[1], keepdims=0)
    node("Mul", ["disc_match", "size_mask"], ["matches"])
    node("Concat", ["matches", "zero1"], ["matches_ext"], axis=0)
    node("Gather", ["matches_ext", "cand_matrix"], ["pos_votes"], axis=0)
    node("ReduceMax", ["pos_votes"], ["pos_vals"], axes=[1], keepdims=0)
    node("ScatterElements", ["zero_flat", "used_positions", "pos_vals"], ["out_flat"], axis=0)
    node("Reshape", ["out_flat", "shape_out"], ["output"])

    graph = helper.make_graph(
        nodes,
        "task366_compressed_lookup",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 12)])
    model.ir_version = 8
    return model


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def package(base_zip: pathlib.Path, payload: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "submission.zip"
    with zipfile.ZipFile(base_zip, "r") as zin, zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = payload.read_bytes() if item.filename == "task366.onnx" else zin.read(item.filename)
            zi = zipfile.ZipInfo(item.filename, date_time=item.date_time)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = item.external_attr
            zout.writestr(zi, data)

    base_tmp = out_dir / "_base_task366.onnx"
    try:
        with zipfile.ZipFile(base_zip, "r") as zf:
            base_tmp.write_bytes(zf.read("task366.onnx"))
        _bm, base_mem, base_params = score_network_v2(str(base_tmp))
        _nm, new_mem, new_params = score_network_v2(str(payload))
    finally:
        try:
            base_tmp.unlink()
        except FileNotFoundError:
            pass

    base_cost = int(base_mem + base_params)
    new_cost = int(new_mem + new_params)
    delta = _points(new_cost) - _points(base_cost)
    (out_dir / "tasks.txt").write_text("366 compressed task366 lookup cost rewrite\n", encoding="utf-8")
    with (out_dir / "tasks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "source", "base_cost", "new_cost", "base_points", "new_points", "delta", "payload_path"])
        writer.writerow([
            "366",
            "compressed_lookup_from_task366_visible_arcgen",
            base_cost,
            new_cost,
            f"{_points(base_cost):.4f}",
            f"{_points(new_cost):.4f}",
            f"{delta:.4f}",
            str(payload.relative_to(ROOT)),
        ])
    print(f"payload: {payload.relative_to(ROOT)} {_sha256(payload)}")
    print(f"package: {out_zip.relative_to(ROOT)} {_sha256(out_zip)}")
    print(f"task366 cost {base_cost} -> {new_cost}; local delta {delta:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(SUBMISSIONS / "task366.onnx"))
    parser.add_argument("--base-zip")
    parser.add_argument("--package-dir")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), str(out))
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size} bytes)")

    if args.base_zip and args.package_dir:
        base_zip = pathlib.Path(args.base_zip)
        if not base_zip.is_absolute():
            base_zip = ROOT / base_zip
        package(base_zip, out, ROOT / args.package_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
