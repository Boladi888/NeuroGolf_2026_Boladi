#!/usr/bin/env python3
"""Exact CP-SAT rank search for task312's selector bank.

The effective bank is BANK[k,c,d] = 1 when c != 5 and d == c, or when
c == 5 and d == k.  Current shipped/staged task312 stores a rank-18 signed
factorization of this bank as U/V/W initializers.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from ortools.sat.python import cp_model


B32_INT_EXACT = 2**24


def bank_array() -> np.ndarray:
    bank = np.zeros((10, 10, 10), dtype=np.int64)
    for k in range(10):
        for c in range(10):
            if c != 5:
                bank[k, c, c] = 1
            else:
                bank[k, 5, k] = 1
    return bank


def e0_array() -> np.ndarray:
    e0 = np.zeros(30, dtype=np.float32)
    e0[0] = 1.0
    return e0


def load_model_from_locator(locator: str) -> onnx.ModelProto:
    if "::" not in locator:
        return onnx.load(locator)
    zip_path, member = locator.split("::", 1)
    with zipfile.ZipFile(zip_path, "r") as zf:
        data = zf.read(member)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return onnx.load(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_rank_factors(model: onnx.ModelProto) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim == 2 and arr.shape[0] == 10:
            arrays.append((init.name, arr.astype(np.int64)))
    ranks = sorted({arr.shape[1] for _, arr in arrays})
    for rank in ranks:
        same_rank = [(name, arr) for name, arr in arrays if arr.shape == (10, rank)]
        if len(same_rank) == 3:
            same_rank.sort(key=lambda item: item[0])
            return same_rank[0][1], same_rank[1][1], same_rank[2][1]
    raise ValueError("could not find exactly three [10,R] selector-bank factors")


def exact_rank(rows: list[list[int]]) -> int:
    mat = [[Fraction(x) for x in row] for row in rows]
    if not mat:
        return 0
    n_rows = len(mat)
    n_cols = len(mat[0])
    rank = 0
    for col in range(n_cols):
        pivot = None
        for row in range(rank, n_rows):
            if mat[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        pv = mat[rank][col]
        mat[rank] = [x / pv for x in mat[rank]]
        for row in range(n_rows):
            if row == rank or mat[row][col] == 0:
                continue
            factor = mat[row][col]
            mat[row] = [x - factor * y for x, y in zip(mat[row], mat[rank])]
        rank += 1
        if rank == n_rows:
            break
    return rank


def flatten_mode(t: np.ndarray, mode: int) -> list[list[int]]:
    k_dim, c_dim, d_dim = t.shape
    if mode == 0:
        return [[int(t[k, c, d]) for c in range(c_dim) for d in range(d_dim)] for k in range(k_dim)]
    if mode == 1:
        return [[int(t[k, c, d]) for k in range(k_dim) for d in range(d_dim)] for c in range(c_dim)]
    if mode == 2:
        return [[int(t[k, c, d]) for k in range(k_dim) for c in range(c_dim)] for d in range(d_dim)]
    raise ValueError(mode)


def flattening_ranks(t: np.ndarray) -> list[int]:
    return [exact_rank(flatten_mode(t, mode)) for mode in range(3)]


def build_model_from_factors(a: np.ndarray, b: np.ndarray, c: np.ndarray, out_path: Path) -> None:
    rank = int(a.shape[1])
    inits = [
        helper.make_tensor("e0", TensorProto.FLOAT, [30], e0_array()),
        helper.make_tensor("u", TensorProto.FLOAT, [10, rank], a.astype(np.float32).ravel()),
        helper.make_tensor("vw", TensorProto.FLOAT, [10, rank], b.astype(np.float32).ravel()),
        helper.make_tensor("w", TensorProto.FLOAT, [10, rank], c.astype(np.float32).ravel()),
    ]
    node = helper.make_node(
        "Einsum",
        ["e0", "input", "u", "vw", "w", "input"],
        ["output"],
        equation="v,bkhv,kt,ct,dt,bchw->bdhw",
    )
    graph = helper.make_graph(
        [node],
        f"t312_colormap_einsum6cp_rank{rank}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(out_path))


def _status_name(solver: cp_model.CpSolver, status: int) -> str:
    try:
        return solver.status_name(status)
    except AttributeError:
        return solver.StatusName(status)


def solve_cp_sat(
    t: np.ndarray,
    rank: int,
    alphabet: list[int],
    time_limit_s: float,
    workers: int,
    log_progress: bool,
    dictionary: bool = False,
) -> dict:
    k_dim, c_dim, d_dim = t.shape
    values = sorted(set(alphabet))
    if 0 not in values:
        raise ValueError("alphabet must contain 0")
    max_abs = max(abs(v) for v in values)
    force_first_one = max_abs == 1

    model = cp_model.CpModel()
    dom_s = cp_model.Domain.from_values(values)

    def uniq_vectors(vectors: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        seen = set()
        out = []
        for vec in vectors:
            if all(x in values for x in vec) and vec not in seen:
                seen.add(vec)
                out.append(vec)
        return out

    def positive_dictionary() -> list[tuple[int, ...]]:
        zero = (0,) * 10
        basis = [tuple(1 if i == j else 0 for i in range(10)) for j in range(10)]
        ones = (1,) * 10
        not5 = tuple(0 if i == 5 else 1 for i in range(10))
        with5 = [
            tuple(1 if i in (5, j) else 0 for i in range(10))
            for j in range(10)
            if j != 5
        ]
        cyclic_pairs = [
            tuple(1 if i in (j, (j + 1) % 10) else 0 for i in range(10))
            for j in range(10)
        ]
        cyclic_triples = [
            tuple(1 if i in (j, (j + 1) % 10, (j + 2) % 10) else 0 for i in range(10))
            for j in range(10)
        ]
        return uniq_vectors([zero] + basis + [ones, not5] + with5 + cyclic_pairs + cyclic_triples)

    def signed_dictionary() -> list[tuple[int, ...]]:
        base = positive_dictionary()
        vectors = list(base)
        vectors.extend(tuple(-x for x in vec) for vec in base if any(vec))
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                vectors.append(tuple((1 if u == i else 0) - (1 if u == j else 0) for u in range(10)))
        return uniq_vectors(vectors)

    def vars_from_dictionary(name: str, rows: int, dict_vectors: list[tuple[int, ...]]):
        x = [[model.new_int_var_from_domain(dom_s, f"{name}_{u}_{r}") for r in range(rank)] for u in range(rows)]
        picks = [[model.new_bool_var(f"{name}_pick_{r}_{m}") for m in range(len(dict_vectors))] for r in range(rank)]
        for r in range(rank):
            model.add_exactly_one(picks[r])
            for u in range(rows):
                model.add(x[u][r] == sum(dict_vectors[m][u] * picks[r][m] for m in range(len(dict_vectors))))
        return x, picks

    if dictionary:
        dict_a = positive_dictionary()
        dict_b = positive_dictionary()
        dict_c = signed_dictionary()
        a, a_pick = vars_from_dictionary("A", k_dim, dict_a)
        b, b_pick = vars_from_dictionary("B", c_dim, dict_b)
        c, c_pick = vars_from_dictionary("C", d_dim, dict_c)
    else:
        dict_a = dict_b = dict_c = []
        a_pick = b_pick = c_pick = []
        a = [[model.new_int_var_from_domain(dom_s, f"A_{k}_{r}") for r in range(rank)] for k in range(k_dim)]
        b = [[model.new_int_var_from_domain(dom_s, f"B_{c}_{r}") for r in range(rank)] for c in range(c_dim)]
        c = [[model.new_int_var_from_domain(dom_s, f"C_{d}_{r}") for r in range(rank)] for d in range(d_dim)]

    def channel_nz_and_pos(x: list[list[cp_model.IntVar]], prefix: str):
        rows = len(x)
        cols = len(x[0])
        nz = [[model.new_bool_var(f"{prefix}_nz_{u}_{r}") for r in range(cols)] for u in range(rows)]
        pos = [[model.new_bool_var(f"{prefix}_pos_{u}_{r}") for r in range(cols)] for u in range(rows)]
        tuples = [(s, int(s != 0), int(s > 0)) for s in values]
        for u in range(rows):
            for r in range(cols):
                model.add_allowed_assignments([x[u][r], nz[u][r], pos[u][r]], tuples)
        return nz, pos

    a_nz, _ = channel_nz_and_pos(a, "A")
    b_nz, _ = channel_nz_and_pos(b, "B")
    c_nz, _ = channel_nz_and_pos(c, "C")

    active = [model.new_bool_var(f"active_{r}") for r in range(rank)]

    def has_nonzero(nz: list[list[cp_model.IntVar]], prefix: str):
        rows = len(nz)
        cols = len(nz[0])
        has = [model.new_bool_var(f"{prefix}_has_{r}") for r in range(cols)]
        for r in range(cols):
            total = sum(nz[u][r] for u in range(rows))
            model.add(total >= 1).only_enforce_if(has[r])
            model.add(total == 0).only_enforce_if(has[r].Not())
        return has

    a_has = has_nonzero(a_nz, "A")
    b_has = has_nonzero(b_nz, "B")
    c_has = has_nonzero(c_nz, "C")
    for r in range(rank):
        model.add(a_has[r] == active[r])
        model.add(b_has[r] == active[r])
        model.add(c_has[r] == active[r])
        if dictionary:
            model.add(a_pick[r][0] == active[r].Not())
            model.add(b_pick[r][0] == active[r].Not())
            model.add(c_pick[r][0] == active[r].Not())

    def normalize_first_nonzero_positive(x: list[list[cp_model.IntVar]], prefix: str) -> None:
        rows = len(x)
        cols = len(x[0])
        for r in range(cols):
            first = [model.new_bool_var(f"{prefix}_first_{u}_{r}") for u in range(rows)]
            model.add(sum(first) == active[r])
            for u in range(rows):
                if force_first_one:
                    model.add(x[u][r] == 1).only_enforce_if(first[u])
                else:
                    model.add(x[u][r] >= 1).only_enforce_if(first[u])
                for v in range(u):
                    model.add(x[v][r] == 0).only_enforce_if(first[u])

    normalize_first_nonzero_positive(a, "A")
    normalize_first_nonzero_positive(b, "B")

    support_len = k_dim + c_dim + d_dim
    if support_len > 62:
        raise ValueError("support key must be chunked for this tensor")
    weights = [1 << (support_len - 1 - p) for p in range(support_len)]
    support_code = [
        model.new_int_var(0, (1 << support_len) - 1, f"support_code_{r}")
        for r in range(rank)
    ]
    for r in range(rank):
        bits = [a_nz[k][r] for k in range(k_dim)] + [b_nz[i][r] for i in range(c_dim)] + [c_nz[j][r] for j in range(d_dim)]
        model.add(support_code[r] == sum(weights[p] * bits[p] for p in range(support_len)))
    for r in range(rank - 1):
        model.add(support_code[r] >= support_code[r + 1])
        model.add(active[r] >= active[r + 1])

    pair_values = sorted({x * y for x in values for y in values})
    term_values = sorted({xy * z for xy in pair_values for z in values})
    dom_pair = cp_model.Domain.from_values(pair_values)
    dom_term = cp_model.Domain.from_values(term_values)
    pair_tuples = [(x, y, x * y) for x in values for y in values]
    term_tuples = [(xy, z, xy * z) for xy in pair_values for z in values]

    ab = [
        [[model.new_int_var_from_domain(dom_pair, f"AB_{k}_{i}_{r}") for r in range(rank)] for i in range(c_dim)]
        for k in range(k_dim)
    ]
    w = [
        [
            [[model.new_int_var_from_domain(dom_term, f"W_{k}_{i}_{j}_{r}") for r in range(rank)] for j in range(d_dim)]
            for i in range(c_dim)
        ]
        for k in range(k_dim)
    ]

    for k in range(k_dim):
        for i in range(c_dim):
            for r in range(rank):
                model.add_allowed_assignments([a[k][r], b[i][r], ab[k][i][r]], pair_tuples)

    for k in range(k_dim):
        for i in range(c_dim):
            for j in range(d_dim):
                for r in range(rank):
                    model.add_allowed_assignments([ab[k][i][r], c[j][r], w[k][i][j][r]], term_tuples)

    for k in range(k_dim):
        for i in range(c_dim):
            for j in range(d_dim):
                prev = None
                for r in range(rank):
                    p = model.new_int_var(-B32_INT_EXACT, B32_INT_EXACT, f"P_{k}_{i}_{j}_{r}")
                    if prev is None:
                        model.add(p == w[k][i][j][r])
                    else:
                        model.add(p == prev + w[k][i][j][r])
                    prev = p
                model.add(prev == int(t[k, i, j]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.linearization_level = 2
    solver.parameters.log_search_progress = log_progress

    start = time.time()
    status = solver.solve(model)
    elapsed = time.time() - start
    status_name = _status_name(solver, status)

    result = {
        "rank": rank,
        "alphabet": values,
        "dictionary": dictionary,
        "status": status_name,
        "elapsed_s": elapsed,
        "time_limit_s": time_limit_s,
        "workers": workers,
        "active_terms_bound": rank,
        "model": {
            "factor_vars": 3 * 10 * rank,
            "pair_vars": 10 * 10 * rank,
            "term_vars": 10 * 10 * 10 * rank,
            "prefix_vars": 10 * 10 * 10 * rank,
            "prefix_guard": B32_INT_EXACT,
            "zero_term_padding": True,
            "dictionary_sizes": [len(dict_a), len(dict_b), len(dict_c)] if dictionary else None,
        },
        "solver_response_stats": solver.response_stats(),
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        a_sol = np.array([[solver.value(a[k][r]) for r in range(rank)] for k in range(k_dim)], dtype=np.int64)
        b_sol = np.array([[solver.value(b[i][r]) for r in range(rank)] for i in range(c_dim)], dtype=np.int64)
        c_sol = np.array([[solver.value(c[j][r]) for r in range(rank)] for j in range(d_dim)], dtype=np.int64)
        rec = np.einsum("kr,cr,dr->kcd", a_sol, b_sol, c_sol)
        if not np.array_equal(rec, t):
            raise AssertionError("solver witness failed exact integer re-check")
        active_count = int(np.count_nonzero(np.any(a_sol != 0, axis=0) & np.any(b_sol != 0, axis=0) & np.any(c_sol != 0, axis=0)))
        result["active_terms_found"] = active_count
        result["A"] = a_sol.tolist()
        result["B"] = b_sol.tolist()
        result["C"] = c_sol.tolist()

    return result


def parse_alphabet(text: str) -> list[int]:
    aliases = {
        "01": [0, 1],
        "012": [0, 1, 2],
        "signed01": [-1, 0, 1],
        "signed012": [-2, -1, 0, 1, 2],
    }
    if text in aliases:
        return aliases[text]
    return [int(part) for part in text.split(",") if part.strip()]


def command_inspect(args: argparse.Namespace) -> None:
    bank = bank_array()
    payload = {
        "shape": list(bank.shape),
        "nnz": int(np.count_nonzero(bank)),
        "sum": int(bank.sum()),
        "flattening_ranks": flattening_ranks(bank),
    }
    if args.onnx:
        model = load_model_from_locator(args.onnx)
        a, b, c = extract_rank_factors(model)
        rec = np.einsum("kr,cr,dr->kcd", a, b, c)
        payload["onnx_factor_shapes"] = [list(a.shape), list(b.shape), list(c.shape)]
        payload["onnx_factor_elements"] = int(a.size + b.size + c.size)
        payload["onnx_reconstructs_bank"] = bool(np.array_equal(rec, bank))
        payload["onnx_factor_alphabet"] = sorted({int(x) for x in np.concatenate([a.ravel(), b.ravel(), c.ravel()])})
    print(json.dumps(payload, indent=2, sort_keys=True))


def command_solve(args: argparse.Namespace) -> None:
    bank = bank_array()
    result = solve_cp_sat(
        bank,
        rank=args.rank,
        alphabet=parse_alphabet(args.alphabet),
        time_limit_s=args.time_limit,
        workers=args.workers,
        log_progress=args.log_progress,
        dictionary=args.dictionary,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.onnx_out and "A" in result:
        build_model_from_factors(
            np.array(result["A"], dtype=np.int64),
            np.array(result["B"], dtype=np.int64),
            np.array(result["C"], dtype=np.int64),
            Path(args.onnx_out),
        )
    summary = {k: result[k] for k in ("rank", "alphabet", "dictionary", "status", "elapsed_s", "time_limit_s")}
    if "active_terms_found" in result:
        summary["active_terms_found"] = result["active_terms_found"]
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect_p = sub.add_parser("inspect")
    inspect_p.add_argument("--onnx", help="ONNX path or zip::member locator")
    inspect_p.set_defaults(func=command_inspect)

    solve_p = sub.add_parser("solve")
    solve_p.add_argument("--rank", type=int, required=True)
    solve_p.add_argument("--alphabet", required=True, help="01, 012, signed01, signed012, or comma list")
    solve_p.add_argument("--time-limit", type=float, default=600)
    solve_p.add_argument("--workers", type=int, default=max(1, min(8, (math.floor((__import__('os').cpu_count() or 8)))) ))
    solve_p.add_argument("--log-progress", action="store_true")
    solve_p.add_argument("--dictionary", action="store_true")
    solve_p.add_argument("--out")
    solve_p.add_argument("--onnx-out")
    solve_p.set_defaults(func=command_solve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
