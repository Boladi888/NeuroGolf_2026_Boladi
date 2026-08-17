"""Mechanical cost-diff between a task's LIVE ONNX and a CANDIDATE ONNX.

Names exactly which *charged* structures changed and why the saving is real, so
reviewers/workers stop inferring cost deltas from prose.

The NeuroGolf charged cost (May-6 scorer) is::

    cost = memory + params

  * params : element count of unique initializers (deduped by name).
  * memory : byte size of intermediate `value_info` tensors, excluding the
             tensors literally named "input"/"output". Initializers do NOT
             count toward memory.

`tools/cost_predictor.py` is the single source of that math. This tool calls
`cost_predictor.predict()` for the authoritative per-graph totals, then
attributes the delta to individual initializers / intermediate tensors with
`onnx` + numpy and *reconciles* its per-structure sums back to
cost_predictor's `params` and `memory` numbers (a RECONCILE line is printed;
a mismatch is flagged loudly).

Usage::

    python tools/ng_graph_cost_diff.py --task N --live LIVE.onnx --candidate CAND.onnx
    python tools/ng_graph_cost_diff.py --task N --candidate CAND.onnx        # live -> Team_Best_Onnx/task{N}.onnx
    python tools/ng_graph_cost_diff.py --task N ... --top 25                 # cap significant lines
    python tools/ng_graph_cost_diff.py --task N ... --json                   # machine-readable

Output (top-N by |byte delta|):
  * charged memory-tensor deltas (added / removed / dtype / shape changed)
  * initializer params deltas (added / removed / dtype / shape changed)
  * node / op-type count deltas
  * dtype changes, shape changes (for same-named structures)
  * duplicate-initializer groups (same bytes reused) on each side
  * removed / added scorer-visible graph outputs

Each significant line gets a best-effort ACTION label from::
  delete_dead_sentinel share_initializer dtype_narrow replace_final_decoder
  fold_gather fold_mask pad_axes avoid_materialized_crop table_remap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NG_ROOT / "tools"))

import onnx  # noqa: E402

# Reuse the cost model + byte table; do NOT re-derive them.
import cost_predictor  # noqa: E402
from cost_predictor import _BYTES_PER_ELEM as BPE  # noqa: E402

try:
    from neurogolf_paths import resolve_canonical_root  # noqa: E402
except Exception:  # noqa: BLE001
    resolve_canonical_root = None  # type: ignore

# ONNX TensorProto elem_type -> readable dtype name (diagnostic only).
_DTYPE_NAME = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64",
    12: "uint32", 13: "uint64", 16: "bfloat16",
}


def _dtype_name(elem_type):
    return _DTYPE_NAME.get(elem_type, f"dt{elem_type}")


def _bpe(elem_type):
    return BPE.get(elem_type, 4)


def _shape_dims(shape_proto):
    """Return list of ints, or None if any dim is dynamic / missing."""
    dims = []
    for dim in shape_proto.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            return None
    return dims


def _elem_count(dims):
    # Match cost_predictor/onnx-tool: empty-shape (scalar) tensor => 0 elems.
    return math.prod(dims) if dims else 0


def _init_bytes(init):
    """Raw byte payload of an initializer, preferring the actual tensor bytes.

    Falls back to elem_count * bytes-per-elem when raw_data is absent (e.g.
    initializers stored in the typed float_data/int64_data fields).
    """
    if init.raw_data:
        return len(init.raw_data)
    dims = list(init.dims)
    return _elem_count(dims) * _bpe(init.data_type)


def _init_content_hash(init):
    """Stable hash of an initializer's *value bytes* for duplicate detection."""
    h = hashlib.sha1()
    h.update(bytes(str(init.data_type), "ascii"))
    h.update(bytes(str(list(init.dims)), "ascii"))
    if init.raw_data:
        h.update(init.raw_data)
    else:
        # Cover typed-field storage; order-stable per field.
        for field in ("float_data", "int32_data", "int64_data",
                      "double_data", "uint64_data"):
            vals = getattr(init, field, None)
            if vals:
                h.update(bytes(field, "ascii"))
                h.update(bytes(str(list(vals)), "ascii"))
    return h.hexdigest()[:12]


def _collect_initializers(model):
    """name -> {dims, dtype, dtype_name, elems, bytes, hash}, deduped by name.

    `params` in the cost model dedupes by name (first occurrence wins), so we
    do the same to keep the reconcile honest.
    """
    out = {}
    for init in model.graph.initializer:
        if init.name in out:
            continue
        dims = list(init.dims)
        elems = _elem_count(dims)
        out[init.name] = {
            "name": init.name,
            "dims": dims,
            "dtype": init.data_type,
            "dtype_name": _dtype_name(init.data_type),
            "elems": elems,
            "bytes": _init_bytes(init),
            "hash": _init_content_hash(init),
        }
    return out


def _collect_memory_tensors(model):
    """name -> {dims, dtype, dtype_name, elems, bytes} for CHARGED memory tensors.

    Mirrors cost_predictor.predict()'s memory rule: shape-infer (strict), then
    sum intermediate value_info byte sizes, excluding names 'input'/'output'.
    Initializers are excluded from memory.
    """
    try:
        m = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception:  # noqa: BLE001
        m = model
    init_names = {init.name for init in m.graph.initializer}
    out = {}
    for vi in m.graph.value_info:
        if vi.name in ("input", "output"):
            continue
        if vi.name in init_names:
            continue
        d = _shape_dims(vi.type.tensor_type.shape)
        if d is None:
            continue
        elem_type = vi.type.tensor_type.elem_type
        elems = _elem_count(d)
        out[vi.name] = {
            "name": vi.name,
            "dims": d,
            "dtype": elem_type,
            "dtype_name": _dtype_name(elem_type),
            "elems": elems,
            "bytes": elems * _bpe(elem_type),
        }
    return out


def _op_counts(model):
    counts = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _graph_output_names(model):
    return [o.name for o in model.graph.output]


def _final_decoder_initializers(model):
    """Initializer names sitting in the *final* operand slot of any node that
    produces a scorer-visible graph output.

    That tail slot is the conventional position for a readout / decode matrix
    (e.g. the last Einsum operand, a final MatMul weight). Restricting to the
    last operand avoids tagging every table that merely touches the output
    node. Used only for the `replace_final_decoder` action hint.
    """
    out_names = {o.name for o in model.graph.output}
    init_names = {init.name for init in model.graph.initializer}
    feeders = set()
    for node in model.graph.node:
        if any(o in out_names for o in node.output) and node.input:
            last = node.input[-1]
            if last in init_names:
                feeders.add(last)
    return feeders


def _dup_groups(inits):
    """Group same-name-deduped initializers by identical value bytes.

    Returns list of {hash, names, bytes_each, elems_each, wasted_bytes} for any
    group of size >= 2 (i.e. an initializer whose exact bytes are materialised
    more than once -> a `share_initializer` opportunity).
    """
    by_hash = {}
    for meta in inits.values():
        by_hash.setdefault(meta["hash"], []).append(meta)
    groups = []
    for h, metas in by_hash.items():
        if len(metas) < 2:
            continue
        names = sorted(m["name"] for m in metas)
        groups.append({
            "hash": h,
            "names": names,
            "count": len(metas),
            "bytes_each": metas[0]["bytes"],
            "elems_each": metas[0]["elems"],
            "wasted_bytes": metas[0]["bytes"] * (len(metas) - 1),
        })
    groups.sort(key=lambda g: -g["wasted_bytes"])
    return groups


# --------------------------------------------------------------------------
# Action labelling (best-effort heuristic — name the closest).
# --------------------------------------------------------------------------
_ACTIONS = {
    "delete_dead_sentinel", "share_initializer", "dtype_narrow",
    "replace_final_decoder", "fold_gather", "fold_mask", "pad_axes",
    "avoid_materialized_crop", "table_remap",
}

# op_types that, when their count drops, hint at a specific structural action.
_OP_ACTION_ON_REMOVAL = {
    "Gather": "fold_gather",
    "GatherND": "fold_gather",
    "GatherElements": "fold_gather",
    "Where": "fold_mask",
    "Slice": "avoid_materialized_crop",
    "Pad": "pad_axes",
}
_OP_ACTION_ON_ADD = {
    "Pad": "pad_axes",
}


def _is_final_decoder(name, ctx):
    """True if `name` feeds the node that produces a scorer-visible output.

    Structural signal (initializer is an operand of an output-producing node)
    takes priority; a name-token fallback catches decoder-ish initializers in
    graphs where shape inference could not tie the operand to the output.
    """
    if name in ctx.get("output_feeders", ()):  # structural: feeds output node
        return True
    low = name.lower()
    return any(tok in low for tok in ("decod", "final", "render", "readout"))


def _label_initializer_change(kind, live_meta, cand_meta, ctx):
    """Return an action label for one initializer delta line.

    kind in {'removed','added','dtype','shape'}.
    ctx carries cross-graph hints (op deltas, dup groups, matching partners).
    """
    if kind == "dtype":
        lb, cb = live_meta["bytes"], cand_meta["bytes"]
        if cb < lb:
            return "dtype_narrow"
        return "dtype_narrow"  # a dtype change on a charged init is always the narrow lever

    if kind == "shape":
        # A shrunk shape on a table-like init reads as a remap/pad-axes shave.
        return "table_remap"

    if kind == "removed":
        name = live_meta["name"]
        # Removed init whose exact bytes still exist on the candidate side under
        # another name -> it was consolidated (shared) into that partner.
        if live_meta["hash"] in ctx["cand_hashes"]:
            return "share_initializer"
        # Removed init feeding the final decode/output node -> decoder swap.
        if _is_final_decoder(name, ctx):
            return "replace_final_decoder"
        # Removed init that has a same-shape/dtype added partner -> table remap.
        if ctx["added_by_shape_dtype"].get((tuple(live_meta["dims"]), live_meta["dtype"])):
            return "table_remap"
        # A removed Gather/Where/Slice op alongside -> that op's action.
        for op, act in _OP_ACTION_ON_REMOVAL.items():
            if ctx["op_delta"].get(op, 0) < 0:
                return act
        # Otherwise it is a dead structure that simply went away.
        return "delete_dead_sentinel"

    if kind == "added":
        name = cand_meta["name"]
        # Added init duplicating existing bytes -> should have been shared.
        if ctx["cand_dup_hashes"] and cand_meta["hash"] in ctx["cand_dup_hashes"]:
            return "share_initializer"
        # Added init feeding the final decode/output node -> decoder swap.
        if _is_final_decoder(name, ctx):
            return "replace_final_decoder"
        # Added init that replaces a removed same-shape one -> table remap.
        if ctx["removed_by_shape_dtype"].get((tuple(cand_meta["dims"]), cand_meta["dtype"])):
            return "table_remap"
        return "table_remap"

    return "table_remap"


def _label_op_delta(op, delta, ctx):
    if delta < 0 and op in _OP_ACTION_ON_REMOVAL:
        return _OP_ACTION_ON_REMOVAL[op]
    if delta > 0 and op in _OP_ACTION_ON_ADD:
        return _OP_ACTION_ON_ADD[op]
    if delta < 0:
        return "delete_dead_sentinel"
    return "table_remap"


# --------------------------------------------------------------------------
# Diff core
# --------------------------------------------------------------------------
def diff(live_path, cand_path, task=None, top=25):
    live_model = onnx.load(str(live_path))
    cand_model = onnx.load(str(cand_path))

    live_pred = cost_predictor.predict(live_path)
    cand_pred = cost_predictor.predict(cand_path)

    live_inits = _collect_initializers(live_model)
    cand_inits = _collect_initializers(cand_model)
    live_mem = _collect_memory_tensors(live_model)
    cand_mem = _collect_memory_tensors(cand_model)
    live_ops = _op_counts(live_model)
    cand_ops = _op_counts(cand_model)
    live_outs = _graph_output_names(live_model)
    cand_outs = _graph_output_names(cand_model)

    # Cross-graph context for action labelling.
    cand_hashes = {m["hash"] for m in cand_inits.values()}
    live_hashes = {m["hash"] for m in live_inits.values()}
    cand_dups = _dup_groups(cand_inits)
    live_dups = _dup_groups(live_inits)
    cand_dup_hashes = {g["hash"] for g in cand_dups}

    added_by_shape_dtype = {}
    for m in cand_inits.values():
        if m["name"] not in live_inits:
            added_by_shape_dtype.setdefault((tuple(m["dims"]), m["dtype"]), []).append(m["name"])
    removed_by_shape_dtype = {}
    for m in live_inits.values():
        if m["name"] not in cand_inits:
            removed_by_shape_dtype.setdefault((tuple(m["dims"]), m["dtype"]), []).append(m["name"])

    op_delta = {}
    for op in set(live_ops) | set(cand_ops):
        d = cand_ops.get(op, 0) - live_ops.get(op, 0)
        if d != 0:
            op_delta[op] = d

    # Initializers sitting in the final decode/readout slot on either side.
    output_feeders = (_final_decoder_initializers(live_model)
                      | _final_decoder_initializers(cand_model))

    ctx = {
        "cand_hashes": cand_hashes,
        "live_hashes": live_hashes,
        "cand_dup_hashes": cand_dup_hashes,
        "added_by_shape_dtype": added_by_shape_dtype,
        "removed_by_shape_dtype": removed_by_shape_dtype,
        "op_delta": op_delta,
        "cand_outputs": cand_outs,
        "output_feeders": output_feeders,
    }

    # ---- initializer (params) deltas ----
    init_lines = []
    for name in sorted(set(live_inits) | set(cand_inits)):
        lm = live_inits.get(name)
        cm = cand_inits.get(name)
        if lm and not cm:
            init_lines.append({
                "structure": "initializer", "kind": "removed", "name": name,
                "dims": lm["dims"], "dtype_name": lm["dtype_name"],
                "elems_delta": -lm["elems"], "bytes_delta": -lm["bytes"],
                "action": _label_initializer_change("removed", lm, None, ctx),
                "detail": f"[{lm['dtype_name']} {lm['dims']}] gone",
            })
        elif cm and not lm:
            init_lines.append({
                "structure": "initializer", "kind": "added", "name": name,
                "dims": cm["dims"], "dtype_name": cm["dtype_name"],
                "elems_delta": cm["elems"], "bytes_delta": cm["bytes"],
                "action": _label_initializer_change("added", None, cm, ctx),
                "detail": f"[{cm['dtype_name']} {cm['dims']}] new",
            })
        else:
            # same name, present both sides — report dtype / shape / value change.
            if lm["dtype"] != cm["dtype"]:
                init_lines.append({
                    "structure": "initializer", "kind": "dtype", "name": name,
                    "dims": cm["dims"], "dtype_name": cm["dtype_name"],
                    "elems_delta": cm["elems"] - lm["elems"],
                    "bytes_delta": cm["bytes"] - lm["bytes"],
                    "action": _label_initializer_change("dtype", lm, cm, ctx),
                    "detail": f"{lm['dtype_name']} -> {cm['dtype_name']} (dims {lm['dims']})",
                })
            elif lm["dims"] != cm["dims"]:
                init_lines.append({
                    "structure": "initializer", "kind": "shape", "name": name,
                    "dims": cm["dims"], "dtype_name": cm["dtype_name"],
                    "elems_delta": cm["elems"] - lm["elems"],
                    "bytes_delta": cm["bytes"] - lm["bytes"],
                    "action": _label_initializer_change("shape", lm, cm, ctx),
                    "detail": f"{lm['dims']} -> {cm['dims']} ({cm['dtype_name']})",
                })
            elif lm["hash"] != cm["hash"]:
                # values changed, cost identical — note it (0 byte delta) so a
                # reviewer knows the graph is not byte-identical.
                init_lines.append({
                    "structure": "initializer", "kind": "values", "name": name,
                    "dims": cm["dims"], "dtype_name": cm["dtype_name"],
                    "elems_delta": 0, "bytes_delta": 0,
                    "action": "table_remap",
                    "detail": f"same shape/dtype, values changed [{cm['dtype_name']} {cm['dims']}]",
                })

    # ---- memory-tensor deltas ----
    mem_lines = []
    for name in sorted(set(live_mem) | set(cand_mem)):
        lm = live_mem.get(name)
        cm = cand_mem.get(name)
        if lm and not cm:
            mem_lines.append({
                "structure": "memory_tensor", "kind": "removed", "name": name,
                "dims": lm["dims"], "dtype_name": lm["dtype_name"],
                "elems_delta": -lm["elems"], "bytes_delta": -lm["bytes"],
                "action": "avoid_materialized_crop",
                "detail": f"[{lm['dtype_name']} {lm['dims']}] no longer materialised",
            })
        elif cm and not lm:
            mem_lines.append({
                "structure": "memory_tensor", "kind": "added", "name": name,
                "dims": cm["dims"], "dtype_name": cm["dtype_name"],
                "elems_delta": cm["elems"], "bytes_delta": cm["bytes"],
                "action": "avoid_materialized_crop",
                "detail": f"[{cm['dtype_name']} {cm['dims']}] newly materialised",
            })
        else:
            if lm["dtype"] != cm["dtype"]:
                mem_lines.append({
                    "structure": "memory_tensor", "kind": "dtype", "name": name,
                    "dims": cm["dims"], "dtype_name": cm["dtype_name"],
                    "elems_delta": cm["elems"] - lm["elems"],
                    "bytes_delta": cm["bytes"] - lm["bytes"],
                    "action": "dtype_narrow",
                    "detail": f"{lm['dtype_name']} -> {cm['dtype_name']} (dims {cm['dims']})",
                })
            elif lm["dims"] != cm["dims"]:
                mem_lines.append({
                    "structure": "memory_tensor", "kind": "shape", "name": name,
                    "dims": cm["dims"], "dtype_name": cm["dtype_name"],
                    "elems_delta": cm["elems"] - lm["elems"],
                    "bytes_delta": cm["bytes"] - lm["bytes"],
                    "action": "avoid_materialized_crop",
                    "detail": f"{lm['dims']} -> {cm['dims']} ({cm['dtype_name']})",
                })

    # ---- node / op-type count deltas ----
    op_lines = []
    for op in sorted(op_delta):
        d = op_delta[op]
        op_lines.append({
            "structure": "op_count", "kind": "node", "name": op,
            "bytes_delta": 0, "count_delta": d,
            "action": _label_op_delta(op, d, ctx),
            "detail": f"{live_ops.get(op, 0)} -> {cand_ops.get(op, 0)} ({d:+d})",
        })

    # ---- scorer-visible outputs ----
    out_lines = []
    for name in live_outs:
        if name not in cand_outs:
            out_lines.append({"structure": "output", "kind": "removed", "name": name,
                              "action": "delete_dead_sentinel",
                              "detail": "scorer-visible output removed"})
    for name in cand_outs:
        if name not in live_outs:
            out_lines.append({"structure": "output", "kind": "added", "name": name,
                              "action": "replace_final_decoder",
                              "detail": "scorer-visible output added"})

    # ---- reconcile per-structure byte sums to cost_predictor totals ----
    # params delta (element counts).
    live_params_sum = sum(m["elems"] for m in live_inits.values())
    cand_params_sum = sum(m["elems"] for m in cand_inits.values())
    params_delta_attributed = cand_params_sum - live_params_sum
    params_delta_pred = cand_pred.get("params", 0) - live_pred.get("params", 0)

    # memory delta (bytes).
    live_mem_sum = sum(m["bytes"] for m in live_mem.values())
    cand_mem_sum = sum(m["bytes"] for m in cand_mem.values())
    mem_delta_attributed = cand_mem_sum - live_mem_sum
    mem_delta_pred = cand_pred.get("memory", 0) - live_pred.get("memory", 0)

    cost_delta_pred = cand_pred.get("cost", 0) - live_pred.get("cost", 0)

    reconcile = {
        "params_delta_attributed": params_delta_attributed,
        "params_delta_predictor": params_delta_pred,
        "params_reconciled": params_delta_attributed == params_delta_pred,
        "memory_delta_attributed_bytes": mem_delta_attributed,
        "memory_delta_predictor_bytes": mem_delta_pred,
        "memory_reconciled": mem_delta_attributed == mem_delta_pred,
        "cost_delta_predictor": cost_delta_pred,
    }

    return {
        "task": task,
        "live_path": str(live_path),
        "cand_path": str(cand_path),
        "live_cost": live_pred.get("cost"),
        "cand_cost": cand_pred.get("cost"),
        "live_memory": live_pred.get("memory"),
        "cand_memory": cand_pred.get("memory"),
        "live_params": live_pred.get("params"),
        "cand_params": cand_pred.get("params"),
        "live_error": live_pred.get("error"),
        "cand_error": cand_pred.get("error"),
        "cost_delta": cost_delta_pred,
        "init_lines": init_lines,
        "mem_lines": mem_lines,
        "op_lines": op_lines,
        "out_lines": out_lines,
        "live_dup_groups": live_dups,
        "cand_dup_groups": cand_dups,
        "reconcile": reconcile,
        "top": top,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _fmt_bytes(n):
    return f"{n:+,} B" if n else "0 B"


def render(result):
    lines = []
    R = result
    t = f"task{int(R['task']):03d}" if R.get("task") is not None else "?"
    lines.append(f"=== ng_graph_cost_diff  {t} ===")
    lines.append(f"live      : {R['live_path']}")
    lines.append(f"candidate : {R['cand_path']}")
    if R.get("live_error"):
        lines.append(f"!! LIVE cost_predictor error: {R['live_error']}")
    if R.get("cand_error"):
        lines.append(f"!! CANDIDATE cost_predictor error: {R['cand_error']}")
    lines.append("")
    lines.append(f"{'':10s} {'memory':>10s} {'params':>10s} {'cost':>10s}")
    lines.append(f"{'live':10s} {R['live_memory']!s:>10s} {R['live_params']!s:>10s} {R['live_cost']!s:>10s}")
    lines.append(f"{'candidate':10s} {R['cand_memory']!s:>10s} {R['cand_params']!s:>10s} {R['cand_cost']!s:>10s}")
    dmem = (R['cand_memory'] or 0) - (R['live_memory'] or 0)
    dpar = (R['cand_params'] or 0) - (R['live_params'] or 0)
    lines.append(f"{'delta':10s} {dmem:>+10d} {dpar:>+10d} {R['cost_delta']:>+10d}")
    lines.append("")

    # Merge all charged-structure lines, rank by |bytes_delta|.
    charged = []
    charged.extend(R["init_lines"])
    charged.extend(R["mem_lines"])
    charged.sort(key=lambda x: -abs(x.get("bytes_delta", 0)))
    shown = charged[: R["top"]]
    hidden = len(charged) - len(shown)

    lines.append(f"--- CHARGED STRUCTURE DELTAS (top {min(R['top'], len(charged))} by |byte delta|) ---")
    if not charged:
        lines.append("  (no charged initializer/memory-tensor differences)")
    else:
        lines.append(f"  {'delta':>12s}  {'kind':<8s} {'structure':<14s} {'name':<16s} {'action':<22s} detail")
        for x in shown:
            db = x.get("bytes_delta", 0)
            dstr = _fmt_bytes(db)
            extra = ""
            if "elems_delta" in x and x["elems_delta"]:
                extra = f"  ({x['elems_delta']:+d} params)"
            lines.append(
                f"  {dstr:>12s}  {x['kind']:<8s} {x['structure']:<14s} "
                f"{x['name'][:16]:<16s} {x['action']:<22s} {x['detail']}{extra}"
            )
        if hidden > 0:
            lines.append(f"  ... {hidden} more (raise --top to see)")
    lines.append("")

    # op/node deltas
    if R["op_lines"]:
        lines.append("--- NODE / OP-TYPE COUNT DELTAS ---")
        for x in R["op_lines"]:
            lines.append(f"  {x['name']:<16s} {x['detail']:<20s} action={x['action']}")
        lines.append("")

    # scorer-visible outputs
    if R["out_lines"]:
        lines.append("--- SCORER-VISIBLE OUTPUT CHANGES ---")
        for x in R["out_lines"]:
            lines.append(f"  {x['kind']:<8s} {x['name']:<16s} action={x['action']}  ({x['detail']})")
        lines.append("")

    # duplicate-initializer groups
    if R["cand_dup_groups"] or R["live_dup_groups"]:
        lines.append("--- DUPLICATE-INITIALIZER GROUPS (same bytes materialised >1x) ---")
        if R["live_dup_groups"]:
            lines.append("  live:")
            for g in R["live_dup_groups"]:
                lines.append(f"    {g['names']} x{g['count']} @ {g['bytes_each']}B each "
                             f"-> wasted {g['wasted_bytes']}B  action=share_initializer")
        if R["cand_dup_groups"]:
            lines.append("  candidate:")
            for g in R["cand_dup_groups"]:
                lines.append(f"    {g['names']} x{g['count']} @ {g['bytes_each']}B each "
                             f"-> wasted {g['wasted_bytes']}B  action=share_initializer")
        else:
            lines.append("  candidate: none")
        lines.append("")

    # reconcile
    rc = R["reconcile"]
    lines.append("--- RECONCILE (per-structure sums vs cost_predictor totals) ---")
    pmark = "OK" if rc["params_reconciled"] else "MISMATCH"
    mmark = "OK" if rc["memory_reconciled"] else "MISMATCH"
    lines.append(f"  params delta : attributed {rc['params_delta_attributed']:+d}  "
                 f"cost_predictor {rc['params_delta_predictor']:+d}  [{pmark}]")
    lines.append(f"  memory delta : attributed {rc['memory_delta_attributed_bytes']:+d} B  "
                 f"cost_predictor {rc['memory_delta_predictor_bytes']:+d} B  [{mmark}]")
    if not (rc["params_reconciled"] and rc["memory_reconciled"]):
        lines.append("  !! WARNING: per-structure attribution does not reconcile to the "
                     "cost model total; treat named deltas as incomplete.")
    lines.append("")

    # net cost delta + sign confirmation
    cd = R["cost_delta"]
    if cd < 0:
        verdict = f"NET COST DELTA {cd:+d}  -> candidate is CHEAPER (saving is real, sign negative)."
    elif cd > 0:
        verdict = f"NET COST DELTA {cd:+d}  -> candidate is MORE EXPENSIVE (regression, sign positive)."
    else:
        verdict = "NET COST DELTA 0  -> tie (no charged-cost change)."
    lines.append(verdict)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _resolve_live(task, explicit_live):
    if explicit_live:
        return pathlib.Path(explicit_live)
    if task is None:
        raise SystemExit("--live omitted and --task not given: cannot resolve live ONNX")
    root = NG_ROOT
    if resolve_canonical_root is not None:
        try:
            root = resolve_canonical_root()
        except Exception:  # noqa: BLE001
            root = NG_ROOT
    return pathlib.Path(root) / "Team_Best_Onnx" / f"task{int(task):03d}.onnx"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", type=int, default=None, help="Task number (used to resolve --live default).")
    ap.add_argument("--live", default=None,
                    help="Live ONNX path; defaults to Team_Best_Onnx/task{N}.onnx when omitted.")
    ap.add_argument("--candidate", required=True, help="Candidate ONNX path.")
    ap.add_argument("--top", type=int, default=25, help="Max charged-structure lines to print (default 25).")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of the human table.")
    args = ap.parse_args()

    live_path = _resolve_live(args.task, args.live)
    cand_path = pathlib.Path(args.candidate)

    if not live_path.exists():
        raise SystemExit(f"live ONNX not found: {live_path}")
    if not cand_path.exists():
        raise SystemExit(f"candidate ONNX not found: {cand_path}")

    result = diff(live_path, cand_path, task=args.task, top=args.top)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(render(result))

    # Exit non-zero if attribution fails to reconcile, so callers can gate on it.
    rc = result["reconcile"]
    if not (rc["params_reconciled"] and rc["memory_reconciled"]):
        sys.exit(3)


if __name__ == "__main__":
    main()
