"""Multi-strategy F16 batch converter for object/color tasks.

Tries each strategy and stages the best (largest negative delta with 100% verify).
Strategies:
  A. Whole-graph F16 (Cast input -> F16, all F32 -> F16, output F16)
  B. Cast-after-first-slice F16 (Cast only after first Slice(input))
  C. Cast-after-all-input-slices F16 (multi-slice variant)
  D. Drop final Cast(F16->F32) (output becomes F16)
"""
import argparse
import json
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tools._score_v2 import score_network_v2

SUBMISSION_ZIP = REPO / "submissions" / "submission.zip"
STAGED = REPO / ".hermes" / "board" / "staged_submits"


def hex8():
    return secrets.token_hex(4)


def strategy_whole_f16(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    new_inits = []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16), name=init.name))
        else:
            new_inits.append(init)
    input_f16_name = "_input_f16_castA"
    new_nodes = [helper.make_node("Cast", inputs=['input'], outputs=[input_f16_name],
                                  name="_cast_input", to=TensorProto.FLOAT16)]
    for n in g.node:
        nn = onnx.NodeProto(); nn.CopyFrom(n)
        for i, inp in enumerate(nn.input):
            if inp == 'input':
                nn.input[i] = input_f16_name
        if nn.op_type == 'Cast':
            for attr in nn.attribute:
                if attr.name == 'to' and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        new_nodes.append(nn)
    new_vi = []
    for vi in g.value_info:
        nvi = onnx.ValueInfoProto(); nvi.CopyFrom(vi)
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            nvi.type.tensor_type.elem_type = TensorProto.FLOAT16
        new_vi.append(nvi)
    inp_shape = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    new_vi.append(helper.make_tensor_value_info(input_f16_name, TensorProto.FLOAT16, inp_shape))
    out_shape = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    orig_out_dt = g.output[0].type.tensor_type.elem_type
    new_out_dt = TensorProto.FLOAT16 if orig_out_dt == TensorProto.FLOAT else orig_out_dt
    new_output = helper.make_tensor_value_info(g.output[0].name, new_out_dt, out_shape)
    new_graph = helper.make_graph(new_nodes, g.name, list(g.input), [new_output], new_inits, value_info=new_vi)
    return helper.make_model(new_graph, opset_imports=list(model.opset_import),
                             ir_version=model.ir_version, producer_name=model.producer_name,
                             producer_version=model.producer_version)


def strategy_cast_after_slices(model: onnx.ModelProto, multi: bool = False) -> onnx.ModelProto:
    g = model.graph
    input_slices = [n for n in g.node if n.op_type == 'Slice' and n.input and n.input[0] == 'input']
    if not input_slices:
        return None
    if not multi:
        input_slices = input_slices[:1]
    new_inits = []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16), name=init.name))
        else:
            new_inits.append(init)
    slice_renames = {s.output[0]: s.output[0] + "_f32_internal" for s in input_slices}
    new_nodes = []
    for n in g.node:
        nn = onnx.NodeProto(); nn.CopyFrom(n)
        if n in input_slices:
            orig = n.output[0]
            f32_name = slice_renames[orig]
            nn.output[0] = f32_name
            new_nodes.append(nn)
            new_nodes.append(helper.make_node("Cast", inputs=[f32_name], outputs=[orig],
                                              name=f"_cast_{orig}_f16", to=TensorProto.FLOAT16))
            continue
        if nn.op_type == 'Cast':
            for attr in nn.attribute:
                if attr.name == 'to' and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        new_nodes.append(nn)
    new_vi = []
    for vi in g.value_info:
        nvi = onnx.ValueInfoProto(); nvi.CopyFrom(vi)
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            nvi.type.tensor_type.elem_type = TensorProto.FLOAT16
        new_vi.append(nvi)
    for orig, f32_name in slice_renames.items():
        for vi in g.value_info:
            if vi.name == orig:
                shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                new_vi.append(helper.make_tensor_value_info(f32_name, TensorProto.FLOAT, shape))
                break
    out_shape = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    orig_out_dt = g.output[0].type.tensor_type.elem_type
    new_out_dt = TensorProto.FLOAT16 if orig_out_dt == TensorProto.FLOAT else orig_out_dt
    new_output = helper.make_tensor_value_info(g.output[0].name, new_out_dt, out_shape)
    new_graph = helper.make_graph(new_nodes, g.name, list(g.input), [new_output], new_inits, value_info=new_vi)
    return helper.make_model(new_graph, opset_imports=list(model.opset_import),
                             ir_version=model.ir_version, producer_name=model.producer_name,
                             producer_version=model.producer_version)


def strategy_drop_final_cast(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    out_name = g.output[0].name
    last_node = None
    for n in g.node:
        if out_name in n.output:
            last_node = n
            break
    if last_node is None:
        return None
    if last_node.op_type == 'Cast':
        for attr in last_node.attribute:
            if attr.name == 'to' and attr.i == TensorProto.FLOAT:
                prev = last_node.input[0]
                new_nodes = []
                for n in g.node:
                    if n is last_node:
                        continue
                    nn = onnx.NodeProto(); nn.CopyFrom(n)
                    for i, inp in enumerate(nn.input):
                        if inp == prev: nn.input[i] = out_name
                    for i, outp in enumerate(nn.output):
                        if outp == prev: nn.output[i] = out_name
                    new_nodes.append(nn)
                new_vi = [vi for vi in g.value_info if vi.name != prev]
                out_shape = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
                new_output = helper.make_tensor_value_info(out_name, TensorProto.FLOAT16, out_shape)
                new_graph = helper.make_graph(new_nodes, g.name, list(g.input), [new_output],
                                              list(g.initializer), value_info=new_vi)
                return helper.make_model(new_graph, opset_imports=list(model.opset_import),
                                         ir_version=model.ir_version, producer_name=model.producer_name,
                                         producer_version=model.producer_version)
    if last_node.op_type == 'Pad':
        pad_in = last_node.input[0]
        cast_node = None
        for n in g.node:
            if pad_in in n.output and n.op_type == 'Cast':
                for attr in n.attribute:
                    if attr.name == 'to' and attr.i == TensorProto.FLOAT:
                        cast_node = n
                        break
                if cast_node:
                    break
        if cast_node:
            prev = cast_node.input[0]
            cast_out = cast_node.output[0]
            new_nodes = []
            for n in g.node:
                if n is cast_node:
                    continue
                nn = onnx.NodeProto(); nn.CopyFrom(n)
                for i, inp in enumerate(nn.input):
                    if inp == cast_out: nn.input[i] = prev
                new_nodes.append(nn)
            new_vi = [vi for vi in g.value_info if vi.name != cast_out]
            out_shape = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
            new_output = helper.make_tensor_value_info(out_name, TensorProto.FLOAT16, out_shape)
            new_graph = helper.make_graph(new_nodes, g.name, list(g.input), [new_output],
                                          list(g.initializer), value_info=new_vi)
            return helper.make_model(new_graph, opset_imports=list(model.opset_import),
                                     ir_version=model.ir_version, producer_name=model.producer_name,
                                     producer_version=model.producer_version)
    return None


STRATEGIES = {
    "whole_f16": strategy_whole_f16,
    "cast_after_first_slice": lambda m: strategy_cast_after_slices(m, multi=False),
    "cast_after_all_slices": lambda m: strategy_cast_after_slices(m, multi=True),
    "drop_final_cast": strategy_drop_final_cast,
}


def run_verify(out_path: Path, tid: int):
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "verify_against_live.py"),
         "--proto", str(out_path), "--tid", str(tid),
         "--live-zip", str(SUBMISSION_ZIP)],
        capture_output=True, text=True, timeout=300
    )
    m = re.search(r"Prototype\s+vs\s+LIVE\s*:\s*(\d+)/(\d+)", r.stdout, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def process_task(tid: str, tag: str, tmpdir: Path) -> dict:
    result = {"task_id": tid, "best_delta": 0, "best_strategy": None}
    try:
        with zipfile.ZipFile(SUBMISSION_ZIP) as z:
            z.extract(f"{tid}.onnx", tmpdir)
        src = tmpdir / f"{tid}.onnx"
        m = onnx.load(str(src))
        r_orig = score_network_v2(str(src))
        if r_orig is None:
            result["status"] = "ORIG_SCORE_FAIL"
            return result
        orig_cost = r_orig[1] + r_orig[2]
        result["orig_cost"] = orig_cost
        tid_num = int(tid.replace("task", ""))

        best = None
        for sname, sfn in STRATEGIES.items():
            try:
                new_m = sfn(m)
                if new_m is None:
                    continue
                out = tmpdir / f"{tid}_{sname}.onnx"
                onnx.save(new_m, str(out))
                try:
                    onnx.checker.check_model(new_m)
                except Exception:
                    continue
                r_new = score_network_v2(str(out))
                if r_new is None:
                    continue
                delta = (r_new[1] + r_new[2]) - orig_cost
                if delta >= 0:
                    continue
                passed, total = run_verify(out, tid_num)
                if total == 0 or passed != total:
                    continue
                if best is None or delta < best[0]:
                    best = (delta, sname, str(out), passed, total, r_new[1]+r_new[2])
            except Exception:
                continue

        if best is None:
            result["status"] = "NO_WIN"
            return result
        delta, sname, out_path, passed, total, new_cost = best
        cand_id = f"{tid}_{sname}_t_{hex8()}_on_5835_30"
        dest = STAGED / cand_id
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, dest / f"{tid}.onnx")
        manifest = {
            "schema": "neurogolf.staged_submit.v1",
            "candidate_id": cand_id,
            "task_id": tid,
            "pattern": sname,
            "tag": tag,
            "scores": {"base_cost": orig_cost, "candidate_cost": new_cost, "cost_delta": delta},
            "tests_run": {"verify_against_live": f"{passed}/{total}"},
            "lineage_base_candidate_id": "canonical_5835_30",
            "staged_at": datetime.now(timezone.utc).isoformat(),
            "worker": "neurogolf-worker-OB_BIG",
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
        result["status"] = "STAGED"
        result["best_delta"] = delta
        result["best_strategy"] = sname
        result["candidate_id"] = cand_id
        return result
    except Exception as e:
        result["status"] = "EXCEPTION"
        result["error"] = str(e)[:200]
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks", help="Comma-separated task IDs")
    p.add_argument("--tag", default="OB_BIG_multi")
    args = p.parse_args()
    task_ids = [t.strip() for t in args.tasks.split(",")]
    tmpdir = Path(tempfile.mkdtemp())
    print(f"Temp: {tmpdir}")
    results = []
    try:
        for i, tid in enumerate(task_ids):
            print(f"\n[{i+1}/{len(task_ids)}] {tid}", flush=True)
            r = process_task(tid, args.tag, tmpdir)
            results.append(r)
            for k, v in r.items():
                if k != 'task_id':
                    print(f"  {k}: {v}", flush=True)

        print("\n=== SUMMARY ===")
        staged = [r for r in results if r["status"] == "STAGED"]
        total = sum(r["best_delta"] for r in staged)
        print(f"Staged: {len(staged)} / {len(task_ids)}")
        print(f"Total bytes saved: {total}")
        by_status = {}
        for r in results:
            by_status.setdefault(r["status"], []).append(r["task_id"])
        for s, ids in by_status.items():
            print(f"  {s}: {len(ids)}  ({', '.join(ids[:10])}{'...' if len(ids)>10 else ''})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
