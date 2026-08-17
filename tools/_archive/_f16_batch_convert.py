"""Batch convert OB_HUGE tasks to F16-throughout with F16 output. Stage winners.

Usage:
  python tools/_f16_batch_convert.py task137,task195,...
  python tools/_f16_batch_convert.py --all-ob-huge
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


def f16_convert(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert all F32 intermediate computations to F16."""
    g = model.graph

    new_inits = []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16), name=init.name))
        else:
            new_inits.append(init)

    input_f16_name = "_input_f16_cast"
    new_nodes = [helper.make_node(
        "Cast", inputs=['input'], outputs=[input_f16_name],
        name="_cast_input", to=TensorProto.FLOAT16
    )]

    for n in g.node:
        nn = onnx.NodeProto()
        nn.CopyFrom(n)
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
        nvi = onnx.ValueInfoProto()
        nvi.CopyFrom(vi)
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            nvi.type.tensor_type.elem_type = TensorProto.FLOAT16
        new_vi.append(nvi)

    inp_shape = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    new_vi.append(helper.make_tensor_value_info(input_f16_name, TensorProto.FLOAT16, inp_shape))

    out_shape = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    orig_out_dtype = g.output[0].type.tensor_type.elem_type
    # Only convert output to F16 if it was F32; preserve other dtypes (bool, F16, etc.)
    new_out_dtype = TensorProto.FLOAT16 if orig_out_dtype == TensorProto.FLOAT else orig_out_dtype
    new_output = helper.make_tensor_value_info(g.output[0].name, new_out_dtype, out_shape)

    new_graph = helper.make_graph(
        new_nodes, g.name, list(g.input), [new_output], new_inits, value_info=new_vi
    )
    return helper.make_model(
        new_graph, opset_imports=list(model.opset_import),
        ir_version=model.ir_version, producer_name=model.producer_name,
        producer_version=model.producer_version,
    )


def run_verify(proto_path: Path, tid: int):
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "verify_against_live.py"),
         "--proto", str(proto_path), "--tid", str(tid),
         "--live-zip", str(SUBMISSION_ZIP)],
        capture_output=True, text=True, timeout=300
    )
    m = re.search(r"Prototype\s+vs\s+LIVE\s*:\s*(\d+)/(\d+)", r.stdout, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), r.stdout
    return 0, 0, r.stdout + r.stderr


def process_task(tid: str, tag: str, tmpdir: Path) -> dict:
    result = {"task_id": tid, "status": "UNKNOWN"}
    try:
        with zipfile.ZipFile(SUBMISSION_ZIP) as z:
            z.extract(f"{tid}.onnx", tmpdir)
        src = tmpdir / f"{tid}.onnx"
        m = onnx.load(str(src))
        new_m = f16_convert(m)
        out = tmpdir / f"{tid}_f16drop.onnx"
        onnx.save(new_m, str(out))

        try:
            onnx.checker.check_model(new_m)
        except Exception as e:
            result["status"] = "CHECKER_FAIL"
            result["error"] = str(e)[:200]
            return result

        r_orig = score_network_v2(str(src))
        r_new = score_network_v2(str(out))
        if r_orig is None or r_new is None:
            result["status"] = "SCORE_FAIL"
            return result

        orig_cost = r_orig[1] + r_orig[2]
        new_cost = r_new[1] + r_new[2]
        delta = new_cost - orig_cost
        result["orig_cost"] = orig_cost
        result["new_cost"] = new_cost
        result["delta"] = delta

        if delta >= 0:
            result["status"] = "NO_IMPROVEMENT"
            return result

        # Verify
        tid_num = int(tid.replace("task", ""))
        passed, total, _ = run_verify(out, tid_num)
        result["verify"] = f"{passed}/{total}"

        if total == 0 or passed != total:
            result["status"] = "VERIFY_FAIL"
            return result

        # Stage
        cand_id = f"{tid}_f16drop_t_{hex8()}_on_5835_30"
        dest = STAGED / cand_id
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, dest / f"{tid}.onnx")

        manifest = {
            "schema": "neurogolf.staged_submit.v1",
            "candidate_id": cand_id,
            "task_id": tid,
            "pattern": "f16_whole_graph_with_f16_output",
            "tag": tag,
            "scores": {
                "base_cost": orig_cost,
                "candidate_cost": new_cost,
                "cost_delta": delta,
            },
            "tests_run": {"verify_against_live": f"{passed}/{total}"},
            "lineage_base_candidate_id": "canonical_5835_30",
            "staged_at": datetime.now(timezone.utc).isoformat(),
            "worker": "neurogolf-worker-OB_HUGE",
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
        result["status"] = "STAGED"
        result["candidate_id"] = cand_id
        return result
    except Exception as e:
        result["status"] = "EXCEPTION"
        result["error"] = str(e)[:300]
        return result


OB_HUGE_ALL = [
    'task138','task264','task205','task153','task208','task202','task201','task069',
    'task008','task029','task018','task396','task216','task088','task097','task137',
    'task384','task036','task195','task159','task400','task377','task193','task085',
    'task238','task206','task209','task122','task170','task325','task290','task157',
    'task213','task014','task244','task049','task308','task065',
]

CL_MID_SMALL_ALL = [
    'task118','task040','task125','task156','task354','task023','task283',
    'task312','task314','task332','task292','task147','task272','task294','task229',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tasks", nargs="?")
    p.add_argument("--all-ob-huge", action="store_true")
    p.add_argument("--tag", default="OB_HUGE_f16_batch")
    p.add_argument("--skip", default="", help="Comma-separated task IDs to skip")
    args = p.parse_args()

    if args.all_ob_huge:
        task_ids = OB_HUGE_ALL
    elif args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
    else:
        p.print_help()
        return 1

    skip = set(t.strip() for t in args.skip.split(",") if t.strip())
    task_ids = [t for t in task_ids if t not in skip]

    tmpdir = Path(tempfile.mkdtemp())
    print(f"Temp: {tmpdir}")
    print(f"Processing {len(task_ids)} tasks...")

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
        total_saved = sum(r["delta"] for r in staged) if staged else 0
        print(f"Staged: {len(staged)} / {len(task_ids)}")
        print(f"Total bytes saved: {total_saved}")
        by_status = {}
        for r in results:
            by_status.setdefault(r["status"], []).append(r["task_id"])
        for s, ids in by_status.items():
            print(f"  {s}: {len(ids)}  ({', '.join(ids[:10])}{'...' if len(ids)>10 else ''})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
