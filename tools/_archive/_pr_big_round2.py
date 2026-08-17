"""PR_BIG round 2: cast-after-slice with merge of downstream redundant Cast."""
import zipfile
import sys
import tempfile
import subprocess
import re
import json
import shutil
import secrets
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import onnx
from onnx import TensorProto, helper, numpy_helper
from tools._score_v2 import score_network_v2
import numpy as np


def hex8():
    return secrets.token_hex(4)


STAGED = REPO / ".hermes" / "board" / "staged_submits"
SUBMISSION_ZIP = REPO / "submissions" / "submission.zip"


def cast_after_slice_with_cast_merge(tid, tmpdir):
    with zipfile.ZipFile(SUBMISSION_ZIP) as z:
        z.extract(tid + ".onnx", tmpdir)
    src = tmpdir / (tid + ".onnx")
    m = onnx.load(str(src))
    g = m.graph

    first_slice = next(
        (n for n in g.node if n.op_type == "Slice" and n.input and n.input[0] == "input"),
        None,
    )
    if first_slice is None:
        return None, None

    slice_out_orig = first_slice.output[0]

    redundant_cast = None
    for n in g.node:
        if n.op_type == "Cast" and n.input and n.input[0] == slice_out_orig:
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT16:
                    redundant_cast = n
                    break
            if redundant_cast:
                break

    new_inits = []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16), name=init.name))
        else:
            new_inits.append(init)

    f32_name = slice_out_orig + "_f32_internal"
    new_nodes = []
    cast_output_name = None
    if redundant_cast is not None:
        cast_output_name = redundant_cast.output[0]

    for n in g.node:
        nn = onnx.NodeProto()
        nn.CopyFrom(n)
        if n is first_slice:
            nn.output[0] = f32_name
            new_nodes.append(nn)
            if redundant_cast is not None:
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[f32_name],
                        outputs=[cast_output_name],
                        name="_merged_cast_slice_f16",
                        to=TensorProto.FLOAT16,
                    )
                )
            else:
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[f32_name],
                        outputs=[slice_out_orig],
                        name="_cast_slice_f16",
                        to=TensorProto.FLOAT16,
                    )
                )
            continue
        if n is redundant_cast:
            continue
        if redundant_cast is not None:
            for i, inp in enumerate(nn.input):
                if inp == slice_out_orig:
                    nn.input[i] = cast_output_name
        if nn.op_type == "Cast":
            for attr in nn.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        new_nodes.append(nn)

    new_vi = []
    for vi in g.value_info:
        nvi = onnx.ValueInfoProto()
        nvi.CopyFrom(vi)
        if redundant_cast is not None and vi.name == slice_out_orig:
            continue
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            nvi.type.tensor_type.elem_type = TensorProto.FLOAT16
        new_vi.append(nvi)

    for vi in g.value_info:
        if vi.name == slice_out_orig:
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            new_vi.append(
                helper.make_tensor_value_info(f32_name, TensorProto.FLOAT, shape)
            )
            break

    out_shape = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    orig_out_dt = g.output[0].type.tensor_type.elem_type
    new_out_dt = (
        TensorProto.FLOAT16 if orig_out_dt == TensorProto.FLOAT else orig_out_dt
    )
    new_output = helper.make_tensor_value_info(
        g.output[0].name, new_out_dt, out_shape
    )

    new_graph = helper.make_graph(
        new_nodes,
        g.name,
        list(g.input),
        [new_output],
        new_inits,
        value_info=new_vi,
    )
    return (
        helper.make_model(
            new_graph,
            opset_imports=list(m.opset_import),
            ir_version=m.ir_version,
            producer_name=m.producer_name,
            producer_version=m.producer_version,
        ),
        str(src),
    )


def main():
    tmpdir = Path(tempfile.mkdtemp())
    hits = []
    targets = ["task117", "task119", "task132", "task055", "task099", "task361", "task071", "task338"]
    for tid in targets:
        result = cast_after_slice_with_cast_merge(tid, tmpdir)
        if result[0] is None:
            print(f"{tid}: NO_SLICE")
            continue
        new_m, src_path = result
        out = tmpdir / f"{tid}_castmerge.onnx"
        onnx.save(new_m, str(out))
        try:
            onnx.checker.check_model(new_m)
        except Exception as e:
            print(f"{tid}: checker_fail {str(e)[:120]}")
            continue
        r_orig = score_network_v2(src_path)
        r_new = score_network_v2(str(out))
        if r_new is None:
            print(f"{tid}: score_fail")
            continue
        delta = (r_new[1] + r_new[2]) - (r_orig[1] + r_orig[2])
        if delta >= 0:
            print(f"{tid}: delta=+{delta}")
            continue
        tid_num = int(tid.replace("task", ""))
        v = subprocess.run(
            [
                sys.executable,
                "tools/verify_against_live.py",
                "--proto",
                str(out),
                "--tid",
                str(tid_num),
                "--live-zip",
                "submissions/submission.zip",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        mv = re.search(r"Prototype\s+vs\s+LIVE\s*:\s*(\d+)/(\d+)", v.stdout)
        if mv and mv.group(1) == mv.group(2):
            print(f"{tid}: delta={delta} verify={mv.group(1)}/{mv.group(2)} HIT")
            hits.append(
                (
                    tid,
                    delta,
                    str(out),
                    r_orig[1] + r_orig[2],
                    r_new[1] + r_new[2],
                    mv.group(1),
                    mv.group(2),
                )
            )
        else:
            print(f"{tid}: delta={delta} verify={mv.group(1) + '/' + mv.group(2) if mv else 'parse_fail'} FAIL")

    for tid, delta, out_path, orig_cost, new_cost, passed, total in hits:
        cand_id = f"{tid}_castmerge_t_{hex8()}_on_5835_30"
        dest = STAGED / cand_id
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, dest / (tid + ".onnx"))
        (dest / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "neurogolf.staged_submit.v1",
                    "candidate_id": cand_id,
                    "task_id": tid,
                    "pattern": "f16_cast_merge_slice_and_downstream",
                    "tag": "PR_BIG_castmerge",
                    "scores": {
                        "base_cost": orig_cost,
                        "candidate_cost": new_cost,
                        "cost_delta": delta,
                    },
                    "tests_run": {"verify_against_live": f"{passed}/{total}"},
                    "lineage_base_candidate_id": "canonical_5835_30",
                    "staged_at": datetime.now(timezone.utc).isoformat(),
                    "worker": "neurogolf-worker-PR_BIG",
                },
                indent=2,
            )
        )
        print(f"STAGED: {cand_id}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
