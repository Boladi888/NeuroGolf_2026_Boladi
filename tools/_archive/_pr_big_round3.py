"""PR_BIG round 3: F16 with MatMul/Conv kept F32."""
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


def hex8():
    return secrets.token_hex(4)


STAGED = REPO / ".hermes" / "board" / "staged_submits"
SUBMISSION_ZIP = REPO / "submissions" / "submission.zip"


F32_PRESERVE_OPS = {"MatMul", "CumSum", "Conv"}


def f16_with_op_preserved(tid, tmpdir, preserve_ops):
    with zipfile.ZipFile(SUBMISSION_ZIP) as z:
        z.extract(tid + ".onnx", tmpdir)
    src = tmpdir / (tid + ".onnx")
    m = onnx.load(str(src))
    g = m.graph

    keep_f32_inits = set()
    for n in g.node:
        if n.op_type in preserve_ops:
            for inp in n.input:
                keep_f32_inits.add(inp)

    new_inits = []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and init.name not in keep_f32_inits:
            arr = numpy_helper.to_array(init)
            new_inits.append(
                numpy_helper.from_array(arr.astype(np.float16), name=init.name)
            )
        else:
            new_inits.append(init)

    input_f16 = "_input_f16"
    new_nodes = [
        helper.make_node(
            "Cast",
            inputs=["input"],
            outputs=[input_f16],
            name="_cast_input",
            to=TensorProto.FLOAT16,
        )
    ]

    init_names = {init.name for init in g.initializer}
    casts_added = {}
    back_counter = [0]
    fwd_counter = [0]

    for n in g.node:
        nn = onnx.NodeProto()
        nn.CopyFrom(n)
        for i, inp in enumerate(nn.input):
            if inp == "input":
                nn.input[i] = input_f16

        if nn.op_type in preserve_ops:
            new_inputs = list(nn.input)
            for i, inp in enumerate(new_inputs):
                if inp in init_names:
                    continue
                cast_key = f"_pre_{inp}"
                if cast_key not in casts_added:
                    back_counter[0] += 1
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            inputs=[inp],
                            outputs=[cast_key],
                            name=f"_back_{back_counter[0]}",
                            to=TensorProto.FLOAT,
                        )
                    )
                    casts_added[cast_key] = True
                new_inputs[i] = cast_key
            nn.ClearField("input")
            nn.input.extend(new_inputs)

            orig_out = nn.output[0]
            f32_intermediate = orig_out + "_op_f32"
            nn.output[0] = f32_intermediate
            new_nodes.append(nn)
            fwd_counter[0] += 1
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[f32_intermediate],
                    outputs=[orig_out],
                    name=f"_fwd_{fwd_counter[0]}",
                    to=TensorProto.FLOAT16,
                )
            )
            continue

        if nn.op_type == "Cast":
            for attr in nn.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
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
    new_vi.append(
        helper.make_tensor_value_info(input_f16, TensorProto.FLOAT16, inp_shape)
    )

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
    # Targets where F16 had MatMul/Conv/CumSum issues
    targets = ["task066", "task074", "task323", "task333", "task350", "task367", "task361"]

    for tid in targets:
        try:
            new_m, src_path = f16_with_op_preserved(tid, tmpdir, F32_PRESERVE_OPS)
        except Exception as e:
            print(f"{tid}: build error {str(e)[:120]}")
            continue
        out = tmpdir / f"{tid}_f16_op_f32.onnx"
        onnx.save(new_m, str(out))
        try:
            onnx.checker.check_model(new_m)
        except Exception as e:
            print(f"{tid}: checker {str(e)[:120]}")
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
            print(
                f"{tid}: delta={delta} verify={mv.group(1) + '/' + mv.group(2) if mv else 'parse_fail'} FAIL"
            )

    for tid, delta, out_path, orig_cost, new_cost, passed, total in hits:
        cand_id = f"{tid}_f16_op_f32_t_{hex8()}_on_5835_30"
        dest = STAGED / cand_id
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, dest / (tid + ".onnx"))
        (dest / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "neurogolf.staged_submit.v1",
                    "candidate_id": cand_id,
                    "task_id": tid,
                    "pattern": "f16_with_matmul_conv_cumsum_preserved",
                    "tag": "PR_BIG_f16_op_f32",
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
