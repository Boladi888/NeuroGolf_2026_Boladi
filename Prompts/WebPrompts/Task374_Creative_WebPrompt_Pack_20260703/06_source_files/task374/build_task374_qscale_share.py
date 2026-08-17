from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[4]
TEAM_BEST = (
    ROOT.parent
    / "K-NeuroGolf-OnnxlyFans"
    / "Team_Submission"
    / "Backend"
    / "Team_Best_Onnx"
    / "task374.onnx"
)
OUT = Path(__file__).with_name("task374_qscale_share.onnx")


def replace_initializer(model, name, array):
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            model.graph.initializer.remove(init)
            model.graph.initializer.insert(i, numpy_helper.from_array(array, name))
            return
    raise KeyError(name)


def remove_initializer(model, name):
    for init in list(model.graph.initializer):
        if init.name == name:
            model.graph.initializer.remove(init)
            return
    raise KeyError(name)


def main():
    model = onnx.load(TEAM_BEST)
    model.graph.name = "task374_twobit_rank_qscale_shared"

    replace_initializer(model, "q_scale", np.array(0.2, dtype=np.float32))
    replace_initializer(model, "row_kernel", np.full((1, 1, 1, 10), 5, dtype=np.uint8))
    replace_initializer(model, "col_kernel", np.full((1, 1, 10, 1), 5, dtype=np.uint8))

    for node in model.graph.node:
        if node.output and node.output[0] == "horizontal_u8":
            node.input[6] = "q_scale"
            break
    else:
        raise RuntimeError("horizontal_u8 QLinearConv not found")

    remove_initializer(model, "h_scale")
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
