import unittest
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import (
    ONEHOT_SHAPE,
    ExportPayload,
    ValidationError,
    app,
    compile_competition_model,
    compile_graph,
    competition_prechecks,
    validate_competition,
    validate_model,
)


client = TestClient(app)


def _require_task(task_id: str):
    from server import _official_task_examples
    try:
        _official_task_examples(task_id)
    except Exception:
        raise unittest.SkipTest(f"competition task JSON for {task_id} not present")


def payload_dict(payload):
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def identity_payload(output_grid=None):
    source = [[0, 1], [2, 3]]
    return ExportPayload(
        projectName="test",
        taskId="task001",
        nodes=[
            {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
            {"id": "identity_1", "type": "op", "data": {"opType": "Identity", "shape": "1,1,30,30"}},
            {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
        ],
        edges=[
            {"source": "input_1", "target": "identity_1"},
            {"source": "identity_1", "target": "output_1"},
        ],
        trainingPairs=[{"input": source, "output": output_grid or source}],
    )


def color_remap_payload():
    source = [[6, 6, 7, 6], [6, 6, 7, 7], [7, 7, 6, 7]]
    target = [[2, 2, 7, 2], [2, 2, 7, 7], [7, 7, 2, 7]]
    return ExportPayload(
        projectName="color-remap",
        taskId="task276",
        nodes=[
            {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
            {"id": "const_6", "type": "op", "data": {"opType": "Constant", "shape": "1,1,30,30", "value": "6"}},
            {"id": "const_2", "type": "op", "data": {"opType": "Constant", "shape": "1,1,30,30", "value": "2"}},
            {"id": "equal_1", "type": "op", "data": {"opType": "Equal", "shape": "1,1,30,30"}},
            {"id": "where_1", "type": "op", "data": {"opType": "Where", "shape": "1,1,30,30"}},
            {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
        ],
        edges=[
            {"source": "input_1", "target": "equal_1", "targetHandle": "a"},
            {"source": "const_6", "target": "equal_1", "targetHandle": "b"},
            {"source": "equal_1", "target": "where_1", "targetHandle": "condition"},
            {"source": "const_2", "target": "where_1", "targetHandle": "true"},
            {"source": "input_1", "target": "where_1", "targetHandle": "false"},
            {"source": "where_1", "target": "output_1", "targetHandle": "input"},
        ],
        trainingPairs=[{"input": source, "output": target}],
    )


def single_op_payload(op_type, attrs=None, output_grid=None):
    source = [[1, 2, 3], [4, 5, 6]]
    nodes = [
        {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
        {"id": "op_1", "type": "op", "data": {"opType": op_type, "shape": "1,1,30,30", "attrs": attrs or {}}},
        {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
    ]
    edges = [
        {"source": "input_1", "target": "op_1", "targetHandle": "input"},
        {"source": "op_1", "target": "output_1", "targetHandle": "input"},
    ]
    return ExportPayload(
        projectName=f"test-{op_type.lower()}",
        taskId="task001",
        nodes=nodes,
        edges=edges,
        trainingPairs=[{"input": source, "output": output_grid or source}],
    )


def binary_op_payload(op_type, attrs=None):
    return ExportPayload(
        projectName=f"test-{op_type.lower()}",
        taskId="task001",
        nodes=[
            {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
            {"id": "const_1", "type": "op", "data": {"opType": "Constant", "shape": "1,1,30,30", "value": "1"}},
            {"id": "op_1", "type": "op", "data": {"opType": op_type, "shape": "1,1,30,30", "attrs": attrs or {}}},
            {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
        ],
        edges=[
            {"source": "input_1", "target": "op_1", "targetHandle": "a"},
            {"source": "const_1", "target": "op_1", "targetHandle": "b"},
            {"source": "op_1", "target": "output_1", "targetHandle": "input"},
        ],
        trainingPairs=[{"input": [[1]], "output": [[1]]}],
    )


def identity_onehot_payload():
    return ExportPayload(
        projectName="onehot-identity",
        taskId="task010",
        representation="onehot",
        nodes=[
            {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,10,30,30"}},
            {"id": "identity_1", "type": "op", "data": {"opType": "Identity", "shape": "1,10,30,30"}},
            {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,10,30,30"}},
        ],
        edges=[
            {"source": "input_1", "target": "identity_1"},
            {"source": "identity_1", "target": "output_1"},
        ],
    )


class CompetitionExportTests(unittest.TestCase):
    """The competition compile path: one-hot I/O named input/output, official semantics."""

    def test_scalar_bridge_produces_competition_io(self):
        model = compile_competition_model(identity_payload())
        self.assertEqual(model.graph.input[0].name, "input")
        self.assertEqual(model.graph.output[0].name, "output")
        dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        self.assertEqual(dims, ONEHOT_SHAPE)
        self.assertEqual(competition_prechecks(model), [])

    def test_onehot_representation_renames_io(self):
        model = compile_competition_model(identity_onehot_payload())
        self.assertEqual(model.graph.input[0].name, "input")
        self.assertEqual(model.graph.output[0].name, "output")
        self.assertEqual(competition_prechecks(model), [])

    def test_onehot_representation_requires_onehot_input_shape(self):
        payload = identity_payload()
        payload.representation = "onehot"
        with self.assertRaisesRegex(ValueError, "onehot representation requires"):
            compile_competition_model(payload)

    def test_scalar_identity_round_trips_official_encoding(self):
        # An identity scalar graph wrapped by the bridge must reproduce the
        # exact one-hot input, including all-zero padding cells, on any
        # same-size task. Use a tiny synthetic example set.
        import numpy as np
        import onnxruntime as ort

        model = compile_competition_model(identity_payload())
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        from server import _grid_to_onehot

        grid = [[0, 1, 2], [3, 0, 9]]
        onehot = _grid_to_onehot(grid)
        result = session.run(["output"], {"input": onehot})[0]
        predicted = (result > 0.0).astype(float)
        self.assertTrue(np.array_equal(predicted, onehot))

    def test_validate_competition_official_sets(self):
        _require_task("task010")
        model = compile_competition_model(identity_onehot_payload())
        report = validate_competition(model, "task010", ["train"])
        self.assertIn("train", report["sets"])
        self.assertGreater(report["sets"]["train"]["pass"] + report["sets"]["train"]["fail"], 0)

    def test_prechecks_reject_wrong_io_names(self):
        model = compile_graph(identity_payload())  # editor model: node-id tensor names
        failures = competition_prechecks(model)
        self.assertTrue(any("must be named 'input'" in failure for failure in failures))

    def test_validate_endpoint_reports_score_fields(self):
        _require_task("task001")
        payload = payload_dict(identity_payload())
        payload["exampleSets"] = ["train"]
        response = client.post("/api/validate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["status"], ("validated", "not_ready"))
        self.assertIn("validation", data)
        self.assertIn("score", data)
        self.assertIn("baseline", data)

    def test_project_status_endpoint(self):
        response = client.get("/api/project-status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("projectRoot", data)
        self.assertIn("scorer", data)


class DoorEngineTests(unittest.TestCase):
    """Doors Maze engine: numpy doors, lookahead, and ONNX lowering equivalence."""

    def test_basic_doors(self):
        import numpy as np
        import door_engine

        grid = np.array([[1, 2], [3, 0]])
        self.assertEqual(door_engine.apply_door(grid, "flip_h", {}).tolist(), [[2, 1], [0, 3]])
        self.assertEqual(door_engine.apply_door(grid, "recolor", {"from": 2, "to": 5}).tolist(), [[1, 5], [3, 0]])
        self.assertEqual(door_engine.apply_door(grid, "rotate", {"turns": 1}).tolist(), [[3, 1], [0, 2]])
        self.assertEqual(door_engine.apply_door(grid, "tile", {"down": 1, "across": 2}).tolist(), [[1, 2, 1, 2], [3, 0, 3, 0]])
        cropped = door_engine.apply_door(np.array([[0, 0, 0], [0, 7, 0], [0, 0, 0]]), "crop_content", {})
        self.assertEqual(cropped.tolist(), [[7]])

    def test_size_cap_raises(self):
        import numpy as np
        import door_engine

        with self.assertRaises(door_engine.DoorError):
            door_engine.apply_door(np.ones((20, 20), dtype=int), "tile", {"down": 2, "across": 2})

    def test_path_state_and_win(self):
        import door_engine

        examples = [{"input": [[1, 2]], "output": [[2, 1]]}, {"input": [[3, 4]], "output": [[4, 3]]}]
        state = door_engine.path_state(examples, [{"op": "flip_h", "params": {}}])
        self.assertTrue(state["win"])
        self.assertEqual(state["distance"], 1.0)

    def test_lookahead_suggests_winning_door(self):
        import door_engine

        examples = [{"input": [[6, 0], [6, 6]], "output": [[2, 0], [2, 2]]}]
        moves = door_engine.lookahead(examples, [])
        self.assertEqual(moves[0]["op"], "recolor")
        self.assertEqual(moves[0]["projectedDistance"], 1.0)

    def test_greedy_solve_finds_flip(self):
        import door_engine

        examples = [{"input": [[1, 2, 3]], "output": [[3, 2, 1]]}]
        path = door_engine.greedy_solve(examples)
        self.assertIsNotNone(path)
        self.assertEqual(path[0]["op"], "flip_h")

    def test_negative_step_slice_shape(self):
        from server import _output_shape_for_slice

        shape = _output_shape_for_slice([1, 1, 4, 7], {"starts": [6], "ends": [-8], "axes": [3], "steps": [-1]})
        self.assertEqual(shape, [1, 1, 4, 7])

    def _assert_lowering_matches_numpy(self, grid, path):
        import numpy as np
        import onnxruntime as ort
        import door_engine
        from server import _grid_to_onehot

        payload = ExportPayload(**door_engine.lower_to_payload("task001", grid, path))
        model = compile_competition_model(payload)
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        raw = session.run(["output"], {"input": _grid_to_onehot(grid)})[0]
        predicted = (raw > 0.0).astype(float)
        expected_grid = door_engine.apply_path(np.asarray(grid), path)
        expected = _grid_to_onehot(expected_grid.tolist())
        self.assertTrue(np.array_equal(predicted, expected))

    def test_lowering_matches_numpy_cellwise(self):
        self._assert_lowering_matches_numpy(
            [[6, 0, 3], [6, 6, 0]],
            [{"op": "recolor", "params": {"from": 6, "to": 2}}, {"op": "keep_color", "params": {"color": 2}}],
        )

    def test_lowering_matches_numpy_geometry(self):
        self._assert_lowering_matches_numpy(
            [[1, 2, 3], [4, 5, 6]],
            [{"op": "rotate", "params": {"turns": 1}}, {"op": "mirror_concat", "params": {"direction": "right"}},
             {"op": "pad", "params": {"amount": 1, "color": 4}}],
        )

    def test_door_library_endpoint(self):
        response = client.get("/api/door/library")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["doors"]), 15)
        self.assertTrue(all("op" in door and "group" in door for door in data["doors"]))

    def test_door_step_endpoint_win_and_lookahead(self):
        _require_task("task276")
        response = client.post("/api/door/step", json={
            "taskId": "task276", "path": [{"op": "recolor", "params": {"from": 6, "to": 2}}]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["win"])
        self.assertEqual(data["sets"]["train"]["fail"], 0)
        response = client.post("/api/door/step", json={"taskId": "task276", "path": []})
        data = response.json()
        self.assertFalse(data["win"])
        self.assertTrue(any(move["op"] == "recolor" for move in data["lookahead"]))

    def test_door_campaign_endpoint(self):
        response = client.get("/api/door/campaign")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data["tutorials"]), 3)
        self.assertIn("queue", data)

    def test_stamp_doors(self):
        import numpy as np
        import door_engine

        grid = np.array([[1, 0], [0, 2]])
        stamped = door_engine.apply_door(grid, "stamp", {})
        self.assertEqual(stamped.shape, (4, 4))
        self.assertEqual(stamped[:2, :2].tolist(), [[1, 0], [0, 2]])  # colored (0,0) -> pattern
        self.assertEqual(stamped[:2, 2:].tolist(), [[0, 0], [0, 0]])  # empty (0,1) -> blank
        inverted = door_engine.apply_door(grid, "stamp_empty", {})
        self.assertEqual(inverted[:2, :2].tolist(), [[0, 0], [0, 0]])
        self.assertEqual(inverted[:2, 2:].tolist(), [[1, 0], [0, 2]])

    def test_stamp_wins_task001(self):
        _require_task("task001")
        response = client.post("/api/door/step", json={
            "taskId": "task001", "path": [{"op": "stamp", "params": {}}]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["win"])
        self.assertEqual(data["sets"]["arc-gen"]["fail"], 0)

    def test_color_math_door(self):
        import numpy as np
        import door_engine

        grid = np.array([[0, 1, 9], [5, 0, 8]])
        out = door_engine.apply_door(grid, "color_math", {"amount": 1})
        self.assertEqual(out.tolist(), [[0, 2, 1], [6, 0, 9]])  # bg stays; 9 wraps to 1

    def test_rule_note_requires_text(self):
        response = client.post("/api/door/rule-note", json={"taskId": "task001", "path": [], "ruleHypothesis": " "})
        self.assertEqual(response.status_code, 400)

    def test_wizard_rejects_unknown_backend(self):
        _require_task("task001")
        response = client.post("/api/door/wizard", json={"taskId": "task001", "path": [], "backend": "gemini"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("backend", response.json()["reason"])

    def test_task_rule_endpoint(self):
        response = client.get("/api/task-rule/task011")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["taskId"], "task011")
        self.assertIn("known", data)
        self.assertIn("recipes", data)
        if data["known"]:  # project root present -> documented rule text comes through
            self.assertIn("block", data["rule"].lower())

    def test_task_rule_ghost_path_from_campaign(self):
        response = client.get("/api/task-rule/task031")  # Tutorial 1 has a ghostPath
        self.assertEqual(response.status_code, 200)
        ghost = response.json()["ghostPath"]
        self.assertTrue(ghost and ghost[0]["op"] == "crop_content")

    def test_task_rule_rejects_bad_id(self):
        response = client.get("/api/task-rule/evil..path")
        self.assertEqual(response.status_code, 400)


class ServerCompilerTests(unittest.TestCase):
    def test_compile_and_validate_identity_graph(self):
        payload = identity_payload()
        model = compile_graph(payload)
        result = validate_model(model, payload)
        self.assertEqual(result["train"], "passed")
        self.assertEqual(result["shape"], "passed")
        self.assertEqual(result["colors"], "passed")

    def test_rejects_banned_ops(self):
        payload = ExportPayload(
            projectName="bad",
            taskId="task001",
            nodes=[
                {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
                {"id": "loop_1", "type": "op", "data": {"opType": "Loop", "shape": "1,1,30,30"}},
                {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
            ],
            edges=[
                {"source": "input_1", "target": "loop_1"},
                {"source": "loop_1", "target": "output_1"},
            ],
            trainingPairs=[{"input": [[0]], "output": [[0]]}],
        )
        with self.assertRaisesRegex(ValueError, "Banned ONNX operation"):
            compile_graph(payload)

    def test_rejects_malformed_graph_edges(self):
        payload = identity_payload()
        payload.edges = [{"source": "missing", "target": "identity_1"}]
        with self.assertRaisesRegex(ValueError, "edge source"):
            compile_graph(payload)

    def test_validation_failure_reports_mismatch(self):
        payload = identity_payload(output_grid=[[9, 9], [9, 9]])
        model = compile_graph(payload)
        with self.assertRaisesRegex(ValidationError, "Strict Equivalence failed"):
            validate_model(model, payload)

    def test_named_input_slots_compile_color_remap(self):
        payload = color_remap_payload()
        model = compile_graph(payload)
        result = validate_model(model, payload)
        self.assertEqual(result["train"], "passed")

    def test_rejects_duplicate_named_input_slot(self):
        payload = color_remap_payload()
        payload.edges[1]["targetHandle"] = "a"
        with self.assertRaisesRegex(ValueError, "multiple edges for input slot 'a'"):
            compile_graph(payload)

    def test_slice_can_crop_output_window(self):
        payload = single_op_payload(
            "Slice",
            attrs={"starts": [0, 0, 0, 0], "ends": [1, 1, 1, 2], "axes": [0, 1, 2, 3], "steps": [1, 1, 1, 1]},
            output_grid=[[1, 2]],
        )
        model = compile_graph(payload)
        result = validate_model(model, payload)
        self.assertEqual(result["train"], "passed")

    def test_transpose_can_swap_grid_axes(self):
        payload = single_op_payload("Transpose", attrs={"perm": [0, 1, 3, 2]}, output_grid=[[1, 4], [2, 5], [3, 6]])
        model = compile_graph(payload)
        result = validate_model(model, payload)
        self.assertEqual(result["train"], "passed")

    def test_compile_spatial_and_coordinate_nodes(self):
        cases = [
            single_op_payload("Pad", attrs={"pads": [0, 0, 1, 1, 0, 0, 0, 0], "value": 0}),
            single_op_payload("Tile", attrs={"repeats": [1, 1, 1, 1]}),
            single_op_payload("Resize", attrs={"sizes": [1, 1, 30, 30], "mode": "nearest"}),
            single_op_payload("Conv", attrs={"weight_shape": [1, 1, 3, 3], "weights": [1, 0, 0, 0, 0, 0, 0, 0, 0], "pads": [1, 1, 1, 1]}),
            ExportPayload(
                projectName="coords",
                taskId="task001",
                nodes=[
                    {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
                    {"id": "row_1", "type": "op", "data": {"opType": "RowIndex", "shape": "1,1,30,30"}},
                    {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,1,30,30"}},
                ],
                edges=[{"source": "row_1", "target": "output_1", "targetHandle": "input"}],
                trainingPairs=[{"input": [[0]], "output": [[0]]}],
            ),
        ]
        for payload in cases:
            with self.subTest(op=payload.nodes[1]["data"]["opType"]):
                compile_graph(payload)

    def test_concat_compiles_two_inputs(self):
        payload = ExportPayload(
            projectName="concat",
            taskId="task001",
            nodes=[
                {"id": "input_1", "type": "op", "data": {"opType": "Input", "shape": "1,1,30,30"}},
                {"id": "const_1", "type": "op", "data": {"opType": "Constant", "shape": "1,1,30,30", "value": "0"}},
                {"id": "concat_1", "type": "op", "data": {"opType": "Concat", "shape": "1,2,30,30", "attrs": {"axis": 1}}},
                {"id": "output_1", "type": "op", "data": {"opType": "Output", "shape": "1,2,30,30"}},
            ],
            edges=[
                {"source": "input_1", "target": "concat_1", "targetHandle": "a"},
                {"source": "const_1", "target": "concat_1", "targetHandle": "b"},
                {"source": "concat_1", "target": "output_1", "targetHandle": "input"},
            ],
            trainingPairs=[{"input": [[1]], "output": [[1]]}],
        )
        compile_graph(payload)

    def test_compile_low_risk_extended_ops(self):
        cases = [
            single_op_payload("Relu"),
            single_op_payload("Abs"),
            single_op_payload("Neg"),
            single_op_payload("Floor"),
            single_op_payload("Clip", attrs={"min": 0, "max": 9}),
            single_op_payload("Sign"),
            single_op_payload("Sqrt"),
            single_op_payload("ReduceMax", attrs={"axes": [1], "keepdims": 1}),
            single_op_payload("ReduceMin", attrs={"axes": [1], "keepdims": 1}),
            binary_op_payload("GreaterOrEqual"),
            binary_op_payload("LessOrEqual"),
            binary_op_payload("Add"),
            binary_op_payload("Sub"),
            binary_op_payload("Mul"),
            binary_op_payload("Div"),
            binary_op_payload("Mod"),
            binary_op_payload("Min"),
            binary_op_payload("Max"),
            binary_op_payload("Sum"),
        ]
        for payload in cases:
            with self.subTest(op=payload.nodes[-2]["data"]["opType"]):
                compile_graph(payload)

    def test_generic_in_numbered_slots_order_inputs(self):
        payload = binary_op_payload("Add")
        payload.edges[0]["targetHandle"] = "in0"
        payload.edges[1]["targetHandle"] = "in1"
        compile_graph(payload)

    def test_compile_endpoint_reports_model_summary_without_upload(self):
        response = client.post("/api/compile", json=payload_dict(identity_payload()))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "compiled")
        self.assertGreater(data["modelBytes"], 0)
        self.assertEqual(data["io"]["inputs"][0]["shape"], [1, 1, 30, 30])
        self.assertEqual(data["io"]["outputs"][0]["shape"], [1, 1, 30, 30])

    def test_run_endpoint_runs_current_input_grid_without_upload(self):
        payload = payload_dict(identity_payload())
        payload["inputGrid"] = [[4, 5], [6, 7]]
        response = client.post("/api/run", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ran")
        self.assertEqual(data["grid"], [[4, 5], [6, 7]])

    def test_best_graph_endpoint_imports_onnx_as_visual_nodes(self):
        best_path = Path(__file__).resolve().parents[1] / "client" / "public" / "best" / "onnx" / "task001.onnx"
        if not best_path.exists():
            self.skipTest("best ONNX assets are not present")
        response = client.get("/api/best-graph/task001")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["taskId"], "task001")
        self.assertTrue(data["meta"]["rawOnnx"])
        self.assertGreater(len(data["nodes"]), 0)
        self.assertGreater(len(data["edges"]), 0)
        self.assertIn("Slice", data["meta"]["opTypes"])


if __name__ == "__main__":
    unittest.main()
