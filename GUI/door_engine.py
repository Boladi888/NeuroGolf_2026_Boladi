"""Doors Maze engine — numpy door library, path replay, lookahead, ONNX lowering.

Single source of truth for the game (see DOORS_DESIGN.md): the same functions
*play* a door, *verify* a winning path, and *lower* an all-macro path into the
GUI compiler's ExportPayload node/edge format so it rides the existing
competition gate (compile -> official validation -> official score).

Grids are native-size numpy int arrays (H x W, values 0-9, <= 30x30).
Background is color 0. A path is a list of {"op": str, "params": dict} steps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_DIM = 30
BG = 0


class DoorError(ValueError):
    """A door cannot be applied to the current grid (player-facing message)."""


# ---------------------------------------------------------------------------
# Door library (v1 — geometric, all ONNX-macro lowerable)
# ---------------------------------------------------------------------------

DOORS: dict[str, dict[str, Any]] = {
    "recolor": {
        "group": "Paint", "emoji": "🎨", "label": "Recolor",
        "params": [{"name": "from", "type": "color"}, {"name": "to", "type": "color"}],
        "blurb": "Turn every cell of one color into another color.",
    },
    "keep_color": {
        "group": "Paint", "emoji": "🎯", "label": "Keep color",
        "params": [{"name": "color", "type": "color"}],
        "blurb": "Keep one color; everything else becomes background.",
    },
    "crop_content": {
        "group": "Scissors", "emoji": "✂️", "label": "Crop to stuff",
        "params": [],
        "blurb": "Cut away the empty border around the picture.",
    },
    "crop_color": {
        "group": "Scissors", "emoji": "🔍", "label": "Crop to color",
        "params": [{"name": "color", "type": "color"}],
        "blurb": "Cut down to the box around one color.",
    },
    "pad": {
        "group": "Scissors", "emoji": "🖼️", "label": "Add border",
        "params": [{"name": "amount", "type": "amount"}, {"name": "color", "type": "color"}],
        "blurb": "Add a border of one color all around.",
    },
    "flip_h": {
        "group": "Mirror", "emoji": "↔️", "label": "Flip sideways",
        "params": [],
        "blurb": "Mirror left-right.",
    },
    "flip_v": {
        "group": "Mirror", "emoji": "↕️", "label": "Flip upside down",
        "params": [],
        "blurb": "Mirror top-bottom.",
    },
    "rotate": {
        "group": "Mirror", "emoji": "🔄", "label": "Rotate",
        "params": [{"name": "turns", "type": "turns"}],
        "blurb": "Turn the picture 90, 180, or 270 degrees.",
    },
    "transpose": {
        "group": "Mirror", "emoji": "⤡", "label": "Diagonal flip",
        "params": [],
        "blurb": "Swap rows and columns.",
    },
    "tile": {
        "group": "Magnet", "emoji": "🧱", "label": "Tile",
        "params": [{"name": "down", "type": "amount"}, {"name": "across", "type": "amount"}],
        "blurb": "Repeat the picture in a grid.",
    },
    "shift": {
        "group": "Magnet", "emoji": "🧲", "label": "Slide",
        "params": [{"name": "direction", "type": "direction"}, {"name": "amount", "type": "amount"}],
        "blurb": "Slide everything; empty space fills with background.",
    },
    "mirror_concat": {
        "group": "Mirror", "emoji": "🪞", "label": "Mirror copy",
        "params": [{"name": "direction", "type": "direction"}],
        "blurb": "Stick a mirrored copy onto one side.",
    },
    # v2 "Magic" doors — numpy-only (no ONNX macro yet); winning paths that use
    # them export as recipes for the agent loop to golf.
    "stamp": {
        "group": "Magic", "emoji": "✨", "label": "Stamp on colors",
        "params": [],
        "blurb": "Replace every COLORED cell with a whole copy of the picture.",
    },
    "stamp_empty": {
        "group": "Magic", "emoji": "🌑", "label": "Stamp on empty",
        "params": [],
        "blurb": "Replace every EMPTY cell with a whole copy of the picture.",
    },
    "color_math": {
        "group": "Magic", "emoji": "🔢", "label": "Color math",
        "params": [{"name": "amount", "type": "amount"}],
        "blurb": "Add N to every colored cell's number (wraps 9 back to 1; background stays).",
    },
}

_DIRECTIONS = {"up", "down", "left", "right"}


def _as_grid(value: Any) -> np.ndarray:
    grid = np.asarray(value, dtype=np.int64)
    if grid.ndim != 2 or grid.size == 0:
        raise DoorError("grids must be non-empty 2D")
    return grid


def _check_size(h: int, w: int, op: str) -> None:
    if h > MAX_DIM or w > MAX_DIM:
        raise DoorError(f"{op} would grow the picture past {MAX_DIM}x{MAX_DIM} ({h}x{w})")
    if h < 1 or w < 1:
        raise DoorError(f"{op} would leave nothing ({h}x{w})")


def _color(params: dict[str, Any], key: str, op: str) -> int:
    value = params.get(key)
    if not isinstance(value, int) or not 0 <= value <= 9:
        raise DoorError(f"{op} needs '{key}' as a color 0-9")
    return value


def _amount(params: dict[str, Any], key: str, op: str, lo: int = 1, hi: int = 5) -> int:
    value = params.get(key)
    if not isinstance(value, int) or not lo <= value <= hi:
        raise DoorError(f"{op} needs '{key}' between {lo} and {hi}")
    return value


def _bbox(mask: np.ndarray, op: str) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        raise DoorError(f"{op}: there is nothing to crop to")
    r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return r0, r1, c0, c1


def apply_door(grid: np.ndarray, op: str, params: dict[str, Any]) -> np.ndarray:
    grid = _as_grid(grid)
    h, w = grid.shape
    if op == "recolor":
        src, dst = _color(params, "from", op), _color(params, "to", op)
        out = grid.copy()
        out[grid == src] = dst
        return out
    if op == "keep_color":
        color = _color(params, "color", op)
        return np.where(grid == color, grid, BG)
    if op == "crop_content":
        r0, r1, c0, c1 = _bbox(grid != BG, op)
        return grid[r0:r1, c0:c1]
    if op == "crop_color":
        color = _color(params, "color", op)
        r0, r1, c0, c1 = _bbox(grid == color, op)
        return grid[r0:r1, c0:c1]
    if op == "pad":
        amount = _amount(params, "amount", op, 1, 5)
        color = _color(params, "color", op)
        _check_size(h + 2 * amount, w + 2 * amount, op)
        return np.pad(grid, amount, constant_values=color)
    if op == "flip_h":
        return grid[:, ::-1]
    if op == "flip_v":
        return grid[::-1, :]
    if op == "rotate":
        turns = params.get("turns")
        if turns not in (1, 2, 3):
            raise DoorError("rotate needs 'turns' of 1, 2, or 3 (x90 clockwise)")
        return np.rot90(grid, k=-turns)
    if op == "transpose":
        return grid.T
    if op == "tile":
        down = _amount(params, "down", op, 1, 6)
        across = _amount(params, "across", op, 1, 6)
        if down == 1 and across == 1:
            raise DoorError("tile needs at least one repeat above 1")
        _check_size(h * down, w * across, op)
        return np.tile(grid, (down, across))
    if op == "shift":
        direction = params.get("direction")
        if direction not in _DIRECTIONS:
            raise DoorError("shift needs a direction: up/down/left/right")
        amount = _amount(params, "amount", op, 1, 10)
        out = np.full_like(grid, BG)
        if direction == "right" and amount < w:
            out[:, amount:] = grid[:, :-amount]
        elif direction == "left" and amount < w:
            out[:, :-amount] = grid[:, amount:]
        elif direction == "down" and amount < h:
            out[amount:, :] = grid[:-amount, :]
        elif direction == "up" and amount < h:
            out[:-amount, :] = grid[amount:, :]
        return out
    if op == "color_math":
        amount = _amount(params, "amount", op, 1, 8)
        shifted = ((grid - 1 + amount) % 9) + 1
        return np.where(grid == BG, grid, shifted)
    if op in ("stamp", "stamp_empty"):
        _check_size(h * h, w * w, op)
        condition = (grid != BG) if op == "stamp" else (grid == BG)
        blocks = np.repeat(np.repeat(condition, h, axis=0), w, axis=1)
        return np.where(blocks, np.tile(grid, (h, w)), BG)
    if op == "mirror_concat":
        direction = params.get("direction")
        if direction not in _DIRECTIONS:
            raise DoorError("mirror copy needs a direction: up/down/left/right")
        if direction in ("left", "right"):
            _check_size(h, w * 2, op)
            flipped = grid[:, ::-1]
            return np.concatenate([grid, flipped] if direction == "right" else [flipped, grid], axis=1)
        _check_size(h * 2, w, op)
        flipped = grid[::-1, :]
        return np.concatenate([grid, flipped] if direction == "down" else [flipped, grid], axis=0)
    raise DoorError(f"unknown door {op!r}")


def apply_path(grid: np.ndarray, path: list[dict[str, Any]]) -> np.ndarray:
    current = _as_grid(grid)
    for index, step in enumerate(path, start=1):
        try:
            current = apply_door(current, str(step.get("op")), step.get("params") or {})
        except DoorError as exc:
            raise DoorError(f"step {index} ({step.get('op')}): {exc}") from exc
    return current


# ---------------------------------------------------------------------------
# Distance + state
# ---------------------------------------------------------------------------

def distance(current: np.ndarray, target: np.ndarray) -> float:
    """Mean matching cells; 0.0 until shapes match (DOORS_DESIGN.md §2)."""
    if current.shape != target.shape:
        return 0.0
    return float(np.mean(current == target))


def path_state(examples: list[dict[str, Any]], path: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay `path` on every example. Raises DoorError naming the first
    example where a step fails."""
    per_example = []
    for index, example in enumerate(examples, start=1):
        target = _as_grid(example["output"])
        try:
            current = apply_path(_as_grid(example["input"]), path)
        except DoorError as exc:
            raise DoorError(f"example {index}: {exc}") from exc
        per_example.append({
            "current": current.tolist(),
            "distance": distance(current, target),
            "shapeOk": current.shape == target.shape,
        })
    overall = float(np.mean([item["distance"] for item in per_example])) if per_example else 0.0
    win = bool(per_example) and all(item["distance"] == 1.0 for item in per_example)
    return {"examples": per_example, "distance": overall, "win": win}


def verify_sets(task_examples: dict[str, list], path: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count exact matches per official example set (train/test/arc-gen)."""
    results: dict[str, dict[str, int]] = {}
    for set_name in ("train", "test", "arc-gen"):
        right = wrong = errors = 0
        for example in task_examples.get(set_name, []):
            try:
                current = apply_path(_as_grid(example["input"]), path)
                target = _as_grid(example["output"])
                if current.shape == target.shape and bool(np.all(current == target)):
                    right += 1
                else:
                    wrong += 1
            except DoorError:
                errors += 1
        results[set_name] = {"pass": right, "fail": wrong + errors}
    return results


# ---------------------------------------------------------------------------
# Tier-1 lookahead — try every door, rank by distance gain
# ---------------------------------------------------------------------------

def _candidates(currents: list[np.ndarray], targets: list[np.ndarray]) -> list[tuple[str, dict[str, Any], str]]:
    present = sorted({int(v) for grid in currents for v in np.unique(grid)})
    wanted = sorted({int(v) for grid in targets for v in np.unique(grid)})
    out: list[tuple[str, dict[str, Any], str]] = []
    for src in present:
        for dst in wanted + ([BG] if BG not in wanted else []):
            if src != dst:
                out.append(("recolor", {"from": src, "to": dst}, f"Recolor {src}→{dst}"))
    for color in present:
        if color != BG:
            out.append(("keep_color", {"color": color}, f"Keep {color}"))
            out.append(("crop_color", {"color": color}, f"Crop to {color}"))
    out.append(("crop_content", {}, "Crop to stuff"))
    for color in wanted:
        out.append(("pad", {"amount": 1, "color": color}, f"Border {color}"))
    out.extend([
        ("flip_h", {}, "Flip ↔"), ("flip_v", {}, "Flip ↕"), ("transpose", {}, "Diagonal ⤡"),
        ("rotate", {"turns": 1}, "Rotate 90°"), ("rotate", {"turns": 2}, "Rotate 180°"),
        ("rotate", {"turns": 3}, "Rotate 270°"),
    ])
    for down, across in ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (3, 3), (2, 3), (3, 2)):
        out.append(("tile", {"down": down, "across": across}, f"Tile {down}x{across}"))
    for direction in ("up", "down", "left", "right"):
        out.append(("shift", {"direction": direction, "amount": 1}, f"Slide {direction}"))
        out.append(("mirror_concat", {"direction": direction}, f"Mirror {direction}"))
    out.append(("stamp", {}, "Stamp on colors"))
    out.append(("stamp_empty", {}, "Stamp on empty"))
    for amount in range(1, 9):
        out.append(("color_math", {"amount": amount}, f"Color +{amount}"))
    return out


def lookahead(examples: list[dict[str, Any]], path: list[dict[str, Any]], top: int = 5) -> list[dict[str, Any]]:
    currents, targets = [], []
    for example in examples:
        try:
            currents.append(apply_path(_as_grid(example["input"]), path))
        except DoorError:
            return []
        targets.append(_as_grid(example["output"]))
    base = float(np.mean([distance(c, t) for c, t in zip(currents, targets)]))
    ranked = []
    for op, params, label in _candidates(currents, targets):
        score = 0.0
        try:
            for current, target in zip(currents, targets):
                score += distance(apply_door(current, op, params), target)
        except DoorError:
            continue
        score /= len(currents)
        if score > base + 1e-9:
            ranked.append({"op": op, "params": params, "label": label,
                           "projectedDistance": round(score, 4), "gain": round(score - base, 4)})
    ranked.sort(key=lambda item: -item["projectedDistance"])
    return ranked[:top]


def greedy_solve(examples: list[dict[str, Any]], max_depth: int = 3, beam: int = 5) -> list[dict[str, Any]] | None:
    """Beam search over doors. Returns a winning path or None. Used for the
    tutorial scan and the ⭐⭐ "par" length."""
    states: list[list[dict[str, Any]]] = [[]]
    for _ in range(max_depth):
        next_states: list[tuple[float, list[dict[str, Any]]]] = []
        for path in states:
            for move in lookahead(examples, path, top=beam):
                candidate = path + [{"op": move["op"], "params": move["params"]}]
                if move["projectedDistance"] == 1.0:
                    return candidate
                next_states.append((move["projectedDistance"], candidate))
        next_states.sort(key=lambda item: -item[0])
        states = [path for _, path in next_states[:beam]]
        if not states:
            return None
    return None


# ---------------------------------------------------------------------------
# ONNX lowering — door path -> ExportPayload nodes/edges (scalar pipeline)
# ---------------------------------------------------------------------------
# Strategy (DOORS_DESIGN.md §6a): keep the working tensor at native size
# [1,1,h,w] through the pipeline. Entry slices the active region out of the
# 30x30 canvas; exit pads back to 30x30 with sentinel -1 so the scalar->onehot
# encode (bridgeMask "none") produces all-zero channels on padding cells.
# Geometry (entry dims, crop bboxes) is frozen from ONE example's trace, so a
# lowered model is only valid when geometry is constant across examples — the
# official validation decides, and failures fall back to recipe-only export.

class _GraphBuilder:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self._n = 0

    def add(self, op: str, data: dict[str, Any] | None = None,
            inputs: list[tuple[str, str]] | None = None) -> str:
        self._n += 1
        node_id = f"d{self._n}_{op.lower()}"
        self.nodes.append({"id": node_id, "type": "op",
                           "data": {"label": node_id, "opType": op, **(data or {})}})
        for source, handle in inputs or []:
            self.edges.append({"id": f"e{len(self.edges) + 1}", "source": source,
                               "target": node_id, "targetHandle": handle})
        return node_id

    def const(self, value: float, h: int, w: int) -> str:
        return self.add("Constant", {"shape": f"1,1,{h},{w}", "value": str(value)})


def _lower_flip(g: _GraphBuilder, src: str, h: int, w: int, axis: int) -> str:
    dim = h if axis == 2 else w
    return g.add("Slice", {"attrs": {"starts": [dim - 1], "ends": [-(dim + 1)],
                                     "axes": [axis], "steps": [-1]}}, [(src, "input")])


_CELLWISE_DOORS = {"recolor", "keep_color"}


def lower_to_payload(task_id: str, example_input: np.ndarray, path: list[dict[str, Any]]) -> dict[str, Any]:
    """Lower a door path to GUI ExportPayload dict (scalar representation).

    Two regimes:
    - cellwise-only paths (recolor/keep_color) operate on the full 30x30
      canvas with the input-region bridge mask — valid for ANY grid size;
    - geometry paths freeze entry dims / crop bboxes from ONE example's trace
      and use sentinel -1 exit padding (bridgeMask none) — only valid when
      geometry is constant across examples; official validation decides.
    """
    grid = _as_grid(example_input)
    cellwise_only = bool(path) and all(str(step.get("op")) in _CELLWISE_DOORS for step in path)
    g = _GraphBuilder()
    src = g.add("Input", {"shape": "1,1,30,30"})
    if cellwise_only:
        canvas = np.zeros((MAX_DIM, MAX_DIM), dtype=np.int64)
        canvas[: grid.shape[0], : grid.shape[1]] = grid
        grid = canvas
    h, w = grid.shape
    if (h, w) != (MAX_DIM, MAX_DIM):  # entry: cut the active region out of the canvas
        src = g.add("Slice", {"attrs": {"starts": [0, 0, 0, 0], "ends": [1, 1, h, w],
                                        "axes": [0, 1, 2, 3], "steps": [1, 1, 1, 1]}}, [(src, "input")])

    for step in path:
        op, params = str(step.get("op")), step.get("params") or {}
        if op == "recolor":
            const_from = g.const(params["from"], h, w)
            equal = g.add("Equal", {}, [(src, "a"), (const_from, "b")])
            const_to = g.const(params["to"], h, w)
            src = g.add("Where", {}, [(equal, "condition"), (const_to, "true"), (src, "false")])
        elif op == "keep_color":
            const_c = g.const(params["color"], h, w)
            equal = g.add("Equal", {}, [(src, "a"), (const_c, "b")])
            const_bg = g.const(BG, h, w)
            src = g.add("Where", {}, [(equal, "condition"), (src, "true"), (const_bg, "false")])
        elif op in ("crop_content", "crop_color"):
            mask = grid != BG if op == "crop_content" else grid == params["color"]
            r0, r1, c0, c1 = _bbox(mask, op)
            src = g.add("Slice", {"attrs": {"starts": [r0, c0], "ends": [r1, c1],
                                            "axes": [2, 3], "steps": [1, 1]}}, [(src, "input")])
        elif op == "pad":
            a = int(params["amount"])
            src = g.add("Pad", {"attrs": {"pads": [0, 0, a, a, 0, 0, a, a],
                                          "value": params["color"]}}, [(src, "input")])
        elif op == "flip_h":
            src = _lower_flip(g, src, h, w, axis=3)
        elif op == "flip_v":
            src = _lower_flip(g, src, h, w, axis=2)
        elif op == "transpose":
            src = g.add("Transpose", {"attrs": {"perm": [0, 1, 3, 2]}}, [(src, "input")])
        elif op == "rotate":
            turns = int(params["turns"])
            if turns == 2:
                src = _lower_flip(g, src, h, w, axis=3)
                src = _lower_flip(g, src, h, w, axis=2)
            else:
                src = g.add("Transpose", {"attrs": {"perm": [0, 1, 3, 2]}}, [(src, "input")])
                src = _lower_flip(g, src, w, h, axis=3 if turns == 1 else 2)
        elif op == "tile":
            src = g.add("Tile", {"attrs": {"repeats": [1, 1, int(params["down"]), int(params["across"])]}},
                        [(src, "input")])
        elif op == "shift":
            direction, a = params["direction"], int(params["amount"])
            begin_r, begin_c = (a if direction == "down" else 0), (a if direction == "right" else 0)
            end_r, end_c = (a if direction == "up" else 0), (a if direction == "left" else 0)
            src = g.add("Pad", {"attrs": {"pads": [0, 0, begin_r, begin_c, 0, 0, end_r, end_c],
                                          "value": BG}}, [(src, "input")])
            r0, c0 = (a if direction == "up" else 0), (a if direction == "left" else 0)
            src = g.add("Slice", {"attrs": {"starts": [r0, c0], "ends": [r0 + h, c0 + w],
                                            "axes": [2, 3], "steps": [1, 1]}}, [(src, "input")])
        elif op == "mirror_concat":
            direction = params["direction"]
            axis = 3 if direction in ("left", "right") else 2
            flipped = _lower_flip(g, src, h, w, axis=axis)
            first, second = (src, flipped) if direction in ("right", "down") else (flipped, src)
            src = g.add("Concat", {"attrs": {"axis": axis}}, [(first, "a"), (second, "b")])
        else:
            raise DoorError(f"door {op!r} has no ONNX macro lowering")
        grid = apply_door(grid, op, params)  # keep the geometry trace in sync
        h, w = grid.shape

    if (h, w) != (MAX_DIM, MAX_DIM):  # exit: back to canvas, sentinel -1 padding
        src = g.add("Pad", {"attrs": {"pads": [0, 0, 0, 0, 0, 0, MAX_DIM - h, MAX_DIM - w],
                                      "value": -1}}, [(src, "input")])
    g.add("Output", {"shape": "1,1,30,30"}, [(src, "input")])

    return {
        "projectName": f"doors-{task_id}",
        "taskId": task_id,
        "nodes": g.nodes,
        "edges": g.edges,
        "representation": "scalar",
        "bridgeMask": "input" if cellwise_only else "none",
    }
