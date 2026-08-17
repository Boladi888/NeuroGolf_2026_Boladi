#!/usr/bin/env python3
"""Render standardized ARC/NeuroGolf task preview PNGs.

The output is a compact contact sheet for rule discovery:
- all train pairs by default
- all test pairs by default, including outputs when present
- a deterministic spread sample from arc-gen pairs

No third-party image dependency is required. The renderer writes PNG files using
only the Python standard library so it works in a fresh GUI checkout.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Iterable


ARC_COLORS = [
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
]

LIGHT_DIGIT_COLORS = {3, 4, 5, 7, 8}

BG = (248, 250, 252)
PANEL_BG = (255, 255, 255)
TEXT = (15, 23, 42)
MUTED = (71, 85, 105)
BORDER = (203, 213, 225)
GRID_LINE = (148, 163, 184)
SECTION_BG = (226, 232, 240)
ACCENT = (15, 118, 110)


FONT_5X7 = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ">": ["10000", "01000", "00100", "00010", "00100", "01000", "10000"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    ",": ["00000", "00000", "00000", "00000", "00100", "00100", "01000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "|": ["00100", "00100", "00100", "00100", "00100", "00100", "00100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render task preview PNG contact sheets.")
    parser.add_argument("tasks", nargs="*", help="Task ids or numbers, e.g. task001 79 250.")
    parser.add_argument("--all", action="store_true", help="Render every task JSON found.")
    parser.add_argument("--tasks-dir", default="client/public/tasks", help="Directory containing taskNNN.json files.")
    parser.add_argument("--out-dir", default="previews", help="Directory for taskNNN_preview.png outputs.")
    parser.add_argument("--arc-gen", type=int, default=4, help="How many arc-gen examples to show.")
    parser.add_argument("--max-train", type=int, default=99, help="Maximum train pairs to show.")
    parser.add_argument("--max-test", type=int, default=99, help="Maximum test pairs to show.")
    parser.add_argument("--sample-strategy", choices=("spread", "first"), default="spread")
    parser.add_argument("--max-grid-px", type=int, default=360, help="Largest grid side in pixels.")
    parser.add_argument("--min-cell", type=int, default=6)
    parser.add_argument("--max-cell", type=int, default=28)
    parser.add_argument("--hide-test-output", action="store_true")
    parser.add_argument("--no-digits", action="store_true", help="Disable readable cell digits.")
    parser.add_argument("--digits-zero", action="store_true", help="Also draw the digit 0 on background cells.")
    return parser.parse_args()


class Canvas:
    def __init__(self, width: int, height: int, color: tuple[int, int, int] = BG):
        self.width = width
        self.height = height
        self.rows = [bytearray(color * width) for _ in range(height)]

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        if w <= 0 or h <= 0:
            return
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        rgb = bytes(color)
        for row in self.rows[y0:y1]:
            start = x0 * 3
            row[start : x1 * 3] = rgb * (x1 - x0)

    def stroke(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int], thickness: int = 1) -> None:
        for offset in range(thickness):
            self.rect(x + offset, y + offset, w - 2 * offset, 1, color)
            self.rect(x + offset, y + h - 1 - offset, w - 2 * offset, 1, color)
            self.rect(x + offset, y + offset, 1, h - 2 * offset, color)
            self.rect(x + w - 1 - offset, y + offset, 1, h - 2 * offset, color)

    def text(self, x: int, y: int, text: str, color: tuple[int, int, int] = TEXT, scale: int = 2) -> None:
        cx = x
        for char in text.upper():
            glyph = FONT_5X7.get(char, FONT_5X7[" "])
            if char == " ":
                cx += 4 * scale
                continue
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cx + gx * scale, y + gy * scale, scale, scale, color)
            cx += 6 * scale

    def png_bytes(self) -> bytes:
        raw = b"".join(b"\x00" + bytes(row) for row in self.rows)
        compressed = zlib.compress(raw, 9)

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b"")
        )


def text_width(text: str, scale: int = 2) -> int:
    total = 0
    for char in text.upper():
        total += (4 if char == " " else 6) * scale
    return total


def normalize_task_id(value: str) -> str:
    raw = value.strip().lower()
    match = re.fullmatch(r"(?:task)?0*(\d{1,3})", raw)
    if not match:
        raise ValueError(f"Bad task id: {value!r}")
    number = int(match.group(1))
    if number < 1 or number > 400:
        raise ValueError(f"Task number out of range: {number}")
    return f"task{number:03d}"


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return rows, cols


def pair_shapes(pair: dict) -> str:
    ih, iw = grid_shape(pair["input"])
    if "output" not in pair:
        return f"{ih}X{iw}"
    oh, ow = grid_shape(pair["output"])
    return f"{ih}X{iw}->{oh}X{ow}"


def all_grids(task: dict) -> Iterable[list[list[int]]]:
    for set_name in ("train", "test", "arc-gen"):
        for pair in task.get(set_name, []):
            if "input" in pair:
                yield pair["input"]
            if "output" in pair:
                yield pair["output"]


def colors_used(task: dict) -> list[int]:
    colors: set[int] = set()
    for grid in all_grids(task):
        for row in grid:
            colors.update(int(value) for value in row)
    return sorted(color for color in colors if 0 <= color <= 9)


def choose_cell_size(task: dict, args: argparse.Namespace) -> int:
    max_dim = 1
    for grid in all_grids(task):
        h, w = grid_shape(grid)
        max_dim = max(max_dim, h, w)
    cell = args.max_grid_px // max_dim
    return max(args.min_cell, min(args.max_cell, cell))


def pick_examples(examples: list[dict], limit: int, strategy: str) -> list[tuple[int, dict]]:
    if limit <= 0 or not examples:
        return []
    if len(examples) <= limit:
        return list(enumerate(examples))
    if strategy == "first":
        return list(enumerate(examples[:limit]))
    if limit == 1:
        return [(0, examples[0])]
    indexes = sorted({round(i * (len(examples) - 1) / (limit - 1)) for i in range(limit)})
    return [(index, examples[index]) for index in indexes]


def grid_pixel_size(grid: list[list[int]], cell: int) -> tuple[int, int]:
    h, w = grid_shape(grid)
    return w * cell + 1, h * cell + 1


def render_grid(
    canvas: Canvas,
    x: int,
    y: int,
    grid: list[list[int]],
    cell: int,
    show_digits: bool,
    show_zero_digits: bool,
) -> tuple[int, int]:
    h, w = grid_shape(grid)
    width = w * cell + 1
    height = h * cell + 1
    canvas.rect(x, y, width, height, GRID_LINE)
    digit_scale = max(1, cell // 9)
    digits = show_digits and cell >= 12
    for r, row in enumerate(grid):
        for c, raw_value in enumerate(row):
            value = int(raw_value)
            color = ARC_COLORS[value] if 0 <= value < len(ARC_COLORS) else (30, 41, 59)
            px = x + c * cell + 1
            py = y + r * cell + 1
            canvas.rect(px, py, cell - 1, cell - 1, color)
            if digits and (value != 0 or show_zero_digits):
                fg = TEXT if value in LIGHT_DIGIT_COLORS else (248, 250, 252)
                label = str(value)
                tw = text_width(label, digit_scale)
                tx = px + max(1, (cell - 1 - tw) // 2)
                ty = py + max(1, (cell - 1 - 7 * digit_scale) // 2)
                canvas.text(tx, ty, label, fg, digit_scale)
    return width, height


def pair_dimensions(pair: dict, cell: int, include_output: bool) -> tuple[int, int]:
    iw, ih = grid_pixel_size(pair["input"], cell)
    if include_output and "output" in pair:
        ow, oh = grid_pixel_size(pair["output"], cell)
        return iw + 34 + ow, max(ih, oh)
    return iw, ih


def draw_pair(
    canvas: Canvas,
    x: int,
    y: int,
    label: str,
    pair: dict,
    cell: int,
    include_output: bool,
    show_digits: bool,
    show_zero_digits: bool,
) -> int:
    canvas.text(x, y, label, MUTED, 2)
    y_grid = y + 22
    iw, ih = render_grid(canvas, x, y_grid, pair["input"], cell, show_digits, show_zero_digits)
    if include_output and "output" in pair:
        arrow_x = x + iw + 10
        arrow_y = y_grid + max(8, ih // 2 - 7)
        canvas.text(arrow_x, arrow_y, "->", MUTED, 2)
        render_grid(canvas, x + iw + 34, y_grid, pair["output"], cell, show_digits, show_zero_digits)
    _, content_h = pair_dimensions(pair, cell, include_output)
    return 22 + content_h


def section_rows(
    title: str,
    examples: list[tuple[int, dict]],
    cell: int,
    include_output: bool,
    max_content_width: int,
) -> tuple[str, list[list[tuple[int, dict]]]]:
    rows: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    current_w = 0
    gap = 22
    for item in examples:
        _, pair = item
        item_w, _ = pair_dimensions(pair, cell, include_output and "output" in pair)
        item_w = max(item_w, 126)
        next_w = item_w if not current else current_w + gap + item_w
        if current and next_w > max_content_width:
            rows.append(current)
            current = [item]
            current_w = item_w
        else:
            current.append(item)
            current_w = next_w
    if current:
        rows.append(current)
    return title, rows


def render_section(
    canvas: Canvas,
    y: int,
    title: str,
    rows: list[list[tuple[int, dict]]],
    set_name: str,
    cell: int,
    include_output: bool,
    show_digits: bool,
    show_zero_digits: bool,
    margin: int,
    width: int,
) -> int:
    if not rows:
        return y
    canvas.rect(margin, y, width - 2 * margin, 24, SECTION_BG)
    canvas.text(margin + 10, y + 6, title, TEXT, 2)
    y += 38
    gap_x = 22
    gap_y = 22
    for row in rows:
        x = margin
        row_h = 0
        for index, pair in row:
            content_include_output = include_output and "output" in pair
            label = f"{set_name}{index + 1} {pair_shapes(pair)}"
            panel_w, panel_h = pair_dimensions(pair, cell, content_include_output)
            panel_w = max(panel_w, text_width(label, 2))
            canvas.rect(x - 8, y - 8, panel_w + 16, panel_h + 44, PANEL_BG)
            canvas.stroke(x - 8, y - 8, panel_w + 16, panel_h + 44, BORDER)
            used_h = draw_pair(
                canvas,
                x,
                y,
                label,
                pair,
                cell,
                content_include_output,
                show_digits,
                show_zero_digits,
            )
            x += panel_w + 16 + gap_x
            row_h = max(row_h, used_h + 16)
        y += row_h + gap_y
    return y + 2


def estimate_height(sections: list[tuple[str, list[list[tuple[int, dict]]], str, bool]], cell: int) -> int:
    height = 106
    for _, rows, _, include_output in sections:
        if not rows:
            continue
        height += 38
        for row in rows:
            row_h = 0
            for _, pair in row:
                _, pair_h = pair_dimensions(pair, cell, include_output and "output" in pair)
                row_h = max(row_h, pair_h + 44)
            height += row_h + 22
        height += 2
    return height + 24


def render_preview(task_id: str, task: dict, args: argparse.Namespace) -> bytes:
    margin = 28
    width = 1600
    content_width = width - 2 * margin
    cell = choose_cell_size(task, args)
    show_digits = not args.no_digits
    show_zero_digits = args.digits_zero

    train = list(enumerate(task.get("train", [])[: args.max_train]))
    test = list(enumerate(task.get("test", [])[: args.max_test]))
    arc_gen = pick_examples(task.get("arc-gen", []), args.arc_gen, args.sample_strategy)

    section_specs = [
        section_rows(f"TRAIN PAIRS ({len(task.get('train', []))})", train, cell, True, content_width),
        section_rows(f"TEST PAIRS ({len(task.get('test', []))})", test, cell, not args.hide_test_output, content_width),
        section_rows(
            f"ARC-GEN SAMPLE ({len(arc_gen)} OF {len(task.get('arc-gen', []))})",
            arc_gen,
            cell,
            True,
            content_width,
        ),
    ]
    sections = [
        (title, rows, set_name, include_output)
        for (title, rows), set_name, include_output in zip(
            section_specs,
            ("TR", "TE", "AG"),
            (True, not args.hide_test_output, True),
        )
    ]

    height = estimate_height(sections, cell)
    canvas = Canvas(width, height)
    canvas.text(margin, 24, f"{task_id} PREVIEW", TEXT, 3)
    meta = (
        f"TRAIN {len(task.get('train', []))} | TEST {len(task.get('test', []))} | "
        f"ARC-GEN {len(task.get('arc-gen', []))} | CELL {cell}PX"
    )
    canvas.text(margin, 58, meta, MUTED, 2)

    used_colors = set(colors_used(task))
    palette_x = width - margin - 10 * 34
    canvas.text(palette_x, 24, "PALETTE", MUTED, 2)
    for idx, color in enumerate(ARC_COLORS):
        x = palette_x + idx * 34
        canvas.rect(x, 48, 24, 24, color)
        canvas.stroke(x, 48, 24, 24, GRID_LINE)
        label_color = ACCENT if idx in used_colors else MUTED
        canvas.text(x + 8, 76, str(idx), label_color, 1)

    y = 106
    for title, rows, set_name, include_output in sections:
        y = render_section(
            canvas,
            y,
            title,
            rows,
            set_name,
            cell,
            include_output,
            show_digits,
            show_zero_digits,
            margin,
            width,
        )
    return canvas.png_bytes()


def task_paths(args: argparse.Namespace) -> list[Path]:
    tasks_dir = Path(args.tasks_dir)
    if args.all:
        return sorted(tasks_dir.glob("task[0-9][0-9][0-9].json"))
    if not args.tasks:
        raise SystemExit("Specify task ids/numbers or pass --all.")
    return [tasks_dir / f"{normalize_task_id(value)}.json" for value in args.tasks]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in task_paths(args):
        if not path.is_file():
            raise FileNotFoundError(path)
        task_id = path.stem
        task = json.loads(path.read_text(encoding="utf-8"))
        png = render_preview(task_id, task, args)
        out_path = out_dir / f"{task_id}_preview.png"
        out_path.write_bytes(png)
        written.append(out_path)
        print(f"wrote {out_path.as_posix()}")
    print(f"rendered {len(written)} preview(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
