#!/usr/bin/env python3
"""Solid GUI window for the read-only ng_gate queue watcher."""
from __future__ import annotations

import argparse
import contextlib
import io
import queue
import sys
import threading
import time
import traceback

import ng_gate_queue_snapshot


def _render(max_rescue: int, state_days: float) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ng_gate_queue_snapshot.print_human(
            ng_gate_queue_snapshot.gather(max_rescue=max_rescue, state_days=state_days)
        )
    return buf.getvalue()


def _run_once(max_rescue: int, state_days: float) -> int:
    print(_render(max_rescue, state_days), end="")
    return 0


def _run_gui(interval: float, max_rescue: int, state_days: float) -> int:
    try:
        import tkinter as tk
        from tkinter import font as tkfont
    except Exception as e:  # noqa: BLE001
        print(f"tkinter unavailable: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    work_q: queue.Queue[tuple[str, str]] = queue.Queue()
    stop = threading.Event()
    refresh_running = threading.Event()

    root = tk.Tk()
    root.title("NG Gate Queue Watch - close window to stop watcher")
    root.geometry("1280x760")
    root.configure(bg="#0b0d10")

    mono = tkfont.Font(family="Consolas", size=10)
    header = tk.Label(
        root,
        text="NG Gate Queue Watch",
        anchor="w",
        bg="#111820",
        fg="#f2f6ff",
        padx=12,
        pady=8,
        font=("Segoe UI", 13, "bold"),
    )
    header.pack(side="top", fill="x")

    status = tk.Label(
        root,
        text="Starting read-only watcher...",
        anchor="w",
        bg="#0b0d10",
        fg="#9fb0c7",
        padx=12,
        pady=5,
        font=("Segoe UI", 9),
    )
    status.pack(side="top", fill="x")

    body = tk.Text(
        root,
        bg="#05070a",
        fg="#e8edf7",
        insertbackground="#e8edf7",
        selectbackground="#274360",
        relief="flat",
        padx=12,
        pady=10,
        wrap="none",
        font=mono,
    )
    body.pack(side="top", fill="both", expand=True)
    body.insert("1.0", "Loading ng_gate queue snapshot...\n")
    body.configure(state="disabled")

    footer = tk.Frame(root, bg="#111820")
    footer.pack(side="bottom", fill="x")

    def request_refresh(force: bool = False) -> None:
        if refresh_running.is_set() and not force:
            return
        refresh_running.set()

        def worker() -> None:
            started = time.time()
            try:
                text = _render(max_rescue=max_rescue, state_days=state_days)
                work_q.put(("ok", text + f"\nSnapshot render: {time.time() - started:.1f}s\n"))
            except Exception:  # noqa: BLE001
                work_q.put(("error", traceback.format_exc()))
            finally:
                refresh_running.clear()

        threading.Thread(target=worker, name="ng-gate-queue-refresh", daemon=True).start()

    def poll_queue() -> None:
        try:
            while True:
                kind, text = work_q.get_nowait()
                body.configure(state="normal")
                body.delete("1.0", "end")
                body.insert("1.0", text)
                body.configure(state="disabled")
                status.configure(
                    text=("Last refresh OK. " if kind == "ok" else "Last refresh failed. ")
                    + f"Next refresh every {interval:g}s. Close this window to stop watcher only.",
                    fg="#84d79a" if kind == "ok" else "#ff8f8f",
                )
        except queue.Empty:
            pass
        if not stop.is_set():
            root.after(300, poll_queue)

    def timed_refresh() -> None:
        if stop.is_set():
            return
        request_refresh()
        root.after(int(max(2.0, interval) * 1000), timed_refresh)

    def close() -> None:
        stop.set()
        root.destroy()

    refresh_btn = tk.Button(
        footer,
        text="Refresh Now",
        command=lambda: request_refresh(force=True),
        bg="#1d2a38",
        fg="#ffffff",
        activebackground="#26384a",
        activeforeground="#ffffff",
        relief="flat",
        padx=12,
        pady=6,
    )
    refresh_btn.pack(side="left", padx=10, pady=8)

    close_btn = tk.Button(
        footer,
        text="Close Watcher",
        command=close,
        bg="#3a2020",
        fg="#ffffff",
        activebackground="#4a2828",
        activeforeground="#ffffff",
        relief="flat",
        padx=12,
        pady=6,
    )
    close_btn.pack(side="left", padx=2, pady=8)

    note = tk.Label(
        footer,
        text="Read-only: does not start, stop, kill, or reprioritize gate runs.",
        anchor="e",
        bg="#111820",
        fg="#9fb0c7",
        padx=12,
    )
    note.pack(side="right", fill="x", expand=True)

    root.protocol("WM_DELETE_WINDOW", close)
    request_refresh(force=True)
    root.after(300, poll_queue)
    root.after(int(max(2.0, interval) * 1000), timed_refresh)
    root.mainloop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Solid GUI ng_gate queue watcher.")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--max-rescue", type=int, default=6)
    ap.add_argument("--state-days", type=float, default=7.0)
    ap.add_argument("--once", action="store_true", help="render once to stdout without opening a GUI")
    args = ap.parse_args()
    if args.once:
        return _run_once(args.max_rescue, args.state_days)
    return _run_gui(args.interval, args.max_rescue, args.state_days)


if __name__ == "__main__":
    raise SystemExit(main())
