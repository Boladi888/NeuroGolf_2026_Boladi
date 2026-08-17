"""verify_ml.py — exercise the GPU ML backends installed for this machine.

Run from whichever ML venv you want to test. The script self-detects which
backends import cleanly and runs an end-to-end check on each:

  - OpenVINO          : train a tiny MLP on PyTorch CPU, export ONNX, convert
                        to OpenVINO IR, run on the Arc GPU, verify outputs
                        match the PyTorch reference within tolerance.
  - PyTorch XPU       : train an MLP for 30 epochs on the Arc, show loss
                        trajectory and per-epoch timing.
  - PyTorch DirectML  : same training, on the DX12 device.

Exit code 0 if every available backend passes, 1 otherwise. Backends that
aren't installed in the current venv are skipped (not failed).

Each backend runs in its own freshly-spawned subprocess. This is required
because OpenVINO's GPU plugin and torch+xpu's SYCL runtime both grab the
Arc and deadlock if loaded into the same Python process — observed
symptom is the second backend's first GPU op hangs forever after oneDNN
init logs.

Usage:
  .\\.venv-ml\\Scripts\\python.exe verify_ml.py                 # all available
  .\\.venv-dml\\Scripts\\python.exe verify_ml.py
  python verify_ml.py --only openvino|xpu|directml              # single backend
"""

import sys
import time
from pathlib import Path

# torch 2.11's new ONNX exporter prints a unicode checkmark; Windows' default
# cp1252 codepage chokes on it. Force UTF-8 so any backend's prints survive.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

INPUT_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 16
N_SAMPLES = 2048
BATCH = 256
EPOCHS = 30
LR = 1e-3


def section(name):
    print()
    print("=" * 60)
    print(f"  {name}")
    print("=" * 60)


def make_data(seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_SAMPLES, INPUT_DIM)).astype("float32")
    s = x.sum(axis=1, keepdims=True)
    base = np.sin(s + np.arange(OUTPUT_DIM)[None, :].astype("float32") * 0.3)
    noise = rng.standard_normal((N_SAMPLES, OUTPUT_DIM)).astype("float32") * 0.05
    return x, (base + noise).astype("float32")


def torch_training(device, sync_fn=None):
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    x_np, y_np = make_data()
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    model = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    losses = []
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        perm = torch.randperm(N_SAMPLES, device=device)
        epoch_loss, nb = 0.0, 0
        for i in range(0, N_SAMPLES, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            pred = model(x[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            nb += 1
        losses.append(epoch_loss / nb)
    if sync_fn:
        sync_fn()
    dt = time.perf_counter() - t0

    print(f"  initial loss: {losses[0]:.4f}")
    print(f"  final   loss: {losses[-1]:.4f}")
    print(f"  reduction:    {(1 - losses[-1] / losses[0]) * 100:.1f}%")
    print(f"  total time:   {dt:.2f}s  ({dt / EPOCHS * 1000:.1f} ms/epoch)")
    if losses[-1] > losses[0] * 0.5:
        return False, f"loss only dropped from {losses[0]:.3f} to {losses[-1]:.3f}"
    return True, None


def test_xpu():
    section("PyTorch + XPU (Intel Arc training)")
    try:
        import torch
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            print("  XPU not available — skipping (this is .venv-dml or torch lacks XPU build)")
            return None
        print(f"  torch:           {torch.__version__}")
        print(f"  device:          {torch.xpu.get_device_name(0)}")
        print(f"  bf16 supported:  {torch.xpu.is_bf16_supported()}")
    except ImportError as e:
        print(f"  torch import failed — skipping ({e})")
        return None
    return torch_training(torch.device("xpu"), sync_fn=torch.xpu.synchronize)


def test_directml():
    section("PyTorch + DirectML (DX12 fallback)")
    try:
        import torch
        import torch_directml as dml
        print(f"  torch:           {torch.__version__}")
        print(f"  device:          {dml.device_name(0)}")
    except ImportError as e:
        print(f"  torch_directml not installed — skipping ({e})")
        return None
    return torch_training(dml.device(), sync_fn=None)


def test_openvino():
    section("OpenVINO (ONNX round-trip + Arc GPU inference)")
    try:
        import openvino as ov
        import numpy as np
    except ImportError as e:
        print(f"  openvino not installed — skipping ({e})")
        return None
    print(f"  openvino:        {ov.__version__}")
    core = ov.Core()
    devs = core.available_devices
    print(f"  devices:         {devs}")
    if "GPU" not in devs:
        return False, "OpenVINO does not see the Arc GPU"

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  torch not in this venv — can't do ONNX export round-trip")
        print("  (still confirmed device enumeration above)")
        return True, None

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    ).eval()

    onnx_path = Path.cwd() / "_verify_ml_tmp.onnx"
    try:
        # dynamo=False forces the legacy exporter (torch.onnx.utils._export):
        # the new dynamo-based exporter prints unicode emoji to stdout (broken
        # on Windows cp1252) and pulls in a slow torch.export tracing path.
        torch.onnx.export(
            model, (torch.randn(1, INPUT_DIM),), str(onnx_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=14,
            dynamo=False,
        )
        print(f"  ONNX export:     {onnx_path.stat().st_size} bytes")

        ov_model = core.read_model(str(onnx_path))
        compiled_gpu = core.compile_model(ov_model, "GPU")
        compiled_cpu = core.compile_model(ov_model, "CPU")
        print("  compiled to GPU and CPU")

        x_test = torch.randn(8, INPUT_DIM).numpy()
        torch_out = model(torch.from_numpy(x_test)).detach().numpy()
        gpu_out = compiled_gpu([x_test])[0]
        cpu_out = compiled_cpu([x_test])[0]
        gpu_diff = float(np.abs(torch_out - gpu_out).max())
        cpu_diff = float(np.abs(torch_out - cpu_out).max())
        print(f"  max |torch - ov_cpu|: {cpu_diff:.6f}")
        print(f"  max |torch - ov_gpu|: {gpu_diff:.6f}")

        N_ITER = 100
        compiled_gpu([x_test])  # warm
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            compiled_gpu([x_test])
        dt = (time.perf_counter() - t0) / N_ITER * 1000
        print(f"  GPU inference:   {dt:.2f} ms / batch of 8 (avg of {N_ITER})")

        if gpu_diff > 1e-3:
            return False, f"GPU output diverges from torch reference ({gpu_diff})"
        return True, None
    finally:
        if onnx_path.exists():
            onnx_path.unlink()


BACKENDS = {
    "openvino": ("OpenVINO",         test_openvino),
    "xpu":      ("PyTorch XPU",      test_xpu),
    "directml": ("PyTorch DirectML", test_directml),
}


def run_one(key):
    """Run a single backend test in this process. Used by --only and by the
    subprocesses the orchestrator spawns."""
    label, fn = BACKENDS[key]
    print(f"verify_ml.py [{key}] — running in {Path(sys.executable).parent.parent.name}")
    print(f"python: {sys.version.split()[0]}")
    try:
        result = fn()
    except Exception as e:
        print(f"  ! {label} CRASHED: {type(e).__name__}: {e}")
        result = (False, f"{type(e).__name__}: {e}")
    section(f"{label} — RESULT")
    if result is None:
        print(f"  -  SKIPPED (not in this venv)")
        sys.exit(2)
    elif result[0]:
        print(f"  +  PASS")
        sys.exit(0)
    else:
        print(f"  X  FAIL ({result[1]})")
        sys.exit(1)


def orchestrate():
    """Run each backend test in its own freshly-spawned python subprocess so
    OpenVINO and torch+xpu cannot share a process (they deadlock the Arc
    when loaded together)."""
    import subprocess
    print(f"verify_ml.py — orchestrating in {Path(sys.executable).parent.parent.name}")
    print(f"python: {sys.version.split()[0]}")
    print("(each backend runs in its own subprocess for runtime isolation)")

    results = {}
    for key, (label, _) in BACKENDS.items():
        section(f"{label}")
        rc = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--only", key],
            env={**__import__("os").environ,
                 "PYTHONIOENCODING": "utf-8",
                 "SYCL_CACHE_PERSISTENT": "1"},
        ).returncode
        # rc: 0=PASS, 1=FAIL, 2=SKIPPED, other=crash (segfault etc.)
        if rc == 0:
            results[label] = ("PASS", None)
        elif rc == 1:
            results[label] = ("FAIL", "see output above")
        elif rc == 2:
            results[label] = ("SKIP", "not installed in this venv")
        else:
            results[label] = ("CRASH", f"exit {rc} (likely native abort)")

    section("RESULTS")
    any_failed = False
    for label, (verdict, detail) in results.items():
        marker = {"PASS": "+", "SKIP": "-", "FAIL": "X", "CRASH": "!"}[verdict]
        line = f"  {marker}  {label:20s} {verdict}"
        if detail and verdict != "PASS":
            line += f"  ({detail})"
        print(line)
        if verdict in ("FAIL", "CRASH"):
            any_failed = True
    sys.exit(1 if any_failed else 0)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--only":
        key = sys.argv[2].lower()
        if key not in BACKENDS:
            print(f"unknown backend '{key}'. choices: {sorted(BACKENDS)}", file=sys.stderr)
            sys.exit(64)
        run_one(key)
    elif len(sys.argv) == 1:
        orchestrate()
    else:
        print("usage: verify_ml.py [--only openvino|xpu|directml]", file=sys.stderr)
        sys.exit(64)


if __name__ == "__main__":
    main()
