"""backends.py — find which (scattered) Python env can run ONNX on the Intel Arc B70 Pro,
and auto-select it for ngs. Replaces the stale ml/README assumptions (.venv-ml/.venv-dml,
A770) with live discovery of the real envs (Q:/AIz/*, the project venv, ...).

Why this exists: the ngs search is ORT-inference-bound and the policy trainer needs torch;
the machine's envs are scattered and only some see the B70 via an ONNX GPU execution provider
(OpenVINO EP / DirectML EP) or torch-xpu. This probes each env in its own subprocess (different
runtimes can't share a process — same reason verify_ml.py forks) and writes a machine-readable
report other ml/ code reads instead of re-probing.

  .venv/Scripts/python.exe -m ml.ngs.backends           # scan, probe, print + write _env/backends.json
  .venv/Scripts/python.exe -m ml.ngs.backends --roots "Q:/AIz" "C:/other"   # extra scan roots
"""
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
OUT = HERE.parent / "_env"
DEFAULT_ROOTS = [str(REPO_ROOT), "Q:/AIz", "Q:/AIz/Intel_Arc_Stuff", "Q:/AIz/LLM"]

# preference order for an ONNX *GPU* execution provider (first available wins)
GPU_EPS = ["OpenVINOExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"]

# Self-contained probe run inside each discovered interpreter; prints one JSON line.
PROBE_SRC = r"""
import json, sys
o = {"python": sys.version.split()[0], "exe": sys.executable}
try:
    import onnxruntime as ort
    o["onnxruntime"] = ort.__version__
    o["ort_providers"] = list(ort.get_available_providers())
except Exception as e:
    o["ort_providers"] = []; o["ort_err"] = str(e)[:140]
try:
    import onnx as _onnx; o["onnx"] = _onnx.__version__
except Exception:
    o["onnx"] = None
try:
    import openvino as ov
    o["openvino"] = ov.__version__
    c = ov.Core(); o["ov_devices"] = list(c.available_devices); o["ov_gpu"] = []
    for d in o["ov_devices"]:
        if d.startswith("GPU"):
            try: o["ov_gpu"].append(c.get_property(d, "FULL_DEVICE_NAME"))
            except Exception: o["ov_gpu"].append(d)
except Exception as e:
    o["openvino"] = None; o["ov_err"] = str(e)[:140]
try:
    import torch
    o["torch"] = torch.__version__
    o["torch_xpu"] = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    if o.get("torch_xpu"):
        try: o["torch_xpu_name"] = torch.xpu.get_device_name(0)
        except Exception: pass
except Exception as e:
    o["torch"] = None; o["torch_err"] = str(e)[:140]
print(json.dumps(o))
"""


def discover_envs(roots, max_depth=6, limit=40):
    """Return python interpreter paths found under the given roots (+ the current one)."""
    found = {sys.executable}
    for root in roots:
        rp = pathlib.Path(root)
        if not rp.exists():
            continue
        base = len(rp.parts)
        for p in rp.rglob("python.exe"):
            try:
                if len(p.parts) - base > max_depth:
                    continue
                # only venv/conda-style interpreters (Scripts/python.exe or bin/python or a python dir)
                if p.parent.name.lower() in ("scripts", "bin") or (p.parent / "Lib").exists():
                    found.add(str(p))
            except Exception:
                continue
            if len(found) >= limit:
                break
    return sorted(found)


def probe(py, timeout=40):
    try:
        r = subprocess.run([py, "-c", PROBE_SRC], capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        line = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
        info = json.loads(line)
    except Exception as e:
        info = {"exe": py, "probe_err": f"{type(e).__name__}: {str(e)[:120]}"}
    info["env_path"] = py
    return info


def _has_gpu_ep(info):
    return [p for p in GPU_EPS if p in info.get("ort_providers", [])]


def _sees_b70(info):
    blob = " ".join(info.get("ov_gpu", []) + [info.get("torch_xpu_name", "")]).lower()
    return ("b70" in blob) or ("arc pro" in blob) or ("battlemage" in blob)


def scan(roots=None):
    roots = roots or DEFAULT_ROOTS
    envs = discover_envs(roots)
    report = {"roots": roots, "n_envs": len(envs), "envs": []}
    for py in envs:
        print(f"  probing {py} ...", flush=True)
        report["envs"].append(probe(py))
    report["best_ort_gpu"] = best_ort_env(report)
    report["best_torch"] = best_torch_env(report)
    report["best_ngs_env"] = best_ngs_env(report)
    report["best_openvino_env"] = best_openvino_env(report)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backends.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def best_ort_env(report):
    """Pick the env+EP that can run ONNX on the GPU (prefers one that sees the B70)."""
    best = None
    for e in report["envs"]:
        eps = _has_gpu_ep(e)
        if not eps:
            continue
        score = (10 if _sees_b70(e) else 0) + (5 if eps[0] == "OpenVINOExecutionProvider" else 0)
        if best is None or score > best[0]:
            best = (score, {"env_path": e["env_path"], "provider": eps[0], "sees_b70": _sees_b70(e)})
    return best[1] if best else None


def best_torch_env(report):
    """Pick the env for training the policy (prefers torch-xpu on the B70)."""
    best = None
    for e in report["envs"]:
        if not e.get("torch"):
            continue
        score = (10 if e.get("torch_xpu") else 0) + (5 if _sees_b70(e) else 0)
        if best is None or score > best[0]:
            best = (score, {"env_path": e["env_path"], "torch": e["torch"],
                            "xpu": bool(e.get("torch_xpu")), "device": e.get("torch_xpu_name", "cpu")})
    return best[1] if best else None


def best_ngs_env(report):
    """The one-stop env that can run the WHOLE ngs pipeline on the B70: onnx + onnxruntime
    (build/run candidate graphs) AND torch (train the policy), prefer torch-xpu on the B70."""
    best = None
    for e in report["envs"]:
        if not (e.get("onnx") and e.get("onnxruntime") and e.get("torch")):
            continue
        score = (10 if e.get("torch_xpu") else 0) + (5 if _sees_b70(e) else 0)
        if best is None or score > best[0]:
            best = (score, {"env_path": e["env_path"], "torch": e["torch"],
                            "xpu": bool(e.get("torch_xpu")), "device": e.get("torch_xpu_name", "cpu"),
                            "onnxruntime": e.get("onnxruntime")})
    return best[1] if best else None


def best_openvino_env(report):
    """Env with the OpenVINO runtime that sees the B70 dGPU — for ov_eval (ONNX inference on the B70)."""
    best = None
    for e in report["envs"]:
        if not e.get("openvino"):
            continue
        has_gpu = any(d.startswith("GPU") for d in e.get("ov_devices", []))
        score = (10 if _sees_b70(e) else 0) + (5 if has_gpu else 0)
        if best is None or score > best[0]:
            best = (score, {"env_path": e["env_path"], "openvino": e["openvino"],
                            "devices": e.get("ov_devices", []), "gpu": e.get("ov_gpu", [])})
    return best[1] if best else None


def load_report():
    p = OUT / "backends.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _print(report):
    print(f"\n== backend scan: {report['n_envs']} env(s) ==")
    for e in report["envs"]:
        tags = []
        if e.get("onnxruntime"):
            gpu = _has_gpu_ep(e)
            tags.append(f"ort{e['onnxruntime']}[{'+'.join(p.replace('ExecutionProvider','') for p in gpu) or 'CPU'}]")
        if e.get("openvino"):
            tags.append(f"ov{e['openvino']}{e.get('ov_gpu') or ''}")
        if e.get("torch"):
            tags.append(f"torch{e['torch']}{'+xpu' if e.get('torch_xpu') else ''}")
        if e.get("probe_err"):
            tags.append("PROBE_ERR")
        flag = "  <-- sees B70" if _sees_b70(e) else ""
        print(f"  {e['env_path']}\n      {', '.join(tags) or '(no ML stack)'}{flag}")
    print("\n  best ONNX-GPU env :", report.get("best_ort_gpu") or "NONE (search runs CPU-only here)")
    print("  best torch env    :", report.get("best_torch") or "NONE (policy training unavailable)")
    print("  best ngs env (B70):", report.get("best_ngs_env") or "NONE (no single env has onnx+ort+torch)")
    print("  best OpenVINO env :", report.get("best_openvino_env") or "NONE (no OpenVINO+GPU env)")
    print(f"\n  report -> {OUT/'backends.json'}")


def main():
    roots = None
    if "--roots" in sys.argv:
        roots = sys.argv[sys.argv.index("--roots") + 1:]
    _print(scan(roots))
    return 0


if __name__ == "__main__":
    sys.exit(main())
