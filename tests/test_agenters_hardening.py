import hashlib
import importlib.util
from pathlib import Path


def _load_module(name, relpath):
    path = Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_run_audit_suggests_sha_matching_task_local_staged_candidate(tmp_path):
    audit = _load_module("run_audit_hardening_test", "Agenters/Orchestrator/run_audit.py")
    data = b"candidate-bytes"
    candidate = (
        tmp_path / "The_400_Tasks" / "Task_by_Task" / "task295" /
        "staged_candidates" / "task295_packscalar_cost1539" / "task295.onnx"
    )
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(data)

    hint = audit.staged_path_hint(
        tmp_path,
        "The_400_Tasks/staged_submits/task295_packscalar_cost1539/task295.onnx",
        _sha(data),
        "task295",
    ).replace("\\", "/")

    assert (
        "SHA-matching candidate exists at "
        "The_400_Tasks/Task_by_Task/task295/staged_candidates/"
        "task295_packscalar_cost1539/task295.onnx"
    ) in hint


def test_captain_classify_requires_probe_sha_bound_to_current_bytes(tmp_path, monkeypatch):
    captain = _load_module("captain_ctl_hardening_test", "Agenters/Submit_Captain/captain_ctl.py")
    monkeypatch.setattr(captain, "HOME", tmp_path)
    data = b"not-a-real-onnx-but-classify-only-checks-the-hash"
    onnx = tmp_path / "candidate.onnx"
    onnx.write_bytes(data)
    sha = _sha(data)
    pkt = {
        "task": "task001",
        "source_lane": "Codex",
        "onnx": "candidate.onnx",
        "cost": 90,
        "sha256": sha,
        "probe_ok": True,
        "probe_sha": sha,
        # STACK-ONLY GATE (captain_ctl.py:532): packets without a stack/local-truth
        # probe_kind are HELD as un-stacked, so the happy path must declare one.
        "probe_kind": "stack",
        "candidate_status": "confirmed_public",
    }

    assert captain.classify(dict(pkt), {"task001": 100}) == ("READY", None)

    missing_probe_sha = dict(pkt)
    missing_probe_sha.pop("probe_sha")
    state, note = captain.classify(missing_probe_sha, {"task001": 100})
    assert state == "HELD"
    assert "missing probe_sha" in note

    stale_probe = dict(pkt, probe_sha="0" * 64)
    state, note = captain.classify(stale_probe, {"task001": 100})
    assert state == "HELD"
    assert "probe_sha" in note and "onnx sha" in note

    unconfirmed = dict(pkt, candidate_status="probe_confirmed_public")
    state, note = captain.classify(unconfirmed, {"task001": 100})
    assert state == "CONFLICT"
    assert "not confirmed_public" in note
