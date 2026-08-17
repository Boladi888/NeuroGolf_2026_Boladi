import csv
import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest


def _load_team_submit():
    submit_py = Path(__file__).resolve().parents[2] / "K-NeuroGolf-OnnxlyFans" / "Team_Submission" / "submit.py"
    if not submit_py.exists():
        pytest.skip(f"team submit.py not found at {submit_py}")
    spec = importlib.util.spec_from_file_location("team_submit_snapshot_test", submit_py)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_zip(path: Path, task: str, data: bytes) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{task}.onnx", data)


def _write_manifest(path: Path, task: str, old_sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "creator", "onnx_sha256", "cost", "points", "updated_utc"],
        )
        writer.writeheader()
        writer.writerow({
            "task": task,
            "creator": "baseline",
            "onnx_sha256": old_sha,
            "cost": "100",
            "points": "20.0000",
            "updated_utc": "2026-06-23T00:00:00Z",
        })


def _snapshot_fixture(tmp_path, monkeypatch):
    submit = _load_team_submit()
    task = "task001"
    submitted = b"submitted-onnx"
    candidate = tmp_path / "Staged_Boladi" / f"{task}.onnx"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(submitted)
    (candidate.with_suffix(".json")).write_text("{}")
    live_zip = tmp_path / "submission.zip"
    _write_zip(live_zip, task, submitted)

    monkeypatch.setattr(submit, "PAST_SUBMISSIONS", tmp_path / "Past_Submissions")
    monkeypatch.setattr(submit, "TEAM_BEST_ONNX", tmp_path / "Team_Best_Onnx")
    monkeypatch.setattr(submit, "MANIFEST", tmp_path / "manifest.csv")
    monkeypatch.setattr(submit, "LIVE_ZIP", live_zip)
    monkeypatch.setattr(submit, "BACKEND", tmp_path / "Backend")
    _write_manifest(submit.MANIFEST, task, old_sha=_sha(b"old-best"))

    shipped = [{
        "task": task,
        "creator": "Boladi",
        "onnx_path": candidate,
        "sha256": _sha(submitted),
        "cost": 90,
        "points": 20.5,
        "msg": "unit",
    }]
    manifest = submit.read_manifest()
    rec = submit.build_record(shipped, manifest, 7331.09, "unit snapshot")
    snapshot = submit.write_submit_snapshot(rec, "UNIT", shipped, manifest, live_zip, expected_delta=0.5)
    return submit, rec, snapshot, candidate, submitted


def test_scorer_output_labels_score_as_if_correct():
    text = (Path(__file__).resolve().parents[1] / "tools" / "verify_submission.py").read_text()
    assert "LOCAL_POINTS_IF_ALL_CORRECT" in text
    assert "CONFIRMED_KAGGLE_PUBLIC_SCORE: UNKNOWN" in text
    assert "CANDIDATE_STATUS: UNVERIFIED" in text


def test_snapshot_manifest_is_written(tmp_path, monkeypatch):
    _submit, rec, snapshot, _candidate, submitted = _snapshot_fixture(tmp_path, monkeypatch)
    record_dir = tmp_path / "Past_Submissions" / rec["submission_id"]
    assert (record_dir / "submission.zip").exists()
    assert (record_dir / "snapshot_manifest.json").exists()
    assert snapshot["zip_sha256"] == _sha((record_dir / "submission.zip").read_bytes())
    assert snapshot["tasks"][0]["onnx_sha256"] == _sha(submitted)
    assert snapshot["tasks"][0]["points_if_correct"] == 20.5


def test_reconcile_refuses_current_hash_mismatch(tmp_path, monkeypatch):
    submit, rec, _snapshot, candidate, _submitted = _snapshot_fixture(tmp_path, monkeypatch)
    candidate.write_bytes(b"mutated-after-submit")
    with pytest.raises(SystemExit, match="SNAPSHOT HASH MISMATCH"):
        submit.reconcile(None, rec)


def test_reconcile_promotes_from_snapshot_not_mutable_staged_file(tmp_path, monkeypatch):
    submit, rec, _snapshot, candidate, submitted = _snapshot_fixture(tmp_path, monkeypatch)
    candidate.write_bytes(b"mutated-after-submit")
    submit.reconcile(None, rec, allow_snapshot_mismatch=True)
    assert (tmp_path / "Team_Best_Onnx" / "task001.onnx").read_bytes() == submitted


def test_hidden_failed_result_does_not_promote(tmp_path, monkeypatch):
    submit, rec, _snapshot, _candidate, _submitted = _snapshot_fixture(tmp_path, monkeypatch)
    submit.finalize_record(rec, "COMPLETE", 7331.09)
    submit.write_record(rec)
    submit.mark_snapshot_candidates(rec, rec["candidate_status"], rec["deviation_flag"], rec["actual_lb"])
    assert rec["candidate_status"] == "hidden_failed"
    with pytest.raises(SystemExit, match="refusing reconcile"):
        submit.reconcile(None, rec, allow_snapshot_mismatch=True)
    assert not (tmp_path / "Team_Best_Onnx" / "task001.onnx").exists()
