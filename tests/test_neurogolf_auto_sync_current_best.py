import json
from pathlib import Path

from tools.neurogolf_auto_sync_current_best import main
from tools.neurogolf_sync_current_best import sha256_file


def seed_repo(tmp_path: Path, *, current_score: float = 1.0) -> None:
    (tmp_path / "submissions").mkdir()
    (tmp_path / "submissions" / "submission.zip").write_bytes(b"old package")
    old_sha = sha256_file(tmp_path / "submissions" / "submission.zip")
    (tmp_path / "submissions" / "current_best_manifest.json").write_text(
        json.dumps(
            {
                "schema": "neurogolf.candidate_registry.v2-draft",
                "current_best": {
                    "candidate_id": "old",
                    "score": current_score,
                    "package_sha256": old_sha,
                },
                "entries": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".hermes" / "board").mkdir(parents=True)
    (tmp_path / ".hermes" / "board" / "candidate_registry.json").write_text(
        '{"schema":"neurogolf.candidate_registry.v2-draft","entries":[]}\n',
        encoding="utf-8",
    )
    (tmp_path / "Logs").mkdir()
    (tmp_path / "Logs" / "submissions_ledger.csv").write_text(
        "date,number,fruit,kind,local_prediction,kaggle_score,verdict,source\n",
        encoding="utf-8",
    )


def test_auto_sync_noops_when_queue_and_canonical_match(tmp_path: Path, capsys):
    seed_repo(tmp_path, current_score=2.0)
    current = json.loads((tmp_path / "submissions" / "current_best_manifest.json").read_text(encoding="utf-8"))
    (tmp_path / ".hermes" / "board" / "submit_queue.json").write_text(
        json.dumps(
            {
                "current_best_manifest": {
                    "candidate_id": "old",
                    "score": 2.0,
                    "package_sha256": current["current_best"]["package_sha256"],
                    "sync_status": "synced",
                },
                "queue": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main(["--repo-root", str(tmp_path), "--always"])

    result = json.loads(capsys.readouterr().out)
    assert result["action"] == "no_action"
    assert "already matches" in result["reason"]


def test_auto_sync_promotes_queue_current_best_when_ahead(tmp_path: Path):
    seed_repo(tmp_path)
    package = tmp_path / "stage" / "submission.zip"
    package.parent.mkdir()
    package.write_bytes(b"new public winner")
    package_sha = sha256_file(package)
    queue_doc = {
        "current_best_manifest": {
            "candidate_id": "stack_task001_task002_on_1",
            "score": 2.5,
            "previous_global_score": 1.0,
            "delta_vs_previous_global": 1.5,
            "package_sha256": package_sha,
            "changed_entries": ["task001.onnx", "task002.onnx"],
            "sync_status": "requested",
        },
        "queue": [
            {
                "queue_id": "submit-test",
                "candidate_id": "task001_candidate",
                "expected_point_gain": 1.5,
                "actual_result": {
                    "stack_candidate_id": "stack_task001_task002_on_1",
                    "public_score": 2.5,
                    "delta_vs_lineage_base": 1.5,
                    "package_sha256": package_sha,
                    "stack_changed_entries": ["task001.onnx", "task002.onnx"],
                    "stack_composition": ["task001_candidate", "task002_candidate"],
                },
                "submit_execution": {
                    "package_path": str(package),
                    "package_sha256": package_sha,
                    "submitted_at": "2026-05-16T01:02:00Z",
                    "observed_complete_at": "2026-05-16T01:05:00Z",
                    "attempt_kind": "stack-2",
                },
                "sync_followup": {"sync_task_id": "t_sync"},
            }
        ],
    }
    (tmp_path / ".hermes" / "board" / "submit_queue.json").write_text(
        json.dumps(queue_doc) + "\n",
        encoding="utf-8",
    )

    main(
        [
            "--repo-root",
            str(tmp_path),
            "--orchestrator-current-best",
            str(tmp_path / "orchestrator" / "current_best" / "submission.zip"),
        ]
    )

    assert sha256_file(tmp_path / "submissions" / "submission.zip") == package_sha
    assert not (tmp_path / "orchestrator" / "current_best" / "submission.zip").exists()
    manifest = json.loads((tmp_path / "submissions" / "current_best_manifest.json").read_text(encoding="utf-8"))
    assert manifest["current_best"]["candidate_id"] == "stack_task001_task002_on_1"
    submit_queue = json.loads((tmp_path / ".hermes" / "board" / "submit_queue.json").read_text(encoding="utf-8"))
    assert submit_queue["current_best_manifest"]["sync_status"] == "synced"
    assert submit_queue["queue"][0]["sync_followup"]["sync_status"] == "synced"


def test_auto_sync_repairs_stale_mirror_when_canonical_matches_queue(tmp_path: Path):
    seed_repo(tmp_path, current_score=2.5)
    package = tmp_path / "stage" / "submission.zip"
    package.parent.mkdir()
    package.write_bytes(b"already canonical winner")
    package_sha = sha256_file(package)
    (tmp_path / "submissions" / "submission.zip").write_bytes(package.read_bytes())
    (tmp_path / "submissions" / "current_best_manifest.json").write_text(
        json.dumps(
            {
                "schema": "neurogolf.candidate_registry.v2-draft",
                "current_best": {
                    "candidate_id": "stack_task001_task002_on_1",
                    "score": 2.5,
                    "previous_global_score": 1.0,
                    "delta_vs_previous_global": 1.5,
                    "package_sha256": package_sha,
                    "changed_entries": ["task001.onnx", "task002.onnx"],
                },
                "entries": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    orchestrator_zip = tmp_path / "orchestrator" / "current_best" / "submission.zip"
    orchestrator_zip.parent.mkdir(parents=True)
    orchestrator_zip.write_bytes(b"stale mirror")
    (orchestrator_zip.parent / "manifest.json").write_text(
        '{"current_best":{"candidate_id":"old","score":1,"package_sha256":"OLD"}}\n',
        encoding="utf-8",
    )
    (tmp_path / ".hermes" / "board" / "submit_queue.json").write_text(
        json.dumps(
            {
                "current_best_manifest": {
                    "candidate_id": "stack_task001_task002_on_1",
                    "score": 2.5,
                    "previous_global_score": 1.0,
                    "delta_vs_previous_global": 1.5,
                    "package_sha256": package_sha,
                    "changed_entries": ["task001.onnx", "task002.onnx"],
                    "sync_status": "synced",
                },
                "queue": [
                    {
                        "queue_id": "submit-test",
                        "candidate_id": "task001_candidate",
                        "expected_point_gain": 1.5,
                        "actual_result": {
                            "stack_candidate_id": "stack_task001_task002_on_1",
                            "public_score": 2.5,
                            "delta_vs_lineage_base": 1.5,
                            "package_sha256": package_sha,
                            "stack_changed_entries": ["task001.onnx", "task002.onnx"],
                            "stack_composition": ["task001_candidate", "task002_candidate"],
                        },
                        "submit_execution": {
                            "package_path": str(package),
                            "package_sha256": package_sha,
                            "submitted_at": "2026-05-16T01:02:00Z",
                            "observed_complete_at": "2026-05-16T01:05:00Z",
                            "attempt_kind": "stack-2",
                        },
                        "sync_followup": {"sync_task_id": "t_sync"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main(
        [
            "--repo-root",
            str(tmp_path),
            "--orchestrator-current-best",
            str(orchestrator_zip),
            "--sync-legacy-mirrors",
        ]
    )

    assert sha256_file(orchestrator_zip) == package_sha
    orchestrator_manifest = json.loads((orchestrator_zip.parent / "manifest.json").read_text(encoding="utf-8"))
    assert orchestrator_manifest["current_best"]["candidate_id"] == "stack_task001_task002_on_1"
    assert orchestrator_manifest["current_best"]["package_sha256"] == package_sha


def test_auto_sync_cleans_superseded_sync_pending_entries(tmp_path: Path):
    seed_repo(tmp_path, current_score=3.0)
    current = json.loads((tmp_path / "submissions" / "current_best_manifest.json").read_text(encoding="utf-8"))
    (tmp_path / ".hermes" / "board" / "submit_queue.json").write_text(
        json.dumps(
            {
                "current_best_manifest": {
                    "candidate_id": "old",
                    "score": 3.0,
                    "package_sha256": current["current_best"]["package_sha256"],
                    "sync_status": "synced",
                },
                "queue": [
                    {
                        "candidate_id": "old_component",
                        "status": "ready",
                        "submit_queue_status": "submitted-complete-promote-sync-pending",
                        "submit_ready": True,
                        "actual_result": {
                            "stack_candidate_id": "older_stack",
                            "public_score": 2.5,
                            "package_sha256": "A" * 64,
                        },
                        "sync_followup": {"sync_task_id": "t_old", "sync_status": "requested"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main(["--repo-root", str(tmp_path)])

    submit_queue = json.loads((tmp_path / ".hermes" / "board" / "submit_queue.json").read_text(encoding="utf-8"))
    entry = submit_queue["queue"][0]
    assert entry["status"] == "submitted-complete-promote-superseded"
    assert entry["submit_ready"] is False
    assert entry["sync_followup"]["sync_status"] == "superseded-by-current-best"


def test_auto_sync_accepts_submit_history_match_when_queue_misses(tmp_path: Path):
    """Validator widening (2026-05-19): when current_best_manifest references
    a candidate that has no queue[] entry but DOES have a submit_history[]
    entry, the validator should NOT exit-1 with "no submit_queue entry matches".

    Real failure mode: alternates31_on_5800_80 had current_best_manifest set
    but queue[] empty for it. The widening lets the sync proceed by falling
    back to submit_history match."""
    seed_repo(tmp_path)
    package = tmp_path / "stage" / "submission.zip"
    package.parent.mkdir()
    package.write_bytes(b"new public winner")
    package_sha = sha256_file(package)
    queue_doc = {
        "current_best_manifest": {
            "candidate_id": "alternates31_on_5800_80",
            "score": 2.5,
            "previous_global_score": 1.0,
            "delta_vs_previous_global": 1.5,
            "package_sha256": package_sha,
            "changed_entries": ["31-task-multi-stack"],
            "sync_status": "requested",
        },
        # queue[] is EMPTY for this candidate — the orphan case
        "queue": [],
        # but submit_history[] DOES have it (e.g. backfilled by the
        # update_submit_queue invariant repair)
        "submit_history": [
            {
                "at": "2026-05-19T13:45:00Z",
                "status": "submitted-complete-promote-synced",
                "stage_id": "alternates31_on_5800_80",
                "candidate_id": "alternates31_on_5800_80",
                "package_sha256": package_sha,
                "submitted_at": "2026-05-19T13:45:00Z",
                "expected_point_gain": 5.31,
                "actual_point_gain": 4.39,
                "lineage_base_score": 1.0,
                "kaggle_score": 2.5,
                "submit_execution": {
                    "package_path": str(package),
                    "package_sha256": package_sha,
                },
            }
        ],
    }
    (tmp_path / ".hermes" / "board" / "submit_queue.json").write_text(
        json.dumps(queue_doc) + "\n",
        encoding="utf-8",
    )

    # Without the widening, this raises SystemExit("no submit_queue entry matches...").
    # With it, the validator accepts the submit_history match and proceeds.
    main(["--repo-root", str(tmp_path)])
    # If main returned without raising, the widening worked.
    assert sha256_file(tmp_path / "submissions" / "submission.zip") == package_sha


def test_auto_sync_still_errors_when_neither_queue_nor_history_match(tmp_path: Path):
    """Defensive: the widening doesn't make the validator infinitely
    permissive. If BOTH queue[] and submit_history[] have no matching
    entry, the original SystemExit must still fire."""
    import pytest
    seed_repo(tmp_path)
    package = tmp_path / "stage" / "submission.zip"
    package.parent.mkdir()
    package.write_bytes(b"new public winner")
    package_sha = sha256_file(package)
    queue_doc = {
        "current_best_manifest": {
            "candidate_id": "ghost_candidate",
            "score": 2.5,
            "previous_global_score": 1.0,
            "delta_vs_previous_global": 1.5,
            "package_sha256": package_sha,
            "changed_entries": ["x.onnx"],
            "sync_status": "requested",
        },
        "queue": [],
        "submit_history": [],
    }
    (tmp_path / ".hermes" / "board" / "submit_queue.json").write_text(
        json.dumps(queue_doc) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="no submit_queue entry matches"):
        main(["--repo-root", str(tmp_path)])
