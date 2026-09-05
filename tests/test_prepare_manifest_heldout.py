"""Offline tests for exact selected-reset materialization."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from vla_training.eval import prepare_manifest_heldout as prepare
from vla_training.eval.eval_manifest import (
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    POLICY_NOISE_KIND,
    seal_episode_manifest,
)

PAYLOADS = {
    "ep_meta.json": b"meta",
    "model.xml.gz": b"model",
    "states.npz": b"states",
}


def _descriptor(payload: bytes) -> dict:
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _manifest(task: str = "TaskA") -> dict:
    episodes = []
    for index in (7, 11):
        episodes.append(
            {
                "task": task,
                "split_set": "atomic_seen",
                "horizon": 20,
                "episode_index": index,
                "reset": {
                    "kind": "heldout_demo",
                    "extras_relpath": f"{task}/extras/episode_{index:06d}",
                    "source": f"s3://dataset/{task}",
                    "artifacts": {name: _descriptor(payload) for name, payload in PAYLOADS.items()},
                },
                "seed": 100 + index,
            }
        )
    return seal_episode_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "split": "target",
            "task_sets": ["atomic_seen"],
            "base_seed": 7,
            "policy_noise": {
                "kind": POLICY_NOISE_KIND,
                "key_fields": ["episode.seed", "env_step"],
            },
            "episodes_per_task": 2,
            "episodes": episodes,
        }
    )


def _write(path: Path, manifest: dict) -> Path:
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_selection_requires_sealed_manifest_and_exact_requested_task(tmp_path):
    path = _write(tmp_path / "manifest.json", _manifest())
    selection = prepare.load_selection(path, ["TaskA"])
    assert [record["episode_index"] for record in selection["TaskA"]] == [7, 11]

    tampered = json.loads(path.read_text())
    tampered["episodes"][0]["seed"] += 1
    _write(path, tampered)
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        prepare.load_selection(path, ["TaskA"])


def test_unsafe_task_component_is_rejected_before_directory_creation(tmp_path):
    path = _write(tmp_path / "manifest.json", _manifest("../escape"))
    with pytest.raises(ValueError, match="unsafe path components"):
        prepare.load_selection(path, ["../escape"])


def test_task_sync_includes_only_selected_episode_directories(tmp_path, monkeypatch):
    path = _write(tmp_path / "manifest.json", _manifest())
    records = prepare.load_selection(path, ["TaskA"])["TaskA"]
    seen = {}

    def fake_run(command, *, check):
        assert check is True
        seen["command"] = command
        destination = Path(command[4])
        for record in records:
            episode = destination / f"episode_{record['episode_index']:06d}"
            episode.mkdir(parents=True)
            for name, payload in PAYLOADS.items():
                (episode / name).write_bytes(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(prepare.subprocess, "run", fake_run)
    task, count = prepare._prepare_task(tmp_path / "heldout", "TaskA", records)
    assert (task, count) == ("TaskA", 2)
    command = seen["command"]
    assert command[5:8] == ["--only-show-errors", "--exclude", "*"]
    includes = [command[index + 1] for index, value in enumerate(command) if value == "--include"]
    assert includes == ["episode_000007/*", "episode_000011/*"]
