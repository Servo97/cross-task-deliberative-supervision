from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robomme_integration.eval.framesamp_am_r1_canary import (
    H100_NAME,
    parse_h100_topology,
    validate_lane_receipt,
)
from robomme_integration.eval.framesamp_am_r1_cloud import (
    validate_environment as validate_cloud_environment,
)
from robomme_integration.eval.framesamp_am_r1_cloud import (
    validate_manifest as validate_cloud_manifest,
)
from robomme_integration.eval.framesamp_am_r1_packet import validate
from robomme_integration.eval.framesamp_am_r1_publish import write_create_once
from robomme_integration.eval.framesamp_am_r1_rollout import (
    _load_response,
    _write_action_request,
)
from robomme_integration.eval.framesamp_r1_sim_worker import _load_action_request


def _request(root: Path, *, cut: int = 748) -> None:
    actions = np.zeros((20, 8), dtype=np.float32)
    path = root / "request-000.npy"
    with path.open("wb") as stream:
        np.save(stream, actions, allow_pickle=False)
    (root / "request-000.json").write_text(
        json.dumps(
            {
                "actions_npy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "causal_cut_step": cut,
                "shape": [20, 8],
            }
        ),
        encoding="utf-8",
    )


def test_multi_replan_worker_authenticates_action_bytes_and_cut(tmp_path):
    _request(tmp_path)
    actions = _load_action_request(tmp_path, 0, 748)
    assert actions.shape == (20, 8)
    assert actions.dtype == np.float32


def test_multi_replan_worker_rejects_stale_cut(tmp_path):
    _request(tmp_path, cut=747)
    with pytest.raises(ValueError, match="causal cut mismatch"):
        _load_action_request(tmp_path, 0, 748)


def test_rollout_request_and_response_are_cut_and_sha_authenticated(tmp_path):
    request = _write_action_request(
        tmp_path,
        replan=0,
        causal_cut_step=748,
        actions=np.zeros((20, 8), dtype=np.float32),
    )
    assert request["causal_cut_step"] == 748
    npz_path = tmp_path / "response-000.npz"
    with npz_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            current_image=np.zeros((4, 4, 3), dtype=np.uint8),
            current_wrist=np.zeros((4, 4, 3), dtype=np.uint8),
            current_state=np.zeros((8,), dtype=np.float32),
            execution_images=np.zeros((16, 4, 4, 3), dtype=np.uint8),
            execution_states=np.zeros((16, 8), dtype=np.float32),
        )
    (tmp_path / "response-000.json").write_text(
        json.dumps(
            {
                "causal_cut_step": 748,
                "executed_actions": 16,
                "executed_actions_total": 16,
                "outcome": "unknown",
                "response_npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                "status": "replan_ready",
                "terminal": False,
            }
        ),
        encoding="utf-8",
    )
    response, payload = _load_response(tmp_path, replan=0, causal_cut_step=748)
    assert response["executed_actions"] == 16
    assert payload["execution_images"].shape[0] == 16


def test_rollout_rejects_response_from_stale_cut(tmp_path):
    npz_path = tmp_path / "response-000.npz"
    with npz_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            current_image=np.zeros((1,), dtype=np.uint8),
            current_wrist=np.zeros((1,), dtype=np.uint8),
            current_state=np.zeros((1,), dtype=np.float32),
            execution_images=np.zeros((1, 1), dtype=np.uint8),
            execution_states=np.zeros((1, 1), dtype=np.float32),
        )
    (tmp_path / "response-000.json").write_text(
        json.dumps(
            {
                "causal_cut_step": 747,
                "executed_actions": 1,
                "executed_actions_total": 1,
                "outcome": "unknown",
                "response_npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                "status": "replan_ready",
                "terminal": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="causal cut mismatch"):
        _load_response(tmp_path, replan=0, causal_cut_step=748)


def test_h100_canary_topology_is_exact_and_distinct():
    output = "\n".join(f"{index}, GPU-{index:08d}, {H100_NAME}, 81559 MiB" for index in range(8))
    topology = parse_h100_topology(output)
    assert len(topology) == 8
    with pytest.raises(ValueError, match="exact indexed H100"):
        parse_h100_topology(output.replace(H100_NAME, "NVIDIA H100 FAKE", 1))


def test_h100_lane_receipt_rejects_scored_or_incomplete_canary():
    receipt = {
        "kind": "robomme_framesamp_am_r1_oracle_rollout_canary",
        "status": "HARD_GREEN",
        "scope": "fs_r1_runtime_canary_not_scored_evidence",
        "task": "VideoPlaceButton",
        "episode": 0,
        "budget": 256,
        "fit_mass": False,
        "canary_replans": 1,
        "replans": 1,
        "success": None,
        "fresh_attested_stack_fraction": 1.0,
        "persistent_oracle_payloads": False,
        "ephemeral_artifacts_deleted_before_receipt": True,
        "policy_runtime": {
            "source": "upstream_uv_lock",
            "jax": "0.5.3",
            "orbax_checkpoint": "0.11.13",
        },
        "device": {"count": 1, "platform": "gpu", "device_kind": H100_NAME},
        "simulator_runtime": {"gpu_name": H100_NAME, "torch_cuda_version": "12.8"},
        "cuts": [
            {
                "layers": [{}] * 18,
                "oracle_payload_deleted_before_simulator_response": True,
                "executed_actions": 16,
            }
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    validate_lane_receipt(
        receipt,
        task="VideoPlaceButton",
        episode=0,
        budget=256,
        fit_mass=False,
    )
    receipt["success"] = True
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(
        (json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="semantic mismatch"):
        validate_lane_receipt(
            receipt,
            task="VideoPlaceButton",
            episode=0,
            budget=256,
            fit_mass=False,
        )


def test_result_write_is_create_once_and_rejects_drift(tmp_path):
    path = tmp_path / "result.complete.json"
    write_create_once(path, b"first\n")
    write_create_once(path, b"first\n")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_create_once(path, b"second\n")


def test_materialized_r1_packet_is_complete_and_no_submit():
    path = (
        Path(__file__).resolve().parents[2]
        / "internal_planning_and_todos/launch_packets/robomme_fs_r1_oracle_screen/fs_r1_screen_packet_v6.json"
    )
    packet = json.loads(path.read_text(encoding="utf-8"))
    validate(packet)
    assert packet["executor"]["state"] == "IMPLEMENTED_PENDING_H100_CANARY"
    assert len(packet["executor"]["canary_lanes"]) == 8
    assert packet["this_packet_authorizes_submission"] is False


def test_materialized_r1_p5_canary_plan_is_sealed_and_no_submit():
    path = (
        Path(__file__).resolve().parents[2]
        / "internal_planning_and_todos/launch_packets/robomme_fs_r1_oracle_screen/fs_r1_canary_plan_r3.json"
    )
    plan = json.loads(path.read_text(encoding="utf-8"))
    validate_cloud_manifest(plan["manifest"])
    validate_cloud_environment(plan["manifest"], plan["environment"])
    assert plan["manifest"]["publication"]["scored_evidence"] is False
    assert plan["manifest"]["infrastructure"]["training_plan_arn"] is None
    assert plan["manifest"]["assets"]["policy_runtime"] == {
        "source": "upstream_uv_lock",
        "jax": "0.5.3",
        "orbax_checkpoint": "0.11.13",
    }
    assert plan["environment"]["SM_USE_RESERVED_CAPACITY"] == "1"
    assert "OPENPI_FORK_S3" not in plan["environment"]
