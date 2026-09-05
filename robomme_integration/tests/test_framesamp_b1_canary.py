from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robomme_integration.eval.framesamp_am_r1_canary import H100_NAME
from robomme_integration.eval.framesamp_b1_canary import (
    KIND as CANARY_KIND,
)
from robomme_integration.eval.framesamp_b1_canary import (
    LANES,
    validate_canary_packet,
    validate_canary_receipt,
)
from robomme_integration.eval.framesamp_b1_transition import (
    KIND as TRANSITION_KIND,
)
from robomme_integration.eval.framesamp_b1_transition import (
    RELEASED_CHECKPOINT_SHA256,
    SCOPE,
    validate_receipt,
)
from robomme_integration.training.framesamp_b1_data import B1_PARTITION_KIND
from robomme_integration.training.framesamp_b1_policy_overlay import (
    PATCHED_MEM_BUFFER_SHA256,
    PATCHED_POLICY_SHA256,
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _actions(seed: str) -> dict:
    return {"dtype": "float32", "shape": [20, 8], "sha256": seed * 64}


def _transition(task: str, episode: int, marker: str = "a") -> dict:
    demo = list(range(16))
    initial = {
        "step_idx": 15,
        "exec_start_idx": 15,
        "demo_indices": demo.copy(),
        "live_indices": [],
        "frame_indices": demo + [-1] * 16,
        "frame_mask_sha256": "b" * 64,
        "token_mask_sha256": "c" * 64,
        "valid_frames": 16,
        "valid_tokens": 256,
        "actions": _actions(marker),
    }
    later = {
        "step_idx": 31,
        "exec_start_idx": 15,
        "demo_indices": demo.copy(),
        "live_indices": list(range(16, 32)),
        "frame_indices": list(range(32)),
        "frame_mask_sha256": "d" * 64,
        "token_mask_sha256": "e" * 64,
        "valid_frames": 32,
        "valid_tokens": 512,
        "actions": _actions(marker),
    }
    unsigned = {
        "schema_version": 1,
        "kind": TRANSITION_KIND,
        "scope": SCOPE,
        "status": "HARD_GREEN",
        "task": task,
        "episode": episode,
        "representation_policy": B1_PARTITION_KIND,
        "overlay_manifest_sha256": "f" * 64,
        "overlay_source_tree_sha256": "0" * 64,
        "patched_policy_sha256": PATCHED_POLICY_SHA256,
        "patched_mem_buffer_sha256": PATCHED_MEM_BUFFER_SHA256,
        "released_checkpoint_sha256": RELEASED_CHECKPOINT_SHA256,
        "device": {"platform": "gpu", "device_kind": H100_NAME, "count": 1},
        "simulator_runtime": {"gpu_name": H100_NAME, "torch_cuda_version": "12.8"},
        "cuts": [initial, later],
        "executed_actions": 16,
        "elapsed_seconds": 1.0,
        "scored_evidence": False,
        "cloud_publication": False,
    }
    result = dict(unsigned)
    result["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    return result


def _canary() -> dict:
    lanes = []
    for lane, (task, episode) in enumerate(LANES):
        receipt = _transition(task, episode, marker=hex(lane)[-1])
        lanes.append(
            {
                "lane": lane,
                "task": task,
                "episode": episode,
                "transition_receipt_sha256": receipt["receipt_sha256"],
                "receipt_file_sha256": str(lane) * 64,
                "receipt": receipt,
            }
        )
    unsigned = {
        "schema_version": 1,
        "kind": CANARY_KIND,
        "status": "HARD_GREEN",
        "scope": "runtime_canary_only_not_scored_evidence",
        "representation_policy": B1_PARTITION_KIND,
        "overlay_manifest_sha256": "f" * 64,
        "topology": [
            {
                "index": lane,
                "uuid": f"GPU-{lane}",
                "name": H100_NAME,
                "memory_total_mib": 81_559,
            }
            for lane in range(8)
        ],
        "lane_count": 8,
        "transition_count": 8,
        "executed_simulator_actions": 128,
        "lanes": lanes,
        "runner_sha256": "1" * 64,
        "sim_worker_sha256": "2" * 64,
        "elapsed_seconds": 2.0,
        "scored_evidence": False,
        "cloud_publication": False,
    }
    result = dict(unsigned)
    result["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    return result


def _reseal(value: dict) -> dict:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    result = dict(unsigned)
    result["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    return result


def test_transition_receipt_accepts_only_frozen_demo_plus_full_live_cut():
    receipt = _transition("PickXtimes", 0)
    validate_receipt(receipt)
    receipt["cuts"][1]["demo_indices"][0] = 9
    with pytest.raises(ValueError, match="frozen demo"):
        validate_receipt(_reseal(receipt))


def test_transition_receipt_rejects_scored_or_wrong_source_identity():
    receipt = _transition("PickXtimes", 0)
    receipt["scored_evidence"] = True
    with pytest.raises(ValueError, match="semantics"):
        validate_receipt(_reseal(receipt))
    receipt = _transition("PickXtimes", 0)
    receipt["patched_policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="semantics"):
        validate_receipt(_reseal(receipt))


def test_aggregate_canary_binds_all_eight_task_transitions():
    receipt = _canary()
    validate_canary_receipt(receipt)
    receipt["lanes"][3]["task"] = "wrong"
    with pytest.raises(ValueError, match="lane binding"):
        validate_canary_receipt(_reseal(receipt))


def test_aggregate_canary_rejects_embedded_transition_drift():
    receipt = _canary()
    transition = receipt["lanes"][0]["receipt"]
    transition["scored_evidence"] = True
    transition = _reseal(transition)
    receipt["lanes"][0]["receipt"] = transition
    receipt["lanes"][0]["transition_receipt_sha256"] = transition["receipt_sha256"]
    with pytest.raises(ValueError, match="semantics"):
        validate_canary_receipt(_reseal(receipt))


def test_frozen_packet_authenticates_runtime_files_and_overlay():
    source_root = Path(__file__).resolve().parents[2]
    packet_path = (
        source_root / "internal_planning_and_todos/launch_packets/robomme_fs_b1_control/"
        "fs_b1_control_canary_packet_v1.json"
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    validate_canary_packet(
        packet,
        source_root=source_root,
        overlay_root=Path("/tmp/robomme-fs-b1-overlay-v1"),
    )
