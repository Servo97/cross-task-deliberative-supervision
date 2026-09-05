from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robomme_integration.eval import framesamp_am_r1_screen as screen
from robomme_integration.eval import framesamp_am_r1_screen_cloud as cloud
from robomme_integration.eval.framesamp_am_r1_canary import H100_NAME, LANES
from robomme_integration.eval.framesamp_am_r1_screen_launch import build

ROOT = Path(__file__).resolve().parents[2]
PACKET = (
    ROOT / "internal_planning_and_todos/launch_packets/robomme_fs_r1_oracle_screen" / "fs_r1_screen_packet_v8.json"
)
CANARY_PLAN = (
    ROOT / "internal_planning_and_todos/launch_packets/robomme_fs_r1_oracle_screen" / "fs_r1_canary_plan_r3.json"
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _seal(value: dict, field: str) -> dict:
    result = dict(value)
    result[field] = hashlib.sha256(_canonical(value)).hexdigest()
    return result


def _canary_lane(task: str, episode: int, budget: int, fit_mass: bool) -> dict:
    return _seal(
        {
            "kind": "robomme_framesamp_am_r1_oracle_rollout_canary",
            "status": "HARD_GREEN",
            "scope": "fs_r1_runtime_canary_not_scored_evidence",
            "task": task,
            "episode": episode,
            "budget": budget,
            "fit_mass": fit_mass,
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
        },
        "receipt_sha256",
    )


def _canary_receipt(manifest: dict) -> dict:
    lanes = []
    for lane, (task, episode, budget, fit_mass) in enumerate(LANES):
        receipt = _canary_lane(task, episode, budget, fit_mass)
        lanes.append(
            {
                "lane": lane,
                "task": task,
                "episode": episode,
                "budget": budget,
                "fit_mass": fit_mass,
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt": receipt,
            }
        )
    source_files = manifest["identity"]["source_files"]
    return _seal(
        {
            "kind": "robomme_framesamp_am_r1_8xh100_runtime_canary",
            "status": "HARD_GREEN",
            "scope": "runtime_canary_only_not_scored_evidence",
            "lane_count": 8,
            "authenticated_cuts": 8,
            "executed_simulator_actions": 128,
            "cloud_publication": False,
            "topology": [{}] * 8,
            "lanes": lanes,
            "runner_sha256": source_files["per_cut_rollout"],
            "sim_worker_sha256": source_files["multi_replan_sim_worker"],
        },
        "receipt_sha256",
    )


def _build_plan(tmp_path: Path) -> dict:
    canary_plan = json.loads(CANARY_PLAN.read_text(encoding="utf-8"))
    receipt_path = tmp_path / "canary.complete.json"
    receipt_path.write_text(
        json.dumps(_canary_receipt(canary_plan["manifest"]), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return build(
        source_dir=ROOT / "robomme_integration",
        packet_path=PACKET,
        canary_plan_path=CANARY_PLAN,
        canary_receipt_path=receipt_path,
    )


def test_screen_plan_is_canary_bound_and_covers_twelve_cells(tmp_path):
    plan = _build_plan(tmp_path)
    cloud.validate_manifest(plan["manifest"])
    cloud.validate_environment(plan["manifest"], plan["environment"])
    publication = plan["manifest"]["publication"]
    assert len(publication["results"]) == 12
    assert len({row["cell_id"] for row in publication["results"]}) == 12
    assert plan["manifest"]["infrastructure"]["cell_schedule"] == ("12 sequential cells x 8 concurrent episode lanes")
    assert plan["manifest"]["identity"]["canary_id"].startswith("fs-r1-p5-canary-")


def test_screen_manifest_rejects_resealed_result_namespace_drift(tmp_path):
    plan = _build_plan(tmp_path)
    manifest = json.loads(json.dumps(plan["manifest"]))
    manifest.pop("manifest_sha256")
    manifest["publication"]["results"][0]["result_s3"] = "s3://wrong/result.json"
    manifest = cloud.seal(manifest, "manifest_sha256")
    with pytest.raises(ValueError, match="publication contract drifted"):
        cloud.validate_manifest(manifest)


def test_screen_runs_cells_in_packet_order_and_writes_terminal_receipt(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    observed = []

    def fake_run_cell(**kwargs):
        cell = kwargs["cell"]
        observed.append(cell["cell_id"])
        return {
            "cell_id": cell["cell_id"],
            "claim_sha256": "1" * 64,
            "claim_file_sha256": "2" * 64,
            "successes": 0,
            "valid_episodes": 8,
            "promote_to_paired_fixed50": False,
            "result_s3": None,
        }

    monkeypatch.setattr(screen, "_run_cell", fake_run_cell)
    monkeypatch.setattr(
        screen.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="unused"),
    )
    monkeypatch.setattr(
        screen,
        "parse_h100_topology",
        lambda output: [{"index": index, "uuid": f"GPU-{index}", "name": H100_NAME} for index in range(8)],
    )
    result = screen.run_screen(
        packet_path=PACKET,
        output_dir=output,
        policy_overlay=tmp_path,
        overlay_manifest_sha256="3" * 64,
        official_checkout=tmp_path,
        checkpoint=tmp_path,
        runtime_root=tmp_path,
    )
    assert observed == [cell["cell_id"] for cell in packet["oracle_cells"]]
    assert result["status"] == "COMPLETE"
    assert result["cell_count"] == 12
    assert result["valid_episodes"] == 96
    assert (output / "screen.complete.json").is_file()


def test_screen_entry_is_release_runtime_and_canary_gated():
    shell = (ROOT / "robomme_integration/gpu_framesamp_am_r1_screen_entry.sh").read_text(encoding="utf-8")
    driver = (ROOT / "robomme_integration/eval/framesamp_am_r1_screen_entry.py").read_text(encoding="utf-8")
    assert "framesamp_am_r1_screen_entry.py" in shell
    assert "canonical FS-R1 canary receipt differs" in driver
    assert '"jax": "0.5.3"' in driver
    assert '"orbax_checkpoint": "0.11.13"' in driver
    assert "--confirm-publish" in driver
