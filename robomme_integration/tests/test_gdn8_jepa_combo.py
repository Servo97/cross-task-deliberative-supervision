"""High-level identity and data-contract tests for the RoboMME GDN+JEPA arm.

The combination is not a renamed JEPA-only arm or a renamed GDN arm: training consumes a causal
``omega`` window for K=8 steering and, from the same decision row, the K=1 future ``omega`` target
for JEPA+SigReg.  Inference continues to use only the GDN window.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pytest

from wsm_settings import ROBOCASA_OPENPI_SRC

OPENPI = pathlib.Path(os.environ.get("ROBOMME_OPENPI_SRC", str(ROBOCASA_OPENPI_SRC)))
if OPENPI.exists():
    sys.path.insert(0, str(OPENPI))

from robomme_integration.ablation_matrix import METHODS  # noqa: E402
from robomme_integration.launch import _arm_spec  # noqa: E402
from robomme_integration.training.arms import (  # noqa: E402
    ARM_IDS,
    ROBOTT_ARMS,
    SEQUENCE_ARMS,
    WORKSPACE_ARMS,
)
from robomme_integration.training.config import (  # noqa: E402
    GDN8_JEPA_ARM,
    build_train_config,
    validate_train_config,
)
from robomme_integration.training.data import RoboMMEWorkspaceExecutionDataset  # noqa: E402
from robomme_integration.training.single_task import task_manifest_sha256  # noqa: E402


class _Columns:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]


class _Dataset:
    def __init__(self, episode, step, demo):
        self.hf_dataset = _Columns({"episode_index": episode, "step_idx": step, "is_demo": demo})

    def __len__(self):
        return len(self.hf_dataset["step_idx"])

    def __getitem__(self, index):
        return {
            "episode_index": int(self.hf_dataset["episode_index"][index]),
            "step_idx": int(self.hf_dataset["step_idx"][index]),
            "is_demo": bool(self.hf_dataset["is_demo"][index]),
        }


@pytest.fixture
def _paths(monkeypatch, tmp_path):
    for name in ("data", "assets", "init", "workspace"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("ROBOMME_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROBOMME_ASSETS_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("WSM_INIT_FROM", str(tmp_path / "init"))
    monkeypatch.setenv("ROBOMME_TASK", "PickXtimes")
    monkeypatch.setenv("ROBOMME_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.delenv("ROBOMME_WORKSPACE_INDEX", raising=False)
    return tmp_path


def _write_workspace(tmp_path: pathlib.Path) -> tuple[_Dataset, np.ndarray]:
    root = tmp_path / "workspace" / "PickXtimes"
    episode_root = root / "episode_500"
    episode_root.mkdir(parents=True, exist_ok=True)
    omega = np.arange(25 * 512, dtype=np.float32).reshape(25, 512)
    path = episode_root / "omega_f16.npy"
    np.save(path, omega.astype(np.float16))
    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": "PickXtimes",
                "task_manifest_sha256": task_manifest_sha256("PickXtimes"),
                "omega_dim": 512,
                "episodes": [500],
                "encoder_id": "d" * 64,
                "causal_contract": {
                    "source_frames": "at_or_before_step_idx",
                    "uses_future_execution_frames": False,
                    "video_prefix": "benchmark_provided_only",
                },
                "records": [
                    {
                        "episode": 500,
                        "steps": 25,
                        "path": "episode_500/omega_f16.npy",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = _Dataset(
        episode=np.repeat([500], 25),
        step=np.arange(25),
        demo=np.zeros((25,), dtype=bool),
    )
    return dataset, omega.astype(np.float16).astype(np.float32)


def test_combo_self_labels_and_dry_config_exposes_both_mechanisms(_paths):
    assert GDN8_JEPA_ARM == "gdn8_jepa_l01_k1"
    assert GDN8_JEPA_ARM in ARM_IDS and GDN8_JEPA_ARM in WORKSPACE_ARMS
    assert GDN8_JEPA_ARM not in ROBOTT_ARMS and GDN8_JEPA_ARM not in SEQUENCE_ARMS

    config = build_train_config(GDN8_JEPA_ARM)
    validate_train_config(config, GDN8_JEPA_ARM)
    assert config.name == "pi05_robomme_gdn8_jepa_l01_k1"
    assert config.exp_name == "pi05_robomme_PickXtimes_gdn8_jepa_l01_k1"
    assert config.batch_size == 64

    model = config.model
    assert model.wsm_tanh and model.wsm_cond_type == "gated_deltanet"
    assert model.wsm_cond_window == 8
    assert model.wsm_jepa and model.wsm_jepa_weight == pytest.approx(0.1)
    assert model.wsm_jepa_sigreg_weight == pytest.approx(0.05)
    assert model.wsm_jepa_num_futures == 1
    assert not model.robottt and not model.wsm_cfg2

    data = config.data
    assert data.workspace_window == 8 and data.workspace_stride == 10
    assert data.workspace_jepa_futures == 1 and data.workspace_jepa_with_window
    assert "wsm_tanh_cond" in config.weight_loader.missing_regex
    assert "wsm_jepa_head" in config.weight_loader.missing_regex

    method = next(method for method in METHODS if method["id"] == GDN8_JEPA_ARM)
    assert method["robomme_arm"] == GDN8_JEPA_ARM
    assert method["single_task"] == "workspace_artifact_required"
    assert method["steering"] == "gated_deltanet_k8"
    assert method["aux"] == "jepa_lambda.1_k1_sigreg.05"

    launch_spec = _arm_spec(GDN8_JEPA_ARM)
    assert launch_spec["steering"] == "gated_deltanet_k8"
    assert launch_spec["jepa"] == {"lambda": 0.1, "futures": 1, "sigreg": 0.05}
    assert launch_spec["workspace_tokens_omega_t"]


def test_combo_record_contains_exact_parent_window_and_parent_jepa_target(tmp_path):
    dataset, omega = _write_workspace(tmp_path)
    common = {
        "root": str(tmp_path / "workspace"),
        "task_name": "PickXtimes",
        "episodes": (500,),
        "stride": 10,
    }
    gdn = RoboMMEWorkspaceExecutionDataset(dataset, window=8, **common)
    jepa = RoboMMEWorkspaceExecutionDataset(dataset, jepa_futures=1, **common)
    combo = RoboMMEWorkspaceExecutionDataset(
        dataset,
        window=8,
        jepa_futures=1,
        jepa_with_window=True,
        **common,
    )

    row = combo[12]
    assert set(row) >= {"wsm_w_window", "wsm_w_target", "wsm_w_target_valid"}
    assert row["wsm_w_window"].shape == (8, 512)
    np.testing.assert_array_equal(row["wsm_w_window"], gdn[12]["wsm_w_window"])
    np.testing.assert_array_equal(row["wsm_w_window"], omega[[0, 0, 0, 0, 0, 0, 2, 12]])
    np.testing.assert_array_equal(row["wsm_w_target"], jepa[12]["wsm_w_target"])
    np.testing.assert_array_equal(row["wsm_w_target"], omega[22])
    assert bool(row["wsm_w_target_valid"])

    last = combo[24]
    np.testing.assert_array_equal(last["wsm_w_target"], omega[24])
    assert not bool(last["wsm_w_target_valid"])


def test_combo_data_contract_fails_closed_without_the_explicit_gate(tmp_path):
    dataset, _ = _write_workspace(tmp_path)
    common = {
        "root": str(tmp_path / "workspace"),
        "task_name": "PickXtimes",
        "episodes": (500,),
    }
    with pytest.raises(ValueError, match="without the combo gate"):
        RoboMMEWorkspaceExecutionDataset(dataset, window=8, jepa_futures=1, **common)
    with pytest.raises(ValueError, match="explicit future target"):
        RoboMMEWorkspaceExecutionDataset(dataset, window=8, jepa_with_window=True, **common)
