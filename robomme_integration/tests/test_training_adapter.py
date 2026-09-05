from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

from wsm_settings import ROBOCASA_OPENPI_SRC

OPENPI = pathlib.Path(
    os.environ.get(
        "ROBOMME_OPENPI_SRC",
        str(ROBOCASA_OPENPI_SRC),
    )
)
if OPENPI.exists():
    sys.path.insert(0, str(OPENPI))

from robomme_integration.sequence import uniformly_sample_prefix  # noqa: E402
from robomme_integration.training.data import (  # noqa: E402
    OFFICIAL_RECIPE_POLICY_IMAGE_KEYS,
    RoboMMEDemoWindowDataset,
    RoboMMEExecutionDataset,
    RoboMMEIidStepDataset,
    RoboMMEInputs,
    RoboMMEWindowDataset,
    RoboMMEWorkspaceExecutionDataset,
    _repair_selected_episode_data_index,
    install_official_recipe_two_view_patch,
    validate_official_recipe_two_view_patch,
)
from robomme_integration.training.single_task import (  # noqa: E402
    EPISODES_SHA256,
    TASK_EPISODES,
    TASKS_SHA256,
    select_task_episodes,
    task_manifest_sha256,
)


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
            "index": int(index),
            "episode_index": int(self.hf_dataset["episode_index"][index]),
            "step_idx": int(self.hf_dataset["step_idx"][index]),
            "is_demo": bool(self.hf_dataset["is_demo"][index]),
        }


def test_execution_filter_and_episode_safe_windows():
    episode = np.repeat(np.arange(2), 51)
    step = np.tile(np.arange(51), 2)
    demo = np.tile(np.arange(51) < 11, 2)
    dataset = _Dataset(episode, step, demo)
    execution = RoboMMEExecutionDataset(dataset)
    assert len(execution) == 80
    assert not any(execution[i]["is_demo"] for i in range(len(execution)))

    windows = RoboMMEWindowDataset(dataset, window_len=3, chunk_stride=10)
    assert len(windows) == 4
    for index in range(len(windows)):
        window = windows[index]
        assert len({item["episode_index"] for item in window}) == 1
        assert not any(item["is_demo"] for item in window)
        assert np.diff([item["step_idx"] for item in window]).tolist() == [10, 10]


def test_a6_iid_regrouping_preserves_exact_window_support_and_seed():
    episode = np.repeat(np.arange(3), 61)
    step = np.tile(np.arange(61), 3)
    demo = np.tile(np.arange(61) < 11, 3)
    dataset = _Dataset(episode, step, demo)
    contiguous = RoboMMEWindowDataset(dataset, window_len=4, chunk_stride=10)
    iid_a = RoboMMEIidStepDataset(dataset, window_len=4, chunk_stride=10, seed=7)
    iid_b = RoboMMEIidStepDataset(dataset, window_len=4, chunk_stride=10, seed=7)

    contiguous_support = sorted(item["index"] for i in range(len(contiguous)) for item in contiguous[i])
    iid_support = sorted(item["index"] for i in range(len(iid_a)) for item in iid_a[i])
    assert iid_support == contiguous_support
    assert [[item["index"] for item in iid_a[i]] for i in range(len(iid_a))] == [
        [item["index"] for item in iid_b[i]] for i in range(len(iid_b))
    ]
    assert any(len({item["episode_index"] for item in iid_a[i]}) > 1 for i in range(len(iid_a)))


def test_uniform_demo_prefix_and_loss_mask_are_fail_closed():
    selected, valid = uniformly_sample_prefix(
        [2, 4, 6, 8, 10],
        3,
        pad_index=11,
    )
    assert selected == (2, 6, 10)
    assert valid.tolist() == [True, True, True]

    selected, valid = uniformly_sample_prefix([2, 4], 4, pad_index=11)
    assert selected == (11, 11, 2, 4)
    assert valid.tolist() == [False, False, True, True]

    episode = np.repeat(np.arange(2), 51)
    step = np.tile(np.arange(51), 2)
    # Episode 0 has an 11-frame demo; episode 1 has no demonstration at all.
    demo = np.concatenate([np.arange(51) < 11, np.zeros((51,), dtype=bool)])
    dataset = _Dataset(episode, step, demo)
    windows = RoboMMEDemoWindowDataset(
        dataset,
        window_len=3,
        chunk_stride=10,
        demo_frames=4,
    )
    assert len(windows) == 6
    saw_demo = saw_no_demo = False
    for index in range(len(windows)):
        window = windows[index]
        assert len(window.steps) == 7
        assert window.loss_mask.tolist() == [0, 0, 0, 0, 1, 1, 1]
        assert not np.any(window.context_mask & (window.loss_mask > 0))
        episode_ids = {item["episode_index"] for item in window.steps}
        assert len(episode_ids) == 1
        if window.steps[-1]["episode_index"] == 0:
            saw_demo = True
            assert window.context_mask.tolist() == [True] * 4 + [False] * 3
            assert all(item["is_demo"] for item in window.steps[:4])
        else:
            saw_no_demo = True
            assert not window.context_mask.any()
            assert not any(item["is_demo"] for item in window.steps)
    assert saw_demo and saw_no_demo


def test_two_camera_input_satisfies_pi05_static_camera_contract():
    base = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    wrist = np.flip(base, axis=1).copy()
    transformed = RoboMMEInputs(model_type=object())(
        {
            "observation/image": base,
            "observation/wrist_image": wrist,
            "observation/state": np.arange(8, dtype=np.float32),
            "actions": np.arange(8, dtype=np.float32),
            "prompt": "test task",
        }
    )

    assert tuple(transformed["image"]) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    np.testing.assert_array_equal(transformed["image"]["base_0_rgb"], base)
    np.testing.assert_array_equal(transformed["image"]["left_wrist_0_rgb"], wrist)
    np.testing.assert_array_equal(
        transformed["image"]["right_wrist_0_rgb"],
        np.zeros_like(wrist),
    )
    assert transformed["image"]["right_wrist_0_rgb"].dtype == wrist.dtype
    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }


def test_official_recipe_adapter_has_exactly_two_policy_images(monkeypatch):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    transformed = RoboMMEInputs(
        model_type=object(),
        include_masked_right_view=False,
    )(
        {
            "observation/image": image,
            "observation/wrist_image": image,
            # State remains available only for the official delta-action transform and the
            # Observation structure; pi0.5/discrete_state_input=False never embeds it.
            "observation/state": np.zeros((8,), dtype=np.float32),
            "actions": np.zeros((20, 8), dtype=np.float32),
            "prompt": "test task",
        }
    )
    assert tuple(transformed["image"]) == ("base_0_rgb", "left_wrist_0_rgb")
    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
    }
    # The pinned modified OpenPI defaults to three image keys; prove the isolated runtime patch
    # selects the official two-key contract rather than merely dropping the adapter field and then
    # failing at model preprocessing.
    import robomme_integration.training.data as data_module

    monkeypatch.setattr(data_module, "_OFFICIAL_RECIPE_TWO_VIEW_PATCHED", False)
    monkeypatch.setattr(
        data_module._model,
        "preprocess_observation",
        data_module._ORIGINAL_PREPROCESS_OBSERVATION,
    )
    install_official_recipe_two_view_patch()
    validate_official_recipe_two_view_patch()
    assert data_module._model.preprocess_observation is data_module._official_recipe_preprocess_observation
    assert OFFICIAL_RECIPE_POLICY_IMAGE_KEYS == ("base_0_rgb", "left_wrist_0_rgb")


def test_serve_conditioning_survives_the_input_adapter():
    condition = np.arange(1024, dtype=np.float32)
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    transformed = RoboMMEInputs(model_type=object())(
        {
            "observation/image": image,
            "observation/wrist_image": image,
            "observation/state": np.zeros((8,), dtype=np.float32),
            "prompt": "test task",
            "robottt_cond": condition,
        }
    )
    np.testing.assert_array_equal(transformed["robottt_cond"], condition)


def _write_pinned_single_task_metadata(root):
    import hashlib
    import json

    meta = root / "meta"
    meta.mkdir(parents=True)
    prompts = [f"prompt {index}" for index in range(116)]
    tasks = b"".join(
        json.dumps({"task_index": index, "task": prompt}).encode() + b"\n" for index, prompt in enumerate(prompts)
    )
    episodes = b"".join(
        json.dumps(
            {
                "episode_index": index,
                "tasks": [prompts[index % len(prompts)]],
                "length": 1,
            }
        ).encode()
        + b"\n"
        for index in range(1600)
    )
    (meta / "tasks.jsonl").write_bytes(tasks)
    (meta / "episodes.jsonl").write_bytes(episodes)
    return hashlib.sha256(episodes).hexdigest(), hashlib.sha256(tasks).hexdigest()


def test_single_task_manifest_is_fail_closed(monkeypatch, tmp_path):
    episode_sha, task_sha = _write_pinned_single_task_metadata(tmp_path)
    monkeypatch.setattr(
        "robomme_integration.training.single_task.EPISODES_SHA256",
        episode_sha,
    )
    monkeypatch.setattr(
        "robomme_integration.training.single_task.TASKS_SHA256",
        task_sha,
    )
    selected = select_task_episodes(tmp_path, "PickXtimes")
    assert selected == tuple(range(500, 600))
    assert len(task_manifest_sha256("PickXtimes")) == 64

    with pytest.raises(ValueError, match="unknown RoboMME task"):
        select_task_episodes(tmp_path, "pickxtimes")
    with (tmp_path / "meta" / "episodes.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="metadata identity mismatch"):
        select_task_episodes(tmp_path, "PickXtimes")


def test_selected_episode_lookup_keeps_global_ids_and_uses_local_row_offsets():
    import torch

    class _Meta:
        total_episodes = 1600
        episodes = {
            500: {"length": 3},
            501: {"length": 5},
        }

    class _Selected:
        meta = _Meta()
        hf_dataset = list(range(8))
        episode_data_index = {
            "from": torch.tensor([0, 3]),
            "to": torch.tensor([3, 8]),
        }

    dataset = _Selected()
    _repair_selected_episode_data_index(dataset, (500, 501))
    assert dataset.episode_data_index["from"].shape == (1600,)
    assert dataset.episode_data_index["from"][500].item() == 0
    assert dataset.episode_data_index["to"][500].item() == 3
    assert dataset.episode_data_index["from"][501].item() == 3
    assert dataset.episode_data_index["to"][501].item() == 8
    assert dataset.episode_data_index["from"][499].item() == -1


def test_workspace_cache_is_task_bound_current_only_and_hash_verified(tmp_path):
    import hashlib
    import json

    root = tmp_path / "workspace" / "PickXtimes"
    records = []
    for episode, offset in ((500, 0), (501, 100)):
        episode_root = root / f"episode_{episode}"
        episode_root.mkdir(parents=True)
        path = episode_root / "omega_f16.npy"
        omega = np.arange(3 * 512, dtype=np.float32).reshape(3, 512) + offset
        np.save(path, omega.astype(np.float16))
        records.append(
            {
                "episode": episode,
                "steps": 3,
                "path": f"episode_{episode}/omega_f16.npy",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "task_name": "PickXtimes",
        "task_manifest_sha256": task_manifest_sha256("PickXtimes"),
        "omega_dim": 512,
        "episodes": [500, 501],
        "encoder_id": "a" * 64,
        "causal_contract": {
            "source_frames": "at_or_before_step_idx",
            "uses_future_execution_frames": False,
            "video_prefix": "benchmark_provided_only",
        },
        "records": records,
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = _Dataset(
        episode=np.repeat([500, 501], 3),
        step=np.tile(np.arange(3), 2),
        demo=np.zeros((6,), dtype=bool),
    )
    augmented = RoboMMEWorkspaceExecutionDataset(
        dataset,
        root=str(tmp_path / "workspace"),
        task_name="PickXtimes",
        episodes=(500, 501),
    )
    at_t = augmented[1]["wsm_w_window"].copy()
    assert at_t.shape == (1, 512) and at_t.dtype == np.float32
    # A later stored token cannot affect selection at t: the loader reads exactly omega_t.
    arrays = augmented._arrays[500]
    arrays._mmap.close()
    future = np.load(root / "episode_500/omega_f16.npy")
    future[2] = -123
    np.save(root / "episode_500/omega_f16.npy", future)
    augmented._arrays.clear()
    np.testing.assert_array_equal(augmented[1]["wsm_w_window"], at_t)

    manifest["causal_contract"]["uses_future_execution_frames"] = True
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="causal_contract mismatch"):
        RoboMMEWorkspaceExecutionDataset(
            dataset,
            root=str(tmp_path / "workspace"),
            task_name="PickXtimes",
            episodes=(500, 501),
        )


def test_multitask_workspace_index_routes_each_episode_to_its_own_task(monkeypatch, tmp_path):
    import hashlib
    import json

    from robomme_integration.training import single_task, workspace_index

    task_order = ("PickXtimes", "ButtonUnmaskSwap")
    task_episodes = {"PickXtimes": (0, 1), "ButtonUnmaskSwap": (2, 3)}
    monkeypatch.setattr(single_task, "TASK_ORDER", task_order)
    monkeypatch.setattr(single_task, "TASK_EPISODES", task_episodes)
    monkeypatch.setattr(workspace_index, "TASK_ORDER", task_order)

    workspace_root = tmp_path / "workspace"
    index_tasks = {}
    rows = []
    for task_ordinal, task in enumerate(task_order):
        task_root = workspace_root / task
        records = []
        encoder_id = f"{task_ordinal + 1:064x}"
        for episode in task_episodes[task]:
            episode_root = task_root / f"episode_{episode}"
            episode_root.mkdir(parents=True)
            omega = np.full((2, 512), task_ordinal * 100 + episode, dtype=np.float16)
            path = episode_root / "omega_f16.npy"
            np.save(path, omega)
            records.append(
                {
                    "episode": episode,
                    "steps": 2,
                    "path": f"episode_{episode}/omega_f16.npy",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            rows.extend((episode, step, False) for step in range(2))
        manifest = {
            "schema_version": 1,
            "task_name": task,
            "task_manifest_sha256": single_task.task_manifest_sha256(task),
            "omega_dim": 512,
            "episodes": list(task_episodes[task]),
            "encoder_id": encoder_id,
            "causal_contract": RoboMMEWorkspaceExecutionDataset._CAUSAL_CONTRACT,
            "records": records,
        }
        manifest_path = task_root / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        index_tasks[task] = {
            "task_manifest_sha256": single_task.task_manifest_sha256(task),
            "encoder_id": encoder_id,
            "omega": {
                "uri": f"s3://bucket/{task}/omega",
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
            "representation": {
                "uri": f"s3://bucket/{task}/representation",
                "completion_sha256": f"{task_ordinal + 3:064x}",
                "step": 10000,
            },
            "supervision": None,
        }
    index_path = tmp_path / "all16.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "RoboMME",
                "scope": "all16",
                "task_order": list(task_order),
                "tasks": index_tasks,
            }
        ),
        encoding="utf-8",
    )
    episode, step, demo = map(np.asarray, zip(*rows, strict=True))
    dataset = _Dataset(episode=episode, step=step, demo=demo)
    routed = RoboMMEWorkspaceExecutionDataset(
        dataset,
        root=str(workspace_root),
        task_name=None,
        episodes=None,
        workspace_index=str(index_path),
    )
    assert len(routed) == 8
    for index in range(len(routed)):
        sample = routed[index]
        expected = (0 if sample["episode_index"] < 2 else 100) + sample["episode_index"]
        np.testing.assert_array_equal(sample["wsm_w_window"], np.full((1, 512), expected))


def test_workspace_cache_exposes_causal_deltanet_windows_and_future_jepa_masks(tmp_path):
    import hashlib
    import json

    root = tmp_path / "workspace" / "PickXtimes"
    episode_root = root / "episode_500"
    episode_root.mkdir(parents=True)
    path = episode_root / "omega_f16.npy"
    omega = np.arange(4 * 512, dtype=np.float32).reshape(4, 512)
    np.save(path, omega.astype(np.float16))
    manifest = {
        "schema_version": 1,
        "task_name": "PickXtimes",
        "task_manifest_sha256": task_manifest_sha256("PickXtimes"),
        "omega_dim": 512,
        "episodes": [500],
        "encoder_id": "b" * 64,
        "causal_contract": {
            "source_frames": "at_or_before_step_idx",
            "uses_future_execution_frames": False,
            "video_prefix": "benchmark_provided_only",
        },
        "records": [
            {
                "episode": 500,
                "steps": 4,
                "path": "episode_500/omega_f16.npy",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = _Dataset(
        episode=np.repeat([500], 4),
        step=np.arange(4),
        demo=np.zeros((4,), dtype=bool),
    )
    deltanet = RoboMMEWorkspaceExecutionDataset(
        dataset,
        root=str(tmp_path / "workspace"),
        task_name="PickXtimes",
        episodes=(500,),
        window=3,
        stride=1,
    )
    window = deltanet[1]["wsm_w_window"]
    assert window.shape == (3, 512)
    np.testing.assert_array_equal(window, omega[[0, 0, 1]])

    jepa = RoboMMEWorkspaceExecutionDataset(
        dataset,
        root=str(tmp_path / "workspace"),
        task_name="PickXtimes",
        episodes=(500,),
        jepa_futures=3,
        stride=1,
    )
    row = jepa[1]
    np.testing.assert_array_equal(row["wsm_w_target"], omega[[2, 3, 3]])
    np.testing.assert_array_equal(row["wsm_w_target_valid"], [True, True, False])


def test_q3_windows_combine_contiguous_execution_and_current_workspace(tmp_path):
    import hashlib
    import json

    root = tmp_path / "workspace" / "PickXtimes"
    episode_root = root / "episode_500"
    episode_root.mkdir(parents=True)
    path = episode_root / "omega_f16.npy"
    omega = np.arange(31 * 512, dtype=np.float32).reshape(31, 512)
    np.save(path, omega.astype(np.float16))
    manifest = {
        "schema_version": 1,
        "task_name": "PickXtimes",
        "task_manifest_sha256": task_manifest_sha256("PickXtimes"),
        "omega_dim": 512,
        "episodes": [500],
        "encoder_id": "c" * 64,
        "causal_contract": {
            "source_frames": "at_or_before_step_idx",
            "uses_future_execution_frames": False,
            "video_prefix": "benchmark_provided_only",
        },
        "records": [
            {
                "episode": 500,
                "steps": 31,
                "path": "episode_500/omega_f16.npy",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = _Dataset(
        episode=np.repeat([500], 31),
        step=np.arange(31),
        demo=np.arange(31) < 11,
    )
    augmented = RoboMMEWorkspaceExecutionDataset(
        dataset,
        root=str(tmp_path / "workspace"),
        task_name="PickXtimes",
        episodes=(500,),
        execution_index_view=False,
    )
    windows = RoboMMEWindowDataset(augmented, window_len=2, chunk_stride=10)
    assert len(windows) == 1
    row0, row1 = windows[0]
    assert [row0["step_idx"], row1["step_idx"]] == [11, 21]
    cached = omega.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(row0["wsm_w_window"], cached[[11]])
    np.testing.assert_array_equal(row1["wsm_w_window"], cached[[21]])


@pytest.mark.parametrize(
    ("arm", "mode", "window", "history_dropout", "jepa_weight", "futures"),
    [
        ("wsm_cfg", "cfg", 1, 0.0, 0.0, 0),
        ("wsm_tanh", "tanh", 1, 0.0, 0.0, 0),
        ("wsm_d8", "deltanet", 8, 0.0, 0.0, 0),
        ("wsm_d8_drop05", "deltanet", 8, 0.5, 0.0, 0),
        ("wsm_d16", "deltanet", 16, 0.0, 0.0, 0),
        ("wsm_d16_drop05", "deltanet", 16, 0.5, 0.0, 0),
        ("ptrm", "ptrm", 8, 0.0, 0.0, 0),
        ("jepa_l01_k1", "jepa", 1, 0.0, 0.1, 1),
        ("jepa_l1_k32", "jepa", 1, 0.0, 1.0, 32),
        ("jepa_l01_k16", "jepa", 1, 0.0, 0.1, 16),
    ],
)
def test_robocasa_winner_recipes_are_explicit_in_robomme(
    _paths,
    monkeypatch,
    tmp_path,
    arm,
    mode,
    window,
    history_dropout,
    jepa_weight,
    futures,
):
    from robomme_integration.training.config import build_train_config, validate_train_config

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ROBOMME_TASK", "PickXtimes")
    monkeypatch.setenv("ROBOMME_WORKSPACE_ROOT", str(workspace))
    config = build_train_config(arm)
    validate_train_config(config, arm)
    assert config.data.workspace_window == window
    assert config.data.workspace_jepa_futures == futures
    assert config.model.wsm_jepa_weight == pytest.approx(jepa_weight)
    assert getattr(config.model, "wsm_cond_history_dropout", 0.0) == pytest.approx(history_dropout)
    assert config.model.wsm_cfg2 is (mode == "cfg")
    assert config.model.wsm_tanh is (mode in {"tanh", "deltanet", "ptrm"})
    assert config.model.wsm_jepa is (mode == "jepa")
    if mode == "deltanet":
        assert config.model.wsm_cond_type == "gated_deltanet"
    if mode == "ptrm":
        assert config.model.wsm_cond_type == "gated_deltanet_ptrm"
        assert (config.model.wsm_ptrm_steps, config.model.wsm_ptrm_q_weight) == (4, 0.1)


def test_official_single_task_metadata_identity():
    # The constants are copied from the pinned official Xet objects, not a mutable local cache.
    assert EPISODES_SHA256 == "4b090c292a86386c82f57558e23430066c2e6429424c594a97100b730d7fab27"
    assert TASKS_SHA256 == "8aa71a980c3dd2ce6a4fbede2ca602eb01a0203c5ef2c1bada9d363efeb9a43e"
    assert set(TASK_EPISODES) == {
        "BinFill",
        "PickXtimes",
        "SwingXtimes",
        "StopCube",
        "VideoUnmask",
        "VideoUnmaskSwap",
        "ButtonUnmask",
        "ButtonUnmaskSwap",
        "PickHighlight",
        "VideoRepick",
        "VideoPlaceButton",
        "VideoPlaceOrder",
        "MoveCube",
        "InsertPeg",
        "PatternLock",
        "RouteStick",
    }


@pytest.fixture
def _paths(monkeypatch, tmp_path):
    for name in ("data", "assets", "init"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("ROBOMME_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROBOMME_ASSETS_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("WSM_INIT_FROM", str(tmp_path / "init"))


@pytest.mark.parametrize(
    ("arm", "robottt", "batch", "window", "demo_frames"),
    [
        ("s0", False, 64, 0, 0),
        ("q0", False, 8, 8, 0),
        ("q2", True, 8, 8, 0),
        ("q0v", False, 8, 8, 16),
        ("q2v", True, 8, 8, 16),
    ],
)
def test_arm_config_identity(_paths, arm, robottt, batch, window, demo_frames):
    from robomme_integration.training.config import build_train_config, validate_train_config

    config = build_train_config(arm)
    validate_train_config(config, arm)
    assert config.model.robottt is robottt
    assert config.batch_size == batch
    assert config.data.stage_q_window_len == window
    assert config.data.stage_q_demo_frames == demo_frames
    assert config.model.action_horizon == 20
    assert config.data.execution_only is (demo_frames == 0)
    if demo_frames:
        assert config.model.robottt_window_len == 24
        assert config.model.robottt_tbptt_segment == 24


def test_a6_and_q3_config_identities_are_exact(_paths, monkeypatch, tmp_path):
    from robomme_integration.training.config import build_train_config, validate_train_config

    a6 = build_train_config("a6")
    validate_train_config(a6, "a6")
    assert a6.data.stage_q_iid_steps is True
    assert a6.data.stage_q_window_len == 8 and a6.batch_size == 8
    assert a6.model.robottt is False and a6.model.wsm_tanh is False

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ROBOMME_TASK", "PickXtimes")
    monkeypatch.setenv("ROBOMME_WORKSPACE_ROOT", str(workspace))
    q3 = build_train_config("q3")
    validate_train_config(q3, "q3")
    assert q3.data.stage_q_iid_steps is False
    assert q3.data.stage_q_window_len == 8 and q3.data.workspace_window == 1
    assert q3.model.robottt is True and q3.model.wsm_tanh is True

    q1 = build_train_config("q1")
    validate_train_config(q1, "q1")
    assert q1.data.stage_q_iid_steps is False
    assert q1.data.stage_q_window_len == 8 and q1.data.workspace_window == 1
    assert q1.model.robottt is False and q1.model.wsm_tanh is True


def test_causal_v1_reuses_train_only_keypatch_head_but_requires_its_own_labels(_paths, monkeypatch, tmp_path):
    from robomme_integration.training.config import build_train_config, validate_train_config

    workspace = tmp_path / "workspace"
    supervision = tmp_path / "causal-labels"
    workspace.mkdir()
    supervision.mkdir()
    monkeypatch.setenv("ROBOMME_TASK", "PickXtimes")
    monkeypatch.setenv("ROBOMME_WORKSPACE_ROOT", str(workspace))
    with pytest.raises(ValueError, match="causal_v1 requires ROBOMME_SALIENT_ROOT"):
        build_train_config("causal_v1")
    monkeypatch.setenv("ROBOMME_SALIENT_ROOT", str(supervision))
    config = build_train_config("causal_v1")
    validate_train_config(config, "causal_v1")
    assert config.model.wsm_jepa and config.model.wsm_salient
    assert config.model.wsm_jepa_weight == 0.0
    assert config.model.wsm_jepa_sigreg_weight == 0.0
    assert config.data.salient_root == str(supervision)


def test_single_task_config_has_collision_free_default_name(_paths, monkeypatch):
    from robomme_integration.training.config import build_train_config, validate_train_config

    monkeypatch.setenv("ROBOMME_TASK", "PickXtimes")
    config = build_train_config("q2")
    validate_train_config(config, "q2")
    assert config.data.task_name == "PickXtimes"
    assert config.exp_name == "pi05_robomme_PickXtimes_q2"

    monkeypatch.setenv("ROBOMME_TASK", "pickxtimes")
    with pytest.raises(ValueError, match="unknown ROBOMME_TASK"):
        build_train_config("q2")


def test_single_task_wsm_cfg_is_explicit_and_collision_free(_paths, monkeypatch, tmp_path):
    from robomme_integration.training.config import build_train_config, validate_train_config

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ROBOMME_TASK", "PickXtimes")
    monkeypatch.setenv("ROBOMME_WORKSPACE_ROOT", str(workspace))
    config = build_train_config("wsm_cfg")
    validate_train_config(config, "wsm_cfg")
    assert config.model.wsm_cfg2 is True
    assert config.model.wsm_cfg is False and config.model.wsm is False
    assert config.model.robottt is False
    assert config.model.wsm_cfg_p_drop == pytest.approx(0.2)
    assert config.data.workspace_root == str(workspace)
    assert config.exp_name == "pi05_robomme_PickXtimes_wsm_cfg"


def test_multitask_workspace_config_requires_and_records_sealed_index(_paths, monkeypatch, tmp_path):
    from robomme_integration.training.config import build_train_config, validate_train_config

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    index = tmp_path / "all16.json"
    index.write_text("{}\n")
    monkeypatch.setenv("ROBOMME_WORKSPACE_ROOT", str(workspace))
    with pytest.raises(ValueError, match="ROBOMME_WORKSPACE_INDEX"):
        build_train_config("wsm_cfg")
    monkeypatch.setenv("ROBOMME_WORKSPACE_INDEX", str(index))
    config = build_train_config("wsm_cfg")
    validate_train_config(config, "wsm_cfg")
    assert config.data.task_name is None
    assert config.data.workspace_root == str(workspace)
    assert config.data.workspace_index == str(index)
    assert config.num_train_steps == 60000


def test_official_recipe_lerobot_config_is_exact_and_fails_closed(_paths, monkeypatch):
    from robomme_integration.training.arms import OFFICIAL_RECIPE_LEROBOT_LABEL
    from robomme_integration.training.config import build_train_config, validate_train_config

    exact = {
        "ROBOMME_RECIPE_LABEL": OFFICIAL_RECIPE_LEROBOT_LABEL,
        "WSM_MAX_STEPS": "80000",
        "WSM_WARMUP_STEPS": "10000",
        "WSM_PEAK_LR": "5e-5",
        "WSM_DECAY_STEPS": "100000",
        "WSM_DECAY_LR": "5e-5",
        "WSM_SEED": "42",
        "WSM_SAVE_INTERVAL": "10000",
    }
    for name, value in exact.items():
        monkeypatch.setenv(name, value)
    config = build_train_config("official_recipe_lerobot")
    validate_train_config(config, "official_recipe_lerobot")
    assert config.model.pi05 and config.model.action_horizon == 20
    assert config.model.max_token_len == 64 and not config.model.discrete_state_input
    assert (config.num_train_steps, config.seed, config.batch_size) == (80_000, 42, 64)
    assert (
        config.lr_schedule.warmup_steps,
        config.lr_schedule.peak_lr,
        config.lr_schedule.decay_steps,
        config.lr_schedule.decay_lr,
    ) == (10_000, 5e-5, 100_000, 5e-5)
    assert (
        config.optimizer.b1,
        config.optimizer.b2,
        config.optimizer.eps,
        config.optimizer.weight_decay,
        config.optimizer.clip_gradient_norm,
    ) == (0.9, 0.95, 1e-8, 1e-10, 1.0)
    assert getattr(config.freeze_filter.pattern, "pattern", None) == ".*img.*"
    assert config.ema_decay == 0.999
    assert (config.save_interval, config.keep_period) == (10_000, None)
    assert config.data.official_recipe_lerobot
    assert config.data.task_name is None and config.data.execution_only

    monkeypatch.setenv("WSM_SEED", "0")
    with pytest.raises(ValueError, match="incomplete or drifted"):
        build_train_config("official_recipe_lerobot")


def test_single_task_schedule_horizon_is_explicit(_paths, monkeypatch):
    from robomme_integration.training.config import build_train_config

    monkeypatch.setenv("WSM_MAX_STEPS", "20000")
    monkeypatch.setenv("WSM_WARMUP_STEPS", "1000")
    monkeypatch.setenv("WSM_PEAK_LR", "5e-5")
    monkeypatch.setenv("WSM_DECAY_STEPS", "20000")
    monkeypatch.setenv("WSM_DECAY_LR", "5e-6")
    monkeypatch.setenv("WSM_SEED", "0")
    config = build_train_config("s0")
    assert config.num_train_steps == 20000
    assert config.lr_schedule.warmup_steps == 1000
    assert config.lr_schedule.peak_lr == pytest.approx(5e-5)
    assert config.lr_schedule.decay_steps == 20000
    assert config.lr_schedule.decay_lr == pytest.approx(5e-6)
    assert config.seed == 0

    monkeypatch.setenv("WSM_WARMUP_STEPS", "20000")
    with pytest.raises(ValueError, match="smaller than"):
        build_train_config("s0")
