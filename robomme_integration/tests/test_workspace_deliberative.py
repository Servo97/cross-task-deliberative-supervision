from __future__ import annotations

import hashlib
import json

import numpy as np

from robomme_integration.training.single_task import TASK_EPISODES, task_manifest_sha256
from robomme_integration.training.workspace_deliberative import (
    FEATURE_DIM,
    MAX_EVENTS,
    WorkspaceBatchSampler,
    init_params,
    loss_and_metrics,
)
from robomme_integration.training.workspace_supervision_cache import sha256_file


def _supervision_tree(tmp_path):
    root = tmp_path / "supervision" / "PickXtimes"
    records = []
    for episode in TASK_EPISODES["PickXtimes"]:
        directory = root / f"episode_{episode}"
        directory.mkdir(parents=True)
        path = directory / "supervision.npz"
        steps = 6
        frame = np.stack([np.full((FEATURE_DIM,), episode + step, dtype=np.float16) for step in range(steps)])
        state = np.stack([np.arange(8, dtype=np.float32) + step for step in range(steps)])
        with path.open("wb") as stream:
            np.savez(
                stream,
                frame_mean_f16=frame,
                state_f32=state,
                event_anchor_i32=np.asarray([0], dtype=np.int32),
                event_patch_id_i16=np.asarray([0], dtype=np.int16),
                event_feature_f16=frame[:1],
            )
        records.append(
            {
                "episode": episode,
                "steps": steps,
                "events": [
                    {
                        "anchor_step": 0,
                        "patch_id": 0,
                        "simple_subgoal_sha256": "a" * 64,
                        "grounded_subgoal_sha256": "b" * 64,
                    }
                ],
                "unpointed_segments": [],
                "path": f"episode_{episode}/supervision.npz",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact": "robomme_wsm_long_lag_supervision",
        "task_name": "PickXtimes",
        "task_manifest_sha256": task_manifest_sha256("PickXtimes"),
        "episodes": list(TASK_EPISODES["PickXtimes"]),
        "feature_dim": FEATURE_DIM,
        "patch_grid": 8,
        "causal_training_contract": {
            "encoder_inputs": "frames_at_or_before_decision_t",
            "target": "event_anchor_at_or_before_t_minus_min_lag",
            "current_frame_masking_required": True,
            "uses_labels_at_inference": False,
        },
        "records": records,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return tmp_path / "supervision"


def test_workspace_sampler_is_episode_disjoint_paired_and_current_masked(tmp_path):
    sampler = WorkspaceBatchSampler(
        _supervision_tree(tmp_path),
        "PickXtimes",
        seed=0,
        min_lag=1,
        future_delta=1,
        history_stride=1,
        max_history=6,
        verify_hashes=True,
    )
    assert set(sampler.train_episodes).isdisjoint(sampler.val_episodes)
    assert len(sampler.train_episodes) == 90 and len(sampler.val_episodes) == 10
    batch = sampler.sample(
        split="train",
        batch_size=4,
        rng=np.random.default_rng(0),
        mask_probability=1.0,
    )
    assert batch["history"].shape == (4, 6, FEATURE_DIM + 8)
    assert batch["event_target"].shape == (4, MAX_EVENTS, FEATURE_DIM)
    assert np.all(batch["event_presence"][:, 0] == 1)
    for history, mask in zip(batch["history"], batch["history_mask"], strict=True):
        current_index = int(np.count_nonzero(mask))
        assert not mask[current_index]
        assert np.all(history[current_index] == 0)


def test_workspace_objective_is_finite_and_weight_bearing(tmp_path):
    import jax
    import jax.numpy as jnp

    sampler = WorkspaceBatchSampler(
        _supervision_tree(tmp_path),
        "PickXtimes",
        seed=0,
        min_lag=1,
        future_delta=1,
        history_stride=1,
        max_history=6,
        verify_hashes=False,
    )
    batch = sampler.sample(
        split="train",
        batch_size=4,
        rng=np.random.default_rng(1),
        mask_probability=0.5,
    )
    params = init_params(jax.random.key(0))
    (loss, metrics), gradients = jax.value_and_grad(loss_and_metrics, has_aux=True)(
        params,
        {key: jnp.asarray(value) for key, value in batch.items()},
        weights={"occ": 0.1, "jepa": 0.1, "sigreg": 0.05},
    )
    assert np.isfinite(np.asarray(loss))
    assert all(np.isfinite(np.asarray(value)) for value in metrics.values())
    grad_norm = np.sqrt(sum(float(np.sum(np.square(np.asarray(value)))) for value in jax.tree.leaves(gradients)))
    assert grad_norm > 0
