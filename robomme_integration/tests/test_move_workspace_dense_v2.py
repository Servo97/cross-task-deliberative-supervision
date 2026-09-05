from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from robomme_integration import move_workspace_dense_v2_launch as launch_v2
from robomme_integration.training import move_workspace_dense_v2_canary as canary_v2
from robomme_integration.training import workspace_deliberative_dense_v2 as train_v2
from robomme_integration.training.move_workspace_dense_v2_artifact import load_completion_claim
from robomme_integration.training.workspace_gpu_producer_dense_v2 import (
    validate_manifest,
    validate_runtime_environment,
)
from robomme_integration.training.workspace_supervision_cache import grounded_patch_id
from robomme_integration.training.workspace_supervision_dense_v2 import (
    ARTIFACT,
    GroundedPoint,
    chronological_dense_events,
    dense_patch_distribution,
    ordered_grounded_points,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pick at <10, 20>", [(10, 20)]),
        ("move from <9, 7> to <220, 201>", [(9, 7), (220, 201)]),
        ("visit <1,2>, then <33, 44>, finally <255,255>", [(1, 2), (33, 44), (255, 255)]),
    ],
)
def test_one_two_and_multiple_points_are_all_retained_in_text_order(text, expected) -> None:
    points = ordered_grounded_points(text)
    assert [(point.x, point.y) for point in points] == expected
    assert [point.order for point in points] == list(range(len(expected)))


def test_historical_move_failure_is_exactly_the_legacy_two_point_guard() -> None:
    with pytest.raises(ValueError, match="must contain exactly one point"):
        grounded_patch_id("move from <9, 7> to <220, 201>")


def test_wrong_coordinate_order_is_not_canonicalized_away() -> None:
    forward = chronological_dense_events(["move"], ["move from <9, 7> to <220, 201>"])
    reverse = chronological_dense_events(["move"], ["move from <220, 201> to <9, 7>"])
    assert [role["point_xy"] for role in forward[0]["roles"]] == [[9, 7], [220, 201]]
    assert [role["point_xy"] for role in reverse[0]["roles"]] == [[220, 201], [9, 7]]
    assert forward[0]["grounded_subgoal_sha256"] != reverse[0]["grounded_subgoal_sha256"]
    assert forward != reverse


def test_repeated_identical_segment_is_anchored_once_but_all_roles_survive() -> None:
    events = chronological_dense_events(
        ["move", "move", "place"],
        [
            "move from <10, 20> to <200, 220>",
            "move from <10, 20> to <200, 220>",
            "place at <100, 120>",
        ],
    )
    assert [event["anchor_step"] for event in events] == [0, 2]
    assert [event["target_count"] for event in events] == [2, 1]
    assert [role["target_index"] for event in events for role in event["roles"]] == [0, 1, 2]


def test_dense_target_is_normalized_full_grid_and_coordinate_sensitive() -> None:
    a = dense_patch_distribution(GroundedPoint(order=0, x=10, y=20))
    b = dense_patch_distribution(GroundedPoint(order=0, x=220, y=201))
    assert a.shape == b.shape == (64,)
    assert np.all(a > 0) and np.all(b > 0)
    assert float(a.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(b.sum()) == pytest.approx(1.0, abs=1e-6)
    assert int(a.argmax()) != int(b.argmax())
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("text", ["<256, 1>", "<-1, 1>", "<1, 999>"])
def test_invalid_or_unparsed_coordinates_fail_closed(text) -> None:
    with pytest.raises(ValueError, match="outside"):
        ordered_grounded_points(text)


@pytest.mark.parametrize(
    "text",
    ["move <x, 1> to <2, 3>", "move <1> to <2, 3>", "move <1, 2 to <2, 3>"],
)
def test_malformed_coordinate_like_token_fails_without_retaining_only_valid_subset(text) -> None:
    with pytest.raises(ValueError, match="malformed"):
        ordered_grounded_points(text)


def test_grounding_targets_cannot_change_encoder_history() -> None:
    sampler = train_v2.DenseWorkspaceBatchSampler.__new__(train_v2.DenseWorkspaceBatchSampler)
    sampler.history_stride = 1
    sampler.max_history = 3
    sampler.state_mean = np.zeros((8,), dtype=np.float32)
    sampler.state_std = np.ones((8,), dtype=np.float32)
    base = {
        "frame_mean_f16": np.arange(4 * train_v2.FEATURE_DIM, dtype=np.float32).reshape(4, train_v2.FEATURE_DIM),
        "state_f32": np.arange(32, dtype=np.float32).reshape(4, 8),
        "target_point_xy_i16": np.asarray([[10, 20], [200, 220]], dtype=np.int16),
        "target_attention_f32": np.ones((2, 64), dtype=np.float32) / 64,
    }
    sampler._load = lambda _episode: base
    before = sampler.history(1, 3, mask_current=False)
    base["target_point_xy_i16"][:] = [[200, 220], [10, 20]]
    base["target_attention_f32"][:] = np.eye(2, 64, dtype=np.float32)
    after = sampler.history(1, 3, mask_current=False)
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert before[0].shape[-1] == train_v2.FEATURE_DIM + 8


def test_v2_is_a_new_visreg_only_identity_with_v4_hyperparameters() -> None:
    args = train_v2._parser().parse_args(
        ["--task", "MoveCube", "--supervision-root", "/tmp/s", "--output-root", "/tmp/o"]
    )
    assert args.learning_rate == 3e-4
    assert args.weight_decay == 1e-6
    assert args.clip_gradient_norm == 10.0
    assert args.ema_decay == 0.999
    assert args.sigreg_weight == 0.0
    assert args.attention_weight == 0.1
    assert args.visreg_weight == 0.05
    assert args.visreg_slices == 128
    assert args.mask_probability == 0.2
    assert train_v2.PROTOCOL.endswith("dense_multipoint_visreg_v2")
    assert ARTIFACT.endswith("dense_multipoint_supervision_v2")


def test_v2_source_does_not_import_or_publish_legacy_workspace_identity() -> None:
    supervision_source = inspect.getsource(
        __import__(
            "robomme_integration.training.workspace_supervision_dense_v2",
            fromlist=["unused"],
        )
    )
    assert "superseded_not_mutated" in supervision_source
    assert 'ARTIFACT = "robomme_wsm_dense_multipoint_supervision_v2"' in supervision_source
    assert "uniform_gpu_v1" not in inspect.getsource(train_v2)


def test_visreg_is_deterministic_sample_count_invariant_and_sigreg_absent() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    base = jax.random.normal(jax.random.key(2), (128, 32))
    doubled = jnp.concatenate([base, base], axis=0)
    one = float(train_v2.visreg_loss(base, jax.random.key(7), num_slices=128))
    again = float(train_v2.visreg_loss(base, jax.random.key(7), num_slices=128))
    two = float(train_v2.visreg_loss(doubled, jax.random.key(7), num_slices=128))
    assert one == again
    assert one == pytest.approx(two, rel=0.05)
    source = inspect.getsource(train_v2.dense_v2_loss_and_metrics)
    assert 'weights.get("sigreg", None) != 0.0' in source
    assert "sigreg_epps_pulley" not in source


def test_v2_ema_update_is_exact_and_nontrivial() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")
    from robomme_integration.training.workspace_deliberative import _train_step_ema

    params = {"w": jnp.asarray([1.0], dtype=jnp.float32)}
    ema = {"w": jnp.asarray([0.0], dtype=jnp.float32)}
    optimizer = optax.sgd(0.1)
    state = optimizer.init(params)

    def loss_function(current, batch, *, weights):
        loss = jnp.square(current["w"] - batch["target"]).sum()
        return loss, {"loss": loss}

    step = jax.pmap(
        lambda p, o, e, b: _train_step_ema(
            p,
            o,
            e,
            b,
            optimizer=optimizer,
            weights={},
            loss_function=loss_function,
            ema_decay=0.999,
        ),
        axis_name="devices",
        in_axes=(None, None, None, 0),
        out_axes=(None, None, None, 0),
    )
    updated, _state, average, _metrics = step(params, state, ema, {"target": np.asarray([[0.0]], dtype=np.float32)})
    assert float(updated["w"][0]) == pytest.approx(0.8)
    assert float(average["w"][0]) == pytest.approx(0.0008)

    from robomme_integration.training.workspace_deliberative import checkpoint_completion_payload

    completion = checkpoint_completion_payload(
        step=2,
        run_config_sha256="a" * 64,
        embedded_hashes={"RUN_CONFIG.json": "b" * 64},
        parameter_source="ema",
        ema_decay=0.999,
    )
    assert completion["parameter_source"] == "ema"
    assert completion["ema_decay"] == 0.999


def test_legacy_completion_payload_remains_byte_compatible_without_v2_ema_fields() -> None:
    from robomme_integration.training.workspace_deliberative import checkpoint_completion_payload

    embedded = {"WSM_BEST.json": "b" * 64, "WSM_RUN_CONFIG.json": "c" * 64}
    completion = checkpoint_completion_payload(
        step=100,
        run_config_sha256="a" * 64,
        embedded_hashes=embedded,
    )
    frozen_v1_shape = {
        "schema_version": 1,
        "step": 100,
        "run_config_sha256": "a" * 64,
        "embedded_sha256": embedded,
    }
    assert completion == frozen_v1_shape
    assert (
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
        == json.dumps(frozen_v1_shape, indent=2, sort_keys=True) + "\n"
    )


def test_legacy_materializer_identity_has_no_v2_parameter_source_field() -> None:
    from robomme_integration.training.workspace_materialize import _encoder_identity

    identity = _encoder_identity(
        run_config_sha256="a" * 64,
        checkpoint_step=100,
        checkpoint_tree_sha256="b" * 64,
        materializer_sha256="c" * 64,
    )
    assert identity == {
        "schema_version": 1,
        "run_config_sha256": "a" * 64,
        "checkpoint_step": 100,
        "checkpoint_tree_sha256": "b" * 64,
        "materializer_sha256": "c" * 64,
    }
    v2_identity = _encoder_identity(
        run_config_sha256="a" * 64,
        checkpoint_step=100,
        checkpoint_tree_sha256="b" * 64,
        materializer_sha256="c" * 64,
        parameter_source="ema",
    )
    assert v2_identity == {**identity, "parameter_source": "ema"}


def test_dense_attention_distribution_is_an_explicit_trained_target() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    params = train_v2.init_params_dense_v2(jax.random.key(3))
    batch = {
        "history": np.zeros((2, 3, train_v2.INPUT_DIM), dtype=np.float32),
        "history_mask": np.ones((2, 3), dtype=np.bool_),
        "event_target": np.ones((2, train_v2.MAX_EVENTS, train_v2.FEATURE_DIM), dtype=np.float32),
        "event_attention": np.zeros((2, train_v2.MAX_EVENTS, 64), dtype=np.float32),
        "event_presence": np.zeros((2, train_v2.MAX_EVENTS), dtype=np.float32),
        "future_target": np.ones((2, train_v2.FEATURE_DIM), dtype=np.float32),
    }
    batch["event_attention"][:, 0, 7] = 1.0
    batch["event_presence"][:, 0] = 1.0
    weights = {
        "occ": 0.1,
        "attention": 0.1,
        "jepa": 0.1,
        "sigreg": 0.0,
        "visreg": 0.05,
        "visreg_slices": 8,
        "visreg_scale": 1.0,
        "visreg_shape": 1.0,
        "visreg_center": 1.0,
    }
    (_loss, metrics), gradients = jax.value_and_grad(train_v2.dense_v2_loss_and_metrics, has_aux=True)(
        params, {key: jnp.asarray(value) for key, value in batch.items()}, weights=weights
    )
    assert float(metrics["attention"]) > 0
    assert float(jnp.linalg.norm(gradients["dense_attention_w"])) > 0
    assert gradients["dense_attention_w"].shape == (train_v2.FEATURE_DIM, 64)


def test_review_packet_is_p5_h100_task_bound_and_has_no_submission_authority() -> None:
    source = Path(launch_v2.__file__).resolve().parent
    plan = launch_v2.build_plan(source)
    manifest = plan["manifest"]
    validate_manifest(
        manifest,
        manifest_sha256=plan["manifest_sha256"],
        expected_source_tree_sha256=plan["source_tree_sha256"],
        expected_entry_sha256=manifest["source"]["entry_sha256"],
    )
    validate_runtime_environment(
        manifest,
        manifest_sha256=plan["manifest_sha256"],
        environment=plan["environment"],
        expected_entry_sha256=manifest["source"]["entry_sha256"],
    )
    assert manifest["identity"]["task"] == "MoveCube"
    assert manifest["infrastructure"]["accelerator"] == "8xH100-80GB-HBM3"
    assert manifest["infrastructure"]["training_plan_arn"] is None
    assert plan["environment"]["SM_USE_RESERVED_CAPACITY"] == "1"
    assert manifest["scientific"]["representation"]["devices"] == 8
    assert manifest["scientific"]["representation"]["batch_size"] == 64
    assert manifest["scientific"]["representation"]["loss_weights"]["dense_attention"] == 0.1
    assert manifest["scientific"]["representation"]["ema_decay"] == 0.999
    assert manifest["submission_gate"]["this_packet_authorizes_submission"] is False
    assert manifest["submission_gate"]["required_receipts"] == [
        "independent_source_protocol_review",
        "task_bound_8xH100_gpu_canary",
    ]
    assert "submit_training_job" not in inspect.getsource(launch_v2)
    assert "workspace_dense_multipoint_visreg_v2" in plan["environment"]["ROBOMME_MOVE_DENSE_V2_ARTIFACT_ROOT_S3"]
    entry = Path(launch_v2.__file__).with_name("gpu_move_workspace_dense_v2_entry.sh").read_text()
    assert "MOVE_DENSE_V2_SOURCE_OK" in entry
    assert "if path.relative_to(root).as_posix() != excluded" in entry
    assert "stat.S_IMODE(entry_path.stat().st_mode) != 0o777" in entry
    assert "mode = 0o755 if relative == entry" in entry
    assert entry.index("MOVE_DENSE_V2_MANIFEST_ENV_OK") < entry.index('publish_once "$MANIFEST"')

    # Numerically reproduce the on-node runtime hash: SageMaker chmods only the selected entry to
    # 0777, which must be required then normalized back to its staged 0755 identity.
    with launch_v2.prepared_source_bundle(source, launch_v2.ENTRY, {}, None) as (
        staged,
        _entry,
        _environment,
    ):
        (staged / launch_v2.STAGED_MANIFEST).write_text(plan["manifest_json"])
        runtime_entry = staged / launch_v2.ENTRY
        runtime_entry.chmod(0o777)
        digest = hashlib.sha256()

        def field(value):
            data = value if isinstance(value, bytes) else str(value).encode()
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)

        paths = [
            path for path in staged.rglob("*") if path.relative_to(staged).as_posix() != launch_v2.STAGED_MANIFEST
        ]
        for path in sorted(paths, key=lambda item: item.relative_to(staged).as_posix()):
            relative = path.relative_to(staged).as_posix()
            field(relative)
            mode = 0o755 if relative == launch_v2.ENTRY else stat.S_IMODE(path.lstat().st_mode)
            field(oct(mode))
            if path.is_symlink():
                field("symlink")
                field(os.readlink(path))
            elif path.is_dir():
                field("directory")
            else:
                field("file")
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        field(block)
        assert stat.S_IMODE(runtime_entry.stat().st_mode) == 0o777
        assert digest.hexdigest() == plan["source_tree_sha256"]
        assert hashlib.sha256(runtime_entry.read_bytes()).hexdigest() == manifest["source"]["entry_sha256"]


def test_manifest_and_receipt_mutations_fail_closed() -> None:
    source = Path(launch_v2.__file__).resolve().parent
    plan = launch_v2.build_plan(source)
    manifest = plan["manifest"]
    mutated = json.loads(json.dumps(manifest))
    mutated["scientific"]["representation"]["loss_weights"]["sigreg"] = 0.05
    with pytest.raises(ValueError, match="manifest SHA"):
        validate_manifest(
            mutated,
            manifest_sha256=plan["manifest_sha256"],
            expected_source_tree_sha256=plan["source_tree_sha256"],
            expected_entry_sha256=manifest["source"]["entry_sha256"],
        )
    for path, value, message in (
        (("infrastructure", "accelerator"), "8xH200", "infrastructure"),
        (("identity", "scientific_spec_sha256"), "0" * 64, "identity"),
        (("source", "source_tree_sha256"), "1" * 64, "trusted expected source"),
        (("source", "entry_sha256"), "0" * 64, "trusted expected entry"),
    ):
        adversarial = json.loads(json.dumps(manifest))
        adversarial[path[0]][path[1]] = value
        resealed = hashlib.sha256(
            (json.dumps(adversarial, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        with pytest.raises(ValueError, match=message):
            validate_manifest(
                adversarial,
                manifest_sha256=resealed,
                expected_source_tree_sha256=plan["source_tree_sha256"],
                expected_entry_sha256=manifest["source"]["entry_sha256"],
            )

    environment = dict(plan["environment"])
    environment["ROBOMME_DATA_S3"] = "s3://wrong/data"
    with pytest.raises(ValueError, match="runtime environment drifted"):
        validate_runtime_environment(
            manifest,
            manifest_sha256=plan["manifest_sha256"],
            environment=environment,
            expected_entry_sha256=manifest["source"]["entry_sha256"],
        )

    receipt = {
        "schema_version": 2,
        "kind": canary_v2.KIND,
        "evidence_class": "operational_canary_not_scientific_evidence",
        "identity": {
            "campaign": manifest["identity"]["campaign"],
            "task": "MoveCube",
            "run_id": manifest["identity"]["run_id"],
            "attempt_id": manifest["identity"]["attempt_id"],
            "manifest_sha256": plan["manifest_sha256"],
            "source_tree_sha256": manifest["source"]["source_tree_sha256"],
        },
        "protocol": {
            "name": train_v2.PROTOCOL,
            "supervision_artifact": ARTIFACT,
            "regularizer": "visreg",
            "visreg_weight": 0.05,
            "visreg_slices": 128,
            "visreg_components": {"scale": 1.0, "shape": 1.0, "center": 1.0},
            "sigreg_weight": 0.0,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 3e-4,
                "weight_decay": 1e-6,
                "global_gradient_clip": 10.0,
                "ema_decay": 0.999,
            },
        },
        "runtime": {
            "platform": "gpu",
            "device_count": 8,
            "device_kinds": ["NVIDIA H100 80GB HBM3"] * 8,
        },
        "proof": {
            "grounding": {
                "point_counts": [1, 2, 3],
                "dense_target_sha256": ["a" * 64, "b" * 64, "c" * 64],
                "wrong_order_changes_roles": True,
                "all_grid_weights_positive": True,
                "all_grid_weights_normalized": True,
            },
            "visreg_value": 1.0,
            "visreg_doubled_value": 1.0,
            "visreg_gradient_norm": 0.5,
            "sample_count_invariant": True,
            "sigreg_executed": False,
            "optimizer_steps": 2,
            "actual_dense_v2_batch_size": 64,
            "raw_update_norm": 1.0,
            "ema_update_norm": 0.001,
            "ema_raw_delta": 0.999,
            "dense_attention_update_norm": 0.1,
            "encoder_input_update_norm": 0.1,
            "step2_metrics": {"grad_norm": 1.0, "attention": 4.0},
            "checkpoint_written": False,
            "cloud_artifact_published": False,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(canary_v2._canonical(receipt)).hexdigest()
    canary_v2.validate_receipt(receipt, manifest=manifest, manifest_sha256=plan["manifest_sha256"])
    mutated_manifest = json.loads(json.dumps(manifest))
    mutated_manifest["infrastructure"]["accelerator"] = "8xA10G"
    mutated_manifest["submission_gate"]["this_packet_authorizes_submission"] = True
    with pytest.raises(ValueError, match="manifest argument SHA-256"):
        canary_v2.validate_receipt(
            receipt,
            manifest=mutated_manifest,
            manifest_sha256=plan["manifest_sha256"],
        )
    wrong_runtime = json.loads(json.dumps(receipt))
    wrong_runtime["runtime"]["device_kinds"] = ["NVIDIA A10G"] * 8
    value = dict(wrong_runtime)
    value.pop("receipt_sha256")
    wrong_runtime["receipt_sha256"] = hashlib.sha256(canary_v2._canonical(value)).hexdigest()
    with pytest.raises(ValueError, match="not exact 8xH100"):
        canary_v2.validate_receipt(wrong_runtime, manifest=manifest, manifest_sha256=plan["manifest_sha256"])
    for mutation, message in (
        (("runtime", "device_kinds", ["NVIDIA H100 FAKE"] * 8), "not exact 8xH100"),
        (("proof", "visreg_value", float("nan")), "proof mismatch"),
        (("proof", "visreg_doubled_value", float("inf")), "proof mismatch"),
        (("proof", "dense_attention_update_norm", float("nan")), "proof mismatch"),
        (("proof", "encoder_input_update_norm", float("nan")), "proof mismatch"),
        (("proof", "step2_metrics", {"grad_norm": float("nan"), "attention": 4.0}), "proof mismatch"),
        (("proof", "grounding", {**receipt["proof"]["grounding"], "dense_target_sha256": []}), "proof mismatch"),
    ):
        adversarial_receipt = json.loads(json.dumps(receipt))
        adversarial_receipt[mutation[0]][mutation[1]] = mutation[2]
        value = dict(adversarial_receipt)
        value.pop("receipt_sha256")
        adversarial_receipt["receipt_sha256"] = hashlib.sha256(canary_v2._canonical(value)).hexdigest()
        with pytest.raises(ValueError, match=message):
            canary_v2.validate_receipt(
                adversarial_receipt,
                manifest=manifest,
                manifest_sha256=plan["manifest_sha256"],
            )
    receipt["protocol"]["sigreg_weight"] = 0.05
    with pytest.raises(ValueError, match="receipt SHA"):
        canary_v2.validate_receipt(receipt, manifest=manifest, manifest_sha256=plan["manifest_sha256"])


def test_v2_completion_claim_cannot_be_aliased_to_legacy_or_sigreg(tmp_path) -> None:
    source = Path(launch_v2.__file__).resolve().parent
    plan = launch_v2.build_plan(source)
    manifest = plan["manifest"]
    encoder_id = "3" * 64
    artifact_root = (
        "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/"
        f"studies/long_context_v1/artifacts/robomme/workspace_dense_multipoint_visreg_v2/MoveCube/{encoder_id}"
    )
    claim = {
        "schema_version": 2,
        "kind": "robomme_move_workspace_dense_v2_complete",
        "campaign": "move_workspace_dense_multipoint_visreg_v2",
        "task": "MoveCube",
        "run_id": manifest["identity"]["run_id"],
        "scientific_spec_sha256": manifest["identity"]["scientific_spec_sha256"],
        "source_tree_sha256": manifest["source"]["source_tree_sha256"],
        "task_manifest_sha256": "4779e982dadd841c481f667c8ca578da8a080ef15873e10e84fbab3a4dde2dda",
        "protocol": train_v2.PROTOCOL,
        "supervision_artifact": ARTIFACT,
        "target_semantics": "ordered_grounded_roles_dense_gaussian_8x8_v2",
        "regularizer": {"name": "visreg", "weight": 0.05, "sigreg_weight": 0.0},
        "ema_decay": 0.999,
        "encoder_id": encoder_id,
        "omega": {"uri": f"{artifact_root}/omega", "manifest_sha256": "4" * 64},
        "supervision": {
            "uri": f"{artifact_root}/supervision",
            "manifest_sha256": "5" * 64,
        },
        "representation": {
            "uri": f"{artifact_root}/representation/step-10000",
            "step": 10_000,
            "completion_sha256": "6" * 64,
        },
    }
    path = tmp_path / "claim.json"
    path.write_text(json.dumps(claim, sort_keys=True) + "\n")
    assert (
        load_completion_claim(
            path,
            expected_manifest=manifest,
            expected_manifest_sha256=plan["manifest_sha256"],
        )["encoder_id"]
        == encoder_id
    )
    claim["regularizer"] = {"name": "sigreg", "weight": 0.05, "sigreg_weight": 0.05}
    path.write_text(json.dumps(claim, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="protocol/identity"):
        load_completion_claim(
            path,
            expected_manifest=manifest,
            expected_manifest_sha256=plan["manifest_sha256"],
        )
