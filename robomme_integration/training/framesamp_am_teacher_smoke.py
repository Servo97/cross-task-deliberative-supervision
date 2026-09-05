"""No-simulator released-checkpoint smoke for the E1 teacher-tap producer.

This command is intentionally fixture-driven.  The fixture must contain the
exact transformed, batched policy observation from an on-policy causal cut,
the initial sampler noise, and actions produced by the unmodified released
checkpoint from those same bytes.  No initial-demo or teacher-trajectory
substitution is allowed.

The command is expensive (it restores the 79999 checkpoint and executes full
10-step action samples), so repository tests cover its contracts with small
synthetic modules.  Run it once on a GPU before claiming the E1 producer is
evaluation-ready.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import TextIO

import numpy as np

from robomme_integration.training.framesamp_am_teacher_producer import (
    RELEASED_CHECKPOINT_SHA256,
    FrameSampTeacherCaptureIdentity,
    SourcePinnedFrameSampTeacherProvider,
    assert_full_action_parity,
    build_framesamp_teacher_query_plan,
    prepare_framesamp_teacher_forward,
    produce_framesamp_teacher_stack,
    verify_source_pinned_released_teacher,
    write_teacher_capture_receipt,
)
from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    TOKEN_BUDGET,
    FrameSampHistory,
)

FIXTURE_SCHEMA_VERSION = 3
FIXTURE_KIND = "robomme_framesamp_am_teacher_parity_fixture"
PARITY_ONLY_SCOPE = "sealed_causal_dataset_source_checkpoint_parity_only_not_online_rollout_attestation"
STATIC_IMAGE_BFLOAT16_ENCODING = "bfloat16_uint16_bits_v1"
STATIC_IMAGE_FLOAT32_ENCODING = "native_float32_v1"
PHASE_LOG_KIND = "robomme_framesamp_am_teacher_smoke_phase"


def _log_phase(
    phase: str,
    *,
    started_at: float,
    stream: TextIO | None = None,
    clock=time.monotonic,
    **fields: str | int | float | bool | None,
) -> None:
    """Emit one bounded, immediately visible progress record to stderr."""

    if not isinstance(phase, str) or not phase:
        raise ValueError("smoke phase must be a nonempty string")
    if any(not isinstance(key, str) or not key for key in fields):
        raise ValueError("smoke phase field names must be nonempty strings")
    elapsed = float(clock()) - float(started_at)
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("smoke phase elapsed time must be finite and nonnegative")
    payload = {
        "elapsed_seconds": round(elapsed, 3),
        "kind": PHASE_LOG_KIND,
        "phase": phase,
        **fields,
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False), file=stream or sys.stderr, flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid smoke fixture manifest: {path}") from error
    required = {
        "schema_version",
        "kind",
        "attestation_scope",
        "task_id",
        "episode_id",
        "causal_cut_step",
        "source_fixture_id",
        "source_chunk_role",
        "source_bundle_manifest_sha256",
        "source_bundle_content_sha256",
        "source_bundle_payload_sha256",
        "source_record_metadata",
        "teacher_checkpoint_sha256",
        "teacher_code_sha",
        "model_seed",
        "sampler_seed",
        "sampler_key_seed",
        "num_steps",
        "static_image_emb_encoding",
        "observation_npz",
        "observation_npz_sha256",
        "official_actions_npy",
        "official_actions_npy_sha256",
        "sampler_noise_npy",
        "sampler_noise_npy_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("smoke fixture manifest fields mismatch")
    if value["schema_version"] != FIXTURE_SCHEMA_VERSION or value["kind"] != FIXTURE_KIND:
        raise ValueError("unsupported smoke fixture schema/kind")
    if value["attestation_scope"] != PARITY_ONLY_SCOPE:
        raise ValueError("teacher smoke fixture must be labelled parity-only, never online-attested")
    if value["teacher_checkpoint_sha256"] != RELEASED_CHECKPOINT_SHA256:
        raise ValueError("teacher smoke fixture does not bind the released FrameSamp checkpoint")
    if value["static_image_emb_encoding"] not in {
        STATIC_IMAGE_BFLOAT16_ENCODING,
        STATIC_IMAGE_FLOAT32_ENCODING,
    }:
        raise ValueError("teacher smoke fixture has an unsupported static-image encoding")
    if not isinstance(value["causal_cut_step"], int) or isinstance(value["causal_cut_step"], bool):
        raise ValueError("smoke fixture causal_cut_step must be an integer")
    if not isinstance(value["sampler_key_seed"], int) or isinstance(value["sampler_key_seed"], bool):
        raise ValueError("smoke fixture sampler_key_seed must be an integer")
    return value


def _resolve_payload(manifest_path: Path, relative: object, expected_sha: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("fixture payload paths must be normalized relative paths")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("fixture payload SHA must be a lowercase SHA256")
    value = (manifest_path.parent / relative).resolve(strict=True)
    try:
        value.relative_to(manifest_path.parent.resolve(strict=True))
    except ValueError as error:
        raise ValueError("fixture payload escaped its manifest directory") from error
    if _sha256_file(value) != expected_sha:
        raise ValueError(f"fixture payload SHA mismatch: {relative}")
    return value


def _load_observation_fixture(manifest_path: Path):
    manifest = _load_manifest(manifest_path)
    observation_path = _resolve_payload(
        manifest_path,
        manifest["observation_npz"],
        manifest["observation_npz_sha256"],
    )
    actions_path = _resolve_payload(
        manifest_path,
        manifest["official_actions_npy"],
        manifest["official_actions_npy_sha256"],
    )
    noise_path = _resolve_payload(
        manifest_path,
        manifest["sampler_noise_npy"],
        manifest["sampler_noise_npy_sha256"],
    )
    with np.load(observation_path, allow_pickle=False) as payload:
        names = set(payload.files)
        image_names = sorted(name.removeprefix("image__") for name in names if name.startswith("image__"))
        mask_names = sorted(name.removeprefix("image_mask__") for name in names if name.startswith("image_mask__"))
        required = {
            "state",
            "static_image_emb",
            "static_mask",
            "static_pos_emb",
            "static_state_emb",
            "frame_indices",
            "frame_mask",
        }
        optional = {
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
        }
        dynamic = {f"image__{name}" for name in image_names} | {f"image_mask__{name}" for name in mask_names}
        if not image_names or image_names != mask_names or not required.issubset(names):
            raise ValueError("smoke observation payload has incomplete image/history fields")
        if names - required - optional - dynamic:
            raise ValueError("smoke observation payload has unknown fields")
        arrays = {name: np.asarray(payload[name]) for name in names}

    encoding = manifest["static_image_emb_encoding"]
    stored_static_image = arrays["static_image_emb"]
    if encoding == STATIC_IMAGE_BFLOAT16_ENCODING:
        if stored_static_image.dtype != np.uint16:
            raise ValueError("bfloat16 static-image payload must be stored as uint16 bits")
        import ml_dtypes

        arrays["static_image_emb"] = stored_static_image.view(ml_dtypes.bfloat16)
    elif stored_static_image.dtype != np.float32:
        raise ValueError("native float32 static-image payload has the wrong dtype")

    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    observation = HistAugObservation(
        images={name: jnp.asarray(arrays[f"image__{name}"]) for name in image_names},
        image_masks={name: jnp.asarray(arrays[f"image_mask__{name}"]) for name in mask_names},
        state=jnp.asarray(arrays["state"]),
        tokenized_prompt=(jnp.asarray(arrays["tokenized_prompt"]) if "tokenized_prompt" in arrays else None),
        tokenized_prompt_mask=(
            jnp.asarray(arrays["tokenized_prompt_mask"]) if "tokenized_prompt_mask" in arrays else None
        ),
        token_ar_mask=(jnp.asarray(arrays["token_ar_mask"]) if "token_ar_mask" in arrays else None),
        token_loss_mask=(jnp.asarray(arrays["token_loss_mask"]) if "token_loss_mask" in arrays else None),
        static_image_emb=jnp.asarray(arrays["static_image_emb"]),
        static_mask=jnp.asarray(arrays["static_mask"]),
        static_pos_emb=jnp.asarray(arrays["static_pos_emb"]),
        static_state_emb=jnp.asarray(arrays["static_state_emb"]),
    )
    if arrays["static_image_emb"].shape[0] != 1 or arrays["static_mask"].shape != (1, TOKEN_BUDGET):
        raise ValueError("smoke fixture must contain one batched 512-token history")
    history = FrameSampHistory(
        image=arrays["static_image_emb"][0].astype(np.float32),
        position=arrays["static_pos_emb"][0].astype(np.float32),
        token_mask=arrays["static_mask"][0].astype(np.bool_),
        frame_indices=arrays["frame_indices"].astype(np.int32),
        frame_mask=arrays["frame_mask"].astype(np.bool_),
    )
    if history.frame_indices.shape != (MAX_FRAMES,) or history.frame_mask.shape != (MAX_FRAMES,):
        raise ValueError("smoke fixture frame map must retain all 32 physical slots")
    history.validate()
    official_actions = np.load(actions_path, allow_pickle=False)
    sampler_noise = np.load(noise_path, allow_pickle=False)
    return manifest, observation, history, official_actions, sampler_noise


def _require_import_below_overlay(value: object, overlay: Path, *, label: str) -> None:
    source = Path(inspect.getfile(value)).resolve(strict=True)
    try:
        source.relative_to(overlay)
    except ValueError as error:
        raise ValueError(f"{label} was not imported from the verified policy overlay") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--parity-only",
        action="store_true",
        help="stop after released-action parity; do not capture or write teacher taps",
    )
    parser.add_argument("--require-full-action-parity", action="store_true", required=True)
    args = parser.parse_args(argv)
    if not args.parity_only and args.output is None:
        parser.error("--output is required unless --parity-only is set")

    started_at = time.monotonic()
    _log_phase(
        "source_overlay_verification_start",
        started_at=started_at,
        overlay_manifest_sha256=args.overlay_manifest_sha256,
    )
    provenance = verify_source_pinned_released_teacher(
        policy_overlay=args.policy_overlay,
        expected_policy_overlay_manifest_sha256=args.overlay_manifest_sha256,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
    )
    _log_phase(
        "source_overlay_verification_end",
        started_at=started_at,
        overlay_manifest_sha256=args.overlay_manifest_sha256,
        source_status="pre_staged_verified",
    )
    overlay_src = args.policy_overlay.resolve(strict=True) / "src"
    sys.path.insert(0, str(overlay_src))

    _log_phase("accelerator_discovery_start", started_at=started_at)
    import jax
    from mme_vla_suite.models.integration.history_observation import preprocess_observation
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, make_attn_mask
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config
    from openpi.shared import nnx_utils

    devices = jax.devices()
    _log_phase(
        "accelerator_discovery_end",
        started_at=started_at,
        device_count=len(devices),
        platforms=",".join(sorted({device.platform for device in devices})),
    )
    _require_import_below_overlay(HistoryPi0, args.policy_overlay.resolve(strict=True), label="HistoryPi0")
    manifest, observation, history, official_actions, sampler_noise = _load_observation_fixture(
        args.fixture.resolve(strict=True)
    )
    if manifest["teacher_code_sha"] != provenance["teacher_code_sha"]:
        raise ValueError("teacher smoke fixture code SHA differs from the verified official teacher")
    # The released history-config loader resolves its YAML relative to the
    # source root.  Bind the verified overlay root explicitly so this command
    # is independent of the caller's working directory.
    previous_cwd = Path.cwd()
    _log_phase(
        "checkpoint_restore_start",
        started_at=started_at,
        checkpoint_step=args.checkpoint.name,
    )
    os.chdir(args.policy_overlay.resolve(strict=True))
    try:
        policy = policy_config.create_trained_policy(
            config.get_config("mme_vla_suite"),
            args.checkpoint,
            seed=7,
            default_prompt=None,
        )
    finally:
        os.chdir(previous_cwd)
    _log_phase(
        "checkpoint_restore_end",
        started_at=started_at,
        checkpoint_step=args.checkpoint.name,
    )
    model = policy._model  # noqa: SLF001 - official loader exposes no model accessor.
    sampler = nnx_utils.module_jit(model.sample_actions)
    _log_phase(
        "first_jit_sample_start",
        started_at=started_at,
        num_steps=10,
        sampler_key_seed=int(manifest["sampler_key_seed"]),
    )
    overlay_actions = np.asarray(
        sampler(
            jax.random.key(int(manifest["sampler_key_seed"])),
            observation,
            num_steps=10,
            noise=sampler_noise,
        )
    )
    _log_phase(
        "first_jit_sample_end",
        started_at=started_at,
        action_elements=int(overlay_actions.size),
    )
    delta = np.abs(overlay_actions.astype(np.float32) - official_actions.astype(np.float32))
    bitwise_equal = bool(np.array_equal(official_actions, overlay_actions))
    _log_phase(
        "action_parity_comparison",
        started_at=started_at,
        bitwise_equal=bitwise_equal,
        max_abs=float(delta.max(initial=0.0)),
        mean_abs=float(delta.mean()) if delta.size else 0.0,
        nonzero_elements=int(np.count_nonzero(delta)),
    )
    assert_full_action_parity(official_actions, overlay_actions)
    if args.parity_only:
        print(
            json.dumps(
                {
                    "status": "released_checkpoint_full_action_parity_passed",
                    "bitwise_equal": True,
                    "overlay_manifest_sha256": args.overlay_manifest_sha256,
                },
                sort_keys=True,
            )
        )
        return 0

    identity = FrameSampTeacherCaptureIdentity.from_history(
        history,
        teacher_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
        teacher_code_sha=str(provenance["teacher_code_sha"]),
        policy_overlay_manifest_sha256=args.overlay_manifest_sha256,
        task_id=str(manifest["task_id"]),
        episode_id=str(manifest["episode_id"]),
        causal_cut_step=int(manifest["causal_cut_step"]),
    )
    prepared = prepare_framesamp_teacher_forward(
        model,
        observation,
        history,
        identity,
        preprocess_observation=preprocess_observation,
        make_attn_mask=make_attn_mask,
    )
    provider = SourcePinnedFrameSampTeacherProvider(
        prepared,
        observation,
        sample_actions=sampler,
        verify_first_suffix_parity=True,
    )
    plan = build_framesamp_teacher_query_plan(
        identity,
        diffusion_timesteps=(0.5,),
        fit_split_seed=101,
        heldout_split_seed=211,
        fit_noise_samples_per_timestep=1,
        heldout_noise_samples_per_timestep=1,
    )
    stack = produce_framesamp_teacher_stack(plan, history, provider)
    assert args.output is not None  # guarded by argparse above.
    receipt_sha = write_teacher_capture_receipt(args.output, stack, current_history=history)
    print(
        json.dumps(
            {
                "status": "released_checkpoint_full_action_parity_and_all18_taps_passed",
                "capture_identity_sha256": identity.sha256(),
                "capture_receipt_sha256": receipt_sha,
                "capture_receipt_scientific_sha256": stack.receipt.sha256(),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
