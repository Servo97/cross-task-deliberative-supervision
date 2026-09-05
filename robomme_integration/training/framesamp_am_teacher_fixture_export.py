"""Export one sealed AMKV causal record into the E1 teacher-smoke format.

The source bundle is read through its existing validator and never modified.
The released policy is loaded through the unmodified official source, exactly
as the serve path loads it.  The resulting fixture is sufficient to prove
source/checkpoint/overlay action parity and all-layer tap geometry.  It is *not*
an online-history attestation for a compressed-policy rollout: an online
rollout can diverge from this sealed dataset trajectory at the same labelled
cut and must be recaptured independently.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from robomme_integration.training.framesamp_am_teacher_producer import (
    ACTION_TOKENS,
    OFFICIAL_POLICY_GIT_SHA,
    RELEASED_CHECKPOINT_SHA256,
)
from robomme_integration.training.framesamp_am_teacher_smoke import (
    FIXTURE_KIND,
    FIXTURE_SCHEMA_VERSION,
    STATIC_IMAGE_BFLOAT16_ENCODING,
    STATIC_IMAGE_FLOAT32_ENCODING,
)
from robomme_integration.training.upstream_framesamp_data import FrameSampHistory
from wsm_settings import ROBOMME_EVAL_ROOT, WSM_DATA_ROOT

ATTESTATION_SCOPE = "sealed_causal_dataset_source_checkpoint_parity_only_not_online_rollout_attestation"
OBSERVATION_FILENAME = "observation.npz"
OFFICIAL_ACTIONS_FILENAME = "official_actions.npy"
SAMPLER_NOISE_FILENAME = "sampler_noise.npy"
MANIFEST_FILENAME = "manifest.json"
MODEL_ACTION_DIM = 32
DEFAULT_SOURCE_BUNDLE = WSM_DATA_ROOT / "wsmv2_scratch" / "amkv_fixtures_v1"
DEFAULT_OFFICIAL_CHECKOUT = ROBOMME_EVAL_ROOT / "official_reference" / "robomme_policy_learning"
DEFAULT_CHECKPOINT = ROBOMME_EVAL_ROOT / "official_reference/checkpoints/perceptual-framesamp-modul/79999"


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_git_sha(checkout: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"cannot verify official policy checkout: {error.stderr.strip()}") from error
    if result != OFFICIAL_POLICY_GIT_SHA:
        raise ValueError(f"official policy checkout is {result}, expected {OFFICIAL_POLICY_GIT_SHA}")
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all", "--", "src"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    if status:
        raise ValueError("official policy src tree is dirty; refusing to export reference actions")
    return result


def _verify_released_checkpoint(checkpoint: Path) -> None:
    checkpoint = checkpoint.resolve(strict=True)
    marker = checkpoint.parent / f".EXTRACTED-{RELEASED_CHECKPOINT_SHA256}"
    history_config = checkpoint.parent / "history_config.txt"
    if checkpoint.name != "79999" or not (checkpoint / "params").is_dir() or not marker.is_file():
        raise ValueError("teacher fixture export requires released perceptual-framesamp-modul step 79999")
    if history_config.read_text(encoding="utf-8").strip() != "perceptual-framesamp-modul.yaml":
        raise ValueError("released teacher history config is wrong")


def _require_import_below(value: object, root: Path, *, label: str) -> None:
    source = Path(inspect.getfile(value)).resolve(strict=True)
    try:
        source.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} was not imported from the verified unmodified official checkout") from error


def _observation_arrays(observation: Any, history: FrameSampHistory) -> dict[str, np.ndarray]:
    history.validate()
    value = observation.to_dict()
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("image"), dict)
        or not isinstance(value.get("image_mask"), dict)
    ):
        raise ValueError("official transformed observation has the wrong mapping structure")
    if set(value["image"]) != set(value["image_mask"]):
        raise ValueError("official transformed observation image/mask names differ")
    arrays: dict[str, np.ndarray] = {}
    for name, image in value.pop("image").items():
        arrays[f"image__{name}"] = np.ascontiguousarray(np.asarray(image))
    for name, mask in value.pop("image_mask").items():
        arrays[f"image_mask__{name}"] = np.ascontiguousarray(np.asarray(mask))
    allowed = {
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
        "static_image_emb",
        "static_mask",
        "static_pos_emb",
        "static_state_emb",
        "recur_image_emb",
        "recur_mask",
        "recur_pos_emb",
        "recur_state_emb",
        "symbolic_tokenized_prompt",
        "symbolic_tokenized_prompt_mask",
    }
    if set(value) - allowed:
        raise ValueError(f"official observation has unknown fields {sorted(set(value) - allowed)}")
    for name, array in value.items():
        if array is not None:
            if name.startswith("recur_") or name.startswith("symbolic_"):
                raise ValueError("FrameSamp teacher fixture unexpectedly contains recurrent/symbolic memory")
            arrays[name] = np.ascontiguousarray(np.asarray(array))
    arrays["frame_indices"] = np.ascontiguousarray(history.frame_indices)
    arrays["frame_mask"] = np.ascontiguousarray(history.frame_mask)
    required = {"state", "static_image_emb", "static_mask", "static_pos_emb", "static_state_emb"}
    if not required.issubset(arrays):
        raise ValueError("official transformed observation lacks required perceptual fields")
    if not np.array_equal(arrays["static_image_emb"][0].astype(np.float32), history.image):
        raise ValueError("exported observation differs from its FrameSamp image history")
    if not np.array_equal(arrays["static_pos_emb"][0].astype(np.float32), history.position):
        raise ValueError("exported observation differs from its FrameSamp position history")
    if not np.array_equal(arrays["static_mask"][0].astype(np.bool_), history.token_mask):
        raise ValueError("exported observation differs from its FrameSamp token mask")
    return arrays


def seal_teacher_smoke_fixture(
    destination: str | Path,
    *,
    observation: Any,
    history: FrameSampHistory,
    official_actions: np.ndarray,
    sampler_noise: np.ndarray,
    task_id: str,
    episode_id: str,
    causal_cut_step: int,
    source_fixture_id: str,
    source_chunk_role: str,
    source_bundle_manifest_sha256: str,
    source_bundle_content_sha256: str,
    source_bundle_payload_sha256: str,
    source_record_metadata: dict[str, object],
    teacher_checkpoint_sha256: str = RELEASED_CHECKPOINT_SHA256,
    teacher_code_sha: str = OFFICIAL_POLICY_GIT_SHA,
    model_seed: int = 7,
    sampler_seed: int = 0,
    num_steps: int = 10,
) -> dict[str, object]:
    """Atomically create one four-file, parity-only teacher smoke fixture."""

    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace teacher smoke fixture: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"teacher smoke fixture parent does not exist: {destination.parent}")
    if teacher_checkpoint_sha256 != RELEASED_CHECKPOINT_SHA256 or teacher_code_sha != OFFICIAL_POLICY_GIT_SHA:
        raise ValueError("teacher smoke fixture must use the released checkpoint and audited official code")
    if source_chunk_role not in {"fit_chunk", "eval_chunk"}:
        raise ValueError("source_chunk_role must be fit_chunk or eval_chunk")
    if not task_id or not episode_id or not source_fixture_id:
        raise ValueError("teacher smoke task/episode/source fixture ids must be nonempty")
    if isinstance(causal_cut_step, bool) or not isinstance(causal_cut_step, int) or causal_cut_step < 0:
        raise ValueError("causal_cut_step must be a nonnegative integer")
    for label, digest in (
        ("source bundle manifest", source_bundle_manifest_sha256),
        ("source bundle content", source_bundle_content_sha256),
        ("source bundle payload", source_bundle_payload_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{label} SHA must be lowercase SHA256")
    if not isinstance(source_record_metadata, dict):
        raise ValueError("source_record_metadata must be an object")
    arrays = _observation_arrays(observation, history)
    static_image = arrays["static_image_emb"]
    if static_image.dtype == np.float32:
        static_image_encoding = STATIC_IMAGE_FLOAT32_ENCODING
    elif static_image.dtype.itemsize == 2 and str(static_image.dtype) in {"bfloat16", "|V2"}:
        # NPY has no native bfloat16 descriptor.  Store the exact bits as
        # uint16 and make the encoding explicit in the sealed manifest.
        arrays["static_image_emb"] = static_image.view(np.uint16)
        static_image_encoding = STATIC_IMAGE_BFLOAT16_ENCODING
    else:
        raise ValueError(f"unsupported static-image dtype {static_image.dtype}")
    actions = np.ascontiguousarray(np.asarray(official_actions))
    noise = np.ascontiguousarray(np.asarray(sampler_noise))
    # ``HistoryPi0.sample_actions`` returns the padded model-space action
    # tensor.  The policy output transform later selects the environment's
    # eight coordinates, but this smoke compares the model itself and therefore
    # must retain all 32 dimensions.
    expected = (1, ACTION_TOKENS, MODEL_ACTION_DIM)
    if actions.shape != expected or noise.shape != expected:
        raise ValueError(f"official actions and sampler noise must each be {expected}")
    if not np.isfinite(actions.astype(np.float32)).all() or not np.isfinite(noise.astype(np.float32)).all():
        raise ValueError("teacher smoke actions/noise contain non-finite values")

    scratch = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    staged = scratch / "staged"
    staged.mkdir()
    try:
        with (staged / OBSERVATION_FILENAME).open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        with (staged / OFFICIAL_ACTIONS_FILENAME).open("wb") as stream:
            np.save(stream, actions, allow_pickle=False)
        with (staged / SAMPLER_NOISE_FILENAME).open("wb") as stream:
            np.save(stream, noise, allow_pickle=False)
        manifest = {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "kind": FIXTURE_KIND,
            "attestation_scope": ATTESTATION_SCOPE,
            "task_id": task_id,
            "episode_id": episode_id,
            "causal_cut_step": causal_cut_step,
            "source_fixture_id": source_fixture_id,
            "source_chunk_role": source_chunk_role,
            "source_bundle_manifest_sha256": source_bundle_manifest_sha256,
            "source_bundle_content_sha256": source_bundle_content_sha256,
            "source_bundle_payload_sha256": source_bundle_payload_sha256,
            "source_record_metadata": source_record_metadata,
            "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
            "teacher_code_sha": teacher_code_sha,
            "model_seed": model_seed,
            "sampler_seed": sampler_seed,
            "sampler_key_seed": sampler_seed + 1,
            "num_steps": num_steps,
            "static_image_emb_encoding": static_image_encoding,
            "observation_npz": OBSERVATION_FILENAME,
            "observation_npz_sha256": _sha256_file(staged / OBSERVATION_FILENAME),
            "official_actions_npy": OFFICIAL_ACTIONS_FILENAME,
            "official_actions_npy_sha256": _sha256_file(staged / OFFICIAL_ACTIONS_FILENAME),
            "sampler_noise_npy": SAMPLER_NOISE_FILENAME,
            "sampler_noise_npy_sha256": _sha256_file(staged / SAMPLER_NOISE_FILENAME),
        }
        (staged / MANIFEST_FILENAME).write_bytes(_canonical_json(manifest))
        for path in staged.iterdir():
            path.chmod(0o444)
        os.rename(staged, destination)
        staged = None
        return manifest
    finally:
        if staged is not None and staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)


def export_from_sealed_causal_bundle(
    *,
    source_bundle: str | Path,
    source_fixture_id: str,
    official_checkout: str | Path,
    checkpoint: str | Path,
    destination: str | Path,
    model_seed: int = 7,
    sampler_seed: int = 0,
) -> dict[str, object]:
    """Restore the official teacher and export one selected sealed record."""

    source_bundle = Path(source_bundle).resolve(strict=True)
    official_checkout = Path(official_checkout).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    _official_git_sha(official_checkout)
    _verify_released_checkpoint(checkpoint)

    from mme_vla_suite.models.integration import history_gemma
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0

    # Read-only reuse of the already sealed 64-record fixture implementation.
    from robomme_integration.amkv import driver, episodes

    _require_import_below(HistoryPi0, official_checkout, label="HistoryPi0")
    _require_import_below(history_gemma, official_checkout, label="history_gemma")
    records = episodes.load_fixture_bundle(source_bundle)
    selected = [record for record in records if record.fixture_id == source_fixture_id]
    if len(selected) != 1:
        raise ValueError(f"source fixture id must resolve exactly once, got {len(selected)}")
    record = selected[0]
    source_manifest_path = source_bundle / episodes.MANIFEST_FILENAME
    source_manifest = episodes.FixtureBundleManifest.from_dict(
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
    )
    # The released config loader resolves its history YAML relative to the
    # official repository root.  Bind that root explicitly; callers may run
    # this exporter from any working directory.
    policy = driver.load_policy(
        checkpoint,
        policy_source_root=official_checkout,
        seed=model_seed,
    )
    _require_import_below(type(policy), official_checkout, label="MME_VLA_Policy")
    _require_import_below(type(policy._model), official_checkout, label="HistoryPi0")  # noqa: SLF001
    observation, history, _meta = driver.build_observation(policy, record)

    import jax
    import jax.numpy as jnp

    noise = jax.random.normal(
        jax.random.key(sampler_seed),
        (1, int(policy._model.action_horizon), int(policy._model.action_dim)),  # noqa: SLF001
        dtype=jnp.float32,
    )
    actions = policy._sample_actions(  # noqa: SLF001
        jax.random.key(sampler_seed + 1),
        observation,
        num_steps=10,
        noise=noise,
    )
    return seal_teacher_smoke_fixture(
        destination,
        observation=observation,
        history=history,
        official_actions=np.asarray(actions),
        sampler_noise=np.asarray(noise),
        task_id=f"robomme_task_{record.task_index:03d}",
        episode_id=record.pair_id,
        causal_cut_step=int(record.step_idx),
        source_fixture_id=record.fixture_id,
        source_chunk_role=record.chunk_role,
        source_bundle_manifest_sha256=_sha256_file(source_manifest_path),
        source_bundle_content_sha256=source_manifest.content_sha256,
        source_bundle_payload_sha256=source_manifest.payload_sha256,
        source_record_metadata=record.metadata(),
        model_seed=model_seed,
        sampler_seed=sampler_seed,
        num_steps=10,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--source-fixture-id", required=True)
    parser.add_argument("--official-checkout", type=Path, default=DEFAULT_OFFICIAL_CHECKOUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--sampler-seed", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = export_from_sealed_causal_bundle(
        source_bundle=args.source_bundle,
        source_fixture_id=args.source_fixture_id,
        official_checkout=args.official_checkout,
        checkpoint=args.checkpoint,
        destination=args.destination,
        model_seed=args.model_seed,
        sampler_seed=args.sampler_seed,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
