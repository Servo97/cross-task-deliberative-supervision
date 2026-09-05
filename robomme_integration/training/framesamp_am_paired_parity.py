"""In-process bitwise parity gate for the released FrameSamp-AM overlay.

Separate JAX processes are not a valid bitwise oracle on the RTX 5090: the
unmodified released policy itself can compile to slightly different action
bits across processes.  This gate therefore executes the released and patched
graphs sequentially in one process, with the same restored checkpoint,
fixture, PRNG key, sampler noise, visible device, and JAX runtime.  It also
requires the released graph to reproduce itself bitwise before comparing the
patched graph.

The two models have disjoint lifetimes so this remains usable while another
workload occupies part of GPU memory.  This command is parity evidence only;
it does not authorize an AM rollout.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robomme_integration.training.framesamp_am_teacher_producer import (
    RELEASED_CHECKPOINT_SHA256,
    assert_full_action_parity,
    verify_source_pinned_released_teacher,
)
from robomme_integration.training.framesamp_am_teacher_smoke import (
    _load_observation_fixture,
)

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_am_in_process_released_policy_parity"


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "raw_bytes_sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _require_import_below(value: object, root: Path, *, label: str) -> str:
    source = Path(inspect.getfile(value)).resolve(strict=True)
    try:
        source.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} was not imported from {root}") from error
    return str(source)


def _purge_mme_modules() -> None:
    for name in tuple(sys.modules):
        if name == "mme_vla_suite" or name.startswith("mme_vla_suite."):
            del sys.modules[name]


def _load_and_sample(
    *,
    source_root: Path,
    working_root: Path,
    checkpoint: Path,
    observation: Any,
    noise: np.ndarray,
    sampler_key_seed: int,
    model_seed: int,
) -> tuple[Any, np.ndarray, str]:
    import jax

    sys.path.insert(0, str(source_root))
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config
    from openpi.shared import nnx_utils

    imported_from = _require_import_below(
        HistoryPi0,
        working_root,
        label="HistoryPi0",
    )
    previous_cwd = Path.cwd()
    os.chdir(working_root)
    try:
        policy = policy_config.create_trained_policy(
            config.get_config("mme_vla_suite"),
            checkpoint,
            seed=model_seed,
            default_prompt=None,
        )
    finally:
        os.chdir(previous_cwd)
    sampler = nnx_utils.module_jit(policy._model.sample_actions)  # noqa: SLF001
    actions = np.asarray(
        sampler(
            jax.random.key(sampler_key_seed),
            observation,
            num_steps=10,
            noise=noise,
        )
    )
    return (policy, actions, imported_from)


def run_paired_parity(
    *,
    official_checkout: Path,
    policy_overlay: Path,
    overlay_manifest_sha256: str,
    checkpoint: Path,
    fixture_manifest: Path,
) -> dict[str, object]:
    """Run and return one sealed-ready in-process parity record."""

    provenance = verify_source_pinned_released_teacher(
        policy_overlay=policy_overlay,
        expected_policy_overlay_manifest_sha256=overlay_manifest_sha256,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=RELEASED_CHECKPOINT_SHA256,
    )
    manifest, observation, _history, fixture_actions, sampler_noise = _load_observation_fixture(
        fixture_manifest.resolve(strict=True)
    )
    if manifest["teacher_code_sha"] != provenance["teacher_code_sha"]:
        raise ValueError("fixture and verified released source commit disagree")
    if manifest["teacher_checkpoint_sha256"] != RELEASED_CHECKPOINT_SHA256:
        raise ValueError("fixture and released checkpoint disagree")

    import jax

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("paired parity requires exactly one visible GPU")
    official_src = official_checkout.resolve(strict=True) / "src"
    overlay_src = policy_overlay.resolve(strict=True) / "src"
    # Fixture reconstruction imports the released HistAugObservation class.
    # Authenticate that early import before loading the rest of the official
    # policy graph; it must never come from the overlay or an ambient checkout.
    from mme_vla_suite.models.integration.history_observation import (
        HistAugObservation,
    )

    _require_import_below(
        HistAugObservation,
        official_checkout,
        label="fixture HistAugObservation",
    )
    model_seed = int(manifest["model_seed"])
    sampler_key_seed = int(manifest["sampler_key_seed"])

    official_policy, official_actions, official_import = _load_and_sample(
        source_root=official_src,
        working_root=official_checkout,
        checkpoint=checkpoint,
        observation=observation,
        noise=sampler_noise,
        sampler_key_seed=sampler_key_seed,
        model_seed=model_seed,
    )
    # A second call must reproduce exactly within the comparison process.  If
    # it does not, there is no meaningful bitwise baseline to compare against.
    from openpi.shared import nnx_utils

    official_repeat = np.asarray(
        nnx_utils.module_jit(official_policy._model.sample_actions)(  # noqa: SLF001
            jax.random.key(sampler_key_seed),
            observation,
            num_steps=10,
            noise=sampler_noise,
        )
    )
    assert_full_action_parity(official_actions, official_repeat)

    # Keep only host action bytes before importing and restoring the patched
    # graph.  Sequential lifetimes avoid holding two ~12.5 GiB states.
    del official_policy, official_repeat
    gc.collect()
    jax.clear_caches()
    gc.collect()
    _purge_mme_modules()
    official_src_resolved = official_src.resolve()
    sys.path[:] = [value for value in sys.path if Path(value or ".").resolve() != official_src_resolved]

    patched_policy, patched_actions, patched_import = _load_and_sample(
        source_root=overlay_src,
        working_root=policy_overlay,
        checkpoint=checkpoint,
        observation=observation,
        noise=sampler_noise,
        sampler_key_seed=sampler_key_seed,
        model_seed=model_seed,
    )
    del patched_policy
    assert_full_action_parity(official_actions, patched_actions)

    # The cross-process fixture actions remain useful provenance, but are not
    # the bitwise oracle.  Preserve their observed relation without weakening
    # or conflating it with the in-process gate.
    fixture_delta = np.abs(fixture_actions.astype(np.float32) - official_actions.astype(np.float32))
    action_identity = _array_identity(official_actions)
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "HARD_GREEN",
        "scope": "parity_only_not_rollout_or_scientific_evidence",
        "comparison_contract": "same_process_same_device_same_checkpoint_same_key_same_noise",
        "official_self_repeat_bitwise_equal": True,
        "official_vs_patched_bitwise_equal": True,
        "action_elements": int(official_actions.size),
        "nonzero_elements": 0,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "actions": action_identity,
        "fixture_cross_process_diagnostic": {
            "bitwise_equal": bool(np.array_equal(fixture_actions, official_actions)),
            "nonzero_elements": int(np.count_nonzero(fixture_delta)),
            "max_abs": float(fixture_delta.max(initial=0.0)),
            "mean_abs": float(fixture_delta.mean()) if fixture_delta.size else 0.0,
        },
        "fixture_manifest": str(fixture_manifest.resolve(strict=True)),
        "fixture_manifest_sha256": _sha256_file(fixture_manifest.resolve(strict=True)),
        "source_fixture_id": manifest["source_fixture_id"],
        "model_seed": model_seed,
        "sampler_key_seed": sampler_key_seed,
        "sampler_noise": _array_identity(sampler_noise),
        "teacher_checkpoint_sha256": RELEASED_CHECKPOINT_SHA256,
        "teacher_code_sha": provenance["teacher_code_sha"],
        "policy_overlay_manifest_sha256": overlay_manifest_sha256,
        "patched_history_gemma_sha256": provenance["patched_history_gemma_sha256"],
        "patched_history_pi0_sha256": provenance["patched_history_pi0_sha256"],
        "official_history_pi0_import": official_import,
        "patched_history_pi0_import": patched_import,
        "jax_version": jax.__version__,
        "device": {
            "count": 1,
            "platform": devices[0].platform,
            "device_kind": devices[0].device_kind,
        },
    }
    record["receipt_sha256"] = hashlib.sha256(_canonical_json(record)).hexdigest()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {args.output.parent}")
    record = run_paired_parity(
        official_checkout=args.official_checkout,
        policy_overlay=args.policy_overlay,
        overlay_manifest_sha256=args.overlay_manifest_sha256,
        checkpoint=args.checkpoint,
        fixture_manifest=args.fixture_manifest,
    )
    with args.output.open("xb") as stream:
        stream.write(_canonical_json(record))
        stream.flush()
        os.fsync(stream.fileno())
    args.output.chmod(0o444)
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
