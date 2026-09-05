"""Deterministic episode manifests, shard outputs, and exact atomic merging for RoboCasa eval.

This module deliberately has no simulator, numpy, torch, or JAX imports, so manifest validation and
merging can run cheaply in an orchestration container. The scientific unit is pinned by the tuple
(task, episode_index, reset, seed); worker count and shard count never alter that identity.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import tempfile
from typing import Iterable

MANIFEST_SCHEMA_VERSION = 2
MANIFEST_KIND = "robocasa_episode_manifest"
SHARD_RESULTS_KIND = "robocasa_episode_shard_results"
POLICY_NOISE_KIND = "pi_diffusion_sha256_v1"
EVALUATION_PROVENANCE_KIND = "pi_stage_s_evaluation_provenance"
EVALUATION_PROVENANCE_SCHEMA_VERSION = 1

_POLICY_TIMING_MS_FIELDS = frozenset(
    {
        # Legacy compatibility. New OpenPI responses also carry the explicit completed/amortized key.
        "infer_ms",
        "policy_model_amortized_ms",
        # Whole inner policy call and online-workspace stages, all completed and amortized per request.
        "policy_call_amortized_ms",
        "wsm_tap_amortized_ms",
        "wsm_encoder_amortized_ms",
        "wsm_prepare_amortized_ms",
        "wsm_end_to_end_amortized_ms",
        # Legacy BatchGather name plus its precise normalized name: enqueue through result delivery.
        "gather_ms",
        "gather_request_ms",
        # Client-observed request/response latency added by the evaluator.
        "client_roundtrip_ms",
        # Reserved for explicitly namespaced websocket timing.
        "server_request_ms",
    }
)
_POLICY_TIMING_COUNT_MINIMUMS = {
    "gather_batch_n": 1,
    "policy_model_batch_n": 1,
    "policy_model_bucket_n": 1,
    "wsm_request_batch_n": 1,
    # Zero means every request reused its already-computed causal grid.
    "wsm_new_grid_batch_n": 0,
}


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _artifact_descriptor(path: str | os.PathLike) -> dict:
    """Content identity for one exact-reset artifact."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "size": size}


def _manifest_digest_payload(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def validate_evaluation_provenance(provenance: dict, *, episode_manifest_sha256: str | None = None) -> dict:
    """Validate the immutable model/eval identity carried by every exact result artifact."""
    if not isinstance(provenance, dict):
        raise ValueError("evaluation_provenance must be a JSON object")
    expected_keys = {
        "schema_version",
        "kind",
        "eval_run_id",
        "eval_manifest_sha256",
        "arm",
        "interface",
        "training_run_id",
        "training_manifest_sha256",
        "checkpoint_uri",
        "checkpoint_step",
        "checkpoint_tree_manifest_sha256",
        "episode_manifest_sha256",
        "episode_manifest_file_sha256",
    }
    if set(provenance) != expected_keys:
        raise ValueError(
            "evaluation_provenance fields differ from the canonical schema: "
            f"missing={sorted(expected_keys - set(provenance))}, "
            f"extra={sorted(set(provenance) - expected_keys)}"
        )
    if provenance.get("schema_version") != EVALUATION_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation_provenance schema_version")
    if provenance.get("kind") != EVALUATION_PROVENANCE_KIND:
        raise ValueError(f"evaluation_provenance kind must be {EVALUATION_PROVENANCE_KIND!r}")
    for field in ("eval_run_id", "training_run_id", "checkpoint_uri"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ValueError(f"evaluation_provenance.{field} must be a nonempty string")
    arm = provenance.get("arm")
    interface = provenance.get("interface")
    # Serve-side interface per arm: base serves s0/s3/q0 (bitwise-identical serving), tanh serves
    # BOTH s1 and q1 (identical serve-time contract), robottt_fast serves q2 (workspace-free online
    # TTT), and tanh_robottt serves q3 (tanh workspace read + fast weights combined). Keep in
    # lockstep with scripts/launch/submit_pi_stage_s_eval.py, validate_stage_s_eval_inputs.py, and
    # internal_training/robocasa/aggregate_eval.py.
    expected_interface = {
        "s0": "base",
        "s1": "tanh",
        "s2": "cfg2",
        "s3": "base",
        "q0": "base",
        "q1": "tanh",
        "q2": "robottt_fast",
        "q3": "tanh_robottt",
        # H13 R1-R4: aux-only, every H13 subtree dropped at load -> plain base serve.
        "h13a": "base",
        "h13b": "base",
        "h13c": "base",
        "h13d": "base",
        "h13c2": "base",
        "h13d2": "base",
        "h13e": "tanh",
        "h13f": "tanh",
        "h13g": "tanh",
        "h13h": "tanh",
        "h13g2": "tanh",
        "h13h2": "tanh",
    }.get(arm)
    if expected_interface is None or interface != expected_interface:
        raise ValueError(f"evaluation_provenance arm/interface mismatch: {arm!r}/{interface!r}")
    step = provenance.get("checkpoint_step")
    if type(step) is not int or step < 0:
        raise ValueError("evaluation_provenance.checkpoint_step must be a nonnegative integer")
    for field in (
        "eval_manifest_sha256",
        "training_manifest_sha256",
        "checkpoint_tree_manifest_sha256",
        "episode_manifest_sha256",
        "episode_manifest_file_sha256",
    ):
        _require_sha256(provenance.get(field), f"evaluation_provenance.{field}")
    if episode_manifest_sha256 is not None and provenance["episode_manifest_sha256"] != episode_manifest_sha256:
        raise ValueError("evaluation_provenance episode manifest does not match the loaded manifest")
    return dict(provenance)


def evaluation_provenance_from_run_manifest(
    run_manifest_path: str | os.PathLike,
    episode_manifest: dict,
    *,
    episode_manifest_path: str | os.PathLike,
) -> dict:
    """Derive exact-result provenance from a sealed Stage-S eval run manifest."""
    with open(run_manifest_path, encoding="utf-8") as stream:
        run_manifest = json.load(stream)
    if not isinstance(run_manifest, dict):
        raise ValueError("eval run manifest must be a JSON object")
    claimed = _require_sha256(run_manifest.get("manifest_sha256"), "eval run manifest seal")
    unsealed = dict(run_manifest)
    unsealed.pop("manifest_sha256", None)
    # Stage-S run manifests use ASCII-canonical JSON when sealed.
    actual = hashlib.sha256(
        json.dumps(unsealed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if actual != claimed:
        raise ValueError(f"eval run manifest seal mismatch: claimed={claimed}, computed={actual}")
    if run_manifest.get("schema_version") != 1 or run_manifest.get("kind") not in (
        "pi_stage_s_robocasa_eval_run",
        # E4 shakedown canaries share the exact rollout/provenance path on a strict task subset;
        # their distinct kind + canary-only prefixes keep them out of decisive results.
        "pi_stage_s_robocasa_eval_canary",
    ):
        raise ValueError("not a schema-v1 Stage-S RoboCasa eval run manifest")
    run = run_manifest.get("training_run")
    protocol = run_manifest.get("protocol")
    if not isinstance(run, dict) or not isinstance(protocol, dict):
        raise ValueError("eval run manifest is missing training_run/protocol")
    checkpoint_tree = run.get("checkpoint_tree_manifest")
    episode_descriptor = protocol.get("episode_manifest")
    if not isinstance(checkpoint_tree, dict) or not isinstance(episode_descriptor, dict):
        raise ValueError("eval run manifest is missing artifact descriptors")
    digest = hashlib.sha256()
    with open(episode_manifest_path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    episode_file_sha256 = digest.hexdigest()
    if episode_descriptor.get("file_sha256") != episode_file_sha256:
        raise ValueError("eval run manifest episode file digest does not match the loaded file")
    provenance = {
        "schema_version": EVALUATION_PROVENANCE_SCHEMA_VERSION,
        "kind": EVALUATION_PROVENANCE_KIND,
        "eval_run_id": run_manifest.get("eval_run_id"),
        "eval_manifest_sha256": claimed,
        "arm": run_manifest.get("arm"),
        "interface": run_manifest.get("interface"),
        "training_run_id": run.get("run_id"),
        "training_manifest_sha256": run.get("manifest_sha256"),
        "checkpoint_uri": run.get("checkpoint_uri"),
        "checkpoint_step": run.get("checkpoint_step"),
        "checkpoint_tree_manifest_sha256": checkpoint_tree.get("file_sha256"),
        "episode_manifest_sha256": episode_manifest.get("manifest_sha256"),
        "episode_manifest_file_sha256": episode_file_sha256,
    }
    return validate_evaluation_provenance(
        provenance,
        episode_manifest_sha256=episode_manifest.get("manifest_sha256"),
    )


def seal_episode_manifest(payload: dict) -> dict:
    """Copy, content-address, and validate a manifest payload from any reset adapter."""
    sealed = dict(payload)
    sealed.pop("manifest_sha256", None)
    sealed["manifest_sha256"] = hashlib.sha256(_canonical_bytes(sealed)).hexdigest()
    return validate_episode_manifest(sealed)


def _seed_for(base_seed: int, task: str, episode_index: int) -> int:
    """Stable per-episode seed, independent of Python hash randomization and shard count."""
    raw = f"{int(base_seed)}\0{task}\0{int(episode_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def policy_noise_seed(episode_seed: int, env_step: int) -> int:
    """Stable uint32 key for one π action-chunk diffusion draw.

    This is independent of request order, server replica, gather batch, and shard topology.
    """
    if not 0 <= int(episode_seed) <= 0xFFFFFFFF:
        raise ValueError(f"episode_seed is outside uint32: {episode_seed}")
    if int(env_step) < 0:
        raise ValueError(f"env_step must be nonnegative, got {env_step}")
    raw = (f"{POLICY_NOISE_KIND}\0{int(episode_seed)}\0{int(env_step)}").encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def episode_identity(record: dict) -> tuple[str, int, str, int]:
    """Hashable canonical form of the four immutable identity fields."""
    return (
        str(record["task"]),
        int(record["episode_index"]),
        _canonical_bytes(record["reset"]).decode("utf-8"),
        int(record["seed"]),
    )


def build_episode_manifest(
    tasks: Iterable[dict],
    num_episodes: int,
    base_seed: int,
    *,
    split: str = "target",
    task_sets: Iterable[str] | None = None,
    reset_kind: str = "gym_seed",
) -> dict:
    """Build the immutable procedural-reset manifest used by the 50 x 100 evaluation.

    Task entries follow eval_common.list_tasks. Seeds are derived per episode identity, so changing
    worker or episode-shard counts cannot change a reset. Held-out-demo tooling may construct the
    same schema with a richer JSON reset descriptor; eval_pi_05 intentionally executes only
    reset.kind == gym_seed until its held-out-reset adapter is wired.
    """
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    entries = []
    for task_entry in tasks:
        task = str(task_entry["task"])
        for episode_index in range(num_episodes):
            entries.append(
                {
                    "task": task,
                    "split_set": str(task_entry["split_set"]),
                    "horizon": int(task_entry["horizon"]),
                    "episode_index": episode_index,
                    "reset": {"kind": reset_kind},
                    "seed": _seed_for(base_seed, task, episode_index),
                }
            )
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "split": str(split),
        "task_sets": list(task_sets or []),
        "base_seed": int(base_seed),
        "policy_noise": {
            "kind": POLICY_NOISE_KIND,
            "key_fields": ["episode.seed", "env_step"],
        },
        "episodes_per_task": int(num_episodes),
        "episodes": entries,
    }
    return seal_episode_manifest(payload)


def build_heldout_episode_manifest(
    tasks: Iterable[dict],
    heldout_root: str | os.PathLike,
    num_episodes: int,
    base_seed: int,
    *,
    split: str = "target",
    task_sets: Iterable[str] | None = None,
) -> dict:
    """Select distinct held-out demos/task and pin their exact extras reset descriptors.

    Each <heldout_root>/<task>/heldout.json is the complement of the training keep-set. We rank that
    complement by SHA-256(base_seed, task, dataset episode index), take exactly num_episodes without
    replacement, then order selected dataset indices numerically. This avoids first-N dataset-order
    bias while remaining stable across processes, Python versions, worker counts, and shard counts.
    """
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    root = os.fspath(heldout_root)
    entries = []
    selection_meta = {}
    for task_entry in tasks:
        task = str(task_entry["task"])
        heldout_path = os.path.join(root, task, "heldout.json")
        with open(heldout_path) as handle:
            heldout = json.load(handle)
        if heldout.get("task") not in (None, task):
            raise ValueError(f"{heldout_path}: task={heldout.get('task')!r}, expected {task!r}")
        try:
            episode_indices = [int(value) for value in heldout["episodes"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{heldout_path}: episodes must be a list of integer dataset indices") from exc
        if len(episode_indices) != len(set(episode_indices)):
            raise ValueError(f"{heldout_path}: duplicate held-out episode indices")
        if any(value < 0 for value in episode_indices):
            raise ValueError(f"{heldout_path}: negative held-out episode index")
        if len(episode_indices) < num_episodes:
            raise ValueError(
                f"{task}: requested {num_episodes} distinct held-out demos, "
                f"but only {len(episode_indices)} are available"
            )

        def rank(episode_index: int) -> tuple[bytes, int]:
            raw = (f"{int(base_seed)}\0{task}\0heldout_demo\0{episode_index}").encode("utf-8")
            return hashlib.sha256(raw).digest(), episode_index

        selected = sorted(sorted(episode_indices, key=rank)[:num_episodes])
        source = heldout.get("source")
        selection_meta[task] = {
            "available": len(episode_indices),
            "selected": len(selected),
            "training_num_episodes": heldout.get("num_train"),
            "training_subset_seed": heldout.get("seed"),
            "source": source,
        }
        for episode_index in selected:
            extras_relpath = os.path.join(task, "extras", f"episode_{episode_index:06d}")
            extras_path = os.path.join(root, extras_relpath)
            required = ("ep_meta.json", "model.xml.gz", "states.npz")
            missing = [filename for filename in required if not os.path.isfile(os.path.join(extras_path, filename))]
            if missing:
                raise ValueError(f"{task}/{episode_index}: missing exact-reset artifacts {missing}")
            artifacts = {filename: _artifact_descriptor(os.path.join(extras_path, filename)) for filename in required}
            entries.append(
                {
                    "task": task,
                    "split_set": str(task_entry["split_set"]),
                    "horizon": int(task_entry["horizon"]),
                    # Dataset episode index, not a re-numbered 0..N-1 trial counter.
                    "episode_index": episode_index,
                    "reset": {
                        "kind": "heldout_demo",
                        "extras_relpath": extras_relpath,
                        "source": source,
                        "artifacts": artifacts,
                    },
                    # Pins environment RNG for any stochastic state not overwritten by extras.
                    "seed": _seed_for(base_seed, task, episode_index),
                }
            )
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "split": str(split),
        "task_sets": list(task_sets or []),
        "base_seed": int(base_seed),
        "policy_noise": {
            "kind": POLICY_NOISE_KIND,
            "key_fields": ["episode.seed", "env_step"],
        },
        "episodes_per_task": int(num_episodes),
        "selection": {
            "kind": "heldout_complement_sha256_without_replacement",
            "per_task": selection_meta,
        },
        "episodes": entries,
    }
    return seal_episode_manifest(payload)


def validate_episode_manifest(manifest: dict) -> dict:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema_version={manifest.get('schema_version')!r}")
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError(f"manifest kind must be {MANIFEST_KIND!r}, got {manifest.get('kind')!r}")
    if manifest.get("policy_noise") != {
        "kind": POLICY_NOISE_KIND,
        "key_fields": ["episode.seed", "env_step"],
    }:
        raise ValueError(f"manifest must pin policy_noise={POLICY_NOISE_KIND!r} keyed by episode.seed/env_step")
    claimed = manifest.get("manifest_sha256")
    actual = hashlib.sha256(_canonical_bytes(_manifest_digest_payload(manifest))).hexdigest()
    if claimed != actual:
        raise ValueError(f"manifest digest mismatch: claimed {claimed!r}, computed {actual}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("manifest episodes must be a non-empty list")

    identities = set()
    task_episode_indices = set()
    counts: dict[str, int] = {}
    for index, record in enumerate(episodes):
        missing = {
            "task",
            "split_set",
            "horizon",
            "episode_index",
            "reset",
            "seed",
        } - set(record)
        if missing:
            raise ValueError(f"manifest episode {index} missing {sorted(missing)}")
        if int(record["episode_index"]) < 0:
            raise ValueError(f"manifest episode {index} has negative episode_index")
        if not 0 <= int(record["seed"]) <= 0xFFFFFFFF:
            raise ValueError(f"manifest episode {index} seed is outside uint32")
        if int(record["horizon"]) < 1:
            raise ValueError(f"manifest episode {index} horizon must be positive")
        identity = episode_identity(record)  # Also proves reset is JSON serializable.
        reset = record["reset"]
        if not isinstance(reset, dict) or not isinstance(reset.get("kind"), str):
            raise ValueError(f"manifest episode {index} reset must be a mapping with string kind")
        if reset["kind"] == "heldout_demo":
            artifacts = reset.get("artifacts")
            required_artifacts = {"ep_meta.json", "model.xml.gz", "states.npz"}
            if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts:
                raise ValueError(
                    f"manifest episode {index} heldout_demo artifacts must be exactly {sorted(required_artifacts)}"
                )
            for filename, descriptor in artifacts.items():
                if not isinstance(descriptor, dict):
                    raise ValueError(f"manifest episode {index} artifact {filename} descriptor must be a mapping")
                digest = descriptor.get("sha256")
                size = descriptor.get("size")
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                ):
                    raise ValueError(f"manifest episode {index} artifact {filename} has invalid sha256")
                if type(size) is not int or size < 0:
                    raise ValueError(f"manifest episode {index} artifact {filename} has invalid size")
        if reset["kind"] == "remembench_ep_meta":
            # ReMemBench pins episodes by replaying the demo's ep_meta inline (no side
            # artifacts to fetch). ep_meta must carry the scene pin; `source` records
            # which demo it came from so a manifest is auditable against the dataset.
            ep_meta = reset.get("ep_meta")
            if not isinstance(ep_meta, dict) or not ep_meta:
                raise ValueError(f"manifest episode {index} remembench_ep_meta requires a non-empty ep_meta")
            for key in ("layout_id", "style_id"):
                if key not in ep_meta:
                    raise ValueError(f"manifest episode {index} remembench_ep_meta ep_meta missing {key!r}")
            source = reset.get("source")
            if not isinstance(source, dict) or not {
                "task",
                "session",
                "demo_key",
            } <= set(source):
                raise ValueError(f"manifest episode {index} remembench_ep_meta source must name task/session/demo_key")
        if identity in identities:
            raise ValueError(f"duplicate manifest identity at episode {index}: {identity}")
        identities.add(identity)
        task_episode = (identity[0], identity[1])
        if task_episode in task_episode_indices:
            raise ValueError(f"duplicate (task, episode_index) at episode {index}: {task_episode}")
        task_episode_indices.add(task_episode)
        counts[record["task"]] = counts.get(record["task"], 0) + 1

    expected = manifest.get("episodes_per_task")
    if expected is not None:
        bad = {task: count for task, count in counts.items() if count != int(expected)}
        if bad:
            raise ValueError(f"manifest episodes_per_task={expected}, mismatched counts={bad}")
    return manifest


def write_episode_manifest(path: str | os.PathLike, manifest: dict) -> None:
    """Publish once without overwrite; an identical existing manifest is idempotent."""
    validate_episode_manifest(manifest)
    path = os.fspath(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    data = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Atomic no-clobber publication. os.replace would violate manifest immutability.
            os.link(temporary, path)
        except FileExistsError:
            with open(path) as handle:
                existing = validate_episode_manifest(json.load(handle))
            if _canonical_bytes(existing) != _canonical_bytes(manifest):
                raise FileExistsError(f"refusing to overwrite different immutable manifest: {path}")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_episode_manifest(path: str | os.PathLike) -> dict:
    with open(path) as handle:
        return validate_episode_manifest(json.load(handle))


def write_json_atomic(path: str | os.PathLike, value: dict) -> None:
    """Write to a same-directory temp file, fsync, then atomically replace path."""
    path = os.fspath(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validate_shard_index(shard_idx: int, num_shards: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_idx < num_shards:
        raise ValueError(f"shard_idx must be in [0, {num_shards}), got {shard_idx}")


def shard_episode_records(records: Iterable[dict], shard_idx: int, num_shards: int) -> list[dict]:
    """Stable within-task ordinal shard, balanced to within one episode."""
    _validate_shard_index(shard_idx, num_shards)
    ordered = sorted(records, key=lambda record: episode_identity(record)[1:])
    tasks = {record["task"] for record in ordered}
    if len(tasks) > 1:
        raise ValueError(f"episode sharding expects one task, got {sorted(tasks)}")
    return [record for ordinal, record in enumerate(ordered) if ordinal % num_shards == shard_idx]


def manifest_records_for_task(manifest: dict, task: str) -> list[dict]:
    validate_episode_manifest(manifest)
    return [record for record in manifest["episodes"] if record["task"] == task]


def shard_stats_path(
    out_dir: str | os.PathLike,
    split_set: str,
    task: str,
    shard_idx: int,
    num_shards: int,
) -> str:
    """K-qualified path; changing K cannot silently reuse an incompatible shard."""
    _validate_shard_index(shard_idx, num_shards)
    return os.path.join(
        os.fspath(out_dir),
        split_set,
        task,
        f"stats_shard{shard_idx}of{num_shards}.json",
    )


def sanitize_policy_timing(value: object) -> dict:
    """Keep only canonical finite JSON timing scalars and exact batch cardinalities."""
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, raw in value.items():
        if key in _POLICY_TIMING_COUNT_MINIMUMS:
            if type(raw) in (int, float):
                number = float(raw)
                if math.isfinite(number):
                    count = int(number)
                    if count >= _POLICY_TIMING_COUNT_MINIMUMS[key] and number == count:
                        out[key] = count
            continue
        if key not in _POLICY_TIMING_MS_FIELDS or type(raw) not in (int, float):
            continue
        number = float(raw)
        if math.isfinite(number) and number >= 0:
            out[key] = round(number, 6)
    # Explicit synchronized/amortized metrics supersede the ambiguous legacy alias in summaries.
    if "policy_model_amortized_ms" in out or "policy_call_amortized_ms" in out:
        out.pop("infer_ms", None)
    if "gather_ms" in out:
        out.setdefault("gather_request_ms", out["gather_ms"])
        out.pop("gather_ms")
    return out


def _percentile(values: list[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_policy_performance(episodes: Iterable[dict], wall_seconds: float) -> dict:
    episode_list = list(episodes)
    calls = [
        sanitized
        for episode in episode_list
        for call in episode.get("policy_timing_calls", [])
        if isinstance(call, dict)
        if (sanitized := sanitize_policy_timing(call))
    ]

    def batch_summary(key: str) -> dict:
        # Values occur once per request, not once per physical batch. This is the experienced-request
        # distribution; effective_batch_size reverses that size bias when every row reports its batch N.
        sizes = [call[key] for call in calls if key in call]
        histogram = {}
        for size in sizes:
            histogram[str(size)] = histogram.get(str(size), 0) + 1
        inverse_sum = sum(1.0 / size for size in sizes if size > 0)
        return {
            "histogram": histogram,
            "max": max(sizes) if sizes else None,
            "request_weighted_mean": (round(sum(sizes) / len(sizes), 3) if sizes else None),
            # Compatibility alias, now explicitly identified by the field above.
            "mean": round(sum(sizes) / len(sizes), 3) if sizes else None,
            "requests_observed": len(sizes),
            "multi_request_fraction": (round(sum(size > 1 for size in sizes) / len(sizes), 6) if sizes else None),
            "effective_batch_size": (round(len(sizes) / inverse_sum, 3) if inverse_sum > 0 else None),
            "realized_multi_request": bool(sizes and max(sizes) > 1),
            "weighting": "per_request",
        }

    batch_keys = sorted({key for call in calls for key in call if key in _POLICY_TIMING_COUNT_MINIMUMS})
    by_stage = {key: batch_summary(key) for key in batch_keys}
    gather = by_stage.get("gather_batch_n", batch_summary("gather_batch_n"))
    latency_keys = sorted({key for call in calls for key in call if key.endswith("_ms")})
    latencies = {}
    for key in latency_keys:
        values = [float(call[key]) for call in calls if key in call]
        latencies[key] = {
            "count": len(values),
            "mean": round(sum(values) / len(values), 3),
            "p50": round(float(_percentile(values, 0.50)), 3),
            "p95": round(float(_percentile(values, 0.95)), 3),
            "max": round(max(values), 3),
        }
    return {
        "policy_calls": len(calls),
        "batching": {
            **gather,
            "by_stage": by_stage,
        },
        "latency_ms": latencies,
        "rollouts_per_hour": round(len(episode_list) * 3600.0 / wall_seconds, 3) if wall_seconds > 0 else None,
    }


def build_shard_results(
    manifest: dict,
    split_set: str,
    task: str,
    shard_idx: int,
    num_shards: int,
    episodes: Iterable[dict],
    *,
    complete: bool,
    wall_seconds: float = 0.0,
    evaluation_provenance: dict | None = None,
) -> dict:
    episode_list = list(episodes)
    successes = [bool(episode["success"]) for episode in episode_list]
    normalized_wall_seconds = round(float(wall_seconds), 1)
    output = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": SHARD_RESULTS_KIND,
        "manifest_sha256": manifest["manifest_sha256"],
        "task": task,
        "split_set": split_set,
        "split": manifest["split"],
        "shard_index": int(shard_idx),
        "num_shards": int(num_shards),
        "complete": bool(complete),
        "num_episodes": len(episode_list),
        "success_rate": (sum(successes) / len(successes) if episode_list else None),
        "wall_seconds": normalized_wall_seconds,
        "performance": summarize_policy_performance(episode_list, normalized_wall_seconds),
        "per_episode": episode_list,
    }
    if evaluation_provenance is not None:
        output["evaluation_provenance"] = validate_evaluation_provenance(
            evaluation_provenance,
            episode_manifest_sha256=manifest["manifest_sha256"],
        )
    return output


def validate_shard_results(
    payload: dict,
    manifest: dict,
    split_set: str,
    task: str,
    shard_idx: int,
    num_shards: int,
    *,
    require_complete: bool,
    evaluation_provenance: dict | None = None,
) -> dict:
    _validate_shard_index(shard_idx, num_shards)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unexpected shard schema_version: {payload.get('schema_version')!r}")
    if payload.get("kind") != SHARD_RESULTS_KIND:
        raise ValueError(f"unexpected shard result kind: {payload.get('kind')!r}")
    expected_header = {
        "manifest_sha256": manifest["manifest_sha256"],
        "task": task,
        "split_set": split_set,
        "split": manifest["split"],
        "shard_index": shard_idx,
        "num_shards": num_shards,
    }
    for key, expected_value in expected_header.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"shard field {key}={payload.get(key)!r}; expected {expected_value!r}")
    if evaluation_provenance is not None:
        expected_provenance = validate_evaluation_provenance(
            evaluation_provenance,
            episode_manifest_sha256=manifest["manifest_sha256"],
        )
        actual_provenance = validate_evaluation_provenance(
            payload.get("evaluation_provenance"),
            episode_manifest_sha256=manifest["manifest_sha256"],
        )
        if actual_provenance != expected_provenance:
            raise ValueError("shard evaluation_provenance does not match this eval run")
    if require_complete and payload.get("complete") is not True:
        raise ValueError("shard is not marked complete")

    expected_records = shard_episode_records(manifest_records_for_task(manifest, task), shard_idx, num_shards)
    expected = {episode_identity(record): record for record in expected_records}
    episodes = payload.get("per_episode")
    if not isinstance(episodes, list):
        raise ValueError("shard per_episode must be a list")
    if payload.get("num_episodes") != len(episodes):
        raise ValueError(f"shard num_episodes={payload.get('num_episodes')!r}; expected {len(episodes)}")
    wall_seconds = payload.get("wall_seconds")
    if type(wall_seconds) not in (int, float) or not math.isfinite(float(wall_seconds)) or float(wall_seconds) < 0:
        raise ValueError("shard wall_seconds must be a finite nonnegative JSON number")
    expected_performance = summarize_policy_performance(episodes, float(wall_seconds))
    if payload.get("performance") != expected_performance:
        raise ValueError("shard performance summary does not match per_episode timings")
    actual = {}
    for index, result in enumerate(episodes):
        missing = {
            "task",
            "episode_index",
            "reset",
            "seed",
            "success",
            "episode_length",
        } - set(result)
        if missing:
            raise ValueError(f"shard result episode {index} missing {sorted(missing)}")
        if type(result["success"]) is not bool:
            raise ValueError(f"shard result episode {index} success must be bool")
        if type(result["episode_length"]) is not int or result["episode_length"] < 0:
            raise ValueError(f"shard result episode {index} episode_length must be a nonnegative int")
        identity = episode_identity(result)
        if identity in actual:
            raise ValueError(f"duplicate episode result in shard: {identity}")
        if identity not in expected:
            raise ValueError(f"episode is not assigned to shard {shard_idx}of{num_shards}: {identity}")
        actual[identity] = result
    if require_complete and set(actual) != set(expected):
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        raise ValueError(f"inexact shard coverage: missing={len(missing)}, extra={len(extra)}")
    return payload


@contextlib.contextmanager
def output_lock(output_path: str | os.PathLike):
    """Fail fast if another process is producing the same task/K/shard output."""
    lock_path = os.fspath(output_path) + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    handle = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns output shard lock: {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _load_json(path: str | os.PathLike) -> dict:
    with open(path) as handle:
        return json.load(handle)


def merge_task_episode_shards(
    out_dir: str | os.PathLike,
    manifest: dict,
    split_set: str,
    task: str,
    num_shards: int,
    *,
    evaluation_provenance: dict | None = None,
) -> dict:
    """Validate exact, duplicate-free coverage and atomically publish stats.json."""
    validate_episode_manifest(manifest)
    expected_records = manifest_records_for_task(manifest, task)
    if not expected_records:
        raise ValueError(f"task {task!r} is absent from manifest")
    expected = {episode_identity(record): record for record in expected_records}
    merged = {}
    wall_seconds = 0.0
    for shard_idx in range(num_shards):
        path = shard_stats_path(out_dir, split_set, task, shard_idx, num_shards)
        payload = _load_json(path)
        validate_shard_results(
            payload,
            manifest,
            split_set,
            task,
            shard_idx,
            num_shards,
            require_complete=True,
            evaluation_provenance=evaluation_provenance,
        )
        wall_seconds += float(payload.get("wall_seconds", 0.0))
        for result in payload["per_episode"]:
            identity = episode_identity(result)
            if identity in merged:
                raise ValueError(f"duplicate episode across shards: {identity}")
            merged[identity] = result
    if set(merged) != set(expected):
        missing = set(expected) - set(merged)
        extra = set(merged) - set(expected)
        raise ValueError(f"inexact merged coverage: missing={len(missing)}, extra={len(extra)}")

    ordered = [merged[episode_identity(record)] for record in expected_records]
    successes = [bool(result["success"]) for result in ordered]
    # Shards may run concurrently, so their summed worker-seconds are not an elapsed wall clock.
    # Preserve the compute-accounting value, but do not manufacture a task rollouts/hour number.
    merged_performance = summarize_policy_performance(ordered, 0.0)
    merged_performance.update(
        {
            "rollouts_per_hour": None,
            "throughput_scope": "unavailable_without_shared_rollout_wall_clock",
            "aggregate_shard_wall_seconds": round(wall_seconds, 1),
        }
    )
    output = {
        "task": task,
        "split_set": split_set,
        "split": manifest["split"],
        "num_episodes": len(ordered),
        "success_rate": sum(successes) / len(successes),
        "successes": successes,
        "episode_lengths": [int(result["episode_length"]) for result in ordered],
        "horizon": int(expected_records[0]["horizon"]),
        "seed": manifest.get("base_seed"),
        "manifest_sha256": manifest["manifest_sha256"],
        "num_episode_shards": int(num_shards),
        "per_episode": ordered,
        "wall_seconds": round(wall_seconds, 1),
        "wall_seconds_kind": "aggregate_shard_worker_seconds",
        "performance": merged_performance,
    }
    if evaluation_provenance is not None:
        output["evaluation_provenance"] = validate_evaluation_provenance(
            evaluation_provenance,
            episode_manifest_sha256=manifest["manifest_sha256"],
        )
    output_path = os.path.join(os.fspath(out_dir), split_set, task, "stats.json")
    write_json_atomic(output_path, output)
    return output
