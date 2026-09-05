#!/usr/bin/env python3
"""E0: offline velocity matching for training-free AM on the history K/V.

For every held-out episode prefix the runner

1. captures the teacher's per-layer memory Q/K/V at serve precision on two
   consecutive policy chunks (``fit_chunk`` at step t, ``eval_chunk`` at t+16);
2. proves the two query banks share no query row;
3. fits AM per layer on the eval chunk's memory using the *fit* chunk's queries
   (never the queries it is scored on) at each requested ratio;
4. re-runs the whole denoise loop with the compacted memory and records
   ``||v_compact - v_full|| / ||v_full||`` at every flow time; and
5. records a deliberately non-claimable Python-unrolled timing diagnostic.

Arms:

``identity``   full teacher K/V served through the AM seam without compression
               -- a mandatory bitwise parity gate, never a speed baseline.
``am{r}_f0``   ratio r over the whole valid memory, nothing kept exact.
``am{r}_f1``   ratio r with the newest frame (16 tokens) kept bit-exact and
               concatenated under one softmax denominator.
``am{r}_stale``artifact fitted on the *fit* chunk's memory and reused at the
               eval chunk -- the deployment question, since FrameSamp re-samples
               its 32 frames every step instead of appending to a cache.

No gradients are taken anywhere; every reported row carries its own labels.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from robomme_integration.amkv import compaction, driver, metrics, query_bank, stage_e0
from robomme_integration.amkv.patch_contract import require_reviewed_amkv_patch, sha256_file

RESULT_SCHEMA_VERSION = 3
RESULT_KIND = "amkv_e0_velocity_matching_result"
RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_KIND = "amkv_e0_velocity_matching_attempt"
EVIDENCE_EPISODE_PAIRS = 32
IDENTITY_ARM = "identity"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Production runs must match the released FrameSamp+Modul tensor geometry.
# Only the fake-policy orchestration test turns this off.
STRICT_OFFICIAL_GEOMETRY = True
KERNEL_FILES = (
    "attention_matching.py",
    "framesamp_attention_matching.py",
    "framesamp_am_jax.py",
)


def _kernel_shas() -> dict[str, str]:
    """Self-labelling: the read-only kernel this run actually imported."""

    root = Path(__file__).resolve().parents[1] / "training"
    return {name: sha256_file(root / name) for name in KERNEL_FILES}


def _determinism_env() -> dict[str, object]:
    import jax

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "env": {
            name: os.environ.get(name)
            for name in (
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "JAX_ENABLE_X64",
                "PYTHONHASHSEED",
                "TF_CUDNN_DETERMINISTIC",
                "XLA_FLAGS",
            )
        },
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_digest(value: object, *, label: str, pattern: re.Pattern[str]) -> str:
    text = str(value or "")
    if not pattern.fullmatch(text):
        width = 64 if pattern is HEX64 else 40
        raise ValueError(f"{label} must be {width} lowercase hexadecimal characters")
    return text


def _validated_result_identity(args: argparse.Namespace, patch) -> dict[str, object]:
    """Bind the result row to the sealed run manifest and every staged input."""

    run_id = str(getattr(args, "run_id", "") or "")
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run_id is missing or unsafe")
    identity = {
        "run_id": run_id,
        "run_manifest_sha256": _require_digest(
            getattr(args, "run_manifest_sha256", None), label="run_manifest_sha256", pattern=HEX64
        ),
        "code_source_tree_sha256": _require_digest(
            getattr(args, "code_source_tree_sha256", None), label="code_source_tree_sha256", pattern=HEX64
        ),
        "policy_source_archive_sha256": _require_digest(
            getattr(args, "policy_source_archive_sha256", None),
            label="policy_source_archive_sha256",
            pattern=HEX64,
        ),
        "policy_source_receipt_sha256": _require_digest(
            getattr(args, "policy_source_receipt_sha256", None),
            label="policy_source_receipt_sha256",
            pattern=HEX64,
        ),
        "policy_git_sha": _require_digest(
            getattr(args, "policy_git_sha", None), label="policy_git_sha", pattern=HEX40
        ),
        "policy_tree_sha1": _require_digest(
            getattr(args, "policy_tree_sha1", None), label="policy_tree_sha1", pattern=HEX40
        ),
        "checkpoint_inventory_sha256": _require_digest(
            getattr(args, "checkpoint_inventory_sha256", None),
            label="checkpoint_inventory_sha256",
            pattern=HEX64,
        ),
        "fixtures_manifest_sha256": _require_digest(
            getattr(args, "fixtures_manifest_sha256", None),
            label="fixtures_manifest_sha256",
            pattern=HEX64,
        ),
    }
    if identity["policy_git_sha"] != patch.policy_git_sha:
        raise ValueError("result policy_git_sha disagrees with the reviewed AM patch")
    if identity["policy_tree_sha1"] != patch.policy_tree_sha1:
        raise ValueError("result policy_tree_sha1 disagrees with the reviewed whole-policy tree")
    identity.update(
        {
            "official_history_gemma_sha256": _require_digest(
                getattr(patch, "official_source_sha256", None),
                label="official_history_gemma_sha256",
                pattern=HEX64,
            ),
            "amkv_patched_module_sha256": _require_digest(
                getattr(patch, "patched_module_sha256", None),
                label="amkv_patched_module_sha256",
                pattern=HEX64,
            ),
        }
    )

    manifest_path = Path(getattr(args, "run_manifest", "") or "")
    if not manifest_path.is_file():
        raise ValueError(f"sealed run manifest is missing: {manifest_path}")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = document.pop("manifest_sha256", None)
    actual = hashlib.sha256(_canonical_json(document).encode()).hexdigest()
    if claimed != identity["run_manifest_sha256"] or actual != identity["run_manifest_sha256"]:
        raise ValueError("run manifest seal does not match run_manifest_sha256")
    if document.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"run manifest schema_version must be {RUN_MANIFEST_SCHEMA_VERSION}")
    if document.get("kind") != RUN_MANIFEST_KIND:
        raise ValueError(f"run manifest kind must be {RUN_MANIFEST_KIND!r}")
    scientific = document.get("scientific", {})
    scientific_sha = hashlib.sha256(_canonical_json(scientific).encode()).hexdigest()
    if document.get("scientific_spec_sha256") != scientific_sha:
        raise ValueError("run manifest scientific_spec_sha256 does not seal its scientific block")
    derived_run_id = f"amkv-e0-{scientific_sha[:16]}"
    if document.get("run_id") != derived_run_id:
        raise ValueError(
            f"run manifest run_id is not derived from its scientific spec: "
            f"{document.get('run_id')!r} != {derived_run_id!r}"
        )
    policy_source = scientific.get("policy_source", {})
    receipt_path = Path(getattr(args, "policy_source_receipt", "") or "")
    try:
        receipt, receipt_sha = stage_e0.load_source_receipt(
            receipt_path, expected_sha256=identity["policy_source_receipt_sha256"]
        )
    except SystemExit as error:
        raise ValueError(f"policy source receipt validation failed: {error}") from error
    extracted = stage_e0.source_tree_identity(Path(args.policy_source))
    if extracted != receipt["extracted_tree"]:
        raise ValueError("runtime policy source tree disagrees with its create-once stage receipt")
    if receipt["archive"]["sha256"] != identity["policy_source_archive_sha256"]:
        raise ValueError("policy source receipt binds a different source archive")
    if policy_source.get("uri") != receipt["archive"]["uri"]:
        raise ValueError("run manifest policy source URI disagrees with its stage receipt")
    if policy_source.get("receipt_uri") != stage_e0.source_receipt_uri(receipt_sha):
        raise ValueError("run manifest policy source receipt URI is not content-addressed")
    if policy_source.get("git_sha") != receipt["git"]["git_sha"]:
        raise ValueError("run manifest policy Git commit disagrees with its stage receipt")
    if policy_source.get("git_tree_sha1") != receipt["git"]["git_tree_sha1"]:
        raise ValueError("run manifest policy Git tree disagrees with its stage receipt")
    identity["policy_source_receipt_sha256"] = receipt_sha
    identity["policy_source_extracted_tree_sha256"] = extracted["tree_sha256"]
    identity["policy_source_extracted_tree_objects"] = extracted["totals"]["objects"]
    expected = {
        "run_id": document.get("run_id"),
        "code_source_tree_sha256": scientific.get("code", {}).get("sanitized_source_tree_sha256"),
        "policy_source_archive_sha256": scientific.get("policy_source", {}).get("sha256"),
        "policy_source_receipt_sha256": policy_source.get("receipt_sha256"),
        "policy_source_extracted_tree_sha256": policy_source.get("extracted_tree_sha256"),
        "policy_source_extracted_tree_objects": policy_source.get("extracted_tree_objects"),
        "policy_git_sha": scientific.get("policy_source", {}).get("git_sha"),
        "policy_tree_sha1": scientific.get("policy_source", {}).get("git_tree_sha1"),
        "checkpoint_inventory_sha256": scientific.get("checkpoint", {}).get("inventory_sha256"),
        "fixtures_manifest_sha256": scientific.get("fixtures", {}).get("manifest_sha256"),
    }
    mismatches = {name: (identity[name], value) for name, value in expected.items() if identity[name] != value}
    if mismatches:
        raise ValueError(f"runtime result identity disagrees with sealed run manifest: {mismatches}")
    evaluation = scientific.get("evaluation", {})
    configured = {
        "ratios": scientific.get("ratios"),
        "runtime_dtype": evaluation.get("runtime_dtype"),
        "num_flow_steps": evaluation.get("num_flow_steps"),
        "model_seed": evaluation.get("model_seed"),
        "noise_seed": evaluation.get("noise_seed"),
        "minimum_episodes": evaluation.get("minimum_episodes"),
        "timing_repeats": evaluation.get("timing_repeats"),
    }
    runtime = {
        "ratios": [float(value) for value in args.ratios],
        "runtime_dtype": args.runtime_dtype,
        "num_flow_steps": int(args.num_steps),
        "model_seed": int(args.model_seed),
        "noise_seed": int(args.noise_seed),
        "minimum_episodes": int(args.minimum_episodes),
        "timing_repeats": int(args.timing_repeats),
    }
    config_mismatches = {
        name: (runtime[name], configured[name]) for name in runtime if runtime[name] != configured[name]
    }
    if config_mismatches:
        raise ValueError(f"runtime evaluation config disagrees with sealed run manifest: {config_mismatches}")
    if int(args.limit) != 0:
        raise ValueError("sealed H10 E0 evidence forbids an unmanifested --limit")
    identity["evidence_input_identity_sha256"] = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
    return identity


@dataclasses.dataclass(frozen=True)
class ChunkCapture:
    """Everything one policy chunk contributes: taps, bank, full-cache trace."""

    fixture_id: str
    role: str
    step_idx: int
    observation: object
    history: object
    trace: driver.DenoiseTrace
    bank: query_bank.ActionQueryBank
    meta: dict


def _make_denoisers(model) -> dict:
    """Build each jitted configuration exactly once.

    A ``Denoiser`` freezes its own ``nnx`` graphdef/state at construction, so
    both survive later module swaps.  Constructing one per arm would recompile
    the whole scanned model hundreds of times.
    """

    # Split the restored official graph before swapping any Linen class.  This
    # is the only trustworthy baseline for proving the source patch itself is
    # an identity operation when no AM artifact is supplied.
    official = driver.Denoiser(model, capture=False)
    with driver.installed_patch(model, capture=True):
        capture = driver.Denoiser(model, capture=True)
    with driver.installed_patch(model, capture=False):
        plain = driver.Denoiser(model, capture=False)
    return {"official": official, "capture": capture, "plain": plain}


def _capture_chunk(denoisers, policy, record, *, role, noise, num_steps, noise_sha):
    observation, history, meta = driver.build_observation(policy, record)
    trace = denoisers["capture"](observation, noise=noise, num_steps=num_steps)
    if trace.queries is None or trace.memory_keys is None:
        raise RuntimeError("tap capture returned no queries; the AM patch is not installed")
    if STRICT_OFFICIAL_GEOMETRY:
        driver.validate_official_capture(trace, num_steps=num_steps)
        if int(meta.get("valid_memory_tokens", -1)) != driver.TOKEN_BUDGET:
            raise ValueError(
                "H10 E0 requires a fully populated 512-token FrameSamp memory; "
                f"{record.fixture_id} has {meta.get('valid_memory_tokens')} valid tokens"
            )
    bank = query_bank.bank_from_traced_queries(
        trace.queries,
        fixture_id=record.fixture_id,
        chunk_role=role,
        step_idx=int(record.step_idx),
        flow_times=trace.flow_times,
        noise_sha256=noise_sha,
    )
    return ChunkCapture(
        fixture_id=record.fixture_id,
        role=role,
        step_idx=int(record.step_idx),
        observation=observation,
        history=history,
        trace=trace,
        bank=bank,
        meta=meta,
    )


def _probe_pack_is_consumed(denoisers, policy, record, *, noise, num_steps, dtype) -> dict:
    """Decisive guard against a silently ignored AM pack.

    If the scanned patch never received the pack -- for example because the
    swapped linen module did not enter the frozen graphdef -- every AM arm
    would quietly reproduce the full cache and E0 would report a perfect,
    meaningless result.  Zeroing every memory value must move the actions.
    """

    observation, _, _ = driver.build_observation(policy, record)
    full = denoisers["capture"](observation, noise=noise, num_steps=num_steps)
    destroyed = _run_arm(
        denoisers,
        observation,
        noise=noise,
        num_steps=num_steps,
        pack_arrays=compaction.destroyed_pack_arrays(
            keys_post_rope=full.memory_keys,
            values_post_projection=full.memory_values,
            runtime_dtype=dtype,
        ),
        dtype=dtype,
    )
    delta = float(np.max(np.abs(destroyed.actions - full.actions)))
    if delta == 0.0:
        raise RuntimeError(
            "zeroing every memory value did not change the policy output: the AM pack is not being "
            "consumed by the scanned MemoryAttention, so no AM number from this run is valid"
        )
    return {"destroyed_memory_max_abs_action_delta": delta, "pack_consumed": True}


def _run_arm(denoisers, observation, *, noise, num_steps, pack_arrays, dtype, teacher_states=None):
    pack = driver.am_pack_from_arrays(pack_arrays, dtype=dtype) if isinstance(pack_arrays, dict) else pack_arrays
    return denoisers["plain"](
        observation, noise=noise, num_steps=num_steps, am_pack=pack, teacher_states=teacher_states
    )


def _run_arm_pair(denoisers, observation, *, noise, num_steps, pack_arrays, dtype, teacher_states):
    """Two passes, because the two estimands are different questions.

    The per-flow-time velocity delta must hold ``x_t`` fixed at the teacher's
    own denoising state, otherwise the compact run has already drifted to a
    different input by t=0.9 and the metric conflates "different function" with
    "different point".  The action delta must NOT be teacher forced: it is the
    closed-loop quantity that a rollout would actually execute.
    """

    forced = _run_arm(
        denoisers,
        observation,
        noise=noise,
        num_steps=num_steps,
        pack_arrays=pack_arrays,
        dtype=dtype,
        teacher_states=teacher_states,
    )
    free = _run_arm(denoisers, observation, noise=noise, num_steps=num_steps, pack_arrays=pack_arrays, dtype=dtype)
    return forced, free


def _identity_trace_gate(reference: driver.DenoiseTrace, candidate: driver.DenoiseTrace, *, fixture_id: str) -> dict:
    """Fail closed unless a full-K/V beta-zero route is exactly the teacher."""

    rows: dict[str, dict[str, object]] = {}
    for name in ("actions", "velocities", "denoise_states"):
        expected = np.asarray(getattr(reference, name))
        actual = np.asarray(getattr(candidate, name))
        if expected.shape != actual.shape:
            raise RuntimeError(f"identity {name} shape mismatch for {fixture_id}: {expected.shape} != {actual.shape}")
        delta = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
        bitwise = bool(np.array_equal(expected, actual))
        if not bitwise:
            raise RuntimeError(f"full-K/V beta-zero identity failed for {fixture_id} {name}; max|delta|={delta}")
        rows[name] = {"bitwise": True, "max_abs_delta": delta}
    return {"fixture_id": fixture_id, "arm_id": IDENTITY_ARM, "fields": rows}


def _fit_arm(history, plan, *, bank_pair, keys, values, dtype):
    bank_pair.validate()
    start = time.perf_counter()
    fits = compaction.fit_all_layers(
        history,
        plan,
        fit_queries=bank_pair.fit.queries,
        heldout_queries=bank_pair.heldout.queries,
        keys_post_rope=keys,
        values_post_projection=values,
    )
    fit_seconds = time.perf_counter() - start
    arrays = compaction.stack_am_pack_arrays(
        fits, plan, keys_post_rope=keys, values_post_projection=values, runtime_dtype=dtype
    )
    served = compaction.served_heldout_diagnostics(
        fits,
        plan,
        heldout_queries=bank_pair.heldout.queries,
        keys_post_rope=keys,
        values_post_projection=values,
        pack_arrays=arrays,
    )
    return fits, arrays, served, fit_seconds


def _arm_specs(ratios: tuple[float, ...]) -> tuple[dict, ...]:
    """Fitted arms plus the two controls that make their numbers readable."""

    hardest = max(ratios)
    specs = [
        {"arm_id": f"am{ratio:g}_f0", "kind": "am", "ratio": ratio, "exact_recent_frames": 0, "stale": False}
        for ratio in ratios
    ]
    specs.append(
        {"arm_id": f"am{hardest:g}_f1", "kind": "am", "ratio": hardest, "exact_recent_frames": 1, "stale": False}
    )
    specs.append(
        {"arm_id": f"am{hardest:g}_f0_stale", "kind": "am", "ratio": hardest, "exact_recent_frames": 0, "stale": True}
    )
    specs.extend(
        {"arm_id": f"drop{ratio:g}_random", "kind": "random", "ratio": ratio, "exact_recent_frames": 0, "stale": False}
        for ratio in ratios
    )
    specs.append(
        {"arm_id": "memory_destroyed", "kind": "destroy", "ratio": 1.0, "exact_recent_frames": 0, "stale": False}
    )
    return tuple(specs)


def run(args: argparse.Namespace) -> dict:
    import jax.numpy as jnp

    from robomme_integration.amkv.episodes import load_fixture_bundle

    patch = require_reviewed_amkv_patch(args.policy_source)
    result_identity = _validated_result_identity(args, patch)
    if args.runtime_dtype != "bfloat16":
        raise ValueError("H10 E0 evidence is restricted to the official bfloat16 serve dtype")
    records = load_fixture_bundle(args.fixtures)
    pairs: dict[str, dict[str, object]] = {}
    for record in records:
        pairs.setdefault(record.pair_id, {})[record.chunk_role] = record
    complete = [
        (pair_id, value[query_bank.FIT_CHUNK], value[query_bank.EVAL_CHUNK])
        for pair_id, value in sorted(pairs.items())
        if query_bank.FIT_CHUNK in value and query_bank.EVAL_CHUNK in value
    ]
    if args.limit:
        complete = complete[: args.limit]
    if STRICT_OFFICIAL_GEOMETRY and len(complete) != EVIDENCE_EPISODE_PAIRS:
        raise SystemExit(
            f"sealed H10 E0 evidence requires exactly {EVIDENCE_EPISODE_PAIRS} complete fixture pairs, "
            f"found {len(complete)}"
        )
    if len(complete) < args.minimum_episodes:
        raise SystemExit(f"E0 needs at least {args.minimum_episodes} complete chunk pairs, found {len(complete)}")

    policy = driver.load_policy(args.checkpoint, policy_source_root=args.policy_source, seed=args.model_seed)
    model = policy._model  # noqa: SLF001
    dtype = jnp.bfloat16
    noise = driver.sample_noise(
        args.noise_seed, batch=1, action_horizon=model.action_horizon, action_dim=model.action_dim
    )
    noise_sha = query_bank.array_sha256(np.asarray(noise, dtype=np.float32))

    first_observation, _, _ = driver.build_observation(policy, complete[0][1])
    # Build the jitted configurations first so the self-test exercises the very
    # denoiser the arms will use, and so the model is left holding the official
    # module for `sample_actions`.
    denoisers = _make_denoisers(model)
    selftest = driver.selftest_matches_official_sampler(
        policy, first_observation, noise=noise, num_steps=args.num_steps, denoiser=denoisers["plain"]
    )
    if STRICT_OFFICIAL_GEOMETRY:
        selftest.update(
            driver.selftest_patched_identity(
                official_denoiser=denoisers["official"],
                capture_denoiser=denoisers["capture"],
                patched_denoiser=denoisers["plain"],
                observation=first_observation,
                noise=noise,
                num_steps=args.num_steps,
                dtype=dtype,
            )
        )
    selftest["am_pack_reaches_the_model"] = _probe_pack_is_consumed(
        denoisers, policy, complete[0][1], noise=noise, num_steps=args.num_steps, dtype=dtype
    )

    specs = _arm_specs(tuple(args.ratios))
    comparisons: dict[str, list[metrics.VelocityComparison]] = {}
    layer_diagnostics: dict[str, list[dict]] = {}
    plans: dict[str, dict] = {}
    fit_seconds: dict[str, list[float]] = {}
    per_episode: list[dict] = []
    parity: list[dict] = []
    payload_quantization: dict[str, dict] = {}
    timing_input: tuple[ChunkCapture, ChunkCapture, query_bank.DisjointQueryBankPair] | None = None

    for pair_id, fit_record, eval_record in complete:
        fit_chunk = _capture_chunk(
            denoisers,
            policy,
            fit_record,
            role=query_bank.FIT_CHUNK,
            noise=noise,
            num_steps=args.num_steps,
            noise_sha=noise_sha,
        )
        eval_chunk = _capture_chunk(
            denoisers,
            policy,
            eval_record,
            role=query_bank.EVAL_CHUNK,
            noise=noise,
            num_steps=args.num_steps,
            noise_sha=noise_sha,
        )
        pair = query_bank.pair_disjoint_banks(fit_chunk.bank, eval_chunk.bank)
        if timing_input is None:
            timing_input = (fit_chunk, eval_chunk, pair)
        full = eval_chunk.trace

        identity = _run_arm(
            denoisers,
            eval_chunk.observation,
            noise=noise,
            num_steps=args.num_steps,
            pack_arrays=driver.identity_am_pack(full.memory_keys, full.memory_values, dtype=dtype),
            dtype=dtype,
        )
        parity_row = _identity_trace_gate(full, identity, fixture_id=eval_chunk.fixture_id)
        parity_row["memory_kv_recomputed_per_flow_step"] = full.memory_kv_recomputed_per_step
        parity.append(parity_row)

        for spec in specs:
            arm_id = spec["arm_id"]
            source = fit_chunk if spec["stale"] else eval_chunk
            fits: tuple = ()
            served: tuple = ()
            seconds = 0.0
            if spec["kind"] == "destroy":
                plan = None
                arrays = compaction.destroyed_pack_arrays(
                    keys_post_rope=eval_chunk.trace.memory_keys,
                    values_post_projection=eval_chunk.trace.memory_values,
                    runtime_dtype=dtype,
                )
            else:
                plan = compaction.plan_compaction(
                    source.history, spec["ratio"], exact_recent_frames=spec["exact_recent_frames"]
                )
                if spec["kind"] == "random":
                    arrays = compaction.random_subset_pack_arrays(
                        plan,
                        keys_post_rope=source.trace.memory_keys,
                        values_post_projection=source.trace.memory_values,
                        seed=args.noise_seed,
                        runtime_dtype=dtype,
                    )
                else:
                    fits, arrays, served, seconds = _fit_arm(
                        source.history,
                        plan,
                        bank_pair=pair,
                        keys=source.trace.memory_keys,
                        values=source.trace.memory_values,
                        dtype=dtype,
                    )
            compact_forced, compact_free = _run_arm_pair(
                denoisers,
                eval_chunk.observation,
                noise=noise,
                num_steps=args.num_steps,
                pack_arrays=arrays,
                dtype=dtype,
                teacher_states=full.denoise_states,
            )
            comparison = metrics.compare_velocity_traces(
                fixture_id=eval_chunk.fixture_id,
                arm_id=arm_id,
                flow_times=full.flow_times,
                full_velocity=full.velocities,
                compact_velocity=compact_forced.velocities,
                full_actions=full.actions,
                compact_actions=compact_free.actions,
            )
            comparisons.setdefault(arm_id, []).append(comparison)
            fit_seconds.setdefault(arm_id, []).append(seconds)
            plans.setdefault(arm_id, plan.label() if plan is not None else {"method": "memory_values_zeroed"})
            if fits and arm_id not in payload_quantization:
                payload_quantization[arm_id] = compaction.payload_quantization_parity(
                    fits,
                    plan,
                    heldout_queries=eval_chunk.bank.queries,
                    keys_post_rope=source.trace.memory_keys,
                    values_post_projection=source.trace.memory_values,
                    runtime_dtype=dtype,
                )
            if fits:
                layer_diagnostics.setdefault(arm_id, []).append(
                    {
                        "fixture_id": eval_chunk.fixture_id,
                        "layers": [
                            {**fit.label(), **served_row} for fit, served_row in zip(fits, served, strict=True)
                        ],
                    }
                )

        per_episode.append(
            {
                "pair_id": pair_id,
                "fit_chunk": fit_chunk.meta,
                "eval_chunk": eval_chunk.meta,
                "query_banks": pair.label(),
            }
        )

    if timing_input is None:
        raise RuntimeError("no complete fixture pair was available for the timing diagnostic")
    timing = _tracing_microbenchmark(denoisers, *timing_input, specs=specs, noise=noise, args=args, dtype=dtype)
    aggregates = {
        arm_id: metrics.aggregate_comparisons(tuple(values)) for arm_id, values in sorted(comparisons.items())
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "experiment": "H10_E0_offline_velocity_matching",
        "identity": result_identity,
        "labels": {
            "patch": patch.to_dict(),
            "kernel_sha256": _kernel_shas(),
            "checkpoint": str(args.checkpoint),
            "fixtures": str(args.fixtures),
            "runtime_dtype": args.runtime_dtype,
            "num_flow_steps": args.num_steps,
            "model_seed": args.model_seed,
            "noise_seed": args.noise_seed,
            "noise_sha256": noise_sha,
            "ratios": [float(value) for value in args.ratios],
            "gradients": "none_serve_path_cache_transform",
            "provenance": result_identity,
            "determinism": _determinism_env(),
        },
        "selftest": selftest,
        "identity_parity": parity,
        "fixture_proof": {
            "complete_pair_count": len(complete),
            "fixture_record_count": len(records),
            "required_pair_count": EVIDENCE_EPISODE_PAIRS,
            "exact_pair_count": len(complete) == EVIDENCE_EPISODE_PAIRS,
            "full_memory_tokens": driver.TOKEN_BUDGET if STRICT_OFFICIAL_GEOMETRY else None,
        },
        "plans": plans,
        "aggregates": aggregates,
        "per_arm_fit_seconds": {
            arm_id: {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "episodes": len(values),
            }
            for arm_id, values in sorted(fit_seconds.items())
        },
        "tracing_microbenchmark": timing,
        "per_episode": per_episode,
        "per_episode_comparisons": {
            arm_id: [comparison.to_dict() for comparison in values] for arm_id, values in sorted(comparisons.items())
        },
        "layer_diagnostics": layer_diagnostics,
        "payload_quantization": payload_quantization,
    }


def _tracing_microbenchmark(
    denoisers,
    fit_chunk: ChunkCapture,
    eval_chunk: ChunkCapture,
    bank_pair: query_bank.DisjointQueryBankPair,
    *,
    specs,
    noise,
    args,
    dtype,
) -> dict:
    """Non-claimable timing of the Python-unrolled denoiser only.

    This deliberately does not pretend to be end-to-end policy latency: fixture
    construction, teacher tap capture, AM fitting, host-to-device placement,
    policy-server serialization, and simulator time are outside the timed call.
    It is useful for catching gross regressions while developing the seam, but
    no speedup may be computed or reported from it.
    """

    import jax.numpy as jnp

    bank_pair.validate()
    if bank_pair.fit.bank_id() != fit_chunk.bank.bank_id() or bank_pair.heldout.bank_id() != eval_chunk.bank.bank_id():
        raise ValueError("timing query pair does not belong to the supplied fit/eval chunks")
    observation = eval_chunk.observation
    history = eval_chunk.history
    capture = eval_chunk.trace
    head_dim = int(capture.memory_keys.shape[-1])
    layers = int(capture.memory_keys.shape[0])
    itemsize = int(jnp.zeros((), dtype=dtype).itemsize)

    rows: dict[str, object] = {}
    denoiser = denoisers["plain"]
    if True:  # keep the measured block indented as one unit
        rows["full_denoise"] = driver.timed(
            lambda: denoiser(observation, noise=noise, num_steps=args.num_steps).actions,
            repeats=args.timing_repeats,
        )
        identity = driver.identity_am_pack(capture.memory_keys, capture.memory_values, dtype=dtype)
        rows["identity_denoise"] = driver.timed(
            lambda: denoiser(observation, noise=noise, num_steps=args.num_steps, am_pack=identity).actions,
            repeats=args.timing_repeats,
        )
        for spec in specs:
            if spec["stale"] or spec["kind"] != "am":
                continue
            plan = compaction.plan_compaction(history, spec["ratio"], exact_recent_frames=spec["exact_recent_frames"])
            fits, arrays, served, seconds = _fit_arm(
                history,
                plan,
                bank_pair=bank_pair,
                keys=capture.memory_keys,
                values=capture.memory_values,
                dtype=dtype,
            )
            del fits, served
            pack = driver.am_pack_from_arrays(arrays, dtype=dtype)
            rows[f"{spec['arm_id']}_denoise"] = driver.timed(
                lambda pack=pack: denoiser(observation, noise=noise, num_steps=args.num_steps, am_pack=pack).actions,
                repeats=args.timing_repeats,
            )
            rows[f"{spec['arm_id']}_fit_seconds"] = seconds
            rows[f"{spec['arm_id']}_kv_bytes"] = compaction.served_kv_bytes(
                plan, layers=layers, head_dim=head_dim, itemsize=itemsize
            )
    rows["memory_kv_recomputed_per_flow_step"] = capture.memory_kv_recomputed_per_step
    rows["layers"] = layers
    rows["head_dim"] = head_dim
    rows["flow_steps"] = args.num_steps
    rows["measurement_kind"] = "python_unrolled_denoiser_microbenchmark_v1"
    rows["speedup_claim_permitted"] = False
    rows["fit_query_bank_id"] = bank_pair.fit.bank_id()
    rows["heldout_query_bank_id"] = bank_pair.heldout.bank_id()
    rows["query_banks_disjoint"] = True
    rows["excluded_from_timed_denoise"] = [
        "observation_and_history_construction",
        "teacher_qkv_capture",
        "attention_matching_fit",
        "artifact_host_to_device_placement",
        "policy_server_rpc_and_serialization",
        "simulator",
    ]
    rows["timing_note"] = (
        "Development-only Python-unrolled denoiser microbenchmark. It is not comparable end-to-end serve latency, "
        "and neither ratios nor speedup claims may be derived from these samples. Per-arm fit_seconds are reported "
        "separately and are also not included in denoise samples."
    )
    return rows


def _parse_ratios(value: str) -> tuple[float, ...]:
    ratios = tuple(float(item) for item in str(value).split(",") if item.strip())
    if not ratios or any(ratio <= 1 for ratio in ratios):
        raise argparse.ArgumentTypeError(f"ratios must be a comma separated list of values > 1, got {value!r}")
    return ratios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratios", type=_parse_ratios, default=(4.0, 8.0))
    parser.add_argument("--num-steps", type=int, default=driver.DEFAULT_NUM_FLOW_STEPS)
    parser.add_argument("--model-seed", type=int, default=driver.DEFAULT_MODEL_SEED)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--runtime-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--minimum-episodes", type=int, default=32)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--run-manifest", required=True)
    # Provenance passed through by the node entry and cross-checked against the
    # sealed manifest.  Missing or inconsistent identity aborts before loading
    # the checkpoint; an unlabelled number cannot be emitted.
    for name in (
        "--run-id",
        "--run-manifest-sha256",
        "--code-source-tree-sha256",
        "--policy-source-receipt",
        "--policy-source-receipt-sha256",
        "--policy-source-archive-sha256",
        "--policy-git-sha",
        "--policy-tree-sha1",
        "--checkpoint-inventory-sha256",
        "--fixtures-manifest-sha256",
    ):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)

    results = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, out)
    print(json.dumps({"wrote": str(out), "arms": sorted(results["aggregates"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
