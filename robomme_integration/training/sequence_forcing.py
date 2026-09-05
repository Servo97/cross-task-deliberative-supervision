"""Isolated shared-flow-time control for the RoboMME Stage-Q ablation.

The stock Q0/Q2 path flattens ``[B, L, ...]`` to ``[B*L, ...]`` before calling
``Pi0.compute_loss``.  Consequently both the flow time and Gaussian action noise are sampled
independently for every chunk.  The two ``*_noforce`` controls change exactly one tensor:

* draw the stock ``B*L`` flow times from the stock time key;
* keep the first draw in each contiguous window and broadcast it across that window;
* leave preprocessing and the stock ``B*L`` Gaussian-noise draw inside ``compute_loss`` intact.

Pi0 historically has no explicit-flow-time API.  We therefore require a content-recorded,
node-local source overlay which adds one optional keyword.  The default branch is syntactically
the old function plus a dead ``if`` and remains the route for every existing arm.  This module
never mutates the shared RoboCasa checkout and never monkeypatches ``jax.random``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, Callable

OVERLAY_VERSION = "robomme-shared-tau-v1"
OVERLAY_MARKER_NAME = "_ROBOMME_SEQUENCE_FORCING_OVERLAY"
PI0_RELATIVE_PATH = Path("src/openpi/models/pi0.py")
BASE_ARCHIVE_SHA256 = "ed923b2c27d2f608d62cc4b5ca89d5b80c14739dba1ab81d6f53d8013bcb66ad"
BASE_PI0_SHA256 = "15b4a12df6f9650f1f53a7d77f9b59f2342502ceb576d2a073eda754c4ceb72b"
PATCHED_PI0_SHA256 = "3cb9c0a6a239315ff6f01c5cd9903776aa93fc26f66698c68000ad29ff00eb72"

_OLD_SIGNATURE = """    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, \"*b ah\"]:
"""
_NEW_SIGNATURE = """    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        flow_time: at.Float[at.Array, \" *b\"] | None = None,
    ) -> at.Float[at.Array, \"*b ah\"]:
"""
_OLD_TIME_BLOCK = """        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
"""
_NEW_TIME_BLOCK = """        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        if flow_time is not None:
            # The no-forcing trainer derives this from the SAME stock time_rng and supplies exactly
            # batch_shape.  Shape validation is static under JIT and prevents accidental broadcast.
            if flow_time.shape != time.shape:
                raise ValueError(
                    f\"explicit flow_time shape {flow_time.shape} does not match batch shape {time.shape}\"
                )
            time = flow_time.astype(time.dtype)
        time_expanded = time[..., None, None]
"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _find_pi0_compute_loss(tree: ast.AST) -> ast.FunctionDef:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Pi0.compute_loss, found {len(matches)}")
    return matches[0]


def _validate_unpatched_source(source: str) -> None:
    """Fail closed unless the source exposes the exact seam this overlay was reviewed against."""

    if OVERLAY_MARKER_NAME in source:
        raise ValueError("source is already a sequence-forcing overlay; refusing a stacked patch")
    if source.count(_OLD_SIGNATURE) != 1:
        raise ValueError("unreviewed Pi0.compute_loss signature; shared-tau overlay not applied")
    if source.count(_OLD_TIME_BLOCK) != 1:
        raise ValueError("unreviewed Pi0 flow-time sampling block; shared-tau overlay not applied")
    fn = _find_pi0_compute_loss(ast.parse(source))
    positional = [arg.arg for arg in fn.args.args]
    keyword_only = [arg.arg for arg in fn.args.kwonlyargs]
    if positional != ["self", "rng", "observation", "actions"] or keyword_only != ["train"]:
        raise ValueError(
            f"unreviewed Pi0.compute_loss arguments: positional={positional}, keyword_only={keyword_only}"
        )


def patch_pi0_source(source: str) -> str:
    """Return the minimal reviewed Pi0 source overlay; never writes the input checkout."""

    _validate_unpatched_source(source)
    marker = f'{OVERLAY_MARKER_NAME} = "{OVERLAY_VERSION}"\n'
    # Put the marker immediately before the first domain separator.  This location is module-level
    # across every reviewed WSM Pi0 source while leaving imports and class structure untouched.
    anchor = "# Stable domain separator for workspace CFG dropout."
    if source.count(anchor) != 1:
        raise ValueError("unreviewed Pi0 module header; overlay marker anchor is missing or ambiguous")
    patched = source.replace(anchor, marker + "\n" + anchor, 1)
    patched = patched.replace(_OLD_SIGNATURE, _NEW_SIGNATURE, 1)
    patched = patched.replace(_OLD_TIME_BLOCK, _NEW_TIME_BLOCK, 1)
    compile(patched, str(PI0_RELATIVE_PATH), "exec")
    return patched


def stage_openpi_overlay(
    source_repo: Path,
    output_repo: Path,
    *,
    source_archive_sha256: str,
) -> dict[str, Any]:
    """Create a small node-local OpenPI runtime containing the guarded Pi0 overlay.

    Only the 2 MiB Python package and training scripts are copied.  No checkpoints, datasets,
    virtual environments, git metadata, caches, or build products are duplicated.
    """

    source_repo = source_repo.resolve()
    output_repo = output_repo.resolve()
    if source_archive_sha256 != BASE_ARCHIVE_SHA256:
        raise ValueError(
            f"shared-tau overlay requires canonical ed923 archive {BASE_ARCHIVE_SHA256}, got {source_archive_sha256}"
        )
    source_pi0 = source_repo / PI0_RELATIVE_PATH
    source_package = source_repo / "src/openpi"
    source_scripts = source_repo / "scripts"
    if not source_pi0.is_file() or not source_package.is_dir() or not source_scripts.is_dir():
        raise ValueError(f"{source_repo} is not a complete OpenPI source runtime")
    if output_repo == source_repo or source_repo in output_repo.parents:
        raise ValueError("overlay output must be outside the source checkout")
    original_bytes = source_pi0.read_bytes()
    original_sha256 = _sha256_bytes(original_bytes)
    if original_sha256 != BASE_PI0_SHA256:
        raise ValueError(f"canonical ed923 Pi0 source drifted: {original_sha256} != {BASE_PI0_SHA256}")
    patched = patch_pi0_source(original_bytes.decode("utf-8"))
    patched_sha256 = _sha256_bytes(patched.encode("utf-8"))
    if patched_sha256 != PATCHED_PI0_SHA256:
        raise ValueError(f"reviewed shared-tau Pi0 overlay drifted: {patched_sha256} != {PATCHED_PI0_SHA256}")
    manifest = {
        "schema_version": 1,
        "overlay_version": OVERLAY_VERSION,
        "base_archive_sha256": source_archive_sha256,
        "source_repo": str(source_repo),
        "source_pi0_sha256": original_sha256,
        "patched_pi0_sha256": patched_sha256,
        "copied_roots": ["src/openpi", "scripts"],
        "scientific_delta": "share flow tau across L=8; epsilon remains per flattened chunk",
        "activate": f"PYTHONPATH={output_repo / 'src'}:$PYTHONPATH",
    }
    manifest_path = output_repo / "robomme_sequence_forcing_overlay.json"
    patched_path = output_repo / PI0_RELATIVE_PATH
    if output_repo.exists():
        # A same-node campaign may run many cells through one shared source cache. Reuse is allowed
        # only after a full content check; partial/stale/corrupt directories fail closed.
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            patched_sha = _sha256_bytes(patched_path.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError(f"existing overlay is incomplete or malformed: {output_repo}") from error
        invariant_keys = (
            "schema_version",
            "overlay_version",
            "base_archive_sha256",
            "source_pi0_sha256",
            "patched_pi0_sha256",
            "scientific_delta",
        )
        drift = {
            key: (existing.get(key), manifest[key]) for key in invariant_keys if existing.get(key) != manifest[key]
        }
        if drift or patched_sha != manifest["patched_pi0_sha256"] or not (output_repo / "scripts/train.py").is_file():
            raise ValueError(
                f"existing overlay failed content verification: drift={drift} actual_patched_sha256={patched_sha}"
            )
        return existing

    output_repo.mkdir(parents=True)
    ignore_runtime_junk = shutil.ignore_patterns("__pycache__", "*.py[co]")
    shutil.copytree(source_package, output_repo / "src/openpi", ignore=ignore_runtime_junk)
    shutil.copytree(source_scripts, output_repo / "scripts", ignore=ignore_runtime_junk)
    patched_path.write_text(patched, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_loaded_overlay() -> None:
    """Verify this process imported the staged Pi0 before a no-forcing run can start."""

    import openpi.models.pi0 as pi0_model

    marker = getattr(pi0_model, OVERLAY_MARKER_NAME, None)
    parameters = inspect.signature(pi0_model.Pi0.compute_loss).parameters
    if marker != OVERLAY_VERSION or "flow_time" not in parameters:
        raise RuntimeError(
            "q0_noforce/q2_noforce require the guarded node-local OpenPI shared-tau overlay; "
            f"loaded={Path(pi0_model.__file__).resolve()} marker={marker!r} "
            f"flow_time_api={'flow_time' in parameters}. Stage it with "
            "`python -m robomme_integration.training.sequence_forcing stage ...` and prepend its "
            "src directory to PYTHONPATH before starting Python."
        )


def shared_flow_time_from_stock_rng(rng, batch_size: int, length: int):
    """Use Pi0's stock time key/draw, then share the first tau within each window.

    Drawing all ``B*L`` values first makes the intervention a transparent tensor transform of the
    default Q0/Q2 draw.  It does not widen the stock three-way split or perturb augmentation/noise
    keys.  ``batch_size`` and ``length`` are static shapes under the compiled train step.
    """

    if batch_size < 1 or length < 1:
        raise ValueError(f"batch_size and length must be positive, got {batch_size}, {length}")
    import jax
    import jax.numpy as jnp

    _, _, time_rng = jax.random.split(rng, 3)
    independent = jax.random.beta(time_rng, 1.5, 1, (batch_size * length,)) * 0.999 + 0.001
    by_window = independent.reshape(batch_size, length)
    return jnp.broadcast_to(by_window[:, :1], by_window.shape).reshape(batch_size * length)


def build_noforce_stage_q_train_step(train_module) -> Callable:
    """Build the Stage-Q step with one reviewed delta: explicit shared flow time."""

    validate_loaded_overlay()
    jax = train_module.jax
    jnp = train_module.jnp
    nnx = train_module.nnx
    optax = train_module.optax
    dataclasses = train_module.dataclasses

    def noforce_stage_q_train_step(config, rng, state, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model, step_rng, observation, actions):
            batch_size, length = actions.shape[0], actions.shape[1]
            obs_flat = jax.tree.map(lambda value: value.reshape(batch_size * length, *value.shape[2:]), observation)
            actions_flat = actions.reshape(batch_size * length, *actions.shape[2:])
            if getattr(model, "robottt", False):
                state_seq = obs_flat.state.reshape(batch_size, length, -1)
                conditions, _ = model.robottt_fast.run_sequence(state_seq, actions)
                conditions = jnp.swapaxes(conditions, 0, 1).reshape(batch_size * length, -1)
                obs_flat = dataclasses.replace(obs_flat, robottt_cond=conditions)
            shared_tau = shared_flow_time_from_stock_rng(step_rng, batch_size, length)
            losses = model.compute_loss(
                step_rng,
                obs_flat,
                actions_flat,
                train=True,
                flow_time=shared_tau,
            )
            return jnp.mean(losses)

        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch
        diff_state = nnx.DiffState(0, config.trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

        # Keep the current upstream staged-recipe and optimizer code, including its exact ordering.
        grads = train_module.staged_mask_grads(config.staged, state.step, grads)
        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        updates = train_module.staged_mask_updates(config.staged, state.step, updates)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_params = nnx.state(model)

        new_state = dataclasses.replace(
            state,
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )
        info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
        if config.staged is not None:
            info["staged_phase1"] = train_module._staged_phase1_indicator(config.staged, state.step)
        return new_state, info

    noforce_stage_q_train_step.__name__ = "noforce_stage_q_train_step"
    noforce_stage_q_train_step.__doc__ = (
        "Stage-Q with shared flow tau per contiguous window and independent action noise per chunk."
    )
    return noforce_stage_q_train_step


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage", help="create a guarded node-local OpenPI overlay")
    stage.add_argument("--source-repo", type=Path, required=True)
    stage.add_argument("--output-repo", type=Path, required=True)
    stage.add_argument("--source-archive-sha256", required=True)
    args = parser.parse_args()
    if args.command == "stage":
        print(
            json.dumps(
                stage_openpi_overlay(
                    args.source_repo,
                    args.output_repo,
                    source_archive_sha256=args.source_archive_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    _main()
