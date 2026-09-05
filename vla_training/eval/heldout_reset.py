"""Exact recorded-demo reset support for the RoboCasa evaluation client.

The sequence mirrors robocasa's robomimic_env_wrapper.reset_to against the gym wrapper's underlying
Kitchen environment. It restores episode metadata, edited MuJoCo XML, and the first recorded state.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path


def _read_verified(path: Path, descriptor: dict | None) -> bytes:
    data = path.read_bytes()
    if descriptor is None:
        return data
    expected_size = descriptor.get("size")
    expected_digest = descriptor.get("sha256")
    actual_digest = hashlib.sha256(data).hexdigest()
    if len(data) != expected_size or actual_digest != expected_digest:
        raise ValueError(
            f"exact-reset artifact integrity mismatch for {path}: "
            f"size {len(data)} (expected {expected_size}), "
            f"sha256 {actual_digest} (expected {expected_digest})"
        )
    return data


def load_episode_state(extras_dir: str | Path, artifacts: dict | None = None) -> dict:
    """Load the immutable reset ingredients from one lerobot extras episode directory."""
    import numpy as np

    directory = Path(extras_dir)
    artifacts = artifacts or {}
    ep_meta = json.loads(_read_verified(directory / "ep_meta.json", artifacts.get("ep_meta.json")))
    model_xml = gzip.decompress(_read_verified(directory / "model.xml.gz", artifacts.get("model.xml.gz"))).decode(
        "utf-8"
    )
    states = np.load(io.BytesIO(_read_verified(directory / "states.npz", artifacts.get("states.npz"))))["states"]
    if len(states) < 1:
        raise ValueError(f"{directory}/states.npz has no simulator states")
    return {
        "ep_meta": ep_meta,
        "model_xml": model_xml,
        "state0": states[0],
    }


def reset_gym_env_to_episode(
    gym_env,
    extras_dir: str | Path,
    *,
    seed: int,
    artifacts: dict | None = None,
):
    """Reset a RoboCasa gym environment to one demo's exact initial state."""
    gym_wrapper = gym_env.unwrapped
    core = gym_wrapper.env
    state = load_episode_state(extras_dir, artifacts=artifacts)

    def set_episode_metadata() -> None:
        if hasattr(core, "set_ep_meta"):
            core.set_ep_meta(state["ep_meta"])
        elif hasattr(core, "set_attrs_from_ep_meta"):
            core.set_attrs_from_ep_meta(state["ep_meta"])
        else:
            raise AttributeError("RoboCasa core exposes neither set_ep_meta nor set_attrs_from_ep_meta")

    set_episode_metadata()
    # Reset through OrderEnforcing/TimeLimit so wrapper state is fresh. The explicit seed pins any
    # simulator RNG that is not subsequently overwritten by the recorded XML and flattened state.
    gym_env.reset(seed=int(seed))
    set_episode_metadata()
    xml = core.edit_model_xml(state["model_xml"])
    core.reset_from_xml_string(xml)
    core.sim.reset()
    core.sim.set_state_from_flattened(state["state0"])
    core.sim.forward()
    if hasattr(core, "update_state"):
        core.update_state()
    elif hasattr(core, "update_sites"):
        core.update_sites()

    raw = (
        core.viewer._get_observations(force_update=True)
        if getattr(core, "viewer_get_obs", False)
        else core._get_observations(force_update=True)
    )
    observation = gym_wrapper.get_observation(dict(raw))
    return observation, {"success": False}
