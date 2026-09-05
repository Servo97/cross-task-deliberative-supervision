#!/usr/bin/env python3
"""Environment construction, demo-pinned reset, and per-step state probing.

Shared by the rollout worker, the teacher-forcing worker and the renderer so all three see
byte-identical scenes. Imports of robocasa/robosuite are deferred: this module is imported
by pure-numpy stages too.
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np


# --------------------------------------------------------------------------------------
# env construction
# --------------------------------------------------------------------------------------
def make_env(bench: str, task: str, seed: int, *, camera_size: int = 256, enable_render: bool = True):
    """Build the gym-wrapped RoboCasa/ReMemBench env for one task.

    ``camera_size`` only sizes the three policy cameras. Extra views for video are taken
    straight off ``sim.render`` (see :func:`render_views`), because the ReMemBench wrapper
    silently overwrites any ``camera_names`` kwarg.
    """
    import gymnasium as gym
    import robocasa  # noqa: F401

    if bench == "remembench":
        import robocasa.wrappers.gym_wrapper  # noqa: F401  registers robocasa/<Task> ids

        return gym.make(
            f"robocasa/{task}",
            enable_render=enable_render,
            seed=int(seed),
            camera_widths=camera_size,
            camera_heights=camera_size,
        )
    return gym.make(
        f"robocasa/{task}",
        split="target",
        seed=int(seed),
        camera_widths=camera_size,
        camera_heights=camera_size,
    )


def localise_asset_paths(model_xml: str) -> str:
    """Repoint every robocasa asset reference in a recorded model XML at the local install.

    ReMemBench's demos were collected on a machine where robocasa lived under
    ``/Users/rutavms/research/gaze/robocasa``, and those absolute paths are baked into each
    demo's ``model_file``. ``Kitchen.edit_model_xml`` only remaps three prefixes —
    ``models/assets/fixtures``, ``models/assets/textures`` and
    ``models/assets/objects/objaverse`` — so anything under ``objects/aigen_objs`` (every
    aigen-registry fruit, i.e. most of the study's episodes) keeps the dead macOS path and
    MuJoCo fails to open the mesh. The sealed ReMemBench evals never hit this because their
    reset builds the scene from the local registry instead of loading the demo XML; a
    demo-pinned reset does load it. Rewriting the prefix is purely a path fix — the asset
    bytes it resolves to are the same ones the registry would have loaded.
    """
    import robocasa

    assets = os.path.join(os.path.dirname(robocasa.__file__), "models", "assets")
    return _ASSET_PATH_RE.sub(assets + "/", model_xml)


_ASSET_PATH_RE = __import__("re").compile(r"[^\"'\s>]*?/robocasa/models/assets/")


def _hide_debug_visuals(core) -> None:
    """ReMemBench only: zero the site alphas the demo XML bakes in (SHOW_SITES=True).

    The sealed ReMemBench eval reset calls this on every reset
    (``ReMemBench/robocasa/wrappers/gym_wrapper.py``). A demo-pinned reset reloads the
    demo's own ``model_file``, which carries the collection-time placement-region sites, so
    without this the policy would see large flat quads painted over the countertops — pixels
    it never saw at train or sealed-eval time.
    """
    try:
        from robocasa.wrappers.gym_wrapper import hide_debug_visuals
    except Exception:
        return
    hide_debug_visuals(core)


def reset_to_demo(gym_env, extras_dir: str, *, seed: int, bench: str):
    """Reset to a recorded demo's exact initial state. Mirrors ``heldout_reset``."""
    gym_wrapper = gym_env.unwrapped
    core = gym_wrapper.env

    with open(os.path.join(extras_dir, "ep_meta.json"), encoding="utf-8") as handle:
        ep_meta = json.load(handle)
    with gzip.open(os.path.join(extras_dir, "model.xml.gz"), "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    state0 = np.load(os.path.join(extras_dir, "states.npz"))["states"][0]

    def set_meta():
        if hasattr(core, "set_ep_meta"):
            core.set_ep_meta(ep_meta)
        else:
            core.set_attrs_from_ep_meta(ep_meta)

    set_meta()
    gym_env.reset(seed=int(seed))
    set_meta()
    core.reset_from_xml_string(core.edit_model_xml(localise_asset_paths(model_xml)))
    core.sim.reset()
    core.sim.set_state_from_flattened(state0)
    core.sim.forward()
    if hasattr(core, "update_state"):
        core.update_state()
    elif hasattr(core, "update_sites"):
        core.update_sites()
    if bench == "remembench":
        _hide_debug_visuals(core)

    raw = (
        core.viewer._get_observations(force_update=True)
        if getattr(core, "viewer_get_obs", False)
        else core._get_observations(force_update=True)
    )
    return gym_wrapper.get_observation(dict(raw))


def set_sim_state(core, state) -> None:
    core.sim.set_state_from_flattened(np.asarray(state))
    core.sim.forward()
    if hasattr(core, "update_state"):
        core.update_state()
    elif hasattr(core, "update_sites"):
        core.update_sites()


# --------------------------------------------------------------------------------------
# per-step probes (all read straight off the sim; no render, no observable rebuild)
# --------------------------------------------------------------------------------------
def eef_site_id(core) -> int:
    site = getattr(core.robots[0], "eef_site_id", None)
    if isinstance(site, dict):
        return int(site.get("right", next(iter(site.values()))))
    if site is not None:
        return int(site)
    for name in ("gripper0_right_grip_site", "gripper0_grip_site", "grip_site"):
        try:
            return int(core.sim.model.site_name2id(name))
        except Exception:
            continue
    raise RuntimeError("cannot locate the end-effector site")


def eef_pose(core, site_id: int):
    """(world position (3,), world quaternion (4,) in wxyz)."""
    pos = np.array(core.sim.data.site_xpos[site_id], dtype=np.float64)
    mat = np.array(core.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return pos, _mat_to_quat(mat)


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (mat[2, 1] - mat[1, 2]) / scale
        y = (mat[0, 2] - mat[2, 0]) / scale
        z = (mat[1, 0] - mat[0, 1]) / scale
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        scale = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
        w = (mat[2, 1] - mat[1, 2]) / scale
        x = 0.25 * scale
        y = (mat[0, 1] + mat[1, 0]) / scale
        z = (mat[0, 2] + mat[2, 0]) / scale
    elif mat[1, 1] > mat[2, 2]:
        scale = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
        w = (mat[0, 2] - mat[2, 0]) / scale
        x = (mat[0, 1] + mat[1, 0]) / scale
        y = 0.25 * scale
        z = (mat[1, 2] + mat[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
        w = (mat[1, 0] - mat[0, 1]) / scale
        x = (mat[0, 2] + mat[2, 0]) / scale
        y = (mat[1, 2] + mat[2, 1]) / scale
        z = 0.25 * scale
    return np.array([w, x, y, z], dtype=np.float64)


def object_names(core) -> list:
    return sorted(getattr(core, "obj_body_id", {}) or {})


def object_positions(core, names) -> np.ndarray:
    body_ids = core.obj_body_id
    return (
        np.stack([np.array(core.sim.data.body_xpos[body_ids[name]], dtype=np.float64) for name in names])
        if names
        else np.zeros((0, 3))
    )


def stove_probe(core):
    """(knob locations, knob joint angles, burner-site world positions) or None."""
    stove = getattr(core, "stove", None)
    if stove is None:
        return None
    try:
        knobs = stove.get_knobs_state(env=core)
    except Exception:
        return None
    locations = sorted(knobs)
    angles = np.array([float(knobs[loc]) for loc in locations], dtype=np.float64)
    burners = []
    for loc in locations:
        site = stove.burner_sites.get(loc)
        if site is None:
            burners.append(np.full(3, np.nan))
        else:
            burners.append(np.array(core.sim.data.get_site_xpos(site.get("name")), dtype=np.float64))
    return locations, angles, np.stack(burners)


def fixture_position(core, attr: str):
    fixture = getattr(core, attr, None)
    if fixture is None:
        return None
    pos = getattr(fixture, "pos", None)
    return None if pos is None else np.array(pos, dtype=np.float64)


def advance_task_hooks(core) -> None:
    """Run the per-step task bookkeeping that ``env.step`` would have run.

    A state-replay pass (expert probe, renderer) writes qpos straight into the sim and never
    calls ``env.step``, so the task's own state machine never advances: ``place_success``,
    ``stove_wait_timer``, ``turn_on/off_stove_success``, ``board_contact_timer`` all stay at
    their reset values. Those are exactly the signals the commitment gate and the
    prospective-timing fields read, so the replay has to drive the same hooks by hand. This
    is the identical code path ``_post_action`` invokes — no reimplementation of any success
    rule.
    """
    if hasattr(core, "_n_steps"):
        core._n_steps = int(getattr(core, "_n_steps", 0)) + 1
    for hook in ("_post_step_update", "_update_success", "update_state"):
        method = getattr(core, hook, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


TASK_STATE_KEYS = (
    "stove_wait_timer",
    "stove_wait_timer_threshold",
    "stove_wait_timer_max_threshold",
    "turn_on_stove_success",
    "turn_off_stove_success",
    "place_success",
    "final_success",
    "board_contact_timer",
)


def task_state(core) -> dict:
    out = {}
    for key in TASK_STATE_KEYS:
        if hasattr(core, key):
            value = getattr(core, key)
            out[key] = float(value) if not isinstance(value, bool) else float(bool(value))
    return out


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------
def render_views(core, cameras, width: int, height: int) -> list:
    """Render each named camera offscreen at (height, width). Vertically flipped to match
    the wrapper's ``process_img`` convention. Missing cameras yield a black tile."""
    frames = []
    for name in cameras:
        try:
            img = core.sim.render(width=width, height=height, camera_name=name)
            frames.append(np.ascontiguousarray(img[::-1, :, :]))
        except Exception:
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
    return frames


def available_cameras(core, wanted) -> list:
    present = set()
    for index in range(core.sim.model.ncam):
        present.add(core.sim.model.camera_id2name(index))
    return [name for name in wanted if name in present]
