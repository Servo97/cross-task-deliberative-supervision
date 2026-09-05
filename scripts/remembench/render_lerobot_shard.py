#!/usr/bin/env python3
"""Re-render ReMemBench train demos at 3x256px and emit LeRobot v2.1 episode payloads.

Why a re-render at all: the published ReMemBench hdf5s carry 128px ``robot0_agentview_center``
+ ``robot0_eye_in_hand`` observations, while our pi0.5 policy trains on 256px
``robot0_agentview_left`` / ``robot0_agentview_right`` / ``robot0_eye_in_hand``. The sim states
are recorded, so the frames are regenerated deterministically by ``reset_to({"states": s_t})``
after pinning the episode with its ``model_file`` + ``ep_meta`` -- the same path
``robocasa/scripts/dataset_states_to_obs.py`` uses. No physics is re-simulated, so actions and
frame counts carry over verbatim.

Output format is LeRobot v2.1 in the GR00T flavour, byte-compatible with the RoboCasa
``datasets/v1.0/target/**/lerobot`` trees the training loader already consumes (see
gr00t ``LeRobotSingleDataset``): float64 ``observation.state`` [16] and ``action`` [12] in
``modality.json`` order, h264/yuv420p video per camera, one parquet per episode.

This script writes only the per-episode payloads (parquet + 3 mp4 + a small sidecar json)
into ``<out>/<task>/_episodes/<episode_index>/``; ``finalize_lerobot_tasks.py`` assembles the
``meta/`` tree afterwards. Splitting it this way keeps shards embarrassingly parallel while
still producing one coherent dataset directory per task.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# LeRobot column order for the 16-d state / 12-d action, taken verbatim from
# robocasa/utils/lerobot_utils.py so the re-render matches the RoboCasa conversion exactly.
STATE_SEGMENTS = (
    ("base_position", "robot0_base_pos", 0, 3),
    ("base_rotation", "robot0_base_quat", 3, 7),
    ("end_effector_position_relative", "robot0_base_to_eef_pos", 7, 10),
    ("end_effector_rotation_relative", "robot0_base_to_eef_quat", 10, 14),
    ("gripper_qpos", "robot0_gripper_qpos", 14, 16),
)
# hdf5 action layout -> lerobot action layout
ACTION_SEGMENTS = (
    ("base_motion", 7, 11, 0, 4),
    ("control_mode", 11, 12, 4, 5),
    ("end_effector_position", 0, 3, 5, 8),
    ("end_effector_rotation", 3, 6, 8, 11),
    ("gripper_close", 6, 7, 11, 12),
)
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
FPS = 20
DEMO_FILENAME = "demo_im128_notp.hdf5"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def open_video_writer(path: Path, size: int):
    """h264/yuv420p/crf23 -- the encoder settings robocasa's LerobotDatasetWrapper pins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size}x{size}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-an",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def patch_missing_objects_joint_state() -> None:
    """Tolerate the ``objects-joint-state`` key the robomimic wrapper assumes but no env emits.

    ``EnvRobocasa.get_observation`` does an unconditional ``di["objects-joint-state"]`` whenever
    ``object-state`` is present, but no observable in this ReMemBench checkout registers that
    modality (the published demos were recorded against an older fork that did). Every Mem* env
    therefore raises KeyError on reset. The key is not part of anything we write -- the LeRobot
    state vector is the five robot0_* keys -- so filling an empty array is inert and keeps us off
    a fork of the wrapper.
    """
    from robocasa.utils.robomimic import robomimic_env_wrapper as wrapper

    if getattr(wrapper.EnvRobocasa, "_wsm_ojs_patched", False):
        return
    original = wrapper.EnvRobocasa.get_observation

    def get_observation(self, di=None):
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        if "object-state" in di and "objects-joint-state" not in di:
            di = dict(di)
            di["objects-joint-state"] = np.zeros(0, dtype=np.float64)
        return original(self, di)

    wrapper.EnvRobocasa.get_observation = get_observation
    wrapper.EnvRobocasa._wsm_ojs_patched = True


def make_env(env_meta: dict, camera_size: int):
    import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils

    patch_missing_objects_joint_state()
    meta = json.loads(json.dumps(env_meta))  # deep copy
    # Segmentation rendering is expensive and unused downstream; the published demos enabled it.
    meta["env_kwargs"]["camera_segmentations"] = None
    return EnvUtils.create_env_for_data_processing(
        env_meta=meta,
        camera_names=list(CAMERAS),
        camera_height=camera_size,
        camera_width=camera_size,
        reward_shaping=False,
    )


#: Teleop mocap markers stock robosuite 1.5.2 adds to every Arena (models/arenas/arena.py).
EEF_TARGET_GEOMS = (
    "left_eef_target_box",
    "left_eef_target_sphere",
    "right_eef_target_box",
    "right_eef_target_sphere",
)


def hide_debug_visuals(env) -> None:
    """Restore ``SHOW_SITES = False`` rendering after a recorded model XML is loaded.

    ReMemBench's ``collect_demos.py`` sets ``macros.SHOW_SITES = True``, so every demo's stored
    ``model_file`` has the placement-region / fixture-reference sites baked in with a visible
    alpha. ``reset_to`` loads that XML verbatim, which paints large flat blue and magenta quads
    over the countertops -- absent from the published 128px demo frames and absent at eval time,
    where envs are constructed fresh with ``SHOW_SITES = False``. Zeroing every site alpha
    reproduces the ``SHOW_SITES = False`` appearance exactly (sites are non-physical markers
    only, so nothing legitimate is hidden) and it is the same correction
    ``robomimic_env_wrapper.reset_to`` applies -- except that its guard is ``if not self._is_v1``,
    which never fires on robosuite 1.x.

    The stock-1.5.2 teleop mocap markers (``left/right_eef_target_*``, geom group 2) are cleared
    at the same time: the ``abs_robot`` fork the demos were recorded on had no such geoms, and
    their pose is restored from the mocap portion of the MuJoCo state.

    ``site_rgba`` / ``geom_rgba`` are model data, not restored by ``reset_to({"states": ...})``,
    so one call per model load is sufficient.
    """
    import mujoco

    model = env.env.sim.model._model
    model.site_rgba[:, 3] = 0.0
    for name in EEF_TARGET_GEOMS:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            model.geom_rgba[geom_id, 3] = 0.0


def render_episode(env, source, demo_key, out_dir: Path, camera_size: int) -> dict:
    group = source["data"][demo_key]
    states = group["states"][()]
    actions = group["actions"][()]
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"{demo_key}: states {states.shape} vs actions {actions.shape}")
    length = int(group.attrs["num_samples"])
    if length != states.shape[0]:
        raise ValueError(f"{demo_key}: num_samples {length} != states {states.shape[0]}")

    initial_state = {
        "states": states[0],
        "model": group.attrs["model_file"],
        "ep_meta": group.attrs["ep_meta"],
    }
    # No bare env.reset() first (unlike dataset_states_to_obs): that clears ep_meta and resets into
    # a randomly sampled scene, and several Mem* tasks then expose "object-state" without
    # "objects-joint-state", which the robomimic wrapper dereferences unconditionally. reset_to
    # already performs a pinned reset (set_ep_meta -> reset(unset_ep_meta=False) -> load xml).
    env.reset_to(initial_state)
    hide_debug_visuals(env)  # after the xml load: reset_to rebuilds the model
    ep_meta = env.env.get_ep_meta()

    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {camera: open_video_writer(out_dir / f"{camera}.mp4", camera_size) for camera in CAMERAS}
    state_rows = np.zeros((length, 16), dtype=np.float64)
    rewards = np.zeros((length,), dtype=np.float32)
    dones = np.zeros((length,), dtype=bool)
    try:
        for t in range(length):
            obs = env.reset_to({"states": states[t]})
            for camera in CAMERAS:
                frame = obs[f"{camera}_image"]
                if frame.shape != (camera_size, camera_size, 3):
                    raise ValueError(f"{demo_key}: {camera} frame {frame.shape}")
                writers[camera].stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            for _name, obs_key, start, end in STATE_SEGMENTS:
                state_rows[t, start:end] = np.asarray(obs[obs_key], dtype=np.float64)
            rewards[t] = float(env.get_reward())
            dones[t] = bool(env.is_success()["task"])
        # done flag: mirror robocasa done_mode=2 (success OR final transition)
        dones[length - 1] = True
    finally:
        for writer in writers.values():
            writer.stdin.close()
        for camera, writer in writers.items():
            if writer.wait() != 0:
                raise RuntimeError(f"{demo_key}: ffmpeg failed for {camera}")

    action_rows = np.zeros((length, 12), dtype=np.float64)
    for _name, h_start, h_end, l_start, l_end in ACTION_SEGMENTS:
        action_rows[:, l_start:l_end] = actions[:, h_start:h_end]

    return {
        "length": length,
        "state": state_rows,
        "action": action_rows,
        "rewards": rewards,
        "dones": dones,
        "lang": ep_meta["lang"],
    }


def write_parquet(payload: dict, episode_index: int, task_index: int, out_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    length = payload["length"]
    table = pa.table(
        {
            "annotation.human.task_description": pa.array(np.full(length, task_index, dtype=np.int64)),
            "annotation.human.task_name": pa.array(np.full(length, payload["task_name_index"], dtype=np.int64)),
            "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(payload["state"].reshape(-1)), 16),
            "action": pa.FixedSizeListArray.from_arrays(pa.array(payload["action"].reshape(-1)), 12),
            "next.reward": pa.array(payload["rewards"], type=pa.float32()),
            "next.done": pa.array(payload["dones"], type=pa.bool_()),
            "timestamp": pa.array((np.arange(length, dtype=np.float32) / FPS).astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(np.arange(length, dtype=np.int64)),
            "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(length, dtype=np.int64)),
            "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", required=True, help="i/N over the flat episode list")
    parser.add_argument("--camera-size", type=int, default=256)
    args = parser.parse_args()

    import h5py

    index, count = (int(part) for part in args.shard.split("/"))
    worklist = json.loads(Path(args.worklist).read_text())
    data_root = Path(args.data_root)
    out_root = Path(args.out)

    # Flat, deterministic work order; group by (task, session) so one env serves many demos.
    jobs = []
    for task in worklist["tasks"]:
        for episode in task["episodes"]:
            jobs.append((task["task"], episode))
    mine = [job for position, job in enumerate(jobs) if position % count == index]
    log(f"shard {index}/{count}: {len(mine)} of {len(jobs)} episodes")

    failures = []
    env = None
    source = None
    source_key = None
    done = 0
    started = time.time()
    for task, episode in mine:
        episode_dir = out_root / task / "_episodes" / f"{episode['episode_index']:06d}"
        if (episode_dir / "episode.json").is_file():
            done += 1
            continue
        try:
            key = (task, episode["session"])
            if source_key != key:
                if source is not None:
                    source.close()
                source = h5py.File(data_root / task / episode["session"] / DEMO_FILENAME, "r")
                source_key = key
                env_meta = json.loads(source["data"].attrs["env_args"])
                if env is not None:
                    env.env.close()
                    env = None
                env = make_env(env_meta, args.camera_size)
            payload = render_episode(env, source, episode["demo_key"], episode_dir, args.camera_size)
            if payload["length"] != episode["length"]:
                raise ValueError(
                    f"frame-count parity failed: rendered {payload['length']} vs worklist {episode['length']}"
                )
            payload["task_name_index"] = -1  # filled at finalize once tasks.jsonl is known
            np.savez(
                episode_dir / "arrays.npz",
                state=payload["state"],
                action=payload["action"],
                rewards=payload["rewards"],
                dones=payload["dones"],
            )
            (episode_dir / "episode.json").write_text(
                json.dumps(
                    {
                        "task": task,
                        "episode_index": episode["episode_index"],
                        "session": episode["session"],
                        "demo_key": episode["demo_key"],
                        "length": payload["length"],
                        "lang": payload["lang"],
                        "worklist_lang": episode["lang"],
                    },
                    indent=1,
                )
            )
            done += 1
            if done % 5 == 0:
                rate = (time.time() - started) / max(done, 1)
                log(f"shard {index}: {done}/{len(mine)} done ({rate:.1f}s/demo)")
        except Exception as error:  # noqa: BLE001 - failures are reported, never silently dropped
            failures.append(
                {
                    "task": task,
                    "episode_index": episode["episode_index"],
                    "demo_key": episode["demo_key"],
                    "session": episode["session"],
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-2000:],
                }
            )
            log(f"FAILED {task}/{episode['demo_key']}: {error}")
            if env is not None:
                try:
                    env.env.close()
                except Exception:  # noqa: BLE001
                    pass
                env = None
            source_key = None
            if source is not None:
                source.close()
                source = None

    report = out_root / "_shards" / f"shard_{index:03d}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "shard": index,
                "shards": count,
                "assigned": len(mine),
                "completed": done,
                "failures": failures,
                "seconds": round(time.time() - started, 1),
            },
            indent=1,
        )
    )
    log(f"shard {index} finished: {done}/{len(mine)} ok, {len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
