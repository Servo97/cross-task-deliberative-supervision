#!/usr/bin/env python3
"""Rollout / teacher-forcing worker for the failure-mode study.

Two modes, one env build, one server connection:

``--mode rollout``
    Closed-loop. Demo-pinned reset, then the sealed control loop (replan every 8 steps,
    ``policy_noise_seed`` per chunk, ``wsm_*`` fields for workspace arms). Records the full
    MuJoCo state trajectory plus the derived geometry the metrics need. No pixels: video is
    a separate replay pass, so rollout throughput never pays for rendering.

``--mode teacher_force``
    Open-loop. Replays the EXPERT state trajectory step by step; every ``replan`` steps it
    rebuilds the observation at that expert state and asks the policy what it would do.
    The predicted chunk's first ``replan`` actions are scored against the expert's actions
    over the same window. Nothing the policy says is executed, so the query distribution is
    the expert's, identical for every checkpoint — the deviation is purely the policy's.
    Queries walk t = 0, 8, 16, ... in order, so a workspace arm's recurrent read sees the
    same causal history it would see on-policy (and ``wsm_t == 0`` opens the episode).

Both modes are content-validated resumable: a cell is skipped only when its ``result.json``
parses, declares ``complete``, and its companion ``.npz`` loads with the recorded length.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm_env  # noqa: E402
import fm_targets  # noqa: E402
from fm_common import (  # noqa: E402
    cell_name,
    policy_noise_seed,
    rollout_noise_base,
    write_json_atomic,
)

RESIZE = 224


def convert_action(action):
    action = np.asarray(action).copy()
    return {
        "action.end_effector_position": action[0:3],
        "action.end_effector_rotation": action[3:6],
        "action.gripper_close": action[6:7],
        "action.base_motion": action[7:11],
        "action.control_mode": action[11:12],
    }


def build_element(obs, task, t, env_id, noise_base, wsm_prompt, obs_image_size=RESIZE):
    """Byte-identical to the sealed runners' observation packing."""
    from openpi_client import image_tools

    def prep(key):
        frame = np.ascontiguousarray(obs[key])
        if frame.shape[0] == obs_image_size and frame.shape[1] == obs_image_size:
            return image_tools.convert_to_uint8(frame)
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, obs_image_size, obs_image_size))

    state = np.concatenate(
        (
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ),
        axis=0,
    )
    element = {
        "observation/image": prep("video.robot0_agentview_left"),
        "observation/wrist_image": prep("video.robot0_eye_in_hand"),
        "observation/right_image": prep("video.robot0_agentview_right"),
        "observation/state": state,
        "prompt": obs["annotation.human.task_description"],
        "wsm_t": int(t),
        "wsm_task": task,
        "wsm_env_id": env_id,
        "wsm_demo_episode": 0,
        "policy_noise_seed": policy_noise_seed(noise_base, int(t)),
    }
    if wsm_prompt is not None:
        element["wsm_prompt"] = wsm_prompt
    return element


# --------------------------------------------------------------------------------------
class Probe:
    """Per-step geometry recorder. Resolved once at reset, cheap to call every step."""

    def __init__(self, core, task, ep_meta):
        self.core = core
        self.task = task
        self.spec = fm_targets.TASK_SPECS[task]
        self.site = fm_env.eef_site_id(core)
        self.resolved = fm_targets.resolve_candidates(core, task, ep_meta)
        self.correct = self.resolved["correct"]
        self.distractor = self.resolved["distractor"]
        self.agent = self.spec["agent"]
        self.agent_object = self.agent.split(":", 1)[1] if self.agent.startswith("object:") else None
        if self.agent_object and self.agent_object not in (core.obj_body_id or {}):
            self.agent_object = None  # fall back to the gripper
        self.stove_locations = self.resolved["stove_locations"]
        self.correct_knob = self.resolved["correct_knob"]
        self.tracked = [
            name
            for name in (
                [self.spec.get("manipulated")]
                + [c["name"] for c in self.spec.get("correct", []) if c["kind"] == "object"]
                + [c["name"] for c in self.spec.get("distractor", []) if c["kind"] == "object"]
            )
            if name and name in (core.obj_body_id or {})
        ]
        self.tracked = sorted(set(self.tracked))
        self.rows = collections.defaultdict(list)

    def record(self):
        core = self.core
        ee_pos, ee_quat = fm_env.eef_pose(core, self.site)
        self.rows["ee_pos"].append(ee_pos)
        self.rows["ee_quat"].append(ee_quat)
        self.rows["state"].append(core.sim.get_state().flatten())

        objects = fm_env.object_positions(core, self.tracked)
        self.rows["obj_pos"].append(objects if len(self.tracked) else np.zeros((0, 3)))

        if self.agent_object is not None:
            agent = fm_env.object_positions(core, [self.agent_object])[0]
        else:
            agent = ee_pos
        self.rows["agent_pos"].append(agent)

        correct = fm_targets.dynamic_positions(core, self.correct)
        wrong = fm_targets.dynamic_positions(core, self.distractor)
        self.rows["d_correct"].append(
            float(np.min(np.linalg.norm(correct - agent, axis=1))) if len(correct) else np.inf
        )
        self.rows["d_distractor"].append(
            float(np.min(np.linalg.norm(wrong - agent, axis=1))) if len(wrong) else np.inf
        )
        self.rows["d_correct_ee"].append(
            float(np.min(np.linalg.norm(correct - ee_pos, axis=1))) if len(correct) else np.inf
        )
        self.rows["d_distractor_ee"].append(
            float(np.min(np.linalg.norm(wrong - ee_pos, axis=1))) if len(wrong) else np.inf
        )

        stove = fm_env.stove_probe(core)
        self.rows["knobs"].append(stove[1] if stove else np.zeros(0))
        state = fm_env.task_state(core)
        self.rows["task_state"].append(state)

    def finish(self):
        out = {
            "ee_pos": np.asarray(self.rows["ee_pos"], dtype=np.float64),
            "ee_quat": np.asarray(self.rows["ee_quat"], dtype=np.float64),
            "agent_pos": np.asarray(self.rows["agent_pos"], dtype=np.float64),
            "states": np.asarray(self.rows["state"], dtype=np.float64),
            "d_correct": np.asarray(self.rows["d_correct"], dtype=np.float64),
            "d_distractor": np.asarray(self.rows["d_distractor"], dtype=np.float64),
            "d_correct_ee": np.asarray(self.rows["d_correct_ee"], dtype=np.float64),
            "d_distractor_ee": np.asarray(self.rows["d_distractor_ee"], dtype=np.float64),
        }
        if self.rows["obj_pos"] and len(self.tracked):
            out["obj_pos"] = np.asarray(self.rows["obj_pos"], dtype=np.float64)
        if self.rows["knobs"] and self.rows["knobs"][0].size:
            out["knobs"] = np.asarray(self.rows["knobs"], dtype=np.float64)
        keys = sorted({key for row in self.rows["task_state"] for key in row})
        for key in keys:
            out[f"ts_{key}"] = np.asarray([row.get(key, np.nan) for row in self.rows["task_state"]], dtype=np.float64)
        return out

    def meta(self):
        return {
            "agent": self.agent if self.agent_object or self.agent == "ee" else "ee(fallback)",
            "tracked_objects": self.tracked,
            "correct_candidates": [label for label, _pos, _dyn in self.correct],
            "distractor_candidates": [label for label, _pos, _dyn in self.distractor],
            "correct_knob": self.correct_knob,
            "stove_locations": self.stove_locations,
            "phase_gate": self.spec.get("phase_gate"),
            "derivation": self.spec["docs"],
        }


# --------------------------------------------------------------------------------------
def cell_complete(out_dir: str, mode: str) -> bool:
    result = os.path.join(out_dir, "result.json")
    array = os.path.join(out_dir, "tf.npz" if mode == "teacher_force" else "traj.npz")
    if not (os.path.exists(result) and os.path.exists(array)):
        return False
    try:
        with open(result) as handle:
            payload = json.load(handle)
        if not payload.get(f"{mode}__complete"):
            return False
        with np.load(array) as data:
            key = "tf_mse" if mode == "teacher_force" else "ee_pos"
            if key not in data:
                return False
            if int(len(data[key])) != int(payload.get(f"{mode}__n_recorded", -1)):
                return False
    except Exception:
        return False
    return True


def run_rollout(env, core, client, episode, args, env_id, wsm_prompt):
    obs = fm_env.reset_to_demo(env, episode["reset"]["extras_dir"], seed=int(episode["seed"]), bench=args.bench)
    with open(os.path.join(episode["reset"]["extras_dir"], "ep_meta.json")) as handle:
        ep_meta = json.load(handle)
    probe = Probe(core, episode["task"], ep_meta)

    horizon = int(episode["horizon"])
    noise_base = rollout_noise_base(int(episode["seed"]), args.rollout_idx)
    plan = collections.deque()
    success = False
    failed_task = False
    n_infer = 0
    started = time.time()
    t = 0
    probe.record()
    while t < horizon:
        if not plan:
            element = build_element(obs, episode["task"], t, env_id, noise_base, wsm_prompt, args.obs_image_size)
            chunk = client.infer(element)["actions"]
            n_infer += 1
            if len(chunk) < args.replan_steps:
                raise AssertionError(f"chunk {len(chunk)} < replan {args.replan_steps}")
            actions = np.asarray(chunk[: args.replan_steps])
            if not np.all(np.isfinite(actions)):
                raise RuntimeError(f"non-finite action at t={t}")
            plan.extend(actions)
        obs, _reward, terminated, truncated, info = env.step(convert_action(plan.popleft()))
        success = bool(info["success"])
        failed_task = bool(info.get("failed_task", False))
        t += 1
        probe.record()
        if success or failed_task or terminated or truncated:
            break

    arrays = probe.finish()
    result = {
        "complete": True,
        "mode": "rollout",
        "success": bool(success and not failed_task),
        "raw_success": bool(success),
        "failed_task": bool(failed_task),
        "episode_length": int(t),
        "n_recorded": int(len(arrays["ee_pos"])),
        "n_infer": int(n_infer),
        "rollout_seconds": round(time.time() - started, 1),
        "target_meta": probe.meta(),
    }
    return result, arrays


def run_teacher_force(env, core, client, episode, args, env_id, wsm_prompt):
    """Query the policy along the expert trajectory; score against the expert's actions."""
    fm_env.reset_to_demo(env, episode["reset"]["extras_dir"], seed=int(episode["seed"]), bench=args.bench)
    gym_wrapper = env.unwrapped
    states = np.load(episode["expert"]["states"])["states"]
    expert_actions = np.load(episode["expert"]["actions"])
    length = int(min(len(states), len(expert_actions)))
    stride = int(args.replan_steps)
    noise_base = rollout_noise_base(int(episode["seed"]), args.rollout_idx)

    action_dim = int(expert_actions.shape[1])
    per_step_mse = np.full(length, np.nan, dtype=np.float64)
    per_step_se = np.full((length, action_dim), np.nan, dtype=np.float64)
    query_steps = []
    started = time.time()
    for t in range(0, length, stride):
        fm_env.set_sim_state(core, states[t])
        raw = (
            core.viewer._get_observations(force_update=True)
            if getattr(core, "viewer_get_obs", False)
            else core._get_observations(force_update=True)
        )
        obs = gym_wrapper.get_observation(dict(raw))
        element = build_element(obs, episode["task"], t, env_id, noise_base, wsm_prompt, args.obs_image_size)
        chunk = np.asarray(client.infer(element)["actions"])
        window = min(stride, length - t)
        predicted = chunk[:window]
        target = expert_actions[t : t + window]
        squared = (predicted - target) ** 2
        per_step_se[t : t + window] = squared
        per_step_mse[t : t + window] = np.mean(squared, axis=1)
        query_steps.append(t)

    valid = np.isfinite(per_step_mse)
    result = {
        "complete": True,
        "mode": "teacher_force",
        "expert_length": length,
        "n_recorded": int(length),
        "n_infer": int(len(query_steps)),
        "stride": stride,
        "tf_seconds": round(time.time() - started, 1),
    }
    arrays = {
        "tf_mse": per_step_mse,
        "tf_se": per_step_se,
        "query_steps": np.asarray(query_steps, dtype=np.int64),
        "valid": valid,
    }
    return result, arrays


def run_expert_probe(env, core, _client, episode, args, _env_id, _wsm_prompt):
    """Replay the expert trajectory and record the SAME geometry a rollout records.

    Server-free. Runs once per (task, reset) and is reused by every checkpoint, so the
    expert reference curves are guaranteed identical across arms. Also yields the expert's
    own target-commitment label, which is the sanity check on the whole classifier: a
    demonstration must come out ``approached_correct``.
    """
    fm_env.reset_to_demo(env, episode["reset"]["extras_dir"], seed=int(episode["seed"]), bench=args.bench)
    with open(os.path.join(episode["reset"]["extras_dir"], "ep_meta.json")) as handle:
        ep_meta = json.load(handle)
    probe = Probe(core, episode["task"], ep_meta)
    states = np.load(episode["expert"]["states"])["states"]
    started = time.time()
    for state in states:
        fm_env.set_sim_state(core, state)
        fm_env.advance_task_hooks(core)
        probe.record()
    arrays = probe.finish()
    result = {
        "complete": True,
        "mode": "expert_probe",
        "n_recorded": int(len(arrays["ee_pos"])),
        "expert_seconds": round(time.time() - started, 1),
        "target_meta": probe.meta(),
    }
    return result, arrays


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bench", choices=["remembench", "robocasa"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--ckpt-label", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--mode", choices=["rollout", "teacher_force", "expert_probe"], default="rollout")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--obs-image-size", type=int, default=RESIZE)
    parser.add_argument("--task-prompt-manifest", default=None)
    parser.add_argument("--limit", type=int, default=None, help="canary: cap resets")
    parser.add_argument(
        "--rollout-idx",
        type=int,
        default=0,
        help=(
            "diffusion-noise draw index for this pass. 0 reuses the episode seed and is "
            "byte-identical to the sealed protocol's rollout 0; 1..k derive independent "
            "streams via rollout_noise_base. The RESET is unchanged across draws, so "
            "reset-pairing across arms and draws stays intact."
        ),
    )
    parser.add_argument("--env-id", default=None)
    args = parser.parse_args()

    with open(args.manifest) as handle:
        manifest = json.load(handle)
    episodes = [e for e in manifest["episodes"] if e["task"] == args.task]
    episodes.sort(key=lambda e: (e["split"] != "heldout", int(e["episode_index"])))
    if args.limit:
        episodes = episodes[: args.limit]
    episodes = episodes[args.shard_idx :: args.num_shards]
    if not episodes:
        print("[fm] nothing to do", flush=True)
        return 0

    wsm_prompt = None
    if args.task_prompt_manifest:
        with open(args.task_prompt_manifest) as handle:
            table = json.load(handle)
        wsm_prompt = {r["task"]: r["prompt"] for r in table["tasks"]}[args.task]

    import robocasa  # noqa: F401

    client = None
    if args.mode != "expert_probe":
        from openpi_client import websocket_client_policy as wcp

        if not args.port:
            raise SystemExit("--port is required unless --mode expert_probe")
        client = wcp.WebsocketClientPolicy(args.host, args.port)
    env = fm_env.make_env(args.bench, args.task, seed=int(manifest["base_seed"]))
    core = env.unwrapped.env
    env_id = args.env_id or f"fm-{args.bench}-w{args.shard_idx}of{args.num_shards}"
    suffix = {"rollout": "traj.npz", "teacher_force": "tf.npz", "expert_probe": "traj.npz"}[args.mode]
    failures = 0

    try:
        for episode in episodes:
            out_dir = os.path.join(
                args.out_root,
                "raw",
                args.bench,
                args.task,
                args.ckpt_label,
                cell_name(episode["reset_id"], args.rollout_idx),
            )
            if cell_complete(out_dir, args.mode):
                print(f"[fm] skip {episode['reset_id']} ({args.mode})", flush=True)
                continue
            os.makedirs(out_dir, exist_ok=True)
            try:
                runner = {
                    "rollout": run_rollout,
                    "teacher_force": run_teacher_force,
                    "expert_probe": run_expert_probe,
                }[args.mode]
                result, arrays = runner(env, core, client, episode, args, env_id, wsm_prompt)
            except Exception:
                failures += 1
                traceback.print_exc()
                write_json_atomic(
                    os.path.join(out_dir, f"{args.mode}_FAILED.json"),
                    {"reset_id": episode["reset_id"], "traceback": traceback.format_exc()},
                )
                continue
            result.update(
                {
                    "bench": args.bench,
                    "task": args.task,
                    "ckpt_label": args.ckpt_label,
                    "reset_id": episode["reset_id"],
                    "rollout_index": int(args.rollout_idx),
                    "split": episode["split"],
                    "episode_index": episode["episode_index"],
                    "seed": episode["seed"],
                    "horizon": episode["horizon"],
                    "expert_length": episode["expert"]["expert_length"],
                    "lang": episode["expert"].get("lang"),
                    "replan_steps": args.replan_steps,
                }
            )
            np.savez_compressed(os.path.join(out_dir, suffix), **arrays)
            existing = {}
            result_path = os.path.join(out_dir, "result.json")
            if os.path.exists(result_path):
                try:
                    with open(result_path) as handle:
                        existing = json.load(handle)
                except Exception:
                    existing = {}
            # rollout and teacher_force write disjoint namespaces into one result.json
            merged = dict(existing)
            merged.update({f"{args.mode}__{k}": v for k, v in result.items()})
            merged.update(
                {
                    k: result[k]
                    for k in (
                        "bench",
                        "task",
                        "ckpt_label",
                        "reset_id",
                        "rollout_index",
                        "split",
                        "episode_index",
                        "seed",
                        "horizon",
                        "expert_length",
                        "lang",
                    )
                }
            )
            merged["complete"] = True
            merged["n_recorded"] = result["n_recorded"]
            write_json_atomic(result_path, merged)
            print(
                f"[fm] {args.mode} {args.task}/{args.ckpt_label}/"
                f"{cell_name(episode['reset_id'], args.rollout_idx)} "
                f"{'success' if result.get('success') else ''} "
                f"n={result['n_recorded']} ({result.get('rollout_seconds', result.get('tf_seconds'))}s)",
                flush=True,
            )
    finally:
        env.close()

    print(f"[fm] worker done, {failures} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
