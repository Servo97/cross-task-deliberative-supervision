#!/usr/bin/env python3
"""RoboCerebra benchmark evaluation driven by an openpi policy server.

The published harness (``evaluation/eval_openvla.py``) loads an OpenVLA-OFT checkpoint
in-process. Ours cannot: the sim needs robosuite 1.4.0 / mujoco 2.3.2, while the policy needs the
openpi venv with robosuite 1.5.2. So this script keeps only the **environment side** and talks to
an openpi ``websocket_policy_server`` over a socket -- the same split ``examples/libero`` already
uses, which conveniently also makes the version conflict a non-issue rather than a blocker.

Run under the LIBERO sim venv:

    LIBERO_CONFIG_PATH=$WSM/robocerebra/libero_config PYTHONPATH=$WSM/robocerebra/code/LIBERO \
    MUJOCO_GL=egl $WSM/robocerebra/venv_sim/bin/python eval_robocerebra_openpi.py \
        --bench-root $WSM/robocerebra/RoboCerebraBench --modes Ideal --trials 10

and serve the policy separately from the openpi venv:

    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config pi05_libero --policy.dir <checkpoint>

Protocol reproduced from the reference harness, verbatim where it matters:

* ``max_steps = switch_steps * num_subtasks`` and ``step_idx = (t // switch_steps) % num_subtasks``
  -- the subtask instruction advances on a **fixed 150-step timer**, not on success. This is the
  "anchor-aligned subtask transition": the benchmark simulates a perfect high-level planner and
  measures only the low-level policy.
* ``resume=True`` (every mode uses it) resets the sim to the demo's ground-truth state at each
  subtask boundary, so subtask k is scored from the state subtask k actually starts in rather
  than from wherever the policy drifted to.
* ``num_steps_wait=15`` dummy actions first, because the sim drops objects on reset.
* Per-mode switches (``eval_openvla.py:573-591``): ``dynamic`` teleports a distractor +-0.15 m in
  y mid-segment (Random_Disturbance, Mix); ``dynamic_shift_description`` offsets the instruction
  index by one so the text disagrees with the state (Observation_Mismatching, Mix).
* Scoring: ``env._check_success(goal)`` returns ``(per_object_dict, total_completed, all_done)``.
  We report episode success (``all_done``) and subtask completion rate, matching the paper.

Ideal / Observation_Mismatching / Random_Disturbance all read task files **and** init states from
the ``Ideal`` directory -- they are three protocols over one task set, not three task sets.

``--wsm`` (opt-in, strictly additive) additionally stamps every policy request with the episode/step
identity that ``serve_pi05_libero_wsm.py`` needs to build omega online -- ``wsm_env_id``, ``wsm_t``,
``wsm_episode_len``, ``wsm_episode_id`` and ``wsm_repin``. Without the flag not one byte of the
request changes, so the base arms (A0/A3, which read no omega) keep running against the plain
``serve_pi05_libero.py`` exactly as before.

``wsm_t`` is the post-wait env step (the 15 dummy actions are not part of the episode) and
``wsm_episode_len`` is ``max_steps = switch_steps * num_subtasks``; the server derives the omega grid
stride from that as ``max(1, round(len / 64))``, reproducing the training tap's 64-frames-per-episode
cadence. ``wsm_repin=True`` marks the first request after each subtask-boundary resume re-pin, which
is where the sim teleports to the demo's ground-truth state and the buffered omega stops describing
the workspace the robot is in.

K ENV RUNNERS PER GPU (``--num-shards`` / ``--shard``)
-----------------------------------------------------
One process, one env, one policy request in flight -- so a pi-class server spends most of its life
idle and the eval spends most of its life waiting on a serial tap. K runner PROCESSES per GPU
against ONE gather-batching server fixes that (``run_eval_sharded.sh``). Two invariants make it a
performance change rather than a different experiment:

1. **Whole (case, trial) units.** Shard s owns trials ``s, s+K, s+2K, ...`` of every case. A trial is
   never split, so the 150-step anchor timer, the ``resume`` re-pins, the 15 warm-up no-ops and the
   replan-5 cadence are byte-for-byte the protocol they always were -- this file's episode loop is
   untouched. Cases are NOT sharded because each shard would then build the same env anyway and the
   long pole is trials, not env construction.
2. **Nothing may depend on who ran it.** Two things did. ``Policy.infer`` splits a mutable server-side
   rng per call, and this file drew distractor teleports from one global ``random.Random(seed)``
   stream. Both are functions of arrival/iteration order, which sharding changes.
   ``--deterministic-seeding`` replaces both with coordinates: every request carries
   ``policy_noise_seed = blake2b(mode|case|trial|step)`` (openpi's documented "one explicit JAX noise
   row, independent of mutable policy request order"), and each episode gets its own
   ``random.Random(blake2b(rng|seed|mode|case|trial))``. Outcomes then depend on the protocol
   coordinate and nothing else -- not K, not the shard, not the gather window, not the arm. See
   ``eval_seeding.py``.

``--deterministic-seeding`` is opt-in and ``--num-shards > 1`` refuses to run without it. Left off,
every request is byte-identical to what this harness sent before sharding existed, so already-scored
cells stay reproducible on the old path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from authors_scorer_port import AuthorsScorer  # noqa: E402
from eval_seeding import episode_rng_seed, policy_noise_seed  # noqa: E402

SWITCH_STEPS = 150
NUM_STEPS_WAIT = 15
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
RESIZE = 224
# Task types whose case directories and init files both live under Ideal/.
IDEAL_ALIASES = {"Ideal", "Observation_Mismatching", "Random_Disturbance"}
MODE_FLAGS = {
    "Ideal": (False, False),
    "Observation_Mismatching": (False, True),
    "Random_Disturbance": (True, False),
    "Mix": (True, True),
    "Memory_Execution": (False, False),
    "Memory_Exploration": (False, False),
}
MOVABLE_OBJECTS = [
    "alphabet_soup",
    "bbq_sauce",
    "butter",
    "chocolate_pudding",
    "cookies",
    "cream_cheese",
    "ketchup",
    "macaroni_and_cheese",
    "milk",
    "orange_juice",
    "popcorn",
    "salad_dressing",
    "new_salad_dressing",
    "tomato_sauce",
    "white_bowl",
    "akita_black_bowl",
    "plate",
    "glazed_rim_porcelain_ramekin",
    "red_coffee_mug",
    "porcelain_mug",
    "white_yellow_mug",
    "chefmate_8_frypan",
    "bowl_drainer",
    "moka_pot",
    "window",
    "faucet",
    "black_book",
    "yellow_book",
    "desk_caddy",
    "wine_bottle",
]
BRACKET_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_steps(path: Path) -> tuple[list[str], list[int]]:
    """Return ``(step_texts, start_frames)`` -- start frames index the raw demo states."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts, starts, i = [], [], 0
    while i < len(lines):
        if lines[i].startswith("Step"):
            texts.append(lines[i].split(":", 1)[1].strip())
            if i + 1 < len(lines) and (match := BRACKET_RE.match(lines[i + 1])):
                starts.append(int(match.group(1)))
                i += 1
        i += 1
    return texts, starts


def load_goal(path: Path) -> tuple[dict, dict]:
    """goal.json -> ``({object: [[verb, subj(, obj)], ...]}, {object: [task_step, ...]})``.

    The first element is the shape ``_check_success`` wants. The second is the per-goal
    ``task_step`` index, which protocol v3 needs to reproduce the authors'
    ``create_step_based_resume_handler`` / ``simulate_resume_completion`` pair
    (``code/evaluation/resume.py``). v2 discarded it, which is why v2 could not attribute
    re-pin credit to resume and leaked it into the agent's count.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    goal, steps = {}, {}
    for object_id, relations in raw.items():
        processed, processed_steps = [], []
        for item in relations:
            if isinstance(item, dict) and "state_pair" in item:
                triple = item["state_pair"]
                if len(triple) in (2, 3):
                    processed.append([triple[0].lower(), *triple[1:]])
                    processed_steps.append(int(item.get("task_step", len(processed) - 1)))
            elif isinstance(item, list) and len(item) in (2, 3):
                processed.append([item[0].lower(), *item[1:]])
                processed_steps.append(len(processed) - 1)
        goal[object_id] = processed
        steps[object_id] = processed_steps
    return goal, steps


def build_resume_handler(goal: dict, goal_steps: dict) -> dict:
    """Port of ``create_step_based_resume_handler`` (code/evaluation/resume.py:17).

    Maps a subtask index to every goal that should already be satisfied BEFORE that subtask.
    """
    if not goal or not goal_steps:
        return {}
    step_to_subtasks: dict[int, list[dict]] = {}
    for object_id, actions in goal.items():
        for i, (action, step) in enumerate(zip(actions, goal_steps.get(object_id, []))):
            step_to_subtasks.setdefault(step, []).append({"object": object_id, "action_index": i, "action": action})
    step_to_prior: dict[int, list[dict]] = {}
    for current in step_to_subtasks:
        prior = []
        for step in step_to_subtasks:
            if step < current:
                prior.extend(step_to_subtasks[step])
        step_to_prior[current] = prior
    return {
        "step_to_subtasks": step_to_subtasks,
        "step_to_prior_subtasks": step_to_prior,
        "max_step": max(step_to_subtasks) if step_to_subtasks else 0,
    }


def simulate_resume_completion(env, resume_handler: dict, current_step: int) -> tuple[int, list[str]]:
    """Port of ``simulate_resume_completion`` (code/evaluation/resume.py:56), verbatim semantics.

    Advances the env's ``_state_progress`` pointer over every goal that the demo state we just
    re-pinned to has already satisfied, and returns how many it advanced. Those goals are then
    charged to RESUME, never to the agent. v2 omitted this call entirely, so the pointer advanced
    lazily over the following steps and that credit landed on the agent instead.

    Mutates only ``env._state_progress`` -- a scoring-side monitor dict. It touches no qpos/qvel and
    no observable, so it cannot change what the policy sees. `--assert-digest-fixture` proves that.
    """
    if not resume_handler or current_step not in resume_handler["step_to_prior_subtasks"]:
        return 0, []
    completed = []
    if hasattr(env, "_state_progress"):
        for subtask in resume_handler["step_to_prior_subtasks"][current_step]:
            object_id, action_index = subtask["object"], subtask["action_index"]
            if object_id in env._state_progress and env._state_progress[object_id] <= action_index:
                env._state_progress[object_id] = action_index + 1
                completed.append(f"{object_id}_{action_index}")
    return len(completed), completed


def find_object_y_addr(sim, name: str) -> int | None:
    for candidate in (f"{name}_1_joint0", f"{name}_joint0", f"{name}_joint"):
        if candidate in sim.model.joint_names:
            return sim.model.get_joint_qpos_addr(candidate)[0] + 1
    return None


def build_env(bddl_path: Path, image_size: int):
    import libero.libero.envs.bddl_utils as BDDLUtils
    from libero.libero.envs import TASK_MAPPING
    from robosuite import load_controller_config

    problem = BDDLUtils.get_problem_info(str(bddl_path))
    env = TASK_MAPPING[problem["problem_name"]](
        bddl_file_name=str(bddl_path),
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=20,
    )
    for site in ("gripper0_grip_site_cylinder", "gripper0_grip_site"):
        if site in env.sim.model.site_names:
            env.sim.model.site_rgba[env.sim.model.site_name2id(site)][3] = 0.0
    return env


def observation_element(obs, prompt: str) -> dict:
    """Exactly openpi ``examples/libero/main.py``'s element: 180-degree rotation, pad-resize, 8-d state."""
    from openpi_client import image_tools
    from robosuite.utils.transform_utils import quat2axisangle

    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), RESIZE, RESIZE)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), RESIZE, RESIZE)
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": prompt,
    }


def pin_state(env, flat_state) -> dict:
    env.sim.set_state_from_flattened(flat_state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()


def wsm_fields(*, env_id: str, episode_id: str, step: int, max_steps: int, repin: bool) -> dict:
    """The omega server's identity/step contract. Only ever merged in under ``--wsm``."""
    return {
        "wsm_env_id": str(env_id),
        "wsm_episode_id": str(episode_id),
        "wsm_t": int(step),
        "wsm_episode_len": int(max_steps),
        "wsm_repin": bool(repin),
    }


def run_episode(
    env,
    client,
    *,
    goal,
    goal_steps,
    step_texts,
    step_states,
    dynamic,
    shift,
    replan,
    image_size,
    rng,
    wsm=False,
    wsm_env_id="env0",
    episode_id="",
    noise_key: tuple[str, str, int] | None = None,
    trace_digest: bool = False,
    expert_actions: bool = False,
    shadow_authors: bool = False,
) -> tuple[bool, int, int, dict]:
    import h5py  # noqa: F401  (kept so the import error surfaces early on a bad venv)

    segments = len(step_texts)
    max_steps = SWITCH_STEPS * segments
    total_goals = sum(len(v) for v in goal.values())

    # `dynamic_shift_description` starts one subtask ahead so the instruction never matches the
    # state the policy is actually in.
    instruction_offset = 1 if shift else 0
    obs = env.reset()
    if step_states is not None:
        obs = pin_state(env, step_states[min(instruction_offset, len(step_states) - 1)])

    distractor_addrs = []
    if dynamic:
        for name in MOVABLE_OBJECTS:
            addr = find_object_y_addr(env.sim, name)
            if addr is not None:
                distractor_addrs.append((addr, float(env.sim.data.qpos[addr])))

    action_plan: list[np.ndarray] = []
    completed_by_agent = 0
    prev_step_idx = 0
    # --- PROTOCOL v3 scoring state -------------------------------------------------------------
    # v2 kept ONE counter and a STICKY skip flag. v3 mirrors code/evaluation/episode.py exactly:
    #   * `skip_increment` is cleared unconditionally ONE step after the re-pin, so the segment's
    #     first genuine completion is counted (v2 discarded it -- the bug that made a perfect
    #     expert score 1.32%, below the policy's 1.97%).
    #   * `simulate_resume_completion` runs at every re-pin, so pointer-advance credit is charged
    #     to resume rather than leaking to the agent (v2 never called it, which inflated the
    #     authors'-rule reading in the other direction).
    # The v2 counter is kept alongside as `agent_subtasks_v2legacy` so every sealed v2 number stays
    # reproducible from a v3 run and re-scoring never needs the sim again.
    resume_handler = build_resume_handler(goal, goal_steps)
    completed_v2legacy = 0
    skip_v2legacy = step_states is not None
    total_prev_v2legacy = 0
    resume_credited = 0
    resume_skipped = 0
    # Authors start skip_increment False except under dynamic_shift_description, where the episode
    # opens already pinned one subtask ahead (episode.py:129-137).
    skip_increment = bool(shift)
    if shift and step_states is not None:
        gained_r, _ = simulate_resume_completion(env, resume_handler, instruction_offset)
        resume_credited += gained_r
    _, total_prev, _ = env._check_success(goal)
    total_prev_v2legacy = total_prev
    shadow = None
    if shadow_authors:
        shadow = AuthorsScorer(env, goal, resume_handler, dynamic_shift_description=bool(shift))
        if shift and step_states is not None:
            shadow.record_resume_completion(gained_r)
        shadow.baseline()
    seg_records: list[dict] = []
    seg_agent = seg_resume = 0
    seg_first_step: int | None = None
    seg_completion_steps: list[int] = []
    toggle, moved_this_segment = 1, False
    # Set at every subtask-boundary re-pin, consumed by the next request: the omega server must clear
    # its window on the first observation taken AFTER the sim jumped, not on the request before it.
    pending_repin = False
    requests_batched: list[int] = []
    digest = hashlib.blake2b(digest_size=8) if trace_digest else None
    # A SECOND digest, over what was SENT. Splits "the two runs disagree" into its only two causes:
    # obs digests differ  -> the SIM diverged and the server is off the hook;
    # obs equal, actions differ -> the server is not a function of its input.
    obs_digest = hashlib.blake2b(digest_size=8) if trace_digest else None
    first_obs_digest = None

    for t in range(max_steps + NUM_STEPS_WAIT):
        if t < NUM_STEPS_WAIT:
            obs, _, _, _ = env.step(DUMMY_ACTION)
            continue
        step = t - NUM_STEPS_WAIT
        step_idx = (step // SWITCH_STEPS) % segments

        if step_idx != prev_step_idx:
            # Close the segment that just ended, at per-segment resolution, so any future
            # re-scoring is an offline pass over the results file rather than another sim run.
            seg_records.append(
                {
                    "seg": prev_step_idx + 1,
                    "subtask": step_texts[(prev_step_idx + instruction_offset) % segments],
                    "agent": int(seg_agent),
                    "resume": int(seg_resume),
                    "first_completion_step": seg_first_step,
                    "completion_steps": list(seg_completion_steps),
                }
            )
            seg_agent = seg_resume = 0
            seg_first_step = None
            seg_completion_steps = []
            # Anchor: re-pin to the demo's ground-truth state for the new subtask.
            if step_states is not None and step_idx < len(step_states):
                obs = pin_state(env, step_states[step_idx])
                env.skip_pick_quat_once = True
                # v3: charge the goals the pinned state already satisfies to RESUME, atomically,
                # the way the authors do (episode.py:365-379). Without this the monitor's pointer
                # advances over the next few steps and the agent is credited for the teleport.
                gained_resume, _ = simulate_resume_completion(env, resume_handler, step_idx)
                resume_credited += gained_resume
                seg_resume += gained_resume
                skip_increment = True
                skip_v2legacy = True
                pending_repin = True
                if shadow is not None:
                    shadow.record_resume_completion(gained_resume)
                    shadow.on_transition()
            _, total_prev, _ = env._check_success(goal)
            total_prev_v2legacy = total_prev
            action_plan.clear()
            moved_this_segment = False
            prev_step_idx = step_idx

        if dynamic and distractor_addrs and not moved_this_segment and step % SWITCH_STEPS == 10:
            addr, base = distractor_addrs[rng.randrange(len(distractor_addrs))]
            env.sim.data.qpos[addr] = base + 0.15 * toggle
            obs = pin_state(env, env.sim.get_state().flatten())
            toggle, moved_this_segment = -toggle, True

        if not action_plan:
            prompt = step_texts[(step_idx + instruction_offset) % segments]
            element = observation_element(obs, prompt)
            if expert_actions:
                element["_expert_seg"] = step_idx
                element["_expert_step_in_seg"] = step % SWITCH_STEPS
            if wsm:
                element.update(
                    wsm_fields(
                        env_id=wsm_env_id, episode_id=episode_id, step=step, max_steps=max_steps, repin=pending_repin
                    )
                )
                pending_repin = False
            if noise_key is not None:
                # THE field that makes K>1 legal. `step` is protocol-determined (a request happens
                # iff the plan is empty, i.e. every `replan` steps and at every subtask boundary),
                # so this coordinate is identical across arms, shards and gather compositions.
                element["policy_noise_seed"] = np.uint32(
                    policy_noise_seed(noise_key[0], noise_key[1], noise_key[2], step)
                )
            if obs_digest is not None:
                payload = b"".join(
                    np.ascontiguousarray(np.asarray(element[key])).tobytes()
                    for key in ("observation/image", "observation/wrist_image", "observation/state")
                )
                payload += str(element["prompt"]).encode("utf-8")
                payload += str(int(np.asarray(element.get("policy_noise_seed", 0)).item())).encode()
                obs_digest.update(payload)
                if first_obs_digest is None:
                    first_obs_digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
            result = client.infer(element)
            if digest is not None:
                # Rolling hash of EVERY action chunk the server returned, in order. Two runs of the
                # same (mode, case, trial) that agree here ran the same trajectory; agreeing only on
                # success/subtasks would also happen for two different episodes that both failed.
                digest.update(np.ascontiguousarray(np.asarray(result["actions"], dtype=np.float32)).tobytes())
            # Did the gather actually gather? A K>1 run whose realised batches are all 1 is paying
            # the gather wait for nothing (sparse arrivals), and the results file should say so
            # rather than leave it to be inferred from wall-clock later.
            realised = (
                result.get("policy_timing", {}).get("gather_batch_n")
                if isinstance(result.get("policy_timing"), dict)
                else None
            )
            if realised is not None:
                requests_batched.append(int(realised))
            chunk = result["actions"]
            action_plan = [np.asarray(a) for a in chunk[:replan]]

        obs, _, _, _ = env.step(action_plan.pop(0).tolist())

        if shadow is not None:
            shadow.update_completion_tracking()
        _, total_now, all_done = env._check_success(goal)
        # --- v3 counter: authors' update_completion_tracking (episode.py:408-436) --------------
        gained = total_now - total_prev
        if gained > 0:
            if skip_increment:
                resume_skipped += gained
                seg_resume += gained
            else:
                completed_by_agent += gained
                seg_agent += gained
                step_in_seg = step % SWITCH_STEPS
                seg_completion_steps.extend([step_in_seg] * gained)
                if seg_first_step is None:
                    seg_first_step = step_in_seg
        total_prev = total_now
        # Cleared unconditionally one step after the transition -- THE v2 fix.
        if skip_increment:
            skip_increment = False
        # --- v2 counter, preserved verbatim so sealed numbers stay reproducible ---------------
        gained_legacy = total_now - total_prev_v2legacy
        if gained_legacy > 0:
            if not skip_v2legacy:
                completed_v2legacy += gained_legacy
            skip_v2legacy = False
            total_prev_v2legacy = total_now

    seg_records.append(
        {
            "seg": prev_step_idx + 1,
            "subtask": step_texts[(prev_step_idx + instruction_offset) % segments],
            "agent": int(seg_agent),
            "resume": int(seg_resume),
            "first_completion_step": seg_first_step,
            "completion_steps": list(seg_completion_steps),
        }
    )
    _, _, all_done = env._check_success(goal)
    telemetry = {
        "protocol": "v3",
        "shadow_authors_agent": (int(shadow.total_agent_subtasks) if shadow is not None else None),
        "shadow_authors_resume": (
            (int(shadow.total_resume_completed), int(shadow.total_resume_skipped)) if shadow is not None else None
        ),
        "agent_subtasks_v2legacy": int(completed_v2legacy),
        "resume_credited": int(resume_credited),
        "resume_skipped": int(resume_skipped),
        "segments": seg_records,
        "action_digest": digest.hexdigest() if digest is not None else None,
        "obs_digest": obs_digest.hexdigest() if obs_digest is not None else None,
        "first_obs_digest": first_obs_digest,
        "requests": len(requests_batched) or None,
        "mean_gather_batch": (round(sum(requests_batched) / len(requests_batched), 2) if requests_batched else None),
        "min_gather_batch": min(requests_batched) if requests_batched else None,
    }
    return bool(all_done), completed_by_agent, total_goals, telemetry


class ExpertPolicy:
    """Debug-only oracle: replays the demo's OWN actions for the current segment.

    Exists so the acceptance test for a scoring change can be run through the REAL harness rather
    than a re-implementation of it. Like ``--random-actions`` it never contacts a server, and like
    ``--random-actions`` it is unreachable on any scored run. It needs the (segment, step) it is
    being asked for, which the harness adds to the element ONLY under ``--expert-actions`` -- so a
    scored request stays byte-identical to what it always was.
    """

    def __init__(self, horizon: int, skip_noops: bool = False) -> None:
        self._horizon = horizon
        self._skip_noops = skip_noops
        self._actions = None
        self._starts = None
        self._per_seg = None

    @staticmethod
    def _is_noop(action, prev) -> bool:
        """The build-time filter (replay_shard.py:73), so the 'active' expert replays exactly the
        frames the policy was TRAINED on -- the right ceiling for a no-op-filtered policy."""
        if prev is None:
            return bool(np.linalg.norm(action[:-1]) < 1e-4)
        return bool(np.linalg.norm(action[:-1]) < 1e-4 and action[-1] == prev[-1])

    def begin_episode(self, raw_actions, start_frames) -> None:
        self._actions, self._starts = raw_actions, list(start_frames)
        if not self._skip_noops:
            self._per_seg = None
            return
        ends = list(start_frames[1:]) + [len(raw_actions)]
        self._per_seg = []
        for a, b in zip(start_frames, ends):
            kept, prev = [], None
            for act in raw_actions[a:b]:
                if not self._is_noop(act, prev):
                    kept.append(act)
                prev = act
            self._per_seg.append(np.asarray(kept if kept else raw_actions[a : a + 1]))

    def infer(self, element: dict) -> dict:
        seg = int(element["_expert_seg"])
        offset = int(element["_expert_step_in_seg"])
        if self._per_seg is not None:
            src = self._per_seg[min(seg, len(self._per_seg) - 1)]
            chunk = np.asarray(src[offset : offset + self._horizon], dtype=np.float32)
        else:
            base = self._starts[min(seg, len(self._starts) - 1)] + offset
            end = min(base + self._horizon, len(self._actions))
            chunk = np.asarray(self._actions[base:end], dtype=np.float32)
        if len(chunk) == 0:  # demo exhausted: hold still, keep the last gripper command
            hold = np.zeros((self._horizon, 7), dtype=np.float32)
            hold[:, 6] = float(self._actions[-1][6])
            return {"actions": hold}
        if len(chunk) < self._horizon:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self._horizon - len(chunk), 0)])
        return {"actions": chunk}


class RandomPolicy:
    """Stand-in for the policy server so the protocol can be smoke-tested without one.

    Exercises every env-side seam -- bddl load, init-file pinning, the 150-step anchor timer,
    subtask re-pinning, distractor teleports, `_check_success` scoring -- and nothing else.
    """

    def __init__(self, horizon: int, rng: random.Random) -> None:
        self._horizon, self._rng = horizon, rng

    def infer(self, element: dict) -> dict:
        del element
        actions = np.zeros((self._horizon, 7), dtype=np.float32)
        actions[:, :6] = np.asarray([[self._rng.uniform(-0.2, 0.2) for _ in range(6)] for _ in range(self._horizon)])
        actions[:, 6] = -1.0
        return {"actions": actions}


def main() -> None:
    global SWITCH_STEPS  # noqa: PLW0603 - --switch-steps is a debug-only protocol override
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--modes", nargs="+", default=list(MODE_FLAGS))
    parser.add_argument("--cases", nargs="+", default=None, help="e.g. case1 case2; default all 10")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replan", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="smoke test: skip the policy server and drive with random actions",
    )
    parser.add_argument(
        "--expert-skip-noops",
        action="store_true",
        help="with --expert-actions, replay only the no-op-filtered frames, i.e. the "
        "frames the policy was trained on. This is the ceiling a no-op-trained "
        "policy is actually racing, not the raw demo's pause-inclusive pace.",
    )
    parser.add_argument(
        "--shadow-authors-scorer",
        action="store_true",
        help="run a verbatim port of the authors' bookkeeping alongside v3 and "
        "record its counts; the two must agree exactly (acceptance oracle b)",
    )
    parser.add_argument(
        "--expert-actions",
        action="store_true",
        help="oracle: skip the policy server and drive with the demo's OWN actions. "
        "The scoring acceptance test -- a correct scorer must credit an expert "
        "that replays the demo, and v2 credited it 1.32%% (below the policy).",
    )
    parser.add_argument(
        "--switch-steps", type=int, default=SWITCH_STEPS, help="debug only; the benchmark protocol is 150"
    )
    parser.add_argument("--out", required=True, help="results json")
    # Self-labelling: every eval cell must carry its own provenance. An unlabelled results file
    # cannot be attributed to an arm/checkpoint/encoder and is therefore not evidence.
    parser.add_argument(
        "--arm", required=True, help="arm id, e.g. A0_base / A1_gdn_w8 / A2_gdn_w16_hd05 / A3_jepa / A4_ptrm"
    )
    parser.add_argument("--ckpt-sha", required=True, help="sha256 (or step-pinned id) of the served policy checkpoint")
    parser.add_argument(
        "--encoder-sha",
        default="none",
        help="sha256 of the omega encoder used at serve time; 'none' for arms that read no omega",
    )
    parser.add_argument("--budget-steps", type=int, default=-1, help="train steps of the served checkpoint")
    # Omega serving (A1/A2/A4). Opt-in and strictly additive: absent, every request is byte-identical
    # to the base arms' requests, so A0/A3 keep running unchanged against serve_pi05_libero.py.
    parser.add_argument(
        "--wsm",
        action="store_true",
        help="send wsm_env_id/wsm_t/wsm_episode_len/wsm_episode_id/wsm_repin with "
        "every request (required by serve_pi05_libero_wsm.py; a base server "
        "would reject the extra keys)",
    )
    parser.add_argument(
        "--wsm-env-id", default="env0", help="this client's env slot on the omega server (one env per client today)"
    )
    parser.add_argument("--note", default="", help="free-form label, e.g. G2_canary / G3_probe_15k")
    # Protocol amendment 1: memory modes get extra trials, run as an ADDITIVE block. This offsets
    # only the reported trial index and the provenance record -- it does not reseed anything,
    # because Memory_Execution/Memory_Exploration consume no harness rng (MODE_FLAGS = False,False),
    # so a second invocation is already a valid independent block. Modes that DO consume the rng
    # stay at 10 trials and are never topped up.
    parser.add_argument(
        "--trial-start",
        type=int,
        default=0,
        help="label offset for an additive trial block (e.g. 10 for trials 11..N)",
    )
    # K env runners per GPU. Whole (case, trial) units only; merge with merge_eval_shards.py.
    parser.add_argument(
        "--num-shards", type=int, default=1, help="how many runner processes split this cell's TRIALS (K envs per GPU)"
    )
    parser.add_argument(
        "--shard", type=int, default=0, help="this runner's index in [0, num_shards); owns trials shard::num_shards"
    )
    parser.add_argument(
        "--trace-digest",
        action="store_true",
        help="record a blake2b digest of every action chunk in each per-trial row. "
        "Turns 'the two runs agree' from a claim about success flags into a "
        "claim about the trajectory; costs a hash per request.",
    )
    parser.add_argument(
        "--deterministic-seeding",
        action="store_true",
        help="send policy_noise_seed=blake2b(mode|case|trial|step) with every request "
        "and seed each episode's harness rng from its own coordinate. Makes the "
        "outcome independent of K, shard, gather composition and arrival order. "
        "Required for --num-shards > 1; OFF reproduces the pre-sharding path byte "
        "for byte.",
    )
    args = parser.parse_args()

    # --- A4 / PTRM serve-config guard -------------------------------------------------------
    # PTRM's inference-time triple must stay (k=1, sigma=0, select="q"). Those exact values consume
    # NO rng at serve; anything else makes the head sample, and a sampling head inside a gather
    # window is sensitive to BATCH POSITION -- which would silently void the whole v2-final seeding
    # guarantee (identical coordinates would stop implying identical actions). The failure would be
    # invisible in the results file, so it is asserted here rather than trusted.
    if args.arm.lower().startswith("a4") or "ptrm" in args.arm.lower():
        try:
            import openpi.training.config as _cfg

            m = _cfg.get_config("pi05_robocerebra_ptrm").model
            triple = (int(m.wsm_ptrm_eval_k), float(m.wsm_ptrm_eval_sigma), str(m.wsm_ptrm_eval_select))
            if triple != (1, 0.0, "q"):
                raise SystemExit(
                    f"A4 PTRM serve config drifted to k/sigma/select={triple}; required (1, 0.0, 'q'). "
                    "A sampling PTRM head is batch-position dependent, which breaks the v2-final "
                    "policy_noise_seed guarantee. Refusing to produce a cell that cannot be trusted."
                )
            log(f"[A4 guard] wsm_ptrm_eval k/sigma/select = {triple} (deterministic, consumes no rng)")
        except ModuleNotFoundError:
            # The sim venv has no openpi; the server-side process asserts the same triple.
            log("[A4 guard] openpi not importable here; relying on the server-side assertion")

    if args.num_shards < 1:
        parser.error(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard {args.shard} outside [0, {args.num_shards})")
    if args.num_shards > 1 and not args.deterministic_seeding:
        parser.error(
            "--num-shards > 1 without --deterministic-seeding. Splitting trials across runners "
            "re-cuts both the server's mutable action-noise rng and this harness's global "
            "distractor rng, so the shards would not reproduce the unsharded run and would not "
            "reproduce each other. Add --deterministic-seeding."
        )

    SWITCH_STEPS = args.switch_steps
    provenance = {
        "arm": args.arm,
        "ckpt_sha": args.ckpt_sha,
        "encoder_sha": args.encoder_sha,
        "budget_steps": args.budget_steps,
        "note": args.note,
        "modes": list(args.modes),
        "cases": args.cases,
        "trials": args.trials,
        "trial_start": args.trial_start,
        "trial_block": f"{args.trial_start + 1}..{args.trial_start + args.trials}",
        "replan": args.replan,
        "switch_steps": SWITCH_STEPS,
        "seed": args.seed,
        "random_actions": bool(args.random_actions),
        # Whether this cell read omega at serve time. An arm label alone cannot say it: A1 served
        # without --wsm is not A1, it is A0 with A1's weights.
        "wsm": bool(args.wsm),
        "wsm_env_id": args.wsm_env_id if args.wsm else None,
        # Sharding + seeding are properties of HOW the numbers were produced, so they are provenance,
        # not options. A shard file that does not say it is a shard cannot be merged safely.
        "num_shards": args.num_shards,
        "shard": args.shard,
        "deterministic_seeding": bool(args.deterministic_seeding),
        "seed_rule": (
            "blake2b(mode|case|trial|step)->uint32 policy_noise_seed"
            if args.deterministic_seeding
            else "server-side mutable rng (order-dependent)"
        ),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log(f"provenance: {json.dumps(provenance)}")

    import h5py

    bench_root = Path(args.bench_root)
    rng = random.Random(args.seed)
    if args.random_actions:
        client = RandomPolicy(args.replan, rng)
    elif args.expert_actions:
        client = ExpertPolicy(args.replan, skip_noops=args.expert_skip_noops)
    else:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    results = []
    per_trial = []  # the merge unit: one row per (mode, case, trial), union-able across shards
    my_trials = list(range(args.shard, args.trials, args.num_shards))
    if args.num_shards > 1:
        log(f"shard {args.shard}/{args.num_shards}: trials {my_trials} of {args.trials} per case")

    for mode in args.modes:
        dynamic, shift = MODE_FLAGS[mode]
        source_dir = bench_root / ("Ideal" if mode in IDEAL_ALIASES else mode)
        init_dir = bench_root / "init_files" / ("Ideal" if mode in IDEAL_ALIASES else mode)
        case_names = args.cases or sorted(
            (p.name for p in source_dir.iterdir() if p.is_dir() and p.name.startswith("case")),
            key=lambda n: int(n[4:]),
        )

        for case_name in case_names:
            case_dir = source_dir / case_name
            bddl = next(case_dir.glob("*.bddl"))
            step_texts, start_frames = parse_steps(case_dir / "task_description.txt")
            goal, goal_steps = load_goal(case_dir / "goal.json")
            with h5py.File(case_dir / "demo.hdf5", "r") as handle:
                raw_states = handle["data"]["demo_1"]["states"][()]
                raw_actions = handle["data"]["demo_1"]["actions"][()] if args.expert_actions else None
            step_states = [raw_states[min(f, len(raw_states) - 1)] for f in start_frames]
            if args.expert_actions:
                client.begin_episode(raw_actions, start_frames)

            init_file = init_dir / f"{case_name}.init"
            initial_state = pickle.loads(init_file.read_bytes()) if init_file.is_file() else None

            env = build_env(bddl, args.image_size)
            successes, agent_subtasks, possible = 0, 0, 0
            for trial in my_trials:
                # The seed COORDINATE is the absolute trial index. With --trial-start (the
                # amendment-1 top-up block) the loop variable restarts at 0, so seeding on it
                # would re-run trials 1..10 byte-for-byte and label them 11..20 -- a no-op that
                # looks like new data. Everything coordinate-derived below uses `trial_abs`.
                trial_abs = args.trial_start + trial
                if args.deterministic_seeding:
                    # THE SIM's rng, and it is not optional. robosuite's placement sampler draws from
                    # the GLOBAL, unseeded ``np.random`` (placement_samplers.py: np.random.uniform),
                    # and ``env.reset()`` with hard_reset re-runs ``_load_model()``. So two processes
                    # -- and two runs in the SAME process -- start the same (mode, case, trial) from
                    # different scenes, and ``pin_state`` cannot undo it: the flattened state restores
                    # qpos/qvel, not what the model was built with. Measured: two identical K=1 runs
                    # of Ideal/case1/trial0 disagreed on the FIRST observation they sent
                    # (first_obs_digest 6e1ee6bf... vs 5cca97dd...). That is a property this harness
                    # always had; it is why "same trial twice" never reproduced, at any K. One
                    # coordinate-derived seed per episode fixes it, and belongs to exactly the same
                    # opt-in as the policy noise seed.
                    np.random.seed(episode_rng_seed(mode, case_name, trial_abs, args.seed))
                env.reset()
                if initial_state is not None:
                    pin_state(env, initial_state)
                # Under --deterministic-seeding the distractor stream is a function of the episode
                # coordinate, not of how many episodes this process happened to run first. That is
                # what lets shard 1 own trials 1,3,5 and still produce trial 3's episode exactly.
                episode_rng = (
                    random.Random(episode_rng_seed(mode, case_name, trial_abs, args.seed))
                    if args.deterministic_seeding
                    else rng
                )
                episode_t0 = time.monotonic()
                done, gained, total, telemetry = run_episode(
                    env,
                    client,
                    goal=goal,
                    goal_steps=goal_steps,
                    step_texts=step_texts,
                    step_states=step_states,
                    dynamic=dynamic,
                    shift=shift,
                    replan=args.replan,
                    image_size=args.image_size,
                    rng=episode_rng,
                    wsm=args.wsm,
                    wsm_env_id=args.wsm_env_id,
                    episode_id=f"{mode}/{case_name}/trial{trial_abs}",
                    noise_key=(mode, case_name, trial_abs) if args.deterministic_seeding else None,
                    trace_digest=args.trace_digest,
                    expert_actions=args.expert_actions,
                    shadow_authors=args.shadow_authors_scorer,
                )
                successes += int(done)
                agent_subtasks += gained
                possible += total
                episode_s = time.monotonic() - episode_t0
                env_steps = SWITCH_STEPS * len(step_texts) + NUM_STEPS_WAIT
                per_trial.append(
                    {
                        "mode": mode,
                        "case": case_name,
                        "trial": trial_abs,
                        "success": bool(done),
                        "agent_subtasks": int(gained),
                        "possible_subtasks": int(total),
                        "num_subtasks": len(step_texts),
                        "subtask_texts": list(step_texts),
                        "bddl": bddl.name,
                        "shard": args.shard,
                        # Throughput accounting lives with the result, so "how fast was it" never has to
                        # be reconstructed from log timestamps after the fact.
                        "wall_s": round(episode_s, 3),
                        "env_steps": int(env_steps),
                        "env_steps_per_s": round(env_steps / episode_s, 3) if episode_s > 0 else None,
                        **telemetry,
                    }
                )
                log(
                    f"{mode}/{case_name} trial {trial_abs}: "
                    f"success={done} subtasks={gained}/{total} "
                    f"(v2legacy={telemetry['agent_subtasks_v2legacy']} "
                    f"resume={telemetry['resume_credited']}+{telemetry['resume_skipped']}) "
                    f"[{episode_s:.1f}s, {env_steps / max(episode_s, 1e-9):.2f} env-steps/s]"
                )
            env.close()

            results.append(
                {
                    "mode": mode,
                    "case": case_name,
                    "trials": len(my_trials),
                    "successes": successes,
                    "success_rate": successes / len(my_trials) if my_trials else 0.0,
                    "agent_subtasks": agent_subtasks,
                    "possible_subtasks": possible,
                    "subtask_rate": agent_subtasks / possible if possible else 0.0,
                    "num_subtasks": len(step_texts),
                    "bddl": bddl.name,
                }
            )
            Path(args.out).write_text(
                json.dumps(
                    {"provenance": provenance, "per_case": results, "per_trial": per_trial, "complete": False},
                    indent=2,
                )
            )

    by_mode: dict[str, dict] = {}
    for row in results:
        bucket = by_mode.setdefault(
            row["mode"], {"successes": 0, "trials": 0, "agent_subtasks": 0, "possible_subtasks": 0}
        )
        for key in bucket:
            bucket[key] += row[key]
    for mode, bucket in by_mode.items():
        log(
            f"{mode}: success {bucket['successes']}/{bucket['trials']} "
            f"subtask {bucket['agent_subtasks']}/{bucket['possible_subtasks']}"
        )
    provenance["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.out).write_text(
        json.dumps(
            {
                "provenance": provenance,
                "per_case": results,
                "per_trial": per_trial,
                "by_mode": by_mode,
                "complete": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
