#!/usr/bin/env python3
"""ReMemBench 13-task rollout runner (box tier) — one worker's shard.

Why this exists instead of ``vla_training/eval/eval_pi_05.py``: that driver asserts
``len(manifest_records_for_task(task)) == cfg.num_trials`` for every task, i.e. a UNIFORM
number of episodes per task. The sealed ReMemBench held-out manifest is deliberately
non-uniform (3..12 episodes/task — the held-out tail is 20% of however many demos were
collected, floored at 3), so that driver cannot consume it. This runner is a thin,
eval-side shim: the observation packing, action decoding, replan cadence, and
``policy_noise_seed`` contract are copied byte-for-byte from ``eval_pi_05.eval_task_manifest``
so the numbers are comparable; only episode bookkeeping differs.

Protocol (locked):
  * Reset  = ``env.reset(seed=episode.seed, options={"ep_meta": episode.reset.ep_meta})``.
    BOTH are required. ep_meta pins object counts/instances (MemPutK* draw counts from the
    global ``np.random``, which no per-env seed reaches); the seed pins the placement
    sampling ep_meta leaves to ``env.rng``. ep_meta alone drifts ~0.15 m between replays.
  * Rollouts = ``--rollouts`` (3) per episode. The reset is IDENTICAL across the three;
    the only thing that varies is the pi diffusion noise, because the reset is fully
    pinned by design. Varying the reset seed instead would change the initial state and
    stop measuring what "3 rollouts of the same held-out episode" is supposed to measure.
    Rollout 0 reuses the episode seed verbatim, so it is bit-identical to a
    single-rollout run of the same manifest; rollouts 1..k-1 derive an independent
    uint32 noise stream (:func:`rollout_noise_base`).
  * Horizon = per-episode, from the manifest (Prospective needs 1500-3200 steps).
  * Termination = success, ``failed_task`` (prospective deadline blown -> HARD FAIL, never
    scored a success), or horizon exhaustion.
  * Reported unit = the 4 memory CATEGORIES, unweighted mean over the variants in each
    (six of the thirteen variants are corner/side permutations of two spatial tasks; a flat
    13-task mean would triple-count the spatial condition).

Sharding: work is grouped by task (one env build per task, reused across its episodes) and
tasks are bin-packed across workers by ``horizon * n_episodes`` with greedy LPT, which is
deterministic and independent of worker count ordering. One worker per GPU, each against
its own policy server.

  python scripts/remembench/run_remembench_eval.py --manifest <sealed.json> \
      --out-dir /data/work/remembench_evals/<arm> --worker-idx 0 --num-workers 4 \
      --port 5800 --rollouts 3 --replan-steps 8
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import socket
import sys
import time
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vla_training.eval.remembench_tasks import (  # noqa: E402
    get_remembench_category,
)

RESIZE = 224  # pi0.5 RoboCasa input resolution — must match training (examples/robocasa/main.py)
NOISE_KIND = "remembench_rollout_v1"


# --------------------------------------------------------------------------------------
# protocol primitives
# --------------------------------------------------------------------------------------
def policy_noise_seed(episode_seed: int, env_step: int) -> int:
    """Stable uint32 key for one pi action-chunk diffusion draw.

    Vendored from ``vla_training.eval.eval_manifest`` (identical bytes, identical kind
    string) so this runner does not drag the manifest module's RoboCasa imports into the
    ReMemBench venv. Independent of request order, server replica, and shard topology.
    """
    if not 0 <= int(episode_seed) <= 0xFFFFFFFF:
        raise ValueError(f"episode_seed is outside uint32: {episode_seed}")
    if int(env_step) < 0:
        raise ValueError(f"env_step must be nonnegative, got {env_step}")
    raw = f"pi_diffusion_sha256_v1\0{int(episode_seed)}\0{int(env_step)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def rollout_noise_base(episode_seed: int, rollout_idx: int) -> int:
    """Per-rollout uint32 noise-stream root. Rollout 0 == the episode seed, unchanged."""
    if rollout_idx == 0:
        return int(episode_seed)
    raw = f"{NOISE_KIND}\0{int(episode_seed)}\0{int(rollout_idx)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def load_task_prompts(path, expected_tasks):
    """Load + validate the canonical demo-independent task->prompt manifest.

    Mirrors ``scripts/launch/validate_stage_s_task_prompts.load_task_prompts`` for the
    ReMemBench namespace. The workspace serve path (``--interface tanh``) forces
    ``--tap-prompt terse``, which sets ``require_wsm_prompt`` server-side: every request
    must carry a non-empty canonical ``wsm_prompt``, used ONLY by the frozen representation
    tap. It never reaches the action-policy prompt.
    """
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    if manifest.get("artifact") != "remembench_train13_task_prompts":
        raise ValueError(f"{path}: unexpected artifact {manifest.get('artifact')!r}")
    if manifest.get("global_language_mode") != "canonical_terse_task_instruction":
        raise ValueError(f"{path}: global_language_mode mismatch")
    if manifest.get("demo_derived") is not False:
        raise ValueError(f"{path}: task prompts must declare demo_derived=false")
    records = manifest["tasks"]
    if not isinstance(records, list) or len(records) != len(expected_tasks):
        raise ValueError(f"{path}: expected {len(expected_tasks)} tasks, got {len(records)}")
    prompts = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"task", "prompt"}:
            raise ValueError(f"{path}: record {index} must contain task,prompt exactly")
        task, prompt = record["task"], record["prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or prompt != prompt.strip():
            raise ValueError(f"{path}: invalid prompt for {task!r}")
        if task in prompts:
            raise ValueError(f"{path}: duplicate task {task!r}")
        prompts[task] = prompt
    missing = set(expected_tasks) - set(prompts)
    if missing:
        raise ValueError(f"{path}: missing prompts for {sorted(missing)}")
    return prompts


def convert_action(action):
    """Flat 12-dim pi action -> the wrapper's action dict.

    Vendored from stock ``robocasa.utils.env_utils.convert_action``: the ReMemBench fork
    (RoboCasa v0.2) predates that helper and does not ship it, but its gym wrapper's
    ``PandaOmronKeyConverter.unmap_action`` consumes exactly these five keys.
    """
    action = np.asarray(action).copy()
    return {
        "action.end_effector_position": action[0:3],
        "action.end_effector_rotation": action[3:6],
        "action.gripper_close": action[6:7],
        "action.base_motion": action[7:11],
        "action.control_mode": action[11:12],
    }


def shard_tasks_lpt(task_costs: dict, worker_idx: int, num_workers: int) -> list:
    """Greedy longest-processing-time bin-pack. Deterministic; returns this worker's tasks."""
    bins = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    for task, cost in sorted(task_costs.items(), key=lambda kv: (-kv[1], kv[0])):
        target = min(range(num_workers), key=lambda i: (loads[i], i))
        bins[target].append(task)
        loads[target] += cost
    return sorted(bins[worker_idx])


def wait_for_port(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(5)
    return False


def write_json_atomic(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------------------
def run_episode(
    env, client, spec, rollout_idx, replan_steps, record_video, wsm_prompt=None, env_id="rmb-w0", obs_image_size=RESIZE
):
    """One pinned rollout. Returns (result dict, frames).

    ``obs_image_size`` is the edge length the observation frames are sent at. DEFAULT 224 = the
    pi0.5 contract, byte-identical to every sealed pi arm. It exists for the GR00T backbone-
    generality arm: GR00T's own processor resizes the frame it is handed, so sending a
    224-pad-resized frame would put a SECOND resample in front of the one training used
    (env 256 -> processor target). Passing 256 (the env's native render size) sends the frame
    untouched and leaves GR00T with exactly the single canonical resize it trained under.
    When the requested size equals the native frame size the resize is skipped entirely rather
    than round-tripped through PIL, so 'native' means native to the byte.
    """
    from openpi_client import image_tools

    reset_spec = spec["reset"]
    if reset_spec.get("kind") != "remembench_ep_meta":
        raise ValueError(f"unsupported reset kind: {reset_spec.get('kind')!r}")

    obs, info = env.reset(
        seed=int(spec["seed"]),
        options={"ep_meta": reset_spec["ep_meta"]},
    )
    lang = obs["annotation.human.task_description"]
    if not isinstance(lang, str) or not lang.strip():
        raise ValueError(f"empty task language for {spec['task']}")

    noise_base = rollout_noise_base(int(spec["seed"]), rollout_idx)
    horizon = int(spec["horizon"])
    plan = collections.deque()
    frames = []
    success = False
    failed_task = False
    started = time.time()
    n_infer = 0
    t = 0

    while t < horizon:
        if not plan:

            def prep(key):
                frame = np.ascontiguousarray(obs[key])
                if frame.shape[0] == obs_image_size and frame.shape[1] == obs_image_size:
                    # Already native at the requested size: no resample at all (see the
                    # obs_image_size note in this function's docstring).
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
                "prompt": lang,
                # Baseline (s0) serves through the plain openpi interface and ignores the
                # wsm_* fields; they are sent so a workspace arm is drop-in comparable.
                "wsm_t": t,
                "wsm_task": spec["task"],
                # STABLE across every episode and rollout this worker runs — the same
                # contract as the sealed client (eval_pi_05.py:187, where env_id is bound
                # once per task-shard, outside the episode loop). The workspace server keys
                # per-env omega state by wsm_env_id and REFUSES to evict live state
                # (serve_pi_05_wsm.py:_validate_batch:438), so a per-episode-unique id makes
                # every episode look like a second concurrent env and trips the bound. The
                # episode boundary is signalled by wsm_t==0, which rebuilds the conditioner
                # (serve_pi_05_wsm.py:520 conditioner.reset + :548 fresh _EpisodeState) —
                # that, not a new identity, is what clears the omega window between episodes.
                "wsm_env_id": env_id,
                "wsm_demo_episode": int(spec["episode_index"]),
                "policy_noise_seed": policy_noise_seed(noise_base, t),
            }
            if wsm_prompt is not None:
                # Private canonical task language for the frozen tap only. The workspace
                # serve strips it before the action-policy transform, so the policy still
                # receives the env's own annotation as `prompt`.
                element["wsm_prompt"] = wsm_prompt
            response = client.infer(element)
            n_infer += 1
            chunk = response["actions"]
            if len(chunk) < replan_steps:
                raise AssertionError(f"chunk {len(chunk)} < replan {replan_steps}")
            actions = np.asarray(chunk[:replan_steps])
            if not np.all(np.isfinite(actions)):
                raise RuntimeError(
                    f"non-finite action from server at {spec['task']} ep={spec['episode_index']} r={rollout_idx} t={t}"
                )
            plan.extend(actions)

        obs, _reward, terminated, truncated, info = env.step(convert_action(plan.popleft()))
        success = bool(info["success"])
        failed_task = bool(info.get("failed_task", False))
        if record_video and (t % 4 == 0 or success):
            frames.append(np.ascontiguousarray(env.render()))
        t += 1
        if success or failed_task or terminated or truncated:
            break

    result = {
        "task": spec["task"],
        "category": spec.get("category") or get_remembench_category(spec["task"]),
        "episode_index": int(spec["episode_index"]),
        "rollout": int(rollout_idx),
        "seed": int(spec["seed"]),
        "noise_base": int(noise_base),
        "horizon": horizon,
        # failed_task is a HARD failure: a blown prospective deadline can never be a success.
        "success": bool(success and not failed_task),
        "failed_task": failed_task,
        "episode_length": int(t),
        "n_infer": n_infer,
        "rollout_seconds": round(time.time() - started, 1),
    }
    return result, frames


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256", default=None, help="expected sealed manifest_sha256")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5800)
    ap.add_argument("--worker-idx", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--replan-steps", type=int, default=8)
    ap.add_argument(
        "--obs-image-size",
        type=int,
        default=RESIZE,
        help=(
            "edge length observation frames are sent at. DEFAULT 224 = the pi0.5 contract, "
            "byte-identical to every sealed pi arm. Use 256 (the env's native render size) for "
            "the GR00T arm so its processor performs the single resize it trained under instead "
            "of resampling an already-224-padded frame."
        ),
    )
    ap.add_argument("--server-wait-s", type=int, default=1200)
    ap.add_argument("--video", choices=["none", "first"], default="none")
    ap.add_argument("--tasks", default=None, help="comma list; restrict to these tasks")
    ap.add_argument(
        "--task-prompt-manifest",
        default=None,
        help=(
            "canonical remembench13 task->prompt manifest; REQUIRED by the workspace serve "
            "path (--interface tanh forces --tap-prompt terse => require_wsm_prompt)"
        ),
    )
    ap.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="preflight only: cap episodes per task",
    )
    ap.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help=(
            "preflight only: skip the first N episodes of each task. Exists to prove "
            "server-side episode isolation: episode k run ALONE must be bit-identical to "
            "episode k run after episode k-1, since both reset and policy noise are pinned"
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.manifest, "rb") as handle:
        raw = handle.read()
    manifest = json.loads(raw)
    file_sha = hashlib.sha256(raw).hexdigest()
    sealed_sha = manifest["manifest_sha256"]
    if args.manifest_sha256 and sealed_sha != args.manifest_sha256:
        raise SystemExit(f"manifest_sha256 mismatch: {sealed_sha} != {args.manifest_sha256}")
    if args.rollouts < 1:
        raise SystemExit("--rollouts must be >= 1")

    by_task = collections.OrderedDict()
    for record in manifest["episodes"]:
        by_task.setdefault(record["task"], []).append(record)
    if args.tasks:
        keep = {t.strip() for t in args.tasks.split(",") if t.strip()}
        by_task = collections.OrderedDict((t, v) for t, v in by_task.items() if t in keep)
    if args.episode_offset or args.max_episodes_per_task:
        for task, records in by_task.items():
            records.sort(key=lambda r: int(r["episode_index"]))
        start = int(args.episode_offset)
        stop = None if not args.max_episodes_per_task else start + args.max_episodes_per_task
        by_task = collections.OrderedDict((t, v[start:stop]) for t, v in by_task.items())
    for task, records in by_task.items():
        records.sort(key=lambda r: int(r["episode_index"]))

    # Prompt provenance is validated against the COMPLETE 13-task universe even when
    # --tasks restricts execution, so a subset can never bless a partial prompt table.
    task_prompts = None
    if args.task_prompt_manifest:
        from vla_training.eval.remembench_tasks import REMEMBENCH_TASKS

        task_prompts = load_task_prompts(args.task_prompt_manifest, REMEMBENCH_TASKS)
        print(
            f"[w{args.worker_idx}] canonical wsm_prompt table: {len(task_prompts)} tasks",
            flush=True,
        )

    costs = {task: sum(int(r["horizon"]) for r in records) for task, records in by_task.items()}
    shard = shard_tasks_lpt(costs, args.worker_idx, args.num_workers)
    n_rollouts = sum(len(by_task[t]) for t in shard) * args.rollouts
    print(
        f"[w{args.worker_idx}/{args.num_workers}] manifest={sealed_sha[:12]} "
        f"file_sha={file_sha[:12]} tasks={shard} rollouts={n_rollouts} "
        f"cost={sum(costs[t] for t in shard) * args.rollouts} steps",
        flush=True,
    )
    if args.dry_run:
        return 0

    import gymnasium as gym
    import imageio

    # The ReMemBench fork registers the ``robocasa/<Task>`` gym ids at the BOTTOM of
    # gym_wrapper, not in robocasa/__init__ (stock RoboCasa v1.0.1 does the latter, which
    # is why eval_pi_05.py can get away with a bare ``import robocasa``). Import the
    # wrapper module explicitly or gym.make raises NamespaceNotFound.
    import robocasa  # noqa: F401
    import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers robocasa/<Task> ids)
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as wcp

    if not wait_for_port(args.host, args.port, args.server_wait_s):
        raise SystemExit(f"[w{args.worker_idx}] policy server {args.host}:{args.port} not up")
    client = wcp.WebsocketClientPolicy(args.host, args.port)

    # One long-lived env identity per worker: this worker drives exactly one environment at a
    # time (episodes and rollouts run strictly back-to-back), so the server should see one
    # env whose episode is reset by wsm_t==0, never a growing set of live identities.
    worker_env_id = f"rmb-w{args.worker_idx}of{args.num_workers}"

    for task in shard:
        records = by_task[task]
        out_path = os.path.join(args.out_dir, "remembench", task, f"stats_w{args.worker_idx}.json")
        if os.path.exists(out_path):
            print(f"[w{args.worker_idx}] skip {task} (stats exist)", flush=True)
            continue
        partial_path = out_path + ".partial"
        done_keys = {}
        if os.path.exists(partial_path):
            with open(partial_path) as handle:
                for item in json.load(handle)["per_episode"]:
                    done_keys[(int(item["episode_index"]), int(item["rollout"]))] = item
            print(
                f"[w{args.worker_idx}] resume {task}: {len(done_keys)} rollouts done",
                flush=True,
            )

        # enable_render=True is REQUIRED: the wrapper zeroes every camera image when it is
        # False (see get_basic_observation), which would silently feed the policy blank frames.
        env = gym.make(f"robocasa/{task}", enable_render=True, seed=int(manifest["base_seed"]))
        t_task = time.time()
        results = list(done_keys.values())
        try:
            for spec in records:
                for rollout_idx in range(args.rollouts):
                    key = (int(spec["episode_index"]), rollout_idx)
                    if key in done_keys:
                        continue
                    record_video = args.video == "first" and spec is records[0] and rollout_idx == 0
                    result, frames = run_episode(
                        env,
                        client,
                        spec,
                        rollout_idx,
                        args.replan_steps,
                        record_video,
                        wsm_prompt=None if task_prompts is None else task_prompts[task],
                        env_id=worker_env_id,
                        obs_image_size=args.obs_image_size,
                    )
                    results.append(result)
                    if frames:
                        video_dir = os.path.join(args.out_dir, "remembench", task, "videos")
                        os.makedirs(video_dir, exist_ok=True)
                        imageio.mimwrite(
                            os.path.join(
                                video_dir,
                                f"ep{result['episode_index']}_r{rollout_idx}_"
                                f"{'success' if result['success'] else 'failure'}.mp4",
                            ),
                            [image_tools.convert_to_uint8(f) for f in frames],
                            fps=10,
                        )
                    n_success = sum(1 for r in results if r["success"])
                    write_json_atomic(
                        partial_path,
                        {
                            "task": task,
                            "manifest_sha256": sealed_sha,
                            "complete": False,
                            "per_episode": results,
                        },
                    )
                    print(
                        f"[w{args.worker_idx}] {task} ep={result['episode_index']} "
                        f"r={rollout_idx} success={result['success']} "
                        f"failed_task={result['failed_task']} "
                        f"len={result['episode_length']}/{result['horizon']} "
                        f"({result['rollout_seconds']:.0f}s) "
                        f"[{n_success}/{len(results)}]",
                        flush=True,
                    )
        finally:
            env.close()

        expected = len(records) * args.rollouts
        if len(results) != expected:
            raise SystemExit(f"{task}: {len(results)} rollouts, expected {expected}")
        results.sort(key=lambda r: (r["episode_index"], r["rollout"]))
        payload = {
            "task": task,
            "category": get_remembench_category(task),
            "manifest_sha256": sealed_sha,
            "manifest_file_sha256": file_sha,
            "worker_idx": args.worker_idx,
            "num_workers": args.num_workers,
            "rollouts_per_episode": args.rollouts,
            "replan_steps": args.replan_steps,
            "complete": True,
            "num_episodes": len(records),
            "num_rollouts": len(results),
            "num_success": sum(1 for r in results if r["success"]),
            "success_rate": sum(1 for r in results if r["success"]) / len(results),
            "wall_seconds": round(time.time() - t_task, 1),
            "per_episode": results,
        }
        write_json_atomic(out_path, payload)
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(
            f"[w{args.worker_idx}] DONE {task}: "
            f"{payload['num_success']}/{payload['num_rollouts']} "
            f"({payload['success_rate'] * 100:.1f}%) in {payload['wall_seconds']:.0f}s",
            flush=True,
        )

    print(f"[w{args.worker_idx}] worker complete", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
