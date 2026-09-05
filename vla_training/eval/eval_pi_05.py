#!/usr/bin/env python3
"""pi0.5 RoboCasa365 eval client (sim venv) — ONE worker's shard of the 50-task TARGET set.

Per-shard websocket client: connects to a pi0.5 policy server (one per GPU), runs its sharded
target tasks, writes per-task stats.json (resumable — skips tasks whose stats.json exists). The
RECIPE comes from the eval YAML (EvalConfigView, split=target, num_trials, video, seed, exec_steps);
RUNTIME placement (worker-idx / num-workers / host / port) comes from the orchestrator
(scripts/eval/eval.sh). Ported from ported_raw/reference_code/eval_runner_pi05.py.

Runs in the torch-free sim venv (robosuite+robocasa + openpi-client). Heavy sim/client imports are
deferred so ``--dry-run`` (shard preview) works anywhere robocasa imports.

  python vla_training/eval/eval_pi_05.py --config scripts/configs/eval/pi05_eval.yaml \
      --worker-idx 0 --num-workers 8 --port 8000
  python ... --dry-run     # print this worker's task shard; no sim/server needed
"""

from __future__ import annotations

import argparse
import collections
import os
import socket
import time
import traceback

import numpy as np

from utils.config_schema import default_eval_config_path, load_eval_config
from vla_training.eval.eval_common import list_tasks, shard_tasks, stats_path, write_stats
from vla_training.eval.eval_manifest import (
    build_episode_manifest,
    build_heldout_episode_manifest,
    build_shard_results,
    evaluation_provenance_from_run_manifest,
    load_episode_manifest,
    manifest_records_for_task,
    merge_task_episode_shards,
    output_lock,
    policy_noise_seed,
    sanitize_policy_timing,
    shard_episode_records,
    shard_stats_path,
    validate_shard_results,
    write_episode_manifest,
    write_json_atomic,
)

BACKBONE = "pi05"
RESIZE = 224  # pi0.5 RoboCasa input resolution (examples/robocasa/main.py)


def workspace_prompt_fields(task: str, task_prompts: dict[str, str] | None) -> dict[str, str]:
    """Return the private frozen-tap prompt without changing the action-policy prompt."""
    if task_prompts is None:
        return {}
    try:
        prompt = task_prompts[task]
    except KeyError as exc:
        raise ValueError(f"canonical workspace prompt is missing task {task!r}") from exc
    if not isinstance(prompt, str) or not prompt.strip() or prompt != prompt.strip():
        raise ValueError(f"canonical workspace prompt for {task!r} is invalid")
    return {"wsm_prompt": prompt}


def wait_for_port(host: str, port: int, timeout_s: int) -> bool:
    """Bounded readiness wait (openpi-client's WebsocketClientPolicy retries forever otherwise)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(5)
    return False


def eval_task(entry, cfg, out_dir, worker_idx, replan_steps, client) -> None:
    import gymnasium as gym
    import imageio
    from openpi_client import image_tools
    from robocasa.utils.env_utils import convert_action

    task, split_set, horizon = entry["task"], entry["split_set"], entry["horizon"]
    spath = stats_path(out_dir, split_set, task)
    if os.path.exists(spath):
        print(f"[w{worker_idx}] skip {task} (stats exist)", flush=True)
        return
    env = gym.make(f"robocasa/{task}", split=cfg.split, seed=cfg.seed)
    video_dir = os.path.join(os.path.dirname(spath), "videos")
    successes, ep_lens = [], []
    t_task = time.time()
    for ep in range(cfg.num_trials):
        obs, info = env.reset()
        lang = obs["annotation.human.task_description"]
        plan = collections.deque()
        record = cfg.video == "all" or (cfg.video == "first" and ep == 0)
        frames, done, t = [], False, 0
        while t < horizon:
            if not plan:
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        np.ascontiguousarray(obs["video.robot0_agentview_left"]), RESIZE, RESIZE
                    )
                )
                wrist = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(np.ascontiguousarray(obs["video.robot0_eye_in_hand"]), RESIZE, RESIZE)
                )
                right = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        np.ascontiguousarray(obs["video.robot0_agentview_right"]), RESIZE, RESIZE
                    )
                )
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
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/right_image": right,
                    "observation/state": state,
                    "prompt": lang,
                    # In-band WSM signaling (openpi has NO reset channel): the WSM server reads `wsm_t`
                    # (env step; t==0 => new episode -> reset the online-omega_t buffer + set task language)
                    # and `wsm_task` (gym task name, the task_lang_table key — NOT the per-episode prompt,
                    # which RoboCasa randomizes). A BASELINE pi server ignores these extra keys
                    # (RobocasaInputs only reads the keys it knows), so this stays a no-op for Eval1.
                    "wsm_t": t,
                    "wsm_task": task,
                    "wsm_env_id": f"w{worker_idx}-legacy",
                    "wsm_demo_episode": ep,
                }
                chunk = client.infer(element)["actions"]
                assert len(chunk) >= replan_steps, f"chunk {len(chunk)} < replan {replan_steps}"
                plan.extend(chunk[:replan_steps])
            action = convert_action(np.asarray(plan.popleft()))
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(info["success"])
            if record and (t % 4 == 0 or done):
                frames.append(np.ascontiguousarray(env.render()))
            if done:
                break
            t += 1
        successes.append(done)
        ep_lens.append(t + 1 if done else t)
        if record and frames:
            os.makedirs(video_dir, exist_ok=True)
            imageio.mimwrite(
                os.path.join(video_dir, f"ep{ep}_{'success' if done else 'failure'}.mp4"),
                [image_tools.convert_to_uint8(f) for f in frames],
                fps=10,
            )
        print(
            f"[w{worker_idx}] {task} ep {ep + 1}/{cfg.num_trials} success={done} "
            f"({sum(successes)}/{len(successes)}, {time.time() - t_task:.0f}s)",
            flush=True,
        )
    write_stats(
        spath,
        {
            "task": task,
            "split_set": split_set,
            "split": cfg.split,
            "num_episodes": len(successes),
            "success_rate": float(np.mean(successes)),
            "successes": [bool(s) for s in successes],
            "episode_lengths": ep_lens,
            "horizon": horizon,
            "seed": cfg.seed,
            "wall_seconds": round(time.time() - t_task, 1),
        },
    )
    try:
        env.env.close()
    except AttributeError:
        env.close()


def eval_task_manifest(
    entry,
    cfg,
    out_dir,
    worker_idx,
    replan_steps,
    client,
    manifest,
    records,
    episode_shard_idx,
    num_episode_shards,
    heldout_root,
    require_realized_batching,
    evaluation_provenance,
    task_prompts,
) -> None:
    """Evaluate one deterministic episode shard with resumable, race-free publication."""
    import gymnasium as gym
    import imageio
    from openpi_client import image_tools
    from robocasa.utils.env_utils import convert_action

    task, split_set, horizon = entry["task"], entry["split_set"], entry["horizon"]
    spath = shard_stats_path(out_dir, split_set, task, episode_shard_idx, num_episode_shards)
    partial_path = spath + ".partial"
    env_id = f"w{worker_idx}-episode-shard{episode_shard_idx}of{num_episode_shards}"

    with output_lock(spath):
        if os.path.exists(spath):
            import json

            with open(spath) as handle:
                finished = json.load(handle)
            validate_shard_results(
                finished,
                manifest,
                split_set,
                task,
                episode_shard_idx,
                num_episode_shards,
                require_complete=True,
                evaluation_provenance=evaluation_provenance,
            )
            if require_realized_batching and not finished.get("performance", {}).get("batching", {}).get(
                "realized_multi_request", False
            ):
                raise RuntimeError("K>1 canary observed only singleton policy gather batches")
            print(
                f"[w{worker_idx}] skip {task} shard {episode_shard_idx}of{num_episode_shards} (validated complete)",
                flush=True,
            )
            return

        completed = {}
        prior_wall_seconds = 0.0
        if os.path.exists(partial_path):
            import json

            with open(partial_path) as handle:
                partial = json.load(handle)
            validate_shard_results(
                partial,
                manifest,
                split_set,
                task,
                episode_shard_idx,
                num_episode_shards,
                require_complete=False,
                evaluation_provenance=evaluation_provenance,
            )
            completed = {int(result["episode_index"]): result for result in partial["per_episode"]}
            prior_wall_seconds = float(partial.get("wall_seconds", 0.0))
            print(
                f"[w{worker_idx}] resume {task} shard "
                f"{episode_shard_idx}of{num_episode_shards}: "
                f"{len(completed)}/{len(records)} complete",
                flush=True,
            )

        env = gym.make(f"robocasa/{task}", split=cfg.split, seed=cfg.seed)
        video_dir = os.path.join(
            os.path.dirname(spath),
            "videos",
            f"shard{episode_shard_idx}of{num_episode_shards}",
        )
        started = time.time()
        try:
            for position, spec in enumerate(records):
                episode_index = int(spec["episode_index"])
                if episode_index in completed:
                    continue
                reset_spec = spec["reset"]
                if not isinstance(reset_spec, dict):
                    raise ValueError(f"episode {task}/{episode_index} reset must be a mapping")

                episode_started = time.time()
                reset_kind = reset_spec.get("kind")
                if reset_kind == "gym_seed":
                    obs, info = env.reset(seed=int(spec["seed"]))
                elif reset_kind == "heldout_demo":
                    if not heldout_root:
                        raise ValueError("heldout_demo manifest requires --heldout-root")
                    expected_relpath = os.path.join(task, "extras", f"episode_{episode_index:06d}")
                    extras_relpath = os.path.normpath(os.fspath(reset_spec.get("extras_relpath", "")))
                    if os.path.isabs(extras_relpath) or extras_relpath != expected_relpath:
                        raise ValueError(
                            f"unsafe or mismatched heldout extras path "
                            f"{extras_relpath!r}; expected {expected_relpath!r}"
                        )
                    from vla_training.eval.heldout_reset import (
                        reset_gym_env_to_episode,
                    )

                    obs, info = reset_gym_env_to_episode(
                        env,
                        os.path.join(os.fspath(heldout_root), extras_relpath),
                        seed=int(spec["seed"]),
                        artifacts=reset_spec["artifacts"],
                    )
                elif reset_kind == "remembench_ep_meta":
                    # ReMemBench episodes are pinned by replaying the ep_meta recorded in
                    # the demo hdf5, carried inline in the manifest (no side artifacts).
                    # BOTH ep_meta and seed are required: ep_meta pins the object counts
                    # (MemPutK* draw them from the global np.random, which no seed
                    # reaches) and the object instances; the seed pins the placement
                    # sampling that ep_meta leaves to env.rng. Verified locally: ep_meta
                    # alone -> placements drift 0.15m between replays; seed alone ->
                    # counts drift; both -> bit-identical.
                    obs, info = env.reset(
                        seed=int(spec["seed"]),
                        options={"ep_meta": reset_spec["ep_meta"]},
                    )
                else:
                    raise ValueError(f"unsupported reset kind for {task}/{episode_index}: {reset_kind!r}")
                lang = obs["annotation.human.task_description"]
                plan = collections.deque()
                should_record = cfg.video == "all" or (
                    cfg.video == "first" and episode_shard_idx == 0 and position == 0
                )
                frames, done, t = [], False, 0
                policy_timing_calls = []
                while t < horizon:
                    if not plan:
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(
                                np.ascontiguousarray(obs["video.robot0_agentview_left"]),
                                RESIZE,
                                RESIZE,
                            )
                        )
                        wrist = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(
                                np.ascontiguousarray(obs["video.robot0_eye_in_hand"]),
                                RESIZE,
                                RESIZE,
                            )
                        )
                        right = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(
                                np.ascontiguousarray(obs["video.robot0_agentview_right"]),
                                RESIZE,
                                RESIZE,
                            )
                        )
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
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/right_image": right,
                            "observation/state": state,
                            "prompt": lang,
                            "wsm_t": t,
                            "wsm_task": task,
                            # Baseline servers ignore these. Stateful WSM servers must key
                            # episode state by wsm_env_id before K > 1 is enabled.
                            "wsm_env_id": env_id,
                            "wsm_demo_episode": episode_index,
                            # Explicit diffusion key makes π action noise invariant to server request
                            # order, gather batch composition, worker count, and episode sharding.
                            "policy_noise_seed": policy_noise_seed(int(spec["seed"]), t),
                            # Private canonical task language for the frozen representation tap.
                            # WSMPiInferWrapper strips this before the action-policy transform, so
                            # the stock policy continues to receive RoboCasa's normal annotation.
                            **workspace_prompt_fields(task, task_prompts),
                        }
                        request_started = time.perf_counter()
                        response = client.infer(element)
                        request_ms = (time.perf_counter() - request_started) * 1000.0
                        raw_timing = response.get("policy_timing")
                        raw_timing = dict(raw_timing) if isinstance(raw_timing, dict) else {}
                        raw_timing["client_roundtrip_ms"] = request_ms
                        timing = sanitize_policy_timing(raw_timing)
                        if timing:
                            policy_timing_calls.append(timing)
                        chunk = response["actions"]
                        assert len(chunk) >= replan_steps, f"chunk {len(chunk)} < replan {replan_steps}"
                        plan.extend(chunk[:replan_steps])
                    action = convert_action(np.asarray(plan.popleft()))
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(info["success"])
                    if should_record and (t % 4 == 0 or done):
                        frames.append(np.ascontiguousarray(env.render()))
                    if done:
                        break
                    t += 1

                episode_length = t + 1 if done else t
                result = {
                    "task": task,
                    "episode_index": episode_index,
                    "reset": spec["reset"],
                    "seed": int(spec["seed"]),
                    "success": bool(done),
                    "episode_length": int(episode_length),
                    "rollout_seconds": round(time.time() - episode_started, 1),
                    "policy_timing_calls": policy_timing_calls,
                }
                completed[episode_index] = result

                if should_record and frames:
                    os.makedirs(video_dir, exist_ok=True)
                    imageio.mimwrite(
                        os.path.join(
                            video_dir,
                            f"ep{episode_index}_{'success' if done else 'failure'}.mp4",
                        ),
                        [image_tools.convert_to_uint8(frame) for frame in frames],
                        fps=10,
                    )

                ordered_partial = [
                    completed[int(record["episode_index"])]
                    for record in records
                    if int(record["episode_index"]) in completed
                ]
                wall_seconds = prior_wall_seconds + time.time() - started
                partial = build_shard_results(
                    manifest,
                    split_set,
                    task,
                    episode_shard_idx,
                    num_episode_shards,
                    ordered_partial,
                    complete=False,
                    wall_seconds=wall_seconds,
                    evaluation_provenance=evaluation_provenance,
                )
                validate_shard_results(
                    partial,
                    manifest,
                    split_set,
                    task,
                    episode_shard_idx,
                    num_episode_shards,
                    require_complete=False,
                    evaluation_provenance=evaluation_provenance,
                )
                write_json_atomic(partial_path, partial)
                successes = sum(bool(item["success"]) for item in ordered_partial)
                print(
                    f"[w{worker_idx}] {task} manifest ep={episode_index} "
                    f"shard={episode_shard_idx}of{num_episode_shards} "
                    f"success={done} ({successes}/{len(ordered_partial)}, "
                    f"{wall_seconds:.0f}s)",
                    flush=True,
                )

            ordered = [completed[int(record["episode_index"])] for record in records]
            final = build_shard_results(
                manifest,
                split_set,
                task,
                episode_shard_idx,
                num_episode_shards,
                ordered,
                complete=True,
                wall_seconds=prior_wall_seconds + time.time() - started,
                evaluation_provenance=evaluation_provenance,
            )
            validate_shard_results(
                final,
                manifest,
                split_set,
                task,
                episode_shard_idx,
                num_episode_shards,
                require_complete=True,
                evaluation_provenance=evaluation_provenance,
            )
            write_json_atomic(spath, final)
            if require_realized_batching and not final["performance"]["batching"]["realized_multi_request"]:
                raise RuntimeError("K>1 canary observed only singleton policy gather batches")
            try:
                os.unlink(partial_path)
            except FileNotFoundError:
                pass
        finally:
            try:
                env.env.close()
            except AttributeError:
                env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="eval YAML (default: scripts/configs/eval/pi05_eval.yaml)")
    ap.add_argument("--worker-idx", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=None, help="default: eval.num_workers")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", default=None, help="default: eval.output_dir")
    ap.add_argument("--server-wait-s", type=int, default=2400)
    # Optional runtime overrides (the SageMaker eval entry drives these from env, like eval_runner_pi05);
    # each defaults to the YAML's eval block when omitted.
    ap.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="explicit action cadence override; Stage-S uses 8 to match WSM stride 8",
    )
    ap.add_argument("--num-trials", type=int, default=None, help="override eval.num_trials")
    ap.add_argument("--video", default=None, choices=[None, "none", "first", "all"], help="override eval.video")
    ap.add_argument("--seed", type=int, default=None, help="override eval.seed")
    ap.add_argument("--task-sets", default=None, help="override eval.task_sets (comma-separated names)")
    ap.add_argument("--tasks", default=None, help="restrict to this comma-list of tasks (small-task POC subset)")
    ap.add_argument("--dry-run", action="store_true", help="print this worker's shard; no sim/server")
    ap.add_argument(
        "--heldout-root",
        default=None,
        help="root with <Task>/{heldout.json,extras/}; selects exact demo resets",
    )
    ap.add_argument(
        "--rollouts-per-demo",
        type=int,
        default=1,
        help="compatibility flag; decisive manifest protocol requires exactly 1",
    )
    ap.add_argument(
        "--episode-manifest",
        default=None,
        help="immutable (task, episode_index, reset, seed) manifest",
    )
    ap.add_argument(
        "--eval-run-manifest",
        default=None,
        help=(
            "sealed Stage-S eval run manifest; binds every exact shard/stat to its "
            "arm, training run, checkpoint tree, and eval identity"
        ),
    )
    ap.add_argument(
        "--task-prompt-manifest",
        default=None,
        help=(
            "validated demo-independent task→prompt manifest used only by the frozen "
            "workspace tap; focused S1/S2 jobs require it"
        ),
    )
    ap.add_argument(
        "--write-episode-manifest",
        default=None,
        help="write a procedural or held-out-reset manifest from this config and exit",
    )
    ap.add_argument("--episode-shard-idx", type=int, default=0)
    ap.add_argument("--num-episode-shards", type=int, default=1)
    ap.add_argument(
        "--server-state-mode",
        choices=[
            "stateless",
            "per_env_isolated",
            "stateless_v1",
            "per_env_isolated_v1",
        ],
        default=None,
        help="declared policy-server state contract; Stage-S uses a capability-checked *_v1 mode",
    )
    ap.add_argument(
        "--merge-episode-shards",
        action="store_true",
        help="validate/merge this worker's task shards and exit; requires --episode-manifest",
    )
    ap.add_argument(
        "--require-realized-batching",
        action="store_true",
        help="fail a K>1 shard if every realized server gather batch is singleton",
    )
    args = ap.parse_args()

    cfg = load_eval_config(args.config or default_eval_config_path(BACKBONE))
    overrides = {}
    if args.replan_steps is not None:
        if args.replan_steps < 1:
            raise SystemExit("--replan-steps must be >= 1")
        overrides["replan_steps"] = args.replan_steps
    if args.num_trials is not None:
        overrides["num_trials"] = args.num_trials
    if args.video is not None:
        overrides["video"] = args.video
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.task_sets is not None:
        overrides["task_sets"] = [t.strip() for t in args.task_sets.split(",") if t.strip()]
    if overrides:
        import dataclasses

        cfg = dataclasses.replace(cfg, **overrides)
    num_workers = args.num_workers or cfg.num_workers
    out_dir = args.out_dir or cfg.output_dir or "results/pi05_eval"
    replan_steps = cfg.replan_steps
    # --tasks restricts execution to a POC subset; prompt provenance is still validated against the
    # complete task universe so a subset can never bless a partial prompt table.
    only = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    tasks = list_tasks(cfg.task_sets, only=only)
    task_prompts = None
    if args.task_prompt_manifest:
        from scripts.launch.validate_stage_s_task_prompts import load_task_prompts

        task_universe = list_tasks(cfg.task_sets)
        expected_names = {entry["task"] for entry in task_universe}
        task_prompts = load_task_prompts(
            args.task_prompt_manifest,
            expected_task_names=expected_names,
            expected_tasks=len(expected_names),
        )
        print(
            f"[eval_pi_05] canonical workspace prompts loaded: {len(task_prompts)} tasks",
            flush=True,
        )

    if args.heldout_root and args.rollouts_per_demo != 1:
        raise SystemExit(
            "the immutable 50x100 protocol requires --rollouts-per-demo=1; "
            f"got {args.rollouts_per_demo}. Use the legacy internal runner only to "
            "resume the older multi-rollout held-out campaign."
        )
    if args.episode_manifest and args.write_episode_manifest:
        raise SystemExit("--episode-manifest and --write-episode-manifest are mutually exclusive")
    if args.eval_run_manifest and not args.episode_manifest:
        raise SystemExit("--eval-run-manifest requires --episode-manifest")

    def make_manifest():
        if args.heldout_root:
            return build_heldout_episode_manifest(
                tasks,
                args.heldout_root,
                cfg.num_trials,
                cfg.seed,
                split=cfg.split,
                task_sets=cfg.task_sets,
            )
        return build_episode_manifest(
            tasks,
            cfg.num_trials,
            cfg.seed,
            split=cfg.split,
            task_sets=cfg.task_sets,
        )

    if args.write_episode_manifest:
        manifest = make_manifest()
        write_episode_manifest(args.write_episode_manifest, manifest)
        print(
            f"[eval_pi_05] immutable manifest -> {args.write_episode_manifest} "
            f"sha256={manifest['manifest_sha256']} episodes={len(manifest['episodes'])}",
            flush=True,
        )
        return

    if args.episode_manifest:
        manifest = load_episode_manifest(args.episode_manifest)
    elif args.heldout_root:
        # Active SageMaker compatibility: all task workers independently derive identical bytes and
        # atomically no-clobber the same run-local manifest before any server connection or rollout.
        manifest = make_manifest()
        auto_manifest_path = os.path.join(out_dir, "episode_manifest.json")
        write_episode_manifest(auto_manifest_path, manifest)
        print(
            f"[eval_pi_05] held-out manifest -> {auto_manifest_path} "
            f"sha256={manifest['manifest_sha256']} "
            f"demos={len(manifest['episodes'])}",
            flush=True,
        )
    else:
        manifest = None
    evaluation_provenance = None
    if args.eval_run_manifest:
        evaluation_provenance = evaluation_provenance_from_run_manifest(
            args.eval_run_manifest,
            manifest,
            episode_manifest_path=args.episode_manifest,
        )
    if args.require_realized_batching and args.num_episode_shards <= 1:
        raise SystemExit("--require-realized-batching is meaningful only with K>1 episode shards")
    if manifest is None:
        if args.num_episode_shards != 1 or args.episode_shard_idx != 0:
            raise SystemExit(
                "episode sharding requires --episode-manifest; refusing duplicated "
                "legacy rollouts and shared stats.json writes"
            )
        if args.merge_episode_shards:
            raise SystemExit("--merge-episode-shards requires --episode-manifest")
    else:
        if manifest["split"] != cfg.split:
            raise SystemExit(f"manifest split={manifest['split']!r} != config split={cfg.split!r}")
        if int(manifest.get("base_seed", -1)) != int(cfg.seed):
            raise SystemExit(f"manifest base_seed={manifest.get('base_seed')!r} != config seed={cfg.seed}")
        if list(manifest.get("task_sets", [])) != list(cfg.task_sets):
            raise SystemExit(f"manifest task_sets={manifest.get('task_sets')!r} != config {list(cfg.task_sets)!r}")
        selected_names = {entry["task"] for entry in tasks}
        manifest_names = {record["task"] for record in manifest["episodes"]}
        missing = selected_names - manifest_names
        extra = manifest_names - selected_names
        if missing or (only is None and extra):
            raise SystemExit(
                f"manifest/task-universe mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra) if only is None else 'allowed by --tasks'}"
            )
        for entry in tasks:
            records = manifest_records_for_task(manifest, entry["task"])
            if len(records) != cfg.num_trials:
                raise SystemExit(
                    f"manifest has {len(records)} episodes for {entry['task']}; "
                    f"config requires --num-trials={cfg.num_trials}"
                )
            for record in records:
                if record["split_set"] != entry["split_set"]:
                    raise SystemExit(
                        f"manifest split_set mismatch for {entry['task']}: "
                        f"{record['split_set']!r} != {entry['split_set']!r}"
                    )
                if int(record["horizon"]) != int(entry["horizon"]):
                    raise SystemExit(
                        f"manifest horizon mismatch for {entry['task']}: {record['horizon']} != {entry['horizon']}"
                    )

        selected_reset_kinds = {
            record["reset"].get("kind")
            for record in manifest["episodes"]
            if record["task"] in selected_names and isinstance(record.get("reset"), dict)
        }
        if args.heldout_root and selected_reset_kinds != {"heldout_demo"}:
            raise SystemExit(
                f"--heldout-root requires only heldout_demo resets; manifest has {sorted(selected_reset_kinds)}"
            )
        if (
            not args.heldout_root
            and "heldout_demo" in selected_reset_kinds
            and not args.merge_episode_shards
            and not args.dry_run
        ):
            raise SystemExit("manifest contains heldout_demo resets but --heldout-root is missing")

    if (
        manifest is not None
        and args.num_episode_shards > 1
        and not args.merge_episode_shards
        and not args.dry_run
        and args.server_state_mode is None
    ):
        raise SystemExit(
            "K>1 rollout requires --server-state-mode=stateless or "
            "--server-state-mode=per_env_isolated; refusing ambiguous singleton state"
        )

    shard = shard_tasks(tasks, args.worker_idx, num_workers)
    print(
        f"[w{args.worker_idx}/{num_workers}] split={cfg.split} "
        f"trials={cfg.num_trials} task_shard({len(shard)})="
        f"{[entry['task'] for entry in shard]}",
        flush=True,
    )
    if manifest is not None:
        shard_counts = {
            entry["task"]: len(
                shard_episode_records(
                    manifest_records_for_task(manifest, entry["task"]),
                    args.episode_shard_idx,
                    args.num_episode_shards,
                )
            )
            for entry in shard
        }
        print(
            f"[w{args.worker_idx}] episode_shard="
            f"{args.episode_shard_idx}of{args.num_episode_shards} "
            f"manifest={manifest['manifest_sha256']} counts={shard_counts}",
            flush=True,
        )

    if args.merge_episode_shards:
        for entry in shard:
            merged = merge_task_episode_shards(
                out_dir,
                manifest,
                entry["split_set"],
                entry["task"],
                args.num_episode_shards,
                evaluation_provenance=evaluation_provenance,
            )
            print(
                f"[w{args.worker_idx}] merged {entry['task']}: "
                f"{merged['num_episodes']} episodes, "
                f"success={merged['success_rate']:.4f}",
                flush=True,
            )
        return
    if args.dry_run:
        print("[eval_pi_05] dry-run OK (no server connect or rollouts).")
        return

    import robocasa  # noqa: F401  (registers the gymnasium envs)
    from openpi_client import websocket_client_policy as _wcp

    if not wait_for_port(args.host, args.port, args.server_wait_s):
        raise SystemExit(
            f"[w{args.worker_idx}] policy server {args.host}:{args.port} not up after {args.server_wait_s}s"
        )
    client = _wcp.WebsocketClientPolicy(args.host, args.port)

    if args.server_state_mode is not None:
        metadata = dict(client.get_server_metadata() or {})
        expected_mode = {
            "stateless": "stateless_v1",
            "per_env_isolated": "per_env_isolated_v1",
        }.get(args.server_state_mode, args.server_state_mode)
        problems = []
        if metadata.get("server_state_mode") != expected_mode:
            problems.append(f"server_state_mode={metadata.get('server_state_mode')!r}, expected {expected_mode!r}")
        if metadata.get("infer_batch") is not True:
            problems.append(f"infer_batch={metadata.get('infer_batch')!r}, expected true")
        if args.num_episode_shards > 1:
            if metadata.get("server_concurrent") is not True:
                problems.append(f"server_concurrent={metadata.get('server_concurrent')!r}, expected true")
            if int(metadata.get("server_batch_envs", -1)) != args.num_episode_shards:
                problems.append(
                    f"server_batch_envs={metadata.get('server_batch_envs')!r}, expected {args.num_episode_shards}"
                )
        if expected_mode == "per_env_isolated_v1":
            if metadata.get("wsm_state_mode") != expected_mode:
                problems.append(f"wsm_state_mode={metadata.get('wsm_state_mode')!r}, expected {expected_mode!r}")
            if int(metadata.get("wsm_max_envs", -1)) < args.num_episode_shards:
                problems.append(f"wsm_max_envs={metadata.get('wsm_max_envs')!r}, need >= {args.num_episode_shards}")
            if int(metadata.get("wsm_stride", -1)) != replan_steps:
                problems.append(f"wsm_stride={metadata.get('wsm_stride')!r} != replan_steps={replan_steps}")
            required = {
                "wsm_env_id",
                "wsm_task",
                "wsm_demo_episode",
                "wsm_t",
            }
            advertised = set(metadata.get("wsm_required_identity_fields", []))
            if not required.issubset(advertised):
                problems.append(f"identity_fields missing {sorted(required - advertised)}")
            required_signals = set(metadata.get("wsm_required_signal_fields", []))
            if task_prompts is not None and "wsm_prompt" not in required_signals:
                problems.append(
                    "canonical task prompts were supplied but the server does not "
                    "advertise required wsm_prompt consumption"
                )
            if "wsm_prompt" in required_signals and task_prompts is None:
                problems.append("server requires wsm_prompt but --task-prompt-manifest is missing")
        elif metadata.get("wsm_state_mode") is not None:
            problems.append("stateless arm unexpectedly advertises workspace trajectory state")
        if problems:
            raise SystemExit(f"[w{args.worker_idx}] server capability check failed: " + "; ".join(problems))
        print(
            f"[w{args.worker_idx}] server capability OK: mode={expected_mode} "
            f"K={args.num_episode_shards} infer_batch=true",
            flush=True,
        )

    failures = []
    for entry in shard:
        try:
            if manifest is None:
                eval_task(entry, cfg, out_dir, args.worker_idx, replan_steps, client)
            else:
                records = shard_episode_records(
                    manifest_records_for_task(manifest, entry["task"]),
                    args.episode_shard_idx,
                    args.num_episode_shards,
                )
                eval_task_manifest(
                    entry,
                    cfg,
                    out_dir,
                    args.worker_idx,
                    replan_steps,
                    client,
                    manifest,
                    records,
                    args.episode_shard_idx,
                    args.num_episode_shards,
                    args.heldout_root,
                    args.require_realized_batching,
                    evaluation_provenance,
                    task_prompts,
                )
                if args.num_episode_shards == 1:
                    merged = merge_task_episode_shards(
                        out_dir,
                        manifest,
                        entry["split_set"],
                        entry["task"],
                        1,
                        evaluation_provenance=evaluation_provenance,
                    )
                    print(
                        f"[w{args.worker_idx}] merged {entry['task']}: "
                        f"{merged['num_episodes']} episodes, "
                        f"success={merged['success_rate']:.4f}",
                        flush=True,
                    )
        except Exception:
            failures.append(entry["task"])
            print(f"[w{args.worker_idx}] TASK FAILED {entry['task']}:\n{traceback.format_exc()}", flush=True)
    print(f"[w{args.worker_idx}] done. failures={failures}", flush=True)
    if failures and manifest is not None:
        # Decisive manifest runs require exact task coverage; any isolated task failure is fatal.
        raise SystemExit(2)
    # Preserve the historical non-manifest runner's all-tasks-failed exit convention.
    raise SystemExit(2 if failures and len(failures) == len(shard) else 0)


if __name__ == "__main__":
    main()
