#!/usr/bin/env python3
"""GR00T N1.7 RoboCasa365 eval client (sim venv, torch-free) — ONE worker's shard of the TARGET set.

Per-shard zmq client: speaks the N1.7 PolicyServer protocol with a vendored minimal client
(importing gr00t.policy pulls torch, so we re-implement the REQ socket + msgpack_numpy serializer,
incl. the object-dtype/pickle guards). Recipe from the eval YAML (EvalConfigView, split=target);
runtime placement from the orchestrator (scripts/eval/eval.sh). Ported from
ported_raw/reference_code/eval_runner_groot.py.

Heavy sim/zmq imports are deferred so ``--dry-run`` (shard preview) works anywhere robocasa imports.

  python vla_training/eval/eval_groot_17.py --config scripts/configs/eval/groot17_eval.yaml \
      --worker-idx 0 --num-workers 8 --port 5600
  python ... --dry-run     # print this worker's task shard; no sim/server needed
"""

from __future__ import annotations

import argparse
import functools
import os
import time
import traceback

import numpy as np

from utils.config_schema import default_eval_config_path, load_eval_config
from vla_training.eval.eval_common import list_tasks, shard_tasks, stats_path, write_stats

BACKBONE = "groot_17"
VIDEO_KEYS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
STATE_KEYS = [
    "base_position",
    "base_rotation",
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
]
ACTION_KEYS = ["end_effector_position", "end_effector_rotation", "gripper_close", "base_motion", "control_mode"]
LANG_KEY = "annotation.human.task_description"


class ZmqPolicyClient:
    """Vendored minimal client for the N1.7 PolicyServer (REQ + msgpack_numpy, pickle-guarded)."""

    def __init__(self, host, port, timeout_ms=300_000):
        import msgpack
        import msgpack_numpy as mnp
        import zmq  # deferred: keeps dry-run import-light (sim venv has these)

        self._zmq, self._msgpack, self._mnp = zmq, msgpack, mnp
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self):
        zmq = self._zmq
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _to_bytes(self, data):
        mnp, msgpack = self._mnp, self._msgpack

        def _safe_encode(obj, chain=None):
            if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
                raise TypeError("Refusing to encode object-dtype ndarray (pickle surface).")
            return mnp.encode(obj, chain=chain)

        return msgpack.packb(data, default=functools.partial(_safe_encode, chain=lambda o: o))

    def _from_bytes(self, data):
        mnp, msgpack = self._mnp, self._msgpack

        def _safe_decode(obj, chain=None):
            if isinstance(obj, dict):
                nd_val = obj.get(b"nd", obj.get("nd"))
                kind_val = obj.get(b"kind", obj.get("kind"))
                if nd_val and kind_val in (b"O", "O"):
                    raise ValueError("Refusing to decode object-dtype ndarray payload (pickle-bearing).")
            return mnp.decode(obj, chain=chain)

        return msgpack.unpackb(data, object_hook=functools.partial(_safe_decode, chain=lambda o: o), raw=False)

    def _call(self, endpoint, data=None, requires_input=True):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self.socket.send(self._to_bytes(request))
            message = self.socket.recv()
        except self._zmq.error.Again:
            self.socket.close(linger=0)
            self._init_socket()
            raise
        if message == b"ERROR":
            raise RuntimeError("Server error (wrong policy server?)")
        response = self._from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self, timeout_ms=2_000):
        old = self.timeout_ms
        try:
            self.socket.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
            self._call("ping", requires_input=False)
            return True
        except (self._zmq.error.ZMQError, RuntimeError):
            self.socket.close(linger=0)
            self._init_socket()
            return False
        finally:
            self.timeout_ms = old
            try:
                self.socket.setsockopt(self._zmq.RCVTIMEO, old)
            except self._zmq.error.ZMQError:
                pass

    def get_action(self, observation):
        action, _info = self._call("get_action", {"observation": observation, "options": None})
        return action

    def reset(self, task: str) -> None:
        """Per-episode reset. For a WSM-conditioned server this clears the online-w_t causal buffer and
        sets the task language; for a baseline server it's a harmless policy reset. Best-effort so eval
        never dies if the server handles reset differently."""
        try:
            self._call("reset", {"options": {"task": task}})
        except Exception as e:
            print(f"[client] reset({task}) ignored: {e}", flush=True)


def wait_for_server(client, timeout_s: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if client.ping():
            return True
        time.sleep(5)
    return False


def pack_obs(obs):
    out = {}
    for k in VIDEO_KEYS:
        out[f"video.{k}"] = np.asarray(obs[f"video.{k}"], dtype=np.uint8)[None, None]
    for k in STATE_KEYS:
        out[f"state.{k}"] = np.asarray(obs[f"state.{k}"], dtype=np.float32)[None, None]
    out[LANG_KEY] = [str(obs[LANG_KEY])]
    return out


def chunk_to_steps(action, exec_steps):
    arrs = {}
    for k in ACTION_KEYS:
        a = np.asarray(action[f"action.{k}"], dtype=np.float64)
        if a.ndim == 3:
            a = a[0]
        arrs[k] = a
    n = min(exec_steps, min(a.shape[0] for a in arrs.values()))
    return [{f"action.{k}": arrs[k][i] for k in ACTION_KEYS} for i in range(n)]


def eval_task(entry, cfg, out_dir, worker_idx, client) -> None:
    import gymnasium as gym
    import imageio

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
        client.reset(
            task
        )  # new episode: clears the WSM online-w_t causal buffer + sets task language (no-op for baseline)
        record = cfg.video == "all" or (cfg.video == "first" and ep == 0)
        frames, done, t = [], False, 0
        while t < horizon and not done:
            steps = chunk_to_steps(client.get_action(pack_obs(obs)), cfg.exec_steps)
            for action in steps:
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(info["success"])
                if record and (t % 4 == 0 or done):
                    frames.append(np.ascontiguousarray(env.render()))
                t += 1
                if done or t >= horizon:
                    break
        successes.append(done)
        ep_lens.append(t)
        if record and frames:
            os.makedirs(video_dir, exist_ok=True)
            imageio.mimwrite(
                os.path.join(video_dir, f"ep{ep}_{'success' if done else 'failure'}.mp4"),
                [np.asarray(f, dtype=np.uint8) for f in frames],
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="eval YAML (default: scripts/configs/eval/groot17_eval.yaml)")
    ap.add_argument("--worker-idx", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=None, help="default: eval.num_workers")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--out-dir", default=None, help="default: eval.output_dir")
    ap.add_argument("--server-wait-s", type=int, default=2400)
    ap.add_argument("--dry-run", action="store_true", help="print this worker's shard; no sim/server")
    args = ap.parse_args()

    cfg = load_eval_config(args.config or default_eval_config_path(BACKBONE))
    num_workers = args.num_workers or cfg.num_workers
    out_dir = args.out_dir or cfg.output_dir or "results/groot17_eval"
    shard = shard_tasks(list_tasks(cfg.task_sets), args.worker_idx, num_workers)
    print(
        f"[w{args.worker_idx}/{num_workers}] split={cfg.split} trials={cfg.num_trials} "
        f"exec_steps={cfg.exec_steps} shard({len(shard)})={[e['task'] for e in shard]}",
        flush=True,
    )
    if args.dry_run:
        print("[eval_groot_17] dry-run OK (shard computed; skipping server connect + rollouts).")
        return

    import robocasa  # noqa: F401  (registers the gymnasium envs)

    client = ZmqPolicyClient(args.host, args.port)
    if not wait_for_server(client, args.server_wait_s):
        raise SystemExit(
            f"[w{args.worker_idx}] policy server {args.host}:{args.port} not up after {args.server_wait_s}s"
        )
    print(f"[w{args.worker_idx}] server ready on :{args.port}", flush=True)

    failures = []
    for entry in shard:
        try:
            eval_task(entry, cfg, out_dir, args.worker_idx, client)
        except Exception:
            failures.append(entry["task"])
            print(f"[w{args.worker_idx}] TASK FAILED {entry['task']}:\n{traceback.format_exc()}", flush=True)
    print(f"[w{args.worker_idx}] done. failures={failures}", flush=True)
    raise SystemExit(2 if failures and len(failures) == len(shard) else 0)


if __name__ == "__main__":
    main()
