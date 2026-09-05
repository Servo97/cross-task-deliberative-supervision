#!/usr/bin/env python3
"""Turn raw rollout / teacher-forcing / expert-probe recordings into the study's metrics.

Writes one JSON per rollout (all curves, so nothing has to be recomputed from sim) plus the
single master CSV the eyeballing session is driven from.

Metrics, in the order the brief asks for them:

(a) TEACHER-FORCED ACTION DEVIATION. The policy is queried at every 8th step ALONG THE
    EXPERT TRAJECTORY (no closed loop), and its 8-action chunk is scored against the
    expert's actions over the same window. Per-dimension squared error is normalised by the
    per-dimension expert action standard deviation POOLED OVER ALL 20 RESETS OF THAT TASK —
    one fixed scaler per task, shared by every checkpoint, so the numbers are comparable
    across arms. Units: multiples of expert action variance. Reported as mean, 90th
    percentile, peak, and the fraction of the way through the episode at which the peak
    occurs (``argmax_frac``) — an early peak means the policy diverged from the expert's
    intent immediately; a late peak means it tracked and then lost it.

    The p90 exists because the MEAN is measurably insensitive: on MemWashAndReturnLeft the
    never-finetuned pretrain-150k checkpoint scores 0/20 successes yet has a mean
    (0.557) indistinguishable from the finetuned baseline (0.530) and deltanet-w8 (0.487).
    Most of a manipulation trajectory is low-variance transit that any checkpoint predicts,
    so a large disagreement confined to the decisive few steps is averaged away. Prefer p90
    and peak over the mean when judging whether this metric separates arms at all.

(b) FREE-ROLLOUT DIVERGENCE. Closed-loop end-effector world position against the expert's,
    aligned by timestep. Reported as the L2 curve, its DTW distance (banded, mean cost per
    aligned pair, so episode length does not inflate it), and FIRST-DIVERGENCE STEP: the
    first t at which the deviation exceeds 10 cm and stays above it for 10 consecutive
    control steps (0.5 s at 20 Hz). 10 cm is roughly a gripper width — smaller deviations
    are ordinary control noise around the same intent; 0.5 s of sustained excess is not.
    Normalised by expert length, so 0.0 means "wrong from the start" and 1.0 means "never".

(c) TARGET COMMITMENT — the memory-vs-control discriminator. See ``fm_targets`` for how
    correct and distractor were derived from each task's own source. Labels:
      approached_correct : went to the right target  -> a failure here is a CONTROL failure
      approached_wrong   : went to a distractor      -> a MEMORY/CONTEXT failure
      no_commitment      : never settled on either   -> collapse

(d) OUTCOME. success, plus ReMemBench's ``failed_task`` (a blown prospective deadline is a
    hard failure and can never be scored a success).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm_targets  # noqa: E402
from fm_common import (  # noqa: E402
    APPROACH_CONSECUTIVE_K,
    APPROACH_RADIUS_M,
    DIVERGENCE_CONSECUTIVE_K,
    DIVERGENCE_THRESHOLD_M,
    cell_name,
    dtw_distance,
    first_divergence_step,
    quat_geodesic,
    write_json_atomic,
)

CSV_COLUMNS = [
    "bench",
    "task",
    "ckpt",
    "reset_id",
    "rollout_index",
    "split",
    "success",
    "failed_task",
    "target_commitment",
    "commit_step_correct",
    "commit_step_wrong",
    "first_divergence_step",
    "first_divergence_frac",
    "agent_travel_max",
    "tf_mse_mean",
    "tf_mse_p90",
    "tf_mse_peak",
    "tf_mse_argmax_frac",
    "dtw",
    "ee_dev_mean",
    "ee_dev_final",
    "rot_dev_mean",
    "episode_length",
    "expert_length",
    "video_path",
]


def action_scaler(manifest, task) -> np.ndarray:
    """Per-dimension expert action std pooled over the task's 20 resets. Fixed per task."""
    chunks = []
    for episode in manifest["episodes"]:
        if episode["task"] != task:
            continue
        chunks.append(np.load(episode["expert"]["actions"]))
    pooled = np.concatenate(chunks, axis=0)
    std = pooled.std(axis=0)
    # Dimensions the expert never moves (e.g. a constant control-mode flag) would divide by
    # zero; leave them unscaled rather than exploding.
    std[std < 1e-6] = 1.0
    return std


def discover_draws(root, bench, task, ckpt, reset_id_value):
    """Which diffusion-noise draws exist on disk for this (arm, reset). Always sorted."""
    found = []
    for index in range(0, 16):
        path = os.path.join(root, "raw", bench, task, ckpt, cell_name(reset_id_value, index), "result.json")
        if os.path.exists(path):
            found.append(index)
    return found


def load_cell(root, bench, task, ckpt, reset_id):
    cell = os.path.join(root, "raw", bench, task, ckpt, reset_id)
    result_path = os.path.join(cell, "result.json")
    if not os.path.exists(result_path):
        return None
    with open(result_path) as handle:
        result = json.load(handle)
    payload = {"result": result, "dir": cell}
    for name, key in (("traj.npz", "traj"), ("tf.npz", "tf")):
        path = os.path.join(cell, name)
        if os.path.exists(path):
            with np.load(path) as data:
                # `states` is the raw MuJoCo state trajectory -- megabytes per rollout, used
                # only by the renderer. Decompressing it here would dominate the metrics pass
                # for data nothing below ever reads.
                payload[key] = {k: data[k] for k in data.files if k != "states"}
    return payload


#: The tracked agent must first travel this far from where it started before any commitment
#: is allowed to register. Without it the metric is degenerate on tasks whose agent BEGINS
#: at its correct target — MemWashAndReturn's fruit starts inside the correct container, and
#: ScrubCuttingBoard's sponge is placed in the cutting board's own region — so step 0 would
#: score "approached_correct" for every arm including a policy that never moves.
DEPARTURE_M = 0.10


def commitment(task, arrays, result):
    """Target-commitment label plus the two commit steps."""
    spec = fm_targets.TASK_SPECS[task]
    correct = np.asarray(arrays.get("d_correct", []), dtype=np.float64)
    wrong = np.asarray(arrays.get("d_distractor", []), dtype=np.float64)
    agent = np.asarray(arrays.get("agent_pos", []), dtype=np.float64)

    meta = result.get("rollout__target_meta") or result.get("expert_probe__target_meta") or {}

    if spec.get("commitment_kind") == "knob_state":
        label, t_correct, t_wrong = fm_targets.classify_knobs(
            meta.get("stove_locations") or [],
            np.asarray(arrays.get("knobs", []), dtype=np.float64),
            meta.get("correct_knob"),
            consecutive=APPROACH_CONSECUTIVE_K,
        )
        return (
            label,
            t_correct,
            t_wrong,
            {
                "kind": "knob_state",
                "correct_knob": meta.get("correct_knob"),
                "stove_locations": meta.get("stove_locations"),
                "note": "burner-distance curves are stored as diagnostics but do not drive this label",
            },
        )

    gate_key = spec.get("phase_gate")
    gate = np.ones(len(correct), dtype=bool)
    departed_step = -1
    if len(agent):
        travelled = np.linalg.norm(agent - agent[0], axis=1)
        departed = np.maximum.accumulate(travelled) > DEPARTURE_M
        departed_step = int(np.argmax(departed)) if departed.any() else -1
        gate &= departed
    if gate_key:
        series = arrays.get(f"ts_{gate_key}")
        if series is not None:
            gate &= np.asarray(series, dtype=np.float64) > 0.5

    label, t_correct, t_wrong = fm_targets.classify(
        correct, wrong, radius=APPROACH_RADIUS_M, consecutive=APPROACH_CONSECUTIVE_K, gate=gate
    )
    detail = {
        "gate_key": gate_key,
        "gate_steps": int(gate.sum()),
        "departure_m": DEPARTURE_M,
        "departed_step": departed_step,
    }

    return label, t_correct, t_wrong, detail


def prospective_fields(arrays):
    """MemHeatPot's timing half: did it turn on, wait long enough, and turn off in time."""
    out = {}
    for key in (
        "turn_on_stove_success",
        "turn_off_stove_success",
        "stove_wait_timer",
        "stove_wait_timer_threshold",
        "stove_wait_timer_max_threshold",
        "place_success",
        "final_success",
    ):
        series = arrays.get(f"ts_{key}")
        if series is None:
            continue
        series = np.asarray(series, dtype=np.float64)
        out[key] = {"final": float(series[-1]), "max": float(np.nanmax(series))}
    return out


def compute(manifest, root, bench, task, ckpt, episode, scaler, video_root, rollout_idx=0):
    reset_id = cell_name(episode["reset_id"], rollout_idx)
    cell = load_cell(root, bench, task, ckpt, reset_id)
    # The expert is a property of the RESET, not of a draw: every draw differences against
    # the same demonstration, which is what keeps the arms paired across draws.
    expert = load_cell(root, bench, task, "expert", episode["reset_id"])
    if cell is None or "traj" not in cell or expert is None or "traj" not in expert:
        return None

    result = cell["result"]
    arrays = cell["traj"]
    expert_arrays = expert["traj"]

    rollout_ee = np.asarray(arrays["ee_pos"], dtype=np.float64)
    expert_ee = np.asarray(expert_arrays["ee_pos"], dtype=np.float64)
    n = int(min(len(rollout_ee), len(expert_ee)))
    deviation = np.linalg.norm(rollout_ee[:n] - expert_ee[:n], axis=1)
    rotation = np.array(
        [quat_geodesic(arrays["ee_quat"][i], expert_arrays["ee_quat"][i]) for i in range(n)],
        dtype=np.float64,
    )
    divergence = first_divergence_step(deviation)
    expert_length = int(len(expert_ee))
    dtw = dtw_distance(rollout_ee, expert_ee)

    agent_pos = np.asarray(arrays.get("agent_pos", []), dtype=np.float64)
    # How far the tracked agent EVER got from where it started. Separates the two very
    # different things that both land in `no_commitment`: a policy that moved decisively but
    # settled on nothing, and one that barely moved at all. Measured on
    # MemFruitInSinkRightFar the rmb baseline travels a median 0.22 m against the expert's
    # 1.19 m -- it freezes rather than choosing wrongly.
    travel_max = float(np.max(np.linalg.norm(agent_pos - agent_pos[0], axis=1))) if len(agent_pos) else float("nan")

    label, t_correct, t_wrong, commit_detail = commitment(task, arrays, result)
    expert_label, _ec, _ew, _ed = commitment(task, expert_arrays, expert["result"])

    tf_mean = tf_p90 = tf_peak = tf_frac = float("nan")
    tf_curve = None
    if cell.get("tf") is not None and "tf_se" in cell["tf"]:
        squared = np.asarray(cell["tf"]["tf_se"], dtype=np.float64)
        normalised = squared / (scaler**2)[None, :]
        tf_curve = np.nanmean(normalised, axis=1)
        finite = np.isfinite(tf_curve)
        if finite.any():
            tf_mean = float(np.nanmean(tf_curve[finite]))
            # p90 as well as the mean: most of a manipulation trajectory is low-variance
            # transit that any checkpoint predicts, so the mean can be dominated by the easy
            # majority and hide a large disagreement confined to the decisive few steps.
            # Measured on MemWashAndReturnLeft the never-finetuned checkpoint (0/20 success)
            # has a MEAN indistinguishable from the finetuned arms.
            tf_p90 = float(np.nanpercentile(tf_curve[finite], 90))
            tf_peak = float(np.nanmax(tf_curve[finite]))
            tf_frac = float(np.nanargmax(np.where(finite, tf_curve, -np.inf)) / max(len(tf_curve) - 1, 1))

    video_path = os.path.join(video_root, bench, task, ckpt, f"{reset_id}.mp4")
    record = {
        "bench": bench,
        "task": task,
        "ckpt": ckpt,
        "reset_id": episode["reset_id"],
        "rollout_index": int(rollout_idx),
        "split": episode["split"],
        "episode_index": episode["episode_index"],
        "seed": episode["seed"],
        "lang": episode["expert"].get("lang"),
        "outcome": {
            "success": bool(result.get("rollout__success")),
            "raw_success": bool(result.get("rollout__raw_success")),
            "failed_task": bool(result.get("rollout__failed_task")),
            "episode_length": int(result.get("rollout__episode_length", n)),
            "horizon": episode["horizon"],
            "expert_length": expert_length,
        },
        "teacher_forced": {
            "mse_mean": tf_mean,
            "mse_p90": tf_p90,
            "mse_peak": tf_peak,
            "mse_argmax_frac": tf_frac,
            "normalisation": "per-dim expert action std pooled over the task's 20 resets",
            "action_std": scaler.tolist(),
            "curve": None if tf_curve is None else tf_curve.tolist(),
        },
        "divergence": {
            "threshold_m": DIVERGENCE_THRESHOLD_M,
            "consecutive_steps": DIVERGENCE_CONSECUTIVE_K,
            "first_divergence_step": divergence,
            "first_divergence_frac": (1.0 if divergence < 0 else float(divergence) / max(expert_length, 1)),
            "ee_deviation_mean": float(np.mean(deviation)) if n else float("nan"),
            "ee_deviation_final": float(deviation[-1]) if n else float("nan"),
            "rotation_deviation_mean_rad": float(np.mean(rotation)) if n else float("nan"),
            "dtw": dtw,
            "curve": deviation.tolist(),
        },
        "target_commitment": {
            "label": label,
            "commit_step_correct": t_correct,
            "commit_step_wrong": t_wrong,
            "radius_m": APPROACH_RADIUS_M,
            "consecutive_steps": APPROACH_CONSECUTIVE_K,
            "expert_label": expert_label,
            "agent_travel_max": travel_max,
            "detail": commit_detail,
            "meta": result.get("rollout__target_meta"),
            "d_correct": np.asarray(arrays["d_correct"]).tolist(),
            "d_distractor": np.asarray(arrays["d_distractor"]).tolist(),
        },
        "prospective": prospective_fields(arrays),
        "video_path": video_path,
    }
    row = {
        "bench": bench,
        "task": task,
        "ckpt": ckpt,
        "reset_id": episode["reset_id"],
        "rollout_index": int(rollout_idx),
        "split": episode["split"],
        "success": int(record["outcome"]["success"]),
        "failed_task": int(record["outcome"]["failed_task"]),
        "target_commitment": label,
        "commit_step_correct": t_correct,
        "commit_step_wrong": t_wrong,
        "first_divergence_step": divergence,
        "first_divergence_frac": round(record["divergence"]["first_divergence_frac"], 4),
        "agent_travel_max": _round(travel_max),
        "tf_mse_mean": _round(tf_mean),
        "tf_mse_p90": _round(tf_p90),
        "tf_mse_peak": _round(tf_peak),
        "tf_mse_argmax_frac": _round(tf_frac),
        "dtw": _round(dtw),
        "ee_dev_mean": _round(record["divergence"]["ee_deviation_mean"]),
        "ee_dev_final": _round(record["divergence"]["ee_deviation_final"]),
        "rot_dev_mean": _round(record["divergence"]["rotation_deviation_mean_rad"]),
        "episode_length": record["outcome"]["episode_length"],
        "expert_length": expert_length,
        "video_path": video_path,
    }
    return record, row


def _round(value, digits=5):
    return "" if value is None or not np.isfinite(value) else round(float(value), digits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--out-root", required=True, help="where per-rollout JSON + CSV land")
    parser.add_argument("--ckpts", required=True, help="comma list of checkpoint labels")
    parser.add_argument("--csv-name", default="master.csv")
    args = parser.parse_args()

    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    rows = []
    for manifest_path in args.manifest:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        bench = manifest["bench"]
        for task in manifest["tasks"]:
            scaler = action_scaler(manifest, task)
            episodes = [e for e in manifest["episodes"] if e["task"] == task]
            episodes.sort(key=lambda e: (e["split"] != "heldout", int(e["episode_index"])))
            for ckpt in ckpts:
                for episode in episodes:
                    for draw in discover_draws(args.root, bench, task, ckpt, episode["reset_id"]):
                        computed = compute(
                            manifest,
                            args.root,
                            bench,
                            task,
                            ckpt,
                            episode,
                            scaler,
                            args.video_root,
                            rollout_idx=draw,
                        )
                        if computed is None:
                            continue
                        record, row = computed
                        name = cell_name(episode["reset_id"], draw)
                        write_json_atomic(
                            os.path.join(args.out_root, bench, task, ckpt, f"{name}.json"),
                            record,
                        )
                        rows.append(row)

    csv_path = os.path.join(args.out_root, args.csv_name)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[metrics] {len(rows)} rollouts -> {csv_path}", flush=True)
    _summary(rows)
    return 0


def _summary(rows):
    if not rows:
        return
    import collections

    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["bench"], row["task"], row["ckpt"])].append(row)
    print(
        f"\n{'bench':<11}{'task':<26}{'ckpt':<22}{'n':>3} {'succ%':>6} "
        f"{'correct/wrong/none':>20} {'med_first_div':>14}"
    )
    for key in sorted(grouped):
        group = grouped[key]
        n = len(group)
        success = 100.0 * sum(r["success"] for r in group) / n
        labels = collections.Counter(r["target_commitment"] for r in group)
        divergences = [r["first_divergence_step"] for r in group if r["first_divergence_step"] >= 0]
        median = int(np.median(divergences)) if divergences else -1
        print(
            f"{key[0]:<11}{key[1]:<26}{key[2]:<22}{n:>3} {success:>6.1f} "
            f"{labels['approached_correct']:>6}/{labels['approached_wrong']:>6}/"
            f"{labels['no_commitment']:>6} {median:>14}"
        )


if __name__ == "__main__":
    sys.exit(main())
