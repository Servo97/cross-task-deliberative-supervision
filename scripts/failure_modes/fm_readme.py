#!/usr/bin/env python3
"""Generate ``failure_modes/README.md`` for the eyeballing session.

Generated rather than hand-written so the per-task correct/distractor derivations and every
threshold are read straight out of the code that computed the numbers, and cannot drift
from it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm_targets  # noqa: E402
from fm_common import (  # noqa: E402
    APPROACH_CONSECUTIVE_K,
    APPROACH_RADIUS_M,
    DIVERGENCE_CONSECUTIVE_K,
    DIVERGENCE_THRESHOLD_M,
    RESETS_PER_TASK,
    VIDEO_CAMERAS,
    VIDEO_H,
    VIDEO_W,
)
from fm_metrics import DEPARTURE_M  # noqa: E402

HEADER = """# Failure-mode study — videos + trajectory-difference metrics

Purpose: decide, by eye against the numbers, whether these metrics can classify a failure
on their own. The question each rollout answers is **why did this fail** —

| the metric says | reading |
|---|---|
| went to the right target, still failed | **control failure** — it knew what to do and fumbled the manipulation |
| went to a plausible wrong target | **memory / context failure** — it ignored the history that disambiguates |
| never settled on any target | **collapse** — no coherent intent to classify |

Everything here is paired: within a benchmark, every checkpoint ran the **same {resets} resets
per task**, so any two arms can be compared reset by reset.

## Layout

```
failure_modes/
  README.md                                  this file
  master.csv                                 one row per rollout, all arms, both benchmarks
  <bench>/<task>/expert/<reset_id>.mp4       the demonstration for that reset (reference)
  <bench>/<task>/<ckpt>/<reset_id>.mp4       the policy rollout
  <bench>/<task>/<ckpt>/<reset_id>.json      that rollout's full metric record, incl. curves
  _provenance/                               per-cell label gates + the reset manifests
```

`<reset_id>` is `heldout_NNN` or `train_NNNN`; the two share the same numbering namespace as
the sealed manifests, so `heldout_003` here is sealed held-out episode 3 of that task.

## Videos

{cameras} composited {tiles}, {vw}x{vh} per view, H.264. The first three are exactly the
images the policy receives; `agentview_center` is an extra human-readable overview it never
sees. The caption bar carries task / arm / reset / outcome and the episode's instruction.

Playback is **real time**: the simulator runs at 20 Hz, every 2nd control step is kept, and
the file is written at 10 fps. Everything is encoded at CRF 23 — rollouts and demonstrations
alike — which puts a typical episode at 2-6 MB.

To eyeball a cell, open the expert video for a reset alongside the arm's video for the same
reset — same scene, same object placement, same instruction, so the only difference is the
behaviour.
"""

PROTOCOL = """## Protocol, and one honest deviation

Every reset in this study is **demo-pinned**: the environment is reset to a recorded
demonstration's own `ep_meta` + `model.xml` + `states[0]`, so the scene the policy sees is
bit-identically the scene that demonstration was collected in, and the demonstration itself
is the ground-truth expert trajectory for that reset.

* **RoboCasa**: this IS the sealed protocol (`reset.kind == "heldout_demo"`). No deviation.
* **ReMemBench**: the sealed protocol resets from `ep_meta` + `seed`, which fixes the layout,
  the object categories and the object instances but **re-samples every object's pose** from
  the seeded RNG. Bit-reproducible, but not the demonstration's placement — so under the
  sealed reset no ground-truth expert trajectory exists for a reset, and a
  trajectory-difference metric has nothing to difference against. This study therefore puts
  ReMemBench on RoboCasa's footing and pins the demonstration state.

  **Consequence:** the ReMemBench success rates here are NOT comparable to the sealed
  cb24fe49 numbers (31.3 / 36.8 / …). Different reset distribution, and n={resets} per task
  instead of 88 episodes x 3 rollouts. They are comparable *to each other*, which is what a
  paired failure-mode comparison needs, and that is all they are used for.

One more ReMemBench-only fix: demo `model.xml` files carry absolute asset paths from the
collection machine, and the fork's own `edit_model_xml` only remaps three of the four asset
prefixes (`objects/aigen_objs` is missed). Those paths are rewritten to the local install
before loading. It is a path fix; the mesh bytes are the ones the registry would have loaded.

Replan cadence 8, matching every sealed eval.

### Diffusion-noise draws, and why one is not always enough

Most cells run **one rollout per reset at draw 0**, whose noise stream is byte-identical to
the sealed protocol's rollout 0. `MemFruitInSinkRightFar` additionally runs **draws 1 and 2**
(the sealed protocol's own three-draw cadence), because one draw could not discriminate there
at all. The `rollout_index` column says which draw a row is.

> **Methodological finding — read fruitRF through this.** At draw 0 every one of the five
> ReMemBench arms scored **0/20** on `MemFruitInSinkRightFar`, with 20/20 `no_commitment`.
> Zero variance across five very different checkpoints is not a policy result. Running the
> **sealed** protocol on the same task and arm and splitting by draw index shows why:
>
> | draw | sealed dnw8, fruitRF |
> |---|---|
> | 0 | **0/5** |
> | 1 | 2/5 |
> | 2 | 1/5 |
> | pooled | 3/15 = 20% |
>
> Draw 0 carries none of the successes. The study's single-draw cell was therefore replaying
> a barren noise stream, and its 0% floor is an artifact of that draw, not a property of the
> policies. **On a high-variance task, a single rollout per reset can floor every arm at once;
> multiple draws are required before an all-arms floor means anything.** The harness itself
> was cleared first — the reset restores the demonstration state bit-exactly
> (`max|reset - states[0]| = 0.000e+00`), the served instruction matched the demonstration's
> 9/9, debug-visual sites were suppressed identically in both reset paths, and the sealed
> scene differs from the demo-pinned one by 2-3 cm.
>
> The same rule is applied data-driven to RoboCasa: it runs at one draw per reset, and a task
> is extended to three draws only if it shows the same all-arms floor at draw 0.

Resets per task: the sealed held-out tail first (it is smaller than {resets} on these tasks),
topped up from the TRAINING demos immediately preceding that tail, so the {resets} resets are
a contiguous suffix of the ordered demo list. The `split` column marks which is which — check
whether the story differs between held-out and train resets before trusting either.
"""

METRICS = """## The metrics

Every number below is in `master.csv`; the per-rollout JSON additionally carries the full
curves (teacher-forced error per step, EE deviation per step, distance to correct and to
distractor per step) so nothing has to be recomputed from simulation.

### (a) Teacher-forced action deviation — `tf_mse_mean`, `tf_mse_peak`, `tf_mse_argmax_frac`

The policy is queried **along the expert's trajectory**, not its own: the simulator is set to
the expert's state at t = 0, 8, 16, …, the observation is rebuilt there, and the policy's
8-action chunk is scored against what the expert actually did over that window. Nothing the
policy says is executed, so the query distribution is the expert's and is identical for every
checkpoint — the deviation is purely the policy's own disagreement, with no compounding.

Per-dimension squared error is normalised by the per-dimension **expert action standard
deviation pooled over that task's {resets} resets** — one fixed scaler per task, shared by all
arms. Units: multiples of expert action variance. 1.0 means "off by about as much as the
expert's actions vary".

* `tf_mse_mean` — average disagreement over the episode. **Known to be insensitive** — see
  the warning below.
* `tf_mse_p90` — 90th percentile. Prefer this over the mean.
* `tf_mse_peak` — worst single step.
* `tf_mse_argmax_frac` — where the peak sits, as a fraction of episode length. **Early peak
  (near 0) = disagreed about the plan from the start. Late peak (near 1) = tracked the expert
  and then lost it.** This is the field that most often separates the two failure modes on
  its own.

Because it is open-loop, this metric survives a rollout that diverged immediately: even a
policy that drove off in the first second still gets scored over the whole expert trajectory.

> **Warning, measured not assumed.** On `MemWashAndReturnLeft` the never-finetuned
> pretrain-150k checkpoint scores **0/20 successes** yet its `tf_mse_mean` (0.557) is
> indistinguishable from the finetuned baseline (0.530) and deltanet-w8 (0.487). On
> `MemFruitInSinkRightFar` the same checkpoint sits at 5.80, i.e. wildly separated. So this
> metric's sensitivity is **task-dependent**, and the mean is the weakest form of it: most of
> a manipulation trajectory is low-variance transit that any checkpoint predicts, and a
> disagreement confined to the decisive few steps is averaged away. Treat a small
> `tf_mse_mean` gap as no evidence either way rather than as evidence of agreement, and read
> `tf_mse_p90` / `tf_mse_peak` / `tf_mse_argmax_frac` instead. Where teacher-forced error is
> flat across arms, `target_commitment` is the metric doing the actual work.

### (b) Free-rollout divergence — `first_divergence_step`, `dtw`, `ee_dev_mean/final`

Closed-loop end-effector world position against the expert's, aligned by timestep.

* `first_divergence_step` — first step where deviation exceeds **{thresh} m** and *stays*
  above it for **{cons} consecutive control steps** ({secs:.1f} s at 20 Hz). {thresh_cm} cm is
  about a gripper width: smaller deviations are ordinary control noise around the same
  intent. Requiring half a second of sustained excess stops a single reach wobble triggering
  it. `-1` = never diverged. `first_divergence_frac` normalises by expert length, so 0.0 is
  "wrong from the start" and 1.0 is "never".
* `dtw` — banded dynamic time warping over the two EE position curves, reported as **mean
  cost per aligned pair (metres)**, so a long episode does not inflate it. Timestep-aligned
  deviation punishes a policy that does the right thing slower; DTW does not. A large
  timestep deviation with a small DTW means *right path, wrong tempo* — usually control, not
  memory.
* `ee_dev_final` — where it ended up relative to the expert's end.

### (c) Target commitment — `target_commitment` — the memory-vs-control discriminator

The one metric designed to answer *why*. Over the rollout, the distance from the tracked
agent (the end-effector, or the manipulated object where that is the meaningful thing) to the
task's CORRECT target and to its plausible DISTRACTOR(s) is measured every step from sim
state. A candidate is **committed to** at the first step where it is the nearer candidate AND
within **{radius} m** for **{acons} consecutive steps**. Whichever is committed to first names
the rollout:

| label | meaning | verdict when the rollout also failed |
|---|---|---|
| `approached_correct` | reached the right target first | **control failure** |
| `approached_wrong` | reached a distractor first | **memory / context failure** |
| `no_commitment` | neither, sustained | **collapse** |

`no_commitment` covers two behaviours worth telling apart, so read it with
**`agent_travel_max`** — the furthest the tracked agent ever got from where it started, in
metres. A large travel with no commitment is a policy that moved decisively and settled on
nothing; a small one is a policy that froze. Measured on `MemFruitInSinkRightFar`, the rmb
baseline travels a median **0.22 m** against the expert's **1.19 m** from an identical
starting state — it does not choose the wrong fruit, it barely moves at all, for the full
1400-step horizon.

Two guards, both necessary:

* **Departure gate.** No commitment can register until the tracked agent has moved
  {departure} m from where it started. Some tasks begin with the agent already at its correct
  target — MemWashAndReturn's fruit starts *inside* the correct container — so without this a
  policy that never moves would score `approached_correct`.
* **Phase gate.** Where a task has a semantic phase boundary, only the relevant leg counts.
  MemWashAndReturnLeft is gated on `place_success`: the choice of container is only
  meaningful on the *return* leg, after the fruit has been to the sink.

`commit_step_correct` / `commit_step_wrong` give the step each was committed to (-1 = never),
so a rollout that went wrong and then recovered is visible rather than hidden by the label.

**Sanity check built in:** the expert demonstration for every reset is pushed through the same
classifier. A demonstration must come out `approached_correct`; the per-rollout JSON records
its label as `expert_label`. Any reset whose expert is not `approached_correct` means the
candidate derivation is wrong for that episode and its rows should be discarded, not explained.

### (d) Outcome — `success`, `failed_task`

`success` is the environment's own `_check_success`. `failed_task` is ReMemBench's hard
prospective failure (the cook deadline blown) — it can never be scored a success, so
`success` is already `raw_success AND NOT failed_task`. For MemHeatPot the JSON also carries
the prospective fields (turn-on, turn-off, wait timer against its threshold), because a
prospective miss is a third failure mode that neither geometry nor action error will show.
"""

FOOTER = """## Recommended eyeballing set — start here

Sorted by how much the metric is claiming, so a disagreement costs you the least time.

1. **`remembench/MemWashAndReturnLeft/*` rows where `target_commitment == approached_wrong`.**
   This is the study's cleanest memory-failure claim: the fruit physically ends up in the
   *other* plate. In the canary those rollouts ended **1.5-6.8 cm from the wrong container and
   72-80 cm from the right one**, so if the metric is ever going to be trustworthy it is here.
   Watch each against `expert/<same reset_id>.mp4`.

   **Interim signal worth your eye (n=20/arm, suggestive not conclusive):** wrong-container
   counts *rise* with the memory-read arms while plain finetuning leads on success —

   | arm | success | approached_wrong |
   |---|---:|---:|
   | base | 70% | 4 |
   | jepa_k16 | 50% | 6 |
   | dnw16_drop | 50% | 8 |
   | dnw8 | 35% | 9 |

   If that survives eyeballing, it is a *mis-binding* story — the read is being used, and used
   to select the wrong referent — which is a sharper claim than "the mechanism didn't help".
   The thing to check in the videos is whether the wrong-container rollouts look *confident*
   (a clean approach to the wrong plate) or *confused* (drifting into it). Confident is
   mis-binding; confused is ordinary control failure that the label is over-reading.

2. **`remembench/MemHeatPot/pretrain150k/*`** — 20/20 `approached_wrong` via the knob-state
   rule, against base's 11 correct / 0 wrong. A never-finetuned checkpoint turning the wrong
   knob every single time is the strongest positive control the classifier has; if these
   videos do not show it reaching for the wrong knob, the knob rule is wrong.

3. **`remembench/MemFruitInSinkRightFar/base/*`** — the collapse class. `agent_travel_max`
   says the baseline moves a median 0.22 m against the expert's 1.19 m. These should look like
   a robot that never commits to going anywhere. Read them with the draw caveat above.

4. **Any row where `target_commitment == approached_correct` but `success == 0`** — the
   control-failure class. These should look like the right intent, botched at contact.

## How to read a cell quickly

1. Sort `master.csv` by `task`, then `reset_id`, then `ckpt`. Every reset gives you one row
   per arm plus the expert's video.
2. For a **failed** rollout, look at `target_commitment` first. `approached_wrong` is the
   claim the study is testing — open that video against the expert's and check whether the
   robot visibly goes to the other container / the other burner / the mirrored counter.
3. Then check `tf_mse_argmax_frac`. A memory failure should disagree with the expert **early**
   (the branch point), a control failure **late** (at the contact).
4. `first_divergence_step` close to 0 with `approached_correct` is usually a navigation or
   tempo difference, not a different intent — cross-check `dtw`.
5. Disagreements between your eye and the label are the actual result. Note the `reset_id`;
   the JSON has every curve needed to see why the classifier decided what it did.

## Caveats worth carrying

* n = {resets} per task per arm. These are orderings and worked examples, not rates with
  usable confidence intervals.
* The distractor for `ScrubCuttingBoard` is *derived*, not placed by the task author — see its
  entry below. Treat its `approached_wrong` rows more sceptically than the others.
* `SearingMeat` has the tightest geometry in the study: its wrong burners sit ~12 cm from the
  correct one, inside the {radius} m approach radius. The label is still well-behaved because
  commitment requires a candidate to be the *nearest* one for {acons} consecutive steps, not
  merely within radius — measured on a demonstration, the pan passes over other burners en
  route and still classifies `approached_correct`. But for this task read
  `commit_step_correct` / `commit_step_wrong` alongside the label rather than the label alone.
* RoboCasa has no memory demand by construction; its three tasks were picked as the largest
  per-task (deltanet-w8 − baseline) deltas in the sealed default-recipe evals, i.e. where the
  mechanism helped most, not where memory is required. `approached_wrong` there is evidence
  about what the mechanism changed, not evidence of a memory demand.
* All numbers are box tier, not sealed study results.
"""


def task_section() -> str:
    lines = ["## Per-task: what counts as correct, and what counts as wrong\n"]
    for bench in ("remembench", "robocasa"):
        title = "ReMemBench" if bench == "remembench" else "RoboCasa"
        lines.append(f"### {title}\n")
        for task, spec in fm_targets.TASK_SPECS.items():
            if spec["bench"] != bench:
                continue
            lines.append(f"#### {task}\n")
            lines.append(f"* tracked agent: `{spec['agent']}`")
            lines.append(f"* manipulated object: `{spec.get('manipulated')}`")
            lines.append("* correct: " + ", ".join(f"`{c}`" for c in _describe(spec.get("correct", []))))
            lines.append("* distractor: " + ", ".join(f"`{c}`" for c in _describe(spec.get("distractor", []))))
            if spec.get("phase_gate"):
                lines.append(f"* phase gate: `{spec['phase_gate']}`")
            if spec.get("commitment_kind") == "knob_state":
                lines.append(
                    "* commitment measured on **knob state**, not geometry — turning the "
                    "correct knob first is `approached_correct`, any other knob first is "
                    "`approached_wrong`, none ever is `no_commitment`"
                )
            lines.append(f"\n{spec['docs']}\n")
    return "\n".join(lines)


def _describe(candidates):
    out = []
    for candidate in candidates:
        kind = candidate["kind"]
        if kind == "object":
            out.append(f"object:{candidate['name']}" + (" (moving)" if candidate.get("dynamic") else ""))
        elif kind == "fixture":
            out.append(f"fixture:{candidate['attr']}")
        elif kind == "burner":
            out.append(f"burner:{candidate['loc']}")
        elif kind == "mirror":
            out.append("mirror of the correct location about the sink")
        else:
            out.append(kind)
    return out or ["(none)"]


def summary_section(csv_path: str) -> str:
    if not os.path.exists(csv_path):
        return ""
    with open(csv_path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ""
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["bench"], row["task"], row["ckpt"])].append(row)
    lines = [
        "## Summary (generated from master.csv)\n",
        "| bench | task | arm | n | success % | correct / wrong / none | median first-divergence |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for key in sorted(grouped):
        group = grouped[key]
        n = len(group)
        success = 100.0 * sum(int(r["success"]) for r in group) / n
        labels = collections.Counter(r["target_commitment"] for r in group)
        divergences = sorted(int(r["first_divergence_step"]) for r in group if int(r["first_divergence_step"]) >= 0)
        median = divergences[len(divergences) // 2] if divergences else "-"
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {n} | {success:.1f} | "
            f"{labels['approached_correct']} / {labels['approached_wrong']} / "
            f"{labels['no_commitment']} | {median} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    body = HEADER.format(
        resets=RESETS_PER_TASK,
        cameras=", ".join(f"`{c}`" for c in VIDEO_CAMERAS),
        tiles="2x2",
        vw=VIDEO_W,
        vh=VIDEO_H,
    )
    body += "\n" + PROTOCOL.format(resets=RESETS_PER_TASK)
    body += "\n" + METRICS.format(
        resets=RESETS_PER_TASK,
        thresh=DIVERGENCE_THRESHOLD_M,
        thresh_cm=int(DIVERGENCE_THRESHOLD_M * 100),
        cons=DIVERGENCE_CONSECUTIVE_K,
        secs=DIVERGENCE_CONSECUTIVE_K / 20.0,
        radius=APPROACH_RADIUS_M,
        acons=APPROACH_CONSECUTIVE_K,
        departure=DEPARTURE_M,
    )
    body += "\n" + task_section()
    if args.csv:
        body += "\n" + summary_section(args.csv)
    body += "\n" + FOOTER.format(
        resets=RESETS_PER_TASK,
        radius=APPROACH_RADIUS_M,
        acons=APPROACH_CONSECUTIVE_K,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write(body)
    print(f"[readme] wrote {args.out} ({os.path.getsize(args.out)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
