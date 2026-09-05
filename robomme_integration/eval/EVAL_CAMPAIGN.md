# RoboMME fixed-50 node-resident evaluation campaign

`eval/campaign.py` amortizes one p5 allocation across a sealed sequence of already-trained,
single-task fixed-50 evaluations. It does not submit or acquire cloud capacity. Execution is local
to an existing node and remains blocked unless `--confirm-run` is passed after explicit approval.

The queue is fail-closed on four identities:

- the canonical task/arm/run/checkpoint completion and checkpoint-tree seals;
- the exact fixed-50 YAML digest and queue-specific eval/result IDs;
- a successful p5/H100 native rendered-reset claim through the exact
  `RoboMMEOfficialHistoryBenchmark`, with nonempty, length-matched demonstration video and
  proprioceptive state histories; and
- a separately self-sealed runtime receipt bound to that exact preflight claim, runtime archive,
  OpenPI archive, EGL environment, executable paths, upstream source, and vision weights.

There is no cross-archive serving assumption. Every cell opens its immutable training-attempt
manifest and verifies the exact OpenPI URI/SHA plus data inventory, task inventory, initialization,
seed, steps, action horizon, window length, and chunk stride. A queue may contain only cells trained
with one identical OpenPI archive, and the p5 preflight/runtime must use that same archive. Thus the
existing 16-cell Pick/Button panel uses canonical `ed923...`; PTRM/GDN+JEPA cells trained with
`24bd...` belong in a separate advanced-source queue. The advanced fork is never asserted to be a
drop-in serving replacement for ordinary checkpoints.

The native evaluator entry is also relocatable: the node creates a sealed wrapper which invokes the
staged OpenPI Python as `-m vla_eval.cli.main`. It never executes the archived `bin/vla-eval`, whose
shebang points to its build machine. The receipt binds this wrapper plus explicit harness,
RoboMME-benchmark, ManiSkill, OpenPI-source, policy-site, and simulator-site paths; wrapper `run
--help` is exercised before any scored cell.

Workspace-serving arms additionally require a hash-pinned producer claim and representation seal.
The local representation is checked through `WSM_GENERATION_COMPLETE.json` and both embedded
configuration hashes. PTRM evaluation parameters are part of the sealed cell and are forwarded to
`launch_gpu_fleet`; non-PTRM cells cannot carry them. JEPA/salience training-only arms correctly
serve through their stateless inference interface and therefore cannot smuggle workspace inputs.

Per-cell behavior is:

1. authenticate and stage the deploy-only checkpoint (and workspace representation when needed);
2. run native sharded fixed-50 through `launch_gpu_fleet`;
3. retry only the hard-coded `robomme-eval-transients-v1` Vulkan device-loss or harness transport /
   worker-interruption classes (at most three total attempts);
4. publish content-addressed evidence and an exact result or terminal-failure claim;
5. atomically record local state and remove the staged checkpoint/workspace/evidence tree;
6. continue only after a scientifically valid result; halt the queue after any terminal failure.

Unknown errors, imports, missing files, identity drift, OOM, policy failures, and generic nonzero
return codes are terminal and never retried. A terminal-failure claim is published for the cell,
then the campaign records `halted_terminal_failure` and stops before staging another cell. No queue
completion receipt is published for that partial campaign. A preempted queue resumes by verifying
immutable remote per-cell results rather than trusting local state; an existing terminal claim
halts that queue again instead of advancing past it.

Dry validation (no AWS/S3 read or write and no simulator/policy process):

```bash
python -m robomme_integration.eval.campaign \
  --queue /path/to/sealed-queue.json \
  --source-root /opt/ml/code \
  --native-preflight-claim /path/to/exact-preflight-claim.json \
  --runtime-receipt /path/to/exact-runtime-receipt.json \
  --work-root /opt/ml/robomme-eval-campaign \
  --dry-run
```

The remaining integration gate is operational, not hidden in this runner: a current source bundle
must first produce the rendered-reset claim and the node must produce the corresponding staged
runtime receipt. Until both exact files are supplied, no scored p5 evaluation campaign is runnable.

## Exact existing Pick/Button evidence path

Build the local draft for the 16 already-trained representation cells. The first command is the
network-dry preview; the second performs only S3 reads and writes one local JSON file. It fails if a
task/arm has zero or multiple completed retrains, if its completion claim does not select one exact
training attempt, or if the cells do not share the same training OpenPI/data/init recipe.

```bash
PYTHONPATH=. .venv/bin/python -m robomme_integration.eval.build_existing_pick_button_queue \
  --output /tmp/robomme-pick-button-fixed50.draft.json

PYTHONPATH=. .venv/bin/python -m robomme_integration.eval.build_existing_pick_button_queue \
  --output /tmp/robomme-pick-button-fixed50.draft.json \
  --confirm-read-s3
```

Dry-build the source-matched p5 renderer preflight. This still does not submit:

```bash
PYTHONPATH=. .venv/bin/python -m robomme_integration.eval.launch_p5_preflight \
  --openpi-profile standard --dry-run
```

After explicit approval, submit that preflight with `--confirm-submit` (and without `--dry-run`).
Download the immutable successful claim from the exact `claim_s3` printed by the preflight plan.
Then dry-build the bulk campaign; lack of a real successful claim, source drift, OpenPI drift, or an
invalid queue blocks before the AWS SDK is loaded:

```bash
PYTHONPATH=. .venv/bin/python -m robomme_integration.eval.launch_p5_campaign \
  --queue-template /tmp/robomme-pick-button-fixed50.draft.json \
  --native-preflight-claim /tmp/p5-native-eval-preflight.complete.json \
  --dry-run
```

Only after another explicit approval may the same command be run with `--confirm-submit`. The node
then stages and verifies the artifacts, reconstructs the predicted receipt byte-for-byte from the
actual paths/files, and only then invokes `eval.campaign --confirm-run`. No runtime receipt is
fabricated by the queue builder, and no eval job has been submitted by these implementation steps.
