# Local RTX 5090 Pick/Button evaluation lane

This lane evaluates the already-trained 16-cell PickXtimes/ButtonUnmaskSwap representation panel
with the same fixed-50 test split and demonstration-video history as the p5 lane. It pins the
training-matched standard-ed923 OpenPI archive; it must not use the later PTRM fork for these
checkpoints. Native NVIDIA EGL is mandatory and lavapipe is disabled.

The first queue is exactly:

- tasks: `PickXtimes`, `ButtonUnmaskSwap`;
- arms: `q3`, `wsm_cfg`, `wsm_tanh`, `wsm_d8`, `jepa_l01_k1`, `jepa_l1_k32`,
  `jepa_l01_k16`, `salient`;
- 50 test episodes per cell, 800 scored episodes total;
- retry queue ID: `pick-button-representation-fixed50-local5090-v2`.

Do not reuse `local_ready9`, `local_ready11`, or the terminally failed `local5090-v1` queue. The campaign retries only its narrow
Vulkan/transport/worker allowlist, verifies exact remote results on resume, and cleans each staged
checkpoint/workspace after publishing evidence. Any terminal cell halts the queue before the next
cell is staged, so a shared harness or serving defect cannot contaminate the panel.

The runtime archive, standard-ed923 archive/venv, harness, ManiSkill, pinned upstream tree, and
pi0.5 SigLIP asset must all pass the read-only inspection. No preflight receipt can be sealed until
the fresh v2 template's PickXtimes/Q3 policy checkpoint and numeric workspace step are staged with
the policy checkpoint's authenticated `checkpoint-tree.json`.

Readiness inspection (no cloud or simulator):

```bash
.venv/bin/python -m robomme_integration.eval.local_rtx5090_preflight
```

Resolve the exact 16 completed checkpoint/workspace claims into the fresh template (read-only S3):

```bash
.venv/bin/python -m robomme_integration.eval.build_existing_pick_button_queue \
  --queue-id pick-button-representation-fixed50-local5090-v2 \
  --output /home/sarveshp/Research/TRI/robomme_eval/campaign-runtime/queues/pick-button-representation-fixed50-local5090-v2.template.json \
  --confirm-read-s3
```

After the combined source tree is frozen and explicit approval is obtained, seal both an unscored
MoveCube demo-history reset and a real two-client PickXtimes/Q3 workspace observation-to-action
probe. Both native-EGL clients share one policy server; each must execute exactly one action with
zero harness errors. All four paths below are mandatory, and the workspace path must end in its
numeric step (for example, `10000`), never the staging root:

```bash
.venv/bin/python -m robomme_integration.eval.local_rtx5090_preflight \
  --workspace-probe-policy-checkpoint /exact/staged/checkpoint/19999 \
  --workspace-probe-checkpoint /exact/staged/workspace/10000 \
  --workspace-probe-queue-template /home/sarveshp/Research/TRI/robomme_eval/campaign-runtime/queues/pick-button-representation-fixed50-local5090-v2.template.json \
  --workspace-probe-policy-tree-manifest /exact/staged/checkpoint/checkpoint-tree.json \
  --confirm-preflight
```

Dry-finalize with the two content-addressed files printed by the preflight:

```bash
.venv/bin/python -m robomme_integration.eval.local_rtx5090_campaign \
  --queue-template /home/sarveshp/Research/TRI/robomme_eval/campaign-runtime/queues/pick-button-representation-fixed50-local5090-v2.template.json \
  --native-preflight-claim /path/printed/preflight-SHA.json \
  --runtime-receipt /path/printed/runtime-SHA.json \
  --sealed-queue-output /home/sarveshp/Research/TRI/robomme_eval/campaign-runtime/queues/pick-button-representation-fixed50-local5090-v2.queue.json \
  --dry-run
```

Only after separate explicit approval, replace `--dry-run` with `--confirm-run`. That is the sole
campaign path which publishes evidence or starts scored policy/simulator processes.
