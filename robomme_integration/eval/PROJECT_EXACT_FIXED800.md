# Project S0/Q0/A6 exact fixed-800 evaluation

`project_exact_runner.py` is the local, provenance-sealed bridge for evaluating the existing
multitask project controls under RoboMME's pinned paper rollout: 16 tasks × 50 episodes, action
horizon 20, execute horizon 16, max 1,300 simulator steps, and diffusion seed 7. Its protocol and
output directories are intentionally distinct from both the released-checkpoint positive controls
and the older project execute-10 results.

The runner does not discover, download, upload, or launch anything remotely. It requires an
already-local checkpoint plus its exact deploy-tree manifest. Before starting either process, it:

- rebuilds the checkpoint manifest from every local `params/` and `assets/` byte and compares the
  result with the explicitly supplied manifest SHA-256;
- verifies and freshly extracts the pinned OpenPI archive
  `ed923b2c27d2f608d62cc4b5ca89d5b80c14739dba1ab81d6f53d8013bcb66ad`;
- stages and hashes the current `robomme_integration` and `vla_eval` packages, then proves the
  policy interpreter imports OpenPI, `openpi_client`, the RoboCasa shim, `vla_eval`, and this
  integration from those staged roots rather than an installed fork;
- checks the official policy, benchmark, and clean ManiSkill Git worktrees against commits
  `ecf086c`, `856bc3`, and `07be6fbc` respectively;
- seals deterministic interpreter/package fingerprints for the policy runtime (Python, JAX,
  jaxlib, Flax, Orbax, NumPy) and simulator runtime (Python, Torch, Gymnasium, SAPIEN, NumPy),
  without hashing either whole virtual environment;
- starts one resident policy process and restarts only the simulator process after the exact known
  `vk::createInstanceUnique: ErrorIncompatibleDriver` signature. It waits for `nvidia-smi -L` to
  recover before consuming another bounded retry.

Run only when two local GPUs are available; the released-checkpoint campaign currently uses the
same local resources. This is the ready S0 invocation, but it is an invocation recipe—not approval
to start it:

```bash
cd ~/Research/TRI/wsmv2
PYTHONPATH="$PWD/robomme_integration/compat:$PWD:$HOME/Research/TRI/vla-evaluation-harness/src" \
  ~/Research/TRI/robomme_eval/openpi/ed923b2c/.venv/bin/python \
  robomme_integration/eval/project_exact_runner.py \
  --arm s0 \
  --checkpoint-root ~/Research/TRI/robomme_eval/checkpoints/s0-all16-step59999 \
  --checkpoint-manifest ~/Research/TRI/robomme_eval/checkpoints/manifests/s0-all16-step59999.tree.json \
  --expected-checkpoint-sha256 891410288b651baef88ebd24b4c52bb5ff5557df294a85a1e242b12ff277a4 \
  --openpi-archive ~/Research/TRI/robomme_eval/artifacts/openpi-ed923b2c.tgz \
  --openpi-python ~/Research/TRI/robomme_eval/openpi/ed923b2c/.venv/bin/python \
  --simulator-python ~/Research/TRI/robomme_eval/runtime-v0.4.0/env-v0.4.0/bin/python \
  --policy-root ~/Research/TRI/robomme_eval/official_reference/robomme_policy_learning \
  --benchmark-root ~/Research/TRI/robomme_eval/official_reference/robomme_benchmark \
  --maniskill-root ~/Research/TRI/robomme_eval/official_reference/ManiSkill-07be6fbc-git \
  --vla-eval-root ~/Research/TRI/vla-evaluation-harness \
  --work-root ~/Research/TRI/robomme_eval/project_exact_work \
  --output ~/Research/TRI/robomme_eval/results/project-exact-s0-fixed800
```

For Q0, change the arm/root/manifest/output and use exact digest
`20f91eac9369e2734459cd3dd3d326dd833abe9dcbd8f8184eaec8256ec99f51`. A6 becomes runnable only
after its local deploy tree and exact manifest digest are supplied in the same form.

Progress is atomically resumed from `evaluation/progress.json`. Each simulator process has its own
`logs/eval-attempt-NNNN.log`; the resident server appends to `logs/server.log`. A result is complete
only after the validated `evaluation/scorecard.json` and `PROJECT_EXACT_FIXED800_COMPLETE` marker
exist. On process/node restart, the wrapper reloads and rehashes the exact content-addressed project
snapshot named by the existing orchestration manifest; it does not restage the mutable working tree.

Q2 is deliberately rejected because its trained commit/stride is 10 while this protocol executes
16 actions. `official_recipe_lerobot` is also rejected because it needs its own two-view evaluator.
Deploy-tree manifests identify checkpoint bytes and URI, but do not themselves prove the semantic
training recipe associated with an arm; keep the training run manifest beside each future A6
manifest until that association is added to the checkpoint artifact schema.
