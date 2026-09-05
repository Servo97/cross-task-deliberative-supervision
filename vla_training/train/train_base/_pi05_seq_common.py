"""Stage-Q (RoboTTT 2x2) train-config builder — all four arms from ONE dataclass.

The four arms Q0/Q1/Q2/Q3 are derived here from a single `StageQArms(fast_weights, workspace)`
(imported from the fork's `robottt_fast_weights`). The ONLY per-arm differences that reach the
openpi TrainConfig are the two Pi0Config booleans `robottt` and `wsm_tanh` and the checkpoint
back-fill regex; the data pipeline, optimizer, schedule, steps, batch, full-finetune surface, and the
RoboTTT geometry are shared by construction. Q2 (fast_weights=True, workspace=False) is vanilla
paper-faithful RoboTTT — a config point, not a bespoke path.

Stage-Q uses the CONTIGUOUS-EPISODE window loader (`StageQRobocasaSequenceDataConfig` ->
`StageQWindowDataset`, stacked [L, ...] items) rather than the Stage-S per-step loader, and trains
with the fork's `stage_q_train_step`: the fast-weight chain is teacher-forced (state, action) only,
so O_0..O_{L-1} come from one cheap `run_sequence` scan and the L flow-matching losses (Eqs 4-5,
TBPTT inside the scan) run as ONE pi forward at batch B*L — same math as the per-step-loop
reference `robottt_sequence_loss`, ~L x cheaper. With fast_weights off the step is EXACTLY the
per-step loss at batch B*L on window-sampled data (the Q0 == baseline reduction). batch_size in
this config counts WINDOWS.

Two OPTIONAL recipe variants (both default-off, both inert when absent from the YAML):
  * A6 ``data.iid_window_steps: true`` — same window support, i.i.d. step grouping (targets the
    Q0-S0 -7.3 recipe gap, steelman G5). Rejected with fast_weights on.
  * A7 top-level ``staged:`` block — phase 1 trains ONLY the robottt subtree (backbone masked) at a
    scaled LR, then the SAME run continues as today's full finetune (targets the Q2-Q0 -3.7
    training-time damage, steelman G6). Requires fast_weights on.
"""

from __future__ import annotations

import os
import sys

from wsm_settings import ROBOCASA_OPENPI_SRC

# The fork is the single source of the 2x2 dataclass and the fast-weight geometry.
sys.path.insert(0, str(ROBOCASA_OPENPI_SRC))


def _robottt():
    from openpi.models.robottt_fast_weights import ALL_STAGE_Q_ARMS, StageQArms  # noqa: F401

    return StageQArms, ALL_STAGE_Q_ARMS


def derive_arm(fast_weights: bool, workspace: bool):
    """Map the two (and only two) flags to the single StageQArms dataclass."""
    StageQArms, _ = _robottt()
    return StageQArms(fast_weights=bool(fast_weights), workspace=bool(workspace))


def all_arms():
    _, arms = _robottt()
    return list(arms)


def iid_window_steps(cfg) -> bool:
    """A6 flag: ``data.iid_window_steps`` in the recipe (default False = today's window loader)."""
    return bool(dict(getattr(cfg.data, "raw", {}) or {}).get("iid_window_steps", False))


_STAGED_KEYS = {"phase1_steps", "phase1_freeze", "phase1_lr_scale_robottt"}


def staged_block(cfg) -> dict | None:
    """A7 block: top-level ``staged:`` in the recipe (default absent = today's single stage)."""
    raw = dict(getattr(cfg, "raw", {}) or {}).get("staged")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"staged: must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _STAGED_KEYS
    if unknown:
        raise ValueError(f"unknown staged keys {sorted(unknown)}; allowed {sorted(_STAGED_KEYS)}")
    if "phase1_steps" not in raw:
        raise ValueError("staged: requires phase1_steps")
    return {
        "phase1_steps": int(raw["phase1_steps"]),
        "phase1_freeze": str(raw.get("phase1_freeze", "backbone")),
        "phase1_lr_scale_robottt": float(raw.get("phase1_lr_scale_robottt", 1.0)),
    }


def check_declared_arm(cfg, arms) -> None:
    """If the recipe declares ``model.stage_q_arm``, the CLI flags MUST agree with it.

    The arm-specific variant recipes (A6 = Q0 only, A7 = Q2 only) are only valid for one arm, so a
    mismatched --fast-weights/--workspace pair must fail before compute is spent rather than train a
    recipe the YAML header does not describe.
    """
    declared = cfg.model.get("stage_q_arm") if hasattr(cfg.model, "get") else None
    if declared is None:
        return
    if str(declared) != arms.name:
        raise ValueError(
            f"recipe declares model.stage_q_arm={declared!r} but the flags derive arm {arms.name!r} "
            f"(fast_weights={arms.fast_weights}, workspace={arms.workspace}); refusing to launch"
        )


def stage_q_env(cfg, arms) -> None:
    """Export the Stage-Q loader/interface env contract for `arms` (mirrors the Stage-S entry).

    workspace=True reuses the promoted Stage-S tanh read UNCHANGED (WSM_TANH=1 + a feature root);
    fast_weights=True turns on the RoboTTT online update. Both, either, or neither may be set — the
    2x2 falls out of these two flags.
    """
    m = cfg.model
    rt = dict(m.get("robottt", {})) if hasattr(m, "get") else {}
    # Action norm-stat nav split: the same recipe knob and the same pre-openpi export as Stage-S
    # (_pi05_common.export_norm_split_nav) — Stage-Q shares LeRobotRobocasaDataConfig's norm-stat
    # fallback, so the merged blob is produced by the identical fork code path.
    from vla_training.train.train_base._pi05_common import export_norm_split_nav

    print(f"[pi-stage-q] norm_split_nav={export_norm_split_nav(cfg)}", flush=True)
    os.environ["WSM_STAGE_Q"] = "1"
    os.environ["WSM_STAGE_Q_ARM"] = arms.name
    os.environ["WSM_SEQ_WINDOW_LEN"] = str(int(rt.get("window_len", 8)))
    os.environ["WSM_SEQ_CHUNK_STRIDE"] = str(int(rt.get("chunk_stride", 8)))
    # workspace read: reuse the exact Stage-S tanh interface + its current-only feature contract.
    if arms.workspace:
        os.environ["WSM_TANH"] = "1"
        os.environ["WSM_K_WINDOW"] = "1"
        if not os.environ.get("WSM_POLICY_FEATS_ROOT"):
            raise ValueError("Q1/Q3 (workspace on) require WSM_POLICY_FEATS_ROOT for the tanh read")
    else:
        os.environ.pop("WSM_TANH", None)


def build_stage_q_train_config(cfg, gs, arms):
    """(TrainConfigView, GroupedSoup, StageQArms) -> openpi TrainConfig. Imports openpi/jax.

    Only `robottt`/`wsm_tanh` and the back-fill regex vary across the 2x2; everything else is shared.
    """
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders
    from flax import nnx

    # Post-import nav-split check, the same fail-loud contract the Stage-S arm families get for
    # their loader-facing knobs (_pi05_common: JEPA horizon, salient targets, deltanet K). Stage-Q
    # exported WSM_NORM_SPLIT_NAV in stage_q_env above; this confirms the openpi archive we just
    # imported actually implements the knob, so an archive predating it fails here instead of
    # silently writing old single-blob norm stats into assets/norm_stats.json.
    from vla_training.train.train_base._pi05_common import (
        assert_norm_split_nav_loader,
        norm_split_nav_enabled,
    )

    assert_norm_split_nav_loader(norm_split_nav_enabled(cfg.data.raw))

    check_declared_arm(cfg, arms)
    iid_steps = iid_window_steps(cfg)
    staged_raw = staged_block(cfg)
    m, o, t = cfg.model, cfg.optim, cfg.train
    rt = dict(m.get("robottt", {}))
    flags = arms.pi0_flags()  # {"robottt": ..., "wsm_tanh": ...} — the ONLY per-arm knobs
    peak_lr = float(o["lr"])
    ckpt = os.environ.get("WSM_INIT_FROM") or t.get("weight_loader") or t.get("init_from")
    num_train_steps = int(os.environ.get("WSM_MAX_STEPS") or t.get("max_steps", 60000))
    final_only = os.environ.get("WSM_FINAL_ONLY_CHECKPOINTS", "0") == "1"
    save_interval = num_train_steps if final_only else int(t.get("save_interval", 5000))

    model = pi0_config.Pi0Config(
        pi05=bool(m.get("pi05", True)),
        max_token_len=int(m.get("max_token_len", 200)),
        # --- the two ablation flags (everything below is shared across all four arms) ---
        robottt=flags["robottt"],
        wsm_tanh=flags["wsm_tanh"],
        wsm_tanh_gate_init=float(os.environ.get("WSM_TANH_GATE_INIT", "0.001")),
        wsm_k_window=1,
        # --- shared RoboTTT geometry ---
        robottt_token_dim=int(rt.get("token_dim", 256)),
        robottt_fast_hidden=int(rt.get("fast_hidden", 128)),
        robottt_num_registers=int(rt.get("num_registers", 16)),
        robottt_base_inner_lr=float(rt.get("base_inner_lr", 0.1)),
        robottt_gate_init=float(rt.get("gate_init", 0.001)),
        robottt_window_len=int(rt.get("window_len", 8)),
        robottt_tbptt_segment=int(rt.get("tbptt_segment", 8)),
    )
    # Arm-suffixed name/exp_name: checkpoint_dir = base/name/exp_name, so without the suffix all
    # four arms would collide into ONE checkpoint dir and wandb run (audit 2026-07-23).
    return _config.TrainConfig(
        name=f"{m.get('config_name', 'pi05_robocasa_stage_q')}_{arms.name}",
        exp_name=f"{t.get('exp_name', 'pi05_rc365_stage_q')}_{arms.name}",
        project_name=os.environ.get("WANDB_PROJECT", "wsm-robocasa"),
        model=model,
        # The Stage-Q WINDOW loader (contiguous-episode, stacked [L, ...]); batch_size below counts
        # windows. The per-step transform chain is identical to Stage-S by construction.
        data=_config.StageQRobocasaSequenceDataConfig(
            data_dirs=gs.soup,
            dataset_weights=gs.pi05_weights,  # ignored by the window path (uniform-over-windows)
            defer_image_resize_to_model_preprocess=False,
            stage_q_window_len=int(rt.get("window_len", 8)),
            stage_q_chunk_stride=int(rt.get("chunk_stride", 8)),
            # A6: same support, i.i.d. step grouping. TrainConfig.__post_init__ rejects it with
            # fast weights on (the chain needs contiguity), so a mis-flagged launch fails here.
            stage_q_iid_steps=iid_steps,
        ),
        # Back-fill exactly the newly-initialized subtree(s) — robottt_fast (stable name) and/or
        # wsm_tanh_cond — from the shared S0/M* checkpoint. Derived from the single dataclass.
        weight_loader=weight_loaders.CheckpointWeightLoader(ckpt, missing_regex=arms.missing_regex()),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(o.get("warmup_steps", 1000)),
            peak_lr=peak_lr,
            decay_steps=int(o.get("decay_steps", num_train_steps)),
            decay_lr=float(o.get("decay_lr", peak_lr / 10)),
        ),
        freeze_filter=nnx.Nothing(),  # full finetune + the new subtree(s), same surface across the 2x2
        # A7: two-phase inside ONE run. NOT freeze_filter — that would bf16-cast the backbone and
        # drop it from the optimizer state at init, making phase 2 impossible in the same process;
        # the phase-1 freeze is a per-step grad+update mask instead (scripts/train.py).
        staged=(_config.StagedRecipe(**staged_raw) if staged_raw else None),
        ema_decay=m.get("ema_decay"),
        resume=os.environ.get("WSM_RESUME") == "1",
        num_train_steps=num_train_steps,
        save_interval=save_interval,
        keep_period=None if final_only else int(os.environ.get("WSM_KEEP_PERIOD", "5000")),
        batch_size=int(t.get("batch_size", 64)),
        num_workers=int(t.get("num_workers", 32)),
        checkpoint_base_dir=os.environ.get("WSM_CKPT_BASE", str(t.get("output_dir", "./checkpoints/wsm_robocasa"))),
    )
