"""Shared GR00T N1.7 (Isaac-GR00T / PyTorch) wiring for the base-VLA adapters.

Ported from the proven reference launcher (ported_raw/reference_code/launch_finetune_stage0.py),
generalized so the dataset list comes from a shared ``GroupedSoup`` (1 spec for balancing-OFF /
finetune, 3 specs for the balancing-ON pretrain arm — each a SingleDatasetConfig with mix_ratio).

GR00T-venv-only: all gr00t/torch imports live inside functions, so this module is import-safe
elsewhere. Both ``pretrain_groot_17`` and ``finetune_groot_17`` call ``build_and_run_groot``.
"""

from __future__ import annotations

import os
from pathlib import Path

# robocasa_panda_modality.py sits next to this file; importing it registers the NEW_EMBODIMENT
# modality config (needed BEFORE config.validate()).
MODALITY_PATH = str(Path(__file__).resolve().parent / "robocasa_panda_modality.py")


def load_modality_config(modality_config_path: str = MODALITY_PATH) -> None:
    import importlib
    import sys

    path = Path(modality_config_path)
    if not (path.exists() and path.suffix == ".py"):
        raise FileNotFoundError(f"modality config not found: {modality_config_path}")
    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    print(f"[groot] loaded modality config: {path}", flush=True)


def install_stage0_trainer(visual_lr_scale: float, adam_betas: tuple[float, float] = (0.9, 0.999)) -> None:
    """Swap Gr00tTrainer for a subclass that puts vision-tower params in their own LR group
    (Isaac-GR00T has no LoRA; this is the sanctioned vision-adapt approximation). At 1.0 it is a
    no-op relative to upstream. Ported verbatim from the reference stage-0 launcher."""
    import gr00t.experiment.experiment as exp_module
    from gr00t.experiment.trainer import Gr00tTrainer

    class Stage0Trainer(Gr00tTrainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            import torch.nn as nn
            from transformers import Trainer
            from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
            from transformers.trainer_pt_utils import get_parameter_names

            model = self.model
            decay_names = {n for n in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in n}

            def is_visual(name: str) -> bool:
                return ".visual." in name or name.endswith(".visual")

            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            base_lr = self.args.learning_rate
            groups = []
            for sel_visual, lr in ((False, base_lr), (True, base_lr * visual_lr_scale)):
                for decay, wd in ((True, self.args.weight_decay), (False, 0.0)):
                    params = [p for n, p in named if is_visual(n) == sel_visual and (n in decay_names) == decay]
                    if params:
                        groups.append({"params": params, "weight_decay": wd, "lr": lr})
            n_vis = sum(p.numel() for n, p in named if is_visual(n))
            n_rest = sum(p.numel() for n, p in named if not is_visual(n))
            print(
                f"[stage0] optimizer: visual {n_vis / 1e6:.1f}M @ lr={base_lr * visual_lr_scale:g}, "
                f"rest {n_rest / 1e6:.1f}M @ lr={base_lr:g} ({len(groups)} groups)",
                flush=True,
            )
            opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, model)
            # GR00T's TrainingConfig cannot pass YAML adam_beta1/2 into HF TrainingArguments.
            # Enforce the recorded recipe at the final optimizer-construction seam.
            opt_kwargs["betas"] = adam_betas
            self.optimizer = opt_cls(groups, **opt_kwargs)
            if "bitsandbytes" in opt_cls.__module__:
                try:
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                    for module in model.modules():
                        if isinstance(module, nn.Embedding) and module.weight.requires_grad:
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                except Exception as e:  # stabilization only; never fail training over it
                    print(f"[stage0] WARN bnb embedding override skipped: {e}", flush=True)
            return self.optimizer

        def train(self, *train_args, **train_kwargs):
            # Resume-on-retry: if the SageMaker entry synced a COMPLETE prior checkpoint and exported
            # WSM_RESUME_CKPT, continue from it (model + optimizer + LR schedule + step count) instead of
            # restarting from step 0 after an InternalServerError. The entry only sets WSM_RESUME_CKPT when a
            # complete checkpoint (model shards + trainer_state) exists, so this never targets a half-written
            # ckpt. Respects an explicit resume_from_checkpoint if gr00t's run() already passes one.
            import os as _os

            rc = _os.environ.get("WSM_RESUME_CKPT")
            if rc and "resume_from_checkpoint" not in train_kwargs:
                train_kwargs["resume_from_checkpoint"] = rc
                print(f"[stage0] RESUMING training from {rc}", flush=True)
            return super().train(*train_args, **train_kwargs)

    exp_module.Gr00tTrainer = Stage0Trainer
    print(f"[groot] Gr00tTrainer patched (visual_lr_scale={visual_lr_scale}, adam_betas={adam_betas})", flush=True)


def install_episode_subsample(num_demos: int, seed: int = 0) -> None:
    """Gated monkeypatch: restrict EVERY LeRobotEpisodeLoader to the deterministic first
    ``num_demos`` episodes (seed-``seed`` shuffle of episode_index values) — byte-identical to
    robocasa's filter_key, so GR00T finetunes on the SAME episodes as pi0.5. No on-disk changes;
    applies per dataset dir and flows through mix_ratio weighting (shorter len -> lower weight).

    CAVEAT: GR00T norm stats glob all data/*/*.parquet independently of the episode loader, so
    stats reflect the FULL set (not the 30% subset). Fine for a representative random draw
    (mean/std/quantiles barely move); materialize filtered dirs only if exact subset-only stats
    are mandated."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    from utils.subsample import episode_index_keep_set

    _orig = LeRobotEpisodeLoader._load_metadata

    def _patched(self):
        _orig(self)  # populates self.episodes_metadata; get_episode_lengths() runs AFTER this
        keep = episode_index_keep_set(self.dataset_path, num_demos, seed)
        if keep is not None:
            before = len(self.episodes_metadata)
            self.episodes_metadata = [e for e in self.episodes_metadata if e["episode_index"] in keep]
            print(
                f"[groot] subsample {self.dataset_path.name}: {len(self.episodes_metadata)}/{before} episodes",
                flush=True,
            )

    LeRobotEpisodeLoader._load_metadata = _patched
    print(f"[groot] episode subsample installed: keep first {num_demos} (seed {seed})", flush=True)


def build_and_run_groot(
    cfg,
    gs,
    *,
    start_from_checkpoint: str,
    visual_lr_scale: float = 1.0,
    episode_subsample_num_demos: int | None = None,
) -> None:
    """Translate (TrainConfigView, GroupedSoup) -> a GR00T config and run training.

    ``gs.groot_specs`` carries one (dirs, mix_ratio) per source group (3 for balancing-ON pretrain,
    1 for OFF / finetune). ``start_from_checkpoint`` is the base model (pretrain) or the phase-1
    checkpoint (finetune).
    """
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.experiment import run

    load_modality_config()
    tag = EmbodimentTag.NEW_EMBODIMENT.value

    datasets = [
        {"dataset_paths": list(spec.dirs), "mix_ratio": float(spec.mix_ratio), "embodiment_tag": tag}
        for spec in gs.groot_specs
    ]
    print(
        f"[groot] {len(datasets)} dataset group(s): "
        + ", ".join(f"{s.group}:{s.mix_ratio:.3f}({len(s.dirs)})" for s in gs.groot_specs),
        flush=True,
    )

    config = get_default_config().load_dict({"data": {"download_cache": False, "datasets": datasets}})
    config.load_config_path = None

    m, o, t = cfg.model, cfg.optim, cfg.train
    fr = m.get("freeze", {}) or {}
    aug = cfg.data.raw.get("augmentation") or {}

    # --- model / freeze / augmentation (the proven N1.7 recipe) ---
    config.model.tune_llm = bool(fr.get("tune_llm", False))
    config.model.tune_visual = bool(fr.get("tune_visual", False))
    config.model.tune_projector = bool(fr.get("tune_projector", True))
    config.model.tune_diffusion_model = bool(fr.get("tune_diffusion_model", True))
    config.model.state_dropout_prob = float(m.get("state_dropout_prob", 0.2))
    config.model.color_jitter_params = aug.get("color_jitter")  # dict[str,float] | None
    config.model.random_rotation_angle = aug.get("random_rotation_angle")
    config.model.extra_augmentation_config = None
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    # HF is only reached for the BACKBONE PROCESSOR (tokenizer + image/video preprocessors);
    # the weights always come from ``start_from_checkpoint`` (a local dir). That repo is GATED, so
    # a plain fetch needs HF_TOKEN — which the launch guardrails forbid in plaintext env and which
    # we cannot supply via Secrets Manager (the role has no secretsmanager:* today). The node
    # instead pre-seeds $HF_HOME/hub with the ~11 MB processor-only cache and sets this flag, which
    # makes transformers resolve from cache without a single Hub API call.
    # ``model_name`` deliberately stays the CANONICAL repo id (not a node path): it is serialized
    # into the output checkpoint's processor_config.json, and a local path there would poison every
    # downstream serve/eval that reconstructs the processor from the checkpoint.
    if os.environ.get("WSM_TRANSFORMERS_LOCAL_FILES_ONLY", "0") == "1":
        config.training.transformers_local_files_only = True
        print("[groot] transformers_local_files_only=True (pre-seeded HF processor cache)", flush=True)
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True  # no-op for OSC-delta keys (none are RELATIVE)

    # --- training ---
    num_gpus = int(os.environ.get("WSM_NUM_GPUS", "8"))
    config.training.experiment_name = t.get("exp_name", "groot17_rc365_pretrain")
    config.training.start_from_checkpoint = start_from_checkpoint
    # The pinned GR00T default is fused AdamW. Use it on H100 DDP unless explicitly disabled.
    fused_adamw = os.environ.get("WSM_FUSED_ADAMW", "1") == "1"
    config.training.optim = (
        "adamw_torch_fused"
        if num_gpus > 1 and fused_adamw
        else ("adamw_torch" if num_gpus > 1 else "paged_adamw_8bit")
    )
    config.training.use_ddp = True
    config.training.num_gpus = num_gpus
    config.training.global_batch_size = int(t.get("batch_size", 128))
    config.training.gradient_accumulation_steps = int(t.get("gradient_accumulation_steps", 1))
    # This is PER DDP RANK: 32 creates 256 workers on an 8-GPU p5.48xlarge.
    workers_per_rank = int(os.environ.get("WSM_NUM_WORKERS_PER_RANK", t.get("num_workers", 8)))
    if workers_per_rank < 0:
        raise ValueError(f"num_workers must be >= 0, got {workers_per_rank}")
    config.training.dataloader_num_workers = workers_per_rank
    config.training.learning_rate = float(o["lr"])
    config.training.weight_decay = float(o.get("weight_decay", 1e-5))
    config.training.warmup_ratio = float(o.get("warmup_ratio", 0.05))
    # env overrides (WSM_MAX_STEPS / WSM_SAVE_INTERVAL) let the small-task POC run a MODEST post-train with
    # frequent checkpoints (default to the YAML); ckpts persist in S3 (the entry's sync has no --delete) so a
    # per-ckpt eval can chart the curve. save_interval=5000 by default -> "evals every 5k updates".
    config.training.max_steps = int(os.environ.get("WSM_MAX_STEPS") or t.get("max_steps", 80000))
    config.training.save_steps = int(os.environ.get("WSM_SAVE_INTERVAL") or t.get("save_interval", 5000))
    # save_total_limit prunes OLD checkpoints LOCALLY. The co-located eval (doc 14) reads LOCAL ckpts, so a
    # small limit means it only ever sees the last N (the 80k groot CFG run was truncated to its last 5 by the
    # default 5). When co-located eval is on, keep ALL ckpts locally (None) so the full curve is evaluable;
    # otherwise keep the disk-hygiene default. (pi/openpi is unaffected — keep_period retains every 5k step.)
    if os.environ.get("WSM_EVAL_DURING_TRAIN") == "1":
        # keep ALL ckpts locally so the co-located eval (reads local) sees the full curve. Use a large INT
        # (not None) — gr00t's config->HF conversion path is int-typed, and None can break it (the groot
        # baseline-80k job died at setup with save_total_limit=None; an int == the proven default's type).
        _steps = int(os.environ.get("WSM_MAX_STEPS") or t.get("max_steps", 80000))
        _save = int(os.environ.get("WSM_SAVE_INTERVAL") or t.get("save_interval", 5000))
        config.training.save_total_limit = max(2, _steps // max(1, _save) + 2)
    else:
        config.training.save_total_limit = int(t.get("save_total_limit", 5))
    # WSM_OUTPUT_DIR lets the SageMaker FT entry redirect saves to the SM-agent-safe /opt/ml path.
    config.training.output_dir = os.environ.get("WSM_OUTPUT_DIR") or t.get("output_dir", "./checkpoints/groot17")
    config.training.use_wandb = os.environ.get("WSM_USE_WANDB", "1") == "1"
    config.training.wandb_project = os.environ.get("WANDB_PROJECT", "wsm-robocasa")

    if episode_subsample_num_demos is not None:
        install_episode_subsample(episode_subsample_num_demos)

    betas = tuple(float(x) for x in o.get("betas", (0.9, 0.999)))
    if len(betas) != 2 or not all(0.0 <= x < 1.0 for x in betas):
        raise ValueError(f"optim.betas must contain two values in [0,1), got {betas}")
    install_stage0_trainer(visual_lr_scale, adam_betas=betas)
    print(
        f"[groot] dispatching: steps={config.training.max_steps} bs={config.training.global_batch_size} "
        f"lr={config.training.learning_rate} optim={config.training.optim} betas={betas} "
        f"workers={workers_per_rank}/rank ({workers_per_rank * num_gpus}/node) "
        f"from={start_from_checkpoint}",
        flush=True,
    )
    run(config)
