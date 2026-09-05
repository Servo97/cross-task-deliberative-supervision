"""Typed views of the nested train/eval YAML recipes + one loader, shared by both backbones.

Fixes two live scaffold bugs the stubs have today:
  * the stub drivers do ``Cfg(**raw)`` on the FLAT YAML, which crashes on the real NESTED
    files (regime/backbone/phase + model/data/optim/train/eval blocks);
  * ``scripts/train/vla_base/train.sh`` derives default config names like ``pi_05_pretrain.yaml``
    / ``groot_17_finetune.yaml`` that do not exist — the real files are ``pi05_pretrain.yaml`` /
    ``groot17_target_finetune.yaml``. ``default_config_path`` reconciles the tokens.

Pure dataclasses + pyyaml. No framework imports; backbone-specific blocks (model/optim/train)
are passed through as raw dicts for each adapter to interpret in its own framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_TRAIN_DIR = REPO_ROOT / "scripts" / "configs" / "train"
_CONFIG_EVAL_DIR = REPO_ROOT / "scripts" / "configs" / "eval"

# Exact, validated key sets — unknown keys raise (cheap guard against silent mixture drift).
_BALANCING_KEYS = {
    "enabled",
    "max_mg_atomic_fraction",
    "mg_atomic_weight",
    "human_atomic_weight",
    "human_composite_weight",
    "strategy",
}
_BALANCING_STRATEGIES = {"per_batch_resample", "source_weighted", "none"}


def normalize_backbone_token(s: str) -> str:
    """Collapse the several spellings to the canonical config-file stem.
    train.sh/--backbone use ``pi_05``/``groot_17``; YAML ``backbone:`` uses ``pi05``/``groot_n17``."""
    t = s.strip().lower().replace("-", "_")
    if t in {"pi05", "pi_05", "pi0_5", "pi0.5"}:
        return "pi05"
    if t in {"groot17", "groot_17", "groot_n17", "grootn17", "gr00t_n17", "gr00t17"}:
        return "groot17"
    raise ValueError(f"unrecognized backbone token: {s!r}")


def _phase_stem(phase: str) -> str:
    p = phase.strip().lower()
    if p == "pretrain":
        return "pretrain"
    if p in {"finetune", "target_finetune"}:
        return "target_finetune"
    raise ValueError(f"unrecognized phase: {phase!r}")


def default_config_path(backbone: str, phase: str) -> Path:
    """Map (backbone, phase) -> the REAL train YAML path (e.g. groot17_pretrain.yaml)."""
    return _CONFIG_TRAIN_DIR / f"{normalize_backbone_token(backbone)}_{_phase_stem(phase)}.yaml"


def default_eval_config_path(backbone: str) -> Path:
    """Map backbone -> the REAL eval YAML path (e.g. groot17_eval.yaml / pi05_eval.yaml)."""
    return _CONFIG_EVAL_DIR / f"{normalize_backbone_token(backbone)}_eval.yaml"


@dataclass(frozen=True)
class DataBalancingView:
    """Typed mirror of the YAML ``data.data_balancing`` block. Framework-agnostic; only the
    APPLICATION differs (see utils.balancing). Defaults = disabled / no-op."""

    enabled: bool = False
    max_mg_atomic_fraction: float = 1.0  # cap on the mg_atomic group's expected mass
    mg_atomic_weight: float = 1.0
    human_atomic_weight: float = 1.0
    human_composite_weight: float = 1.0
    strategy: str = "per_batch_resample"

    @classmethod
    def from_yaml(cls, d: dict | None) -> "DataBalancingView":
        d = d or {}
        unknown = set(d) - _BALANCING_KEYS
        if unknown:
            raise ValueError(f"unknown data_balancing keys {sorted(unknown)}; allowed {sorted(_BALANCING_KEYS)}")
        strat = str(d.get("strategy", "per_batch_resample"))
        if strat not in _BALANCING_STRATEGIES:
            raise ValueError(f"strategy={strat!r} not in {sorted(_BALANCING_STRATEGIES)}")
        return cls(
            enabled=bool(d.get("enabled", False)),
            max_mg_atomic_fraction=float(d.get("max_mg_atomic_fraction", 1.0)),
            mg_atomic_weight=float(d.get("mg_atomic_weight", 1.0)),
            human_atomic_weight=float(d.get("human_atomic_weight", 1.0)),
            human_composite_weight=float(d.get("human_composite_weight", 1.0)),
            strategy=strat,
        )


@dataclass(frozen=True)
class DataView:
    """Typed mirror of the YAML ``data`` block."""

    soup: str
    source: str
    split: str
    balancing: DataBalancingView
    # demo_fraction for the (target) subsample; None => native full (filter applied per-meta).
    subsample_fraction: float | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, d: dict) -> "DataView":
        # Lenient on the data block: backbone-specific extensions (e.g. GR00T data.augmentation)
        # are expected and preserved in .raw. Required keys are enforced by direct access below;
        # only data_balancing is strictly key-validated (mixture-drift guard).
        sub = d.get("subsample") or {}
        frac = sub.get("target_fraction") if isinstance(sub, dict) else None
        return cls(
            soup=str(d["soup"]),
            source=str(d.get("source", "human")),
            split=str(d["split"]),
            balancing=DataBalancingView.from_yaml(d.get("data_balancing")),
            subsample_fraction=(float(frac) if frac is not None else None),
            raw=dict(d),
        )


@dataclass(frozen=True)
class TrainConfigView:
    """Top-level nested view of a train recipe. model/optim/train stay raw for the adapter."""

    regime: str
    backbone: str
    phase: str
    model: dict
    data: DataView
    optim: dict
    train: dict
    eval: dict | None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, d: dict) -> "TrainConfigView":
        for k in ("regime", "backbone", "phase", "model", "data", "optim", "train"):
            if k not in d:
                raise ValueError(f"train config missing required top-level key {k!r}")
        return cls(
            regime=str(d["regime"]),
            backbone=str(d["backbone"]),
            phase=str(d["phase"]),
            model=dict(d["model"]),
            data=DataView.from_yaml(d["data"]),
            optim=dict(d["optim"]),
            train=dict(d["train"]),
            eval=(dict(d["eval"]) if d.get("eval") else None),
            raw=dict(d),
        )


@dataclass(frozen=True)
class EvalConfigView:
    """Eval-phase view. ``split`` is the gym SCENE split, resolved from the nested ``eval:`` block;
    it MUST be 'target' for foundation-model-learning eval (the legacy 'pretrain' default was the
    bug flagged in internal_planning_and_todos/00_START_HERE.md). ``task_sets`` may be plain strings
    or ``{name, num_tasks}`` mappings; we keep the ordered names and the per-set expected counts."""

    regime: str
    backbone: str
    split: str
    task_sets: list[str]  # ordered task-set names (atomic_seen, composite_seen, ...)
    task_num: dict  # name -> expected num_tasks (for the task-weighted metric)
    num_trials: int
    num_workers: int
    exec_steps: int  # GR00T: action-chunk steps executed per inference
    replan_steps: int  # pi0.5: replan interval (reference default 5, NOT exec_steps)
    seed: int
    video: str
    execution_order: str
    checkpoint: str | None
    output_dir: str | None
    model: dict
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, d: dict) -> "EvalConfigView":
        ev = d.get("eval") or {}
        split = str(ev.get("split", "target"))
        if split != "target":
            raise ValueError(
                f"eval split={split!r}; foundation-model-learning eval must use 'target' "
                "(see 00_START_HERE.md: the 'pretrain' default was the legacy bug)"
            )
        raw_sets = ev.get("task_sets") or ["atomic_seen", "composite_seen", "composite_unseen"]
        names, task_num = [], {}
        for ts in raw_sets:
            if isinstance(ts, dict):
                names.append(str(ts["name"]))
                if "num_tasks" in ts:
                    task_num[str(ts["name"])] = int(ts["num_tasks"])
            else:
                names.append(str(ts))
        model = dict(d.get("model") or {})
        return cls(
            regime=str(d.get("regime", "vla_base")),
            backbone=str(d.get("backbone", "")),
            split=split,
            task_sets=names,
            task_num=task_num,
            num_trials=int(ev.get("num_trials", 50)),
            num_workers=int(ev.get("num_workers", 8)),
            exec_steps=int(ev.get("exec_steps", 8)),
            replan_steps=int(ev.get("replan_steps", 5)),
            seed=int(ev.get("seed", 7)),
            video=str(ev.get("video", "first")),
            execution_order=str(ev.get("execution_order", "atomic_first")),
            checkpoint=(str(model["checkpoint"]) if model.get("checkpoint") else None),
            output_dir=(str(ev["output_dir"]) if ev.get("output_dir") else None),
            model=model,
            raw=dict(d),
        )


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def load_train_config(path: str | Path) -> TrainConfigView:
    return TrainConfigView.from_yaml(_load_yaml(path))


def load_eval_config(path: str | Path) -> EvalConfigView:
    return EvalConfigView.from_yaml(_load_yaml(path))


def resolve_train_config_path(backbone: str, phase: str, explicit: str | None = None) -> Path:
    """Used by drivers: honor an explicit --config, else the canonical default path."""
    if explicit:
        return Path(explicit)
    return default_config_path(backbone, phase)
