"""Framework-agnostic glue shared by the base-VLA backbone adapters.

Both backbones (pi0.5/openpi/JAX and GR00T/PyTorch) start identically: load the nested YAML
recipe, resolve the shared ``GroupedSoup`` (soup + per-source balancing), and log it for run
provenance. Only AFTER this do the adapters diverge into framework-specific config building.

Imports ONLY the shared ``utils`` layer (numpy + robocasa) — never jax/torch — so it is safe to
import from either backbone venv.
"""

from __future__ import annotations

from utils import GroupedSoup, TrainConfigView, load_train_config, resolve_train_config_path


def load_recipe(config_path: str | None, *, backbone: str, phase: str) -> tuple[TrainConfigView, GroupedSoup]:
    """Resolve (--config or the canonical default) -> (typed recipe view, GroupedSoup).

    The GroupedSoup is the single framework-neutral DataSpec the adapter then translates into its
    native dataloader. Logs the resolved soup/balancing summary for provenance.
    """
    path = resolve_train_config_path(backbone, phase, config_path)
    cfg = load_train_config(path)
    if cfg.phase not in (phase, "target_finetune" if phase == "finetune" else phase):
        # soft check: warn rather than fail if the YAML phase label differs from the driver's
        print(f"[{backbone}] WARN: config phase={cfg.phase!r} but driver phase={phase!r}")
    print(f"[{backbone}/{cfg.phase}] recipe: {path}", flush=True)
    gs = GroupedSoup.from_data_view(cfg.data)
    print(gs.summary(), flush=True)
    return cfg, gs
