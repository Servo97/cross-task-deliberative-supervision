"""wsmv2 shared, framework-agnostic plumbing.

This package owns the DATA SPECIFICATION shared by both backbones (pi0.5 / openpi / JAX and
GR00T N1.7 / PyTorch). It depends ONLY on stdlib + numpy + ``robocasa`` (the pure-python
registry) and must NEVER import jax or torch, so it imports cleanly in either backbone venv.

Boundary: pi0.5 and GR00T cannot share a Dataset object (different frameworks). They share the
*spec* — which tasks/sources, the per-source-group balancing masses, the config schema — and
each backbone adapter (vla_training/train/train_base/*.py) translates that spec into its native
dataloader (openpi ``dataset_weights`` vs GR00T ``dataset_spec`` mix_ratio). Compute-once, apply-twice.
"""

from utils.balancing import (
    GrootGroupSpec,
    GroupedSoup,
    compute_group_masses,
    groot_group_specs,
    pi05_dataset_weights,
)
from utils.config_schema import (
    DataBalancingView,
    DataView,
    EvalConfigView,
    TrainConfigView,
    default_config_path,
    default_eval_config_path,
    load_eval_config,
    load_train_config,
    normalize_backbone_token,
    resolve_train_config_path,
)
from utils.soup import (
    GROUPS,
    REMEMBENCH13_SOUP,
    combined_target_soup,
    dirs_for_group,
    partition_by_group,
    remembench_soup,
    resolve_soup,
    source_group_of,
)
from utils.subsample import episode_index_keep_set, num_demos_from_filter_key, uniform_num_demos

__all__ = [
    # soup / registry
    "GROUPS",
    "resolve_soup",
    "combined_target_soup",
    "source_group_of",
    "partition_by_group",
    "dirs_for_group",
    "REMEMBENCH13_SOUP",
    "remembench_soup",
    # subsample (deterministic episode selection, robocasa-parity)
    "episode_index_keep_set",
    "num_demos_from_filter_key",
    "uniform_num_demos",
    # balancing
    "GroupedSoup",
    "GrootGroupSpec",
    "compute_group_masses",
    "pi05_dataset_weights",
    "groot_group_specs",
    # config schema
    "TrainConfigView",
    "EvalConfigView",
    "DataView",
    "DataBalancingView",
    "load_train_config",
    "load_eval_config",
    "default_config_path",
    "default_eval_config_path",
    "resolve_train_config_path",
    "normalize_backbone_token",
]
