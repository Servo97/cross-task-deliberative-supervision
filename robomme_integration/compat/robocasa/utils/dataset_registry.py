"""Fail-closed placeholders for OpenPI's eagerly registered RoboCasa recipes."""

from __future__ import annotations


class _UnavailableSoup(tuple):
    def __new__(cls, name: str):
        value = super().__new__(cls)
        value.name = name
        return value

    def __iter__(self):
        raise RuntimeError(f"RoboCasa dataset soup {self.name!r} is unavailable in an isolated RoboMME job")


DATASET_SOUP_REGISTRY = {
    name: _UnavailableSoup(name)
    for name in (
        "target50",
        "target_atomic_seen",
        "target_composite_seen",
        "target_composite_unseen",
        "pretrain_human300_mg60",
        "pretrain_human300",
    )
}


def get_ds_meta(**_kwargs) -> dict:
    # One historical OpenPI config calls this while constructing its module-level registry.  The
    # sentinel cannot resolve to real data and therefore also fails closed if that config is used.
    return {"path": "__ROBOMME_IMPORT_ONLY__", "filter_key": None}
