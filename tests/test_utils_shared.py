"""Unit tests for the shared utils/ data layer (framework-agnostic; runs in any robocasa venv).

Run:  PYTHONPATH=<repo> /path/to/robocasa_env/bin/python tests/test_utils_shared.py
(also pytest-compatible)
"""

from __future__ import annotations

import numpy as np

from utils.balancing import GroupedSoup, compute_group_masses
from utils.config_schema import (
    DataBalancingView,
    DataView,
    default_config_path,
    load_train_config,
    normalize_backbone_token,
)
from utils.soup import (
    GROUPS,
    REMEMBENCH13_SOUP,
    combined_target_soup,
    remembench_soup,
    resolve_soup,
    source_group_of,
)


def test_backbone_tokens_and_paths():
    assert normalize_backbone_token("pi_05") == "pi05"
    assert normalize_backbone_token("pi05") == "pi05"
    assert normalize_backbone_token("groot_17") == "groot17"
    assert normalize_backbone_token("groot_n17") == "groot17"
    assert default_config_path("pi_05", "pretrain").name == "pi05_pretrain.yaml"
    assert default_config_path("groot_17", "pretrain").name == "groot17_pretrain.yaml"
    # phase 'finetune' (driver stem) and 'target_finetune' (yaml) both map to the real file
    assert default_config_path("groot_17", "finetune").name == "groot17_target_finetune.yaml"
    assert default_config_path("pi_05", "target_finetune").name == "pi05_target_finetune.yaml"


def test_load_pretrain_configs():
    for bb in ("pi_05", "groot_17"):
        cfg = load_train_config(default_config_path(bb, "pretrain"))
        assert cfg.phase == "pretrain"
        assert cfg.data.soup == "pretrain_human300_mg60"
        b = cfg.data.balancing
        assert b.enabled is True
        # 33/33/33 policy: equal weights + no cap
        assert b.mg_atomic_weight == b.human_atomic_weight == b.human_composite_weight == 1.0
        assert b.max_mg_atomic_fraction == 1.0


def test_grouped_soup_on():
    cfg = load_train_config(default_config_path("pi_05", "pretrain"))
    gs = GroupedSoup.from_data_view(cfg.data)
    print("\n" + gs.summary())
    # all three source groups present in the human+mg pretrain soup
    for g in GROUPS:
        assert gs.groups[g], f"group {g} unexpectedly empty"
    # masses ~1/3 each, sum to 1
    assert abs(sum(gs.group_masses.values()) - 1.0) < 1e-9
    for g in GROUPS:
        assert abs(gs.group_masses[g] - 1 / 3) < 1e-6, (g, gs.group_masses[g])
    # pi05 weights: aligned to soup order, primary == 1.0, per-group normalized prob == mass
    assert gs.pi05_weights is not None and len(gs.pi05_weights) == len(gs.soup)
    w = np.asarray(gs.pi05_weights)
    assert abs(w.max() - 1.0) < 1e-12, "no primary weight == 1.0"
    p = w / w.sum()  # the per-step sampling distribution openpi will realize
    for g in GROUPS:
        idx = [i for i, m in enumerate(gs.soup) if source_group_of(m) == g]
        assert abs(p[idx].sum() - gs.group_masses[g]) < 1e-6, (g, p[idx].sum())
    # groot specs: one per group, mix_ratio == 1/3
    assert len(gs.groot_specs) == 3
    assert all(abs(s.mix_ratio - 1 / 3) < 1e-6 for s in gs.groot_specs)
    assert sum(len(s.dirs) for s in gs.groot_specs) == len(gs.soup)


def test_grouped_soup_off():
    cfg = load_train_config(default_config_path("pi_05", "pretrain"))
    data_off = DataView(
        soup=cfg.data.soup,
        source=cfg.data.source,
        split=cfg.data.split,
        balancing=DataBalancingView(enabled=False),
    )
    gs = GroupedSoup.from_data_view(data_off)
    assert gs.group_masses == {}
    assert gs.pi05_weights is None  # -> openpi native size power-law
    assert len(gs.groot_specs) == 1
    assert gs.groot_specs[0].group == "all" and gs.groot_specs[0].mix_ratio == 1.0
    assert len(gs.groot_specs[0].dirs) == len(gs.soup)


def test_cap_policy():
    groups = {g: [{}] for g in GROUPS}  # all present, nonempty
    # old policy (mg=1,ha=1,hc=2, cap=0.5): base {mg:.25,ha:.25,hc:.5}; mg<cap -> unchanged
    m = compute_group_masses(
        groups,
        DataBalancingView(
            enabled=True,
            mg_atomic_weight=1,
            human_atomic_weight=1,
            human_composite_weight=2,
            max_mg_atomic_fraction=0.5,
        ),
    )
    assert abs(m["mg_atomic"] - 0.25) < 1e-9 and abs(m["human_composite"] - 0.5) < 1e-9
    # a capping case: mg heavily weighted (8:1:1 -> .8) but capped to .4; humans split the rest .6 equally
    m2 = compute_group_masses(
        groups,
        DataBalancingView(
            enabled=True,
            mg_atomic_weight=8,
            human_atomic_weight=1,
            human_composite_weight=1,
            max_mg_atomic_fraction=0.4,
        ),
    )
    assert abs(m2["mg_atomic"] - 0.4) < 1e-9
    assert abs(m2["human_atomic"] - 0.3) < 1e-9 and abs(m2["human_composite"] - 0.3) < 1e-9
    assert abs(sum(m2.values()) - 1.0) < 1e-9
    # OFF -> {}
    assert compute_group_masses(groups, DataBalancingView(enabled=False)) == {}


def test_combined_target_30():
    soup = combined_target_soup(0.30)
    assert len(soup) == 50, len(soup)  # 18 atomic_seen + 16 composite_seen + 16 composite_unseen
    assert all(m["source"] == "human" for m in soup)
    assert all(m["filter_key"] == "150_demos" for m in soup), {m["filter_key"] for m in soup}


def test_resolve_soup_is_copy():
    a = resolve_soup("target50")
    a[0]["filter_key"] = "MUTATED"
    b = resolve_soup("target50")
    assert b[0]["filter_key"] != "MUTATED", "resolve_soup must return a copy"


# ---------------------------------------------------------------------------------------------
# ReMemBench (flat layout, robocasa-free). These run in ANY venv — no robocasa import is reachable.
# ---------------------------------------------------------------------------------------------
RMB_TASKS = {"MemHeatPot": 4, "MemWashAndReturnLeft": 2, "MemRetrieveOilsFromCounterRL": 3}


def _fake_remembench_tree(tmp_path):
    import json

    root = tmp_path / "v1.0" / "target" / "train"
    for task, count in RMB_TASKS.items():
        meta = root / task / "20260803" / "lerobot" / "meta"
        meta.mkdir(parents=True)
        with (meta / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(count):
                stream.write(json.dumps({"episode_index": index, "tasks": [task]}) + "\n")
    return root


def test_remembench_soup_is_flat_full_mass_and_registry_free(tmp_path):
    root = _fake_remembench_tree(tmp_path)
    soup = remembench_soup(str(root))
    assert len(soup) == len(RMB_TASKS)
    assert {m["task"] for m in soup} == set(RMB_TASKS)
    # filter_key None => robocasa's get_subset_demos_filter_key keeps EVERY episode. No subsample.
    assert all(m["filter_key"] is None for m in soup)
    # An explicit group means source_group_of never touches the (ReMemBench-free) robocasa registry.
    assert all(m["group"] in GROUPS for m in soup)
    assert all(source_group_of(m) == m["group"] for m in soup)
    assert all(m["path"].endswith("/lerobot") for m in soup)


def test_remembench_soup_env_and_failure_modes(tmp_path, monkeypatch):
    root = _fake_remembench_tree(tmp_path)
    monkeypatch.delenv("WSM_SOUP_FROM_DIRS", raising=False)
    monkeypatch.setenv("WSM_REMEMBENCH_ROOT", str(root))
    assert len(remembench_soup()) == len(RMB_TASKS)
    # WSM_SOUP_FROM_DIRS is honoured as a fallback (used verbatim, not with v1.0/target appended).
    monkeypatch.delenv("WSM_REMEMBENCH_ROOT")
    monkeypatch.setenv("WSM_SOUP_FROM_DIRS", str(root))
    assert len(remembench_soup()) == len(RMB_TASKS)
    # WSM_TASKS still restricts the set, exactly as it does for the combined target soup.
    monkeypatch.setenv("WSM_TASKS", "MemHeatPot")
    assert [m["task"] for m in remembench_soup()] == ["MemHeatPot"]
    monkeypatch.delenv("WSM_TASKS")
    # Empty tree and unset root both fail loudly rather than training on nothing.
    with np.testing.assert_raises(RuntimeError):
        remembench_soup(str(tmp_path / "nonexistent"))
    monkeypatch.delenv("WSM_SOUP_FROM_DIRS")
    with np.testing.assert_raises(RuntimeError):
        remembench_soup()


def test_remembench_grouped_soup_resolves_with_balancing_off(tmp_path, monkeypatch):
    root = _fake_remembench_tree(tmp_path)
    monkeypatch.setenv("WSM_REMEMBENCH_ROOT", str(root))
    data = DataView(
        soup=REMEMBENCH13_SOUP,
        source="human",
        split="target",
        balancing=DataBalancingView(enabled=False),
    )
    gs = GroupedSoup.from_data_view(data)
    assert len(gs.soup) == len(RMB_TASKS)
    assert gs.group_masses == {}
    assert gs.pi05_weights is None  # -> openpi native mixing
    assert len(gs.groot_specs) == 1  # -> one "all" spec
    assert gs.groot_specs[0].group == "all" and gs.groot_specs[0].mix_ratio == 1.0
    assert gs.summary()  # no registry lookup, no crash
    # A subsample block on this soup is a recipe error, not a silent no-op.
    with np.testing.assert_raises(ValueError):
        GroupedSoup.from_data_view(
            DataView(
                soup=REMEMBENCH13_SOUP,
                source="human",
                split="target",
                balancing=DataBalancingView(enabled=False),
                subsample_fraction=0.30,
            )
        )


def test_remembench_yaml_configs_load_and_declare_their_arms():
    """The four rmb arms parse through the REAL schema and carry the intended interface flags."""
    from utils.config_schema import _CONFIG_TRAIN_DIR

    expected = {
        "base": {},
        "tanh": {"cond_type": "tanh"},
        "deltanet": {"cond_type": "gated_deltanet", "cond_window": 8},
        "jw01k16": {"jepa_weight": 0.1, "jepa_num_futures": 16},
    }
    for arm, flags in expected.items():
        cfg = load_train_config(_CONFIG_TRAIN_DIR / f"pi05_rmb_{arm}_finetune.yaml")
        assert cfg.data.soup == REMEMBENCH13_SOUP
        assert cfg.data.subsample_fraction is None  # every demo, no seed-0 selection
        assert cfg.data.balancing.enabled is False
        assert cfg.model["config_name"] == f"pi05_robocasa_rmb_{arm}"
        assert cfg.train["max_steps"] == 15000 and cfg.train["save_interval"] == 2500
        assert cfg.optim["warmup_steps"] == 500 and cfg.optim["decay_steps"] == 25000
        assert cfg.train["init_from"].endswith("mg60_bal33/run/149999/params")
        for key, value in flags.items():
            assert cfg.model[key] == value, (arm, key)
        for absent in {"cond_type", "cond_window", "jepa_weight", "jepa_num_futures"} - set(flags):
            assert absent not in cfg.model, (arm, absent)


if __name__ == "__main__":
    import inspect

    # Fixture-taking tests (tmp_path/monkeypatch) are pytest-only; the bare runner skips them.
    fns = [
        v
        for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v) and not inspect.signature(v).parameters
    ]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED (fixture-taking tests need pytest)")
