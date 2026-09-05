"""ReMemBench task registry, task-set gating, and episode-manifest builder.

The load-bearing property is the gate: adding ReMemBench must not change what
`eval_common.list_tasks` / `aggregate_eval` do for any RoboCasa task set.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vla_training.eval.eval_manifest import validate_episode_manifest  # noqa: E402
from vla_training.eval.remembench_tasks import (  # noqa: E402
    REMEMBENCH_CATEGORIES,
    REMEMBENCH_CATEGORY_TASKS,
    REMEMBENCH_HORIZONS,
    REMEMBENCH_TASK_CATEGORIES,
    REMEMBENCH_TASK_SETS,
    REMEMBENCH_TASKS,
    heldout_split,
    is_remembench_task_set,
    list_remembench_tasks,
    summarize_by_category,
)


def _load_builder():
    path = REPO_ROOT / "scripts" / "launch" / "build_remembench_episode_manifest.py"
    spec = importlib.util.spec_from_file_location("_rmb_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thirteen_variants_cover_four_categories_with_horizons():
    assert len(REMEMBENCH_TASKS) == 13
    assert len(set(REMEMBENCH_TASKS)) == 13
    assert set(REMEMBENCH_TASK_CATEGORIES) == set(REMEMBENCH_TASKS)
    assert set(REMEMBENCH_HORIZONS) == set(REMEMBENCH_TASKS)
    assert set(REMEMBENCH_CATEGORIES) == set(REMEMBENCH_TASK_CATEGORIES.values())
    assert sum(len(v) for v in REMEMBENCH_CATEGORY_TASKS.values()) == 13
    # Prospective tasks need the long horizons; the flat RoboCasa 1000 would truncate them.
    assert REMEMBENCH_HORIZONS["MemHeatPotMultiple"] > REMEMBENCH_HORIZONS["MemHeatPot"] > 1000


def test_task_set_gate_only_claims_remembench_names():
    assert is_remembench_task_set("remembench")
    for category in REMEMBENCH_CATEGORIES:
        assert is_remembench_task_set(f"remembench_{category}")
    for name in ("atomic_seen", "composite_seen", "composite_unseen", "target50", ""):
        assert not is_remembench_task_set(name)
    assert set(REMEMBENCH_TASK_SETS["remembench"]) == set(REMEMBENCH_TASKS)


def test_list_remembench_tasks_matches_eval_common_entry_contract():
    entries = list_remembench_tasks("remembench")
    assert len(entries) == 13
    for entry in entries:
        assert set(entry) == {"task", "split_set", "horizon", "category"}
        assert entry["split_set"] == "remembench"
        assert entry["horizon"] == REMEMBENCH_HORIZONS[entry["task"]]
        assert entry["category"] == REMEMBENCH_TASK_CATEGORIES[entry["task"]]
    prospective = list_remembench_tasks("remembench_prospective")
    assert {e["task"] for e in prospective} == {"MemHeatPot", "MemHeatPotMultiple"}
    # a per-category set labels split_set with its own name, so results land in their own dir
    assert {e["split_set"] for e in prospective} == {"remembench_prospective"}


def test_eval_common_gate_leaves_robocasa_task_sets_alone(monkeypatch):
    """list_tasks must not import or consult the ReMemBench registry for RoboCasa sets,
    and must produce the exact same 3-key entries it did before."""
    from vla_training.eval import eval_common

    fake_registry = {"atomic_seen": ["CloseFridge", "TurnOnStove"]}
    fake_horizons = {"CloseFridge": 500, "TurnOnStove": 700}

    registry_mod = type(sys)("robocasa.utils.dataset_registry")
    registry_mod.TASK_SET_REGISTRY = fake_registry
    utils_mod = type(sys)("robocasa.utils.dataset_registry_utils")
    utils_mod.get_task_horizon = fake_horizons.__getitem__
    pkg = type(sys)("robocasa")
    pkg_utils = type(sys)("robocasa.utils")
    for name, module in {
        "robocasa": pkg,
        "robocasa.utils": pkg_utils,
        "robocasa.utils.dataset_registry": registry_mod,
        "robocasa.utils.dataset_registry_utils": utils_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert eval_common.list_tasks(["atomic_seen"]) == [
        {"task": "CloseFridge", "split_set": "atomic_seen", "horizon": 500},
        {"task": "TurnOnStove", "split_set": "atomic_seen", "horizon": 700},
    ]
    # `only=` filtering still keeps the correct split_set label and fails loud.
    assert eval_common.list_tasks(["atomic_seen"], only=["TurnOnStove"]) == [
        {"task": "TurnOnStove", "split_set": "atomic_seen", "horizon": 700}
    ]
    with pytest.raises(SystemExit):
        eval_common.list_tasks(["atomic_seen"], only=["NotATask"])


def test_eval_common_resolves_remembench_without_robocasa(monkeypatch):
    """A ReMemBench-only run must not need the RoboCasa registry at all."""
    from vla_training.eval import eval_common

    # Poison every robocasa module so any attempt to consult the RoboCasa registry raises.
    for name in list(sys.modules):
        if name == "robocasa" or name.startswith("robocasa."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in (
        "robocasa",
        "robocasa.utils",
        "robocasa.utils.dataset_registry",
        "robocasa.utils.dataset_registry_utils",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    entries = eval_common.list_tasks(["remembench"])
    assert len(entries) == 13
    assert {e["split_set"] for e in entries} == {"remembench"}


def test_by_category_rollup_is_unweighted_over_variants():
    rates = {
        "MemHeatPot": 0.0,
        "MemHeatPotMultiple": 0.5,
        "MemWashAndReturnLeft": 1.0,
    }
    summary = summarize_by_category(rates)
    assert set(summary) == {"prospective", "object_associative"}
    assert summary["prospective"]["mean"] == pytest.approx(0.25)
    assert summary["prospective"]["n_tasks_done"] == 2
    assert summary["prospective"]["n_tasks_expected"] == 2
    # partial coverage is reported, not silently averaged as if complete
    assert summary["object_associative"]["n_tasks_done"] == 1
    assert summary["object_associative"]["n_tasks_expected"] == 3
    assert summarize_by_category({}) == {}


def test_heldout_split_takes_the_tail_with_a_floor_of_three():
    train, heldout = heldout_split(range(50))
    assert heldout == list(range(40, 50))
    assert train + heldout == list(range(50))
    # 20% of 12 is 2.4 -> ceil 3, and the floor also holds
    assert heldout_split(range(12))[1] == [9, 10, 11]
    assert heldout_split(range(5))[1] == [2, 3, 4]
    # never more demos than exist
    assert heldout_split(range(2))[1] == [0, 1]
    assert heldout_split([]) == ([], [])


def _fake_demos(n, task):
    return [
        {
            "session": "s0",
            "demo_key": f"{task}_PandaOmron_demo_{i}",
            "demo_index": i,
            "length": 500 + i,
            "ep_meta": {"layout_id": 1, "style_id": 0, "lang": "do the thing", "n": i},
        }
        for i in range(n)
    ]


def test_manifest_builder_seals_a_valid_content_addressed_manifest():
    builder = _load_builder()
    per_task = {t: _fake_demos(20, t) for t in REMEMBENCH_TASKS}
    manifest, payload, digest = builder.build_manifest(per_task)

    validate_episode_manifest(manifest)
    assert manifest["task_sets"] == ["remembench"]
    assert "episodes_per_task" not in manifest
    assert len(manifest["episodes"]) == 13 * 4  # ceil(0.2*20) = 4 per task

    for record in manifest["episodes"]:
        reset = record["reset"]
        assert reset["kind"] == "remembench_ep_meta"
        assert reset["ep_meta"]["layout_id"] == 1 and reset["ep_meta"]["style_id"] == 0
        assert set(reset["source"]) >= {"task", "session", "demo_key"}
        assert record["horizon"] == REMEMBENCH_HORIZONS[record["task"]]
        assert record["category"] == REMEMBENCH_TASK_CATEGORIES[record["task"]]
        assert 0 <= record["seed"] <= 0x7FFFFFFF
        # the tail, not the head
        assert reset["source"]["demo_index"] >= 16

    # content addressing: byte-identical rebuild, and any input change moves the digest
    assert builder.build_manifest(per_task)[2] == digest
    assert json.loads(payload)["manifest_sha256"] == manifest["manifest_sha256"]
    per_task["MemHeatPot"][-1]["ep_meta"]["n"] = 9999
    assert builder.build_manifest(per_task)[2] != digest


def test_manifest_builder_rejects_ep_meta_without_a_scene_pin():
    builder = _load_builder()
    per_task = {t: _fake_demos(20, t) for t in REMEMBENCH_TASKS}
    for demo in per_task["MemHeatPot"]:
        demo["ep_meta"] = {"lang": "no layout pin"}
    with pytest.raises(ValueError, match="missing 'layout_id'"):
        builder.build_manifest(per_task)


def test_horizons_match_the_remembench_wrapper_when_it_is_importable():
    try:
        import robocasa.wrappers.gym_wrapper as gym_wrapper
    except Exception:
        pytest.skip("ReMemBench checkout not importable in this environment")
    if not hasattr(gym_wrapper, "TASK_HORIZONS"):
        pytest.skip("robocasa on the path is not the ReMemBench fork")
    assert gym_wrapper.TASK_HORIZONS == REMEMBENCH_HORIZONS
    assert gym_wrapper.TASK_CATEGORIES == REMEMBENCH_TASK_CATEGORIES
