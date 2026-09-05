"""Offline tests for the sealed Stage-S per-task language table (builder + validator)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import stage_s_fixtures as fx

from scripts.launch import build_stage_s_task_lang_table as builder
from scripts.launch import stage_s_provenance as prov
from scripts.launch import validate_stage_s_task_lang_table as tlt


def _build(tmp_path, *, tasks=2, demos=3):
    target, features, labels, prompt_id, manifest_path, _m = fx.full_source_fixture(tmp_path, tasks=tasks, demos=demos)
    prompt_manifest = tmp_path / "prompts" / f"{prompt_id}.json"
    out = tmp_path / "table"
    path, table_id, uri, manifest = builder.build_task_lang_table(
        source_features_root=features,
        source_features_manifest=manifest_path,
        source_features_manifest_sha256=prov.sha256_file(manifest_path),
        target_root=target,
        labels_root=labels,
        task_prompt_manifest=prompt_manifest,
        task_prompt_manifest_sha256=prompt_id,
        task_prompt_manifest_uri=f"{fx.STUDY_ROOT}/manifests/artifacts/workspace/task_prompts/robocasa_target50/{prompt_id}.json",
        feature_source_inventory_id=fx.INVENTORY_ID,
        output_dir=out,
        study_root=fx.STUDY_ROOT,
        expected_tasks=tasks,
        demos_per_task=demos,
    )
    return dict(
        target=target,
        features=features,
        labels=labels,
        prompt_id=prompt_id,
        source_manifest=manifest_path,
        path=path,
        table_id=table_id,
        uri=uri,
        manifest=manifest,
        out=out,
    )


def test_build_and_validate_roundtrip(tmp_path):
    ctx = _build(tmp_path)
    summary = tlt.validate_task_lang_table(ctx["path"], expected_tasks=2)
    assert summary == {"tasks": 2}
    # spot-check the mean matches a float64 recompute over the source features
    manifest = json.loads(Path(ctx["source_manifest"]).read_text())
    mapping = tlt.load_task_lang_table(ctx["path"], expected_tasks=2)
    for row in manifest["tasks"]:
        acc = np.zeros(tlt.LANGUAGE_DIM, np.float64)
        n = 0
        for ep in row["episodes"]:
            with np.load(ctx["features"] / ep["feats"]["path"]) as a:
                lp = np.asarray(a["lang_per_frame"], np.float64)
            acc += lp.sum(0)
            n += lp.shape[0]
        expected = (acc / n).astype(np.float16)
        assert np.array_equal(mapping[row["task"]], expected)
    assert mapping[manifest["tasks"][0]["task"]].dtype == np.float16


@pytest.mark.parametrize("mutation", ["prompt", "drop_demo", "inventory", "feats_byte", "code"])
def test_any_input_changes_table_id(tmp_path, mutation):
    ctx = _build(tmp_path)
    base = ctx["table_id"]
    m = json.loads(json.dumps(ctx["manifest"]))
    if mutation == "prompt":
        m["task_prompt_manifest"]["id"] = "1" * 64
    elif mutation == "drop_demo":
        m["dataset"]["demos_per_task"] = 2
    elif mutation == "inventory":
        m["feature_source"]["checkpoint_inventory_id"] = "2" * 64
    elif mutation == "feats_byte":
        m["source_features"]["manifest_id"] = "3" * 64
    elif mutation == "code":
        m["producing_code"]["sha256"] = "4" * 64
    assert tlt.table_id(m) != base


def test_missing_demo_fails_closed(tmp_path):
    ctx = _build(tmp_path)
    # Corrupt the source manifest to reference a feats file we then delete, and rebuild.
    manifest = json.loads(Path(ctx["source_manifest"]).read_text())
    victim = ctx["features"] / manifest["tasks"][0]["episodes"][0]["feats"]["path"]
    victim.unlink()
    out2 = tmp_path / "table2"
    # The source manifest's sha no longer matches (a declared file vanished), so the build fails
    # closed at the sha gate before reading anything.
    with pytest.raises(ValueError):
        builder.build_task_lang_table(
            source_features_root=ctx["features"],
            source_features_manifest=ctx["source_manifest"],
            source_features_manifest_sha256=prov.sha256_file(ctx["source_manifest"]),
            target_root=ctx["target"],
            labels_root=ctx["labels"],
            task_prompt_manifest=tmp_path / "prompts" / f"{ctx['prompt_id']}.json",
            task_prompt_manifest_sha256=ctx["prompt_id"],
            task_prompt_manifest_uri=ctx["uri"],
            feature_source_inventory_id=fx.INVENTORY_ID,
            output_dir=out2,
            study_root=fx.STUDY_ROOT,
            expected_tasks=2,
            demos_per_task=3,
        )
    if out2.exists():
        assert not list(out2.glob(".*.incomplete"))  # no partial-write residue


def test_expanded_key_rejected(tmp_path):
    ctx = _build(tmp_path)
    npz = ctx["out"] / "task_lang_table.npz"
    with np.load(npz) as a:
        tasks, lang = a["tasks"], a["lang"]
    np.savez(npz, tasks=tasks, lang=lang, expanded=np.array(["x", "y"]))
    # sha will now mismatch; disable sha to reach the key check
    with pytest.raises(ValueError, match="expanded"):
        tlt.load_task_lang_table(ctx["path"], expected_tasks=2, verify_sha=False)


def test_legacy_builder_fails_closed(monkeypatch, tmp_path):
    import workspace_models.features.make_task_lang_table as legacy

    monkeypatch.setattr(
        "sys.argv",
        ["make_task_lang_table", "--root", str(tmp_path), "--out", str(tmp_path / "t.npz")],
    )
    with pytest.raises(SystemExit) as excinfo:
        legacy.main()
    assert "build_stage_s_task_lang_table.py" in str(excinfo.value)


def test_task_set_must_match(tmp_path):
    ctx = _build(tmp_path)
    with pytest.raises(ValueError, match="task set differs"):
        tlt.load_task_lang_table(ctx["path"], expected_task_names={"Task00", "Nope"}, expected_tasks=2)


def test_producing_code_sha_stable():
    assert builder.producing_code_sha256() == builder.producing_code_sha256()
    assert len(builder.producing_code_sha256()) == 64
