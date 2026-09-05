"""TODO-1 done gate: the whole Stage-S representation chain, offline, on a real-dim tiny fixture.

prompt manifest -> canonical source features -> sealed task-language table -> (trained) frozen
encoder -> exact omega cache -> omega manifest that the strict validator accepts, with the
encoder_id derived from provenance. Plus a rejection sweep proving any single tampering fails closed.

Real production dims (omega 512, lang 2048, patch 192x2048) are REQUIRED — the omega validator pins
them as constants — with a shrunk encoder (n_layers/k_slots) and tiny F/tasks/demos for speed.
Marked slow (real-dim CPU forward). Torch env.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import stage_s_fixtures as fx

from scripts.launch import build_stage_s_task_lang_table as tlt_builder
from scripts.launch import stage_s_provenance as prov
from scripts.launch import validate_stage_s_policy_features as omega_val
from workspace_models.features import build_pairs
from workspace_models.features import generate_stage_s_policy_features as omega
from workspace_models.train.train_wsm_base import _wsm_stage_s_train as st
from workspace_models.train.train_wsm_base.data import load_demo_pi_stage_s

TASKS, DEMOS = 2, 4
PROMPT_URI = "{s}/manifests/artifacts/workspace/task_prompts/robocasa_target50/{d}.json"


def _run_chain(tmp_path: Path):
    target, features, labels, prompt_id, src_manifest_path, _m = fx.full_source_fixture(
        tmp_path, tasks=TASKS, demos=DEMOS, frames=3
    )
    prompt_manifest = tmp_path / "prompts" / f"{prompt_id}.json"
    prompt_uri = PROMPT_URI.format(s=fx.STUDY_ROOT, d=prompt_id)

    # 1) task-language table
    table_manifest, table_id, _uri, _m2 = tlt_builder.build_task_lang_table(
        source_features_root=features,
        source_features_manifest=src_manifest_path,
        source_features_manifest_sha256=prov.sha256_file(src_manifest_path),
        target_root=target,
        labels_root=labels,
        task_prompt_manifest=prompt_manifest,
        task_prompt_manifest_sha256=prompt_id,
        task_prompt_manifest_uri=prompt_uri,
        feature_source_inventory_id=fx.INVENTORY_ID,
        output_dir=tmp_path / "table",
        study_root=fx.STUDY_ROOT,
        expected_tasks=TASKS,
        demos_per_task=DEMOS,
    )
    table_npz = tmp_path / "table" / "task_lang_table.npz"

    # 2) pairs + a (shrunk) trained encoder
    pairs = build_pairs.build_stage_s_pairs(
        source_features_manifest=src_manifest_path,
        features_root=features,
        labels_root=labels,
        n_partners=3,
        seed=0,
    )
    pairs_path = tmp_path / "pairs.parquet"
    pairs.to_parquet(pairs_path, index=False)
    from scripts.launch.validate_stage_s_task_lang_table import load_task_lang_table

    table_map = {
        k: np.asarray(v)
        for k, v in load_task_lang_table(table_manifest, table_path=table_npz, expected_tasks=TASKS).items()
    }
    cfg = dict(
        lambda_align=1.0,
        target_mode="next",
        dropout=1.0,
        steps=2,
        batch_size=2,
        lr=3e-4,
        warmup_steps=1,
        min_lr_frac=0.1,
        input_norm=False,
        val_frac=0.25,
        seed_split=1,
    )
    ckpt = st.train_stage_s(
        rows=pairs.to_dict("records"),
        load_fn=partial(load_demo_pi_stage_s, table=table_map),
        cfg=cfg,
        out_dir=tmp_path / "enc",
        device="cpu",
        num_workers=0,
        val_every=2,
        model_overrides={"n_layers": 1, "n_dec_layers": 1, "k_slots": 4, "n_heads": 4},
    )
    ckpt_sha = prov.sha256_file(ckpt)
    ckpt_uri = f"{fx.STUDY_ROOT}/artifacts/workspace/encoders/{ckpt_sha}.pt"

    # 3) omega generation + manifest
    out_root = tmp_path / "omega"
    common = dict(
        target_root=target,
        source_features_manifest=src_manifest_path,
        task_lang_table_manifest=table_manifest,
        task_prompt_manifest=prompt_manifest,
        task_prompt_manifest_uri=prompt_uri,
        feature_source_inventory_id=fx.INVENTORY_ID,
        encoder_ckpt=ckpt,
        encoder_ckpt_uri=ckpt_uri,
        expected_tasks=TASKS,
        demos_per_task=DEMOS,
        seed=0,
    )
    omega.generate(
        source_features_root=features,
        labels_root=labels,
        task_lang_table=table_npz,
        out_root=out_root,
        study_root=fx.STUDY_ROOT,
        shard="0/1",
        device="cpu",
        **common,
    )
    manifest_out = tmp_path / "omega_manifest.json"
    encoder_id, manifest_sha = omega.build_manifest(out_root=out_root, out_path=manifest_out, **common)
    return dict(
        target=target,
        features=features,
        labels=labels,
        out_root=out_root,
        manifest_out=manifest_out,
        encoder_id=encoder_id,
        prompt_id=prompt_id,
        prompt_uri=prompt_uri,
        common=common,
        table_npz=table_npz,
        ckpt=ckpt,
    )


@pytest.mark.slow
def test_end_to_end_chain_validates(tmp_path):
    ctx = _run_chain(tmp_path)
    # the strict omega validator accepts the produced cache + manifest
    summary = omega_val.validate_manifest(
        features_root=ctx["out_root"],
        target_root=ctx["target"],
        manifest_path=ctx["manifest_out"],
        encoder_id=ctx["encoder_id"],
        expected_feature_source_inventory_id=fx.INVENTORY_ID,
        expected_task_prompt_manifest_id=ctx["prompt_id"],
        expected_task_prompt_manifest_uri=ctx["prompt_uri"],
        expected_tasks=TASKS,
        demos_per_task=DEMOS,
        seed=0,
        workers=2,
    )
    assert summary == {"tasks": TASKS, "episodes": TASKS * DEMOS}
    # encoder_id is the canonical hash of the provenance, not caller-invented
    manifest = json.loads(Path(ctx["manifest_out"]).read_text())
    assert manifest["encoder_id"] == prov.canonical_json_sha256(manifest["encoder_provenance"])
    # every omega archive is the canonical contract
    w0 = ctx["out_root"] / manifest["tasks"][0]["episodes"][0]["path"]
    with np.load(w0) as a:
        assert a["w"].dtype == np.float16 and a["w"].shape[1] == omega_val.OMEGA_DIM
        assert a["lang_global"].shape == (omega_val.LANGUAGE_DIM,)


@pytest.mark.slow
@pytest.mark.parametrize("tamper", ["omega_byte", "delete_omega", "inventory", "encoder_bytes"])
def test_rejection_sweep(tmp_path, tamper):
    ctx = _run_chain(tmp_path)
    common = ctx["common"]
    if tamper == "omega_byte":
        manifest = json.loads(Path(ctx["manifest_out"]).read_text())
        path = ctx["out_root"] / manifest["tasks"][0]["episodes"][0]["path"]
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="sha256 mismatch"):
            omega_val.validate_manifest(
                features_root=ctx["out_root"],
                target_root=ctx["target"],
                manifest_path=ctx["manifest_out"],
                encoder_id=ctx["encoder_id"],
                expected_feature_source_inventory_id=fx.INVENTORY_ID,
                expected_task_prompt_manifest_id=ctx["prompt_id"],
                expected_task_prompt_manifest_uri=ctx["prompt_uri"],
                expected_tasks=TASKS,
                demos_per_task=DEMOS,
                seed=0,
                workers=2,
            )
    elif tamper == "delete_omega":
        manifest = json.loads(Path(ctx["manifest_out"]).read_text())
        (ctx["out_root"] / manifest["tasks"][0]["episodes"][0]["path"]).unlink()
        with pytest.raises(ValueError, match="missing omega archive"):
            omega.build_manifest(out_root=ctx["out_root"], out_path=tmp_path / "m2.json", **common)
    elif tamper == "inventory":
        bad = dict(common)
        bad["feature_source_inventory_id"] = "0" * 64
        with pytest.raises(ValueError, match="inventory id"):
            omega.build_manifest(out_root=ctx["out_root"], out_path=tmp_path / "m3.json", **bad)
    elif tamper == "encoder_bytes":
        # a different ckpt uri (not content-addressed to its bytes) is rejected
        bad = dict(common)
        bad["encoder_ckpt_uri"] = f"{fx.STUDY_ROOT}/artifacts/workspace/encoders/{'0' * 64}.pt"
        with pytest.raises(ValueError, match="content-addressed"):
            omega.build_manifest(out_root=ctx["out_root"], out_path=tmp_path / "m4.json", **bad)
