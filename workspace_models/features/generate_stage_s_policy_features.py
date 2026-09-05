"""Stage-S strict omega producer: frozen WorkspaceModel -> omega_t per demo, exact 7,500/7,500.

Every input is SHA-verified and cross-checked before a single omega is written, and the output
manifest's ``encoder_id`` is the SHA-256 of the canonical ``encoder_provenance`` (never a
caller-invented id). It iterates the SOURCE-FEATURE manifest's declared episodes — never a glob,
never a silent ``continue`` — so a missing demo fails the run instead of shrinking coverage.

  <out-root>/<task>/demo_<ep>/w.npz   { w [F,512] fp16, frame_indices [F] int64, lang_global [2048] fp16 }
  <out-root>/<manifest-sha>.json      the pi05_workspace_omega manifest (validated in-process)

The chain checked here: task-prompt manifest SHA == source manifest's task_prompt_manifest.id ==
table manifest's task_prompt_manifest.id; source manifest id == table manifest's
source_features.manifest_id; the frozen pi feature-source inventory id is the same everywhere. Any
break aborts naming the edge. Run in the torch env; multi-GPU via ``--shard i/N``.

Dataset shape is parametrized by the shared ``--dataset-name/--task-dir-glob/--expected-tasks/
--demos-per-task(-map)/--seed`` flags (defaults = RoboCasa target50, so the target50 chain is
unchanged). A non-default ``--dataset-name`` additionally commits a ``dataset`` block into
``encoder_provenance``, which is what makes a second dataset over the SAME frozen encoder weights
resolve to a different ``encoder_id`` instead of colliding with the first cache's create-once
prefix (see ``validate_stage_s_policy_features.provenance_includes_dataset``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from workspace_models.networks.wsm_model import WorkspaceModel, WSMConfig

try:
    from scripts.launch import stage_s_provenance as prov
    from scripts.launch import validate_stage_s_policy_features as omega_val
    from scripts.launch import validate_stage_s_source_features as sfv
    from scripts.launch import validate_stage_s_task_lang_table as tlt
    from scripts.launch.validate_stage_s_task_prompts import load_task_prompts
except ImportError:  # on-node sys.path fallback
    import stage_s_provenance as prov  # type: ignore
    import validate_stage_s_policy_features as omega_val  # type: ignore
    import validate_stage_s_source_features as sfv  # type: ignore
    import validate_stage_s_task_lang_table as tlt  # type: ignore
    from validate_stage_s_task_prompts import load_task_prompts  # type: ignore

DONE_MARKER = ".done_policy_feats_v2"
_PRODUCING_CODE = [
    Path(__file__).resolve(),
    prov.REPO_ROOT / "workspace_models" / "networks" / "wsm_model.py",
    prov.REPO_ROOT / "workspace_models" / "networks" / "workspace_latent.py",
    prov.REPO_ROOT / "workspace_models" / "networks" / "keyframe_patch_head.py",
    prov.REPO_ROOT / "scripts" / "launch" / "validate_stage_s_policy_features.py",
    prov.REPO_ROOT / "scripts" / "launch" / "validate_stage_s_source_features.py",
]


def producing_code_sha256() -> str:
    return prov.stage_s_code_sha256(_PRODUCING_CODE)


def load_wsm_stage_s(ckpt_path: str | Path, device: str):
    """Load the FROZEN Stage-S WorkspaceModel, rebuilding the EXACT WSMConfig it was trained with.

    Refuses non-finite weights (the NaN-encoder lesson): a diverged encoder would emit NaN omega ->
    NaN modulator -> 0% eval."""
    blob = torch.load(Path(ckpt_path).expanduser(), map_location="cpu")
    if not isinstance(blob, dict) or "model" not in blob or "cfg" not in blob:
        raise ValueError(f"{ckpt_path} is not a Stage-S trainer checkpoint (need model+cfg)")
    cfg = blob["cfg"]
    model = WorkspaceModel(WSMConfig(**cfg)).to(device).eval()
    model.load_state_dict(blob["model"], strict=True)
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise ValueError(f"WSM encoder {ckpt_path} has NON-FINITE weights in {len(bad)} tensors; refusing")
    for p in model.parameters():
        p.requires_grad_(False)
    return model, blob


@torch.no_grad()
def encode_demo(model, feats_path: Path, patch_path: Path, task_lang: np.ndarray, device: str):
    """Encode one demo over its cached stride-8 grid -> (w [F,512] fp16, frame_indices, lang_global)."""
    patch = np.load(patch_path, mmap_mode="r")  # [F,192,2048] fp16
    with np.load(feats_path, allow_pickle=False) as f:
        proprio = np.asarray(f["lang_per_frame"], dtype=np.float16)  # [F,2048]
        frame_indices = f["frame_indices"].astype(np.int64)
    F = patch.shape[0]
    lang_global = np.asarray(task_lang, dtype=np.float16)  # [2048] sealed task vector
    cond_lang = np.broadcast_to(lang_global, (F, lang_global.shape[-1])).copy()
    pt = torch.from_numpy(np.asarray(patch))[None].to(device).float()
    pr = torch.from_numpy(proprio)[None].to(device).float()
    lg = torch.from_numpy(cond_lang)[None].to(device).float()
    w = model.encode(pt, pr, lg)[0]  # [F,512]
    if not torch.isfinite(w).all():
        raise ValueError(f"non-finite omega for {patch_path} — aborting precompute")
    return w.to(torch.float16).cpu().numpy(), frame_indices, lang_global


def _verify_chain(
    *,
    source_manifest: dict,
    table_manifest: dict,
    prompt_manifest_sha: str,
    feature_source_inventory_id: str,
) -> str:
    """Cross-check the provenance chain; return the source-feature manifest id."""
    src_id = sfv.source_manifest_id(source_manifest)
    if source_manifest["task_prompt_manifest"]["id"] != prompt_manifest_sha:
        raise ValueError("source manifest task_prompt_manifest.id != task-prompt manifest sha")
    if source_manifest["feature_source"]["checkpoint_inventory_id"] != feature_source_inventory_id:
        raise ValueError("source manifest feature_source inventory id mismatch")
    if table_manifest["task_prompt_manifest"]["id"] != prompt_manifest_sha:
        raise ValueError("table manifest task_prompt_manifest.id != task-prompt manifest sha")
    if table_manifest["source_features"]["manifest_id"] != src_id:
        raise ValueError("table manifest source_features.manifest_id != source manifest id")
    if table_manifest["feature_source"]["checkpoint_inventory_id"] != feature_source_inventory_id:
        raise ValueError("table manifest feature_source inventory id mismatch")
    return src_id


def generate(
    *,
    source_features_root: Path,
    source_features_manifest: Path,
    target_root: Path,
    labels_root: Path,
    task_lang_table: Path,
    task_lang_table_manifest: Path,
    task_prompt_manifest: Path,
    task_prompt_manifest_uri: str,
    feature_source_inventory_id: str,
    encoder_ckpt: Path,
    encoder_ckpt_uri: str,
    out_root: Path,
    study_root: str,
    shard: str = "0/1",
    device: str = "cpu",
    expected_tasks: int = 50,
    demos_per_task=150,
    seed: int = 0,
    dataset_name: str = omega_val.DATASET_NAME,
    task_dir_globs=omega_val.DEFAULT_TASK_DIR_GLOBS,
    task_prompt_artifact: str | None = None,
) -> None:
    """GPU pass: write w.npz for this shard's tasks (from the SOURCE manifest's declared episodes)."""
    if not task_prompt_manifest_uri.startswith("s3://"):
        raise ValueError("task_prompt_manifest_uri must be an s3:// URI")
    # SHA-verify every input, then validate the two upstream manifests fully.
    prompt_sha = prov.sha256_file(task_prompt_manifest)
    src_manifest = json.loads(Path(source_features_manifest).read_text(encoding="utf-8"))
    table_map = tlt.load_task_lang_table(
        task_lang_table_manifest,
        table_path=task_lang_table,
        expected_tasks=expected_tasks,
        dataset_name=dataset_name,
        seed=seed,
    )
    table_manifest = json.loads(Path(task_lang_table_manifest).read_text(encoding="utf-8"))
    sfv.validate_source_manifest(
        features_root=source_features_root,
        target_root=target_root,
        labels_root=labels_root,
        manifest_path=source_features_manifest,
        expected_task_prompt_manifest_id=prompt_sha,
        expected_feature_source_inventory_id=feature_source_inventory_id,
        expected_tasks=expected_tasks,
        demos_per_task=demos_per_task,
        seed=seed,
        dataset_name=dataset_name,
        task_dir_globs=task_dir_globs,
    )
    _verify_chain(
        source_manifest=src_manifest,
        table_manifest=table_manifest,
        prompt_manifest_sha=prompt_sha,
        feature_source_inventory_id=feature_source_inventory_id,
    )
    # confirm the prompt map is loadable and covers the tasks (fail-closed contract)
    load_task_prompts(
        task_prompt_manifest,
        target_root=target_root,
        expected_tasks=expected_tasks,
        task_dir_globs=task_dir_globs,
        **({"expected_artifact": task_prompt_artifact} if task_prompt_artifact else {}),
    )

    model, _blob = load_wsm_stage_s(encoder_ckpt, device)
    i, n = (int(x) for x in shard.split("/"))
    tasks = [r for k, r in enumerate(src_manifest["tasks"]) if k % n == i]
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    t0, written, skipped = time.time(), 0, 0
    for task_rec in tasks:
        task = task_rec["task"]
        task_lang = np.asarray(table_map[task], dtype=np.float16)
        for episode in task_rec["episodes"]:
            ep = int(episode["episode_index"])
            demo_rel = f"{task}/demo_{ep:06d}"
            out_dir = out_root / demo_rel
            if (out_dir / DONE_MARKER).exists() and (out_dir / "w.npz").exists():
                skipped += 1
                continue
            feats_path = source_features_root / episode["feats"]["path"]
            patch_path = source_features_root / episode["patch_tokens"]["path"]
            w, frame_indices, lang_global = encode_demo(model, feats_path, patch_path, task_lang, device)
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / ".w.npz.incomplete"
            with tmp.open("wb") as stream:
                np.savez(stream, w=w, frame_indices=frame_indices, lang_global=lang_global)
            os.replace(tmp, out_dir / "w.npz")
            marker = out_dir / (DONE_MARKER + ".incomplete")
            marker.write_text("ok")
            os.replace(marker, out_dir / DONE_MARKER)
            written += 1
    print(f"[omega] shard {shard}: {written} written, {skipped} skipped ({time.time() - t0:.0f}s)", flush=True)


def build_manifest(
    *,
    out_root: Path,
    target_root: Path,
    source_features_manifest: Path,
    task_lang_table_manifest: Path,
    task_prompt_manifest: Path,
    task_prompt_manifest_uri: str,
    feature_source_inventory_id: str,
    encoder_ckpt: Path,
    encoder_ckpt_uri: str,
    out_path: Path,
    expected_tasks: int = 50,
    demos_per_task=150,
    seed: int = 0,
    dataset_name: str = omega_val.DATASET_NAME,
    task_dir_globs=omega_val.DEFAULT_TASK_DIR_GLOBS,
    task_prompt_artifact: str | None = None,
) -> tuple[str, str]:
    """Assemble encoder_provenance -> encoder_id, enumerate w.npz files, write + validate the omega
    manifest. Returns (encoder_id, manifest_sha256)."""
    prompt_sha = prov.sha256_file(task_prompt_manifest)
    src_manifest = json.loads(Path(source_features_manifest).read_text(encoding="utf-8"))
    table_manifest = json.loads(Path(task_lang_table_manifest).read_text(encoding="utf-8"))
    src_id = _verify_chain(
        source_manifest=src_manifest,
        table_manifest=table_manifest,
        prompt_manifest_sha=prompt_sha,
        feature_source_inventory_id=feature_source_inventory_id,
    )

    blob = torch.load(Path(encoder_ckpt).expanduser(), map_location="cpu")
    ckpt_sha = prov.sha256_file(encoder_ckpt)
    if not encoder_ckpt_uri.endswith(f"{ckpt_sha}.pt"):
        raise ValueError("encoder-ckpt-uri must be content-addressed as <ckpt_sha>.pt")
    model_config = {k: (v if not isinstance(v, float) else float(v)) for k, v in blob["cfg"].items()}
    model_config["run_config_sha256"] = blob.get("run_config_sha256")

    provenance = {
        "encoder_checkpoint": {"uri": encoder_ckpt_uri, "sha256": ckpt_sha},
        "workspace_model": {
            "architecture": "WorkspaceModel-v1",
            "config": model_config,
            "step": int(blob.get("step", 0)),
        },
        "frozen_pi_feature_source": {"checkpoint_inventory_id": feature_source_inventory_id},
        "source_features": {"manifest_id": src_id},
        "conditioning": {
            "subgoal_dropout": 1.0,
            "global_language_mode": "canonical_terse_task_instruction",
            "canonical_task_prompt_manifest_id": prompt_sha,
            "canonical_task_prompt_manifest_uri": task_prompt_manifest_uri,
            # Every omega is conditioned on the sealed per-task language table, so encoder_id must
            # commit to its exact identity — the chain check alone only proves the table CLAIMS the
            # same source manifest, not that the bytes match (audit 2026-07-23).
            "task_lang_table_manifest_sha256": prov.sha256_file(task_lang_table_manifest),
        },
        "producing_code": {"sha256": producing_code_sha256()},
    }
    dataset_block = {
        "name": dataset_name,
        "episode_subsample_seed": seed,
        "demos_per_task": omega_val.canonical_demos_per_task(demos_per_task),
    }
    if omega_val.provenance_includes_dataset(dataset_name):
        # A second dataset over the SAME frozen encoder must not hash to the first dataset's
        # encoder_id (it would collide on the create-once $STUDY/caches/<id>/omega prefix).
        provenance["dataset"] = dataset_block
    encoder_id = prov.canonical_json_sha256(provenance)

    dataset_tasks = sfv._dataset_task_dirs(target_root, task_dir_globs=task_dir_globs)
    task_records = []
    for task in sorted(dataset_tasks):
        keep = sorted(
            sfv._expected_episode_ids(
                dataset_tasks[task],
                demos_per_task=omega_val.demos_for_task(demos_per_task, task),
                seed=seed,
            )
        )
        episodes = []
        for ep in keep:
            rel = f"{task}/demo_{ep:06d}/w.npz"
            path = out_root / rel
            if not path.is_file():
                raise ValueError(f"missing omega archive {path}")
            episodes.append(
                {
                    "episode_index": ep,
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "sha256": prov.sha256_file(path),
                }
            )
        task_records.append({"task": task, "episodes": episodes})

    manifest = {
        "schema_version": 1,
        "artifact": omega_val.ARTIFACT,
        "encoder_id": encoder_id,
        "encoder_provenance": provenance,
        "dataset": dataset_block,
        "tasks": task_records,
    }
    data = prov.canonical_json(manifest) + "\n"
    import hashlib

    manifest_sha = hashlib.sha256(data.encode("utf-8")).hexdigest()
    out_path = Path(out_path)
    if out_path.is_dir():
        out_path = out_path / f"{manifest_sha}.json"
    tmp = out_path.with_name(f".{out_path.name}.incomplete")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, out_path)

    omega_val.validate_manifest(
        features_root=out_root,
        target_root=target_root,
        manifest_path=out_path,
        encoder_id=encoder_id,
        expected_feature_source_inventory_id=feature_source_inventory_id,
        expected_task_prompt_manifest_id=prompt_sha,
        expected_task_prompt_manifest_uri=task_prompt_manifest_uri,
        expected_tasks=expected_tasks,
        demos_per_task=demos_per_task,
        seed=seed,
        dataset_name=dataset_name,
        task_dir_globs=task_dir_globs,
    )
    return encoder_id, manifest_sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-features-root", required=True)
    ap.add_argument("--source-features-manifest", required=True)
    ap.add_argument("--source-features-manifest-sha256", required=True)
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--task-lang-table", required=True)
    ap.add_argument("--task-lang-table-manifest", required=True)
    ap.add_argument("--task-lang-table-manifest-sha256", required=True)
    ap.add_argument("--task-prompt-manifest", required=True)
    ap.add_argument("--task-prompt-manifest-sha256", required=True)
    ap.add_argument("--task-prompt-manifest-uri", required=True)
    ap.add_argument("--feature-source-inventory-id", required=True)
    ap.add_argument("--encoder-ckpt", required=True)
    ap.add_argument("--encoder-ckpt-uri", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--task-prompt-artifact",
        default=None,
        help="expected task-prompt manifest artifact (default: the RoboCasa target50 one)",
    )
    omega_val.add_dataset_shape_args(ap)
    args = ap.parse_args()
    shape = omega_val.dataset_shape_kwargs(args)

    for path, sha, label in (
        (args.source_features_manifest, args.source_features_manifest_sha256, "source-features"),
        (args.task_lang_table_manifest, args.task_lang_table_manifest_sha256, "task-lang-table"),
        (args.task_prompt_manifest, args.task_prompt_manifest_sha256, "task-prompt"),
    ):
        if prov.sha256_file(path) != sha:
            raise SystemExit(f"{label} manifest sha256 does not match the file bytes")

    common = dict(
        target_root=Path(args.target_root).expanduser(),
        source_features_manifest=Path(args.source_features_manifest).expanduser(),
        task_lang_table_manifest=Path(args.task_lang_table_manifest).expanduser(),
        task_prompt_manifest=Path(args.task_prompt_manifest).expanduser(),
        task_prompt_manifest_uri=args.task_prompt_manifest_uri,
        feature_source_inventory_id=args.feature_source_inventory_id,
        encoder_ckpt=Path(args.encoder_ckpt).expanduser(),
        encoder_ckpt_uri=args.encoder_ckpt_uri,
        task_prompt_artifact=args.task_prompt_artifact,
        **shape,
    )
    if args.build_manifest:
        if not args.out:
            raise SystemExit("--build-manifest requires --out")
        encoder_id, manifest_sha = build_manifest(
            out_root=Path(args.out_root).expanduser(),
            out_path=Path(args.out).expanduser(),
            **common,
        )
        episodes = sum(
            omega_val.demos_for_task(shape["demos_per_task"], task)
            for task in omega_val._dataset_task_dirs(common["target_root"], task_dir_globs=shape["task_dir_globs"])
        )
        print(
            f"[omega] encoder_id={encoder_id} manifest_sha256={manifest_sha} "
            f"dataset={shape['dataset_name']} tasks={shape['expected_tasks']} "
            f"episodes={episodes} upload=false",
            flush=True,
        )
        return

    generate(
        source_features_root=Path(args.source_features_root).expanduser(),
        source_features_manifest=common["source_features_manifest"],
        target_root=common["target_root"],
        labels_root=Path(args.labels_root).expanduser(),
        task_lang_table=Path(args.task_lang_table).expanduser(),
        task_lang_table_manifest=common["task_lang_table_manifest"],
        task_prompt_manifest=common["task_prompt_manifest"],
        task_prompt_manifest_uri=args.task_prompt_manifest_uri,
        feature_source_inventory_id=args.feature_source_inventory_id,
        encoder_ckpt=common["encoder_ckpt"],
        encoder_ckpt_uri=args.encoder_ckpt_uri,
        out_root=Path(args.out_root).expanduser(),
        study_root=args.study_root,
        shard=args.shard,
        device=args.device,
        task_prompt_artifact=args.task_prompt_artifact,
        **shape,
    )


if __name__ == "__main__":
    main()
