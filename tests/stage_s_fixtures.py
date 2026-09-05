"""Shared synthetic fixtures for the Stage-S representation-chain tests.

Builds, at REAL production dims (patch [F,192,2048], lang [F,2048], omega [F,512]) but tiny frame
counts and task/demo counts, a self-consistent set of Stage-S artifacts:

  * a target dataset root with meta/episodes.jsonl (episode_index + one canonical `tasks` prompt),
    laid out so the omega/source validators derive the same seed-0 keep-set the manifest declares;
  * a canonical-terse source-feature cache (patch_tokens.npy + feats.npz{lang_per_frame,
    frame_indices}) + teacher-label npz + a validated source-feature manifest;
  * the canonical task-prompt manifest (via the real builder).

numpy + stdlib only — importable from the sm_launch, openpi-jax-latest, and pytorch envs alike.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from scripts.launch import build_stage_s_task_prompts as prompt_builder
from scripts.launch import stage_s_provenance as prov
from scripts.launch import validate_stage_s_source_features as sfv

STUDY_ROOT = "s3://bucket/study/long_context_v1"
INVENTORY_ID = "f" * 64
PATCH_GRID = sfv.PATCH_GRID
BACKBONE_DIM = sfv.BACKBONE_DIM
LANGUAGE_DIM = sfv.LANGUAGE_DIM
OMEGA_DIM = 512


def _kind(task_index: int) -> str:
    return "atomic" if task_index % 2 == 0 else "composite"


def make_target_dataset(
    target_root: Path,
    *,
    tasks: int = 2,
    demos: int = 3,
    spare: int = 2,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Write meta/episodes.jsonl per task; return the seed-0 keep-set (sorted) each task declares."""
    selected: dict[str, list[int]] = {}
    for task_index in range(tasks):
        task = f"Task{task_index:02d}"
        dataset = target_root / _kind(task_index) / task / "20260101" / "lerobot"
        (dataset / "meta").mkdir(parents=True)
        all_ids = list(range(task_index * 100, task_index * 100 + demos + spare))
        prompt = f"perform task {task_index}"
        with (dataset / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for episode_index in all_ids:
                stream.write(json.dumps({"episode_index": episode_index, "tasks": [prompt]}) + "\n")
        shuffled = list(all_ids)
        random.Random(seed).shuffle(shuffled)
        selected[task] = sorted(shuffled[:demos])
    return selected


def make_prompt_manifest(target_root: Path, output_dir: Path, *, tasks: int = 2) -> tuple[Path, str]:
    """Build the canonical task-prompt manifest with the real builder; return (path, id==sha)."""
    path, digest, _uri, _manifest = prompt_builder.build_task_prompt_manifest(
        target_root, output_dir=output_dir, study_root=STUDY_ROOT, expected_tasks=tasks
    )
    return path, digest


def _frame_indices(n: int, *, odd_last: bool = False) -> np.ndarray:
    idx = (np.arange(n) * sfv.FRAME_STRIDE).astype(np.int64)
    if odd_last and n >= 2:
        idx[-1] = idx[-2] + 3  # exercise the "include last frame" sub-stride final step
    return idx


def make_source_features(
    features_root: Path,
    labels_root: Path,
    target_root: Path,
    selected: dict[str, list[int]],
    *,
    prompt_manifest_id: str,
    frames: int = 3,
    inventory_id: str = INVENTORY_ID,
    unlabeled: set[tuple[str, int]] | None = None,
    producing_code_sha: str | None = None,
) -> tuple[Path, dict]:
    """Materialize the canonical feature cache + labels + a VALID source-feature manifest on disk."""
    unlabeled = unlabeled or set()
    rng = np.random.default_rng(0)
    task_records = []
    for task, ids in sorted(selected.items()):
        episodes = []
        for pos, episode_index in enumerate(ids):
            n = frames + (pos % 2)  # vary F a little
            demo_rel = f"{task}/demo_{episode_index:06d}"
            demo_dir = features_root / demo_rel
            demo_dir.mkdir(parents=True)
            patch = rng.standard_normal((n, PATCH_GRID, BACKBONE_DIM)).astype(np.float16)
            np.save(demo_dir / "patch_tokens.npy", patch)
            lang = rng.standard_normal((n, LANGUAGE_DIM)).astype(np.float16)
            frame_indices = _frame_indices(n, odd_last=(pos == 0))
            feats_path = demo_dir / "feats.npz"
            np.savez(feats_path, lang_per_frame=lang, frame_indices=frame_indices)

            available = (task, episode_index) not in unlabeled
            label_rel = f"{task}/vlm_episode_pi_{episode_index:06d}.npz"
            label_sha = None
            if available:
                label_path = labels_root / label_rel
                label_path.parent.mkdir(parents=True, exist_ok=True)
                keyframes = np.array([frame_indices[min(1, n - 1)]], dtype=np.int64)
                salient = np.array([np.array([0, 1])], dtype=object)
                np.savez(
                    label_path,
                    keyframes=keyframes,
                    salient_global=salient,
                    cumulative_global=salient,
                )
                label_sha = prov.sha256_file(label_path)

            patch_bytes = (demo_dir / "patch_tokens.npy").read_bytes()
            feats_bytes = feats_path.read_bytes()
            episodes.append(
                {
                    "episode_index": episode_index,
                    "patch_tokens": {
                        "path": f"{demo_rel}/patch_tokens.npy",
                        "size_bytes": len(patch_bytes),
                        "sha256": prov.sha256_file(demo_dir / "patch_tokens.npy"),
                        "shape": [n, PATCH_GRID, BACKBONE_DIM],
                        "dtype": "float16",
                    },
                    "feats": {
                        "path": f"{demo_rel}/feats.npz",
                        "size_bytes": len(feats_bytes),
                        "sha256": prov.sha256_file(feats_path),
                    },
                    "frame_count": n,
                    "teacher_labels": {
                        "available": available,
                        "path": label_rel,
                        "sha256": label_sha,
                    },
                }
            )
        task_records.append({"task": task, "episodes": episodes})

    if producing_code_sha is None:
        producing_code_sha = "e" * 64
    manifest = {
        "schema_version": 1,
        "artifact": sfv.ARTIFACT,
        "global_language_mode": sfv.GLOBAL_LANGUAGE_MODE,
        "dataset": {
            "name": sfv.DATASET_NAME,
            "episode_subsample_seed": 0,
            "demos_per_task": len(next(iter(selected.values()))),
        },
        "task_prompt_manifest": {
            "id": prompt_manifest_id,
            "uri": f"{STUDY_ROOT}/manifests/artifacts/workspace/task_prompts/"
            f"robocasa_target50/{prompt_manifest_id}.json",
        },
        "feature_source": {"checkpoint_inventory_id": inventory_id},
        "frame_grid": {"stride": sfv.FRAME_STRIDE, "include_last": True, "image_hw": [256, 256]},
        "producing_code": {"sha256": producing_code_sha},
        "tasks": task_records,
    }
    manifest_path = features_root / "source_features_manifest.json"
    manifest_path.write_text(prov.canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest_path, manifest


def full_source_fixture(
    tmp_path: Path,
    *,
    tasks: int = 2,
    demos: int = 3,
    frames: int = 3,
    unlabeled: set[tuple[str, int]] | None = None,
):
    """One call → (target_root, features_root, labels_root, prompt_id, manifest_path, manifest)."""
    target_root = tmp_path / "target"
    features_root = tmp_path / "features"
    labels_root = tmp_path / "labels"
    prompt_dir = tmp_path / "prompts"
    selected = make_target_dataset(target_root, tasks=tasks, demos=demos)
    _prompt_path, prompt_id = make_prompt_manifest(target_root, prompt_dir, tasks=tasks)
    manifest_path, manifest = make_source_features(
        features_root,
        labels_root,
        target_root,
        selected,
        prompt_manifest_id=prompt_id,
        frames=frames,
        unlabeled=unlabeled,
    )
    return target_root, features_root, labels_root, prompt_id, manifest_path, manifest
