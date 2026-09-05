"""Stage-S feature producer: cache frozen pi0.5 tap features with the CANONICAL TERSE prompt.

Replaces the historical ``pi_cache_features`` (which tapped the Qwen-EXPANDED prompt and stored
per-demo language + subgoals). This producer feeds the frozen H300+MG pi tap ONLY the demo-
independent canonical terse task instruction (from the Stage-S task-prompt manifest), and writes
the minimal canonical schema:

  <cache_root>/<task>/demo_{ep:06d}/
    patch_tokens.npy  [F,192,2048] fp16   (bin-averaged SigLIP grids, LABEL view order via the tap)
    feats.npz         { lang_per_frame [F,2048] fp16, frame_indices [F] int64 }
    .done_features_v2                      (written only after a reopen re-validation)

Task-level language lives ONLY in the packet-01 task-language table; teacher labels
(vlm_episode_pi_*.npz) live under a separate labels root and are recorded per-demo (available or
not) — a missing teacher label never shrinks frozen-feature coverage. The producer never consumes
``expanded_prompt`` or ``subgoals``.

The seed-0 T30 keep-set (150 demos/task) is taken from the exact same helper the validators use, so
frames must have been extracted for precisely those episodes (the node entry runs
``extract_frames --num-demos 150 --seed 0 --stride 8`` first).

Two passes:
  --shard i/N   GPU tap pass over tasks[k % N == i]   (writes cache + .done_features_v2)
  --build-manifest   single-process, no GPU: assemble + validate the source-feature manifest

Run the GPU pass in the openpi-jax-latest env (jax + openpi + robocasa).

Dataset shape is parametrized by the shared ``--dataset-name/--task-dir-glob/--expected-tasks/
--demos-per-task(-map)/--seed`` flags (defaults = RoboCasa target50). ``--lerobot-dir-source``
selects where ``observation.state`` is read from: ``soup`` (default, the RoboCasa registry via
``utils.soup``) or ``target-root``, which uses the very lerobot directory the task glob already
found — the robocasa-free path a non-RoboCasa dataset needs.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from workspace_models.labels.geometry import VIEWS

try:
    from scripts.launch import stage_s_provenance as prov
    from scripts.launch import validate_stage_s_source_features as sfv
    from scripts.launch import validate_stage_s_task_prompts as prompts
except ImportError:  # on-node sys.path fallback
    import stage_s_provenance as prov  # type: ignore
    import validate_stage_s_source_features as sfv  # type: ignore
    import validate_stage_s_task_prompts as prompts  # type: ignore

DONE_MARKER = ".done_features_v2"
# The exact, ordered code inputs whose bytes define this producer's identity.
_PRODUCING_CODE = [
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "pi_backbone_tap.py",
    prov.REPO_ROOT / "scripts" / "launch" / "validate_stage_s_source_features.py",
    prov.REPO_ROOT / "scripts" / "launch" / "validate_stage_s_task_prompts.py",
]


def producing_code_sha256() -> str:
    return prov.stage_s_code_sha256(_PRODUCING_CODE)


def _state_at(lerobot_dir: Path, ep: int, frame_indices: np.ndarray) -> np.ndarray:
    """observation.state at the extracted frame_indices -> [F, D] (as in pi_cache_features)."""
    import pandas as pd

    df = pd.read_parquet(lerobot_dir / f"data/chunk-000/episode_{ep:06d}.parquet")
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    return state[frame_indices]


def _reopen_ok(demo_dir: Path, n: int) -> bool:
    """Re-read what we just wrote and re-check the canonical contract before the done marker."""
    try:
        patch = np.load(demo_dir / "patch_tokens.npy", mmap_mode="r")
        with np.load(demo_dir / "feats.npz") as feats:
            keys = set(feats.files)
            lang = feats["lang_per_frame"]
            frame_indices = feats["frame_indices"]
    except (OSError, EOFError, KeyError, ValueError):
        return False
    if patch.shape != (n, sfv.PATCH_GRID, sfv.BACKBONE_DIM) or patch.dtype != np.float16:
        return False
    if keys != set(sfv.FEATS_KEYS) or lang.shape != (n, sfv.LANGUAGE_DIM) or lang.dtype != np.float16:
        return False
    if frame_indices.dtype != np.int64 or frame_indices.shape != (n,):
        return False
    if not np.isfinite(np.asarray(lang, dtype=np.float32)).all():
        return False
    return True


def cache_task(
    tap,
    task: str,
    keep_ids: list[int],
    frames_dir: Path,
    cache_root: Path,
    prompt: str,
    *,
    state_fn,
    lerobot_dir: Path | None,
    batch_size: int = 32,
) -> tuple[int, int]:
    """Tap the canonical terse prompt for each kept demo; return (written, skipped_done)."""
    written = skipped = 0
    task_frames = frames_dir / task
    out_task = cache_root / task
    for ep in keep_ids:
        out_dir = out_task / f"demo_{ep:06d}"
        if (out_dir / DONE_MARKER).exists():
            skipped += 1
            continue
        fp = task_frames / f"ep{ep:03d}_frames.npz"
        if not fp.exists():
            raise FileNotFoundError(
                f"[{task}] required frames for the seed-0 keep-set are missing: {fp} "
                f"(extract_frames must run for the exact keep-set before the tap)"
            )
        d = np.load(fp, allow_pickle=True)
        framesets = {v: d[f"frames_{v}"] for v in VIEWS}
        fidx = d["frame_indices"].astype(np.int64)
        state = state_fn(lerobot_dir, ep, fidx)
        n = len(fidx)

        patches, langs = [], []
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            frames = {v: framesets[v][lo:hi] for v in VIEWS}
            r = tap.tap(frames, state[lo:hi], prompt)  # canonical terse prompt, str broadcast
            patches.append(np.asarray(r.patch_tokens, dtype=np.float16))
            langs.append(np.asarray(r.lang_emb, dtype=np.float16))
        patch_tokens = np.concatenate(patches)  # [F,192,2048]
        lang_per_frame = np.concatenate(langs)  # [F,2048]

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_npy = out_dir / ".patch_tokens.npy.incomplete"
        with tmp_npy.open("wb") as stream:  # file handle: np.save must not append '.npy'
            np.save(stream, patch_tokens)
        os.replace(tmp_npy, out_dir / "patch_tokens.npy")
        tmp_feats = out_dir / ".feats.npz.incomplete"
        with tmp_feats.open("wb") as stream:
            np.savez(stream, lang_per_frame=lang_per_frame, frame_indices=fidx)
        os.replace(tmp_feats, out_dir / "feats.npz")
        if not _reopen_ok(out_dir, n):
            raise ValueError(f"[{task}] ep{ep:06d}: reopen validation failed; refusing done marker")
        marker_tmp = out_dir / (DONE_MARKER + ".incomplete")
        marker_tmp.write_text("ok")
        os.replace(marker_tmp, out_dir / DONE_MARKER)
        written += 1
        if written % 50 == 0:
            print(f"[stage-s-feats] {task}: {written} written (ep{ep:06d} F={n})", flush=True)
    return written, skipped


def build_source_manifest(
    *,
    cache_root: Path,
    target_root: Path,
    labels_root: Path,
    prompt_manifest_path: Path,
    prompt_manifest_id: str,
    prompt_manifest_uri: str,
    feature_source_inventory_id: str,
    out_path: Path,
    expected_tasks: int = 50,
    demos_per_task=150,
    seed: int = 0,
    dataset_name: str = sfv.DATASET_NAME,
    task_dir_globs=sfv.DEFAULT_TASK_DIR_GLOBS,
) -> tuple[str, dict]:
    """Assemble + validate the content-addressed source-feature manifest; return (id, manifest)."""
    dataset_tasks = sfv._dataset_task_dirs(target_root, task_dir_globs=task_dir_globs)
    if len(dataset_tasks) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} target tasks; found {len(dataset_tasks)}")
    task_records = []
    for task in sorted(dataset_tasks):
        keep = sorted(
            sfv._expected_episode_ids(
                dataset_tasks[task],
                demos_per_task=sfv.demos_for_task(demos_per_task, task),
                seed=seed,
            )
        )
        episodes = []
        for ep in keep:
            demo_rel = f"{task}/demo_{ep:06d}"
            patch_path = cache_root / demo_rel / "patch_tokens.npy"
            feats_path = cache_root / demo_rel / "feats.npz"
            if not (cache_root / demo_rel / DONE_MARKER).exists():
                raise ValueError(f"missing canonical feature for {demo_rel} (no {DONE_MARKER})")
            n = int(np.load(patch_path, mmap_mode="r").shape[0])  # header-only read
            label_rel = f"{task}/vlm_episode_pi_{ep:06d}.npz"
            label_path = labels_root / label_rel
            available = label_path.is_file()
            episodes.append(
                {
                    "episode_index": ep,
                    "patch_tokens": {
                        "path": f"{demo_rel}/patch_tokens.npy",
                        "size_bytes": patch_path.stat().st_size,
                        "sha256": prov.sha256_file(patch_path),
                        "shape": [n, sfv.PATCH_GRID, sfv.BACKBONE_DIM],
                        "dtype": "float16",
                    },
                    "feats": {
                        "path": f"{demo_rel}/feats.npz",
                        "size_bytes": feats_path.stat().st_size,
                        "sha256": prov.sha256_file(feats_path),
                    },
                    "frame_count": n,
                    "teacher_labels": {
                        "available": available,
                        "path": label_rel,
                        "sha256": prov.sha256_file(label_path) if available else None,
                    },
                }
            )
        task_records.append({"task": task, "episodes": episodes})

    manifest = {
        "schema_version": 1,
        "artifact": sfv.ARTIFACT,
        "global_language_mode": sfv.GLOBAL_LANGUAGE_MODE,
        "dataset": {
            "name": dataset_name,
            "episode_subsample_seed": seed,
            "demos_per_task": sfv.canonical_demos_per_task(demos_per_task),
        },
        "task_prompt_manifest": {"id": prompt_manifest_id, "uri": prompt_manifest_uri},
        "feature_source": {"checkpoint_inventory_id": feature_source_inventory_id},
        "frame_grid": {"stride": sfv.FRAME_STRIDE, "include_last": True, "image_hw": [256, 256]},
        "producing_code": {"sha256": producing_code_sha256()},
        "tasks": task_records,
    }
    manifest_id = sfv.source_manifest_id(manifest)
    data = prov.canonical_json(manifest) + "\n"
    out_path = Path(out_path)
    if out_path.is_dir():
        out_path = out_path / f"{manifest_id}.json"
    tmp = out_path.with_name(f".{out_path.name}.incomplete")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, out_path)
    # Final gate: the strict validator over the on-disk cache.
    sfv.validate_source_manifest(
        features_root=cache_root,
        target_root=target_root,
        labels_root=labels_root,
        manifest_path=out_path,
        expected_task_prompt_manifest_id=prompt_manifest_id,
        expected_feature_source_inventory_id=feature_source_inventory_id,
        expected_tasks=expected_tasks,
        demos_per_task=demos_per_task,
        seed=seed,
        dataset_name=dataset_name,
        task_dir_globs=task_dir_globs,
    )
    return manifest_id, manifest


def _keep_ids(dataset_tasks: dict, task: str, demos_per_task, seed: int) -> list[int]:
    if task not in dataset_tasks:
        raise SystemExit(f"task {task!r} not in target root")
    return sorted(
        sfv._expected_episode_ids(
            dataset_tasks[task],
            demos_per_task=sfv.demos_for_task(demos_per_task, task),
            seed=seed,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-root", required=True, help="materialized T30 target dataset root")
    ap.add_argument("--frames-dir", required=True, help="extract_frames output root (<root>/<task>/ep*)")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--labels-root", default=None, help="teacher-label root (build-manifest only)")
    ap.add_argument("--task-prompt-manifest", required=True)
    ap.add_argument("--task-prompt-manifest-sha256", required=True)
    ap.add_argument("--task-prompt-manifest-uri", default=None, help="s3:// canonical URI (build-manifest)")
    ap.add_argument("--feature-source-inventory-id", default=None, help="64hex (build-manifest)")
    ap.add_argument("--ckpt", default=None, help="frozen pi05 checkpoint dir (GPU pass)")
    ap.add_argument("--config", default="pi05_rc_mg60_bal33")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard", default="0/1", help="i/N — process tasks whose index %% N == i")
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--out", default=None, help="manifest output path/dir (build-manifest)")
    ap.add_argument(
        "--lerobot-dir-source",
        choices=("soup", "target-root"),
        default="soup",
        help="where observation.state is read from: the robocasa registry soup (default) or the "
        "lerobot dir the task glob already resolved (robocasa-free)",
    )
    ap.add_argument("--task-prompt-artifact", default=prompts.ARTIFACT)
    sfv.add_dataset_shape_args(ap)
    args = ap.parse_args()

    shape = sfv.dataset_shape_kwargs(args)
    target_root = Path(args.target_root).expanduser()
    cache_root = Path(args.cache_root).expanduser()
    # Verify the prompt manifest bytes match the claimed id, then load the exact task->prompt map.
    manifest_path = Path(args.task_prompt_manifest).expanduser()
    if prov.sha256_file(manifest_path) != args.task_prompt_manifest_sha256:
        raise SystemExit("task-prompt manifest sha256 does not match the file bytes")
    prompt_map = prompts.load_task_prompts(
        manifest_path,
        target_root=target_root,
        expected_tasks=shape["expected_tasks"],
        expected_artifact=args.task_prompt_artifact,
        task_dir_globs=shape["task_dir_globs"],
    )

    if args.build_manifest:
        if not (args.labels_root and args.feature_source_inventory_id and args.out and args.task_prompt_manifest_uri):
            raise SystemExit(
                "--build-manifest requires --labels-root, --feature-source-inventory-id, "
                "--task-prompt-manifest-uri, and --out"
            )
        manifest_id, _m = build_source_manifest(
            cache_root=cache_root,
            target_root=target_root,
            labels_root=Path(args.labels_root).expanduser(),
            prompt_manifest_path=manifest_path,
            prompt_manifest_id=args.task_prompt_manifest_sha256,
            prompt_manifest_uri=args.task_prompt_manifest_uri,
            feature_source_inventory_id=args.feature_source_inventory_id,
            out_path=Path(args.out).expanduser(),
            **shape,
        )
        episodes = sum(
            sfv.demos_for_task(shape["demos_per_task"], task)
            for task in sfv._dataset_task_dirs(target_root, task_dir_globs=shape["task_dir_globs"])
        )
        print(
            f"[stage-s-feats] manifest_id={manifest_id} tasks={shape['expected_tasks']} "
            f"episodes={episodes} upload=false",
            flush=True,
        )
        return

    if not args.ckpt:
        raise SystemExit("the GPU tap pass requires --ckpt")
    from workspace_models.features.pi_backbone_tap import Pi05BackboneTap

    i, nshards = (int(x) for x in args.shard.split("/"))
    dataset_tasks = sfv._dataset_task_dirs(target_root, task_dir_globs=shape["task_dir_globs"])
    tasks = [t for k, t in enumerate(sorted(dataset_tasks)) if k % nshards == i]
    tap = Pi05BackboneTap(args.ckpt, args.config)
    frames_dir = Path(args.frames_dir).expanduser()
    t0 = time.time()
    for task in tasks:
        keep = _keep_ids(dataset_tasks, task, shape["demos_per_task"], shape["seed"])
        if args.lerobot_dir_source == "target-root":
            lerobot_dir = dataset_tasks[task]
        else:
            lerobot_dir = None  # resolved via utils.soup inside state_fn when needed
            from utils.soup import combined_target_soup

            metas = [m for m in combined_target_soup(demo_fraction=1.0) if m["task"] == task]
            if metas:
                lerobot_dir = Path(metas[0]["path"]).expanduser()
        w, s = cache_task(
            tap,
            task,
            keep,
            frames_dir,
            cache_root,
            prompt_map[task],
            state_fn=_state_at,
            lerobot_dir=lerobot_dir,
            batch_size=args.batch_size,
        )
        print(f"[stage-s-feats] {task}: {w} written, {s} already-done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
