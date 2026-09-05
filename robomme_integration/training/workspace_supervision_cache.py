"""Build deliberate, long-lag WSM supervision from pinned RoboMME features and subgoals.

This is intentionally a representation artifact, not a policy input.  For each episode it stores a
small per-frame global feature stream plus chronological event targets.  An event target is the
*historical* frozen visual feature at the grounded patch when a new subgoal segment begins.  A later
workspace token can therefore be supervised to recover an earlier value instead of copying the
current frame at the same spatial index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from .single_task import select_task_episodes, task_manifest_sha256
from .upstream_feature_cache import UPSTREAM_REPO_ID, UPSTREAM_REVISION
from .upstream_ttt_data import verify_compact_manifest

IMAGE_SIZE = 256
PATCH_GRID = 8
FEATURE_DIM = 2048
_POINT = re.compile(r"at <(?P<x>[0-9]+), (?P<y>[0-9]+)>")


def sha256_file(path: str | Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def grounded_patch_id(text: str) -> int:
    matches = list(_POINT.finditer(str(text)))
    if len(matches) != 1:
        raise ValueError(f"grounded subgoal must contain exactly one point: {text!r}")
    x, y = int(matches[0]["x"]), int(matches[0]["y"])
    if not (0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE):
        raise ValueError(f"grounded point outside {IMAGE_SIZE}x{IMAGE_SIZE}: {(x, y)}")
    col = min(PATCH_GRID - 1, x * PATCH_GRID // IMAGE_SIZE)
    row = min(PATCH_GRID - 1, y * PATCH_GRID // IMAGE_SIZE)
    return row * PATCH_GRID + col


def chronological_events(simple: list[str], grounded: list[str]) -> list[dict]:
    if not simple or len(simple) != len(grounded):
        raise ValueError("simple/grounded subgoal streams must be nonempty and aligned")
    events = []
    previous = None
    for step, (simple_text, grounded_text) in enumerate(zip(simple, grounded, strict=True)):
        simple_text = str(simple_text)
        grounded_text = str(grounded_text)
        identity = (simple_text, grounded_text)
        if identity == previous:
            continue
        previous = identity
        # Some deliberately generated subgoals (for example "put down the container") have no
        # grounded point.  They are not silently converted into a guessed visual target: the
        # manifest records their hashes/coverage separately and only point-grounded segments
        # become salient-patch reconstruction targets.
        if not _POINT.search(grounded_text):
            continue
        events.append(
            {
                "anchor_step": step,
                "patch_id": grounded_patch_id(grounded_text),
                "simple_subgoal_sha256": hashlib.sha256(simple_text.encode()).hexdigest(),
                "grounded_subgoal_sha256": hashlib.sha256(grounded_text.encode()).hexdigest(),
            }
        )
    return events


def unpointed_segments(simple: list[str], grounded: list[str]) -> list[dict]:
    """Return hashed identities for segment transitions that lack a visual point."""
    skipped = []
    previous = None
    for step, (simple_text, grounded_text) in enumerate(zip(simple, grounded, strict=True)):
        identity = (str(simple_text), str(grounded_text))
        if identity == previous:
            continue
        previous = identity
        if not _POINT.search(identity[1]):
            skipped.append(
                {
                    "anchor_step": step,
                    "simple_subgoal_sha256": hashlib.sha256(identity[0].encode()).hexdigest(),
                    "grounded_subgoal_sha256": hashlib.sha256(identity[1].encode()).hexdigest(),
                }
            )
    return skipped


def _episode_parquet(data_root: Path, episode: int) -> Path:
    candidates = list((data_root / "data").glob(f"chunk-*/episode_{episode:06d}.parquet"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one parquet for episode {episode}, found {candidates}")
    return candidates[0]


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def build_workspace_supervision_cache(
    *,
    task_name: str,
    lerobot_root: str | Path,
    upstream_cache_root: str | Path,
    output_root: str | Path,
    verify_upstream_hashes: bool = True,
) -> dict:
    import ml_dtypes
    import pyarrow.parquet as pq

    data_root = Path(lerobot_root)
    episodes = select_task_episodes(data_root, task_name)
    compact, upstream_manifest = verify_compact_manifest(
        upstream_cache_root,
        task_name,
        episodes,
        verify_hashes=verify_upstream_hashes,
    )
    root = Path(output_root) / task_name
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal, episode in enumerate(episodes, 1):
        destination = root / f"episode_{episode}" / "supervision.npz"
        source_record = next(record for record in upstream_manifest["records"] if int(record["episode"]) == episode)
        image_bits = np.load(compact / f"episode_{episode}" / "image_bf16_bits.npy", mmap_mode="r")
        states = np.load(compact / f"episode_{episode}" / "state_f64.npy", mmap_mode="r")
        steps = int(source_record["steps"])
        if image_bits.shape != (steps, 1, 64, FEATURE_DIM) or states.shape != (steps, 8):
            raise ValueError(f"upstream compact shape mismatch for episode {episode}")

        table = pq.read_table(
            _episode_parquet(data_root, episode),
            columns=["episode_index", "step_idx", "simple_subgoal", "grounded_subgoal"],
        )
        episode_column = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        step_column = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
        if len(table) != steps or not np.all(episode_column == episode):
            raise ValueError(f"parquet/feature episode mismatch for episode {episode}")
        if not np.array_equal(step_column, np.arange(steps, dtype=np.int64)):
            raise ValueError(f"non-contiguous step_idx for episode {episode}")
        simple = table["simple_subgoal"].to_pylist()
        grounded = table["grounded_subgoal"].to_pylist()
        events = chronological_events(simple, grounded)
        skipped = unpointed_segments(simple, grounded)
        if not events:
            raise ValueError(f"episode {episode} has no point-grounded salient event")
        anchor = np.asarray([event["anchor_step"] for event in events], dtype=np.int32)
        patch_id = np.asarray([event["patch_id"] for event in events], dtype=np.int16)

        # Convert exactly once from official bfloat16 payloads.  The compact supervision artifact
        # is small enough for high-throughput random sampling and retains fp16 frozen features.
        image_f32 = image_bits.view(ml_dtypes.bfloat16).astype(np.float32)
        frame_mean = image_f32[:, 0].mean(axis=1).astype(np.float16)
        event_feature = np.stack(
            [image_f32[int(step), 0, int(patch)] for step, patch in zip(anchor, patch_id, strict=True)]
        ).astype(np.float16)
        _atomic_savez(
            destination,
            frame_mean_f16=frame_mean,
            state_f32=np.asarray(states, dtype=np.float32),
            event_anchor_i32=anchor,
            event_patch_id_i16=patch_id,
            event_feature_f16=event_feature,
        )
        with np.load(destination) as reopened:
            expected_keys = {
                "frame_mean_f16",
                "state_f32",
                "event_anchor_i32",
                "event_patch_id_i16",
                "event_feature_f16",
            }
            if set(reopened.files) != expected_keys:
                raise RuntimeError(f"reopen key mismatch for episode {episode}")
            if reopened["frame_mean_f16"].shape != (steps, FEATURE_DIM):
                raise RuntimeError(f"reopen frame shape mismatch for episode {episode}")
            if reopened["event_feature_f16"].shape != (len(events), FEATURE_DIM):
                raise RuntimeError(f"reopen event shape mismatch for episode {episode}")
        records.append(
            {
                "episode": episode,
                "steps": steps,
                "events": events,
                "unpointed_segments": skipped,
                "path": f"episode_{episode}/supervision.npz",
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
        print(
            f"[wsm-supervision] {ordinal}/{len(episodes)} episode={episode} "
            f"steps={steps} events={len(events)} unpointed={len(skipped)}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "artifact": "robomme_wsm_long_lag_supervision",
        "task_name": task_name,
        "task_manifest_sha256": task_manifest_sha256(task_name),
        "episodes": list(episodes),
        "feature_dim": FEATURE_DIM,
        "patch_grid": PATCH_GRID,
        "frame_feature": "mean_of_8x8_official_frozen_siglip_features",
        "event_target": "frozen_patch_value_at_subgoal_segment_anchor",
        "upstream": {
            "repo_id": UPSTREAM_REPO_ID,
            "revision": UPSTREAM_REVISION,
            "compact_manifest_sha256": upstream_manifest["manifest_sha256"],
        },
        "causal_training_contract": {
            "encoder_inputs": "frames_at_or_before_decision_t",
            "target": "event_anchor_at_or_before_t_minus_min_lag",
            "current_frame_masking_required": True,
            "uses_labels_at_inference": False,
        },
        "records": records,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    temporary = root / ".MANIFEST.json.incomplete"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, root / "MANIFEST.json")
    print(
        f"[wsm-supervision] complete task={task_name} manifest_sha256={manifest['manifest_sha256']}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--upstream-cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--skip-upstream-hashes", action="store_true")
    args = parser.parse_args()
    build_workspace_supervision_cache(
        task_name=args.task,
        lerobot_root=args.lerobot_root,
        upstream_cache_root=args.upstream_cache_root,
        output_root=args.output_root,
        verify_upstream_hashes=not args.skip_upstream_hashes,
    )


if __name__ == "__main__":
    main()
