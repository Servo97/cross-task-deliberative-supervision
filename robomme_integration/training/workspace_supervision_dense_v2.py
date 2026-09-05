"""Dense, ordered multi-point workspace supervision for RoboMME v2.

This is a new artifact family.  It does not read, rewrite, or relabel the legacy point-only
``robomme_wsm_long_lag_supervision`` cache.  Every coordinate in a grounded segment becomes one
ordered target role.  A role target is a normalized dense distribution over the full 8x8 frozen
visual grid and its attention-weighted frozen feature at the segment anchor.

Grounded text is target construction metadata only.  The causal encoder input remains frame means
plus proprioception at or before the decision, so neither coordinates, target roles, nor future
frames can enter the deployed workspace encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .single_task import select_task_episodes, task_manifest_sha256
from .upstream_feature_cache import UPSTREAM_REPO_ID, UPSTREAM_REVISION
from .upstream_ttt_data import verify_compact_manifest
from .workspace_supervision_cache import FEATURE_DIM, IMAGE_SIZE, PATCH_GRID, sha256_file

SCHEMA_VERSION = 2
ARTIFACT = "robomme_wsm_dense_multipoint_supervision_v2"
TARGET_SEMANTICS = "ordered_grounded_roles_dense_gaussian_8x8_v2"
DENSE_SIGMA_PATCHES = 0.75
MAX_TARGET_ROLES = 16
_POINT = re.compile(r"<\s*(?P<x>[+-]?[0-9]+)\s*,\s*(?P<y>[+-]?[0-9]+)\s*>")
_ANGLE_TOKEN = re.compile(r"<[^<>]*>")


@dataclass(frozen=True)
class GroundedPoint:
    order: int
    x: int
    y: int


def ordered_grounded_points(text: str) -> tuple[GroundedPoint, ...]:
    """Return every valid coordinate in text order; never sort by value or deduplicate roles."""
    text = str(text)
    tokens = list(_ANGLE_TOKEN.finditer(text))
    if text.count("<") != len(tokens) or text.count(">") != len(tokens):
        raise ValueError(f"malformed grounded coordinate token: {text!r}")
    points = []
    for order, token in enumerate(tokens):
        match = _POINT.fullmatch(token.group())
        if match is None:
            raise ValueError(f"malformed grounded coordinate token: {token.group()!r}")
        x, y = int(match["x"]), int(match["y"])
        if not (0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE):
            raise ValueError(f"grounded point outside {IMAGE_SIZE}x{IMAGE_SIZE}: {(x, y)}")
        points.append(GroundedPoint(order=order, x=x, y=y))
    return tuple(points)


def dense_patch_distribution(
    point: GroundedPoint,
    *,
    sigma_patches: float = DENSE_SIGMA_PATCHES,
) -> np.ndarray:
    """Map one pixel coordinate to a normalized, strictly dense 8x8 spatial target."""
    if not np.isfinite(sigma_patches) or sigma_patches <= 0:
        raise ValueError("sigma_patches must be finite and positive")
    centers = (np.arange(PATCH_GRID, dtype=np.float64) + 0.5) * IMAGE_SIZE / PATCH_GRID
    grid_y, grid_x = np.meshgrid(centers, centers, indexing="ij")
    sigma_pixels = sigma_patches * IMAGE_SIZE / PATCH_GRID
    logits = -((grid_x - point.x) ** 2 + (grid_y - point.y) ** 2) / (2.0 * sigma_pixels**2)
    logits -= logits.max()
    weights = np.exp(logits).reshape(-1)
    weights /= weights.sum()
    result = weights.astype(np.float32)
    if result.shape != (PATCH_GRID * PATCH_GRID,) or not np.all(result > 0):
        raise RuntimeError("dense spatial target is not strictly positive over the 8x8 grid")
    if not np.isclose(result.sum(dtype=np.float64), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("dense spatial target is not normalized")
    return result


def chronological_dense_events(simple: list[str], grounded: list[str]) -> list[dict]:
    """Create segment events with all coordinate roles retained in source-text order."""
    if not simple or len(simple) != len(grounded):
        raise ValueError("simple/grounded subgoal streams must be nonempty and aligned")
    events: list[dict] = []
    previous = None
    target_offset = 0
    for anchor_step, (simple_text, grounded_text) in enumerate(zip(simple, grounded, strict=True)):
        simple_text, grounded_text = str(simple_text), str(grounded_text)
        identity = (simple_text, grounded_text)
        if identity == previous:
            continue
        previous = identity
        points = ordered_grounded_points(grounded_text)
        if not points:
            continue
        roles = [
            {
                "role_index": point.order,
                "target_index": target_offset + point.order,
                "point_xy": [point.x, point.y],
                "dominant_patch_id": int(np.argmax(dense_patch_distribution(point))),
            }
            for point in points
        ]
        events.append(
            {
                "event_index": len(events),
                "anchor_step": anchor_step,
                "target_begin": target_offset,
                "target_count": len(points),
                "roles": roles,
                "simple_subgoal_sha256": hashlib.sha256(simple_text.encode()).hexdigest(),
                "grounded_subgoal_sha256": hashlib.sha256(grounded_text.encode()).hexdigest(),
            }
        )
        target_offset += len(points)
    if target_offset > MAX_TARGET_ROLES:
        raise ValueError(f"episode has {target_offset} grounded roles, above v2 limit {MAX_TARGET_ROLES}")
    return events


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


def build_dense_supervision_cache(
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
        if len(table) != steps:
            raise ValueError(f"parquet/feature length mismatch for episode {episode}")
        episode_column = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        step_column = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
        if not np.all(episode_column == episode) or not np.array_equal(step_column, np.arange(steps, dtype=np.int64)):
            raise ValueError(f"parquet identity/order mismatch for episode {episode}")
        events = chronological_dense_events(table["simple_subgoal"].to_pylist(), table["grounded_subgoal"].to_pylist())
        if not events:
            raise ValueError(f"episode {episode} has no grounded role")

        image_f32 = image_bits.view(ml_dtypes.bfloat16).astype(np.float32)
        frame_mean = image_f32[:, 0].mean(axis=1).astype(np.float16)
        anchors, event_indices, role_indices, point_xy, attention, features = [], [], [], [], [], []
        for event in events:
            anchor = int(event["anchor_step"])
            for role in event["roles"]:
                point = GroundedPoint(
                    order=int(role["role_index"]),
                    x=int(role["point_xy"][0]),
                    y=int(role["point_xy"][1]),
                )
                weights = dense_patch_distribution(point)
                anchors.append(anchor)
                event_indices.append(int(event["event_index"]))
                role_indices.append(point.order)
                point_xy.append((point.x, point.y))
                attention.append(weights)
                features.append(weights @ image_f32[anchor, 0])
        # Keep the distribution in f32.  A corner target gives the farthest grid cells weights near
        # 1e-43; f16 would silently underflow those entries to zero and violate the declared dense
        # (full-support) target contract.
        attention_array = np.stack(attention).astype(np.float32)
        feature_array = np.stack(features).astype(np.float16)
        _atomic_savez(
            destination,
            frame_mean_f16=frame_mean,
            state_f32=np.asarray(states, dtype=np.float32),
            target_anchor_i32=np.asarray(anchors, dtype=np.int32),
            target_event_i16=np.asarray(event_indices, dtype=np.int16),
            target_role_i16=np.asarray(role_indices, dtype=np.int16),
            target_point_xy_i16=np.asarray(point_xy, dtype=np.int16),
            target_attention_f32=attention_array,
            target_feature_f16=feature_array,
        )
        with np.load(destination) as reopened:
            if reopened["target_attention_f32"].shape != (len(anchors), 64):
                raise RuntimeError(f"dense attention shape mismatch for episode {episode}")
            if not np.all(reopened["target_attention_f32"] > 0):
                raise RuntimeError(f"dense attention lost full support for episode {episode}")
            sums = reopened["target_attention_f32"].sum(axis=1)
            if not np.allclose(sums, 1.0, atol=2e-3, rtol=0.0):
                raise RuntimeError(f"dense attention normalization drift for episode {episode}")
            if reopened["target_feature_f16"].shape != (len(anchors), FEATURE_DIM):
                raise RuntimeError(f"dense feature shape mismatch for episode {episode}")
        records.append(
            {
                "episode": episode,
                "steps": steps,
                "events": events,
                "target_roles": len(anchors),
                "path": f"episode_{episode}/supervision.npz",
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
        print(
            f"[wsm-dense-v2] {ordinal}/{len(episodes)} episode={episode} events={len(events)} roles={len(anchors)}",
            flush=True,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "artifact_family_disposition": {
            "legacy_uniform_gpu_v1": "superseded_not_mutated",
            "legacy_sigreg_outputs_reused": False,
        },
        "task_name": task_name,
        "task_manifest_sha256": task_manifest_sha256(task_name),
        "episodes": list(episodes),
        "feature_dim": FEATURE_DIM,
        "patch_grid": PATCH_GRID,
        "implementation_sha256": sha256_file(Path(__file__)),
        "target_semantics": {
            "name": TARGET_SEMANTICS,
            "role_order": "left_to_right_coordinate_occurrence_in_grounded_text",
            "attention": "normalized_strictly_positive_gaussian_over_all_8x8_patches",
            "sigma_patches": DENSE_SIGMA_PATCHES,
            "feature": "attention_weighted_official_frozen_siglip_grid_at_segment_anchor",
            "max_target_roles": MAX_TARGET_ROLES,
        },
        "upstream": {
            "repo_id": UPSTREAM_REPO_ID,
            "revision": UPSTREAM_REVISION,
            "compact_manifest_sha256": upstream_manifest["manifest_sha256"],
        },
        "causal_training_contract": {
            "encoder_inputs": "frame_mean_and_state_at_or_before_decision_t_only",
            "targets": "ordered_grounded_roles_anchored_at_or_before_t_minus_min_lag",
            "grounded_text_or_coordinates_in_encoder": False,
            "future_frames_in_encoder": False,
            "uses_labels_at_inference": False,
        },
        "records": records,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    temporary = root / ".MANIFEST.json.incomplete"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, root / "MANIFEST.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--upstream-cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--skip-upstream-hashes", action="store_true")
    args = parser.parse_args()
    build_dense_supervision_cache(
        task_name=args.task,
        lerobot_root=args.lerobot_root,
        upstream_cache_root=args.upstream_cache_root,
        output_root=args.output_root,
        verify_upstream_hashes=not args.skip_upstream_hashes,
    )


if __name__ == "__main__":
    main()
