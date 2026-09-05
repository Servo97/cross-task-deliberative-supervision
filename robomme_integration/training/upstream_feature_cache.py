"""Build a compact, provenance-locked cache of official RoboMME history features.

The official ``Yinpei/robomme_preprocessed_data`` repository stores one zip per episode.  Each
archive contains hundreds of pickled ``token_emb_<step>.npy`` dictionaries, including several
spatial resolutions and a copy of the deterministic positional embedding at every step.  The
recurrent-TTT recipe consumes only the 8x8 tensors.  This builder therefore:

* downloads only the 100 archives belonging to one pinned single-task manifest;
* keeps the source archives for auditability and resumable downloads;
* reads them directly with ``zipfile`` (no exploded small-file tree);
* writes one contiguous bfloat16 image tensor and one tiny state tensor per episode; and
* verifies/deduplicates the position tensor into one task-level table.

No existing path is deleted.  Completion markers are written only after all hashes and shapes have
been re-opened and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .single_task import select_task_episodes, task_manifest_sha256

UPSTREAM_REPO_ID = "Yinpei/robomme_preprocessed_data"
UPSTREAM_REVISION = "ddf0baf55b633cc6657dcd53ac0e089a273de612"
IMAGE_SHAPE = (1, 64, 2048)
POSITION_SHAPE = (1, 64, 768)
STATE_SHAPE = (8,)
_TOKEN_MEMBER = re.compile(r"^episode_(?P<episode>[0-9]+)/token_emb_(?P<step>[0-9]+)\.npy$")


def sha256_file(path: str | Path, *, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def recurrent_history_indices(
    step_idx: int,
    exec_start_idx: int,
    *,
    input_obs_horizon: int = 16,
    max_recur_steps: int = 64,
    max_video_steps: int = 40,
) -> tuple[int, ...]:
    """Exact index policy from upstream ``MemoryBufferRecurrent`` (made independently testable)."""
    if step_idx < 0 or exec_start_idx < 0 or step_idx < exec_start_idx:
        raise ValueError("execution step must be nonnegative and at/after exec_start_idx")
    if input_obs_horizon < 2 or max_recur_steps < 1 or max_video_steps < 1:
        raise ValueError("invalid recurrent-history geometry")

    if exec_start_idx == 0:
        if step_idx < input_obs_horizon:
            selected = [step_idx]
        else:
            start_idx = step_idx % input_obs_horizon
            selected = list(range(start_idx, step_idx + 1, input_obs_horizon))[-max_recur_steps:]
    else:
        if exec_start_idx <= input_obs_horizon * 2:
            video = list(range(0, exec_start_idx, input_obs_horizon // 2))
        elif exec_start_idx <= max_video_steps * input_obs_horizon:
            video = list(range(0, exec_start_idx, input_obs_horizon))
        else:
            video = np.linspace(0, exec_start_idx - 1, max_video_steps, dtype=int).tolist()

        if step_idx - exec_start_idx < input_obs_horizon:
            execution = [step_idx]
        else:
            start_idx = (step_idx - exec_start_idx) % input_obs_horizon + exec_start_idx
            execution = list(range(start_idx, step_idx + 1, input_obs_horizon))
        selected = (video + execution)[-max_recur_steps:]

    if not 1 <= len(selected) <= max_recur_steps or selected != sorted(selected):
        raise RuntimeError(f"invalid upstream recurrent indices: {selected}")
    if selected[-1] > step_idx:
        raise RuntimeError("recurrent history leaked a future frame")
    return tuple(int(value) for value in selected)


def _archive_members(archive: zipfile.ZipFile, episode: int) -> dict[int, str]:
    members: dict[int, str] = {}
    for name in archive.namelist():
        match = _TOKEN_MEMBER.fullmatch(name)
        if match is None or int(match.group("episode")) != episode:
            continue
        step = int(match.group("step"))
        if step in members:
            raise ValueError(f"duplicate token feature for episode {episode}, step {step}")
        members[step] = name
    if not members or sorted(members) != list(range(max(members) + 1)):
        raise ValueError(f"episode {episode} feature steps are not contiguous from zero")
    return members


def _load_token_payload(archive: zipfile.ZipFile, member: str) -> dict[str, np.ndarray]:
    # The pickle is accepted only inside a sha256-recorded archive from the pinned official revision.
    with archive.open(member) as stream:
        payload = np.load(io.BytesIO(stream.read()), allow_pickle=True).item()
    required = {"image_emb_8x8", "pos_emb_8x8", "state_emb"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"invalid official token payload in {member}")
    image = np.asarray(payload["image_emb_8x8"])
    position = np.asarray(payload["pos_emb_8x8"])
    state = np.asarray(payload["state_emb"])
    if image.shape != IMAGE_SHAPE or image.dtype.name != "bfloat16":
        raise ValueError(f"invalid image feature {image.shape}/{image.dtype} in {member}")
    if position.shape != POSITION_SHAPE or position.dtype != np.float32:
        raise ValueError(f"invalid position feature {position.shape}/{position.dtype} in {member}")
    if state.shape != STATE_SHAPE or not np.issubdtype(state.dtype, np.floating):
        raise ValueError(f"invalid state feature {state.shape}/{state.dtype} in {member}")
    if not all(np.isfinite(np.asarray(value, dtype=np.float32)).all() for value in (image, position, state)):
        raise ValueError(f"non-finite official feature in {member}")
    return {"image": image, "position": position, "state": state}


def _atomic_memmap(path: Path, *, dtype, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    incomplete = path.with_name(f".{path.name}.incomplete")
    return incomplete, np.lib.format.open_memmap(incomplete, mode="w+", dtype=dtype, shape=shape)


def _finish_memmap(incomplete: Path, destination: Path, array) -> None:
    array.flush()
    del array
    os.replace(incomplete, destination)


def consolidate_episode(
    archive_path: str | Path,
    output_root: str | Path,
    episode: int,
    position_by_step: dict[int, np.ndarray],
) -> dict:
    """Consolidate one official archive and update the shared exact-position table."""
    archive_path = Path(archive_path)
    output_dir = Path(output_root) / f"episode_{episode}"
    marker = output_dir / "COMPLETE.json"
    archive_sha = sha256_file(archive_path)
    if marker.is_file():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        image_path = output_dir / "image_bf16_bits.npy"
        state_path = output_dir / "state_f64.npy"
        if (
            prior.get("source_archive_sha256") == archive_sha
            and image_path.is_file()
            and state_path.is_file()
            and prior.get("image_sha256") == sha256_file(image_path)
            and prior.get("state_sha256") == sha256_file(state_path)
        ):
            # Rebuild the shared position table on partial-task resume without keeping a repeated
            # per-episode copy in the compact cache.
            with zipfile.ZipFile(archive_path) as archive:
                members = _archive_members(archive, episode)
                for step in range(len(members)):
                    position = _load_token_payload(archive, members[step])["position"]
                    known = position_by_step.get(step)
                    if known is None:
                        position_by_step[step] = position.copy()
                    elif not np.array_equal(known, position):
                        raise ValueError(f"position embedding differs at shared step {step} (episode {episode})")
            return prior
        raise ValueError(f"stale or corrupted completed cache at {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"zip CRC failure in {archive_path}: {bad_member}")
        members = _archive_members(archive, episode)
        steps = len(members)
        image_tmp, image_out = _atomic_memmap(
            output_dir / "image_bf16_bits.npy",
            # NumPy's .npy header loses ml_dtypes.bfloat16 metadata under open_memmap.  Store the
            # exact two-byte payload as uint16 and reinterpret it as bfloat16 in the loader.
            dtype=np.uint16,
            shape=(steps, *IMAGE_SHAPE),
        )
        state_tmp, state_out = _atomic_memmap(
            output_dir / "state_f64.npy",
            dtype=np.float64,
            shape=(steps, *STATE_SHAPE),
        )
        for step in range(steps):
            payload = _load_token_payload(archive, members[step])
            image_out[step] = payload["image"].view(np.uint16)
            state_out[step] = payload["state"]
            position = payload["position"]
            known = position_by_step.get(step)
            if known is None:
                position_by_step[step] = position.copy()
            elif not np.array_equal(known, position):
                raise ValueError(f"position embedding differs at shared step {step} (episode {episode})")
        _finish_memmap(image_tmp, output_dir / "image_bf16_bits.npy", image_out)
        _finish_memmap(state_tmp, output_dir / "state_f64.npy", state_out)

    image_check = np.load(output_dir / "image_bf16_bits.npy", mmap_mode="r")
    state_check = np.load(output_dir / "state_f64.npy", mmap_mode="r")
    if image_check.shape != (steps, *IMAGE_SHAPE) or image_check.dtype != np.uint16:
        raise RuntimeError(f"re-open validation failed for {output_dir}/image_bf16_bits.npy")
    if state_check.shape != (steps, *STATE_SHAPE) or state_check.dtype != np.float64:
        raise RuntimeError(f"re-open validation failed for {output_dir}/state_f64.npy")
    record = {
        "episode": episode,
        "steps": steps,
        "source_archive": archive_path.name,
        "source_archive_bytes": archive_path.stat().st_size,
        "source_archive_sha256": archive_sha,
        "image_sha256": sha256_file(output_dir / "image_bf16_bits.npy"),
        "state_sha256": sha256_file(output_dir / "state_f64.npy"),
    }
    incomplete_marker = marker.with_name(".COMPLETE.json.incomplete")
    incomplete_marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incomplete_marker, marker)
    return record


def _download_one(raw_root: Path, episode: int) -> Path:
    from huggingface_hub import hf_hub_download

    relative = f"features/episode_{episode}.zip"
    downloaded = Path(
        hf_hub_download(
            repo_id=UPSTREAM_REPO_ID,
            repo_type="dataset",
            revision=UPSTREAM_REVISION,
            filename=relative,
            cache_dir=raw_root / ".hf-cache",
            local_dir=raw_root,
        )
    )
    if not downloaded.is_file() or downloaded.stat().st_size < 1:
        raise RuntimeError(f"failed to materialize {relative}")
    return downloaded


def build_task_cache(
    *,
    task_name: str,
    lerobot_root: str | Path,
    output_root: str | Path,
    download_workers: int = 4,
) -> dict:
    episodes = select_task_episodes(lerobot_root, task_name)
    root = Path(output_root) / task_name
    raw_root = root / "source_archives"
    compact_root = root / "compact"
    raw_root.mkdir(parents=True, exist_ok=True)
    compact_root.mkdir(parents=True, exist_ok=True)

    archives: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {pool.submit(_download_one, raw_root, episode): episode for episode in episodes}
        for future in as_completed(futures):
            episode = futures[future]
            archives[episode] = future.result()
            print(f"[upstream-cache] downloaded episode={episode} path={archives[episode]}", flush=True)

    position_by_step: dict[int, np.ndarray] = {}
    records = []
    for ordinal, episode in enumerate(episodes, 1):
        record = consolidate_episode(archives[episode], compact_root, episode, position_by_step)
        records.append(record)
        print(f"[upstream-cache] consolidated {ordinal}/{len(episodes)} episode={episode}", flush=True)

    max_step = max(position_by_step)
    if sorted(position_by_step) != list(range(max_step + 1)):
        raise RuntimeError("shared position table has gaps")
    position_path = compact_root / "position_f32.npy"
    position_tmp, position_out = _atomic_memmap(
        position_path,
        dtype=np.float32,
        shape=(max_step + 1, *POSITION_SHAPE),
    )
    for step in range(max_step + 1):
        position_out[step] = position_by_step[step]
    _finish_memmap(position_tmp, position_path, position_out)
    position_check = np.load(position_path, mmap_mode="r")
    if position_check.shape != (max_step + 1, *POSITION_SHAPE) or position_check.dtype != np.float32:
        raise RuntimeError("shared position table re-open validation failed")

    manifest = {
        "schema_version": 1,
        "task_name": task_name,
        "task_manifest_sha256": task_manifest_sha256(task_name),
        "episodes": list(episodes),
        "upstream": {"repo_id": UPSTREAM_REPO_ID, "revision": UPSTREAM_REVISION},
        "source_archives_retained": True,
        "source_archive_bytes": sum(record["source_archive_bytes"] for record in records),
        "position_steps": max_step + 1,
        "position_sha256": sha256_file(position_path),
        "records": records,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    final_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    incomplete = root / ".MANIFEST.json.incomplete"
    incomplete.write_bytes(final_bytes)
    os.replace(incomplete, root / "MANIFEST.json")
    print(
        f"[upstream-cache] complete task={task_name} episodes={len(episodes)} "
        f"source_gib={manifest['source_archive_bytes'] / 2**30:.2f} "
        f"manifest_sha256={manifest['manifest_sha256']}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--download-workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.download_workers <= 16:
        raise SystemExit("--download-workers must lie in [1, 16]")
    build_task_cache(
        task_name=args.task,
        lerobot_root=args.lerobot_root,
        output_root=args.output_root,
        download_workers=args.download_workers,
    )


if __name__ == "__main__":
    main()
