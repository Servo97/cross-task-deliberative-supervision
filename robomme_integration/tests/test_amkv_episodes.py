from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import numpy as np
import pytest

from robomme_integration.amkv.episodes import (
    DEFAULT_EPISODE_COUNT,
    DEFAULT_EXEC_OFFSET,
    EVAL_ROLE,
    EXECUTION_HORIZON,
    FIT_ROLE,
    IMAGE_SHAPE,
    MANIFEST_FILENAME,
    MIN_CAUSAL_STEP,
    MIN_EPISODE_LENGTH,
    PAYLOAD_FILENAME,
    STATE_DIM,
    DatasetSource,
    EpisodeCutUnavailable,
    FixtureRecord,
    SelectionSpec,
    build_fixture_records,
    bundle_size_report,
    candidate_episode_order,
    load_fixture_bundle,
    main,
    parquet_integrity_problem,
    read_dataset_meta,
    read_fixture_manifest,
    validate_fixture_records,
    write_fixture_bundle,
)
from robomme_integration.training.upstream_framesamp_data import (
    MAX_FRAMES,
    even_sampling_indices,
)

TASK_COUNT = 4
SYNTH_EXEC_START = 32
SYNTH_FIT = max(MIN_CAUSAL_STEP, SYNTH_EXEC_START + DEFAULT_EXEC_OFFSET)
SYNTH_EVAL = SYNTH_FIT + EXECUTION_HORIZON


@dataclasses.dataclass(frozen=True)
class _EpisodeSpec:
    task_index: int
    length: int
    rows: int | None = None
    step_shift: int = 0
    parquet_task_index: int | None = None
    exec_start_idx: int = 32


def _frame_pixels(episode_index: int, step: int, *, view: int) -> np.ndarray:
    """Deterministic, mostly-zero image that encodes its own provenance."""

    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    image[0, 0] = (episode_index % 251, step % 251, view)
    image[1, 1] = ((episode_index * 7 + step) % 251, view, 9)
    return image


def _frame_state(episode_index: int, step: int) -> np.ndarray:
    return np.arange(STATE_DIM, dtype=np.float32) + np.float32(episode_index * 100 + step)


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    try:
        import polars as pl
    except ModuleNotFoundError:
        pl = None
    if pl is not None:
        frame = pl.DataFrame(
            {
                "image": [row["image"] for row in rows],
                "wrist_image": [row["wrist_image"] for row in rows],
                "state": pl.Series(
                    "state",
                    [row["state"] for row in rows],
                    dtype=pl.Array(pl.Float32, STATE_DIM),
                ),
                "actions": pl.Series(
                    "actions",
                    [row["state"] for row in rows],
                    dtype=pl.Array(pl.Float32, STATE_DIM),
                ),
                "exec_start_idx": pl.Series("exec_start_idx", [row["exec_start_idx"] for row in rows], dtype=pl.Int32),
                "is_demo": pl.Series("is_demo", [row["is_demo"] for row in rows], dtype=pl.Boolean),
                "step_idx": pl.Series("step_idx", [row["step_idx"] for row in rows], dtype=pl.Int32),
                "epis_idx": pl.Series("epis_idx", [row["episode_index"] for row in rows], dtype=pl.Int32),
                "frame_index": pl.Series("frame_index", [row["frame_index"] for row in rows], dtype=pl.Int64),
                "episode_index": pl.Series("episode_index", [row["episode_index"] for row in rows], dtype=pl.Int64),
                "task_index": pl.Series("task_index", [row["task_index"] for row in rows], dtype=pl.Int64),
            }
        )
        frame.write_parquet(path)
        return
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "image": pa.array([row["image"] for row in rows], type=image_type),
            "wrist_image": pa.array([row["wrist_image"] for row in rows], type=image_type),
            "state": pa.array([row["state"] for row in rows], type=pa.list_(pa.float32(), STATE_DIM)),
            "actions": pa.array([row["state"] for row in rows], type=pa.list_(pa.float32(), STATE_DIM)),
            "exec_start_idx": pa.array([row["exec_start_idx"] for row in rows], type=pa.int32()),
            "is_demo": pa.array([row["is_demo"] for row in rows], type=pa.bool_()),
            "step_idx": pa.array([row["step_idx"] for row in rows], type=pa.int32()),
            "epis_idx": pa.array([row["episode_index"] for row in rows], type=pa.int32()),
            "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
            "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
            "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def _make_dataset(root: Path, specs: list[_EpisodeSpec], *, task_count: int = TASK_COUNT) -> Path:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    tasks = [{"task_index": index, "task": f"synthetic task {index}"} for index in range(task_count)]
    (meta / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks), encoding="utf-8")
    episodes = [
        {
            "episode_index": index,
            "tasks": [tasks[spec.task_index]["task"]],
            "length": spec.length,
        }
        for index, spec in enumerate(specs)
    ]
    (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8")
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "total_episodes": len(specs),
                "total_tasks": task_count,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": 10,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
        ),
        encoding="utf-8",
    )
    for index, spec in enumerate(specs):
        rows_written = spec.rows if spec.rows is not None else spec.length
        rows = []
        for step in range(rows_written):
            rows.append(
                {
                    "image": {
                        "bytes": _png_bytes(_frame_pixels(index, step, view=0)),
                        "path": f"frame_{step:06d}.png",
                    },
                    "wrist_image": {
                        "bytes": _png_bytes(_frame_pixels(index, step, view=1)),
                        "path": f"frame_{step:06d}.png",
                    },
                    "state": _frame_state(index, step).tolist(),
                    "exec_start_idx": spec.exec_start_idx,
                    "is_demo": step < spec.exec_start_idx,
                    "step_idx": step + (spec.step_shift if step else 0),
                    "frame_index": step,
                    "episode_index": index,
                    "task_index": (spec.task_index if spec.parquet_task_index is None else spec.parquet_task_index),
                }
            )
        _write_parquet(data / f"episode_{index:06d}.parquet", rows)
    return root


def _default_specs() -> list[_EpisodeSpec]:
    specs = []
    for task_index in range(TASK_COUNT):
        specs.append(_EpisodeSpec(task_index=task_index, length=96))
        specs.append(_EpisodeSpec(task_index=task_index, length=88))
        specs.append(_EpisodeSpec(task_index=task_index, length=60))
    return specs


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory) -> Path:
    return _make_dataset(tmp_path_factory.mktemp("robomme_synthetic"), _default_specs())


def test_memory_indices_match_upstream_even_sampling(synthetic_root):
    records = build_fixture_records(synthetic_root, episodes=2, seed=0)
    assert len(records) == 4
    for record in records:
        expected = np.asarray(even_sampling_indices(record.step_idx), dtype=np.int32)
        assert np.array_equal(record.memory_step_indices, expected)
        assert record.memory_step_indices.dtype == np.int32
        assert record.memory_step_indices.shape == (MAX_FRAMES,)
        assert int(record.memory_step_indices[0]) == 0
        assert int(record.memory_step_indices[-1]) == record.step_idx
        assert np.all(np.diff(record.memory_step_indices) > 0)
        assert len(set(record.memory_step_indices.tolist())) == MAX_FRAMES
    assert [record.step_idx for record in records] == [SYNTH_FIT, SYNTH_EVAL] * 2
    assert all(record.fit_step == SYNTH_FIT and record.eval_step == SYNTH_EVAL for record in records)
    assert [record.chunk_role for record in records] == [FIT_ROLE, EVAL_ROLE] * 2
    assert records[1].step_idx - records[0].step_idx == EXECUTION_HORIZON


def test_pixels_and_states_round_trip_from_png(synthetic_root):
    records = build_fixture_records(synthetic_root, episodes=1, seed=0)
    record = records[0]
    for position, step in enumerate(record.memory_step_indices.tolist()):
        assert np.array_equal(
            record.memory_images[position],
            _frame_pixels(record.episode_index, step, view=0),
        )
        assert np.array_equal(
            record.memory_states[position],
            _frame_state(record.episode_index, step),
        )
    assert np.array_equal(record.obs_image, _frame_pixels(record.episode_index, record.step_idx, view=0))
    assert np.array_equal(record.obs_wrist_image, _frame_pixels(record.episode_index, record.step_idx, view=1))
    assert not np.array_equal(record.obs_image, record.obs_wrist_image)
    assert record.memory_images.dtype == np.uint8 and record.memory_states.dtype == np.float32
    assert record.fixture_id == f"ep{record.episode_index:06d}-t{record.step_idx:05d}"
    assert record.pair_id == f"ep{record.episode_index:06d}"


def test_selection_is_deterministic_seeded_and_spans_tasks(synthetic_root):
    source, episodes, _ = read_dataset_meta(synthetic_root)
    assert source.total_episodes == len(_default_specs())
    selection = SelectionSpec(
        seed=0,
        episode_count=4,
        min_episode_length=MIN_EPISODE_LENGTH,
        exec_offset=DEFAULT_EXEC_OFFSET,
    )
    first = candidate_episode_order(episodes, selection)
    second = candidate_episode_order(episodes, selection)
    assert [item.episode_index for item in first] == [item.episode_index for item in second]
    # Round-robin: the first pass covers every task exactly once.
    assert [item.task_index for item in first[:TASK_COUNT]] == list(range(TASK_COUNT))
    assert len({item.task_index for item in first[:TASK_COUNT]}) == TASK_COUNT

    other = candidate_episode_order(episodes, dataclasses.replace(selection, seed=7))
    assert [item.task_index for item in other[:TASK_COUNT]] == list(range(TASK_COUNT))
    assert {item.episode_index for item in other} == {item.episode_index for item in first}
    assert [item.episode_index for item in other] == [
        item.episode_index for item in candidate_episode_order(episodes, dataclasses.replace(selection, seed=7))
    ]

    built = build_fixture_records(synthetic_root, episodes=4, seed=0)
    assert [record.episode_index for record in built[::2]] == [item.episode_index for item in first[:4]]
    assert [record.episode_index for record in built] == [
        record.episode_index for record in build_fixture_records(synthetic_root, episodes=4, seed=0)
    ]
    assert len({record.task_index for record in built}) == TASK_COUNT


def test_short_episodes_are_skipped(synthetic_root):
    source, episodes, _ = read_dataset_meta(synthetic_root)
    short = {item.episode_index for item in episodes if item.length < MIN_EPISODE_LENGTH}
    assert short, "the synthetic dataset must contain skippable episodes"
    selection = SelectionSpec(
        seed=0,
        episode_count=2,
        min_episode_length=MIN_EPISODE_LENGTH,
        exec_offset=DEFAULT_EXEC_OFFSET,
    )
    ordering = candidate_episode_order(episodes, selection)
    assert len(ordering) == len(episodes) - len(short)
    assert short.isdisjoint({item.episode_index for item in ordering})
    records = build_fixture_records(synthetic_root, episodes=len(ordering), seed=0)
    assert short.isdisjoint({record.episode_index for record in records})
    assert all(record.episode_length >= MIN_EPISODE_LENGTH for record in records)
    with pytest.raises(ValueError, match="satisfy the selection rule"):
        build_fixture_records(synthetic_root, episodes=len(episodes), seed=0)


def test_write_load_round_trip_verifies_shas(synthetic_root, tmp_path):
    records = build_fixture_records(synthetic_root, episodes=3, seed=1)
    bundle = tmp_path / "bundle"
    manifest = write_fixture_bundle(records, bundle)
    assert {path.name for path in bundle.iterdir()} == {MANIFEST_FILENAME, PAYLOAD_FILENAME}
    assert manifest.record_count == 6 and manifest.selection.episode_count == 3
    assert manifest.selection.seed == 1
    assert manifest.dataset_source.root == str(synthetic_root)
    assert manifest.selection_rule == manifest.selection.rule()
    assert "round-robin over task_index" in manifest.selection_rule

    loaded = load_fixture_bundle(bundle)
    assert len(loaded) == len(records)
    for original, restored in zip(records, loaded, strict=True):
        assert restored.metadata() == original.metadata()
        for name, array in original.arrays().items():
            assert np.array_equal(getattr(restored, name), array)
            assert getattr(restored, name).dtype == array.dtype

    on_disk = read_fixture_manifest(bundle)
    assert on_disk == manifest
    stored = json.loads((bundle / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert stored["records"][0]["memory_step_indices"] == list(even_sampling_indices(SYNTH_FIT))
    assert "memory_images" not in json.dumps(stored)

    report = bundle_size_report(bundle)
    assert report["record_count"] == 6
    assert report["bytes_per_record"] > MAX_FRAMES * int(np.prod(IMAGE_SHAPE)) // 4
    assert report["payload_sha256"] == manifest.payload_sha256

    with pytest.raises(FileExistsError):
        write_fixture_bundle(records, bundle)


def test_tampered_payload_and_manifest_are_rejected(synthetic_root, tmp_path):
    records = build_fixture_records(synthetic_root, episodes=1, seed=0)
    bundle = tmp_path / "tamper"
    write_fixture_bundle(records, bundle)

    payload = bundle / PAYLOAD_FILENAME
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 0xFF
    payload.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_fixture_bundle(bundle)

    # Metadata that lives only in the manifest is still sealed by content_sha256.
    second = tmp_path / "tamper_manifest"
    write_fixture_bundle(records, second)
    stored = json.loads((second / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    stored["records"][0]["task"] += " (tampered)"
    (second / MANIFEST_FILENAME).write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="content SHA256 mismatch"):
        load_fixture_bundle(second)

    # Cut-bearing metadata is additionally cross-checked before any digest.
    fifth = tmp_path / "tamper_exec_start"
    write_fixture_bundle(records, fifth)
    stored = json.loads((fifth / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    stored["records"][0]["exec_start_idx"] += 1
    (fifth / MANIFEST_FILENAME).write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="execution_step disagrees with its cut"):
        load_fixture_bundle(fifth)

    third = tmp_path / "tamper_indices"
    write_fixture_bundle(records, third)
    stored = json.loads((third / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    stored["records"][0]["memory_step_indices"][3] += 1
    (third / MANIFEST_FILENAME).write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="official rule"):
        load_fixture_bundle(third)

    fourth = tmp_path / "tamper_files"
    write_fixture_bundle(records, fourth)
    (fourth / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        load_fixture_bundle(fourth)


def test_record_invariants_fail_closed(synthetic_root):
    record = build_fixture_records(synthetic_root, episodes=1, seed=0)[0]
    record.validate()

    broken = dataclasses.replace(record, memory_step_indices=record.memory_step_indices.astype(np.int64))
    with pytest.raises(ValueError, match="memory_step_indices must be int32"):
        broken.validate()

    shifted = record.memory_step_indices.copy()
    shifted[0] = 1
    with pytest.raises(ValueError, match="official FrameSamp rule"):
        dataclasses.replace(record, memory_step_indices=shifted).validate()

    truncated = record.memory_step_indices[:-1]
    with pytest.raises(ValueError, match=r"int32\[32\]"):
        dataclasses.replace(record, memory_step_indices=truncated).validate()

    with pytest.raises(ValueError, match="must sit at"):
        dataclasses.replace(record, chunk_role=EVAL_ROLE).validate()

    with pytest.raises(ValueError, match="chunk_role must be one of"):
        dataclasses.replace(record, chunk_role="warmup").validate()

    with pytest.raises(ValueError, match="does not encode"):
        dataclasses.replace(record, fixture_id="ep000000-t00000").validate()

    with pytest.raises(ValueError, match="memory_images must be uint8"):
        dataclasses.replace(record, memory_images=record.memory_images.astype(np.int16)).validate()

    with pytest.raises(ValueError, match="obs_state contains non-finite"):
        dataclasses.replace(record, obs_state=np.full((STATE_DIM,), np.nan, dtype=np.float32)).validate()

    detached = record.obs_image.copy()
    detached[0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="final memory frame"):
        dataclasses.replace(record, obs_image=detached).validate()

    with pytest.raises(ValueError, match="final memory state"):
        dataclasses.replace(record, obs_state=record.obs_state + np.float32(1.0)).validate()

    with pytest.raises(ValueError, match="outside an episode"):
        dataclasses.replace(record, episode_length=10).validate()


def test_bundle_level_invariants_fail_closed(synthetic_root):
    records = build_fixture_records(synthetic_root, episodes=2, seed=0)
    validate_fixture_records(records)

    with pytest.raises(ValueError, match="fit/eval pairs"):
        validate_fixture_records(records[:3])
    with pytest.raises(ValueError, match="consecutive"):
        validate_fixture_records((records[1], records[0], records[2], records[3]))
    with pytest.raises(ValueError, match="unique"):
        validate_fixture_records((records[0], records[1], records[0], records[1]))
    with pytest.raises(ValueError, match="at least one record"):
        validate_fixture_records(())
    with pytest.raises(ValueError, match="sequence of FixtureRecord"):
        validate_fixture_records(records[0])

    foreign = DatasetSource(
        root="/elsewhere",
        info_sha256="0" * 64,
        episodes_sha256="1" * 64,
        tasks_sha256="2" * 64,
        total_episodes=1,
        total_tasks=1,
        chunks_size=1000,
        unreadable_episodes=(),
    )
    mixed = (records[0], dataclasses.replace(records[1], source=foreign), records[2], records[3])
    with pytest.raises(ValueError, match="one dataset source"):
        validate_fixture_records(mixed)

    with pytest.raises(ValueError, match="requested"):
        validate_fixture_records(records[:2])


def test_dataset_drift_is_detected(tmp_path):
    truncated = _make_dataset(
        tmp_path / "truncated",
        [_EpisodeSpec(task_index=0, length=96, rows=70)],
        task_count=1,
    )
    with pytest.raises(ValueError, match="episodes.jsonl declares"):
        build_fixture_records(truncated, episodes=1, seed=0)

    shifted = _make_dataset(
        tmp_path / "shifted",
        [_EpisodeSpec(task_index=0, length=96, step_shift=3)],
        task_count=1,
    )
    with pytest.raises(ValueError, match="not contiguous"):
        build_fixture_records(shifted, episodes=1, seed=0)

    mislabeled = _make_dataset(
        tmp_path / "mislabeled",
        [_EpisodeSpec(task_index=0, length=96, parquet_task_index=3)],
        task_count=1,
    )
    with pytest.raises(ValueError, match="disagrees with meta task_index"):
        build_fixture_records(mislabeled, episodes=1, seed=0)

    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="missing dataset metadata"):
        read_dataset_meta(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        read_dataset_meta(tmp_path / "absent")


def test_corrupt_shards_are_screened_and_recorded(tmp_path):
    root = _make_dataset(
        tmp_path / "corrupt",
        [
            _EpisodeSpec(task_index=0, length=96),
            _EpisodeSpec(task_index=0, length=96),
            _EpisodeSpec(task_index=1, length=96),
        ],
        task_count=2,
    )
    damaged = root / "data" / "chunk-000" / "episode_000000.parquet"
    raw = bytearray(damaged.read_bytes())
    assert bytes(raw[:4]) == b"PAR1"
    raw[:4] = b"\x00\x00\x00\x00"  # exactly how the local HF blobs are damaged
    damaged.write_bytes(bytes(raw))
    assert "header magic" in (parquet_integrity_problem(damaged) or "")
    assert parquet_integrity_problem(root / "data" / "chunk-000" / "episode_000001.parquet") is None
    assert parquet_integrity_problem(root / "data" / "chunk-000" / "absent.parquet") == "missing"

    source, episodes, _ = read_dataset_meta(root)
    assert source.unreadable_episodes == (0,)
    selection = SelectionSpec(
        seed=0,
        episode_count=2,
        min_episode_length=MIN_EPISODE_LENGTH,
        exec_offset=DEFAULT_EXEC_OFFSET,
    )
    ordering = candidate_episode_order(episodes, selection, exclude=frozenset(source.unreadable_episodes))
    assert [item.episode_index for item in ordering] == [1, 2]

    records = build_fixture_records(root, episodes=2, seed=0)
    assert sorted({record.episode_index for record in records}) == [1, 2]
    bundle = tmp_path / "corrupt_bundle"
    manifest = write_fixture_bundle(records, bundle)
    assert manifest.dataset_source.unreadable_episodes == (0,)
    assert load_fixture_bundle(bundle)[0].source.unreadable_episodes == (0,)
    assert bundle_size_report(bundle)["unreadable_source_episodes"] == 1


def test_exec_start_and_prompt_are_carried(synthetic_root):
    records = build_fixture_records(synthetic_root, episodes=1, seed=0)
    for record in records:
        assert record.exec_start_idx == 32
        assert record.task == f"synthetic task {record.task_index}"
        assert record.episode_length >= MIN_EPISODE_LENGTH
        assert record.step_idx <= record.episode_length - 1


def test_cli_writes_and_reverifies_a_bundle(synthetic_root, tmp_path, capsys):
    out = tmp_path / "cli_bundle"
    assert main(["--out", str(out), "--episodes", "2", "--seed", "0", "--dataset-root", str(synthetic_root)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["record_count"] == 4
    assert printed["episode_count"] == 2
    assert printed["bytes_per_record"] > 0
    assert printed["task_indices"] == [0, 1]
    loaded = load_fixture_bundle(out)
    assert len(loaded) == 4
    assert [record.chunk_role for record in loaded] == [FIT_ROLE, EVAL_ROLE] * 2


def test_selection_spec_rejects_incoherent_cuts():
    workable = SelectionSpec(
        seed=0,
        episode_count=1,
        min_episode_length=MIN_EPISODE_LENGTH,
        exec_offset=DEFAULT_EXEC_OFFSET,
    )
    workable.validate()
    # The floor keeps all 32 memory frames real when execution starts early.
    assert workable.cuts_for(exec_start_idx=0, episode_length=200) == (
        MIN_CAUSAL_STEP,
        MIN_CAUSAL_STEP + EXECUTION_HORIZON,
    )
    assert workable.cuts_for(exec_start_idx=100, episode_length=200) == (116, 132)
    with pytest.raises(EpisodeCutUnavailable, match="exceeds the last step"):
        workable.cuts_for(exec_start_idx=180, episode_length=200)
    with pytest.raises(EpisodeCutUnavailable, match="below the"):
        workable.cuts_for(exec_start_idx=0, episode_length=MIN_EPISODE_LENGTH - 1)

    with pytest.raises(ValueError, match="min_episode_length must be at least"):
        SelectionSpec(
            seed=0,
            episode_count=1,
            min_episode_length=MIN_CAUSAL_STEP,
            exec_offset=DEFAULT_EXEC_OFFSET,
        ).validate()
    with pytest.raises(ValueError, match="exec_offset must be an integer"):
        SelectionSpec(
            seed=0,
            episode_count=1,
            min_episode_length=MIN_EPISODE_LENGTH,
            exec_offset="16",
        ).validate()
    assert DEFAULT_EPISODE_COUNT == 32 and DEFAULT_EXEC_OFFSET == 16
    assert isinstance(FixtureRecord, type)


def test_cuts_are_execution_relative_per_episode(tmp_path):
    root = _make_dataset(
        tmp_path / "phases",
        [
            _EpisodeSpec(task_index=0, length=200, exec_start_idx=0),
            _EpisodeSpec(task_index=1, length=200, exec_start_idx=100),
        ],
        task_count=2,
    )
    records = build_fixture_records(root, episodes=2, seed=0)
    by_episode = {record.episode_index: record for record in records if record.chunk_role == FIT_ROLE}

    early = by_episode[0]
    assert early.exec_start_idx == 0
    assert (early.fit_step, early.eval_step) == (MIN_CAUSAL_STEP, MIN_CAUSAL_STEP + EXECUTION_HORIZON)
    assert early.execution_step == MIN_CAUSAL_STEP

    late = by_episode[1]
    assert late.exec_start_idx == 100
    assert (late.fit_step, late.eval_step) == (116, 132)
    assert late.execution_step == DEFAULT_EXEC_OFFSET

    for record in records:
        assert record.step_idx >= record.exec_start_idx
        assert record.execution_step >= 0
        assert record.metadata()["execution_step"] == record.step_idx - record.exec_start_idx
    evaluation = next(record for record in records if record.episode_index == 1 and record.chunk_role == EVAL_ROLE)
    assert evaluation.execution_step == DEFAULT_EXEC_OFFSET + EXECUTION_HORIZON


def test_episodes_whose_demo_outruns_the_cut_are_skipped(tmp_path):
    root = _make_dataset(
        tmp_path / "late_demo",
        [
            # eval cut would be 102 > last step 95: unusable.
            _EpisodeSpec(task_index=0, length=96, exec_start_idx=70),
            _EpisodeSpec(task_index=0, length=96, exec_start_idx=75),
            _EpisodeSpec(task_index=0, length=96, exec_start_idx=8),
        ],
        task_count=1,
    )
    records = build_fixture_records(root, episodes=1, seed=0)
    assert {record.episode_index for record in records} == {2}
    assert records[0].fit_step == MIN_CAUSAL_STEP and records[0].eval_step == 47
    for record in records:
        assert record.step_idx >= record.exec_start_idx
    # Proof the builder walked past the unusable candidates instead of emitting them.
    with pytest.raises(ValueError, match="ran out of candidate episodes"):
        build_fixture_records(root, episodes=2, seed=0)


def test_consumer_contract_attribute_names(synthetic_root):
    record = build_fixture_records(synthetic_root, episodes=1, seed=0)[0]
    assert record.prompt == record.task
    assert record.chunk_role in ("fit_chunk", "eval_chunk")
    assert record.memory_images.shape == (MAX_FRAMES, *IMAGE_SHAPE)
    assert record.memory_states.shape == (MAX_FRAMES, STATE_DIM)
    assert record.memory_step_indices.shape == (MAX_FRAMES,)
    # The driver stacks a singleton view axis before handing memory to the
    # official MemoryBuffer; keep that reshape exact.
    views = record.memory_images[:, None, ...]
    assert views.shape == (MAX_FRAMES, 1, *IMAGE_SHAPE)
    assert tuple(record.memory_step_indices.tolist()) == even_sampling_indices(record.step_idx)
