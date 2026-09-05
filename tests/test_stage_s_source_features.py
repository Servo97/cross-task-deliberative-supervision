"""Offline tests for the Stage-S canonical-terse source-feature manifest validator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import stage_s_fixtures as fx

from scripts.launch import validate_stage_s_source_features as sfv


def _validate(target, features, labels, manifest_path, prompt_id, *, tasks=2, demos=3):
    return sfv.validate_source_manifest(
        features_root=features,
        target_root=target,
        labels_root=labels,
        manifest_path=manifest_path,
        expected_task_prompt_manifest_id=prompt_id,
        expected_feature_source_inventory_id=fx.INVENTORY_ID,
        expected_tasks=tasks,
        demos_per_task=demos,
        seed=0,
        workers=2,
    )


def _rewrite(manifest_path: Path, manifest: dict) -> None:
    from scripts.launch import stage_s_provenance as prov

    manifest_path.write_text(prov.canonical_json(manifest) + "\n", encoding="utf-8")


def test_roundtrip_passes(tmp_path):
    target, features, labels, prompt_id, manifest_path, _m = fx.full_source_fixture(tmp_path)
    assert _validate(target, features, labels, manifest_path, prompt_id) == {
        "tasks": 2,
        "episodes": 6,
        "labeled": 6,
    }


def test_partial_labels_counted_not_blocking(tmp_path):
    target, features, labels, prompt_id, manifest_path, _m = fx.full_source_fixture(
        tmp_path,
        unlabeled={("Task00", None)},  # placeholder replaced below
    )
    # Rebuild cleanly with a concrete unlabeled demo.
    import shutil

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    target = tmp_path / "target"
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    selected = fx.make_target_dataset(target, tasks=2, demos=3)
    _pp, prompt_id = fx.make_prompt_manifest(target, tmp_path / "prompts", tasks=2)
    one = (sorted(selected)[0], selected[sorted(selected)[0]][0])
    manifest_path, _m = fx.make_source_features(
        features, labels, target, selected, prompt_manifest_id=prompt_id, unlabeled={one}
    )
    summary = _validate(target, features, labels, manifest_path, prompt_id)
    assert summary == {"tasks": 2, "episodes": 6, "labeled": 5}


def test_undeclared_patch_file_rejected(tmp_path):
    target, features, labels, prompt_id, manifest_path, _m = fx.full_source_fixture(tmp_path)
    extra = features / "Task00" / "demo_999999" / "patch_tokens.npy"
    extra.parent.mkdir(parents=True)
    np.save(extra, np.zeros((1, fx.PATCH_GRID, fx.BACKBONE_DIM), dtype=np.float16))
    with pytest.raises(ValueError, match="undeclared"):
        _validate(target, features, labels, manifest_path, prompt_id)


def test_missing_declared_file_rejected(tmp_path):
    target, features, labels, prompt_id, manifest_path, manifest = fx.full_source_fixture(tmp_path)
    victim = features / manifest["tasks"][0]["episodes"][0]["feats"]["path"]
    victim.unlink()
    with pytest.raises(ValueError, match="differs from manifest|is missing"):
        _validate(target, features, labels, manifest_path, prompt_id)


def test_wrong_keepset_episode_rejected(tmp_path):
    target, features, labels, prompt_id, manifest_path, manifest = fx.full_source_fixture(tmp_path)
    # Point a record at an episode index outside the seed-0 keep-set (still a valid-looking file).
    rec = manifest["tasks"][0]["episodes"][0]
    wrong = 999
    for key, suffix in (("patch_tokens", "patch_tokens.npy"), ("feats", "feats.npz")):
        old = features / rec[key]["path"]
        new_rel = f"Task00/demo_{wrong:06d}/{suffix}"
        new = features / new_rel
        new.parent.mkdir(parents=True, exist_ok=True)
        old.replace(new)
        rec[key]["path"] = new_rel
    rec["episode_index"] = wrong
    rec["teacher_labels"] = {"available": False, "path": f"Task00/vlm_episode_pi_{wrong:06d}.npz", "sha256": None}
    _rewrite(manifest_path, manifest)
    with pytest.raises(ValueError, match="keep-set"):
        _validate(target, features, labels, manifest_path, prompt_id)


def test_corrupt_bytes_rejected(tmp_path):
    target, features, labels, prompt_id, manifest_path, manifest = fx.full_source_fixture(tmp_path)
    path = features / manifest["tasks"][0]["episodes"][0]["patch_tokens"]["path"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _validate(target, features, labels, manifest_path, prompt_id)


def test_forbidden_feats_key_rejected(tmp_path):
    target, features, labels, prompt_id, manifest_path, manifest = fx.full_source_fixture(tmp_path)
    rec = manifest["tasks"][0]["episodes"][0]
    path = features / rec["feats"]["path"]
    with np.load(path) as archive:
        lang = archive["lang_per_frame"]
        frame_indices = archive["frame_indices"]
    np.savez(path, lang_per_frame=lang, frame_indices=frame_indices, lang_global=np.zeros(2048, np.float16))
    from scripts.launch import stage_s_provenance as prov

    rec["feats"]["size_bytes"] = path.stat().st_size
    rec["feats"]["sha256"] = prov.sha256_file(path)
    _rewrite(manifest_path, manifest)
    with pytest.raises(ValueError, match="feats keys"):
        _validate(target, features, labels, manifest_path, prompt_id)


def test_frame_grid_violation_rejected(tmp_path):
    target, features, labels, prompt_id, manifest_path, manifest = fx.full_source_fixture(tmp_path)
    rec = manifest["tasks"][0]["episodes"][0]
    path = features / rec["feats"]["path"]
    with np.load(path) as archive:
        lang = archive["lang_per_frame"]
    bad = np.array([1] + [1 + fx.sfv.FRAME_STRIDE * i for i in range(1, lang.shape[0])], dtype=np.int64)
    np.savez(path, lang_per_frame=lang, frame_indices=bad)
    from scripts.launch import stage_s_provenance as prov

    rec["feats"]["size_bytes"] = path.stat().st_size
    rec["feats"]["sha256"] = prov.sha256_file(path)
    _rewrite(manifest_path, manifest)
    with pytest.raises(ValueError, match="must start at 0"):
        _validate(target, features, labels, manifest_path, prompt_id)


def test_label_availability_must_be_truthful(tmp_path):
    # available=true but file absent
    target, features, labels, prompt_id, manifest_path, manifest = fx.full_source_fixture(tmp_path)
    rec = manifest["tasks"][0]["episodes"][0]
    (labels / rec["teacher_labels"]["path"]).unlink()
    _rewrite(manifest_path, manifest)
    with pytest.raises(ValueError, match="available=true but label file is missing"):
        _validate(target, features, labels, manifest_path, prompt_id)


def test_prompt_manifest_id_must_match(tmp_path):
    target, features, labels, _prompt_id, manifest_path, _m = fx.full_source_fixture(tmp_path)
    with pytest.raises(ValueError, match="task_prompt_manifest.id"):
        _validate(target, features, labels, manifest_path, "a" * 64)


def test_inventory_id_must_match(tmp_path):
    target, features, labels, prompt_id, manifest_path, _m = fx.full_source_fixture(tmp_path)
    with pytest.raises(ValueError, match="feature_source.checkpoint_inventory_id"):
        sfv.validate_source_manifest(
            features_root=features,
            target_root=target,
            labels_root=labels,
            manifest_path=manifest_path,
            expected_task_prompt_manifest_id=prompt_id,
            expected_feature_source_inventory_id="0" * 64,
            expected_tasks=2,
            demos_per_task=3,
            seed=0,
            workers=2,
        )


def test_manifest_id_changes_with_inputs(tmp_path):
    target, features, labels, prompt_id, manifest_path, manifest = fx.full_source_fixture(tmp_path)
    base = sfv.source_manifest_id(manifest)
    mutated = json.loads(json.dumps(manifest))
    mutated["feature_source"]["checkpoint_inventory_id"] = "1" * 64
    assert sfv.source_manifest_id(mutated) != base
    mutated2 = json.loads(json.dumps(manifest))
    mutated2["producing_code"]["sha256"] = "2" * 64
    assert sfv.source_manifest_id(mutated2) != base


# ---- producer main-loop + manifest-builder tests (stub tap, no GPU) --------------------------

from workspace_models.features import stage_s_cache_features as producer  # noqa: E402


class _StubTap:
    """Deterministic fake pi tap: returns finite random-but-seeded patch/lang of the right shape."""

    def __init__(self):
        self._rng = np.random.default_rng(7)

    def tap(self, frames, state, prompt):
        b = next(iter(frames.values())).shape[0]
        assert isinstance(prompt, str) and prompt.strip()

        class _R:
            patch_tokens = self._rng.standard_normal((b, fx.PATCH_GRID, fx.BACKBONE_DIM)).astype(np.float16)
            lang_emb = self._rng.standard_normal((b, fx.LANGUAGE_DIM)).astype(np.float16)

        return _R()


def _write_frames(frames_dir, task, ep, n):
    d = frames_dir / task
    d.mkdir(parents=True, exist_ok=True)
    frame_indices = (np.arange(n) * producer.sfv.FRAME_STRIDE).astype(np.int64)
    np.savez(
        d / f"ep{ep:03d}_frames.npz",
        frame_indices=frame_indices,
        n_frames=np.int64(n * producer.sfv.FRAME_STRIDE),
        **{f"frames_{v}": np.zeros((n, 8, 8, 3), np.uint8) for v in producer.VIEWS},
    )


def test_producer_writes_canonical_cache_and_manifest_validates(tmp_path):
    target = tmp_path / "target"
    cache = tmp_path / "cache"
    labels = tmp_path / "labels"
    frames_dir = tmp_path / "frames"
    selected = fx.make_target_dataset(target, tasks=2, demos=3)
    _pp, prompt_id = fx.make_prompt_manifest(target, tmp_path / "prompts", tasks=2)
    prompt_map = producer.prompts.load_task_prompts(
        tmp_path / "prompts" / f"{prompt_id}.json", target_root=target, expected_tasks=2
    )
    tap = _StubTap()

    def state_fn(_ld, _ep, fidx):
        return np.zeros((len(fidx), 8), np.float32)

    for task, ids in selected.items():
        for pos, ep in enumerate(ids):
            _write_frames(frames_dir, task, ep, n=3 + (pos % 2))
        producer.cache_task(
            tap,
            task,
            ids,
            frames_dir,
            cache,
            prompt_map[task],
            state_fn=state_fn,
            lerobot_dir=None,
            batch_size=2,
        )
        # teacher labels: leave the first demo of the first task unlabeled to exercise the count
        for pos, ep in enumerate(ids):
            if task == sorted(selected)[0] and pos == 0:
                continue
            lp = labels / f"{task}/vlm_episode_pi_{ep:06d}.npz"
            lp.parent.mkdir(parents=True, exist_ok=True)
            np.savez(lp, keyframes=np.array([0], np.int64))

    manifest_id, manifest = producer.build_source_manifest(
        cache_root=cache,
        target_root=target,
        labels_root=labels,
        prompt_manifest_path=tmp_path / "prompts" / f"{prompt_id}.json",
        prompt_manifest_id=prompt_id,
        prompt_manifest_uri=f"{fx.STUDY_ROOT}/manifests/artifacts/workspace/task_prompts/robocasa_target50/{prompt_id}.json",
        feature_source_inventory_id=fx.INVENTORY_ID,
        out_path=cache,
        expected_tasks=2,
        demos_per_task=3,
        seed=0,
    )
    summary = _validate(target, cache, labels, cache / f"{manifest_id}.json", prompt_id)
    assert summary == {"tasks": 2, "episodes": 6, "labeled": 5}


def test_producer_missing_frames_fails_listing_demo(tmp_path):
    target = tmp_path / "target"
    cache = tmp_path / "cache"
    frames_dir = tmp_path / "frames"
    selected = fx.make_target_dataset(target, tasks=1, demos=2)
    _pp, prompt_id = fx.make_prompt_manifest(target, tmp_path / "prompts", tasks=1)
    task = sorted(selected)[0]
    # extract frames for only the FIRST kept demo
    _write_frames(frames_dir, task, selected[task][0], n=3)
    with pytest.raises(FileNotFoundError, match="frames for the seed-0 keep-set are missing"):
        producer.cache_task(
            _StubTap(),
            task,
            selected[task],
            frames_dir,
            cache,
            "do it",
            state_fn=lambda _l, _e, f: np.zeros((len(f), 8), np.float32),
            lerobot_dir=None,
            batch_size=2,
        )


def test_producer_done_marker_skips_recompute(tmp_path):
    target = tmp_path / "target"
    cache = tmp_path / "cache"
    frames_dir = tmp_path / "frames"
    selected = fx.make_target_dataset(target, tasks=1, demos=2)
    task = sorted(selected)[0]
    for ep in selected[task]:
        _write_frames(frames_dir, task, ep, n=3)
    args = dict(state_fn=lambda _l, _e, f: np.zeros((len(f), 8), np.float32), lerobot_dir=None, batch_size=2)
    w1, s1 = producer.cache_task(_StubTap(), task, selected[task], frames_dir, cache, "go", **args)
    w2, s2 = producer.cache_task(_StubTap(), task, selected[task], frames_dir, cache, "go", **args)
    assert (w1, s1) == (2, 0) and (w2, s2) == (0, 2)


def test_producing_code_sha_is_stable_and_derived():
    a = producer.producing_code_sha256()
    b = producer.producing_code_sha256()
    assert a == b and len(a) == 64
