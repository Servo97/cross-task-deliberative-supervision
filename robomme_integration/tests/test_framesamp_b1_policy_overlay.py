from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from robomme_integration.training.framesamp_b1_data import B1_PARTITION_KIND
from robomme_integration.training.framesamp_b1_policy_overlay import (
    MEM_BUFFER_RELATIVE_PATH,
    OFFICIAL_MEM_BUFFER_SHA256,
    OFFICIAL_POLICY_SHA256,
    OVERLAY_MANIFEST,
    PATCHED_MEM_BUFFER_SHA256,
    PATCHED_POLICY_SHA256,
    POLICY_RELATIVE_PATH,
    stage_framesamp_b1_policy_overlay,
    verify_framesamp_b1_policy_overlay,
)
from wsm_settings import ROBOMME_EVAL_ROOT

OFFICIAL_CHECKOUT = Path(
    os.environ.get(
        "ROBOMME_OFFICIAL_POLICY_ROOT",
        str(ROBOMME_EVAL_ROOT / "official_reference" / "robomme_policy_learning"),
    )
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(tmp_path: Path) -> tuple[Path, str]:
    if not (OFFICIAL_CHECKOUT / ".git").is_dir():
        pytest.skip("pinned RoboMME policy checkout is unavailable")
    destination = tmp_path / "b1_overlay"
    manifest_sha = stage_framesamp_b1_policy_overlay(OFFICIAL_CHECKOUT, destination)
    return destination, manifest_sha


def test_b1_overlay_is_exactly_two_source_changes_and_self_authenticating(tmp_path):
    destination, manifest_sha = _stage(tmp_path)
    manifest = verify_framesamp_b1_policy_overlay(destination, expected_manifest_sha256=manifest_sha)
    assert manifest["representation_policy"]["kind"] == B1_PARTITION_KIND
    assert manifest["representation_policy"]["total_frames"] == 32
    assert manifest["representation_policy"]["total_tokens"] == 512
    assert manifest["representation_policy"]["single_attention_denominator"] is True
    assert manifest["representation_policy"]["compression"] == "none"
    assert manifest["official_equivalence"] is False
    assert _sha(OFFICIAL_CHECKOUT.joinpath(*POLICY_RELATIVE_PATH.parts)) == OFFICIAL_POLICY_SHA256
    assert _sha(OFFICIAL_CHECKOUT.joinpath(*MEM_BUFFER_RELATIVE_PATH.parts)) == OFFICIAL_MEM_BUFFER_SHA256
    assert _sha(destination.joinpath(*POLICY_RELATIVE_PATH.parts)) == PATCHED_POLICY_SHA256
    assert _sha(destination.joinpath(*MEM_BUFFER_RELATIVE_PATH.parts)) == PATCHED_MEM_BUFFER_SHA256

    changed = []
    for source in (OFFICIAL_CHECKOUT / "src").rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(OFFICIAL_CHECKOUT)
        staged = destination / relative
        if staged.is_file() and source.read_bytes() != staged.read_bytes():
            changed.append(relative.as_posix())
    assert sorted(changed) == [
        POLICY_RELATIVE_PATH.as_posix(),
        MEM_BUFFER_RELATIVE_PATH.as_posix(),
    ]


def test_staged_b1_buffer_uses_fixed_partitions_and_exact_recent_suffix(tmp_path):
    destination, _ = _stage(tmp_path)
    staged_path = destination.joinpath(*MEM_BUFFER_RELATIVE_PATH.parts)
    spec = importlib.util.spec_from_file_location("staged_framesamp_b1_mem_buffer", staged_path)
    assert spec is not None and spec.loader is not None
    staged = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staged)

    memory = staged.MemoryBuffer.__new__(staged.MemoryBuffer)
    memory.num_views = 1
    demo, live = memory.get_frame_sampling_b1_indices(140, 99)
    assert demo == np.linspace(0, 99, 16, dtype=np.int32).tolist()
    assert live == list(range(125, 141))

    # Exercise the physical partition assembly without loading a vision model.
    def gather(indices, *args, **kwargs):
        del args, kwargs
        return {index: {"index": index} for index in indices}

    def prepare(history, indices, token_budget, token_per_image):
        del history, token_per_image
        capacity = token_budget // 16
        values = np.zeros((capacity * 16, 1), dtype=np.float32)
        mask = np.zeros((capacity * 16,), dtype=np.bool_)
        for slot, index in enumerate(indices):
            values[slot * 16 : (slot + 1) * 16] = index
            mask[slot * 16 : (slot + 1) * 16] = True
        return values, values.copy(), values.copy(), mask

    memory._prepare_frame_sampling = prepare
    image, position, state, mask = memory.prepare_frame_sampling_b1(105, 99, 512, 16, gather)
    assert image.shape == position.shape == state.shape == (512, 1)
    assert mask.shape == (512,)
    frame_values = image.reshape(32, 16, 1)[:, 0, 0]
    assert frame_values[:16].astype(int).tolist() == np.linspace(0, 99, 16, dtype=np.int32).tolist()
    assert frame_values[16:22].astype(int).tolist() == list(range(100, 106))
    assert np.count_nonzero(frame_values[22:]) == 0


def test_overlay_rejects_manifest_or_source_drift(tmp_path):
    destination, manifest_sha = _stage(tmp_path)
    manifest_path = destination / OVERLAY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["representation_policy"]["demo_slots"] = [0, 23]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA"):
        verify_framesamp_b1_policy_overlay(destination, expected_manifest_sha256=manifest_sha)


def test_overlay_refuses_noncanonical_compute_geometry(tmp_path):
    destination, _ = _stage(tmp_path)
    staged_path = destination.joinpath(*MEM_BUFFER_RELATIVE_PATH.parts)
    spec = importlib.util.spec_from_file_location("staged_framesamp_b1_bad_geometry", staged_path)
    assert spec is not None and spec.loader is not None
    staged = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staged)
    memory = staged.MemoryBuffer.__new__(staged.MemoryBuffer)
    memory.num_views = 1
    with pytest.raises(ValueError, match="512 tokens"):
        memory.prepare_frame_sampling_b1(20, 10, 256, 16, lambda indices: {})
