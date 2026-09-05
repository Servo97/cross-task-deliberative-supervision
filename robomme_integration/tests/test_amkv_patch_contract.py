"""The AM patch is only usable while both source hashes match their pins."""

from __future__ import annotations

import inspect

import pytest

from robomme_integration.amkv import patch_contract, stage_e0
from robomme_integration.training import framesamp_am_jax
from wsm_settings import ROBOMME_EVAL_ROOT

OFFICIAL_ROOT = ROBOMME_EVAL_ROOT / "official_reference" / "robomme_policy_learning"


def test_vendored_patch_matches_its_reviewed_hash():
    actual = patch_contract.sha256_file(patch_contract.PATCHED_MODULE_PATH)
    assert actual == patch_contract.REVIEWED_AMKV_PATCH_SHA256, (
        "the vendored AM patch changed; re-review the diff against the five patch points, then "
        "regenerate REVIEWED_AMKV_PATCH_SHA256 with `python -m robomme_integration.amkv.patch_contract --record`"
    )
    assert patch_contract.OFFICIAL_POLICY_TREE_SHA1 == stage_e0.PINNED_POLICY_TREE_SHA1


def test_patch_drift_is_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(patch_contract, "REVIEWED_AMKV_PATCH_SHA256", "0" * 64)
    source = tmp_path / "src" / "mme_vla_suite" / "models" / "integration"
    source.mkdir(parents=True)
    (source / "history_gemma.py").write_bytes(b"")
    with pytest.raises((ValueError, FileNotFoundError)):
        patch_contract.require_reviewed_amkv_patch(tmp_path)


def test_official_base_source_drift_is_detected(tmp_path):
    root = tmp_path
    target = root / patch_contract.OFFICIAL_HISTORY_GEMMA_RELATIVE
    target.parent.mkdir(parents=True)
    target.write_text("# not the audited source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted from the audited base"):
        patch_contract.require_reviewed_amkv_patch(root)


def test_missing_official_source_is_reported_by_path():
    with pytest.raises(FileNotFoundError, match="official MemoryAttention source is missing"):
        patch_contract.official_history_gemma_path("/nonexistent/policy/root")


@pytest.mark.skipif(not OFFICIAL_ROOT.is_dir(), reason="official policy source is not staged locally")
def test_reviewed_patch_accepts_the_pinned_official_source():
    patch = patch_contract.require_reviewed_amkv_patch(OFFICIAL_ROOT)
    assert patch.policy_git_sha == framesamp_am_jax.OFFICIAL_POLICY_GIT_SHA
    assert patch.policy_tree_sha1 == patch_contract.OFFICIAL_POLICY_TREE_SHA1
    assert patch.official_source_sha256 == framesamp_am_jax.OFFICIAL_HISTORY_GEMMA_SHA256
    assert patch.patched_module_sha256 == patch_contract.REVIEWED_AMKV_PATCH_SHA256
    assert patch.module_class.endswith("MemoryAttention")
    assert set(patch.to_dict()) == {
        "policy_git_sha",
        "policy_tree_sha1",
        "official_source_sha256",
        "patched_module_sha256",
        "module_class",
    }


def test_amkv_registry_is_independent_of_the_robomme_lane_registry():
    """Two lanes patch the same seam; neither may silently authorise the other.

    The RoboMME lane ships its own source-rewriting overlay with its own pin.
    This build never consults that pin, and its own pin must not be mistaken
    for it -- a run is labelled by the module it actually executed.
    """

    assert patch_contract.REVIEWED_AMKV_PATCH_SHA256 != framesamp_am_jax.REVIEWED_PATCHED_HISTORY_GEMMA_SHA256
    source = inspect.getsource(patch_contract.require_reviewed_amkv_patch)
    assert "REVIEWED_PATCHED_HISTORY_GEMMA_SHA256" not in source
    assert "REVIEWED_AMKV_PATCH_SHA256" in source


def test_imported_kernel_signature_is_pinned():
    """A read-only lane may evolve; its contract with this build may not drift silently."""

    signature = inspect.signature(framesamp_am_jax.memory_attention_am_core)
    positional = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == [
        "queries_post_rope_pre_scale",
        "compact_keys_post_rope",
        "compact_values_post_projection",
        "compact_beta_am",
        "recent_keys_post_rope",
        "recent_values_post_projection",
        "recent_token_mask",
    ]
    assert {"scale", "query_position_offset"} <= set(signature.parameters)


def test_full_cache_branch_preserves_released_rope_scale_operation_order():
    """Regression: Q-RoPE -> Q-scale -> K-RoPE is part of BF16 identity."""

    from robomme_integration.amkv.patched_history_gemma import MemoryAttentionAM

    source = inspect.getsource(MemoryAttentionAM.__call__)
    full_cache = source.split("if am_pack is None:", 1)[1].split("else:", 1)[0]
    q_rope = full_cache.index("q = _apply_rope(q")
    q_scale = full_cache.index("q = q * head_dim**-0.5")
    k_rope = full_cache.index("k = _apply_rope(k")
    assert q_rope < q_scale < k_rope
