"""Second, evaluation-only source overlay for FrameSamp-AM policy plumbing.

This layer starts from the reviewed module-only overlay and patches only
``HistoryPi0.sample_actions``.  The seven fixed-shape arrays are ordinary JAX
arguments, so they remain dynamic after ``nnx_utils.module_jit`` freezes model
state.  Training ``compute_loss`` and the policy/server route are intentionally
unchanged and unauthorized.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from robomme_integration.training.framesamp_am_flax_overlay import (
    HISTORY_GEMMA_RELATIVE_PATH,
    HISTORY_PI0_RELATIVE_PATH,
    OFFICIAL_HISTORY_PI0_SHA256,
    OFFICIAL_POLICY_GIT_SHA,
    PATCHED_HISTORY_GEMMA_SHA256,
    _canonical_json,
    _path_list_sha256,
    _sha256_bytes,
    _sha256_file,
    _staged_source_files,
    _tree_sha256,
    stage_framesamp_am_flax_overlay,
    verify_framesamp_am_flax_overlay,
)
from robomme_integration.training.framesamp_am_flax_overlay import (
    OVERLAY_MANIFEST as MODULE_OVERLAY_MANIFEST,
)

POLICY_OVERLAY_SCHEMA_VERSION = 1
POLICY_OVERLAY_MANIFEST = "framesamp_am_policy_overlay.json"
MODULE_OVERLAY_PROVENANCE = "framesamp_am_module_overlay.provenance.json"
MEMORY_PARTITION_KIND = "compact_all_valid_framesamp_tokens_no_recent_v1"


_SAMPLE_SIGNATURE_BASE = """    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> Actions:
"""


_SAMPLE_SIGNATURE_PATCHED = """    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        framesamp_am_compact_k=None,
        framesamp_am_compact_v=None,
        framesamp_am_compact_beta=None,
        framesamp_am_compact_mask=None,
        framesamp_am_recent_positions=None,
        framesamp_am_recent_mem_seq=None,
        framesamp_am_recent_mem_mask=None,
    ) -> Actions:
"""


_BATCH_BLOCK_BASE = """        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng, (batch_size, self.action_horizon, self.action_dim)
            )
"""


_BATCH_BLOCK_PATCHED = """        batch_size = observation.state.shape[0]
        framesamp_am_values = (
            framesamp_am_compact_k,
            framesamp_am_compact_v,
            framesamp_am_compact_beta,
            framesamp_am_compact_mask,
            framesamp_am_recent_positions,
            framesamp_am_recent_mem_seq,
            framesamp_am_recent_mem_mask,
        )
        if any(value is None for value in framesamp_am_values) and not all(
            value is None for value in framesamp_am_values
        ):
            raise ValueError("all seven dynamic FrameSamp-AM arrays must be supplied together")
        use_framesamp_am = framesamp_am_compact_k is not None
        if use_framesamp_am:
            if self.integration_type != "modulation":
                raise ValueError("FrameSamp-AM requires modulation integration")
            action_config = _gemma.get_config(self.config.action_expert_variant)
            layers, routed_batch, compact_tokens, kv_heads, head_dim = (
                framesamp_am_compact_k.shape
            )
            if batch_size != 1 or routed_batch != 1:
                raise ValueError("FrameSamp-AM evaluation is restricted to B=1")
            if (
                layers != action_config.depth
                or compact_tokens < 1
                or kv_heads != 1
                or head_dim != 256
            ):
                raise ValueError("FrameSamp-AM compact K must be [action_layers,1,M,1,256]")
            if framesamp_am_compact_v.shape != framesamp_am_compact_k.shape:
                raise ValueError("FrameSamp-AM compact V must match compact K")
            if framesamp_am_compact_beta.shape != (layers, 1, compact_tokens):
                raise ValueError("FrameSamp-AM beta must be [action_layers,1,M]")
            if framesamp_am_compact_mask.shape != (layers, 1, compact_tokens):
                raise ValueError("FrameSamp-AM compact mask must be [action_layers,1,M]")
            if framesamp_am_compact_beta.dtype != jnp.float32:
                raise ValueError("FrameSamp-AM beta must be float32")
            if framesamp_am_compact_mask.dtype != jnp.bool_:
                raise ValueError("FrameSamp-AM compact mask must be bool")
            if (
                framesamp_am_compact_k.dtype != jnp.dtype(self.config.dtype)
                or framesamp_am_compact_v.dtype != jnp.dtype(self.config.dtype)
            ):
                raise ValueError("FrameSamp-AM compact K/V must match the policy model dtype")
            if framesamp_am_recent_positions.shape != (1, 0):
                raise ValueError("sealed FrameSamp-AM v2 requires recent positions [1,0]")
            if framesamp_am_recent_positions.dtype != jnp.int32:
                raise ValueError("FrameSamp-AM recent positions must be int32")
            if framesamp_am_recent_mem_seq.shape != (1, 0, action_config.width):
                raise ValueError("sealed FrameSamp-AM v2 requires recent memory [1,0,1024]")
            if framesamp_am_recent_mem_seq.dtype != jnp.dtype(self.config.dtype):
                raise ValueError("FrameSamp-AM recent memory must match the policy model dtype")
            if (
                framesamp_am_recent_mem_mask.shape != (1, 0)
                or framesamp_am_recent_mem_mask.dtype != jnp.bool_
            ):
                raise ValueError("sealed FrameSamp-AM v2 requires Boolean recent mask [1,0]")
        if noise is None:
            noise = jax.random.normal(
                rng, (batch_size, self.action_horizon, self.action_dim)
            )
"""


_MODULATION_MEMORY_BASE = """        elif self.integration_type == "modulation":
            prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
            )
            mem_seq, mem_mask, _, _, _ = self.embed_memory(observation)
"""


_MODULATION_MEMORY_PATCHED = """        elif self.integration_type == "modulation":
            prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
            )
            if use_framesamp_am:
                # The trusted host router resolves a distinct task/episode/cut
                # stack before every replan.  Schema v2 is compact-all/R=0, so
                # invoking the perceptual encoder here would duplicate memory.
                mem_seq = framesamp_am_recent_mem_seq
                mem_mask = framesamp_am_recent_mem_mask
            else:
                mem_seq, mem_mask, _, _, _ = self.embed_memory(observation)
"""


_DENOISE_MODULATION_BASE = """            elif self.integration_type == "modulation":
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                    mem_seq=[None, mem_seq],
                    mem_mask=[None, mem_mask],
                )
"""


_DENOISE_MODULATION_PATCHED = """            elif self.integration_type == "modulation":
                if use_framesamp_am:
                    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                        [None, suffix_tokens],
                        mask=full_attn_mask,
                        positions=positions,
                        kv_cache=kv_cache,
                        adarms_cond=[None, adarms_cond],
                        mem_seq=[None, mem_seq],
                        mem_mask=[None, mem_mask],
                        framesamp_am_compact_k=framesamp_am_compact_k,
                        framesamp_am_compact_v=framesamp_am_compact_v,
                        framesamp_am_compact_beta=framesamp_am_compact_beta,
                        framesamp_am_compact_mask=framesamp_am_compact_mask,
                        framesamp_am_recent_positions=framesamp_am_recent_positions,
                    )
                else:
                    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                        [None, suffix_tokens],
                        mask=full_attn_mask,
                        positions=positions,
                        kv_cache=kv_cache,
                        adarms_cond=[None, adarms_cond],
                        mem_seq=[None, mem_seq],
                        mem_mask=[None, mem_mask],
                    )
"""


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label} patch anchor count must be 1, got {count}")
    return source.replace(old, new, 1)


def render_patched_history_pi0(base_source: bytes) -> bytes:
    """Patch only evaluation sampling from the exact released source bytes."""

    actual = _sha256_bytes(base_source)
    if actual != OFFICIAL_HISTORY_PI0_SHA256:
        raise ValueError(
            f"history_pi0.py drifted from audited base: expected {OFFICIAL_HISTORY_PI0_SHA256}, got {actual}"
        )
    source = base_source.decode("utf-8")
    source = _replace_once(
        source,
        _SAMPLE_SIGNATURE_BASE,
        _SAMPLE_SIGNATURE_PATCHED,
        label="sample_actions signature",
    )
    source = _replace_once(
        source,
        _BATCH_BLOCK_BASE,
        _BATCH_BLOCK_PATCHED,
        label="dynamic FrameSamp-AM preflight",
    )
    source = _replace_once(
        source,
        _MODULATION_MEMORY_BASE,
        _MODULATION_MEMORY_PATCHED,
        label="compact-all memory selection",
    )
    source = _replace_once(
        source,
        _DENOISE_MODULATION_BASE,
        _DENOISE_MODULATION_PATCHED,
        label="denoising HistoryGemma call",
    )
    return source.encode("utf-8")


# Populated only after the staged source passes the focused module_jit call-chain
# test.  The module-only gemma SHA remains independently pinned and unchanged.
PATCHED_HISTORY_PI0_SHA256 = "38d2b8f26e3dc201374046503f439a3152f899b97e9118c310bccb2607ef981f"
EXPECTED_MODULE_OVERLAY_MANIFEST_SHA256 = "da1114295301205f4fdbb35f8f0d0acc8d27189b52983d4f550174e141ff8eae"


def stage_framesamp_am_policy_overlay(
    official_checkout: str | Path,
    destination: str | Path,
) -> str:
    """Build a new policy overlay atop the independently verified module overlay."""

    official = Path(official_checkout).resolve(strict=True)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace staged FrameSamp-AM policy overlay: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"policy-overlay parent does not exist: {destination.parent}")

    scratch = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    staged = scratch / "staged"
    try:
        module_manifest_sha = stage_framesamp_am_flax_overlay(official, staged)
        if module_manifest_sha != EXPECTED_MODULE_OVERLAY_MANIFEST_SHA256:
            raise ValueError(
                "module-only overlay manifest drifted: "
                f"expected {EXPECTED_MODULE_OVERLAY_MANIFEST_SHA256}, got {module_manifest_sha}"
            )
        verify_framesamp_am_flax_overlay(staged, expected_manifest_sha256=module_manifest_sha)
        module_manifest = staged / MODULE_OVERLAY_MANIFEST
        module_manifest.rename(staged / MODULE_OVERLAY_PROVENANCE)

        pi0_path = staged.joinpath(*HISTORY_PI0_RELATIVE_PATH.parts)
        patched_pi0 = render_patched_history_pi0(pi0_path.read_bytes())
        patched_pi0_sha = _sha256_bytes(patched_pi0)
        if patched_pi0_sha != PATCHED_HISTORY_PI0_SHA256:
            raise ValueError(
                "rendered HistoryPi0 source does not match reviewed SHA: "
                f"expected {PATCHED_HISTORY_PI0_SHA256}, got {patched_pi0_sha}"
            )
        pi0_path.write_bytes(patched_pi0)

        files = _staged_source_files(staged)
        manifest = {
            "schema_version": POLICY_OVERLAY_SCHEMA_VERSION,
            "kind": "robomme_framesamp_am_evaluation_policy_overlay",
            "official_policy_git_sha": OFFICIAL_POLICY_GIT_SHA,
            "module_overlay_manifest_sha256": module_manifest_sha,
            "patched_history_gemma_sha256": PATCHED_HISTORY_GEMMA_SHA256,
            "base_history_pi0_sha256": OFFICIAL_HISTORY_PI0_SHA256,
            "patched_history_pi0_sha256": PATCHED_HISTORY_PI0_SHA256,
            "memory_partition_kind": MEMORY_PARTITION_KIND,
            "runtime_route": "dynamic_task_episode_causal_cut_before_each_replan",
            "training_compute_loss_status": "unchanged_not_implemented",
            "policy_server_status": "blocked_missing_authoritative_task_episode_cut_binding",
            "policy_server_blocker": (
                "mme_vla_suite.policies.policy.MME_VLA_Policy.infer lacks externally authenticated "
                "task_id/episode_id/causal_cut_step and pinned stack/index SHA inputs before _sample_actions"
            ),
            "source_file_count": len(files),
            "source_paths_sha256": _path_list_sha256(files),
            "source_tree_sha256": _tree_sha256(staged, files),
        }
        payload = _canonical_json(manifest)
        (staged / POLICY_OVERLAY_MANIFEST).write_bytes(payload)
        os.rename(staged, destination)
        staged = None
        return _sha256_bytes(payload)
    finally:
        if staged is not None and staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)


def verify_framesamp_am_policy_overlay(
    destination: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, object]:
    """Verify both source layers and keep the server/training gates explicit."""

    root = Path(destination).resolve(strict=True)
    payload = (root / POLICY_OVERLAY_MANIFEST).read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected_manifest_sha256:
        raise ValueError(
            f"FrameSamp-AM policy manifest SHA mismatch: expected {expected_manifest_sha256}, got {actual}"
        )
    try:
        manifest = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("FrameSamp-AM policy manifest is invalid UTF-8 JSON") from error
    expected = {
        "schema_version": POLICY_OVERLAY_SCHEMA_VERSION,
        "kind": "robomme_framesamp_am_evaluation_policy_overlay",
        "official_policy_git_sha": OFFICIAL_POLICY_GIT_SHA,
        "module_overlay_manifest_sha256": EXPECTED_MODULE_OVERLAY_MANIFEST_SHA256,
        "patched_history_gemma_sha256": PATCHED_HISTORY_GEMMA_SHA256,
        "base_history_pi0_sha256": OFFICIAL_HISTORY_PI0_SHA256,
        "patched_history_pi0_sha256": PATCHED_HISTORY_PI0_SHA256,
        "memory_partition_kind": MEMORY_PARTITION_KIND,
        "runtime_route": "dynamic_task_episode_causal_cut_before_each_replan",
        "training_compute_loss_status": "unchanged_not_implemented",
        "policy_server_status": "blocked_missing_authoritative_task_episode_cut_binding",
        "policy_server_blocker": (
            "mme_vla_suite.policies.policy.MME_VLA_Policy.infer lacks externally authenticated "
            "task_id/episode_id/causal_cut_step and pinned stack/index SHA inputs before _sample_actions"
        ),
    }
    if not isinstance(manifest, dict) or set(manifest) != set(expected) | {
        "source_file_count",
        "source_paths_sha256",
        "source_tree_sha256",
    }:
        raise ValueError("FrameSamp-AM policy-overlay manifest fields mismatch")
    for key, value in expected.items():
        if manifest[key] != value:
            raise ValueError(f"FrameSamp-AM policy-overlay manifest drifted at {key}")

    provenance = root / MODULE_OVERLAY_PROVENANCE
    if _sha256_file(provenance) != EXPECTED_MODULE_OVERLAY_MANIFEST_SHA256:
        raise ValueError("module-overlay provenance manifest SHA mismatch")
    gemma = root.joinpath(*HISTORY_GEMMA_RELATIVE_PATH.parts)
    pi0 = root.joinpath(*HISTORY_PI0_RELATIVE_PATH.parts)
    if _sha256_file(gemma) != PATCHED_HISTORY_GEMMA_SHA256:
        raise ValueError("policy overlay changed the reviewed HistoryGemma module")
    if _sha256_file(pi0) != PATCHED_HISTORY_PI0_SHA256:
        raise ValueError("staged HistoryPi0 evaluation patch SHA mismatch")
    files = _staged_source_files(root)
    if manifest["source_file_count"] != len(files):
        raise ValueError("policy-overlay source file set drifted")
    if manifest["source_paths_sha256"] != _path_list_sha256(files):
        raise ValueError("policy-overlay source path-list SHA mismatch")
    if manifest["source_tree_sha256"] != _tree_sha256(root, files):
        raise ValueError("policy-overlay source tree SHA mismatch")
    return manifest
