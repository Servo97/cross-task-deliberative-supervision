"""Deterministic, source-pinned Flax overlay for RoboMME FrameSamp-AM.

The released RoboMME checkout is evidence and must remain immutable.  This
module verifies that checkout at the audited commit, copies only Git-tracked
``src`` files into a new directory, and applies the narrow MemoryAttention
patch described by :mod:`framesamp_am_jax` to the copy.  It deliberately does
not claim end-to-end policy support: ``history_pi0.py`` remains byte-identical
until the trusted artifact-index route is threaded through the observation and
policy-server boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from robomme_integration.training.framesamp_am_jax import (
    OFFICIAL_HISTORY_GEMMA_SHA256,
    OFFICIAL_POLICY_GIT_SHA,
)

OFFICIAL_HISTORY_PI0_SHA256 = "a48dbfd412268a4ee689e10e6a6fd26c044eaa2b03cea8c075e2a34de49c4e57"
HISTORY_GEMMA_RELATIVE_PATH = PurePosixPath("src/mme_vla_suite/models/integration/history_gemma.py")
HISTORY_PI0_RELATIVE_PATH = PurePosixPath("src/mme_vla_suite/models/integration/history_pi0.py")
OVERLAY_MANIFEST = "framesamp_am_flax_overlay.json"
OVERLAY_SCHEMA_VERSION = 1


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


_MEMORY_ATTENTION_BASE = '''class MemoryAttention(nn.Module):
    """
    Cross Attention for Memory Modulation
    Use action sequence to attend memory sequence.
    """
    @nn.compact
    def __call__(self, x, mem_seq, mem_mask):
        # x: [B, T, D], mem_seq: [B, S, D], mem_mask: [B, S]
        B, mem_len, mem_width = mem_seq.shape
        B, x_len, x_width = x.shape
        # Let's hardcode the values for now
        num_heads, num_kv_heads, head_dim, width = (
            4,
            1,
            256,
            1024,
        )  # same dim as the action expert in pi05
        assert mem_width == x_width == width
        q_einsum = lora.Einsum(
            shape=(num_heads, width, head_dim),
            name="q_einsum_mem",
            init_fn=nn.initializers.lecun_normal(
                in_axis=-2, out_axis=-1, batch_axis=(0,)
            ),
        )
        kv_einsum = lora.Einsum(
            shape=(2, num_kv_heads, width, head_dim),
            name="kv_einsum_mem",
            init_fn=nn.initializers.lecun_normal(
                in_axis=-2, out_axis=-1, batch_axis=(0, 1)
            ),
        )
        rms_norm = MemoryRMSNorm(name="mem_rms_norm")
        x = rms_norm(x)
        q = q_einsum("BTD,NDH->BTNH", x)
        
        mem_seq = rms_norm(mem_seq)
        k, v = kv_einsum("BSD,2KDH->2BSKH", mem_seq)
        
        q_positions = einops.repeat(
            jnp.arange(mem_len, x_len + mem_len), "t -> b t", b=B
        )
        k_positions = einops.repeat(jnp.arange(mem_len), "t -> b t", b=B)
        
        q = _apply_rope(q, positions=q_positions)
        q *= head_dim**-0.5
        k = _apply_rope(k, positions=k_positions)
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)

        logits = jnp.einsum(
            "BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32
        )
        attn_mask = mem_mask[:, None, None, None, :]  # (B, 1, 1, 1, S)
        masked_logits = jnp.where(attn_mask, logits, -2.3819763e38)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out_einsum = lora.Einsum(
            shape=(num_heads, head_dim, width),
            name="out_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
        )
        return out_einsum("BTNH,NHD->BTD", encoded)
'''


_MEMORY_ATTENTION_PATCHED = '''class AMMemoryAttention(nn.Module):
    """Cross-attention for memory modulation, with an opt-in sealed AM path.

    Compact keys are already projected and RoPE-applied; compact values are
    already projected.  The sealed v2 producer is compact-all, so the reviewed
    AM path currently requires a genuinely empty ``mem_seq`` (R=0).  The
    explicit recent-position seam is retained for a future disjoint-partition
    schema but is rejected today.  Supplying a partial AM tuple is always an
    error.
    """

    @nn.compact
    def __call__(
        self,
        x,
        mem_seq,
        mem_mask,
        framesamp_am_compact_k=None,
        framesamp_am_compact_v=None,
        framesamp_am_compact_beta=None,
        framesamp_am_compact_mask=None,
        framesamp_am_recent_positions=None,
        capture_framesamp_am_taps=False,
    ):
        B, mem_len, mem_width = mem_seq.shape
        B, x_len, x_width = x.shape
        num_heads, num_kv_heads, head_dim, width = (4, 1, 256, 1024)
        assert mem_width == x_width == width

        am_values = (
            framesamp_am_compact_k,
            framesamp_am_compact_v,
            framesamp_am_compact_beta,
            framesamp_am_compact_mask,
            framesamp_am_recent_positions,
        )
        if any(value is None for value in am_values) and not all(
            value is None for value in am_values
        ):
            raise ValueError("FrameSamp-AM K/V/beta/recent positions must be supplied together")
        use_framesamp_am = framesamp_am_compact_k is not None
        if use_framesamp_am:
            if mem_len != 0:
                raise ValueError(
                    "sealed FrameSamp-AM v2 is compact-all/R=0; raw recent memory is not yet routable"
                )
            compact_tokens = framesamp_am_compact_k.shape[1]
            if compact_tokens < 1:
                raise ValueError("FrameSamp-AM compact memory must contain at least one token")
            if framesamp_am_compact_k.shape != (B, compact_tokens, 1, head_dim):
                raise ValueError("FrameSamp-AM compact K must be [B,M,1,256]")
            if framesamp_am_compact_v.shape != framesamp_am_compact_k.shape:
                raise ValueError("FrameSamp-AM compact V must match compact K")
            if framesamp_am_compact_beta.shape != (B, compact_tokens):
                raise ValueError("FrameSamp-AM beta must be [B,M]")
            if framesamp_am_compact_beta.dtype != jnp.float32:
                raise ValueError("FrameSamp-AM beta must be float32")
            if framesamp_am_compact_mask.shape != (B, compact_tokens):
                raise ValueError("FrameSamp-AM compact mask must be [B,M]")
            if framesamp_am_compact_mask.dtype != jnp.bool_:
                raise ValueError("FrameSamp-AM compact mask must be bool")
            if framesamp_am_compact_k.dtype != x.dtype or framesamp_am_compact_v.dtype != x.dtype:
                raise ValueError("FrameSamp-AM compact K/V must match the action-expert dtype")
            if framesamp_am_recent_positions.shape != (B, mem_len):
                raise ValueError("FrameSamp-AM recent positions must be [B,R]")
            if not jnp.issubdtype(framesamp_am_recent_positions.dtype, jnp.integer):
                raise ValueError("FrameSamp-AM recent positions must have integer dtype")

        q_einsum = lora.Einsum(
            shape=(num_heads, width, head_dim),
            name="q_einsum_mem",
            init_fn=nn.initializers.lecun_normal(
                in_axis=-2, out_axis=-1, batch_axis=(0,)
            ),
        )
        kv_einsum = lora.Einsum(
            shape=(2, num_kv_heads, width, head_dim),
            name="kv_einsum_mem",
            init_fn=nn.initializers.lecun_normal(
                in_axis=-2, out_axis=-1, batch_axis=(0, 1)
            ),
        )
        rms_norm = MemoryRMSNorm(name="mem_rms_norm")
        x = rms_norm(x)
        q = q_einsum("BTD,NDH->BTNH", x)

        mem_seq = rms_norm(mem_seq)
        k, v = kv_einsum("BSD,2KDH->2BSKH", mem_seq)

        if use_framesamp_am:
            # The logical teacher always has 512 memory positions.  Neither M
            # nor R may shift action-query RoPE positions.
            q_positions = einops.repeat(
                jnp.arange(512, 512 + x_len), "t -> b t", b=B
            )
            k_positions = framesamp_am_recent_positions
        else:
            q_positions = einops.repeat(
                jnp.arange(mem_len, x_len + mem_len), "t -> b t", b=B
            )
            k_positions = einops.repeat(jnp.arange(mem_len), "t -> b t", b=B)

        q = _apply_rope(q, positions=q_positions)
        if capture_framesamp_am_taps:
            self.sow("framesamp_am_taps", "q_post_rope_pre_scale", q)

        # Preserve the released operation order in the no-AM graph: Q RoPE,
        # Q scale, then K RoPE.  The prior overlay reordered the independent K
        # RoPE, so it could not support a bitwise source-parity claim even
        # though eager CPU module tests remained exact.
        q *= head_dim**-0.5
        k = _apply_rope(k, positions=k_positions)
        if capture_framesamp_am_taps:
            self.sow("framesamp_am_taps", "recent_k_post_rope", k)
            self.sow("framesamp_am_taps", "recent_v_post_projection", v)
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)

        if use_framesamp_am:
            # Compact K/V bypass RMSNorm, kv_einsum and RoPE.  The algebra is
            # future-safe for a disjoint recent block, but v2 admits only R=0.
            k = jnp.concatenate([framesamp_am_compact_k, k], axis=1)
            v = jnp.concatenate([framesamp_am_compact_v, v], axis=1)
            beta = jnp.concatenate(
                [
                    framesamp_am_compact_beta,
                    jnp.zeros((B, mem_len), dtype=jnp.float32),
                ],
                axis=1,
            )
            mem_mask = jnp.concatenate(
                [framesamp_am_compact_mask, mem_mask], axis=1
            )
        else:
            beta = None

        logits = jnp.einsum(
            "BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32
        )
        if beta is not None:
            logits = logits + beta[:, None, None, None, :]
        attn_mask = mem_mask[:, None, None, None, :]
        masked_logits = jnp.where(attn_mask, logits, -2.3819763e38)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
        if capture_framesamp_am_taps:
            self.sow("framesamp_am_taps", "encoded_pre_out", encoded)

        # Keep the released name and shape so all existing checkpoints load.
        out_einsum = lora.Einsum(
            shape=(num_heads, head_dim, width),
            name="out_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
        )
        return out_einsum("BTNH,NHD->BTD", encoded)
'''


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label} patch anchor count must be 1, got {count}")
    return source.replace(old, new, 1)


def render_patched_history_gemma(base_source: bytes) -> bytes:
    """Render the exact reviewed patch only from the audited source bytes."""

    actual = _sha256_bytes(base_source)
    if actual != OFFICIAL_HISTORY_GEMMA_SHA256:
        raise ValueError(
            f"history_gemma.py drifted from audited base: expected {OFFICIAL_HISTORY_GEMMA_SHA256}, got {actual}"
        )
    try:
        source = base_source.decode("utf-8")
    except UnicodeDecodeError as error:  # pragma: no cover - SHA gate is stronger.
        raise ValueError("audited history_gemma.py is not UTF-8") from error

    # Preserve the released MemoryAttention and HistoryBlock definitions
    # byte-for-byte.  The no-AM route must execute the exact released graph;
    # even algebraically neutral extra scan operands can change XLA lowering
    # and therefore action bits.  The AM implementation lives beside the
    # released implementation and is selected only when an artifact is routed.
    source = _replace_once(
        source,
        _MEMORY_ATTENTION_BASE,
        _MEMORY_ATTENTION_BASE + "\n\n" + _MEMORY_ATTENTION_PATCHED,
        label="AMMemoryAttention insertion",
    )

    history_block_start = source.index("@at.typecheck\nclass HistoryBlock")
    history_block_end = source.index("\n\nKVCache:", history_block_start)
    history_block = source[history_block_start:history_block_end]
    am_history_block = history_block.replace(
        "class HistoryBlock(nn.Module):",
        "class AMHistoryBlock(nn.Module):",
        1,
    ).replace(
        'MemoryAttention(name="mem_attn")',
        'AMMemoryAttention(name="mem_attn")',
        1,
    )
    am_history_block = _replace_once(
        am_history_block,
        """        mem_mask,
        deterministic=True,
    ):  # noqa: FBT002
""",
        """        mem_mask,
        framesamp_am_compact_k,
        framesamp_am_compact_v,
        framesamp_am_compact_beta,
        framesamp_am_compact_mask,
        framesamp_am_recent_positions,
        capture_framesamp_am_taps=False,
        deterministic=True,
    ):  # noqa: FBT002
""",
        label="AMHistoryBlock signature",
    )
    am_history_block = _replace_once(
        am_history_block,
        """                    mem_mod_vec = mem_attn(x, mem_seq[-1], mem_mask[-1])
""",
        """                    if framesamp_am_compact_k.shape[1] == 0:
                        mem_mod_vec = mem_attn(
                            x,
                            mem_seq[-1],
                            mem_mask[-1],
                            capture_framesamp_am_taps=capture_framesamp_am_taps,
                        )
                    else:
                        mem_mod_vec = mem_attn(
                            x,
                            mem_seq[-1],
                            mem_mask[-1],
                            framesamp_am_compact_k,
                            framesamp_am_compact_v,
                            framesamp_am_compact_beta,
                            framesamp_am_compact_mask,
                            framesamp_am_recent_positions,
                            capture_framesamp_am_taps,
                        )
""",
        label="AMHistoryBlock MemoryAttention call",
    )
    source = source[:history_block_end] + "\n\n" + am_history_block + source[history_block_end:]

    am_scan_setup = """        am_block_cls = nn.remat(
            AMHistoryBlock,
            prevent_cse=False,
            static_argnums=(12, 13),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.am_layers = nn.scan(
            am_block_cls,
            variable_axes={"params": 0, "framesamp_am_taps": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                0,
                0,
                0,
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            integration_type=self.integration_type,
        )
        nn.share_scope(self.am_layers, self.layers)
"""
    source = _replace_once(
        source,
        """        self.final_norms = [
""",
        am_scan_setup
        + """        self.final_norms = [
""",
        label="AM scan insertion",
    )
    source = _replace_once(
        source,
        """        mem_mask: Sequence[at.Bool[at.Array, "b lmem"] | None] | None = None,
        deterministic: bool = True,
""",
        """        mem_mask: Sequence[at.Bool[at.Array, "b lmem"] | None] | None = None,
        framesamp_am_compact_k=None,
        framesamp_am_compact_v=None,
        framesamp_am_compact_beta=None,
        framesamp_am_compact_mask=None,
        framesamp_am_recent_positions=None,
        capture_framesamp_am_taps: bool = False,
        deterministic: bool = True,
""",
        label="Module signature",
    )
    source = _replace_once(
        source,
        """        embedded, kv_cache = self.layers(
            embedded,
            kv_cache,
            positions,
            mask,
            adarms_cond,
            mem_seq,
            mem_mask,
            deterministic,
        )
""",
        """        am_values = (
            framesamp_am_compact_k,
            framesamp_am_compact_v,
            framesamp_am_compact_beta,
            framesamp_am_compact_mask,
            framesamp_am_recent_positions,
        )
        if any(value is None for value in am_values) and not all(
            value is None for value in am_values
        ):
            raise ValueError("FrameSamp-AM scan inputs must be supplied together")
        if framesamp_am_compact_k is None and not capture_framesamp_am_taps:
            # Keep the released scan call verbatim.  This is a structural
            # parity requirement, not merely an algebraic one.
            embedded, kv_cache = self.layers(
                embedded,
                kv_cache,
                positions,
                mask,
                adarms_cond,
                mem_seq,
                mem_mask,
                deterministic,
            )
        else:
            depth = self.configs[0].depth
            if framesamp_am_compact_k is None:
                # Tap capture is an explicit diagnostic route through the AM
                # scan with M=0.  It observes the released full-memory math but
                # is never used by ordinary no-AM policy inference.
                batch = next(value.shape[0] for value in embedded if value is not None)
                dtype = next(value.dtype for value in embedded if value is not None)
                framesamp_am_compact_k = jnp.empty((depth, batch, 0, 1, 256), dtype=dtype)
                framesamp_am_compact_v = jnp.empty((depth, batch, 0, 1, 256), dtype=dtype)
                framesamp_am_compact_beta = jnp.empty((depth, batch, 0), dtype=jnp.float32)
                framesamp_am_compact_mask = jnp.empty((depth, batch, 0), dtype=jnp.bool_)
                framesamp_am_recent_positions = jnp.empty((batch, 0), dtype=jnp.int32)
            else:
                if self.integration_type != "modulation":
                    raise ValueError("FrameSamp-AM is only defined for modulation integration")
                if framesamp_am_compact_k.ndim != 5 or framesamp_am_compact_k.shape[0] != depth:
                    raise ValueError("FrameSamp-AM compact K must be [layers,B,M,1,256]")
                if framesamp_am_compact_v.shape != framesamp_am_compact_k.shape:
                    raise ValueError("FrameSamp-AM compact V must match compact K")
                if framesamp_am_compact_beta.shape != framesamp_am_compact_k.shape[:3]:
                    raise ValueError("FrameSamp-AM beta must be [layers,B,M]")
                if framesamp_am_compact_beta.dtype != jnp.float32:
                    raise ValueError("FrameSamp-AM beta must be float32")
                if framesamp_am_compact_mask.shape != framesamp_am_compact_k.shape[:3]:
                    raise ValueError("FrameSamp-AM compact mask must be [layers,B,M]")
                if framesamp_am_compact_mask.dtype != jnp.bool_:
                    raise ValueError("FrameSamp-AM compact mask must be bool")
                if mem_seq is None or mem_seq[-1] is None:
                    raise ValueError("FrameSamp-AM requires an explicit empty compact-all memory route")
                if framesamp_am_recent_positions.shape != mem_seq[-1].shape[:2]:
                    raise ValueError("FrameSamp-AM recent position route disagrees with recent memory")
                if mem_seq[-1].shape[1] != 0:
                    raise ValueError("sealed FrameSamp-AM v2 requires compact-all/R=0")
            embedded, kv_cache = self.am_layers(
                embedded,
                kv_cache,
                positions,
                mask,
                adarms_cond,
                mem_seq,
                mem_mask,
                framesamp_am_compact_k,
                framesamp_am_compact_v,
                framesamp_am_compact_beta,
                framesamp_am_compact_mask,
                framesamp_am_recent_positions,
                capture_framesamp_am_taps,
                deterministic,
            )
""",
        label="Module scan call",
    )
    return source.encode("utf-8")


# Filled from the only output of render_patched_history_gemma at the pinned
# base.  Validation remains closed if a transform or source byte changes.
PATCHED_HISTORY_GEMMA_SHA256 = "8d5084e92374296af2bcf9dcff27195df7f02884ca3b3399e3a6289147ce270e"


def _git_output(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise ValueError(f"failed to audit official Git checkout: {detail}") from error


def _audited_tracked_source_files(root: Path) -> tuple[PurePosixPath, ...]:
    raw = _git_output(root, "ls-files", "-z", "--", "src")
    paths = tuple(PurePosixPath(value.decode()) for value in raw.split(b"\0") if value)
    if not paths:
        raise ValueError("official checkout has no tracked src files")
    for relative in paths:
        if relative.is_absolute() or relative.parts[0] != "src" or ".." in relative.parts:
            raise ValueError(f"unsafe tracked source path: {relative}")
        source = root.joinpath(*relative.parts)
        mode = source.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"tracked source is not a regular file: {relative}")
    return tuple(sorted(paths, key=lambda value: value.as_posix()))


def _verify_official_checkout(root: Path) -> tuple[PurePosixPath, ...]:
    root = root.resolve(strict=True)
    head = _git_output(root, "rev-parse", "HEAD").decode().strip()
    if head != OFFICIAL_POLICY_GIT_SHA:
        raise ValueError(f"official policy Git SHA mismatch: expected {OFFICIAL_POLICY_GIT_SHA}, got {head}")
    dirty = _git_output(root, "status", "--porcelain", "--untracked-files=no", "--", "src")
    if dirty:
        raise ValueError("official tracked src tree is dirty; refusing to stage an overlay")
    gemma = root.joinpath(*HISTORY_GEMMA_RELATIVE_PATH.parts)
    pi0 = root.joinpath(*HISTORY_PI0_RELATIVE_PATH.parts)
    if _sha256_file(gemma) != OFFICIAL_HISTORY_GEMMA_SHA256:
        raise ValueError("official history_gemma.py hash mismatch")
    if _sha256_file(pi0) != OFFICIAL_HISTORY_PI0_SHA256:
        raise ValueError("official history_pi0.py hash mismatch")
    return _audited_tracked_source_files(root)


def _tree_sha256(root: Path, files: tuple[PurePosixPath, ...]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        path = root.joinpath(*relative.parts)
        payload = path.read_bytes()
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _path_list_sha256(files: tuple[PurePosixPath, ...]) -> str:
    return _sha256_bytes(b"\0".join(path.as_posix().encode() for path in files))


def _staged_source_files(root: Path) -> tuple[PurePosixPath, ...]:
    """Enumerate the sealed source tree while ignoring interpreter caches."""

    files: list[PurePosixPath] = []
    source_root = root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"staged overlay is missing src/: {root}")
    for path in source_root.rglob("*"):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"staged source contains a non-regular file: {relative}")
        files.append(relative)
    return tuple(sorted(files, key=lambda value: value.as_posix()))


def stage_framesamp_am_flax_overlay(
    official_checkout: str | Path,
    destination: str | Path,
) -> str:
    """Create one immutable staged source tree and return its manifest SHA."""

    official = Path(official_checkout).resolve(strict=True)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace staged FrameSamp-AM overlay: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"overlay parent does not exist: {destination.parent}")
    tracked = _verify_official_checkout(official)
    patched = render_patched_history_gemma(official.joinpath(*HISTORY_GEMMA_RELATIVE_PATH.parts).read_bytes())
    patched_sha = _sha256_bytes(patched)
    if patched_sha != PATCHED_HISTORY_GEMMA_SHA256:
        raise ValueError(
            "rendered FrameSamp-AM source does not match reviewed patch SHA: "
            f"expected {PATCHED_HISTORY_GEMMA_SHA256}, got {patched_sha}"
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        for relative in tracked:
            source = official.joinpath(*relative.parts)
            target = temporary.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            os.chmod(target, stat.S_IMODE(source.stat().st_mode))
        temporary.joinpath(*HISTORY_GEMMA_RELATIVE_PATH.parts).write_bytes(patched)
        source_tree_sha = _tree_sha256(temporary, tracked)
        manifest = {
            "schema_version": OVERLAY_SCHEMA_VERSION,
            "kind": "robomme_framesamp_am_flax_source_overlay",
            "official_policy_git_sha": OFFICIAL_POLICY_GIT_SHA,
            "base_history_gemma_sha256": OFFICIAL_HISTORY_GEMMA_SHA256,
            "patched_history_gemma_sha256": PATCHED_HISTORY_GEMMA_SHA256,
            "history_pi0_sha256": OFFICIAL_HISTORY_PI0_SHA256,
            "history_pi0_status": "unchanged_policy_artifact_route_required",
            "supported_memory_partition": "compact_all_valid_framesamp_tokens_no_recent_v1",
            "policy_runtime_status": "blocked_before_jitted_policy_boundary",
            "unimplemented_call_chain_seam": (
                "HistAugObservation/preprocess_observation -> HistoryPi0.compute_loss/sample_actions "
                "-> history_gemma.Module dynamic task+episode+cut tensors"
            ),
            "tracked_source_file_count": len(tracked),
            "staged_source_paths_sha256": _path_list_sha256(tracked),
            "staged_source_tree_sha256": source_tree_sha,
        }
        manifest_payload = _canonical_json(manifest)
        (temporary / OVERLAY_MANIFEST).write_bytes(manifest_payload)
        os.rename(temporary, destination)
        temporary = None
        return _sha256_bytes(manifest_payload)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def verify_framesamp_am_flax_overlay(
    destination: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, object]:
    """Fail closed on patched source, unchanged policy seam, or manifest drift."""

    root = Path(destination).resolve(strict=True)
    manifest_path = root / OVERLAY_MANIFEST
    payload = manifest_path.read_bytes()
    actual_manifest_sha = _sha256_bytes(payload)
    if actual_manifest_sha != expected_manifest_sha256:
        raise ValueError(
            f"FrameSamp-AM overlay manifest SHA mismatch: expected {expected_manifest_sha256}, got {actual_manifest_sha}"
        )
    try:
        manifest = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("FrameSamp-AM overlay manifest is invalid UTF-8 JSON") from error
    expected_keys = {
        "schema_version",
        "kind",
        "official_policy_git_sha",
        "base_history_gemma_sha256",
        "patched_history_gemma_sha256",
        "history_pi0_sha256",
        "history_pi0_status",
        "supported_memory_partition",
        "policy_runtime_status",
        "unimplemented_call_chain_seam",
        "tracked_source_file_count",
        "staged_source_paths_sha256",
        "staged_source_tree_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("FrameSamp-AM overlay manifest fields mismatch")
    expected_values = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "kind": "robomme_framesamp_am_flax_source_overlay",
        "official_policy_git_sha": OFFICIAL_POLICY_GIT_SHA,
        "base_history_gemma_sha256": OFFICIAL_HISTORY_GEMMA_SHA256,
        "patched_history_gemma_sha256": PATCHED_HISTORY_GEMMA_SHA256,
        "history_pi0_sha256": OFFICIAL_HISTORY_PI0_SHA256,
        "history_pi0_status": "unchanged_policy_artifact_route_required",
        "supported_memory_partition": "compact_all_valid_framesamp_tokens_no_recent_v1",
        "policy_runtime_status": "blocked_before_jitted_policy_boundary",
        "unimplemented_call_chain_seam": (
            "HistAugObservation/preprocess_observation -> HistoryPi0.compute_loss/sample_actions "
            "-> history_gemma.Module dynamic task+episode+cut tensors"
        ),
    }
    for key, expected in expected_values.items():
        if manifest[key] != expected:
            raise ValueError(f"FrameSamp-AM overlay manifest drifted at {key}")
    gemma = root.joinpath(*HISTORY_GEMMA_RELATIVE_PATH.parts)
    pi0 = root.joinpath(*HISTORY_PI0_RELATIVE_PATH.parts)
    if _sha256_file(gemma) != PATCHED_HISTORY_GEMMA_SHA256:
        raise ValueError("staged patched history_gemma.py hash mismatch")
    if _sha256_file(pi0) != OFFICIAL_HISTORY_PI0_SHA256:
        raise ValueError("staged history_pi0.py changed despite the declared policy seam")
    staged_files = _staged_source_files(root)
    if manifest["tracked_source_file_count"] != len(staged_files):
        raise ValueError("staged source file set drifted from the sealed overlay")
    if manifest["staged_source_paths_sha256"] != _path_list_sha256(staged_files):
        raise ValueError("staged source path-list SHA mismatch")
    if manifest["staged_source_tree_sha256"] != _tree_sha256(root, staged_files):
        raise ValueError("staged source tree SHA mismatch")
    return manifest
