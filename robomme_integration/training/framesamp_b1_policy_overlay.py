"""Evaluation-only source overlay for the FrameSamp B1 control.

The released checkpoint and model graph remain unchanged.  This overlay alters
only online history selection in ``MME_VLA_Policy``/``MemoryBuffer`` so the
released FrameSamp+Modul policy receives the versioned B1 fixed-demo/raw-live
512-token representation.  Offline training dataset assembly and
``HistoryPi0.compute_loss`` are untouched.
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

from robomme_integration.training.framesamp_b1_data import (
    B1_DEMO_FRAMES,
    B1_LIVE_FRAMES,
    B1_PARTITION_KIND,
    B1_SCHEMA_VERSION,
)

# Kept framework-free: this overlay is authenticated by the control plane
# before the node creates the JAX policy environment.
OFFICIAL_POLICY_GIT_SHA = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"


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


def _verify_official_checkout(root: Path) -> tuple[PurePosixPath, ...]:
    root = root.resolve(strict=True)
    if _git_output(root, "rev-parse", "HEAD").decode().strip() != OFFICIAL_POLICY_GIT_SHA:
        raise ValueError("official policy Git SHA mismatch")
    if _git_output(root, "status", "--porcelain", "--untracked-files=no", "--", "src"):
        raise ValueError("official tracked src tree is dirty")
    raw = _git_output(root, "ls-files", "-z", "--", "src")
    paths = tuple(PurePosixPath(value.decode()) for value in raw.split(b"\0") if value)
    if not paths:
        raise ValueError("official checkout has no tracked src files")
    for relative in paths:
        source = root.joinpath(*relative.parts)
        if (
            relative.is_absolute()
            or relative.parts[0] != "src"
            or ".." in relative.parts
            or not stat.S_ISREG(source.lstat().st_mode)
        ):
            raise ValueError(f"unsafe tracked source path: {relative}")
    return tuple(sorted(paths, key=lambda value: value.as_posix()))


def _staged_source_files(root: Path) -> tuple[PurePosixPath, ...]:
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


def _tree_sha256(root: Path, files: tuple[PurePosixPath, ...]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        payload = root.joinpath(*relative.parts).read_bytes()
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _path_list_sha256(files: tuple[PurePosixPath, ...]) -> str:
    return _sha256_bytes(b"\0".join(path.as_posix().encode() for path in files))


POLICY_RELATIVE_PATH = PurePosixPath("src/mme_vla_suite/policies/policy.py")
MEM_BUFFER_RELATIVE_PATH = PurePosixPath("src/mme_vla_suite/shared/mem_buffer.py")
OFFICIAL_POLICY_SHA256 = "fd7852a99853908b388363094077b5c6ee550a8eb2779982bc52d61cdd7671c7"
OFFICIAL_MEM_BUFFER_SHA256 = "ed37406c663cc9be9035cbadb2b961ad2babcc26d7fb8883535aaff1698efef7"
PATCHED_POLICY_SHA256 = "8aa592a5fb9832b05f75c6d1aa1a1bfecf49934d209cf489a694f0fbc014eb31"
PATCHED_MEM_BUFFER_SHA256 = "f48f2f18aff27cd6f4475dce9de00fb3244cfea30f1dea5a5d6bb1d060edeb24"

OVERLAY_MANIFEST = "framesamp_b1_policy_overlay.json"
OVERLAY_KIND = "robomme_framesamp_b1_evaluation_policy_overlay"


_POLICY_CALL_BASE = """                static_image_emb, static_pos_emb, static_state_emb, static_mask = \\
                    self.mem_buffer.prepare_frame_sampling(
                        self.step_idx, token_budget, token_per_image, history_feats_gather_fn)
"""


_POLICY_CALL_PATCHED = """                static_image_emb, static_pos_emb, static_state_emb, static_mask = \\
                    self.mem_buffer.prepare_frame_sampling_b1(
                        self.step_idx, self.exec_start_idx, token_budget, token_per_image,
                        history_feats_gather_fn)
"""


_MEM_BUFFER_ANCHOR = """    def get_frame_sampling_indices(self, step_idx, token_budget, token_per_image):
"""


_MEM_BUFFER_B1_METHODS = (
    '''    @staticmethod
    def get_frame_sampling_b1_indices(step_idx, exec_start_idx):
        """Return frozen-demo and recent-live frame indices for B1 schema v1."""
        if isinstance(step_idx, (bool, np.bool_)) or not isinstance(step_idx, (int, np.integer)):
            raise TypeError("B1 step_idx must be an integer")
        if isinstance(exec_start_idx, (bool, np.bool_)) or not isinstance(
            exec_start_idx, (int, np.integer)
        ):
            raise TypeError("B1 exec_start_idx must be an integer")
        step_idx = int(step_idx)
        exec_start_idx = int(exec_start_idx)
        if exec_start_idx < 0 or step_idx < exec_start_idx:
            raise ValueError("B1 requires 0 <= exec_start_idx <= step_idx")

        if exec_start_idx < 16:
            demo_indices = list(range(exec_start_idx + 1))
        else:
            demo_indices = np.linspace(0, exec_start_idx, 16, dtype=np.int32).tolist()
        live_start = exec_start_idx + 1
        if step_idx < live_start:
            live_indices = []
        else:
            live_indices = list(range(max(live_start, step_idx - 15), step_idx + 1))
        return demo_indices, live_indices

    def prepare_frame_sampling_b1(
        self,
        step_idx,
        exec_start_idx,
        token_budget,
        token_per_image,
        history_feats_gather_fn,
        *args,
        **kwargs,
    ):
        """Assemble fixed demo16 + raw recent16 in distinct physical slots."""
        if self.num_views != 1 or token_budget != 512 or token_per_image != 16:
            raise ValueError("B1 schema v1 requires one view, 512 tokens, and 16 tokens/frame")
        demo_indices, live_indices = self.get_frame_sampling_b1_indices(
            step_idx, exec_start_idx
        )
        demo_history = history_feats_gather_fn(demo_indices, *args, **kwargs)
        demo = self._prepare_frame_sampling(
            demo_history, demo_indices, 256, token_per_image
        )
        if live_indices:
            live_history = history_feats_gather_fn(live_indices, *args, **kwargs)
            live = self._prepare_frame_sampling(
                live_history, live_indices, 256, token_per_image
            )
        else:
            live = tuple(np.zeros_like(value) for value in demo)
        return tuple(
            np.concatenate([demo_value, live_value], axis=0)
            for demo_value, live_value in zip(demo, live, strict=True)
        )


'''
    + _MEM_BUFFER_ANCHOR
)


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label} patch anchor count must be 1, got {count}")
    return source.replace(old, new, 1)


def render_patched_policy(base_source: bytes) -> bytes:
    if _sha256_bytes(base_source) != OFFICIAL_POLICY_SHA256:
        raise ValueError("official policy.py drifted from the audited B1 base")
    return _replace_once(
        base_source.decode("utf-8"),
        _POLICY_CALL_BASE,
        _POLICY_CALL_PATCHED,
        label="FrameSamp policy history route",
    ).encode("utf-8")


def render_patched_mem_buffer(base_source: bytes) -> bytes:
    if _sha256_bytes(base_source) != OFFICIAL_MEM_BUFFER_SHA256:
        raise ValueError("official mem_buffer.py drifted from the audited B1 base")
    return _replace_once(
        base_source.decode("utf-8"),
        _MEM_BUFFER_ANCHOR,
        _MEM_BUFFER_B1_METHODS,
        label="FrameSamp B1 buffer methods",
    ).encode("utf-8")


def _expected_manifest(source_files: tuple[PurePosixPath, ...], source_tree_sha256: str) -> dict:
    return {
        "schema_version": B1_SCHEMA_VERSION,
        "kind": OVERLAY_KIND,
        "official_policy_git_sha": OFFICIAL_POLICY_GIT_SHA,
        "base_policy_sha256": OFFICIAL_POLICY_SHA256,
        "patched_policy_sha256": PATCHED_POLICY_SHA256,
        "base_mem_buffer_sha256": OFFICIAL_MEM_BUFFER_SHA256,
        "patched_mem_buffer_sha256": PATCHED_MEM_BUFFER_SHA256,
        "representation_policy": {
            "kind": B1_PARTITION_KIND,
            "total_frames": B1_DEMO_FRAMES + B1_LIVE_FRAMES,
            "tokens_per_frame": 16,
            "total_tokens": 512,
            "demo_slots": [0, B1_DEMO_FRAMES - 1],
            "demo_selection": "uniform_inclusive_once_over_0_through_exec_start_idx",
            "live_slots": [B1_DEMO_FRAMES, B1_DEMO_FRAMES + B1_LIVE_FRAMES - 1],
            "live_selection": "most_recent_16_execution_frames_chronological",
            "padding": "each_partition_independently_right_padded",
            "time_features": "episode_global_source_step",
            "memory_attention_physical_slots": "fixed_partition_slot_0_through_31",
            "action_query_rope_offset": 512,
            "single_attention_denominator": True,
            "compression": "none",
        },
        "checkpoint_compatibility": "released_perceptual_framesamp_modul_79999_unchanged",
        "training_dataset_path": "unchanged_official_framesamp_not_b1",
        "scope": "evaluation_only_representation_policy_control",
        "official_equivalence": False,
        "source_file_count": len(source_files),
        "source_paths_sha256": _path_list_sha256(source_files),
        "source_tree_sha256": source_tree_sha256,
    }


def stage_framesamp_b1_policy_overlay(
    official_checkout: str | Path,
    destination: str | Path,
) -> str:
    """Stage an immutable B1 source tree from the exact released checkout."""

    official = Path(official_checkout).resolve(strict=True)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace B1 overlay: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"B1 overlay parent does not exist: {destination.parent}")
    tracked = _verify_official_checkout(official)
    policy = render_patched_policy(official.joinpath(*POLICY_RELATIVE_PATH.parts).read_bytes())
    memory = render_patched_mem_buffer(official.joinpath(*MEM_BUFFER_RELATIVE_PATH.parts).read_bytes())
    if _sha256_bytes(policy) != PATCHED_POLICY_SHA256:
        raise ValueError("rendered B1 policy.py does not match the reviewed SHA")
    if _sha256_bytes(memory) != PATCHED_MEM_BUFFER_SHA256:
        raise ValueError("rendered B1 mem_buffer.py does not match the reviewed SHA")

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        for relative in tracked:
            source = official.joinpath(*relative.parts)
            target = temporary.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            os.chmod(target, stat.S_IMODE(source.stat().st_mode))
        temporary.joinpath(*POLICY_RELATIVE_PATH.parts).write_bytes(policy)
        temporary.joinpath(*MEM_BUFFER_RELATIVE_PATH.parts).write_bytes(memory)
        source_files = _staged_source_files(temporary)
        manifest = _expected_manifest(source_files, _tree_sha256(temporary, source_files))
        payload = _canonical_json(manifest)
        (temporary / OVERLAY_MANIFEST).write_bytes(payload)
        os.rename(temporary, destination)
        temporary = None
        return _sha256_bytes(payload)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def verify_framesamp_b1_policy_overlay(
    destination: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict:
    """Authenticate the B1 overlay manifest and complete staged source tree."""

    root = Path(destination).resolve(strict=True)
    payload = (root / OVERLAY_MANIFEST).read_bytes()
    if _sha256_bytes(payload) != expected_manifest_sha256:
        raise ValueError("B1 overlay manifest SHA mismatch")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("B1 overlay manifest is invalid UTF-8 JSON") from error
    source_files = _staged_source_files(root)
    expected = _expected_manifest(source_files, _tree_sha256(root, source_files))
    if manifest != expected:
        raise ValueError("B1 overlay manifest or source inventory drifted")
    if _sha256_file(root.joinpath(*POLICY_RELATIVE_PATH.parts)) != PATCHED_POLICY_SHA256:
        raise ValueError("B1 policy.py SHA mismatch")
    if _sha256_file(root.joinpath(*MEM_BUFFER_RELATIVE_PATH.parts)) != PATCHED_MEM_BUFFER_SHA256:
        raise ValueError("B1 mem_buffer.py SHA mismatch")
    return manifest
