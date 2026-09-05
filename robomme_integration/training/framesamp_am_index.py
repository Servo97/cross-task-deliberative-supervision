"""Create-once trusted index for sealed RoboMME FrameSamp-AM bundles.

An AM bundle cannot nominate its own trusted manifest SHA.  This index is the
out-of-band allowlist: callers must pin the index file SHA, and every record is
keyed by teacher checkpoint/code, the full environment/layer route, requested
budget, and the bundle's complete scientific-manifest SHA.  Loading reopens the
bundle through :mod:`framesamp_am_artifact`, including payload verification.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from robomme_integration.training.framesamp_am_artifact import (
    MANIFEST_FILENAME,
    PAYLOAD_FILENAME,
    ExpectedFrameSampAMIdentity,
    FrameSampAMManifest,
    LoadedFrameSampAMArtifact,
    load_framesamp_am_artifact,
)

TRUSTED_INDEX_SCHEMA_VERSION = 1
TRUSTED_INDEX_KIND = "robomme_framesamp_attention_matching_bundle_index"
_HEX = frozenset("0123456789abcdef")


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _require_sha(value: object, *, label: str, lengths: tuple[int, ...] = (64,)) -> str:
    value = _require_nonempty(value, label=label)
    if len(value) not in lengths or any(character not in _HEX for character in value):
        raise ValueError(f"{label} must be a lowercase {'/'.join(map(str, lengths))}-hex SHA")
    return value


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_manifest(bundle: Path) -> FrameSampAMManifest:
    path = bundle / MANIFEST_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"indexed AM bundle is missing {MANIFEST_FILENAME}: {bundle}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"indexed AM manifest is not valid UTF-8 JSON: {bundle}") from error
    return FrameSampAMManifest.from_dict(raw)


def _expected(manifest: FrameSampAMManifest) -> ExpectedFrameSampAMIdentity:
    return ExpectedFrameSampAMIdentity(
        teacher_checkpoint_sha256=manifest.teacher_checkpoint_sha256,
        teacher_code_sha=manifest.teacher_code_sha,
        task_id=manifest.task_id,
        episode_id=manifest.episode_id,
        causal_cut_step=manifest.causal_cut_step,
        layer_index=manifest.layer_index,
        kv_head_index=manifest.kv_head_index,
        requested_budget=manifest.requested_budget,
        manifest_sha256=manifest.scientific_sha256(),
    )


def _safe_relative_bundle(bundle: str | Path, root: Path) -> tuple[Path, str]:
    candidate = Path(bundle)
    if not candidate.exists():
        raise FileNotFoundError(f"indexed AM bundle does not exist: {candidate}")
    if candidate.is_symlink():
        raise ValueError(f"indexed AM bundle must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"indexed AM bundle must be below the trusted-index directory: {candidate}") from error
    if not relative.parts:
        raise ValueError("trusted-index directory itself cannot be an AM bundle")
    for name in (MANIFEST_FILENAME, PAYLOAD_FILENAME):
        child = resolved / name
        if not child.exists():
            raise FileNotFoundError(f"indexed AM bundle is missing {name}: {resolved}")
        if child.is_symlink():
            raise ValueError(f"indexed AM bundle file must not be a symlink: {child}")
    return resolved, relative.as_posix()


@dataclasses.dataclass(frozen=True)
class FrameSampAMIndexRecord:
    bundle_relative_path: str
    teacher_checkpoint_sha256: str
    teacher_code_sha: str
    task_id: str
    episode_id: str
    causal_cut_step: int
    layer_index: int
    kv_head_index: int
    requested_budget: int
    variant_sha256: str
    manifest_sha256: str
    artifact_key_sha256: str

    def _key_dict(self) -> dict[str, object]:
        return {
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "teacher_code_sha": self.teacher_code_sha,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "causal_cut_step": self.causal_cut_step,
            "layer_index": self.layer_index,
            "kv_head_index": self.kv_head_index,
            "requested_budget": self.requested_budget,
            "variant_sha256": self.variant_sha256,
            "manifest_sha256": self.manifest_sha256,
        }

    def validate(self) -> None:
        path = PurePosixPath(_require_nonempty(self.bundle_relative_path, label="bundle_relative_path"))
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise ValueError("bundle_relative_path must be a normalized relative POSIX path")
        if path.as_posix() != self.bundle_relative_path:
            raise ValueError("bundle_relative_path is not canonically normalized")
        _require_sha(self.teacher_checkpoint_sha256, label="teacher_checkpoint_sha256")
        _require_sha(self.teacher_code_sha, label="teacher_code_sha", lengths=(40, 64))
        _require_nonempty(self.task_id, label="task_id")
        _require_nonempty(self.episode_id, label="episode_id")
        _require_int(self.causal_cut_step, label="causal_cut_step")
        _require_int(self.layer_index, label="layer_index")
        if _require_int(self.kv_head_index, label="kv_head_index") != 0:
            raise ValueError("official MemoryAttention KV head must be 0")
        _require_int(self.requested_budget, label="requested_budget", minimum=1)
        _require_sha(self.variant_sha256, label="variant_sha256")
        _require_sha(self.manifest_sha256, label="manifest_sha256")
        _require_sha(self.artifact_key_sha256, label="artifact_key_sha256")
        expected_key = _sha256_bytes(_canonical_json(self._key_dict()))
        if self.artifact_key_sha256 != expected_key:
            raise ValueError("artifact_key_sha256 disagrees with teacher/routing/budget/manifest key")

    def base_route(self) -> tuple[object, ...]:
        return (
            self.teacher_checkpoint_sha256,
            self.teacher_code_sha,
            self.task_id,
            self.episode_id,
            self.causal_cut_step,
            self.layer_index,
            self.kv_head_index,
            self.requested_budget,
        )

    def route(self) -> tuple[object, ...]:
        """Unambiguous producer/solver variant at one task/layer/budget route."""

        return self.base_route() + (self.variant_sha256,)

    def full_key(self) -> tuple[object, ...]:
        return self.route() + (self.manifest_sha256,)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "FrameSampAMIndexRecord":
        if not isinstance(value, dict):
            raise ValueError("trusted-index record must be an object")
        names = {field.name for field in dataclasses.fields(cls)}
        if set(value) != names:
            raise ValueError("trusted-index record fields mismatch")
        result = cls(**value)
        result.validate()
        return result

    @classmethod
    def from_manifest(cls, relative_path: str, manifest: FrameSampAMManifest) -> "FrameSampAMIndexRecord":
        manifest.validate()
        variant_sha = _sha256_bytes(
            _canonical_json(
                {
                    "artifact_method": manifest.artifact_method,
                    "fit_mass": manifest.fit_mass,
                    "mass_solver": manifest.mass_solver,
                    "value_solver": manifest.value_solver,
                    "mass_ridge": manifest.mass_ridge,
                    "value_ridge": manifest.value_ridge,
                    "fit_query_bank_sha256": manifest.fit_query_bank_sha256,
                    "heldout_query_bank_sha256": manifest.heldout_query_bank_sha256,
                    "storage_dtype": manifest.storage_dtype,
                }
            )
        )
        values = {
            "bundle_relative_path": relative_path,
            "teacher_checkpoint_sha256": manifest.teacher_checkpoint_sha256,
            "teacher_code_sha": manifest.teacher_code_sha,
            "task_id": manifest.task_id,
            "episode_id": manifest.episode_id,
            "causal_cut_step": manifest.causal_cut_step,
            "layer_index": manifest.layer_index,
            "kv_head_index": manifest.kv_head_index,
            "requested_budget": manifest.requested_budget,
            "variant_sha256": variant_sha,
            "manifest_sha256": manifest.scientific_sha256(),
        }
        artifact_key = _sha256_bytes(
            _canonical_json({key: values[key] for key in values if key != "bundle_relative_path"})
        )
        result = cls(**values, artifact_key_sha256=artifact_key)
        result.validate()
        return result

    def expected_identity(self) -> ExpectedFrameSampAMIdentity:
        self.validate()
        return ExpectedFrameSampAMIdentity(
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            teacher_code_sha=self.teacher_code_sha,
            task_id=self.task_id,
            episode_id=self.episode_id,
            causal_cut_step=self.causal_cut_step,
            layer_index=self.layer_index,
            kv_head_index=self.kv_head_index,
            requested_budget=self.requested_budget,
            manifest_sha256=self.manifest_sha256,
        )


@dataclasses.dataclass(frozen=True)
class FrameSampAMTrustedIndex:
    records: tuple[FrameSampAMIndexRecord, ...]
    schema_version: int = TRUSTED_INDEX_SCHEMA_VERSION
    kind: str = TRUSTED_INDEX_KIND

    def validate(self) -> None:
        if _require_int(self.schema_version, label="schema_version") != TRUSTED_INDEX_SCHEMA_VERSION:
            raise ValueError(f"unsupported trusted-index schema {self.schema_version}")
        if self.kind != TRUSTED_INDEX_KIND:
            raise ValueError("unsupported trusted-index kind")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("trusted index must contain at least one immutable record")
        for record in self.records:
            record.validate()
        expected_order = tuple(
            sorted(self.records, key=lambda record: record.full_key() + (record.bundle_relative_path,))
        )
        if self.records != expected_order:
            raise ValueError("trusted-index records are not in canonical key order")
        routes: dict[tuple[object, ...], str] = {}
        full_keys: set[tuple[object, ...]] = set()
        manifest_routes: dict[str, tuple[object, ...]] = {}
        paths: set[str] = set()
        for record in self.records:
            if record.full_key() in full_keys or record.bundle_relative_path in paths:
                raise ValueError("trusted index contains a duplicate artifact key or bundle path")
            full_keys.add(record.full_key())
            paths.add(record.bundle_relative_path)
            previous_manifest = routes.get(record.route())
            if previous_manifest is not None:
                if previous_manifest == record.manifest_sha256:
                    raise ValueError("trusted index contains a duplicate artifact route")
                raise ValueError("trusted index contains a routing collision with different manifest SHAs")
            routes[record.route()] = record.manifest_sha256
            previous_route = manifest_routes.get(record.manifest_sha256)
            if previous_route is not None and previous_route != record.route():
                raise ValueError("trusted index maps one full manifest SHA to multiple routes")
            manifest_routes[record.manifest_sha256] = record.route()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, value: object) -> "FrameSampAMTrustedIndex":
        if not isinstance(value, dict) or set(value) != {"schema_version", "kind", "records"}:
            raise ValueError("trusted FrameSamp-AM index root fields mismatch")
        raw_records = value["records"]
        if not isinstance(raw_records, list):
            raise ValueError("trusted-index records must be a list")
        result = cls(
            schema_version=value["schema_version"],
            kind=value["kind"],
            records=tuple(FrameSampAMIndexRecord.from_dict(record) for record in raw_records),
        )
        result.validate()
        return result


def _bundle_for_record(index_path: Path, record: FrameSampAMIndexRecord) -> Path:
    root = index_path.parent.resolve(strict=True)
    unresolved = root / PurePosixPath(record.bundle_relative_path)
    if not unresolved.exists():
        raise FileNotFoundError(f"trusted-index artifact is missing: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if unresolved.is_symlink() or resolved != unresolved.absolute():
        raise ValueError(f"trusted-index artifact path traverses a symlink: {unresolved}")
    try:
        resolved.relative_to(root)
    except ValueError as error:  # pragma: no cover - guarded by normalized relative path plus symlink rejection.
        raise ValueError("trusted-index artifact escapes its root") from error
    for name in (MANIFEST_FILENAME, PAYLOAD_FILENAME):
        child = resolved / name
        if not child.exists():
            raise FileNotFoundError(f"trusted-index artifact is missing {name}: {resolved}")
        if child.is_symlink():
            raise ValueError(f"trusted-index artifact file is a symlink: {child}")
    return resolved


def _verify_record_bundle(index_path: Path, record: FrameSampAMIndexRecord) -> LoadedFrameSampAMArtifact:
    bundle = _bundle_for_record(index_path, record)
    loaded = load_framesamp_am_artifact(bundle, expected=record.expected_identity())
    derived = FrameSampAMIndexRecord.from_manifest(record.bundle_relative_path, loaded.manifest)
    if record != derived:
        raise ValueError("trusted-index routing record disagrees with its sealed artifact")
    return loaded


@dataclasses.dataclass(frozen=True)
class LoadedFrameSampAMTrustedIndex:
    index: FrameSampAMTrustedIndex
    path: Path
    file_sha256: str

    def resolve(self, expected: ExpectedFrameSampAMIdentity) -> LoadedFrameSampAMArtifact:
        """Resolve one exact full key; never fall back to route-only matching."""

        expected.validate()
        expected_base_route = (
            expected.teacher_checkpoint_sha256,
            expected.teacher_code_sha,
            expected.task_id,
            expected.episode_id,
            expected.causal_cut_step,
            expected.layer_index,
            expected.kv_head_index,
            expected.requested_budget,
        )
        for record in self.index.records:
            if record.base_route() == expected_base_route and record.manifest_sha256 == expected.manifest_sha256:
                return _verify_record_bundle(self.path, record)
        if any(record.base_route() == expected_base_route for record in self.index.records):
            raise ValueError("manifest SHA is not trusted for the requested artifact route")
        if any(record.manifest_sha256 == expected.manifest_sha256 for record in self.index.records):
            raise ValueError("trusted manifest SHA was requested with the wrong routing identity")
        raise KeyError("requested FrameSamp-AM artifact is absent from the trusted index")


def create_framesamp_am_trusted_index(destination: str | Path, bundles: Iterable[str | Path]) -> str:
    """Integrity-check bundles and atomically create a non-replaceable index.

    The returned SHA must be stored in experiment configuration and supplied to
    :func:`load_framesamp_am_trusted_index`; the index never self-authorizes.
    """

    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace trusted FrameSamp-AM index: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"trusted-index parent does not exist: {destination.parent}")
    root = destination.parent.resolve(strict=True)
    records: list[FrameSampAMIndexRecord] = []
    for bundle in bundles:
        resolved, relative = _safe_relative_bundle(bundle, root)
        manifest = _read_manifest(resolved)
        load_framesamp_am_artifact(resolved, expected=_expected(manifest))
        records.append(FrameSampAMIndexRecord.from_manifest(relative, manifest))
    records.sort(key=lambda record: record.full_key() + (record.bundle_relative_path,))
    index = FrameSampAMTrustedIndex(records=tuple(records))
    payload = _canonical_json(index.to_dict())
    digest = _sha256_bytes(payload)

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        # Hard-link publication is atomic and fails if a competing creator won;
        # unlike os.replace it can never overwrite an existing trusted index.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    load_framesamp_am_trusted_index(destination, expected_sha256=digest, verify_artifacts=True)
    return digest


def load_framesamp_am_trusted_index(
    path: str | Path,
    *,
    expected_sha256: str,
    verify_artifacts: bool = True,
) -> LoadedFrameSampAMTrustedIndex:
    """Load only under an externally pinned SHA and optionally verify all bundles."""

    expected_sha256 = _require_sha(expected_sha256, label="trusted index SHA256")
    path = Path(path)
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError(f"trusted FrameSamp-AM index SHA256 mismatch: {actual} != {expected_sha256}")
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("trusted FrameSamp-AM index is not valid UTF-8 JSON") from error
    index = FrameSampAMTrustedIndex.from_dict(raw)
    loaded = LoadedFrameSampAMTrustedIndex(index=index, path=path.resolve(strict=True), file_sha256=actual)
    if verify_artifacts:
        for record in index.records:
            _verify_record_bundle(loaded.path, record)
    return loaded
