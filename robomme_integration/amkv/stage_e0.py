#!/usr/bin/env python3
"""Content-addressed, create-once S3 staging for the H10 attention-matching E0 job.

Three inputs must exist in S3 before ``launch_e0`` may seal a run manifest:

1. the official ``robomme_policy_learning`` source as a deterministic tarball
   plus a create-once receipt binding its extracted bytes to the clean Git tree,
2. the ``perceptual-framesamp-modul/79999`` orbax checkpoint tree plus a sealed
   size/SHA-256 inventory the node re-verifies after download, and
3. the E0 fixtures bundle emitted by :mod:`robomme_integration.amkv.episodes`.

Every destination key is derived from the content it holds, so staging is
idempotent: an existing key is verified and left alone, never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_UTILS = REPO_ROOT / "scripts" / "launch"
for _entry in (str(REPO_ROOT), str(LAUNCH_UTILS)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from build_deterministic_archive import build_archive  # noqa: E402
from launch_guardrails import DEFAULT_RESULTS_BUCKET, STUDY_OWNER, wsm_settings  # noqa: E402

OWNER = STUDY_OWNER
STUDY = "long_context_v1"
STUDY_ROOT = f"s3://{DEFAULT_RESULTS_BUCKET}/{OWNER}/wsm_robocasa/studies/{STUDY}"
AMKV_ROOT = f"{STUDY_ROOT}/amkv"

POLICY_SOURCE_COMPONENT = "robomme_policy_learning"
PINNED_POLICY_GIT_SHA = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
PINNED_POLICY_TREE_SHA1 = "cb8c33dce3a6f19f731481f30bbab4dca66ee768"
CHECKPOINT_ARTIFACT = "framesamp_modul_79999"
CHECKPOINT_PREFIX = f"{AMKV_ROOT}/artifacts/checkpoints/{CHECKPOINT_ARTIFACT}"
INVENTORY_PREFIX = f"{AMKV_ROOT}/manifests/inventories/amkv"
FIXTURES_PREFIX = f"{AMKV_ROOT}/artifacts/amkv_fixtures"
SOURCE_RECEIPT_PREFIX = f"{AMKV_ROOT}/manifests/source_receipts/{POLICY_SOURCE_COMPONENT}"

OFFICIAL_REFERENCE = wsm_settings.ROBOMME_EVAL_ROOT / "official_reference"
POLICY_SOURCE_DIR = OFFICIAL_REFERENCE / POLICY_SOURCE_COMPONENT
CHECKPOINT_DIR = OFFICIAL_REFERENCE / "checkpoints" / "perceptual-framesamp-modul"
# The official loader reads ``checkpoint_dir.parent / "history_config.txt"``,
# so the staged tree must be the STEP PARENT, not the step directory alone.
CHECKPOINT_STEP = "79999"
CHECKPOINT_REQUIRED = (
    f"{CHECKPOINT_STEP}/params",
    f"{CHECKPOINT_STEP}/assets",
    f"{CHECKPOINT_STEP}/_CHECKPOINT_METADATA",
    "history_config.txt",
)
FIXTURES_PAYLOAD = "fixtures.npz"
FIXTURES_MANIFEST = "manifest.json"
INVENTORY_SCHEMA_VERSION = 1
SOURCE_RECEIPT_SCHEMA_VERSION = 1
SOURCE_RECEIPT_KIND = "amkv_policy_source_stage_receipt"
SOURCE_TREE_ALGORITHM = "amkv_extracted_source_tree_sha256_v1"
DEFAULT_ARCHIVE_DIR = Path(tempfile.gettempdir()) / "amkv-stage"


# ---------------------------------------------------------------- URI helpers


def policy_source_uri(sha256: str) -> str:
    return f"{AMKV_ROOT}/code/{POLICY_SOURCE_COMPONENT}/{sha256}.tgz"


def source_receipt_uri(sha256: str) -> str:
    return f"{SOURCE_RECEIPT_PREFIX}/{sha256}.json"


def checkpoint_uri() -> str:
    return CHECKPOINT_PREFIX


def checkpoint_inventory_uri(sha256: str) -> str:
    return f"{INVENTORY_PREFIX}/{sha256}.json"


def fixtures_uri(manifest_sha256: str) -> str:
    return f"{FIXTURES_PREFIX}/{manifest_sha256}"


def split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise SystemExit(f"not an s3 uri: {uri}")
    location = uri[len("s3://") :]
    bucket, _, key = location.partition("/")
    if not bucket or not key:
        raise SystemExit(f"incomplete s3 uri: {uri}")
    return bucket, key


# ------------------------------------------------------------- local hashing


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _safe_relative_key(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise SystemExit(f"unsafe source archive member: {value!r}")
    return value


def source_tree_identity(root: Path) -> dict:
    """Hash every extracted file, directory, and symlink under ``root``.

    This is deliberately computed from an extraction of the deterministic
    archive, rather than from the Git checkout.  The receipt therefore binds
    the exact bytes the node will import, including file modes and symlink
    targets, to the reviewed clean Git commit/tree.
    """

    root = Path(root).resolve()
    if not root.is_dir():
        raise SystemExit(f"source identity root is missing: {root}")
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        key = _safe_relative_key(path.relative_to(root).as_posix())
        status = path.lstat()
        record: dict[str, object] = {
            "key": key,
            "mode": stat.S_IMODE(status.st_mode),
        }
        if stat.S_ISLNK(status.st_mode):
            record.update(type="symlink", target=os.readlink(path))
        elif stat.S_ISDIR(status.st_mode):
            record.update(type="directory")
        elif stat.S_ISREG(status.st_mode):
            record.update(type="file", size_bytes=status.st_size, sha256=sha256_file(path))
        else:
            raise SystemExit(f"unsupported extracted source entry: {path}")
        records.append(record)
    if not records:
        raise SystemExit(f"extracted source is empty: {root}")
    tree_sha = hashlib.sha256(canonical_json({"objects": records}).encode()).hexdigest()
    return {
        "algorithm": SOURCE_TREE_ALGORITHM,
        "tree_sha256": tree_sha,
        "objects": records,
        "totals": {
            "objects": len(records),
            "files": sum(record["type"] == "file" for record in records),
            "bytes": sum(int(record.get("size_bytes", 0)) for record in records),
        },
    }


def safe_extract_source_archive(archive_path: Path, destination: Path) -> None:
    """Extract the generated source archive without accepting path traversal."""

    archive_path = Path(archive_path)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [_safe_relative_key(member.name.rstrip("/")) for member in members]
        if len(names) != len(set(names)):
            raise SystemExit("source archive contains duplicate members")
        for member in members:
            key = member.name.rstrip("/")
            target = destination / key
            for parent in target.parents:
                if parent == destination:
                    break
                if parent.is_symlink():
                    raise SystemExit(f"source archive traverses a symlink ancestor: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=False)
                target.chmod(member.mode)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise SystemExit(f"cannot read source archive member: {member.name}")
                with target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                target.chmod(member.mode)
            elif member.issym():
                link = Path(member.linkname)
                if link.is_absolute() or ".." in link.parts:
                    raise SystemExit(f"unsafe source symlink target: {member.name} -> {member.linkname}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)
            else:
                raise SystemExit(f"unsupported source archive member type: {member.name}")


def validate_source_receipt(document: dict) -> dict:
    """Validate a receipt and return it unchanged, or fail closed."""

    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "kind",
        "component",
        "git",
        "archive",
        "extracted_tree",
    }:
        raise SystemExit("source receipt has an unknown or incomplete top-level schema")
    if document["schema_version"] != SOURCE_RECEIPT_SCHEMA_VERSION:
        raise SystemExit(f"unsupported source receipt schema {document['schema_version']}")
    if document["kind"] != SOURCE_RECEIPT_KIND or document["component"] != POLICY_SOURCE_COMPONENT:
        raise SystemExit("source receipt kind/component mismatch")
    git = document["git"]
    if git != {
        "git_sha": PINNED_POLICY_GIT_SHA,
        "git_tree_sha1": PINNED_POLICY_TREE_SHA1,
        "worktree_status": "clean_including_untracked_and_submodules",
    }:
        raise SystemExit("source receipt does not bind the pinned clean policy Git tree")
    archive = document["archive"]
    if not isinstance(archive, dict) or set(archive) != {"bytes", "sha256", "uri"}:
        raise SystemExit("source receipt archive block is malformed")
    digest = str(archive.get("sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit("source receipt archive SHA-256 is malformed")
    if archive["uri"] != policy_source_uri(digest) or not isinstance(archive["bytes"], int) or archive["bytes"] < 1:
        raise SystemExit("source receipt archive identity is inconsistent")
    tree = document["extracted_tree"]
    if not isinstance(tree, dict) or set(tree) != {"algorithm", "tree_sha256", "objects", "totals"}:
        raise SystemExit("source receipt extracted-tree block is malformed")
    if tree["algorithm"] != SOURCE_TREE_ALGORITHM or not isinstance(tree["objects"], list) or not tree["objects"]:
        raise SystemExit("source receipt extracted-tree algorithm/objects are invalid")
    keys = []
    for record in tree["objects"]:
        if not isinstance(record, dict):
            raise SystemExit("source receipt contains a non-object tree record")
        key = _safe_relative_key(str(record.get("key", "")))
        keys.append(key)
        if record.get("type") not in {"file", "directory", "symlink"}:
            raise SystemExit(f"source receipt has an invalid entry type for {key}")
        if not isinstance(record.get("mode"), int):
            raise SystemExit(f"source receipt has an invalid mode for {key}")
        if record["type"] == "file":
            sha = str(record.get("sha256", ""))
            if len(sha) != 64 or any(character not in "0123456789abcdef" for character in sha):
                raise SystemExit(f"source receipt has an invalid file hash for {key}")
            if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0:
                raise SystemExit(f"source receipt has an invalid file size for {key}")
        elif record["type"] == "symlink" and not isinstance(record.get("target"), str):
            raise SystemExit(f"source receipt has an invalid symlink target for {key}")
    if keys != sorted(set(keys)):
        raise SystemExit("source receipt extracted-tree objects are not sorted and unique")
    expected_tree_sha = hashlib.sha256(canonical_json({"objects": tree["objects"]}).encode()).hexdigest()
    if tree["tree_sha256"] != expected_tree_sha:
        raise SystemExit("source receipt extracted-tree digest mismatch")
    expected_totals = {
        "objects": len(tree["objects"]),
        "files": sum(record["type"] == "file" for record in tree["objects"]),
        "bytes": sum(int(record.get("size_bytes", 0)) for record in tree["objects"]),
    }
    if tree["totals"] != expected_totals:
        raise SystemExit("source receipt extracted-tree totals mismatch")
    return document


def source_receipt_bytes(*, archive_path: Path, archive_sha256: str, git_identity: dict[str, str]) -> bytes:
    """Build the canonical receipt from a fresh extraction of ``archive_path``."""

    archive_path = Path(archive_path)
    if sha256_file(archive_path) != archive_sha256:
        raise SystemExit("source archive changed while its receipt was being built")
    with tempfile.TemporaryDirectory(prefix="amkv-source-receipt-") as temporary:
        extracted = Path(temporary) / "source"
        safe_extract_source_archive(archive_path, extracted)
        tree = source_tree_identity(extracted)
    document = {
        "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "kind": SOURCE_RECEIPT_KIND,
        "component": POLICY_SOURCE_COMPONENT,
        "git": {
            **git_identity,
            "worktree_status": "clean_including_untracked_and_submodules",
        },
        "archive": {
            "uri": policy_source_uri(archive_sha256),
            "sha256": archive_sha256,
            "bytes": archive_path.stat().st_size,
        },
        "extracted_tree": tree,
    }
    validate_source_receipt(document)
    return (canonical_json(document) + "\n").encode()


def write_source_receipt_once(*, payload: bytes, archive_dir: Path) -> tuple[Path, str]:
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(archive_dir) / "source-receipts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(f"content-addressed source receipt collision: {path}")
    else:
        temporary = path.with_suffix(".json.incomplete")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return path, digest


def load_source_receipt(path: Path, *, expected_sha256: str | None = None) -> tuple[dict, str]:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"source receipt is missing: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SystemExit(f"source receipt digest mismatch: expected {expected_sha256}, got {digest}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"source receipt is not valid JSON: {error}") from error
    return validate_source_receipt(document), digest


def local_tree(root: Path) -> list[dict]:
    """Sorted relative-path/size/sha256 records for every regular file under ``root``."""
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise SystemExit(f"refusing to stage a symlink: {path}")
        if not path.is_file():
            continue
        records.append(
            {
                "key": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise SystemExit(f"nothing to stage under {root}")
    return records


def build_inventory(root: Path, *, artifact: str, root_s3: str) -> tuple[str, str]:
    """Return the canonical inventory document and its own SHA-256."""
    objects = local_tree(root)
    document = canonical_json(
        {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "artifact": artifact,
            "root_s3": root_s3,
            "objects": objects,
            "totals": {
                "objects": len(objects),
                "bytes": sum(record["size_bytes"] for record in objects),
            },
        }
    )
    return document, hashlib.sha256(document.encode()).hexdigest()


def require_pinned_clean_policy_source(source_dir: Path) -> dict[str, str]:
    """Require the exact clean official Git tree before an archive can exist.

    The archive SHA binds submitted bytes, while the commit and root-tree object
    bind those bytes to the reviewed upstream source rather than merely to the
    one ``history_gemma.py`` file patched by AMKV.
    """

    source_dir = Path(source_dir).resolve()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(source_dir), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise SystemExit(f"cannot verify official policy Git source: {' '.join(args)}: {result.stderr.strip()}")
        return result.stdout.strip()

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    if root != source_dir:
        raise SystemExit(f"policy source must be the Git toplevel {root}, got {source_dir}")
    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    if commit != PINNED_POLICY_GIT_SHA:
        raise SystemExit(f"official policy HEAD drifted: expected {PINNED_POLICY_GIT_SHA}, got {commit}")
    if tree != PINNED_POLICY_TREE_SHA1:
        raise SystemExit(f"official policy root tree drifted: expected {PINNED_POLICY_TREE_SHA1}, got {tree}")
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    if status:
        raise SystemExit(f"official policy source is not clean; refusing to stage:\n{status}")
    return {"git_sha": commit, "git_tree_sha1": tree}


# ---------------------------------------------------------------- aws helpers


def aws(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["aws", *arguments], capture_output=True, text=True, check=check)


def object_exists(uri: str) -> bool:
    bucket, key = split_uri(uri)
    result = aws("s3api", "head-object", "--bucket", bucket, "--key", key, check=False)
    if result.returncode == 0:
        return True
    message = f"{result.stdout}\n{result.stderr}".lower()
    if any(marker in message for marker in ("404", "not found", "nosuchkey")):
        return False
    raise subprocess.CalledProcessError(
        result.returncode,
        result.args,
        output=result.stdout,
        stderr=result.stderr,
    )


def list_prefix(prefix: str) -> dict[str, int]:
    """Relative key -> size for every object under ``prefix`` (empty when absent)."""
    bucket, key = split_uri(prefix)
    key = key.rstrip("/") + "/"
    listing: dict[str, int] = {}
    token: str | None = None
    while True:
        arguments = ["s3api", "list-objects-v2", "--bucket", bucket, "--prefix", key]
        if token:
            arguments += ["--continuation-token", token]
        result = aws(*arguments)
        payload = json.loads(result.stdout or "{}")
        for record in payload.get("Contents", []):
            listing[record["Key"][len(key) :]] = record["Size"]
        token = payload.get("NextContinuationToken")
        if not token:
            return listing


def put_object_once(body: Path, uri: str) -> str:
    """Create-once upload; an identical existing object is accepted, a different one fails."""
    bucket, key = split_uri(uri)
    result = aws(
        "s3api",
        "put-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--body",
        str(body),
        "--if-none-match",
        "*",
        check=False,
    )
    if result.returncode == 0:
        return "created"
    if not object_exists(uri):
        raise SystemExit(f"upload failed for {uri}: {result.stderr.strip()}")
    return "exists"


def sync_once(local: Path, prefix: str, expected: dict[str, int]) -> str:
    """``aws s3 sync`` a tree that either is absent or already matches ``expected``."""
    present = list_prefix(prefix)
    if present == expected:
        return "exists"
    if present:
        raise SystemExit(f"refusing to overwrite {prefix}: {len(present)} objects already differ from the plan")
    aws("s3", "sync", str(local), prefix.rstrip("/"), "--only-show-errors", "--no-follow-symlinks")
    verified = list_prefix(prefix)
    if verified != expected:
        missing = sorted(set(expected) - set(verified))
        raise SystemExit(f"post-upload verification failed for {prefix}: missing/mismatched {missing[:5]}")
    return "created"


# ------------------------------------------------------------------- stages


def stage_source(*, source_dir: Path, archive_dir: Path, dry_run: bool) -> dict:
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise SystemExit(f"official policy source is missing: {source_dir}")
    git_identity = require_pinned_clean_policy_source(source_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    path, digest, uri = build_archive(
        source_dir,
        output_dir=archive_dir,
        component=POLICY_SOURCE_COMPONENT,
        study_root=AMKV_ROOT,
    )
    if uri != policy_source_uri(digest):
        raise SystemExit(f"archive builder returned a noncanonical policy URI: {uri}")
    receipt_payload = source_receipt_bytes(
        archive_path=path,
        archive_sha256=digest,
        git_identity=git_identity,
    )
    receipt_path, receipt_sha = write_source_receipt_once(
        payload=receipt_payload,
        archive_dir=archive_dir,
    )
    summary = {
        "stage": "source",
        "local": str(source_dir),
        "archive": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        **git_identity,
        "uri": uri,
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "receipt_uri": source_receipt_uri(receipt_sha),
        "dry_run": dry_run,
    }
    if dry_run:
        summary["action"] = "planned"
        summary["receipt_action"] = "planned"
        return summary
    summary["action"] = put_object_once(path, uri)
    if not object_exists(uri):
        raise SystemExit(f"post-upload verification failed for {uri}")
    summary["receipt_action"] = put_object_once(receipt_path, summary["receipt_uri"])
    if not object_exists(summary["receipt_uri"]):
        raise SystemExit(f"post-upload verification failed for {summary['receipt_uri']}")
    return summary


def stage_checkpoint(*, checkpoint_dir: Path, dry_run: bool) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    missing = [name for name in CHECKPOINT_REQUIRED if not (checkpoint_dir / name).exists()]
    if missing:
        raise SystemExit(f"checkpoint {checkpoint_dir} is missing {missing}")
    document, inventory_sha = build_inventory(checkpoint_dir, artifact=CHECKPOINT_ARTIFACT, root_s3=checkpoint_uri())
    objects = json.loads(document)["objects"]
    expected = {record["key"]: record["size_bytes"] for record in objects}
    summary = {
        "stage": "checkpoint",
        "local": str(checkpoint_dir),
        "objects": len(objects),
        "bytes": sum(expected.values()),
        "uri": checkpoint_uri(),
        "inventory_uri": checkpoint_inventory_uri(inventory_sha),
        "inventory_sha256": inventory_sha,
        "dry_run": dry_run,
    }
    if dry_run:
        summary["action"] = "planned"
        return summary
    summary["action"] = sync_once(checkpoint_dir, checkpoint_uri(), expected)
    with tempfile.TemporaryDirectory(prefix="amkv-inventory-") as temporary:
        staged = Path(temporary) / f"{inventory_sha}.json"
        staged.write_text(document, encoding="utf-8")
        summary["inventory_action"] = put_object_once(staged, summary["inventory_uri"])
    return summary


def stage_fixtures(*, bundle_dir: Path, dry_run: bool) -> dict:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / FIXTURES_MANIFEST
    payload_path = bundle_dir / FIXTURES_PAYLOAD
    for path in (manifest_path, payload_path):
        if not path.is_file():
            raise SystemExit(f"fixtures bundle {bundle_dir} is missing {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = manifest.get("payload_sha256")
    actual = sha256_file(payload_path)
    if claimed != actual:
        raise SystemExit(f"fixtures payload_sha256 mismatch: manifest={claimed} actual={actual}")
    manifest_sha = sha256_file(manifest_path)
    objects = local_tree(bundle_dir)
    expected = {record["key"]: record["size_bytes"] for record in objects}
    summary = {
        "stage": "fixtures",
        "local": str(bundle_dir),
        "objects": len(objects),
        "bytes": sum(expected.values()),
        "payload_sha256": actual,
        "manifest_sha256": manifest_sha,
        "uri": fixtures_uri(manifest_sha),
        "dry_run": dry_run,
    }
    summary["action"] = "planned" if dry_run else sync_once(bundle_dir, summary["uri"], expected)
    return summary


def status(*, source_dir: Path, checkpoint_dir: Path, deep: bool) -> dict:
    def s3(call):
        try:
            return call()
        except (SystemExit, subprocess.CalledProcessError, json.JSONDecodeError) as error:
            return {"error": str(error)}

    report: dict = {"root": AMKV_ROOT, "deep": deep, "stages": {}}
    source = {"local": str(source_dir), "local_present": Path(source_dir).is_dir()}
    if source["local_present"]:
        try:
            git_identity = require_pinned_clean_policy_source(source_dir)
            with tempfile.TemporaryDirectory(prefix="amkv-status-") as temporary:
                path, digest, uri = build_archive(
                    source_dir,
                    output_dir=Path(temporary),
                    component=POLICY_SOURCE_COMPONENT,
                    study_root=AMKV_ROOT,
                )
                receipt_payload = source_receipt_bytes(
                    archive_path=path,
                    archive_sha256=digest,
                    git_identity=git_identity,
                )
            receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
            receipt_uri = source_receipt_uri(receipt_sha)
            source.update(
                sha256=digest,
                uri=uri,
                receipt_sha256=receipt_sha,
                receipt_uri=receipt_uri,
                git_sha=git_identity["git_sha"],
                git_tree_sha1=git_identity["git_tree_sha1"],
            )
            source["staged"] = s3(lambda: object_exists(uri))
            source["receipt_staged"] = s3(lambda: object_exists(receipt_uri))
        except SystemExit as error:
            source["local_validation_error"] = str(error)
    report["stages"]["source"] = source

    checkpoint = {
        "local": str(checkpoint_dir),
        "local_present": Path(checkpoint_dir).is_dir(),
        "uri": checkpoint_uri(),
    }
    listing = s3(lambda: list_prefix(checkpoint_uri()))
    if isinstance(listing, dict) and "error" in listing:
        checkpoint["staged"] = listing
    else:
        checkpoint["staged_objects"] = len(listing)
        checkpoint["staged_bytes"] = sum(listing.values())
    if deep and checkpoint["local_present"]:
        _document, inventory_sha = build_inventory(
            checkpoint_dir, artifact=CHECKPOINT_ARTIFACT, root_s3=checkpoint_uri()
        )
        checkpoint["inventory_sha256"] = inventory_sha
        checkpoint["inventory_uri"] = checkpoint_inventory_uri(inventory_sha)
        checkpoint["inventory_staged"] = s3(lambda: object_exists(checkpoint_inventory_uri(inventory_sha)))
    report["stages"]["checkpoint"] = checkpoint

    inventories = s3(lambda: list_prefix(INVENTORY_PREFIX))
    fixtures = s3(lambda: list_prefix(FIXTURES_PREFIX))
    report["stages"]["inventories"] = {"prefix": INVENTORY_PREFIX, "keys": inventories}
    report["stages"]["fixtures"] = {
        "prefix": FIXTURES_PREFIX,
        "bundles": (
            fixtures
            if isinstance(fixtures, dict) and "error" in fixtures
            else sorted({key.split("/")[0] for key in fixtures})
        ),
    }
    return report


# --------------------------------------------------------- environment preflight

PINNED_PYTHON = "3.11"
ENV_PREFLIGHT_MARKER = "amkv_env_preflight.json"
ENV_PREFLIGHT_IMPORTS = ("jax", "openpi", "mme_vla_suite")


def env_preflight_marker_path(archive_dir: Path = DEFAULT_ARCHIVE_DIR) -> Path:
    return Path(archive_dir) / ENV_PREFLIGHT_MARKER


def lock_sha256(source_dir: Path) -> str:
    lock = Path(source_dir) / "uv.lock"
    if not lock.is_file():
        raise SystemExit(f"official policy source has no uv.lock: {lock}")
    return sha256_file(lock)


def env_preflight(
    *,
    source_dir: Path,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
    python: str = PINNED_PYTHON,
    keep: bool = False,
) -> dict:
    """Resolve and BUILD the node environment in a clean scratch tree.

    Standing practice for this lane after E0 attempt 1: an already-built local
    venv cannot see the resolution step, so it cannot catch "this interpreter
    has no wheel for a transitive dependency and will fall through to an sdist".
    That is exactly what burned a node (mujoco 2.3.7 is cp311-only in the lock;
    an unpinned sync picked 3.12 and tried to build it from source).  This runs
    the real ``uv sync --frozen`` against a fresh environment directory and then
    imports what the job imports.
    """

    source_dir = Path(source_dir).resolve()
    lock_sha = lock_sha256(source_dir)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="amkv-envcheck-") as temporary:
        environment = Path(temporary) / "venv"
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(environment), UV_PYTHON=python)
        subprocess.run(
            ["uv", "sync", "--frozen", "--python", python],
            cwd=source_dir,
            env=env,
            check=True,
        )
        interpreter = environment / "bin" / "python"
        if not interpreter.is_file():
            raise SystemExit(f"uv sync produced no interpreter at {interpreter}")
        actual = subprocess.run(
            [str(interpreter), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != python:
            raise SystemExit(f"clean-scratch sync resolved python {actual}, expected {python}")
        subprocess.run(
            [str(interpreter), "-c", "import " + ", ".join(ENV_PREFLIGHT_IMPORTS)],
            cwd=source_dir,
            check=True,
            env=dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES=""),
        )
        if keep:
            shutil.copytree(environment, archive_dir / "envcheck-venv", dirs_exist_ok=True)
    marker = {
        "schema_version": 1,
        "kind": "amkv_env_preflight",
        "policy_source_dir": str(source_dir),
        "uv_lock_sha256": lock_sha,
        "python": python,
        "imports_verified": list(ENV_PREFLIGHT_IMPORTS),
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = env_preflight_marker_path(archive_dir)
    path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"stage": "env-preflight", "marker": str(path), **marker}


def require_env_preflight(*, source_dir: Path, archive_dir: Path = DEFAULT_ARCHIVE_DIR) -> dict:
    """Refuse to submit until a clean-scratch build of THIS lock has passed."""

    path = env_preflight_marker_path(archive_dir)
    if not path.is_file():
        raise SystemExit(
            "no environment preflight on record; run "
            "`python -m robomme_integration.amkv.stage_e0 env-preflight` before submitting "
            "(standing practice after E0 attempt 1)"
        )
    marker = json.loads(path.read_text(encoding="utf-8"))
    expected = lock_sha256(source_dir)
    if marker.get("uv_lock_sha256") != expected:
        raise SystemExit(
            "environment preflight is stale: it covered uv.lock "
            f"{marker.get('uv_lock_sha256')}, the staged source has {expected}"
        )
    if marker.get("python") != PINNED_PYTHON:
        raise SystemExit(f"environment preflight pinned python {marker.get('python')}, expected {PINNED_PYTHON}")
    return marker


# ---------------------------------------------------------------------- cli


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=("source", "checkpoint", "fixtures", "status", "env-preflight"))
    value.add_argument("--source-dir", type=Path, default=POLICY_SOURCE_DIR)
    value.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    value.add_argument("--fixtures-dir", type=Path, help="directory holding fixtures.npz + manifest.json")
    value.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    value.add_argument("--deep", action="store_true", help="status: also hash the checkpoint tree")
    value.add_argument("--dry-run", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.stage == "source":
        summary = stage_source(source_dir=args.source_dir, archive_dir=args.archive_dir, dry_run=args.dry_run)
    elif args.stage == "checkpoint":
        summary = stage_checkpoint(checkpoint_dir=args.checkpoint_dir, dry_run=args.dry_run)
    elif args.stage == "env-preflight":
        summary = env_preflight(source_dir=args.source_dir, archive_dir=args.archive_dir)
    elif args.stage == "fixtures":
        if not args.fixtures_dir:
            raise SystemExit("fixtures staging requires --fixtures-dir")
        summary = stage_fixtures(bundle_dir=args.fixtures_dir, dry_run=args.dry_run)
    else:
        summary = status(source_dir=args.source_dir, checkpoint_dir=args.checkpoint_dir, deep=args.deep)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
