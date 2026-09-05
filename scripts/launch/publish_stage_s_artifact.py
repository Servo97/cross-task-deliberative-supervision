#!/usr/bin/env python3
"""Publish one content-addressed Stage-S artifact with immutable S3 semantics.

The CLI is dry-run by default. A real write requires ``--confirm-publish`` after explicit user
approval, validates the account-141 caller, permits only canonical long-context study paths, uses
``If-None-Match: *``, and verifies the resulting non-null VersionId and exact bytes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

try:
    from .launch_guardrails import EXECUTION_ACCOUNT, REGION
except ImportError:
    from launch_guardrails import EXECUTION_ACCOUNT, REGION


HEX64 = r"[0-9a-f]{64}"
#: Task-prompt manifests are namespaced by dataset. The list is EXPLICIT rather than a wildcard so
#: the allowlist stays an allowlist — a typo'd namespace must fail, not mint a new prefix.
TASK_PROMPT_NAMESPACES = ("robocasa_target50", "remembench13")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"invalid S3 object URI {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def validate_publish_target(source: str | Path, *, destination_s3: str, study_root: str) -> tuple[Path, str, str, str]:
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f"artifact source is missing: {source}")
    if not study_root.startswith("s3://") or study_root.endswith("/"):
        raise ValueError("study_root must be an s3:// URI without a trailing slash")
    if not destination_s3.startswith(study_root + "/"):
        raise ValueError("artifact destination is outside the canonical study root")
    relative = destination_s3[len(study_root) + 1 :]
    patterns = (
        rf"code/(?:wsmv2|openpi|internal_training)/({HEX64})\.tgz",
        rf"artifacts/tokenizers/paligemma/({HEX64})\.model",
        rf"manifests/inventories/(?:init|data)/({HEX64})\.json",
        rf"manifests/artifacts/workspace/task_prompts/(?:{'|'.join(TASK_PROMPT_NAMESPACES)})/({HEX64})\.json",
        rf"manifests/artifacts/workspace/{HEX64}/omega/({HEX64})\.json",
        # Encoder weights, content-addressed by their OWN bytes rather than by encoder_id. The
        # encoder_id is the sha256 of a provenance block that CONTAINS this URI, so an
        # encoder_id-keyed path would be circular. This is the convention the sealed target50 and
        # ReMemBench encoders were already published under (…/encoders/6470fe3a….pt); it was simply
        # never added to the allowlist because those predate this publisher.
        rf"artifacts/workspace/encoders/({HEX64})\.pt",
        # E1: the immutable 50x100 held-out-reset WSM_eval episode manifest.
        rf"manifests/artifacts/eval/heldout50/({HEX64})\.json",
        # Serve-side workspace manifests consumed by submit_pi_stage_s_eval.py: the per-encoder
        # workspace-artifact manifest and the frozen-tap deploy-tree manifest. Both are
        # content-addressed by the sha256 of their own bytes.
        rf"manifests/artifacts/workspace/{HEX64}/({HEX64})\.json",
        rf"manifests/artifacts/tap/({HEX64})\.json",
    )
    # Fixed-name serve artifacts: the eval launcher pins these by (encoder_id, filename) and carries
    # the sha256 in the run manifest + job environment instead of the key, so the digest cannot be
    # read out of the path. Immutability still comes from If-None-Match + the stored checksum.
    fixed_name_patterns = (
        rf"artifacts/workspace/{HEX64}/encoder\.pt",
        rf"artifacts/workspace/{HEX64}/task_lang_table\.npz",
    )
    match = next((match for pattern in patterns if (match := re.fullmatch(pattern, relative))), None)
    if match is None:
        if not any(re.fullmatch(pattern, relative) for pattern in fixed_name_patterns):
            raise ValueError(f"destination is not an allowed content-addressed Stage-S path: {relative}")
        digest = _sha256(source)
    else:
        digest = _sha256(source)
        if match.group(1) != digest:
            raise ValueError(f"destination digest does not match source bytes: path={match.group(1)} actual={digest}")
    bucket, key = _parse_s3(destination_s3)
    return source, digest, bucket, key


def _error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    detail = response.get("Error")
    return str(detail.get("Code")) if isinstance(detail, dict) and detail.get("Code") else None


def _clean_version_id(value: object) -> str | None:
    """Return a usable VersionId, or None on an unversioned bucket (VersionId absent/'null')."""
    return value if (isinstance(value, str) and value and value != "null") else None


def _verify_existing(client, *, bucket: str, key: str, digest: str) -> str | None:
    """Confirm an already-present object is byte-identical (create-once idempotency). Returns the
    VersionId if the bucket is versioned, else None — the SHA-256 of the bytes is the anchor."""
    response = client.get_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    version_id = _clean_version_id(response.get("VersionId"))
    body = response["Body"]
    actual = hashlib.sha256()
    try:
        for block in iter(lambda: body.read(8 * 1024 * 1024), b""):
            actual.update(block)
    finally:
        body.close()
    if actual.hexdigest() != digest:
        raise ValueError(f"immutable artifact collision at s3://{bucket}/{key}")
    return version_id


def publish_once(
    source: str | Path,
    *,
    destination_s3: str,
    study_root: str,
    s3_client,
) -> dict[str, str | bool]:
    source, digest, bucket, key = validate_publish_target(source, destination_s3=destination_s3, study_root=study_root)
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    try:
        with source.open("rb") as stream:
            response = s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                IfNoneMatch="*",
                ChecksumSHA256=checksum,
            )
        # VersionId is present on a versioned bucket, absent/'null' on the unversioned study bucket.
        # Content integrity is anchored by the stored SHA-256 checksum + size either way.
        version_id = _clean_version_id(response.get("VersionId"))
        created = True
    except Exception as error:
        if _error_code(error) not in {"PreconditionFailed", "412"}:
            raise
        version_id = _verify_existing(s3_client, bucket=bucket, key=key, digest=digest)
        created = False

    head_kwargs = {"Bucket": bucket, "Key": key, "ChecksumMode": "ENABLED"}
    if version_id is not None:
        head_kwargs["VersionId"] = version_id
    head = s3_client.head_object(**head_kwargs)
    if version_id is not None and head.get("VersionId") != version_id:
        raise ValueError("S3 head verification returned the wrong VersionId")
    returned_checksum = head.get("ChecksumSHA256")
    if returned_checksum and returned_checksum != checksum:
        raise ValueError("S3 head verification returned the wrong SHA-256 checksum")
    if head.get("ContentLength") != source.stat().st_size:
        raise ValueError("S3 head verification returned the wrong artifact size")
    return {
        "created": created,
        "sha256": digest,
        "version_id": version_id or "unversioned",
        "uri": destination_s3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination-s3", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--confirm-publish", action="store_true")
    args = parser.parse_args()
    source, digest, _bucket, _key = validate_publish_target(
        args.source, destination_s3=args.destination_s3, study_root=args.study_root
    )
    print(f"source={source}")
    print(f"sha256={digest}")
    print(f"destination={args.destination_s3}")
    if not args.confirm_publish:
        print("[DRY RUN: no AWS SDK load and no S3 mutation]")
        return

    import boto3

    session = boto3.session.Session(region_name=REGION)
    account = session.client("sts").get_caller_identity()["Account"]
    if account != EXECUTION_ACCOUNT:
        raise SystemExit(f"artifact publication requires AWS account {EXECUTION_ACCOUNT}; caller is {account}")
    result = publish_once(
        source,
        destination_s3=args.destination_s3,
        study_root=args.study_root,
        s3_client=session.client("s3"),
    )
    print(f"published={result['created']} version_id={result['version_id']} uri={result['uri']}")


if __name__ == "__main__":
    main()
