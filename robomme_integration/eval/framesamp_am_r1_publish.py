"""Validate and create-once publish one FS-R1 eight-episode screen result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from robomme_integration.eval.framesamp_am_r1_canary import H100_NAME
from robomme_integration.eval.framesamp_am_r1_packet import validate as validate_packet

SCHEMA_VERSION = 1
KIND = "robomme_framesamp_am_r1_oracle_screen_result"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_screen_receipt(
    receipt: dict[str, Any],
    *,
    identity: dict[str, Any],
    episode: int,
) -> None:
    seal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if seal != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("FS-R1 screen lane receipt seal mismatch")
    if (
        receipt.get("kind") != "robomme_framesamp_am_r1_oracle_rollout"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("scope") != "fs_r1_screen_receipt_only_not_statistical_evidence"
        or receipt.get("task") != identity["task"]
        or receipt.get("episode") != episode
        or receipt.get("budget") != identity["budget"]
        or receipt.get("fit_mass") is not identity["fit_mass"]
        or receipt.get("mass_ridge") != identity["mass_ridge"]
        or receipt.get("value_ridge") != identity["value_ridge"]
        or receipt.get("storage_dtype") != identity["storage_dtype"]
        or receipt.get("policy_overlay_manifest_sha256") != identity["overlay_manifest_sha256"]
        or receipt.get("teacher_checkpoint_sha256") != identity["checkpoint_sha256"]
        or receipt.get("canary_replans") is not None
        or not isinstance(receipt.get("success"), bool)
        or receipt.get("fresh_attested_stack_fraction") != 1.0
        or receipt.get("persistent_oracle_payloads") is not False
        or receipt.get("ephemeral_artifacts_deleted_before_receipt") is not True
    ):
        raise ValueError("FS-R1 screen lane receipt semantic mismatch")
    device = receipt.get("device")
    simulator = receipt.get("simulator_runtime")
    if (
        not isinstance(device, dict)
        or device.get("count") != 1
        or device.get("platform") != "gpu"
        or device.get("device_kind") != H100_NAME
        or not isinstance(simulator, dict)
        or simulator.get("gpu_name") != H100_NAME
        or simulator.get("torch_cuda_version") != "12.8"
    ):
        raise ValueError("FS-R1 screen receipt is not exact p5/H100 execution")
    cuts = receipt.get("cuts")
    if not isinstance(cuts, list) or not cuts:
        raise ValueError("FS-R1 screen receipt has no causal replans")
    prior_cut = -1
    executed_total = 0
    for cut in cuts:
        if (
            not isinstance(cut, dict)
            or len(cut.get("layers", [])) != 18
            or cut.get("oracle_payload_deleted_before_simulator_response") is not True
            or not isinstance(cut.get("causal_cut_step"), int)
            or cut["causal_cut_step"] <= prior_cut
            or not 1 <= cut.get("executed_actions", 0) <= 16
        ):
            raise ValueError("FS-R1 screen receipt contains an invalid causal cut")
        prior_cut = cut["causal_cut_step"]
        executed_total += cut["executed_actions"]
        if cut.get("executed_actions_total") != executed_total:
            raise ValueError("FS-R1 screen receipt cumulative action count mismatch")
    if receipt.get("executed_actions") != executed_total or cuts[-1].get("terminal") is not True:
        raise ValueError("FS-R1 screen receipt did not reach an authenticated terminal outcome")


def build_result_claim(
    *,
    packet_path: Path,
    cell_id: str,
    receipt_paths: list[Path],
) -> dict[str, object]:
    packet = _load_json(packet_path)
    validate_packet(packet)
    cells = [cell for cell in packet["oracle_cells"] if cell["cell_id"] == cell_id]
    if len(cells) != 1:
        raise ValueError("FS-R1 result cell ID is absent or duplicated")
    cell = cells[0]
    identity = cell["identity"]
    if len(receipt_paths) != 8:
        raise ValueError("FS-R1 screen result requires exactly eight lane receipts")
    records = []
    for path in receipt_paths:
        receipt = _load_json(path)
        episode = receipt.get("episode")
        if isinstance(episode, bool) or not isinstance(episode, int):
            raise ValueError("FS-R1 receipt episode must be an integer")
        _validate_screen_receipt(receipt, identity=identity, episode=episode)
        records.append(
            {
                "episode": episode,
                "success": receipt["success"],
                "receipt_file_sha256": _sha256_file(path),
                "receipt_sha256": receipt["receipt_sha256"],
                "replans": receipt["replans"],
                "executed_actions": receipt["executed_actions"],
            }
        )
    records.sort(key=lambda row: row["episode"])
    if [row["episode"] for row in records] != list(range(8)):
        raise ValueError("FS-R1 result receipts must cover unique episodes 0..7")
    successes = sum(bool(row["success"]) for row in records)
    floor = packet["promotion"]["minimum_successes"][identity["task"]]
    claim: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "COMPLETE",
        "scope": "n8_regression_screen_not_statistical_evidence",
        "packet_file_sha256": _sha256_file(packet_path),
        "packet_sha256": packet["packet_sha256"],
        "cell_id": cell_id,
        "cell_identity_sha256": cell["identity_sha256"],
        "cell_identity": identity,
        "episodes": records,
        "valid_episodes": 8,
        "harness_failures": 0,
        "successes": successes,
        "promotion_floor": floor,
        "promote_to_paired_fixed50": successes >= floor,
        "fresh_attested_stack_fraction": 1.0,
        "persistent_oracle_payloads": False,
        "interpretation": "infrastructure/large-regression screen only; not statistical evidence",
    }
    claim["claim_sha256"] = hashlib.sha256(_canonical(claim)).hexdigest()
    return claim


def write_create_once(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different FS-R1 result: {path}") from None


def _split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("FS-R1 publication URI must be a full s3://bucket/key URI")
    return parsed.netloc, parsed.path.lstrip("/")


def publish_s3_create_once(uri: str, payload: bytes) -> None:
    import boto3
    from botocore.exceptions import ClientError

    bucket, key = _split_s3_uri(uri)
    client = boto3.client("s3")
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as error:
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status != 412:
            raise
        existing = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        if existing != payload:
            raise FileExistsError(f"different immutable FS-R1 result already exists: {uri}") from error
    stored = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if stored != payload:
        raise RuntimeError("FS-R1 result read-after-write byte verification failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish-s3")
    parser.add_argument("--confirm-publish", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.publish_s3) != args.confirm_publish:
        raise ValueError("S3 publication requires both --publish-s3 and --confirm-publish")
    claim = build_result_claim(
        packet_path=args.packet,
        cell_id=args.cell_id,
        receipt_paths=args.receipt,
    )
    payload = json.dumps(claim, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    write_create_once(args.output, payload)
    if args.publish_s3:
        publish_s3_create_once(args.publish_s3, payload)
    print(json.dumps({"claim_sha256": claim["claim_sha256"], "published": bool(args.publish_s3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
