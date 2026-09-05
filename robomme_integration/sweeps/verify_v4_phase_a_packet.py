#!/usr/bin/env python3
"""Verify byte-exact identity between the v4 cell packet and all clumped lane bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from robomme_integration.campaign import validate_manifest


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def verify(packet_path: Path, lane_root: Path) -> dict:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    clean = dict(packet)
    claimed = clean.pop("packet_sha256", None)
    actual = hashlib.sha256(_canonical(clean).encode()).hexdigest()
    if claimed != actual:
        raise ValueError(f"packet self-seal mismatch: {claimed} != {actual}")
    packet_cells = {(cell["task"], cell["arm"]): cell for cell in packet["ready_cells"]}
    lane_cells: dict[tuple[str, str], dict] = {}
    campaign_ids: set[str] = set()
    manifest_uris: set[str] = set()
    attempts: set[str] = set()
    lane_count = 0
    for lane in sorted(path for path in lane_root.iterdir() if path.is_dir()):
        campaign_path = lane / "_robomme_gpu_campaign_manifest.json"
        if not campaign_path.is_file():
            continue
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        validate_manifest(campaign)
        lane_count += 1
        if campaign["campaign_id"] in campaign_ids:
            raise ValueError("duplicate campaign ID across v4 lanes")
        campaign_ids.add(campaign["campaign_id"])
        for record in campaign["cells"]:
            identity = (record["task"], record["arm"])
            if identity in lane_cells:
                raise ValueError(f"duplicate lane cell {identity}")
            staged = lane / record["run_manifest_source"]
            manifest = json.loads(staged.read_text(encoding="utf-8"))
            lane_cells[identity] = manifest
            uri = manifest["manifest_s3"]
            attempt = manifest["attempt_id"]
            if uri in manifest_uris or attempt in attempts:
                raise ValueError("duplicate run attempt or manifest namespace across lanes")
            manifest_uris.add(uri)
            attempts.add(attempt)
            expected = packet_cells.get(identity)
            if expected is None:
                raise ValueError(f"lane cell absent from packet: {identity}")
            if manifest != expected["sealed_manifest"]:
                raise ValueError(f"lane/packet immutable manifest collision for {identity}")
            if record["run_manifest_sha256"] != expected["manifest_sha256"]:
                raise ValueError(f"lane/packet manifest SHA mismatch for {identity}")
    if set(lane_cells) != set(packet_cells):
        raise ValueError(
            f"lane coverage differs from packet: missing={sorted(set(packet_cells) - set(lane_cells))} "
            f"extra={sorted(set(lane_cells) - set(packet_cells))}"
        )
    return {
        "schema_version": 1,
        "kind": "robomme_v4_phase_a_packet_verification",
        "cloud_action": False,
        "packet_sha256": claimed,
        "prepared_source_tree_sha256": packet["source_identity"]["submitted_prepared_source_tree_sha256"],
        "lane_count": lane_count,
        "cell_count": len(lane_cells),
        "unique_campaign_ids": len(campaign_ids),
        "unique_attempt_ids": len(attempts),
        "unique_manifest_namespaces": len(manifest_uris),
        "byte_exact_lane_packet_manifests": True,
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--lane-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = verify(args.packet, args.lane_root)
    rendered = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
