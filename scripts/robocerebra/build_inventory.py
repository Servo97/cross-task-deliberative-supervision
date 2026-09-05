#!/usr/bin/env python3
"""Build a content-addressed inventory manifest for a RoboCerebra dataset tree. LOCAL ONLY.

Same discipline and same canonical-JSON encoding as
``scripts/launch/build_stage_s_inventory.py`` and ``scripts/remembench/upload_and_inventory.py``:
the manifest filename is the sha256 of the manifest's own canonical bytes, and every object
carries its own sha256. The difference from the remembench builder is that this one **does not
touch S3** -- no upload, no version ids, no bucket. It writes ``<sha>.json`` next to the tree so
the dataset can be pinned and diffed locally; S3 staging is a separate, gated step that only has
to add ``version_id``/``etag`` per object.

    python build_inventory.py --root <lerobot dataset dir> --artifact robocerebra_train_v1 \
        --out <wsm_data>/robocerebra/manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="LeRobot dataset directory to pin")
    parser.add_argument("--artifact", required=True, help="artifact name, e.g. robocerebra_train_v1")
    parser.add_argument("--out", required=True, help="directory to write <sha>.json into")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"no files under {root}")

    objects = [
        {
            "key": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "checksum_sha256": sha256_file(path),
        }
        for path in files
    ]

    info = json.loads((root / "meta" / "info.json").read_text())
    provenance_path = root / "meta" / "episode_provenance.jsonl"
    provenance = (
        [json.loads(line) for line in provenance_path.read_text().splitlines()] if provenance_path.is_file() else []
    )
    scenes: dict[str, int] = {}
    for entry in provenance:
        scenes[entry["scene"]] = scenes.get(entry["scene"], 0) + 1

    manifest = {
        "schema_version": 1,
        "artifact": args.artifact,
        "root_s3": None,  # local-only manifest; S3 staging is a separate gated step
        "content_addressing": "sha256",
        "selection": {
            "name": args.artifact,
            "kind": "robocerebra_all_cases",
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "tasks": info["total_tasks"],
            "scenes": dict(sorted(scenes.items())),
            "cases": sorted(f"{e['scene']}/{e['case']}" for e in provenance),
        },
        "objects": objects,
    }
    digest = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"{digest}.json"
    destination.write_bytes(canonical_bytes(manifest))
    print(
        f"artifact={args.artifact} objects={len(objects)} "
        f"bytes={sum(o['size_bytes'] for o in objects)} -> {destination}"
    )


if __name__ == "__main__":
    main()
