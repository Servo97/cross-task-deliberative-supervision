#!/usr/bin/env python3
"""Precompute ω over every RoboCerebra episode with the retrained encoder, + a local manifest.

Runs only after the retrained encoder clears the pre-registered G1b bar. Emits one ``w.npz`` per
episode in the same layout the robocasa study store uses (``w``, ``frame_indices``,
``lang_global``), so the mechanism arms' ω readers consume it unchanged.

Convention, identical to training: pi05_libero tap, 2 views (agentview + eye-in-hand), 128
tokens. Nothing is written to S3 — the manifest is local and content-addressed, matching the
sealed-store discipline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import torch


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tap", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifests", required=True)
    parser.add_argument("--artifact", default="robocerebra_omega_v1")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from workspace_models.networks.workspace_latent import WorkspaceEncoder

    device = torch.device("cuda")
    blob = torch.load(args.encoder, map_location="cpu")
    cfg = SimpleNamespace(**blob["cfg"])
    encoder = WorkspaceEncoder(cfg)
    encoder.load_state_dict(
        {k[len("encoder.") :]: v for k, v in blob["model"].items() if k.startswith("encoder.")}, strict=False
    )
    encoder.eval().to(device)

    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(pathlib.Path(args.tap).glob("episode_*.npz"))
    frames_total = 0
    for path in paths:
        episode = int(path.stem.split("_")[1])
        destination = out_root / f"episode_{episode:06d}" / "w.npz"
        if destination.exists():
            continue
        data = np.load(path)
        with torch.no_grad():
            omega = (
                encoder(
                    torch.from_numpy(data["tokens"].astype(np.float32)).unsqueeze(0).to(device),
                    torch.from_numpy(data["pooled_img"].astype(np.float32)).unsqueeze(0).to(device),
                    torch.from_numpy(data["pooled_lang"].astype(np.float32)).unsqueeze(0).to(device),
                )[0]
                .float()
                .cpu()
                .numpy()
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            w=omega.astype(np.float32),
            frame_indices=data["frame_idx"],
            lang_global=data["pooled_lang"].astype(np.float16),
            subtask_index=data["subtask_index"],
        )
        frames_total += len(omega)

    files = sorted(p for p in out_root.rglob("*") if p.is_file())
    objects = [
        {"key": str(p.relative_to(out_root)), "size_bytes": p.stat().st_size, "checksum_sha256": sha256_file(p)}
        for p in files
    ]
    manifest = {
        "schema_version": 1,
        "artifact": args.artifact,
        "root_s3": None,
        "content_addressing": "sha256",
        "selection": {
            "kind": "robocerebra_omega",
            "encoder": {
                "path": str(args.encoder),
                "step": int(blob.get("step", -1)),
                "sha256": sha256_file(pathlib.Path(args.encoder)),
                "eval": blob.get("eval"),
            },
            "convention": {"tap": "pi05_libero", "views": 2, "patch_tokens": 128},
            "episodes": len(paths),
            "frames": frames_total,
        },
        "objects": objects,
    }
    digest = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    manifests = pathlib.Path(args.manifests)
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{digest}.json").write_bytes(canonical_bytes(manifest))
    print(
        f"omega episodes={len(paths)} frames={frames_total} "
        f"bytes={sum(o['size_bytes'] for o in objects)} -> {manifests / (digest + '.json')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
