#!/usr/bin/env python3
"""Finalize a completed producer run: publish its omega manifest canonically + print submits.

The producer writes the omega cache and its content-addressed manifest into ITS RUN OUTPUT
(`.../producer/producer/<run_id>/omega/…`), but every downstream launcher pins the manifest at the
CANONICAL immutable URI `<study>/manifests/artifacts/workspace/<encoder_id>/omega/<sha>.json`
(the train entry refuses anything else). This tool bridges the two — the one manual step between
"producer finished" and "S1/S2/Q1/Q3 are one command each":

  1. locate + download the omega manifest in the run output, verify sha == filename;
  2. re-validate the manifest chain fields and extract encoder_id;
  3. publish it create-once to the canonical URI (needs --confirm-publish);
  4. print the FILLED submit commands (canary + production) for s1, s2, q1, q3.

Weekend usage (after the producer completion notification):
  eval "$(aws configure export-credentials --format env)"
  python scripts/launch/finalize_producer_omega.py --producer-run <run_id> --confirm-publish
  # then paste the printed canary commands (needs the per-run go-ahead), later the productions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from launch_guardrails import DEXJOCO_IMAGE_REPO, LONG_CONTEXT_STUDY_S3, wsm_settings

STUDY = LONG_CONTEXT_STUDY_S3
SM_PY = str(wsm_settings.ENVS_ROOT / "sm_launch" / "bin" / "python")
LAUNCH = Path(__file__).resolve().parent
IMAGE = f"{DEXJOCO_IMAGE_REPO}@sha256:798592894178d6430f3265060c5ea745abb77eee5818d5ff9a831ef2652266f2"
TOKENIZER_SHA = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
INIT_INV_SHA = "34932efcfeee9b11181a5915ecce3be47aaeb01b5bf9e3f5057c022f4db01b04"
TARGET_INV_SHA = "b2366b34156b76aee7030bdb33c33b89a9668a4e39b202d623ff34844cb4f41c"
PROMPT_SHA = "277e7f07c20186ce66dee0b15e7a09660a6ab25f69a280d60e9bd4d85bf5ec3d"


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stderr[-500:]}")
    return r.stdout


def locate_manifest(run_id: str) -> tuple[str, str]:
    """(manifest_s3_uri, sha) for the single <64hex>.json under the run's omega output."""
    prefix = f"{STUDY}/producer/producer/{run_id}/omega/"
    listing = _run(["aws", "s3", "ls", prefix])
    names = [line.split()[-1] for line in listing.splitlines() if line.strip().endswith(".json")]
    hex64 = [n for n in names if len(n) == 69 and all(c in "0123456789abcdef" for c in n[:64])]
    if len(hex64) != 1:
        raise SystemExit(f"{prefix}: expected exactly one <sha256>.json omega manifest, got {names}")
    return prefix + hex64[0], hex64[0][:64]


def fetch_and_verify(uri: str, sha: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = Path(tmp.name)
    _run(["aws", "s3", "cp", uri, str(path), "--only-show-errors"])
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha:
        raise SystemExit(f"omega manifest content sha {actual} != filename {sha}")
    manifest = json.loads(data)
    for key in ("encoder_id", "encoder_provenance", "tasks"):
        if key not in manifest:
            raise SystemExit(f"omega manifest missing {key}")
    n_tasks = len(manifest["tasks"])
    n_eps = sum(len(t["episodes"]) for t in manifest["tasks"])
    print(f"omega manifest OK: encoder_id={manifest['encoder_id']} tasks={n_tasks} episodes={n_eps}")
    if n_tasks != 50 or n_eps != 7500:
        raise SystemExit("expected 50 tasks x 150 episodes = 7500 omega archives")
    manifest["_local_path"] = str(path)
    return manifest


def publish(manifest: dict, sha: str, confirm: bool) -> str:
    canonical = f"{STUDY}/manifests/artifacts/workspace/{manifest['encoder_id']}/omega/{sha}.json"
    cmd = [
        SM_PY,
        str(LAUNCH / "publish_stage_s_artifact.py"),
        "--source",
        manifest["_local_path"],
        "--destination-s3",
        canonical,
        "--study-root",
        STUDY,
    ]
    if confirm:
        cmd.append("--confirm-publish")
    print(_run(cmd).strip())
    return canonical


def print_submits(run_id: str, manifest: dict, sha: str, canonical_uri: str, wsmv2_s3: str, openpi_s3: str) -> None:
    omega_cache = f"{STUDY}/producer/producer/{run_id}/omega"
    common = f"""  --wsmv2-source-s3 {wsmv2_s3} \\
  --openpi-source-s3 {openpi_s3} \\
  --tokenizer-s3 {STUDY}/artifacts/tokenizers/paligemma/{TOKENIZER_SHA}.model \\
  --tokenizer-sha256 {TOKENIZER_SHA} \\
  --init-inventory-s3 {STUDY}/manifests/inventories/init/{INIT_INV_SHA}.json \\
  --init-inventory-sha256 {INIT_INV_SHA} \\
  --target-inventory-s3 {STUDY}/manifests/inventories/data/{TARGET_INV_SHA}.json \\
  --target-inventory-sha256 {TARGET_INV_SHA} \\
  --image-uri {IMAGE} \\
  --encoder-id {manifest["encoder_id"]} \\
  --policy-features-s3 {omega_cache} \\
  --policy-features-manifest-s3 {canonical_uri} \\
  --policy-features-manifest-sha256 {sha} \\
  --task-prompt-manifest-s3 {STUDY}/manifests/artifacts/workspace/task_prompts/robocasa_target50/{PROMPT_SHA}.json \\
  --task-prompt-manifest-sha256 {PROMPT_SHA}"""
    print("\n================ FILLED SUBMIT COMMANDS (each needs the per-run go-ahead) ================")
    for arm in ("s1", "s2", "s3", "q1", "q3"):
        print(f"\n# --- {arm} CANARY (priority 1) ---")
        print(f"{SM_PY} scripts/launch/submit_pi_stage_s.py --arm {arm} --canary --attempt-index 1 \\")
        print(common + " \\\n  --priority 1 --max-run-seconds 21600 --confirm-submit")
        print(f"\n# --- {arm} PRODUCTION (priority 600, after its canary passes) ---")
        print(f"{SM_PY} scripts/launch/submit_pi_stage_s.py --arm {arm} --attempt-index 1 \\")
        print(common + " \\\n  --priority 600 --max-run-seconds 432000 --confirm-submit")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--producer-run", required=True, help="e.g. producer-18425bf72508d09e")
    ap.add_argument("--wsmv2-source-s3", default=None, help="default: newest published wsmv2 archive")
    ap.add_argument("--openpi-source-s3", default=None)
    ap.add_argument("--confirm-publish", action="store_true")
    args = ap.parse_args()

    def newest(component: str) -> str:
        listing = _run(["aws", "s3", "ls", f"{STUDY}/code/{component}/"])
        rows = sorted(line.split() for line in listing.splitlines() if line.strip().endswith(".tgz"))
        if not rows:
            raise SystemExit(f"no published {component} archives")
        return f"{STUDY}/code/{component}/{rows[-1][-1]}"  # newest by timestamp sort

    wsmv2_s3 = args.wsmv2_source_s3 or newest("wsmv2")
    openpi_s3 = args.openpi_source_s3 or newest("openpi")
    print(f"archives: wsmv2={wsmv2_s3.rsplit('/', 1)[-1]} openpi={openpi_s3.rsplit('/', 1)[-1]}")

    uri, sha = locate_manifest(args.producer_run)
    manifest = fetch_and_verify(uri, sha)
    canonical = publish(manifest, sha, args.confirm_publish)
    if not args.confirm_publish:
        print("\nDRY: rerun with --confirm-publish to publish the canonical manifest, then the")
        print("printed submit commands become valid.")
    print_submits(args.producer_run, manifest, sha, canonical, wsmv2_s3, openpi_s3)


if __name__ == "__main__":
    main()
