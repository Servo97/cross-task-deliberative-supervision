#!/usr/bin/env python3
"""Run a resumable sequence of sealed single-task RoboMME fixed-50 evaluations locally."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path

from robomme_integration.fleet.checkpoint import verify as verify_checkpoint_manifest
from robomme_integration.training.arms import ARM_IDS, OFFICIAL_RECIPE_LEROBOT_ARM
from robomme_integration.training.single_task import TASK_EPISODES

STUDY_ROOT = "s3://sagemaker-us-west-2-141701954645/sarvesh.patil/wsm_robocasa/studies/long_context_v1"
DEFAULT_ROOT = Path("/home/sarveshp/Research/TRI/robomme_eval")
DEFAULT_SPEC = Path(__file__).with_name("local_ready9_fixed50_v1.json")
CONFIGS = {
    "PickXtimes": "pickxtimes.yaml",
    "StopCube": "stopcube.yaml",
    "ButtonUnmaskSwap": "buttonunmaskswap.yaml",
    "VideoUnmaskSwap": "videounmaskswap.yaml",
    "PickHighlight": "pickhighlight.yaml",
    "VideoRepick": "videorepick.yaml",
    "MoveCube": "movecube.yaml",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _read_spec(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1:
        raise SystemExit("unsupported local eval queue schema")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("local eval queue has no jobs")
    identities = set()
    for job in jobs:
        task, arm = job.get("task"), job.get("arm")
        identity = (task, arm)
        if task not in CONFIGS or task not in TASK_EPISODES:
            raise SystemExit(f"unsupported local fixed-50 task {task!r}")
        if arm not in ARM_IDS or arm in {"q0v", "q2v", OFFICIAL_RECIPE_LEROBOT_ARM}:
            raise SystemExit(f"unsupported local fixed-50 arm {arm!r}")
        if identity in identities:
            raise SystemExit(f"duplicate local eval identity {identity}")
        identities.add(identity)
        if not re.fullmatch(r"st-v1-[a-z0-9_-]+", str(job.get("run_id", ""))):
            raise SystemExit(f"invalid run id for {identity}")
        if not HEX64.fullmatch(str(job.get("manifest_sha256", ""))):
            raise SystemExit(f"invalid manifest digest for {identity}")
        expected_tail = f"/{task}/{arm}/seed0/{job['run_id']}"
        if not str(job.get("output", "")).endswith(expected_tail):
            raise SystemExit(f"checkpoint output does not match {identity}")
    return jobs


def _paths(root: Path, source_root: Path, job: dict) -> dict[str, Path | str]:
    eval_id = f"{job['run_id']}-fixed50-local5090-v1"
    output = root / "results" / "single_task_v1" / job["task"] / job["arm"] / eval_id
    work = root / "work" / "single_task_fixed50_queue_v1"
    return {
        "eval_id": eval_id,
        "output": output,
        "checkpoint": work / "checkpoint" / job["run_id"] / "19999",
        "claim": work / "claims" / f"{job['run_id']}.json",
        "tree": work / "claims" / f"{job['run_id']}.tree.json",
        "config": source_root / "robomme_integration" / "eval" / "configs" / CONFIGS[job["task"]],
    }


def _single_site(root: Path) -> Path:
    values = list(root.glob("python*/site-packages"))
    if len(values) != 1:
        raise SystemExit(f"expected one Python site-packages under {root}, found {values}")
    return values[0]


def _runtime(root: Path) -> dict[str, Path]:
    policy = root / "openpi" / "ed923b2c" / ".venv" / "bin" / "python"
    runtime = root / "runtime-v0.4.0"
    vla_eval = runtime / "env-v0.4.0" / "bin" / "vla-eval"
    values = {
        "policy": policy,
        "vla_eval": vla_eval,
        "robomme_src": runtime / "robomme-v0.4.0" / "src",
        "policy_site": _single_site(policy.parent.parent / "lib"),
        "sim_site": _single_site(runtime / "env-v0.4.0" / "lib"),
    }
    missing = [str(path) for path in values.values() if not path.exists()]
    if missing:
        raise SystemExit(f"local RoboMME runtime is incomplete: {missing}")
    return values


def _s3_parts(uri: str) -> tuple[str, str]:
    match = re.fullmatch(r"s3://([^/]+)/(.+)", uri)
    if not match:
        raise ValueError(f"invalid S3 URI {uri!r}")
    return match.group(1), match.group(2)


def _publish_once(local: Path, uri: str) -> None:
    bucket, key = _s3_parts(uri)
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            str(local),
            "--if-none-match",
            "*",
            "--region",
            "us-west-2",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return
    existing = local.with_suffix(local.suffix + ".existing")
    try:
        _run(["aws", "s3", "cp", uri, str(existing), "--only-show-errors", "--region", "us-west-2"])
        if local.read_bytes() != existing.read_bytes():
            raise RuntimeError(f"immutable S3 collision at {uri}")
    finally:
        existing.unlink(missing_ok=True)


def _gpu_idle() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"local GPUs already have compute processes: {result.stdout.strip()}")


def _ports_free(base_port: int) -> None:
    for port in (base_port, base_port + 1):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"local policy port {port} is occupied") from error


def _stage_checkpoint(root: Path, source_root: Path, job: dict, paths: dict) -> dict:
    claim_uri = f"{STUDY_ROOT}/manifests/claims/train/{job['run_id']}/step-19999.complete.json"
    claim_path, tree_path = Path(paths["claim"]), Path(paths["tree"])
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    _run(["aws", "s3", "cp", claim_uri, str(claim_path), "--only-show-errors", "--region", "us-west-2"])
    claim = json.loads(claim_path.read_text())
    expected = {
        "run_id": job["run_id"],
        "step": 19999,
        "checkpoint_uri": f"{job['output']}/deploy/19999",
        "run_manifest_sha256": job["manifest_sha256"],
    }
    for key, value in expected.items():
        if claim.get(key) != value:
            raise RuntimeError(f"training completion claim mismatch {key}: {claim.get(key)!r} != {value!r}")
    tree_sha = claim.get("tree_manifest_sha256")
    tree_uri = claim.get("tree_manifest_uri")
    if not isinstance(tree_sha, str) or not HEX64.fullmatch(tree_sha):
        raise RuntimeError("training completion claim has no tree digest")
    if not isinstance(tree_uri, str) or not tree_uri.endswith(f"/{tree_sha}.json"):
        raise RuntimeError("training completion claim has an invalid tree URI")

    checkpoint = Path(paths["checkpoint"])
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    checkpoint.mkdir(parents=True)
    _run(
        [
            "aws",
            "s3",
            "sync",
            claim["checkpoint_uri"],
            str(checkpoint),
            "--only-show-errors",
            "--exclude",
            "*",
            "--include",
            "params/*",
            "--include",
            "assets/*",
            "--region",
            "us-west-2",
        ]
    )
    _run(["aws", "s3", "cp", tree_uri, str(tree_path), "--only-show-errors", "--region", "us-west-2"])
    if _sha256(tree_path) != tree_sha:
        raise RuntimeError("downloaded checkpoint tree manifest has the wrong digest")
    verified = verify_checkpoint_manifest(checkpoint, tree_path, expected_uri=claim["checkpoint_uri"])
    if verified != tree_sha:
        raise RuntimeError("downloaded checkpoint tree does not match its completion claim")
    return claim


def _result_claim(output: Path, job: dict, eval_id: str, checkpoint_uri: str) -> Path:
    supervisor = json.loads((output / "supervisor.json").read_text())
    launch = json.loads((output / "eval" / "launch_manifest.json").read_text())
    if supervisor.get("launcher_returncode") != 0 or supervisor.get("failure") is not None:
        raise RuntimeError("policy/evaluation supervisor did not complete cleanly")
    if any(launch.get("returncodes", [])) or launch.get("episode_audit", {}).get("harness_failures"):
        raise RuntimeError("fixed-50 has a simulator or harness failure")
    candidates = []
    for record in launch.get("materialized_results", []):
        path = Path(record["path"])
        if path.suffix != ".json":
            continue
        value = json.loads(path.read_text())
        episodes = [episode for task in value.get("tasks", []) for episode in task.get("episodes", [])]
        if episodes:
            candidates.append((path, value, episodes))
    exact = [item for item in candidates if len(item[2]) == 50]
    if len(exact) != 1 or launch.get("episode_audit", {}).get("episodes") != 50:
        raise RuntimeError(f"fixed-50 materialization is ambiguous: {[(str(p), len(e)) for p, _, e in candidates]}")
    _path, aggregate, episodes = exact[0]
    tasks = aggregate.get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("task") != job["task"]:
        raise RuntimeError("fixed-50 aggregate contains the wrong task")
    indices = sorted(episode.get("episode_idx") for episode in episodes)
    if indices != list(range(50)):
        raise RuntimeError("fixed-50 aggregate does not contain episode indices 0..49")
    outcomes = [episode.get("metrics", {}).get("success") for episode in episodes]
    if any(not isinstance(value, bool) for value in outcomes):
        raise RuntimeError("fixed-50 aggregate has a non-boolean outcome")
    claim = {
        "schema_version": 2,
        "kind": "robomme_fixed50_complete",
        "training_scope": "single_task",
        "run_id": job["run_id"],
        "eval_id": eval_id,
        "task": job["task"],
        "arm": job["arm"],
        "episodes": 50,
        "successes": sum(outcomes),
        "checkpoint_uri": checkpoint_uri,
        "finished_utc": _utc(),
    }
    path = output / "result-claim.json"
    _write_json(path, claim)
    return path


def _archive_evidence(root: Path, output: Path, claim_path: Path, eval_id: str) -> tuple[Path, str]:
    staging = root / "work" / "single_task_fixed50_queue_v1" / f"evidence-{eval_id}"
    archive = staging.with_suffix(".tgz")
    if staging.exists():
        shutil.rmtree(staging)
    archive.unlink(missing_ok=True)
    required = [claim_path, output / "supervisor.json", output / "eval" / "launch_manifest.json"]
    launch = json.loads(required[-1].read_text())
    selected = [*required, *(Path(record["path"]) for record in launch.get("materialized_results", []))]
    selected.extend(sorted(output.glob("server-*.log")))
    selected.extend(sorted((output / "eval" / "logs").glob("*.log")))
    for source in selected:
        source = source.resolve()
        if not source.is_file() or not source.is_relative_to(output.resolve()):
            raise RuntimeError(f"unsafe or missing evidence file {source}")
        target = staging / source.relative_to(output.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _run(
        [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-czf",
            str(archive),
            "-C",
            str(staging.parent),
            staging.name,
        ]
    )
    return archive, _sha256(archive)


def _launch(root: Path, source_root: Path, runtime: dict[str, Path], job: dict, paths: dict, base_port: int) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                str(path) for path in (runtime["robomme_src"], runtime["policy_site"], runtime["sim_site"])
            ),
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    command = [
        str(runtime["policy"]),
        "-m",
        "robomme_integration.eval.launch_gpu_fleet",
        "--source-root",
        str(source_root),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--arm",
        job["arm"],
        "--task",
        job["task"],
        "--benchmark-config",
        str(paths["config"]),
        "--vla-eval",
        str(runtime["vla_eval"]),
        "--output-root",
        str(paths["output"]),
        "--eval-id",
        str(paths["eval_id"]),
        "--gpus",
        "0,1",
        "--base-port",
        str(base_port),
        "--shards",
        "8",
        "--cpu-range",
        "0-127",
        "--xla-memory-fraction",
        "0.65",
        "--native-simulator",
        "--simulator-pythonpath",
        str(source_root),
        "--simulator-gpus",
        "0,1",
        "--pin-native-cpus",
    ]
    _run(command, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=18100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root, source_root = args.root.resolve(), args.source_root.resolve()
    jobs = _read_spec(args.spec.resolve())
    runtime = _runtime(root)
    plans = [_paths(root, source_root, job) for job in jobs]
    for plan in plans:
        if not Path(plan["config"]).is_file():
            raise SystemExit(f"missing task config {plan['config']}")
    payload = {
        "schema_version": 1,
        "jobs": [
            {**{key: job[key] for key in ("task", "arm", "run_id")}, **{k: str(v) for k, v in plan.items()}}
            for job, plan in zip(jobs, plans, strict=True)
        ],
        "source_root": str(source_root),
        "root": str(root),
        "base_port": args.base_port,
        "topology": {"policy_gpus": [0, 1], "simulator_shards": 8, "cpu_range": "0-127"},
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    _gpu_idle()
    _ports_free(args.base_port)
    state_path = root / "results" / "single_task_v1" / "QUEUE_STATE.json"
    state = {"schema_version": 1, "started_utc": _utc(), "jobs": [], "source_root": str(source_root)}
    _write_json(state_path, state)
    for index, (job, paths) in enumerate(zip(jobs, plans, strict=True), 1):
        output = Path(paths["output"])
        if (output / "COMPLETED").is_file() and (output / "result-claim.json").is_file():
            print(f"SKIP completed {index}/{len(jobs)} {job['task']} {job['arm']}", flush=True)
            state["jobs"].append({"task": job["task"], "arm": job["arm"], "status": "skipped_complete"})
            _write_json(state_path, state)
            continue
        if output.exists():
            raise RuntimeError(f"refusing partial/noncanonical output {output}")
        checkpoint = Path(paths["checkpoint"])
        record = {"task": job["task"], "arm": job["arm"], "run_id": job["run_id"], "started_utc": _utc()}
        state["jobs"].append(record)
        _write_json(state_path, state)
        try:
            print(f"START {index}/{len(jobs)} task={job['task']} arm={job['arm']}", flush=True)
            training_claim = _stage_checkpoint(root, source_root, job, paths)
            _ports_free(args.base_port)
            _launch(root, source_root, runtime, job, paths, args.base_port)
            result = _result_claim(output, job, str(paths["eval_id"]), training_claim["checkpoint_uri"])
            archive, archive_sha = _archive_evidence(root, output, result, str(paths["eval_id"]))
            evidence_uri = (
                f"{STUDY_ROOT}/results/robomme/pi05/single_task_v1/{job['task']}/{job['arm']}/"
                f"{job['run_id']}/{paths['eval_id']}/{archive_sha}.tgz"
            )
            _publish_once(archive, evidence_uri)
            value = json.loads(result.read_text())
            value.update(evidence_archive_sha256=archive_sha, evidence_archive_uri=evidence_uri)
            _write_json(result, value)
            result_uri = f"{STUDY_ROOT}/manifests/claims/eval/{job['run_id']}/{paths['eval_id']}.complete.json"
            _publish_once(result, result_uri)
            record.update(
                status="complete",
                finished_utc=_utc(),
                successes=value["successes"],
                episodes=50,
                result_claim_uri=result_uri,
                evidence_archive_uri=evidence_uri,
            )
            print(f"COMPLETE {index}/{len(jobs)} {job['task']} {job['arm']} {value['successes']}/50", flush=True)
        except BaseException as error:
            record.update(status="failed", finished_utc=_utc(), error=f"{type(error).__name__}: {error}")
            _write_json(state_path, state)
            raise
        finally:
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
            evidence = root / "work" / "single_task_fixed50_queue_v1" / f"evidence-{paths['eval_id']}"
            if evidence.exists():
                shutil.rmtree(evidence)
            evidence.with_suffix(".tgz").unlink(missing_ok=True)
        _write_json(state_path, state)
    state["finished_utc"] = _utc()
    state["status"] = "complete"
    _write_json(state_path, state)
    (state_path.parent / "QUEUE_COMPLETED").write_text(state["finished_utc"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
