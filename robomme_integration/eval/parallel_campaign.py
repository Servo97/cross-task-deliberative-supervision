#!/usr/bin/env python3
"""Resource-sealed parallel scheduling for independent fixed-50 campaign cells.

Each lane runs the existing :class:`campaign.CampaignRunner` per-cell transaction.  The layer
changes only when independent cells run and which disjoint GPU/port/CPU resources execute them;
the original queue identity, 50 episode IDs, benchmark config, fixed-50 audit, evidence archive,
and per-cell result/failure claims remain authoritative and are never merged across lanes.

Host GPU/port leases prevent duplicate local supervisors, but they are not a distributed lock.
Operations must maintain one active supervisor per sealed queue identity across hosts; resume
rejects any cell for which conflicting immutable success and failure claims already exist.
"""

from __future__ import annotations

import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable

from robomme_integration.eval import campaign

PARALLEL_TOPOLOGY_SCHEMA = 1
PARALLEL_EXECUTION_MODE = "parallel_fixed50_lanes_v1"
LANE_EXECUTION_MODE = "parallel_fixed50_lane_v1"
GIB = 1024**3
MIB = 1024**2
SYSTEMIC_FAILURE_CLASSES = frozenset({"control_plane_or_identity", "gpu_resource_exhausted", "resource_admission"})
_LANE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
HOST_LEASE_ROOT = Path("/tmp/robomme-fixed50-resource-leases")
LANE_PORT_RELEASE_TIMEOUT_SECONDS = 30.0
LANE_PORT_RELEASE_POLL_SECONDS = 0.25


class ResourceAdmissionError(ValueError):
    """The sealed lane resources cannot safely fit on the current host."""


class ParallelCampaignSignal(BaseException):
    """SIGTERM converted into cooperative lane cancellation on the main thread."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"parallel campaign received signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class GpuState:
    index: int
    name: str
    uuid: str
    total_bytes: int
    free_bytes: int
    utilization_percent: int
    compute_pids: tuple[int, ...] = ()

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "uuid": self.uuid,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "utilization_percent": self.utilization_percent,
            "compute_pids": list(self.compute_pids),
        }


@dataclass(frozen=True)
class LaneSpec:
    lane_id: str
    policy_gpu: int
    simulator_gpu: int
    port: int
    cpu_range: str
    simulator_shards: int
    xla_memory_fraction: float
    shard_prewarm_seconds: float
    shard_stagger_seconds: float
    gpu_name_contains: str
    policy_reservation_bytes: int
    simulator_reservation_bytes: int
    gpu_headroom_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.lane_id, str) or not _LANE_ID.fullmatch(self.lane_id):
            raise ValueError(f"invalid parallel lane id: {self.lane_id!r}")
        if (
            not isinstance(self.policy_gpu, int)
            or isinstance(self.policy_gpu, bool)
            or self.policy_gpu < 0
            or not isinstance(self.simulator_gpu, int)
            or isinstance(self.simulator_gpu, bool)
            or self.simulator_gpu != self.policy_gpu
        ):
            raise ValueError("each fixed50 lane must place policy and native simulators on one GPU")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1024 <= self.port <= 65535:
            raise ValueError(f"invalid lane port: {self.port}")
        _cpu_set(self.cpu_range)
        if self.simulator_shards != 4:
            raise ValueError("parallel fixed50 lanes require exactly four native simulator shards")
        if not 0.1 <= self.xla_memory_fraction <= 0.95:
            raise ValueError("lane XLA memory fraction must lie in [0.1, 0.95]")
        if not 0.0 <= self.shard_prewarm_seconds <= 600.0:
            raise ValueError("lane shard prewarm must lie in [0, 600] seconds")
        if not 0.0 <= self.shard_stagger_seconds <= 600.0:
            raise ValueError("lane shard stagger must lie in [0, 600] seconds")
        if not self.gpu_name_contains:
            raise ValueError("lane GPU name admission fragment is empty")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (
                self.policy_reservation_bytes,
                self.simulator_reservation_bytes,
                self.gpu_headroom_bytes,
            )
        ):
            raise ValueError("lane GPU memory reservations must be positive integer bytes")

    @property
    def required_free_gpu_bytes(self) -> int:
        return (
            self.policy_reservation_bytes
            + self.simulator_shards * self.simulator_reservation_bytes
            + self.gpu_headroom_bytes
        )

    def required_free_gpu_bytes_for(self, total_bytes: int) -> int:
        """Include the allocator's sealed preallocation, not only a nominal policy estimate."""
        xla_reservation = math.ceil(self.xla_memory_fraction * total_bytes)
        return (
            max(self.policy_reservation_bytes, xla_reservation)
            + self.simulator_shards * self.simulator_reservation_bytes
            + self.gpu_headroom_bytes
        )

    def as_dict(self) -> dict:
        return {
            "lane_id": self.lane_id,
            "policy_gpu": self.policy_gpu,
            "simulator_gpu": self.simulator_gpu,
            "port": self.port,
            "cpu_range": self.cpu_range,
            "simulator_shards": self.simulator_shards,
            "xla_memory_fraction": self.xla_memory_fraction,
            "shard_prewarm_seconds": self.shard_prewarm_seconds,
            "shard_stagger_seconds": self.shard_stagger_seconds,
            "gpu_name_contains": self.gpu_name_contains,
            "policy_reservation_bytes": self.policy_reservation_bytes,
            "simulator_reservation_bytes": self.simulator_reservation_bytes,
            "gpu_headroom_bytes": self.gpu_headroom_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> LaneSpec:
        expected = {
            "lane_id",
            "policy_gpu",
            "simulator_gpu",
            "port",
            "cpu_range",
            "simulator_shards",
            "xla_memory_fraction",
            "shard_prewarm_seconds",
            "shard_stagger_seconds",
            "gpu_name_contains",
            "policy_reservation_bytes",
            "simulator_reservation_bytes",
            "gpu_headroom_bytes",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("parallel lane schema drift")
        return cls(**value)

    def launch_topology(self) -> dict:
        return {
            "execution_mode": LANE_EXECUTION_MODE,
            "lane_id": self.lane_id,
            "policy_gpus": [self.policy_gpu],
            "simulator_gpus": [self.simulator_gpu],
            "simulator_shards": self.simulator_shards,
            "cpu_range": self.cpu_range,
            "base_port": self.port,
            "xla_memory_fraction": self.xla_memory_fraction,
            "native_shard_prewarm_seconds": self.shard_prewarm_seconds,
            "native_shard_stagger_seconds": self.shard_stagger_seconds,
        }


@dataclass(frozen=True)
class ParallelTopology:
    topology_id: str
    lanes: tuple[LaneSpec, ...]
    minimum_free_disk_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.topology_id, str) or not _LANE_ID.fullmatch(self.topology_id):
            raise ValueError(f"invalid parallel topology id: {self.topology_id!r}")
        if not 1 <= len(self.lanes) <= 8:
            raise ValueError("parallel topology must contain between one and eight lanes")
        if len({lane.lane_id for lane in self.lanes}) != len(self.lanes):
            raise ValueError("parallel lane ids must be unique")
        if len({lane.policy_gpu for lane in self.lanes}) != len(self.lanes):
            raise ValueError("parallel lanes must use disjoint physical GPUs")
        if len({lane.port for lane in self.lanes}) != len(self.lanes):
            raise ValueError("parallel lanes must use disjoint policy ports")
        cpu_sets = [_cpu_set(lane.cpu_range) for lane in self.lanes]
        for index, cpus in enumerate(cpu_sets):
            if any(cpus & other for other in cpu_sets[:index]):
                raise ValueError("parallel lane CPU ranges overlap")
        if not isinstance(self.minimum_free_disk_bytes, int) or self.minimum_free_disk_bytes < 1:
            raise ValueError("parallel topology disk admission floor must be positive bytes")

    def _unsealed_queue_topology(self) -> dict:
        cpu_union = set().union(*(_cpu_set(lane.cpu_range) for lane in self.lanes))
        if cpu_union != set(range(min(cpu_union), max(cpu_union) + 1)):
            raise ValueError("parallel lane CPU ranges must form one contiguous aggregate range")
        return {
            "schema_version": PARALLEL_TOPOLOGY_SCHEMA,
            "execution_mode": PARALLEL_EXECUTION_MODE,
            "topology_id": self.topology_id,
            "policy_gpus": [lane.policy_gpu for lane in self.lanes],
            "simulator_gpus": [lane.simulator_gpu for lane in self.lanes],
            "simulator_shards": sum(lane.simulator_shards for lane in self.lanes),
            "cpu_range": f"{min(cpu_union)}-{max(cpu_union)}",
            "base_port": min(lane.port for lane in self.lanes),
            "xla_memory_fraction": max(lane.xla_memory_fraction for lane in self.lanes),
            "minimum_free_disk_bytes": self.minimum_free_disk_bytes,
            "lanes": [lane.as_dict() for lane in self.lanes],
        }

    def as_queue_topology(self) -> dict:
        value = self._unsealed_queue_topology()
        value["parallel_topology_sha256"] = _topology_digest(value)
        return value

    @classmethod
    def from_queue_topology(cls, value: object) -> ParallelTopology:
        if not isinstance(value, dict):
            raise ValueError("parallel queue topology must be an object")
        lanes = value.get("lanes")
        if not isinstance(lanes, list):
            raise ValueError("parallel queue topology has no lane list")
        topology = cls(
            topology_id=value.get("topology_id"),
            lanes=tuple(LaneSpec.from_dict(lane) for lane in lanes),
            minimum_free_disk_bytes=value.get("minimum_free_disk_bytes"),
        )
        if value != topology.as_queue_topology():
            raise ValueError("parallel queue topology aggregate or self-seal drift")
        return topology


def _cpu_set(value: str) -> set[int]:
    if not isinstance(value, str) or not re.fullmatch(r"\d+-\d+", value):
        raise ValueError(f"invalid inclusive CPU range: {value!r}")
    start_text, end_text = value.split("-", 1)
    start, end = int(start_text), int(end_text)
    if start < 0 or end < start:
        raise ValueError(f"invalid inclusive CPU range: {value!r}")
    return set(range(start, end + 1))


def _topology_digest(value: dict) -> str:
    clean = dict(value)
    clean.pop("parallel_topology_sha256", None)
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def local_2x5090_topology() -> ParallelTopology:
    """Two concurrent cells: one policy plus four staggered native shards on each RTX 5090."""
    lanes = tuple(
        LaneSpec(
            lane_id=f"local5090-gpu{gpu}",
            policy_gpu=gpu,
            simulator_gpu=gpu,
            port=18100 + gpu,
            cpu_range="0-63" if gpu == 0 else "64-127",
            simulator_shards=4,
            xla_memory_fraction=0.55,
            shard_prewarm_seconds=180.0,
            shard_stagger_seconds=30.0,
            gpu_name_contains="RTX 5090",
            policy_reservation_bytes=27 * GIB,
            simulator_reservation_bytes=1 * GIB,
            gpu_headroom_bytes=128 * MIB,
        )
        for gpu in range(2)
    )
    return ParallelTopology(
        topology_id="local-2x5090-fixed50-v1",
        lanes=lanes,
        minimum_free_disk_bytes=48 * GIB,
    )


def p5_8xh100_topology() -> ParallelTopology:
    """Eight concurrent cells: one H100 policy and four native shards per disjoint lane."""
    lanes = tuple(
        LaneSpec(
            lane_id=f"p5-h100-gpu{gpu}",
            policy_gpu=gpu,
            simulator_gpu=gpu,
            port=18100 + gpu,
            cpu_range=f"{gpu * 24}-{gpu * 24 + 23}",
            simulator_shards=4,
            xla_memory_fraction=0.65,
            shard_prewarm_seconds=180.0,
            shard_stagger_seconds=15.0,
            gpu_name_contains="H100",
            policy_reservation_bytes=52 * GIB,
            simulator_reservation_bytes=1536 * MIB,
            gpu_headroom_bytes=8 * GIB,
        )
        for gpu in range(8)
    )
    return ParallelTopology(
        topology_id="p5-8xh100-fixed50-v1",
        lanes=lanes,
        minimum_free_disk_bytes=128 * GIB,
    )


def with_parallel_topology(unsealed_queue: dict, topology: ParallelTopology) -> dict:
    """Bind lane resources before the caller seals a fresh queue identity."""
    if "queue_manifest_sha256" in unsealed_queue:
        raise ValueError("parallel topology must be attached before the queue is sealed")
    return {**unsealed_queue, "topology": topology.as_queue_topology()}


def query_gpu_states() -> dict[int, GpuState]:
    try:
        inventory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ResourceAdmissionError(f"GPU inventory probe failed: {error}") from error
    rows: dict[int, dict] = {}
    for raw in csv.reader(inventory.stdout.splitlines(), skipinitialspace=True):
        if len(raw) != 6:
            raise ResourceAdmissionError(f"malformed nvidia-smi GPU inventory row: {raw}")
        index = int(raw[0])
        rows[index] = {
            "index": index,
            "name": raw[1].strip(),
            "uuid": raw[2].strip(),
            "total_bytes": int(raw[3]) * MIB,
            "free_bytes": int(raw[4]) * MIB,
            "utilization_percent": int(raw[5]),
            "compute_pids": [],
        }
    try:
        applications = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ResourceAdmissionError(f"GPU process probe failed: {error}") from error
    uuid_to_index = {row["uuid"]: index for index, row in rows.items()}
    for raw in csv.reader(applications.stdout.splitlines(), skipinitialspace=True):
        if not raw:
            continue
        if len(raw) != 2 or raw[0].strip() not in uuid_to_index:
            raise ResourceAdmissionError(f"malformed nvidia-smi compute-app row: {raw}")
        rows[uuid_to_index[raw[0].strip()]]["compute_pids"].append(int(raw[1]))
    return {
        index: GpuState(**{**row, "compute_pids": tuple(sorted(row["compute_pids"]))}) for index, row in rows.items()
    }


@dataclass
class HostResourceLeases:
    """Process-lifetime advisory locks for the exact GPU UUIDs and policy ports in use."""

    streams: list[IO[str]]
    keys: tuple[str, ...]

    def close(self) -> None:
        while self.streams:
            stream = self.streams.pop()
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()


def acquire_host_resource_leases(
    topology: ParallelTopology,
    lanes: tuple[LaneSpec, ...],
    *,
    lease_root: Path = HOST_LEASE_ROOT,
    gpu_states: dict[int, GpuState] | None = None,
) -> HostResourceLeases:
    """Atomically reserve cooperating GPU/port lanes, before re-probing under the locks."""
    if not lanes:
        return HostResourceLeases([], ())
    states = query_gpu_states() if gpu_states is None else gpu_states
    lane_ids = {lane.lane_id for lane in topology.lanes}
    if any(lane.lane_id not in lane_ids for lane in lanes):
        raise ResourceAdmissionError("resource lease requested a lane outside the sealed topology")
    keys: set[str] = set()
    for lane in lanes:
        state = states.get(lane.policy_gpu)
        if state is None:
            raise ResourceAdmissionError(f"parallel lane GPU {lane.policy_gpu} is absent")
        uuid_digest = hashlib.sha256(state.uuid.encode("utf-8")).hexdigest()[:24]
        keys.add(f"gpu-{uuid_digest}")
        keys.add(f"port-{lane.port}")
    lease_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    streams: list[IO[str]] = []
    ordered = tuple(sorted(keys))
    try:
        for key in ordered:
            path = lease_root / f"{key}.lock"
            stream = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                stream.close()
                raise ResourceAdmissionError(f"parallel campaign resource lease is already held: {key}") from error
            streams.append(stream)
    except BaseException:
        HostResourceLeases(streams, ordered).close()
        raise
    return HostResourceLeases(streams, ordered)


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # Match the policy server's reusable listener semantics.  Without this, a prior
        # connection in TIME_WAIT can look like an owner even though the next server can bind.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _port_has_listener(port: int) -> bool:
    """Return true only for a live TCP listener, not a draining prior connection."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def wait_for_lane_port_release(
    lane: LaneSpec,
    *,
    timeout_seconds: float = LANE_PORT_RELEASE_TIMEOUT_SECONDS,
    poll_seconds: float = LANE_PORT_RELEASE_POLL_SECONDS,
    port_available: Callable[[int], bool] = _port_available,
    port_has_listener: Callable[[int], bool] = _port_has_listener,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Bound a same-lane handoff while the prior policy socket finishes draining.

    ``launch_gpu_fleet`` returns only after terminating its policy process group, but the
    kernel can briefly reject an immediate bind while the prior TCP connections drain.  A
    listening socket is not that benign transition and fails immediately as an external owner.
    """
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("lane port-release timeout and poll interval must be positive")
    started = monotonic()
    polls = 0
    while not port_available(lane.port):
        if port_has_listener(lane.port):
            raise ResourceAdmissionError(
                f"parallel lane {lane.lane_id} policy port {lane.port} has a live listener "
                "after the prior evaluator exited"
            )
        elapsed = monotonic() - started
        if elapsed >= timeout_seconds:
            raise ResourceAdmissionError(
                f"parallel lane {lane.lane_id} policy port {lane.port} did not release "
                f"within {timeout_seconds:g} seconds after the prior evaluator exited"
            )
        sleep(min(poll_seconds, timeout_seconds - elapsed))
        polls += 1
    return {
        "lane_id": lane.lane_id,
        "port": lane.port,
        "wait_seconds": max(0.0, monotonic() - started),
        "polls": polls,
    }


def _admit_lane_snapshot(
    lane: LaneSpec,
    state: GpuState,
    *,
    allowed_cpus: set[int],
    port_available: Callable[[int], bool],
) -> dict:
    if lane.gpu_name_contains not in state.name:
        raise ResourceAdmissionError(
            f"parallel lane {lane.lane_id} expected {lane.gpu_name_contains!r}, got {state.name!r}"
        )
    if state.compute_pids:
        raise ResourceAdmissionError(f"parallel lane {lane.lane_id} GPU is occupied by PIDs {state.compute_pids}")
    if state.utilization_percent > 5:
        raise ResourceAdmissionError(
            f"parallel lane {lane.lane_id} GPU utilization is not idle: {state.utilization_percent}%"
        )
    required_free_gpu_bytes = lane.required_free_gpu_bytes_for(state.total_bytes)
    if state.total_bytes < required_free_gpu_bytes:
        raise ResourceAdmissionError(f"parallel lane {lane.lane_id} GPU capacity is below its sealed peak budget")
    if state.free_bytes < required_free_gpu_bytes:
        raise ResourceAdmissionError(f"parallel lane {lane.lane_id} free GPU memory is below its sealed peak budget")
    if not _cpu_set(lane.cpu_range) <= allowed_cpus:
        raise ResourceAdmissionError(f"parallel lane {lane.lane_id} CPU range escapes process affinity")
    if not port_available(lane.port):
        raise ResourceAdmissionError(f"parallel lane {lane.lane_id} policy port {lane.port} is occupied")
    return {
        "lane_id": lane.lane_id,
        "required_free_gpu_bytes": required_free_gpu_bytes,
        "xla_allocator_reservation_bytes": math.ceil(lane.xla_memory_fraction * state.total_bytes),
        "gpu": state.as_dict(),
    }


def admit_lane_resources(lane: LaneSpec) -> dict:
    """Re-admit one idle lane immediately before staging its next independent cell."""
    states = query_gpu_states()
    state = states.get(lane.policy_gpu)
    if state is None:
        raise ResourceAdmissionError(f"parallel lane GPU {lane.policy_gpu} is absent")
    return _admit_lane_snapshot(
        lane,
        state,
        allowed_cpus=set(os.sched_getaffinity(0)),
        port_available=_port_available,
    )


def admit_parallel_resources(
    queue: dict,
    topology: ParallelTopology,
    work_root: Path,
    *,
    gpu_states: dict[int, GpuState] | None = None,
    allowed_cpus: set[int] | None = None,
    port_available: Callable[[int], bool] = _port_available,
    lanes: tuple[LaneSpec, ...] | None = None,
) -> dict:
    """Fail before staging unless every sealed lane has exclusive, sufficient host resources."""
    gpu_states = query_gpu_states() if gpu_states is None else gpu_states
    allowed_cpus = set(os.sched_getaffinity(0)) if allowed_cpus is None else set(allowed_cpus)
    disk_floor = max(
        topology.minimum_free_disk_bytes,
        int(queue["limits"]["minimum_free_bytes"]),
    )
    free_disk = shutil.disk_usage(work_root).free
    if free_disk < disk_floor:
        raise ResourceAdmissionError(
            f"parallel campaign disk admission failed: free={free_disk} required={disk_floor}"
        )
    selected_lanes = topology.lanes if lanes is None else lanes
    if any(lane not in topology.lanes for lane in selected_lanes):
        raise ResourceAdmissionError("resource admission selected a lane outside the topology")
    admitted = []
    for lane in selected_lanes:
        state = gpu_states.get(lane.policy_gpu)
        if state is None:
            raise ResourceAdmissionError(f"parallel lane GPU {lane.policy_gpu} is absent")
        admitted.append(
            _admit_lane_snapshot(
                lane,
                state,
                allowed_cpus=allowed_cpus,
                port_available=port_available,
            )
        )
    return {
        "schema_version": 1,
        "parallel_topology_sha256": topology.as_queue_topology()["parallel_topology_sha256"],
        "free_disk_bytes": free_disk,
        "required_free_disk_bytes": disk_floor,
        "lanes": admitted,
    }


EvaluatorFactory = Callable[[LaneSpec, threading.Event], campaign.Evaluator]
Admission = Callable[[dict, ParallelTopology, Path], dict]
LaneAdmission = Callable[[LaneSpec], dict]
LaneReleaseWait = Callable[[LaneSpec], dict]
LeaseFactory = Callable[[ParallelTopology, tuple[LaneSpec, ...]], HostResourceLeases]
DiskFree = Callable[[Path], int]


def dry_run_payload(
    queue: dict,
    source_root: Path,
    runtime: campaign.Runtime,
    work_root: Path,
) -> dict:
    """Render exact lane commands without resource probes, S3 access, or child processes."""
    topology = ParallelTopology.from_queue_topology(queue["topology"])
    cells = []
    for index, cell in enumerate(queue["cells"]):
        lane = topology.lanes[index % len(topology.lanes)]
        cell_root = work_root / "cells" / f"cell-{cell['ordinal']:03d}-{cell['cell_id']}"
        workspace = (
            cell_root / "workspace" / str(cell["workspace"]["step"])
            if cell["arm"] in campaign.WORKSPACE_EVAL_ARMS
            else None
        )
        launch_queue = {**queue, "topology": lane.launch_topology()}
        cells.append(
            {
                "cell_id": cell["cell_id"],
                "task": cell["task"],
                "arm": cell["arm"],
                "lane_id": lane.lane_id,
                "max_attempts": queue["retry"]["max_attempts"],
                "launch_command": campaign.build_launch_command(
                    launch_queue,
                    cell,
                    source_root=source_root,
                    runtime=runtime,
                    checkpoint=cell_root / "checkpoint" / str(campaign.checkpoint_step(cell)),
                    workspace=workspace,
                    output=cell_root / "attempts/attempt-1/output",
                ),
            }
        )
    return {
        "schema_version": 1,
        "dry_run": True,
        "execution_mode": PARALLEL_EXECUTION_MODE,
        "queue_id": queue["queue_id"],
        "queue_manifest_sha256": queue["queue_manifest_sha256"],
        "native_preflight_claim_sha256": runtime.preflight_claim_sha256,
        "runtime_receipt_sha256": runtime.receipt_sha256,
        "parallel_topology": topology.as_queue_topology(),
        "cells": cells,
        "note": "no S3 access, resource admission, simulator, or policy process was started",
    }


@dataclass
class ParallelCampaignRunner:
    queue: dict
    source_root: Path
    work_root: Path
    runtime: campaign.Runtime
    store: campaign.ObjectStore
    stager: campaign.Stager
    artifacts: campaign.Artifacts
    evaluator_factory: EvaluatorFactory | None = None
    resource_admission: Admission | None = None
    lane_admission: LaneAdmission | None = None
    lane_release_wait: LaneReleaseWait | None = None
    lease_factory: LeaseFactory | None = None
    disk_free: DiskFree | None = None
    _transaction: campaign.CampaignRunner = field(init=False)
    _topology: ParallelTopology = field(init=False)
    _cancel: threading.Event = field(init=False, default_factory=threading.Event)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _records: dict[str, dict] = field(init=False, default_factory=dict)
    _resume_records: dict[str, dict | None] = field(init=False, default_factory=dict)
    _systemic_cell: str | None = field(init=False, default=None)
    _resource_blocked: bool = field(init=False, default=False)
    _disk_blocked: bool = field(init=False, default=False)
    _deadline_exhausted: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.source_root = self.source_root.resolve()
        self.work_root = self.work_root.resolve()
        campaign.validate_queue(self.queue, source_root=self.source_root)
        self._topology = ParallelTopology.from_queue_topology(self.queue["topology"])
        if isinstance(self.stager, campaign.AwsStager):
            if self.stager.cancel_event not in (None, self._cancel):
                raise ValueError("AWS stager is already bound to a different cancellation event")
            self.stager.cancel_event = self._cancel
            if isinstance(self.stager.store, campaign.AwsCliStore):
                if self.stager.store.cancel_event not in (None, self._cancel):
                    raise ValueError("AWS staging object store is bound to a different cancellation event")
                # Object publication is a commit point and uses ``self.store`` without cooperative
                # cancellation.  Staging gets a separate read-side client so identity reads and
                # downloads stop promptly without aborting an already-started immutable commit.
                self.stager.store = campaign.AwsCliStore(
                    region=self.stager.store.region,
                    cancel_event=self._cancel,
                    cancel_poll_seconds=self.stager.store.cancel_poll_seconds,
                )
        placeholder = campaign.SubprocessEvaluator(self.source_root, self.runtime)
        self._transaction = campaign.CampaignRunner(
            queue=self.queue,
            source_root=self.source_root,
            work_root=self.work_root,
            runtime=self.runtime,
            store=self.store,
            stager=self.stager,
            evaluator=placeholder,
            artifacts=self.artifacts,
        )

    def _new_evaluator(self, lane: LaneSpec) -> campaign.Evaluator:
        if self.evaluator_factory is not None:
            return self.evaluator_factory(lane, self._cancel)
        return campaign.SubprocessEvaluator(
            self.source_root,
            self.runtime,
            topology_override=lane.launch_topology(),
            cancel_event=self._cancel,
        )

    def _ordered_records(self) -> list[dict]:
        return [self._records[cell["cell_id"]] for cell in self.queue["cells"] if cell["cell_id"] in self._records]

    def _write_parallel_state(self, status: str, admission: dict | None = None) -> None:
        path = self.work_root / "state" / "parallel.json"
        current = json.loads(path.read_text()) if path.is_file() else {}
        campaign._atomic_json(
            path,
            {
                "schema_version": 1,
                "queue_id": self.queue["queue_id"],
                "queue_manifest_sha256": self.queue["queue_manifest_sha256"],
                "parallel_topology": self._topology.as_queue_topology(),
                "lane_cells": {
                    lane.lane_id: [
                        cell["cell_id"]
                        for index, cell in enumerate(self.queue["cells"])
                        if index % len(self._topology.lanes) == self._topology.lanes.index(lane)
                    ]
                    for lane in self._topology.lanes
                },
                "admission": admission if admission is not None else current.get("admission"),
                "systemic_cell": self._systemic_cell,
                "status": status,
                "updated_utc": campaign._utc(),
            },
        )

    def _record(self, lane: LaneSpec, cell: dict, record: dict, *, status: str) -> None:
        with self._lock:
            self._records[cell["cell_id"]] = record
            self._transaction._write_cell_state(cell, record)
            self._transaction._write_campaign_state(self._ordered_records(), status=status)
            campaign._atomic_json(
                self.work_root / "state" / "lanes" / f"{lane.lane_id}.json",
                {
                    "schema_version": 1,
                    "queue_id": self.queue["queue_id"],
                    "queue_manifest_sha256": self.queue["queue_manifest_sha256"],
                    "lane": lane.as_dict(),
                    "last_cell_id": cell["cell_id"],
                    "last_cell_status": record["status"],
                    "updated_utc": campaign._utc(),
                },
            )

    def _existing_record(self, cell: dict) -> dict | None:
        existing = campaign._verify_existing_result(self.queue, cell, self.store, self.runtime)
        failure = campaign._verify_existing_failure(self.queue, cell, self.store)
        if existing is not None and failure is not None:
            raise ValueError(f"cell {cell['cell_id']} has conflicting immutable success and failure claims")
        if existing is not None:
            return {
                "cell_id": cell["cell_id"],
                "status": "skipped_exact_complete",
                "successes": existing["successes"],
                "result_claim_s3": cell["result_claim_s3"],
                "evidence_archive_uri": existing["evidence_archive_uri"],
            }
        if failure is None:
            return None
        return {
            "cell_id": cell["cell_id"],
            "status": "skipped_terminal_failure",
            "failure_class": failure.get("failure_class", "unclassified"),
            "failure_claim_s3": failure["failure_claim_s3"],
            "evidence_archive_uri": failure["evidence_archive_uri"],
        }

    def _scan_resume_records(self) -> tuple[LaneSpec, ...]:
        self._resume_records = {cell["cell_id"]: self._existing_record(cell) for cell in self.queue["cells"]}
        active: list[LaneSpec] = []
        for cell in self.queue["cells"]:
            record = self._resume_records[cell["cell_id"]]
            if (
                record is not None
                and record["status"] == "skipped_terminal_failure"
                and record.get("failure_class") in SYSTEMIC_FAILURE_CLASSES
            ):
                self._signal_systemic(cell["cell_id"])
                return ()
        lane_count = len(self._topology.lanes)
        for lane_index, lane in enumerate(self._topology.lanes):
            for cell in self.queue["cells"][lane_index::lane_count]:
                record = self._resume_records[cell["cell_id"]]
                if record is None:
                    active.append(lane)
                    break
                if record["status"] == "skipped_terminal_failure":
                    break
        return tuple(active)

    def _signal_systemic(self, cell_id: str) -> bool:
        with self._lock:
            owns_claim = self._systemic_cell is None
            if self._systemic_cell is None:
                self._systemic_cell = cell_id
            self._cancel.set()
            return owns_claim

    def _lane_worker(self, lane_index: int, deadline: float) -> None:
        lane = self._topology.lanes[lane_index]
        evaluator = self._new_evaluator(lane)
        if isinstance(evaluator, campaign.SubprocessEvaluator):
            evaluator.deadline_monotonic = deadline
        cells = self.queue["cells"][lane_index :: len(self._topology.lanes)]
        requires_release_barrier = False
        for cell in cells:
            if self._cancel.is_set():
                return
            record = self._resume_records[cell["cell_id"]]
            if record is not None:
                self._record(lane, cell, record, status="running_parallel")
                if record["status"] in {
                    "terminal_failure",
                    "skipped_terminal_failure",
                }:
                    if record.get("failure_class") in SYSTEMIC_FAILURE_CLASSES:
                        self._signal_systemic(cell["cell_id"])
                    return
                continue
            remaining = deadline - time.monotonic()
            required = self.queue["limits"]["estimated_cell_seconds"] + self.queue["limits"]["runtime_reserve_seconds"]
            if remaining < required:
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "deferred_runtime_budget",
                    "remaining_seconds": max(0, int(remaining)),
                    "required_seconds": required,
                }
                self._record(lane, cell, record, status="running_parallel_deferred_lane")
                return
            minimum_free = max(
                self.queue["limits"]["minimum_free_bytes"],
                self._topology.minimum_free_disk_bytes,
            )
            free_disk = (
                shutil.disk_usage(self.work_root).free if self.disk_free is None else self.disk_free(self.work_root)
            )
            if free_disk < minimum_free:
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "blocked_disk_floor",
                    "minimum_free_bytes": minimum_free,
                }
                self._record(lane, cell, record, status="running_parallel_blocked_lane")
                with self._lock:
                    self._disk_blocked = True
                    self._cancel.set()
                return
            try:
                if requires_release_barrier:
                    (self.lane_release_wait or wait_for_lane_port_release)(lane)
                    requires_release_barrier = False
                (self.lane_admission or admit_lane_resources)(lane)
            except ResourceAdmissionError as error:
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "blocked_resource_admission",
                    "detail": str(error)[:1000],
                }
                self._record(
                    lane,
                    cell,
                    record,
                    status="running_parallel_blocked_resource_lane",
                )
                with self._lock:
                    self._resource_blocked = True
                    self._systemic_cell = cell["cell_id"]
                    self._cancel.set()
                return
            try:
                record = self._transaction.run_cell_transaction(
                    cell,
                    evaluator=evaluator,
                    cancel_event=self._cancel,
                    systemic_callback=lambda _failure_class, cell_id=cell["cell_id"]: (self._signal_systemic(cell_id)),
                )
            except campaign.EvaluatorCancelled:
                self._transaction._write_cell_state(
                    cell,
                    {"status": "cancelled_without_claim", "attempts": []},
                )
                return
            except campaign.EvaluatorDeadlineExceeded:
                record = {
                    "cell_id": cell["cell_id"],
                    "status": "deferred_runtime_budget",
                    "remaining_seconds": 0,
                    "required_seconds": required,
                }
                self._record(
                    lane,
                    cell,
                    record,
                    status="running_parallel_deferred_deadline",
                )
                with self._lock:
                    self._deadline_exhausted = True
                    self._cancel.set()
                return
            self._record(lane, cell, record, status="running_parallel")
            requires_release_barrier = record["status"] == "complete"
            if record["status"] in {"terminal_failure", "skipped_terminal_failure"}:
                if record.get("failure_class") in SYSTEMIC_FAILURE_CLASSES:
                    self._signal_systemic(cell["cell_id"])
                return

    def run(self) -> int:
        previous_sigterm = None
        if threading.current_thread() is threading.main_thread():
            previous_sigterm = signal.signal(
                signal.SIGTERM,
                lambda signum, _frame: (_ for _ in ()).throw(ParallelCampaignSignal(signum)),
            )
        try:
            return self._run()
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)

    def _run(self) -> int:
        expected_manifest = campaign._canonical(self.queue)
        existing_manifest = self.store.read_bytes(self.queue["claims"]["manifest"])
        existing_completion = campaign._json_bytes(
            self.store.read_bytes(self.queue["claims"]["completion"]),
            label=self.queue["claims"]["completion"],
        )
        if existing_completion is not None:
            if existing_manifest != expected_manifest:
                raise ValueError("existing queue completion has no exact canonical manifest object")
            status = campaign._verify_queue_completion(self.queue, existing_completion, self.store, self.runtime)
            return 0 if status == "complete" else 2
        if existing_manifest is not None and existing_manifest != expected_manifest:
            raise ValueError("existing queue manifest object has conflicting bytes")

        active_lanes = self._scan_resume_records()
        leases: HostResourceLeases | None = None
        try:
            if active_lanes:
                lease_fn = self.lease_factory
                if lease_fn is None and self.resource_admission is None:
                    lease_fn = acquire_host_resource_leases
                if lease_fn is not None:
                    leases = lease_fn(self._topology, active_lanes)
                if self.resource_admission is None:
                    admission = admit_parallel_resources(
                        self.queue,
                        self._topology,
                        self.work_root,
                        lanes=active_lanes,
                    )
                else:
                    admission = self.resource_admission(self.queue, self._topology, self.work_root)
            else:
                admission = {
                    "schema_version": 1,
                    "status": "not_required_all_runnable_cells_already_claimed",
                    "parallel_topology_sha256": self._topology.as_queue_topology()["parallel_topology_sha256"],
                    "lanes": [],
                }
        except ResourceAdmissionError as error:
            if leases is not None:
                leases.close()
            self._transaction._write_campaign_state([], status="blocked_resource_admission")
            self._write_parallel_state("blocked_resource_admission")
            print(f"[parallel-eval-campaign] resource admission blocked: {error}", flush=True)
            return 3
        except BaseException:
            if leases is not None:
                leases.close()
            raise

        try:
            return self._run_admitted(admission)
        finally:
            if leases is not None:
                leases.close()

    def _run_admitted(self, admission: dict) -> int:
        self.store.put_bytes_once(campaign._canonical(self.queue), self.queue["claims"]["manifest"])
        self._transaction._write_campaign_state([], status="running_parallel")
        self._write_parallel_state("running_parallel", admission)
        deadline = time.monotonic() + self.queue["limits"]["max_run_seconds"]
        executor: concurrent.futures.ThreadPoolExecutor | None = None
        futures: list[concurrent.futures.Future] = []
        try:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self._topology.lanes),
                thread_name_prefix="robomme-fixed50-lane",
            )
            futures = [
                executor.submit(self._lane_worker, lane_index, deadline)
                for lane_index in range(len(self._topology.lanes))
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            self._cancel.set()
            for future in futures:
                future.cancel()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            self._transaction._write_campaign_state(self._ordered_records(), status="interrupted_parallel")
            self._write_parallel_state("interrupted_parallel")
            raise
        else:
            if executor is not None:
                executor.shutdown(wait=True)

        records = self._ordered_records()
        if self._resource_blocked:
            self._transaction._write_campaign_state(records, status="blocked_resource_admission")
            self._write_parallel_state("blocked_resource_admission")
            return 3
        if self._disk_blocked:
            self._transaction._write_campaign_state(records, status="blocked_disk_floor")
            self._write_parallel_state("blocked_disk_floor")
            return 3
        if self._systemic_cell is not None:
            self._transaction._write_campaign_state(records, status="halted_systemic_failure")
            self._write_parallel_state("halted_systemic_failure")
            return 2
        if any(record["status"] in {"terminal_failure", "skipped_terminal_failure"} for record in records):
            self._transaction._write_campaign_state(records, status="halted_lane_terminal_failure")
            self._write_parallel_state("halted_lane_terminal_failure")
            return 2
        if any(record["status"] == "blocked_disk_floor" for record in records):
            self._transaction._write_campaign_state(records, status="blocked_disk_floor")
            self._write_parallel_state("blocked_disk_floor")
            return 3
        if self._deadline_exhausted:
            self._transaction._write_campaign_state(records, status="deferred_runtime_budget")
            self._write_parallel_state("deferred_runtime_budget")
            return 0
        if len(records) != len(self.queue["cells"]) or any(
            record["status"] == "deferred_runtime_budget" for record in records
        ):
            self._transaction._write_campaign_state(records, status="deferred_runtime_budget")
            self._write_parallel_state("deferred_runtime_budget")
            return 0
        if any(record["status"] not in {"complete", "skipped_exact_complete"} for record in records):
            raise RuntimeError("parallel campaign reached an unrecognized terminal state")
        completion = {
            "schema_version": 1,
            "kind": campaign.QUEUE_COMPLETION_KIND,
            "queue_id": self.queue["queue_id"],
            "queue_manifest_sha256": self.queue["queue_manifest_sha256"],
            "native_preflight_claim_sha256": self.runtime.preflight_claim_sha256,
            "runtime_receipt_sha256": self.runtime.receipt_sha256,
            "status": "complete",
            "records": records,
        }
        self.store.put_bytes_once(campaign._canonical(completion), self.queue["claims"]["completion"])
        self._transaction._write_campaign_state(records, status="complete")
        self._write_parallel_state("complete")
        return 0
