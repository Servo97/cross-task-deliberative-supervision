#!/usr/bin/env python3
"""Serve-side Stage-E ω for the pi0.5 ReMemBench eval lane (the consumer §23.2 said was missing).

WHAT THIS IS. The sealed workspace serve (`serve_pi_05_wsm_cfg.py`) produces ω online with a WSMv1
`WorkspaceModel` (`load_wsm`), which rejects a Stage-E encoder (`backbone_dim 512`, no `decoder.*`).
This module supplies the Stage-E front end for the SAME serve stack and nothing else:

    frozen pi tap (unchanged)  ->  patch tokens [192,2048]
      -> frozen WSMv1 PatchPool (`pi_pooled_tap.load_pool`, the pooler the `wsm_pooled` store used)
      -> fp16 round trip (what `Corpus` stored)  ->  Stage-E domain adapter  ->  shared trunk  -> ω_t

    `WSMEvalConditioner` (sealed) owns the causal prefix, the stride-8 grid and the K-row window;
    `WSMPiInferWrapper` (sealed) owns env identity / reset-at-t=0 / ordering; the policy receives
    `wsm_w_window` [K,512] exactly as for the WSMv1 arms.

THE WINDOW/STRIDE CONTRACT — one rule, both sides. Training selects ω rows with the fork's
`groot_openpi_dataset._wsm_causal_window(frame_indices, t, K)`; serving selects them with
`wsm_align.causal_window_indices(frame_indices, t, K)` inside `WSMEvalConditioner.step_many`. The
fork cannot import wsmv2, so the loader carries a verbatim copy. `assert_window_rule_lockstep`
executes the loader's OWN source (AST-extracted from the fork tree actually on `sys.path`, i.e. the
content-addressed openpi the arm's manifest pins) against `causal_window_indices` over a battery of
grids and refuses to serve on any disagreement. Fail closed, at startup, before a GPU is touched.

CONDITIONING. Stage-E rmb encoders are trained under `--lang-mode serve` = `task_mean`
(train_stage_e.SERVE_CONSISTENT_LANG), and `export_omega_store` writes the vector training actually
used into every `w.npz` as `lang_global`. The serve table is therefore DERIVED FROM THE STORE, one
vector per task, and the derivation refuses a store whose per-task vectors are not identical (an
`episode_mean` store is not serveable — §25.8). `--lang-table-mode task_mean_of_store` exists for
smoke tests on such stores and announces itself; it is never a scored configuration.

PARITY (D7 discipline). `parity_check` replays stored pooled frames (`wsm_pooled/.../p.npz`)
through the REAL serve stack (`WSMEvalConditioner` + this front end, pool bypassed because the input
is already pooled) one grid frame at a time and requires, per frame, cos(online ω, stored ω) ≥ 0.999
and max|Δ| within the fp16 storage floor — the D7 bar — AND that the served K-row window equals the
rows `wsm_align.window_at` (the training-side rule) selects from the store at the same grid time.
Run it standalone (`python -m vla_training.eval.stage_e_serve parity ...`, CPU is fine) or let the
server run it at startup (`--stage-e-parity-demos N`) and refuse to serve on FAIL.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace_models.features.stage_e_omega_producer import (  # noqa: E402
    StageEServeEncoder,
    _as_trained,
    load_stage_e,
)
from workspace_models.features.wsm_align import causal_window_indices, window_at  # noqa: E402
from wsm_settings import ROBOCASA_OPENPI_SRC  # noqa: E402

#: The stride of the Stage-E ω store grid (`pi_pooled_tap.STRIDE`): frame_indices = 0, 8, 16, ...
STORE_STRIDE = 8
#: D7 bar (`stage_e_omega_parity.COS_BAR`), not to be tuned against.
COS_BAR = 0.999
LANG_TABLE_MODES = ("strict", "task_mean_of_store")
#: Where the fork's loader lives inside an openpi tree.
FORK_DATASET_RELPATH = Path("groot_utils") / "groot_openpi_dataset.py"
#: Last-resort location when openpi is not importable (a CPU test host).
FORK_CHECKOUT_DATASET = ROBOCASA_OPENPI_SRC / "openpi" / "groot_utils" / "groot_openpi_dataset.py"


# --------------------------------------------------------------------------- window rule lock-step
def locate_fork_dataset(explicit: str | os.PathLike | None = None) -> Path:
    """The fork loader source the served policy was trained with: explicit > importable openpi > checkout."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--fork-dataset-py {path} does not exist")
        return path
    env = os.environ.get("WSM_FORK_DATASET_PY")
    if env:
        return locate_fork_dataset(env)
    try:
        import importlib.util

        spec = importlib.util.find_spec("openpi")
        origin = Path(spec.origin).parent if spec is not None and spec.origin else None
    except Exception:  # noqa: BLE001 - any import trouble means "not importable"
        origin = None
    if origin is not None and (origin / FORK_DATASET_RELPATH).is_file():
        return origin / FORK_DATASET_RELPATH
    if FORK_CHECKOUT_DATASET.is_file():
        return FORK_CHECKOUT_DATASET
    raise FileNotFoundError(
        "cannot locate the fork's groot_openpi_dataset.py (openpi not importable and no checkout at "
        f"{FORK_CHECKOUT_DATASET}); pass --fork-dataset-py or set WSM_FORK_DATASET_PY"
    )


def load_fork_window_rule(source_path: str | os.PathLike):
    """Execute ONLY the loader's `_wsm_causal_window` (importing the module needs robocasa)."""
    source = Path(source_path).read_text(encoding="utf-8")
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_wsm_causal_window"
    ]
    if len(nodes) != 1:
        raise RuntimeError(f"{source_path} defines _wsm_causal_window {len(nodes)} times; expected exactly one")
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)  # noqa: S102
    return namespace["_wsm_causal_window"]


def window_rule_battery() -> list[tuple[np.ndarray, int, int]]:
    """(frame_indices, t, k) cases: serve grids, training grids with the extra final frame, off-grid t."""
    cases = []
    for frames in range(1, 41):
        grid = np.arange(frames, dtype=np.int64) * STORE_STRIDE
        for k in (1, 2, 8, 16, 32):
            cases.append((grid, int(grid[-1]), k))  # serve: t on the newest grid row
            cases.append((grid, int(grid[-1]) + 3, k))  # train: t between grid rows
    for length in (9, 17, 141, 400):  # training grids incl. the final frame
        grid = np.arange(0, length, STORE_STRIDE, dtype=np.int64)
        if grid[-1] != length - 1:
            grid = np.concatenate([grid, [length - 1]])
        for t in (0, 1, 7, 8, 9, length // 2, length - 2, length - 1):
            for k in (1, 8, 16):
                cases.append((grid, int(t), k))
    return cases


def assert_window_rule_lockstep(fork_dataset_py: str | os.PathLike | None = None) -> dict:
    """Refuse unless the fork's training rule and the serve rule agree on every battery case."""
    path = locate_fork_dataset(fork_dataset_py)
    fork_rule = load_fork_window_rule(path)
    checked = 0
    for grid, t, k in window_rule_battery():
        serve = np.asarray(causal_window_indices(grid, t, k), dtype=np.int64)
        train = np.asarray(fork_rule(np.asarray(grid), int(t), int(k)), dtype=np.int64)
        if serve.shape != (k,) or not np.array_equal(serve, train):
            raise RuntimeError(
                "[stage-e-serve] WINDOW RULE MISMATCH between the training loader "
                f"({path}) and wsm_align.causal_window_indices at frame_indices[-1]={int(grid[-1])} "
                f"t={t} k={k}: train={train.tolist()} serve={serve.tolist()}. Refusing to serve."
            )
        checked += 1
    return {"fork_dataset_py": str(path), "cases": checked}


# ------------------------------------------------------------------------------- task lang table
def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def store_task_dirs(omega_root: str | os.PathLike) -> list[Path]:
    root = Path(omega_root).expanduser()
    tasks = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not tasks:
        raise FileNotFoundError(f"no task directories under {root}")
    return tasks


def load_stage_e_task_lang_table(omega_root: str | os.PathLike, *, mode: str = "strict") -> dict:
    """{task: lang[2048] fp32} from a Stage-E ω store's `w.npz` `lang_global` fields.

    `strict` requires every demo of a task to carry the SAME vector (the task_mean contract the
    encoder trained under); `task_mean_of_store` averages them (smoke only — that store's
    encoder trained on per-episode vectors and no serve vector is faithful to it, §25.8).
    """
    if mode not in LANG_TABLE_MODES:
        raise ValueError(f"lang table mode must be one of {LANG_TABLE_MODES}, got {mode!r}")
    table: dict[str, np.ndarray] = {}
    for task_dir in store_task_dirs(omega_root):
        vectors = []
        for demo_dir in sorted(task_dir.iterdir()):
            w_path = demo_dir / "w.npz"
            if not w_path.is_file():
                continue
            with np.load(w_path, allow_pickle=False) as blob:
                if "lang_global" not in blob.files:
                    raise RuntimeError(f"{w_path} has no lang_global; not a Stage-E ω store")
                vectors.append(np.asarray(blob["lang_global"], dtype=np.float32).reshape(-1))
        if not vectors:
            raise RuntimeError(f"no w.npz under {task_dir}")
        stack = np.stack(vectors)
        spread = float(np.abs(stack - stack[0]).max())
        if mode == "strict":
            if spread != 0.0:
                raise RuntimeError(
                    f"[stage-e-serve] ω store {omega_root} is NOT serve-consistent: task "
                    f"{task_dir.name} carries {len(vectors)} demos whose lang_global differ "
                    f"(max|Δ| {spread:.3g}); the encoder conditioned on per-episode vectors and no "
                    "single serve vector reproduces its ω (§25.8). Refusing to derive a table."
                )
            table[task_dir.name] = stack[0]
        else:
            table[task_dir.name] = stack.astype(np.float32).mean(0).astype(np.float16).astype(np.float32)
    return table


def write_task_lang_table(table: dict, path: str | os.PathLike) -> Path:
    """Persist the derived table in the sealed `task_lang_table.npz` schema ({tasks, lang fp16})."""
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted(table)
    np.savez(out, tasks=np.asarray(tasks), lang=np.stack([table[t] for t in tasks]).astype(np.float16))
    return out


def load_store_meta(omega_root: str | os.PathLike) -> dict:
    """`_meta.json` written by export_omega_store, one level above the domain dir (or beside it)."""
    root = Path(omega_root).expanduser()
    for candidate in (root / "_meta.json", root.parent / "_meta.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------------- front end
class StageEServeFrontEnd(StageEServeEncoder):
    """`StageEServeEncoder` + the fp16 round trip on the LANGUAGE vector + an already-pooled mode.

    `_project` in the producer (the D7-gated reference) feeds the adapter `_as_trained(p)` AND
    `_as_trained(lang)`; the base class round-trips only `p`. Serve tables are fp16-exact so this is
    a no-op there, but the parity harness must make no such assumption. `pooled=True` accepts
    pre-pooled tokens [B,T,feat_dim] (the parity replay of `p.npz`); serving always uses raw
    patches [B,T,192,2048] through the frozen pool.
    """

    def __init__(
        self, encoder, domain: str, pool=None, patch_norm=None, device: str = "cuda", pooled: bool = False
    ) -> None:
        if not pooled and pool is None:
            raise ValueError("raw-patch serving needs the frozen pool; pass pool= or pooled=True")
        super().__init__(encoder, domain, pool, patch_norm, device)
        self.pooled = bool(pooled)
        self.feat_dim = int(self.cfg.backbone_dim)
        if pool is not None:
            pool_dim = int(pool.query.shape[-1])
            if pool_dim != self.feat_dim:
                raise ValueError(
                    f"frozen pool emits {pool_dim}-d tokens but the Stage-E encoder consumes "
                    f"{self.feat_dim}-d pooled tokens; wrong pool checkpoint"
                )

    def fuse_inputs(self, patches, proprio, cond_lang):
        del proprio  # Stage-E's pooled contract is proprio-free by construction.
        lang = _as_trained(cond_lang, self.device)
        if self.pooled:
            p = _as_trained(patches, self.device)
            if p.shape[-1] != self.feat_dim:
                raise ValueError(f"pooled input dim {p.shape[-1]} != feat_dim {self.feat_dim}")
        else:
            x = patches.to(self.device).float()
            if x.ndim != 4:
                raise ValueError(f"raw patches must be [B,T,P,D]; got {tuple(x.shape)}")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
                if self.patch_norm is not None:
                    x = self.patch_norm(x)
                p = self.pool(x)
            p = p.float().half().float()
        f, c = self.adapter(p, lang)
        return f, self.encoder.trunk.lang_proj(c)


@dataclass
class StageEServeStack:
    front_end: StageEServeFrontEnd
    conditioner: object
    table: dict
    provenance: dict = field(default_factory=dict)


def load_stage_e_front_end(
    encoder_ckpt: str,
    *,
    domain: str,
    device: str,
    pool_ckpt: str | None,
    expect_sha256: str | None = None,
    expect_encoder_id: str | None = None,
    pooled: bool = False,
) -> tuple[StageEServeFrontEnd, dict]:
    """Load + verify the encoder (sha, encoder_id, domain, dims) and the frozen pool; return the front end."""
    ckpt = Path(encoder_ckpt).expanduser()
    actual_sha = sha256_file(ckpt)
    if expect_sha256 is not None and actual_sha != str(expect_sha256).lower():
        raise RuntimeError(
            f"[stage-e-serve] encoder {ckpt} sha256 {actual_sha} != expected {expect_sha256}; "
            "refusing to serve an unverified encoder (the NaN-encoder lesson)"
        )
    encoder, blob = load_stage_e(ckpt, device)
    encoder_id = str(blob.get("encoder_id", ""))
    if expect_encoder_id is not None and encoder_id != str(expect_encoder_id):
        raise RuntimeError(
            f"[stage-e-serve] encoder_id {encoder_id!r} != the ω store's {expect_encoder_id!r}: this "
            "encoder did not produce the ω the policy trained on"
        )
    if domain not in encoder.adapters:
        raise RuntimeError(f"[stage-e-serve] domain {domain!r} not in adapters {list(encoder.adapters)}")
    if int(encoder.cfg.lang_dim) != 2048:
        raise RuntimeError(f"[stage-e-serve] cfg.lang_dim {encoder.cfg.lang_dim} != the pi tap's 2048")
    norm = pool = None
    pool_sha = None
    if not pooled:
        if not pool_ckpt:
            raise RuntimeError("[stage-e-serve] raw-patch serving requires --pool-ckpt")
        from workspace_models.features.pi_pooled_tap import load_pool

        norm, pool, _pool_id, pool_sha = load_pool(Path(pool_ckpt).expanduser(), device)
    front_end = StageEServeFrontEnd(encoder, domain, pool, norm, device, pooled=pooled)
    provenance = {
        "encoder_ckpt": str(ckpt),
        "encoder_sha256": actual_sha,
        "encoder_id": encoder_id,
        "encoder_step": blob.get("step"),
        "domains": list(blob.get("domains", [])),
        "domain_index": list(blob.get("domain_index", [])),
        "domain": domain,
        "cfg": dict(blob.get("cfg", {})),
        "pool_ckpt": (str(Path(pool_ckpt).expanduser()) if pool_ckpt else None),
        "pool_sha256": pool_sha,
        "patch_in_norm": norm is not None,
    }
    return front_end, provenance


def build_stage_e_serve_stack(
    *,
    encoder_ckpt: str,
    pool_ckpt: str | None,
    domain: str,
    omega_root: str | None,
    task_lang_table: str | None,
    lang_table_mode: str,
    expect_sha256: str | None,
    fork_dataset_py: str | None,
    k_window: int,
    stride: int,
    device: str,
    interface: str,
    parity_demos: int = 0,
    pooled_root: str | None = None,
    parity_frames: int = 24,
    table_out: str | None = None,
) -> StageEServeStack:
    """Everything the serve needs for `--encoder-kind stage_e`, every check fail-closed."""
    from vla_training.eval._groot_wsm_eval import WSMEvalConditioner, load_task_lang_table

    if interface != "tanh":
        raise RuntimeError(f"[stage-e-serve] Stage-E ω arms ride the tanh interface; got {interface!r}")
    if int(stride) != STORE_STRIDE:
        raise RuntimeError(
            f"[stage-e-serve] the Stage-E ω store grid is stride {STORE_STRIDE}; serving at "
            f"--stride {stride} would put ω rows on a different grid than training"
        )
    lockstep = assert_window_rule_lockstep(fork_dataset_py)
    print(
        f"[stage-e-serve] ✓ window rule lock-step: loader == wsm_align on {lockstep['cases']} cases "
        f"({lockstep['fork_dataset_py']})",
        flush=True,
    )

    expect_id = None
    meta = load_store_meta(omega_root) if omega_root else {}
    if meta.get("encoder_id"):
        expect_id = str(meta["encoder_id"])
    front_end, provenance = load_stage_e_front_end(
        encoder_ckpt,
        domain=domain,
        device=device,
        pool_ckpt=pool_ckpt,
        expect_sha256=expect_sha256,
        expect_encoder_id=expect_id,
    )
    if (
        meta.get("encoder_step") is not None
        and provenance["encoder_step"] is not None
        and int(meta["encoder_step"]) != int(provenance["encoder_step"])
    ):
        raise RuntimeError(
            f"[stage-e-serve] ω store was exported by encoder step {meta['encoder_step']} but the "
            f"checkpoint is step {provenance['encoder_step']} (encoder_best.pt vs encoder.pt?)"
        )

    if not omega_root and not task_lang_table:
        raise RuntimeError(
            "[stage-e-serve] need --stage-e-omega-root (derive the task table from the store) and/or --task-lang-table"
        )
    table = None
    table_source = None
    if omega_root:
        table = load_stage_e_task_lang_table(omega_root, mode=lang_table_mode)
        table_source = f"derived from {omega_root} ({lang_table_mode})"
        if lang_table_mode != "strict":
            print(
                "[stage-e-serve] *** SMOKE-ONLY lang table: task_mean_of_store on a store that is "
                "not serve-consistent — NOT a scored configuration ***",
                flush=True,
            )
    if task_lang_table:
        given = load_task_lang_table(task_lang_table)
        if table is not None:
            missing = sorted(set(table) - set(given))
            if missing:
                raise RuntimeError(f"[stage-e-serve] --task-lang-table lacks store tasks {missing}")
            for task, vec in table.items():
                if not np.array_equal(np.asarray(given[task], np.float32), vec):
                    raise RuntimeError(
                        f"[stage-e-serve] --task-lang-table disagrees with the ω store's lang_global "
                        f"for {task}: the policy trained on the store's vector"
                    )
            table_source += f" == {task_lang_table}"
        else:
            table, table_source = given, str(task_lang_table)
    if table_out:
        write_task_lang_table(table, table_out)
        table_source += f" -> {table_out}"
    provenance.update(
        {
            "task_lang_table": table_source,
            "tasks": len(table),
            "k_window": int(k_window),
            "stride": int(stride),
            "lockstep": lockstep,
        }
    )
    print(
        f"[stage-e-serve] encoder_id={provenance['encoder_id']} step={provenance['encoder_step']} "
        f"sha256={provenance['encoder_sha256'][:16]}… pool_sha256="
        f"{(provenance['pool_sha256'] or 'n/a')[:16]}… domain={domain} tasks={len(table)} "
        f"K={k_window} stride={stride} table={table_source}",
        flush=True,
    )

    if parity_demos:
        if not (omega_root and pooled_root):
            raise RuntimeError(
                "[stage-e-serve] --stage-e-parity-demos needs --stage-e-omega-root and --stage-e-pooled-root"
            )
        pooled_front, _ = load_stage_e_front_end(
            encoder_ckpt, domain=domain, device=device, pool_ckpt=None, pooled=True
        )
        report = parity_check(
            pooled_front,
            omega_root=omega_root,
            pooled_root=pooled_root,
            table=table,
            k_window=k_window,
            stride=stride,
            device=device,
            demos=int(parity_demos),
            frames=int(parity_frames),
            lang_mode="table",
        )
        print_parity_report(report)
        if report["verdict"] != "PASS":
            raise RuntimeError("[stage-e-serve] startup parity FAILED; refusing to serve")
        provenance["startup_parity"] = {k: report[k] for k in ("verdict", "worst", "fp16_floor", "demos")}

    conditioner = WSMEvalConditioner(front_end, k_window=int(k_window), stride=int(stride), device=device)
    return StageEServeStack(front_end=front_end, conditioner=conditioner, table=table, provenance=provenance)


# ------------------------------------------------------------------------------------- parity
def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-12)


def fp16_floor(reference: np.ndarray) -> float:
    ref = np.abs(np.asarray(reference, dtype=np.float32))
    ulp = np.where(ref > 0, np.spacing(ref.astype(np.float16)).astype(np.float32), np.float32(6e-8))
    return float(ulp.max())


def pick_parity_demos(omega_root: Path, pooled_root: Path, want: int) -> list[tuple[str, Path, Path]]:
    """Deterministic round-robin over tasks (same policy as stage_e_omega_parity.pick_demos)."""
    per_task: dict[str, list] = {}
    for task_dir in store_task_dirs(omega_root):
        for demo_dir in sorted(task_dir.iterdir()):
            w, p = demo_dir / "w.npz", pooled_root / task_dir.name / demo_dir.name / "p.npz"
            if w.is_file() and p.is_file():
                per_task.setdefault(task_dir.name, []).append((task_dir.name, w, p))
    picked, rnd = [], 0
    while len(picked) < want:
        added = False
        for task in sorted(per_task):
            if rnd < len(per_task[task]):
                picked.append(per_task[task][rnd])
                added = True
                if len(picked) == want:
                    break
        if not added:
            break
        rnd += 1
    return picked


@torch.no_grad()
def parity_check(
    front_end: StageEServeFrontEnd,
    *,
    omega_root: str,
    pooled_root: str,
    table: dict | None,
    k_window: int,
    stride: int,
    device: str,
    demos: int,
    frames: int,
    lang_mode: str = "table",
    cos_bar: float = COS_BAR,
) -> dict:
    """Replay stored pooled frames through the real serve stack and score against the stored ω.

    Per grid frame i (t = i*stride) of each demo:
      * newest served row == stored ω[i]  (cos >= cos_bar, |Δ| <= fp16 floor)   -- D7 bar
      * served K-row window == wsm_align.window_at(stored ω, serve grid, t, K)   -- train/serve rule
    `lang_mode`: 'table' conditions on the derived per-task vector (what the server does);
    'store' on each demo's own stored lang_global (identity check for non-serve-consistent stores).
    """
    from vla_training.eval._groot_wsm_eval import WSMEvalConditioner

    if not front_end.pooled:
        raise ValueError("parity_check needs a pooled=True front end (it replays p.npz)")
    if lang_mode not in ("table", "store"):
        raise ValueError(f"lang_mode must be 'table' or 'store', got {lang_mode!r}")
    if lang_mode == "table" and not table:
        raise ValueError("lang_mode 'table' needs the derived task table")
    picked = pick_parity_demos(Path(omega_root).expanduser(), Path(pooled_root).expanduser(), demos)
    if len(picked) < demos:
        raise RuntimeError(f"only {len(picked)} paired (w.npz, p.npz) demos found, wanted {demos}")
    rows, worst = [], {"cos": 1.0, "abs": 0.0, "window_abs": 0.0}
    floor_all = 0.0
    for task, w_path, p_path in picked:
        with np.load(w_path, allow_pickle=False) as wb, np.load(p_path, allow_pickle=False) as pb:
            w_ref = np.asarray(wb["w"], dtype=np.float32)
            lang_store = np.asarray(wb["lang_global"], dtype=np.float32)
            p = np.asarray(pb["p"])
            if not np.array_equal(
                np.asarray(pb["frame_indices"], np.int64), np.asarray(wb["frame_indices"], np.int64)
            ):
                raise RuntimeError(f"{task}/{p_path.parent.name}: p.npz and w.npz grids differ")
        n = min(len(p), frames) if frames > 0 else len(p)
        lang = table[task] if lang_mode == "table" else lang_store
        if lang_mode == "table" and task not in table:
            raise RuntimeError(f"task {task} missing from the lang table")
        conditioner = WSMEvalConditioner(front_end, k_window=int(k_window), stride=int(stride), device=device)
        conditioner.reset(lang)
        served_newest, window_abs = [], 0.0
        grid = np.arange(n, dtype=np.int64) * int(stride)
        for i in range(n):
            window, _lang_out = conditioner.step(p[i], np.zeros(1, dtype=np.float32))
            window = window.detach().float().cpu().numpy()
            if window.shape != (int(k_window), w_ref.shape[1]):
                raise RuntimeError(f"served window shape {window.shape} != ({k_window}, {w_ref.shape[1]})")
            served_newest.append(window[-1])
            # The training-side rule applied to the STORED ω at the same grid time must select the
            # same rows the server shipped (within fp16 of the stored values).
            expected = window_at(w_ref[:n], grid, int(grid[i]), int(k_window))
            window_abs = max(window_abs, float(np.abs(window - expected).max()))
        served = np.stack(served_newest)
        c = cosine(served, w_ref[:n])
        a = float(np.abs(served - w_ref[:n]).max())
        floor = fp16_floor(w_ref[:n])
        floor_all = max(floor_all, floor)
        worst["cos"] = min(worst["cos"], float(c.min()))
        worst["abs"] = max(worst["abs"], a)
        worst["window_abs"] = max(worst["window_abs"], window_abs)
        rows.append(
            {
                "task": task,
                "demo": p_path.parent.name,
                "frames": int(n),
                "cos_min": float(c.min()),
                "cos_mean": float(c.mean()),
                "absmax": a,
                "window_absmax": window_abs,
                "fp16_floor": floor,
            }
        )
    ok = worst["cos"] >= cos_bar and worst["abs"] <= floor_all and worst["window_abs"] <= floor_all
    return {
        "verdict": "PASS" if ok else "FAIL",
        "cos_bar": cos_bar,
        "fp16_floor": floor_all,
        "worst": worst,
        "k_window": int(k_window),
        "stride": int(stride),
        "lang_mode": lang_mode,
        "demos": len(rows),
        "rows": rows,
    }


def print_parity_report(report: dict) -> None:
    for r in report["rows"]:
        print(
            f"  {r['task']:26s} {r['demo']} F={r['frames']:4d}  cos>={r['cos_min']:.6f} "
            f"|Δ|<={r['absmax']:.2e}  window|Δ|<={r['window_absmax']:.2e}  (floor {r['fp16_floor']:.2e})",
            flush=True,
        )
    w = report["worst"]
    print(
        f"-- gate (cos >= {report['cos_bar']}, max|Δ| <= fp16 floor {report['fp16_floor']:.2e}; "
        f"K={report['k_window']} stride={report['stride']} lang={report['lang_mode']}) --"
    )
    print(f"  newest-row worst cos {w['cos']:.6f}  max|Δ| {w['abs']:.2e}  window max|Δ| {w['window_abs']:.2e}")
    print(f"VERDICT: {report['verdict']}", flush=True)


# ---------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lock = sub.add_parser("lockstep", help="assert the fork loader's window rule == wsm_align")
    lock.add_argument("--fork-dataset-py", default=None)
    par = sub.add_parser("parity", help="online ω vs stored ω through the real serve stack (CPU ok)")
    par.add_argument("--encoder", required=True)
    par.add_argument("--omega-root", required=True, help=".../omega/<cell>/remembench")
    par.add_argument("--pooled-root", required=True, help=".../wsm_pooled/rmb_pi_100k")
    par.add_argument("--domain", default="remembench")
    par.add_argument("--k-window", type=int, required=True, help="the policy's trained window (pos_decay_bias)")
    par.add_argument("--stride", type=int, default=STORE_STRIDE)
    par.add_argument("--demos", type=int, default=3)
    par.add_argument("--frames", type=int, default=24, help="grid frames per demo (0 = all)")
    par.add_argument("--lang-mode", default="table", choices=("table", "store"))
    par.add_argument("--lang-table-mode", default="strict", choices=LANG_TABLE_MODES)
    par.add_argument("--expect-encoder-sha256", default=None)
    par.add_argument("--fork-dataset-py", default=None)
    par.add_argument("--device", default="cpu")
    par.add_argument("--table-out", default=None, help="write the derived table (sealed npz schema)")
    par.add_argument("--out", default=None, help="JSON report")
    args = ap.parse_args(argv)

    if args.cmd == "lockstep":
        info = assert_window_rule_lockstep(args.fork_dataset_py)
        print(f"LOCKSTEP OK: {info['cases']} cases against {info['fork_dataset_py']}")
        return 0

    lockstep = assert_window_rule_lockstep(args.fork_dataset_py)
    print(f"[parity] window rule lock-step OK ({lockstep['cases']} cases, {lockstep['fork_dataset_py']})")
    meta = load_store_meta(args.omega_root)
    front_end, prov = load_stage_e_front_end(
        args.encoder,
        domain=args.domain,
        device=args.device,
        pool_ckpt=None,
        pooled=True,
        expect_sha256=args.expect_encoder_sha256,
        expect_encoder_id=meta.get("encoder_id"),
    )
    print(f"[parity] encoder_id={prov['encoder_id']} step={prov['encoder_step']} sha256={prov['encoder_sha256']}")
    table = None
    if args.lang_mode == "table":
        table = load_stage_e_task_lang_table(args.omega_root, mode=args.lang_table_mode)
        if args.table_out:
            print(f"[parity] wrote {write_task_lang_table(table, args.table_out)}")
    report = parity_check(
        front_end,
        omega_root=args.omega_root,
        pooled_root=args.pooled_root,
        table=table,
        k_window=args.k_window,
        stride=args.stride,
        device=args.device,
        demos=args.demos,
        frames=args.frames,
        lang_mode=args.lang_mode,
    )
    report["encoder"] = prov
    report["lockstep"] = lockstep
    print_parity_report(report)
    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(json.dumps(report, indent=1, default=str))
        print(f"wrote {args.out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
