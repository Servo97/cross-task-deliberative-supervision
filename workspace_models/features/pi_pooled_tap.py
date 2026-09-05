#!/usr/bin/env python3
"""pi_pooled_tap — FUSED frozen pi0.5 tap + frozen WSMv1 pool, straight to the `wsm_pooled` schema.

Why fused. The sanctioned RoboCasa path is two stages: `pi_cache_features.py` writes
`patch_tokens.npy [F,192,2048] fp16` (~300 GB for RoboCasa, ~61 GB for ReMemBench) and
`pool_patch_tokens.py` then folds it through the FROZEN WSMv1 `patch_in_norm + PatchPool` into
`p.npz [F,512]`. The intermediate is pure waste for a domain we only ever consume pooled, and the
"~61 GB of intermediates" was one of the two reasons ReMemBench was ruled OUT of the Stage-E joint
encoder (h14_p0_status §14.2). This module runs the two frozen stages back-to-back per frame batch
so the raw tokens never touch disk, and emits BYTE-COMPATIBLE `p.npz` files.

Contract — reproduced EXACTLY from `wsm_pooled/pi_100k/<Task>/demo_%06d/p.npz`:

    p              [F,512]  fp16    frozen pool over the 192 bin-averaged SigLIP patches
    frame_indices  [F]      int64   episode frame indices, stride 8, ALWAYS incl. the final frame
    lang_global    [2048]   float32 mean over frames of the tap's masked-mean language embedding
    encoder_id     ()       str     "wsm_pool:<first 16 hex of the pool ckpt sha256>"
    pool_sha256    ()       str     full sha256 of the frozen pool checkpoint
  + `.done_pooled` marker (the resume gate every downstream consumer already knows)

and, additive (ignored by every existing reader, present so provenance is not folklore):

    backbone_id    ()       str     the frozen pi0.5 checkpoint directory name
    prompt_source  ()       str     "expanded_prompt" | "lerobot_instruction"

Prompt. RoboCasa's tap fed the Qwen-EXPANDED prompt when an `ep*_subgoals.json` existed and fell
back to the raw LeRobot instruction otherwise (`pi_cache_features.py:83`). No subgoal tree exists
for ReMemBench, so ReMemBench takes that same in-code fallback branch — the raw instruction, which
is also exactly what the policy is served at eval. Recorded per file in `prompt_source` rather than
assumed, because the prefix is bidirectional: the prompt reaches the patch tokens.

Views/geometry/state are RoboCasa's unchanged: ReMemBench `remembench_v02` is a LeRobot tree with
the same 3 views at 256 px, the same `video_path` template, fps 20, and the same 16-d
`observation.state` (verified 2026-08-28), so `Pi05BackboneTap` applies with zero adaptation.

Run in the openpi-jax env (jax + openpi + torch + av):

  WSM_CONFIGS_DIR=~/Research/TRI/internal_training/robocasa PYTHONPATH=<repo> \
  CUDA_VISIBLE_DEVICES=1 ~/Research/envs/openpi-jax-latest/bin/python -m \
    workspace_models.features.pi_pooled_tap \
      --domain remembench \
      --dataset-root ~/Research/TRI/wsm_data/remembench_v02/train \
      --labels-root  ~/Research/TRI/wsm_data/wsm_labels_causal_v1/remembench13 \
      --ckpt         ~/Research/TRI/wsm_data/local_ckpts/pi05_on_149999 \
      --pool-ckpt    ~/Research/TRI/wsm_data/wsm_runs/pi_wsm_v1/wsm_step100000.pt \
      --out-root     ~/Research/TRI/wsm_data/wsm_pooled/rmb_pi_100k
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workspace_models.labels.geometry import VIEW_LEROBOT_KEY, VIEWS  # noqa: E402

STRIDE = 8  # measured off wsm_pooled/pi_100k: frame_indices = 0,8,16,... plus the final frame


# ------------------------------------------------------------------------------------ dataset io
def resolve_lerobot_dir(dataset_root: Path, task: str) -> Path | None:
    """`<root>/<task>/<session>/lerobot` — the same glob caption_segments.resolve_lerobot_dir uses."""
    hits = sorted(dataset_root.glob(f"{task}/*/lerobot")) or sorted(dataset_root.glob(f"*/{task}/*/lerobot"))
    return hits[0] if hits else None


def load_episode_meta(root: Path) -> tuple[dict, dict]:
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = {}
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        episodes[int(rec["episode_index"])] = {
            "length": int(rec["length"]),
            "prompt": str((rec.get("tasks") or [""])[0]),
        }
    return episodes, info


def episode_video_path(root: Path, info: dict, video_key: str, ep: int) -> Path:
    chunk = ep // int(info.get("chunks_size", 1000))
    return root / info["video_path"].format(episode_chunk=chunk, video_key=video_key, episode_index=ep)


def frame_selection(n: int) -> np.ndarray:
    """The pi_100k convention: stride 8 from 0, with the final frame always appended."""
    sel = np.arange(0, n, STRIDE, dtype=np.int64)
    if len(sel) == 0 or sel[-1] != n - 1:
        sel = np.append(sel, n - 1)
    return sel


def decode_frames_at(path: Path, indices: np.ndarray, expected_len: int) -> np.ndarray:
    """Sequential PyAV decode (no seeking) -> [K,H,W,3] uint8. Fails loud on truncation."""
    import av

    wanted = {int(i) for i in indices}
    out: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        pos = 0
        stop = max(wanted)
        for frame in container.decode(stream):
            if pos in wanted:
                out[pos] = frame.to_ndarray(format="rgb24")
            pos += 1
            if pos > stop:
                break
    if pos < min(expected_len, max(wanted) + 1):
        raise RuntimeError(f"{path}: decoded {pos} frames, expected >= {expected_len}")
    missing = sorted(wanted - set(out))
    if missing:
        raise RuntimeError(f"{path}: decoder did not produce frames {missing[:5]}")
    return np.stack([out[int(i)] for i in indices]).astype(np.uint8)


def state_at(root: Path, info: dict, ep: int, sel: np.ndarray) -> np.ndarray:
    import pandas as pd

    chunk = ep // int(info.get("chunks_size", 1000))
    path = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    df = pd.read_parquet(path)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    return state[sel]


# ---------------------------------------------------------------------------------- frozen pool
def load_pool(ckpt_path: Path, device: str):
    """(patch_in_norm | None, PatchPool, encoder_id) — the RoboCasa pooler, byte-for-byte.

    Delegates to `pool_patch_tokens.load_pool`, i.e. the SAME frozen `patch_in_norm + PatchPool`
    the `wsm_pooled/pi_100k` store was produced with, loaded through the same `load_wsm` (which
    reads the ckpt's `input_norm` flag and refuses a non-finite encoder). proprio_dim=2048 is the
    pi loader arg; it touches no weight the pool uses, but strict=True needs the right shape.
    """
    import hashlib

    from workspace_models.features.pool_patch_tokens import load_pool as _load_pool

    norm, pool, _meta = _load_pool(str(Path(ckpt_path).expanduser()), device, proprio_dim=2048)
    ck = Path(ckpt_path).expanduser()
    # CONTENT-ADDRESSED, not path-derived. The old id was f"{parent.name}/{name}", which encodes
    # WHERE the checkpoint happened to sit: the same frozen pooler produced "pi_wsm_v1/wsm_step100000.pt"
    # locally and "work/wsm_step100000.pt" on a SageMaker node (§32.1), so two stores built from
    # byte-identical weights carried different provenance strings and looked like they disagreed.
    # The hash cannot drift with the working directory. Existing stores are NOT rewritten.
    h = hashlib.sha256()
    with ck.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    return norm, pool, f"wsm_pool:{sha[:16]}", sha


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="remembench", choices=("remembench", "robocasa"))
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument(
        "--labels-root",
        default="",
        help="causal_v1 npz root; its vlm_episode_pi_*.npz names define the episode set "
        "(and therefore the episode ids the Stage-E label artifact uses)",
    )
    ap.add_argument("--ckpt", required=True, help="frozen pi05 checkpoint dir (params/ + assets/)")
    ap.add_argument("--config", default="pi05_rc_mg60_bal33")
    ap.add_argument("--pool-ckpt", required=True, help="frozen WSMv1 ckpt supplying patch_in_norm+pool")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--batch-size", type=int, default=32, help="frames per backbone forward")
    ap.add_argument("--limit", type=int, default=0, help="cap episodes per task (0 = all)")
    ap.add_argument(
        "--prompts-json",
        default="",
        help="optional {task: {episode: expanded_prompt}} to reproduce an expanded-prompt tap",
    )
    ap.add_argument("--worker-idx", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()

    import torch

    from workspace_models.features.pi_backbone_tap import Pi05BackboneTap

    dataset_root = Path(args.dataset_root).expanduser()
    labels_root = Path(args.labels_root).expanduser() if args.labels_root else None
    out_root = Path(args.out_root).expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    expanded = json.loads(Path(args.prompts_json).expanduser().read_text()) if args.prompts_json else {}

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        if labels_root is None:
            raise SystemExit("--tasks or --labels-root is required to enumerate the corpus")
        tasks = sorted(d.name for d in labels_root.iterdir() if d.is_dir() and not d.name.startswith("_"))

    # ---- enumerate the corpus FIRST, then shard, then apply the resume gate (h14 §13's fix) ----
    jobs = []
    unresolved = []
    for task in tasks:
        root = resolve_lerobot_dir(dataset_root, task)
        if root is None:
            unresolved.append(task)
            continue
        ep_meta, info = load_episode_meta(root)
        if labels_root is not None and (labels_root / task).exists():
            eps = [
                int(re.search(r"(\d+)\.npz$", p.name).group(1))
                for p in sorted((labels_root / task).glob("vlm_episode_pi_*.npz"))
            ]
        else:
            eps = sorted(ep_meta)
        if args.limit:
            eps = eps[: args.limit]
        for ep in eps:
            meta = ep_meta.get(ep)
            if meta is None:
                unresolved.append(f"{task}/ep{ep}")
                continue
            jobs.append((task, ep, root, info, meta))
    jobs.sort(key=lambda j: (j[0], j[1]))
    jobs = jobs[args.worker_idx :: args.num_workers]
    todo = [j for j in jobs if not (out_root / j[0] / f"demo_{j[1]:06d}" / ".done_pooled").exists()]
    print(
        f"[tap] {len(todo)} episodes to do / {len(jobs) - len(todo)} already done "
        f"(shard {args.worker_idx}/{args.num_workers}); unresolved={len(unresolved)}",
        flush=True,
    )
    if unresolved:
        print(f"[warn] unresolved: {unresolved[:5]}{' ...' if len(unresolved) > 5 else ''}", flush=True)
    if not todo:
        return

    norm, pool, encoder_id, pool_sha256 = load_pool(Path(args.pool_ckpt), device)
    tap = Pi05BackboneTap(str(Path(args.ckpt).expanduser()), args.config)
    backbone_id = Path(args.ckpt).expanduser().name
    print(
        f"[tap] backbone={backbone_id} pool={encoder_id} device={device} "
        f"patch_in_norm={'yes' if norm is not None else 'absent'}",
        flush=True,
    )

    t_start = time.time()
    n_frames_total = 0
    for i, (task, ep, root, info, meta) in enumerate(todo):
        t0 = time.time()
        n = int(meta["length"])
        sel = frame_selection(n)
        prompt = str(expanded.get(task, {}).get(str(ep), "")) or str(meta["prompt"])
        prompt_source = "expanded_prompt" if expanded.get(task, {}).get(str(ep)) else "lerobot_instruction"
        frames = {v: decode_frames_at(episode_video_path(root, info, VIEW_LEROBOT_KEY[v], ep), sel, n) for v in VIEWS}
        state = state_at(root, info, ep, sel)

        F = len(sel)
        pooled = torch.empty(F, pool.query.shape[-1], dtype=torch.float16)
        langs = []
        for lo in range(0, F, args.batch_size):
            hi = min(lo + args.batch_size, F)
            # Pad the trailing batch up to batch_size and drop the padding after the forward. The
            # tap is jitted on the batch shape, so a ragged last batch would trigger a fresh XLA
            # compile for almost every episode; padding keeps the whole run on ONE compiled shape.
            # Rows are independent in the prefix forward, so the padding cannot touch a kept row.
            pad = args.batch_size - (hi - lo)
            if pad:
                sl = {v: np.concatenate([frames[v][lo:hi], np.repeat(frames[v][hi - 1 : hi], pad, 0)]) for v in VIEWS}
                st = np.concatenate([state[lo:hi], np.repeat(state[hi - 1 : hi], pad, 0)])
            else:
                sl, st = {v: frames[v][lo:hi] for v in VIEWS}, state[lo:hi]
            res = tap.tap(sl, st, prompt)
            if pad:
                res.patch_tokens = res.patch_tokens[: hi - lo]
                res.lang_emb = res.lang_emb[: hi - lo]
            langs.append(res.lang_emb)
            x = torch.from_numpy(res.patch_tokens.astype(np.float32)).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                if norm is not None:
                    x = norm(x)
                p = pool(x[None])[0]  # PatchPool takes [B,T,P,D] -> [B,T,512]
            pooled[lo:hi] = p.float().half().cpu()

        lang_per_frame = np.concatenate(langs)  # [F,2048] fp16
        lang_global = lang_per_frame.astype(np.float32).mean(0).astype(np.float16)  # RoboCasa order
        if not torch.isfinite(pooled.float()).all():
            raise SystemExit(f"[tap] NON-FINITE pooled tokens for {task}/ep{ep}")
        if pooled.float().abs().max() > 60000:
            raise SystemExit(f"[tap] fp16 overflow risk for {task}/ep{ep}")

        out_dir = out_root / task / f"demo_{ep:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "p.npz",
            p=pooled.numpy(),
            frame_indices=sel.astype(np.int64),
            lang_global=np.asarray(lang_global, dtype=np.float32),
            encoder_id=np.array(encoder_id),
            pool_sha256=np.array(pool_sha256),
            backbone_id=np.array(backbone_id),
            prompt_source=np.array(prompt_source),
        )
        (out_dir / ".done_pooled").touch()
        n_frames_total += F
        dt = time.time() - t0
        if i % 10 == 0 or i == len(todo) - 1:
            elapsed = time.time() - t_start
            print(
                f"[tap] {i + 1}/{len(todo)} {task}/ep{ep} F={F} ({n} frames) {dt:.1f}s "
                f"| cum {n_frames_total} frames, {n_frames_total / max(elapsed, 1e-9):.2f} fr/s, "
                f"{(i + 1) / max(elapsed, 1e-9) * 3600:.0f} ep/h",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"[tap] COMPLETE: {len(todo)} episodes, {n_frames_total} frames in {elapsed / 60:.1f} min "
        f"({n_frames_total / max(elapsed, 1e-9):.2f} frames/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
