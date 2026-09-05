"""Policy-feature stage: precompute the frozen WSM workspace latent w_t per demo for WSM-conditioned
VLA post-training.

The WSM encoder is FROZEN, and its inputs (the frozen-VLM patch tokens + proprio + global language)
are ALREADY cached on disk by the feature stage (cache_features.py / pi_cache_features.py). So this
stage just runs ONE cheap forward of the ~23.5M-param WorkspaceEncoder per demo over the demo's full
stride-8 grid and stores w_t — NO VLM re-run. The result is what the policy conditions on:

    w = WorkspaceModel.encode(patches[1,F,192,2048], proprio[1,F,Dp], cond_lang[1,F,2048]) -> [1,F,512]

w_t is produced ONLY on the cache's subsampled (stride-8) grid — exactly the grid the encoder was
trained on (the time-embedding is indexed by grid POSITION 0..F-1, so feeding native-rate frames would
be out-of-distribution). The policy/dataloader aligns w_t to a native timestep t with a CAUSAL window
(the last K grid tokens with frame_indices <= t); that K/stride logic lives in the CONSUMER, not here.

CANARY LOCK (GR00T): cond_lang = the global expanded-prompt embedding for every frame (subgoal-dropout
= 1.0), because eval has no online subgoal decomposition — train/eval must use the same language cond.

  <out-root>/<task>/demo_<ep>/w.npz   { w [F,512] fp16, frame_indices [F] int64 }  +  .done_policy_feats
  <out-root>/_meta.json               provenance (wsm ckpt, step, config, subgoal_dropout)

Model-agnostic over backbones (groot now; pi errors until train_wsm_from_pi_05.py is built + a pi WSM
encoder is trained). Frozen, eval/no-grad. See internal_planning_and_todos (WSM-conditioned canary).

  python -m workspace_models.features.generate_policy_features \
      --backbone groot --cache-root ~/Research/TRI/wsm_data/wsm_cache \
      --wsm-ckpt ~/Research/TRI/wsm_data/wsm_ckpts/groot_wsm/wsm_step60000.pt \
      --out-root ~/Research/TRI/wsm_data/wsm_policy_feats --device cuda:0
  # 8-way shard across the H100s:  --shard 0/8 ... --shard 7/8  (one GPU each)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from workspace_models.networks.wsm_model import WorkspaceModel, WSMConfig

# backbone -> the feats.npz key holding the per-frame "proprio" embedding fed to the WSM encoder, and
# the encoder's proprio_dim. groot: the CategorySpecificMLP state embedding [F,1536]. pi: pi0.5 bakes
# robot state into the prompt, so lang_per_frame [F,2048] fills the proprio slot (see pi_cache_features.py
# / data.load_demo_pi); the pi WSM encoder uses proprio_dim=2048 (matches train_wsm_from_pi_05).
_PROPRIO_KEY = {"groot": "state_emb", "pi": "lang_per_frame"}
_PROPRIO_DIM = {"groot": 1536, "pi": 2048}


def load_wsm(ckpt_path: str, device: str, proprio_dim: int = 1536) -> tuple[WorkspaceModel, dict]:
    """Load the FROZEN WorkspaceModel from a trainer checkpoint ({model, cfg, feat_scale, step})."""
    blob = torch.load(Path(ckpt_path).expanduser(), map_location="cpu")
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    # Read arch flags the ckpt was trained with (e.g. input_norm), so old (flag absent -> False) and new
    # stabilized encoders both load under strict=True.
    saved_cfg = blob.get("cfg", {}) if isinstance(blob, dict) else {}
    input_norm = bool(saved_cfg.get("input_norm", False))
    model = WorkspaceModel(WSMConfig(proprio_dim=proprio_dim, input_norm=input_norm)).to(device).eval()
    model.load_state_dict(sd, strict=True)  # strict: catch any dim/arch drift loudly
    # GUARD: refuse a diverged (NaN/Inf) encoder. A non-finite encoder silently produces NaN w_t ->
    # NaN modulator -> 0% eval (cost us a full groot canary). Fail loud at load instead.
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise ValueError(
            f"WSM encoder {ckpt_path} has NON-FINITE weights in {len(bad)} tensors "
            f"(e.g. {bad[:3]}) — diverged checkpoint; refusing to precompute w_t from it."
        )
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {k: v for k, v in blob.items() if k != "model"} if isinstance(blob, dict) else {}
    return model, meta


def _cond_lang_global(lang_global: np.ndarray, F: int) -> np.ndarray:
    """Global expanded-prompt embedding broadcast over F frames, kept fp16 for host transfer."""
    return np.broadcast_to(lang_global.astype(np.float16), (F, lang_global.shape[-1])).copy()


@torch.no_grad()
def encode_demo(
    model: WorkspaceModel, demo_dir: Path, backbone: str, subgoal_dropout: float, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the frozen encoder over one demo's cached features -> (w [F,512] fp16, frame_indices [F])."""
    if subgoal_dropout != 1.0:
        raise NotImplementedError(
            "only subgoal_dropout=1.0 (global language) is supported for the "
            "canary; per-frame subgoals would need the label npz + keyframes."
        )
    patch = np.load(demo_dir / "patch_tokens.npy", mmap_mode="r")  # [F,192,2048] fp16
    f = np.load(demo_dir / "feats.npz", allow_pickle=True)
    frame_indices = f["frame_indices"].astype(np.int64)  # [F]
    F = patch.shape[0]
    proprio = np.asarray(f[_PROPRIO_KEY[backbone]], dtype=np.float16)  # [F,Dp]
    lang_global = np.asarray(f["lang_global"], dtype=np.float16)  # [2048] (the modulator's lang cond)
    cond_lang = _cond_lang_global(lang_global, F)  # [F,2048]

    # Keep the large arrays fp16 over CPU RAM/PCIe, then cast on-device for the FP32 encoder.
    pt = torch.from_numpy(np.asarray(patch))[None].to(device).float()  # [1,F,192,2048]
    pr = torch.from_numpy(proprio)[None].to(device).float()  # [1,F,Dp]
    lg = torch.from_numpy(cond_lang)[None].to(device).float()  # [1,F,2048]
    w = model.encode(pt, pr, lg)[0]  # [F,512]
    # GUARD: never write NaN/Inf w_t. If the encoder produces non-finite latents (diverged ckpt, bad
    # input), abort the whole precompute rather than poison the policy post-training silently.
    if not torch.isfinite(w).all():
        raise ValueError(
            f"non-finite w_t for {demo_dir} (finite frac "
            f"{torch.isfinite(w).float().mean().item():.3f}) — aborting precompute."
        )
    return w.to(torch.float16).cpu().numpy(), frame_indices, lang_global


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="groot", choices=["groot", "pi"])
    ap.add_argument("--cache-root", required=True, help="frozen-feature cache root (<root>/<task>/demo_*/)")
    ap.add_argument("--wsm-ckpt", required=True, help="FROZEN WorkspaceModel checkpoint (wsm_step*.pt)")
    ap.add_argument("--out-root", required=True, help="where w.npz lands (mirrors the cache layout)")
    ap.add_argument("--subgoal-dropout", type=float, default=1.0, help="1.0 = global language (canary lock)")
    ap.add_argument("--tasks", default="", help="comma-separated task subset (default: all in cache-root)")
    ap.add_argument("--limit", type=int, default=0, help="cap demos per task (0 = all; for smoke tests)")
    ap.add_argument("--shard", default="0/1", help="i/N — process tasks whose index %% N == i (multi-GPU)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cache_root = Path(args.cache_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    i, n = (int(x) for x in args.shard.split("/"))

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or sorted(
        d.name for d in cache_root.iterdir() if d.is_dir() and not d.name.startswith("_")
    )
    tasks = [t for k, t in enumerate(tasks) if k % n == i]

    model, ckpt_meta = load_wsm(args.wsm_ckpt, args.device, proprio_dim=_PROPRIO_DIM[args.backbone])
    step = ckpt_meta.get("step", "?")
    print(
        f"[polfeat] backbone={args.backbone} wsm_step={step} shard={args.shard} "
        f"tasks={len(tasks)} device={args.device} dropout={args.subgoal_dropout}",
        flush=True,
    )
    # Only shard 0 publishes shared provenance; concurrent shards writing this file was a race.
    if i == 0:
        meta_tmp = out_root / "._meta.json.incomplete"
        meta_tmp.write_text(
            json.dumps(
                {
                    "backbone": args.backbone,
                    "wsm_ckpt": str(Path(args.wsm_ckpt).expanduser()),
                    "wsm_step": step,
                    "subgoal_dropout": args.subgoal_dropout,
                    "cache_root": str(cache_root),
                    "num_shards": n,
                },
                indent=2,
            )
        )
        os.replace(meta_tmp, out_root / "_meta.json")

    t0, done, skipped = time.time(), 0, 0
    for task in tasks:
        demos = sorted((cache_root / task).glob("demo_*"))
        if args.limit:
            demos = demos[: args.limit]
        for demo_dir in demos:
            if not (demo_dir / ".done_features").exists():
                continue  # incomplete cache entry
            out_dir = out_root / task / demo_dir.name
            done_marker, out_npz = out_dir / ".done_policy_feats", out_dir / "w.npz"
            if done_marker.exists() and out_npz.exists():
                skipped += 1
                continue
            w, frame_indices, lang_global = encode_demo(
                model, demo_dir, args.backbone, args.subgoal_dropout, args.device
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            # lang_global rides along: the GR00T processor drops metadata, so per-sample language for the
            # modulator must be shipped inside w.npz (collator auto-stacks it into action_input.lang_global).
            tmp_npz = out_dir / ".w.npz.incomplete"
            with tmp_npz.open("wb") as f:
                np.savez(f, w=w, frame_indices=frame_indices, lang_global=lang_global)
            os.replace(tmp_npz, out_npz)
            marker_tmp = out_dir / ".done_policy_feats.incomplete"
            marker_tmp.write_text("ok")
            os.replace(marker_tmp, done_marker)
            done += 1
            if done % 200 == 0:
                print(
                    f"[polfeat] {done} demos ({task} {demo_dir.name} F={w.shape[0]}) {time.time() - t0:.0f}s",
                    flush=True,
                )
    print(
        f"[polfeat] DONE shard {args.shard}: {done} written, {skipped} skipped, {time.time() - t0:.0f}s -> {out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
