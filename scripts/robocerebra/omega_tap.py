#!/usr/bin/env python3
"""pi0.5 backbone tap on RoboCerebra frames -> the inputs the Stage-S ω encoder consumes.

Adapted from ``Isaac-GR00T/wsm/pi05_tap_cache.py``, which already emits the **2-view /
128-token** layout (base + wrist) that the RoboCerebra ω convention calls for. Nothing is
fabricated to reach 3 views: RoboCerebra is run in the encoder's P-agnostic 2-view mode, and
every downstream chance baseline uses 128, not 192.

Per sampled frame it writes:

* ``tokens``      [T, 128, 2048] fp16 — base then wrist, each the SigLIP 14x14 grid
  bin-averaged to 8x8 exactly as ``pool_grid`` does upstream (edges = linspace(0,14,9).astype(int)).
* ``pooled_img``  [T, 2048] — mean over valid image-token positions (the encoder's ``proprio``
  slot; Stage-S was trained with proprio_dim 2048 = this pooled feature, not raw robot state).
* ``pooled_lang`` [T, 2048] — mean over valid language positions (the encoder's ``cond_lang``).
* ``pooled_all``  [T, 2048], ``frame_idx`` [T], ``subtask_index`` [T].

Runs in the openpi venv against a local pi05_libero checkpoint.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

GRID_IN, GRID_OUT = 14, 8
N_IMG_TOK = GRID_IN * GRID_IN  # 196 tokens per view before binning


def pool_grid(tokens_196: np.ndarray) -> np.ndarray:
    """[196, D] (14x14) -> [64, D] (8x8) by bin-averaging. Verbatim from the upstream tap."""
    g = tokens_196.reshape(GRID_IN, GRID_IN, -1)
    edges = np.linspace(0, GRID_IN, GRID_OUT + 1).astype(int)
    out = np.empty((GRID_OUT, GRID_OUT, g.shape[-1]), dtype=g.dtype)
    for i in range(GRID_OUT):
        for j in range(GRID_OUT):
            out[i, j] = g[edges[i] : edges[i + 1], edges[j] : edges[j + 1]].mean(axis=(0, 1))
    return out.reshape(GRID_OUT * GRID_OUT, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="RoboCerebra LeRobot dataset dir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--frames-per-ep", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import dataclasses

    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config
    import pandas as pd
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    checkpoint = pathlib.Path(args.checkpoint).resolve()
    link_root = checkpoint.parent / "_serve_assets"
    link_root.mkdir(parents=True, exist_ok=True)
    if not (link_root / args.config).exists():
        (link_root / args.config).symlink_to(checkpoint / "assets", target_is_directory=True)
    train_config = dataclasses.replace(_config.get_config(args.config), assets_base_dir=str(link_root))
    policy = policy_config.create_trained_policy(train_config, checkpoint)
    model = policy._model
    input_transform = getattr(policy, "_input_transform", None) or policy.input_transform

    def embed_batch(examples: list[dict]):
        transformed = [input_transform(e) for e in examples]
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *transformed)
        obs = _model.Observation.from_dict(batch)
        prefix_tokens, prefix_mask, ar_mask = model.embed_prefix(obs)
        attn = make_attn_mask(prefix_mask, ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (hidden, _), _ = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=positions)
        hidden = np.asarray(hidden.astype(jnp.float32))
        mask = np.asarray(prefix_mask)
        # pi0.5 always lays out 3 image slots; RoboCerebra fills 2 and zero-masks the third, so
        # the language span still starts at 3 * N_IMG_TOK even though only 2 views are real.
        n_img_total = 3 * N_IMG_TOK
        img_mask = mask.copy()
        img_mask[:, n_img_total:] = False
        lang_mask = mask.copy()
        lang_mask[:, :n_img_total] = False

        def masked_mean(x, m):
            m3 = m[..., None]
            return (x * m3).sum(1) / np.maximum(m3.sum(1), 1)

        tokens = np.stack(
            [
                np.concatenate([pool_grid(hidden[b, :N_IMG_TOK]), pool_grid(hidden[b, N_IMG_TOK : 2 * N_IMG_TOK])])
                for b in range(hidden.shape[0])
            ]
        ).astype(np.float16)  # [B, 128, 2048] — base then wrist, the 2-view layout
        return (masked_mean(hidden, img_mask), masked_mean(hidden, lang_mask), masked_mean(hidden, mask), tokens)

    dataset = pathlib.Path(args.dataset)
    tasks = {
        json.loads(line)["task_index"]: json.loads(line)["task"]
        for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()
    }
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for episode in args.episodes:
        out_path = out_root / f"episode_{episode:06d}.npz"
        if out_path.exists():
            print(f"skip {out_path}", flush=True)
            continue
        frame = pd.read_parquet(dataset / f"data/chunk-000/episode_{episode:06d}.parquet")
        states = np.stack(frame["state"].to_numpy()).astype(np.float32)
        task_index = frame["task_index"].to_numpy()
        subtask_index = frame["subtask_index"].to_numpy()
        length = len(frame)
        picked = np.unique(np.linspace(0, length - 1, args.frames_per_ep).astype(int))

        readers = {
            name: imageio.get_reader(str(dataset / f"videos/chunk-000/{name}/episode_{episode:06d}.mp4"), "ffmpeg")
            for name in ("image", "wrist_image")
        }
        frames = {name: {} for name in readers}
        wanted = set(picked.tolist())
        for name, reader in readers.items():
            for i, img in enumerate(reader):
                if i in wanted:
                    frames[name][i] = np.asarray(img)
                if i > picked[-1]:
                    break
            reader.close()

        examples = [
            {
                "observation/image": frames["image"][int(t)],
                "observation/wrist_image": frames["wrist_image"][int(t)],
                "observation/state": states[int(t)],
                "prompt": tasks[int(task_index[int(t)])],
            }
            for t in picked
        ]

        chunks = [[], [], [], []]
        for start in range(0, len(examples), args.batch_size):
            for store, value in zip(chunks, embed_batch(examples[start : start + args.batch_size])):
                store.append(value)
        pooled_img, pooled_lang, pooled_all, tokens = (np.concatenate(c) for c in chunks)
        np.savez_compressed(
            out_path,
            tokens=tokens,
            pooled_img=pooled_img,
            pooled_lang=pooled_lang,
            pooled_all=pooled_all,
            frame_idx=picked,
            subtask_index=subtask_index[picked],
            task_index=task_index[picked],
        )
        print(f"wrote {out_path} tokens={tokens.shape}", flush=True)


if __name__ == "__main__":
    main()
