#!/usr/bin/env python3
"""B4 - precompute the FROZEN pi0.5 text-tower embedding of every caption segment.

WHY PRECOMPUTE. The R3/R4 language head aligns `w_t` to the text embedding of the segment the frame
is in. That target must be (a) frozen, (b) deterministic, and (c) free at train time. Embedding
on-node per step would put a Gemma-2B text forward inside the training loop for a target that never
changes - so this runs ONCE, offline, and ships as a content-addressed asset exactly like the
keypatch labels.

WHICH TOWER, AND WHY NOT `feats.npz`. The target is the pi0.5 LLM's own contextual encoding of the
caption, taken from **the study pretrain checkpoint** - the identical init every H13 arm starts from,
so the target is a fixed property of the study rather than of any trained arm. The `subgoal_embs`
already sitting in `feats.npz` are a TRAP and are deliberately not used: `pi_cache_features.py:92`
produces them via `tap.tap(frames, state, name)`, i.e. conditioned on the keyframe IMAGE and robot
state. Aligning `w` to those would make the "language" aux part image-reconstruction - predicting a
future frame's visual content from the policy's own features - and they are tied to a different
frozen checkpoint besides.

ACTIVE SEGMENT SEMANTICS. Segment k covers `[t0_k, t1_k)` and those intervals are read verbatim from
the caption JSON. They are NOT recomputed with `searchsorted(keyframes, ...)`: 4.08% of episodes in
the pinned label store carry genuinely UNSORTED `keyframes` (see aug_12/h13_build_status.md 5e), on
which `np.searchsorted` silently returns a wrong index. The stored intervals are monotonic and cover
`[0, n_frames)` in every episode, and this script asserts exactly that per episode.

OUTPUT (one npz per episode, mirroring the keypatch store's layout so the loader stays trivial):
    <out>/<Task>/ep_%06d.capemb.npz
        seg_t0   [K] int64      segment start (inclusive)
        seg_t1   [K] int64      segment end (exclusive)
        emb      [K, D] float16 frozen text-tower embedding, L2-normalised in float32 then cast
        n_frames ()  int64
Plus <out>/_manifest.json carrying the content address and full provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CAPTION_GLOB = "ep_*.captions.json"


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_episode(path: Path) -> dict:
    """Read one caption file and enforce the segment invariants the loader will rely on."""
    record = json.loads(path.read_text())
    segments = record["segments"]
    n_frames = int(record["n_frames"])
    if not segments:
        raise ValueError(f"{path}: no segments")
    t0 = np.asarray([int(s["t0"]) for s in segments], dtype=np.int64)
    t1 = np.asarray([int(s["t1"]) for s in segments], dtype=np.int64)
    # The invariants the train-time lookup depends on. Checked here, once, offline - rather than
    # trusting them per sample in a dataloader worker.
    if t0[0] != 0:
        raise ValueError(f"{path}: first segment starts at {t0[0]}, not 0")
    if t1[-1] != n_frames:
        raise ValueError(f"{path}: last segment ends at {t1[-1]}, not n_frames={n_frames}")
    if not np.all(t1 > t0):
        raise ValueError(f"{path}: empty segment")
    if not np.array_equal(t0[1:], t1[:-1]):
        raise ValueError(f"{path}: segments are not contiguous (gap or overlap)")
    return {
        "episode_id": int(record["episode_id"]),
        "task": str(record["task"]),
        "n_frames": n_frames,
        "t0": t0,
        "t1": t1,
        "texts": [str(s["text"]) for s in segments],
        "prompt_sha": str(record.get("prompt_sha", "")),
        "model": str(record.get("model", "")),
    }


def embed_texts(texts: list[str], *, params_path: str, tokenizer_path: str, max_len: int, batch: int):
    """Unique caption strings -> L2-normalised frozen text-tower embeddings [U, D] float32.

    The captions are run through the pi0.5 prefix LLM as a text-only sequence (no image tokens) and
    mean-pooled over the real tokens. Deduplicated first: identical caption strings must receive
    byte-identical targets, and a large fraction of segments repeat.
    """
    import jax
    import jax.numpy as jnp
    import sentencepiece
    from flax import nnx
    from openpi.models import model as _model
    from openpi.models import pi0 as _pi0
    from openpi.models.pi0_config import Pi0Config

    # openpi's PaligemmaTokenizer downloads the model from GCS; bind the SHA-PINNED local copy
    # instead so the target embeddings are provably a function of the sealed tokenizer artifact.
    # The encoding below reproduces `PaligemmaTokenizer.tokenize`'s no-state branch EXACTLY (same
    # text cleaning, same BOS, same trailing "\n" start-of-answer token, same pad/truncate + mask).
    # The pi05 with-state format is deliberately NOT used: it would splice discretised robot state
    # into a text-only target.
    with open(tokenizer_path, "rb") as handle:
        sp = sentencepiece.SentencePieceProcessor(model_proto=handle.read())

    def tokenize(text: str):
        cleaned = text.strip().replace("_", " ").replace("\n", " ")
        tokens = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
        if len(tokens) < max_len:
            padding = [False] * (max_len - len(tokens))
            mask = [True] * len(tokens) + padding
            tokens = tokens + padding
        else:
            tokens = tokens[:max_len]
            mask = [True] * max_len
        return np.asarray(tokens), np.asarray(mask)

    config = Pi0Config(pi05=True, action_horizon=50, max_token_len=max_len)
    model = nnx.eval_shape(lambda: config.create(jax.random.key(0)))
    graphdef, state = nnx.split(model)
    # bf16, for two reasons: it halves the 12.4 GB param footprint (an fp32 host copy plus an
    # fp32 device copy OOMs a 32 GB card), and it is the dtype the policy's own text tower
    # actually computes in (Pi0Config.dtype is bfloat16), so the target matches what the model
    # would produce internally. The pooled output is accumulated and stored in float32.
    params = _model.restore_params(params_path, restore_type=np.ndarray, dtype=jnp.bfloat16)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)

    # nnx.jit, not jax.jit: `encode` uses the nnx Module, and closing an nnx Module over a bare
    # jax.jit traces its graph incorrectly (observed: TracerArrayConversionError on the token array).
    # The documented idiom threads the module through as an argument.
    @nnx.jit
    def encode(model, tokens, mask):
        # Text-only prefix: embed the tokens, run the LLM with a full (non-causal) mask over the
        # real tokens, and mean-pool the contextual hidden states. Same LLM call
        # `tap_prefix_hidden` makes, minus the image tokens.
        embedded = model.PaliGemma.llm(tokens, method="embed")
        ar_mask = jnp.zeros((tokens.shape[1],), dtype=jnp.bool_)  # full attention within the text
        attn = _pi0.make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        (hidden, _), _ = model.PaliGemma.llm([embedded, None], mask=attn, positions=positions)
        weights = mask[..., None].astype(jnp.float32)
        pooled = (hidden.astype(jnp.float32) * weights).sum(1) / jnp.maximum(weights.sum(1), 1.0)
        return pooled / (jnp.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-8)

    out = None
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        toks, masks = [], []
        for text in chunk:
            token, mask = tokenize(text)
            toks.append(np.asarray(token))
            masks.append(np.asarray(mask))
        pooled = np.asarray(encode(model, jnp.asarray(np.stack(toks)), jnp.asarray(np.stack(masks))))
        if out is None:
            out = np.zeros((len(texts), pooled.shape[-1]), dtype=np.float32)
        out[start : start + len(chunk)] = pooled
        print(f"[capemb] embedded {min(start + batch, len(texts))}/{len(texts)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions-root", required=True)
    ap.add_argument("--params", required=True, help="frozen pi0.5 pretrain <step>/params")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=48, help="caption token budget (captions are <=30 tokens)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--init-inventory-sha256", default="", help="provenance: the init ckpt's pinned sha")
    args = ap.parse_args()

    root = Path(args.captions_root)
    tasks = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))
    episodes = []
    for task in tasks:
        for path in sorted((root / task).glob(CAPTION_GLOB)):
            episodes.append(load_episode(path))
    print(f"[capemb] {len(episodes)} episodes across {len(tasks)} tasks; segment invariants OK", flush=True)

    unique = sorted({text for ep in episodes for text in ep["texts"]})
    index = {text: i for i, text in enumerate(unique)}
    print(f"[capemb] {sum(len(e['texts']) for e in episodes)} segments -> {len(unique)} unique captions", flush=True)

    table = embed_texts(
        unique, params_path=args.params, tokenizer_path=args.tokenizer, max_len=args.max_len, batch=args.batch
    )

    out = Path(args.out)
    digest = hashlib.sha256()
    written = 0
    for ep in episodes:
        rows = table[[index[t] for t in ep["texts"]]].astype(np.float16)
        dest = out / ep["task"]
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(
            dest / f"ep_{ep['episode_id']:06d}.capemb.npz",
            seg_t0=ep["t0"],
            seg_t1=ep["t1"],
            emb=rows,
            n_frames=np.asarray(ep["n_frames"], dtype=np.int64),
        )
        digest.update(f"{ep['task']}/{ep['episode_id']}".encode())
        digest.update(ep["t0"].tobytes())
        digest.update(ep["t1"].tobytes())
        digest.update(rows.tobytes())
        written += 1
    content_sha = digest.hexdigest()

    manifest = {
        "schema_version": 1,
        "kind": "h13_caption_text_embeddings",
        "episodes": written,
        "tasks": len(tasks),
        "unique_captions": len(unique),
        "embedding_dim": int(table.shape[1]),
        "pooling": "masked mean over prefix LLM hidden states, then L2 normalise",
        "text_tower": "pi0.5 PaliGemma LLM (prefix), text-only sequence, no image tokens",
        "frozen_init_inventory_sha256": args.init_inventory_sha256,
        "tokenizer_sha256": hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest(),
        "caption_model": episodes[0]["model"] if episodes else "",
        "caption_prompt_sha": episodes[0]["prompt_sha"] if episodes else "",
        "not_used": "feats.npz subgoal_embs (image-conditioned; see module docstring)",
        "content_sha256": content_sha,
    }
    (out / "_manifest.json").write_text(_canonical(manifest) + "\n")
    print(f"[capemb] wrote {written} episodes -> {out}")
    print(f"[capemb] content_sha256={content_sha}")


if __name__ == "__main__":
    main()
