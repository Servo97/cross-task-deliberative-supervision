#!/usr/bin/env python3
"""Cache-parity harness for the online-omega sidecar: does it reproduce the canonical omega?

THE FIDELITY PROOF. The omega the GR00T dnw8 conditioner was trained on came from ONE producer —
the offline chain in ``rmb_chain.sh``: 256 px LeRobot frames (``workspace_models.labels
.extract_frames``) -> frozen pi0.5 tap (``stage_s_cache_features``, ckpt ``tap_149999``) -> frozen
WorkspaceModel over the demo's FULL stride-8 grid (``generate_stage_s_policy_features.encode_demo``,
one ``model.encode`` call) -> ``w.npz``. The sidecar produces omega a completely different way:
INCREMENTALLY, one grid frame at a time, growing a causal prefix. Those two agreeing is not obvious
and it is the only thing standing between a real deltanet eval and a plausible-looking wrong one, so
it is measured rather than argued.

The harness replays cached TRAIN demos through the production ``OmegaSidecar`` + ``PiOmegaProducer``
(not a copy of them) and reports the max absolute difference against the cached ``w`` per grid frame.

TWO STAGES, because they falsify different things:

  --stage encoder   Injects the demo's ALREADY-TAPPED features (``patch_tokens.npy`` +
                    ``feats.npz:lang_per_frame``) as the tap. Isolates encoder + incremental-prefix
                    + causal-window semantics. No jax, no GPU, seconds per demo. If this drifts, the
                    window logic is wrong.
  --stage full      Runs the REAL frozen tap on the demo's raw 256 px frames, i.e. the entire chain
                    end to end. Needs jax (``JAX_PLATFORMS=cpu`` works; it is slow). If `encoder`
                    passes and this drifts, the drift is in image handling / prompt / state.

``--stage full`` MUST be run with ``--tap-image-size 0`` (the default here): the cache was tapped
from native 256 px frames, so any resize would be measuring a different quantity. That is the
opposite of the SERVE default (224, the resolution every sealed pi workspace serve received) and the
difference is deliberate — see ``omega_sidecar.PI_SERVE_IMAGE_SIZE``.

TOLERANCE. The cache stores ``w`` as fp16 while the sidecar returns fp32, so the floor is fp16
quantization of the value itself (~1e-3 relative), never zero. ``--rtol/--atol`` default to that
floor plus room for CPU-vs-GPU kernel differences on the `full` stage; the harness always PRINTS the
observed maxima so a tightening/loosening decision is made on numbers.

  # encoder stage (CPU, no GPU, no jax)
  PYTHONPATH=. python vla_training/eval/omega_sidecar_parity.py --stage encoder \
      --task MemHeatPot --demos 0,1,2 --grid-frames 12 \
      --source-features-root /data/work/rmb/source_features \
      --omega-root /data/work/rmb/omega \
      --encoder-ckpt /data/work/wsm_artifacts/rmb/encoder.pt \
      --task-lang-table /data/work/wsm_artifacts/rmb/task_lang_table.npz \
      --task-prompt-manifest /data/work/rmb/task_prompts.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from vla_training.eval.omega_sidecar import OmegaSidecar, PiOmegaProducer, pin_jax_memory_env

#: The offline grid the cache was built on (rmb_chain.sh: extract_frames --stride 8).
CACHE_STRIDE = 8
#: Frame-view names in the frames npz, keyed by the wire key the sidecar expects.
FRAMES_NPZ_VIEW = {
    "observation/image": "frames_agentview_left",
    "observation/wrist_image": "frames_eye_in_hand",
    "observation/right_image": "frames_agentview_right",
}


class CachedTapReplay:
    """A ``Pi05BackboneTap``-shaped object that returns the demo's ALREADY-CACHED tap output.

    Deliberately positional-cursor based: the sidecar advances exactly one grid frame per call, so
    the cursor and the cache grid stay in lock-step, and a double-call (the bug that would mean the
    grid bookkeeping is wrong) desynchronizes them and shows up as a parity failure rather than
    passing quietly.
    """

    def __init__(self, patch_tokens: np.ndarray, lang_per_frame: np.ndarray):
        self._patch = patch_tokens
        self._lang = lang_per_frame
        self.cursor = 0

    def tap(self, frames, state, prompt):
        """Return the next cached frame, TILED to whatever batch the producer asked for.

        The producer pads its single real frame up to ``tap_batch_size`` rows to land on the cache's
        XLA kernel; with cached features there is no kernel to match, so the padding is answered by
        repeating the row. The cursor still advances exactly ONCE per call, which is what keeps a
        double-advance in the grid bookkeeping detectable rather than silently absorbed.
        """
        from workspace_models.features.pi_backbone_tap import PiTapResult

        if self.cursor >= len(self._patch):
            raise RuntimeError(
                f"[omega-parity] tap called {self.cursor + 1} times for a {len(self._patch)}-frame "
                f"demo: the sidecar advanced the causal grid more often than the cache has frames"
            )
        index, self.cursor = self.cursor, self.cursor + 1
        rows = len(prompt) if isinstance(prompt, (list, tuple)) else 1
        return PiTapResult(
            patch_tokens=np.repeat(np.asarray(self._patch[index])[None], rows, axis=0),
            lang_emb=np.repeat(np.asarray(self._lang[index])[None], rows, axis=0),
        )


def load_canonical_prompt(manifest: Path, task: str) -> str:
    """The canonical terse instruction the cache was tapped with (omega manifest's
    ``conditioning.global_language_mode: canonical_terse_task_instruction``)."""
    payload = json.loads(Path(manifest).expanduser().read_text())
    for entry in payload["tasks"]:
        if entry["task"] == task:
            return entry["prompt"]
    raise SystemExit(f"[omega-parity] task {task!r} is not in {manifest}")


def demo_frames(frames_dir: Path, task: str, episode: int) -> dict:
    """The raw 256 px grid frames the offline tap consumed (``extract_frames`` output)."""
    path = Path(frames_dir).expanduser() / task / f"ep{episode:03d}_frames.npz"
    with np.load(path, allow_pickle=True) as data:
        return {
            "frame_indices": data["frame_indices"].astype(np.int64),
            "prompt": str(data["prompt"]),
            **{wire: np.asarray(data[key]) for wire, key in FRAMES_NPZ_VIEW.items()},
        }


def demo_states(lerobot_root: Path, task: str, episode: int, frame_indices: np.ndarray) -> np.ndarray:
    """The same robot state rows the offline tap consumed.

    Inlined rather than imported from ``stage_s_cache_features._state_at`` (whose module-level
    provenance import needs the launch-script sys.path). Kept byte-identical to it on purpose: the
    parquet path, the stack-and-cast, and the fancy-index by ``frame_indices`` are what determine
    the state the frozen tap saw, and a harness that reads state differently would be measuring a
    different quantity than the cache it is comparing against.
    """
    import pandas as pd

    candidates = sorted(Path(lerobot_root).expanduser().glob(f"{task}/*/lerobot"))
    if not candidates:
        raise SystemExit(f"[omega-parity] no {task}/*/lerobot under {lerobot_root}")
    frame = pd.read_parquet(candidates[0] / f"data/chunk-000/episode_{episode:06d}.parquet")
    state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    return state[frame_indices]


def compare_demo(
    producer: PiOmegaProducer,
    *,
    task: str,
    episode: int,
    prompt: str,
    grid_frames: int,
    cached_w: np.ndarray,
    frame_indices: np.ndarray,
    request_frames: dict | None,
    states: np.ndarray | None,
) -> dict:
    """Replay one demo through the production sidecar; return per-frame diffs vs the cache."""
    sidecar = OmegaSidecar(producer, stride=CACHE_STRIDE, max_envs=1, max_grid_frames=max(grid_frames, 1))
    n = min(int(grid_frames), len(cached_w))
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    diffs, rels = [], []
    for index in range(n):
        step = int(frame_indices[index])
        if step != index * CACHE_STRIDE:
            raise SystemExit(
                f"[omega-parity] {task} demo {episode}: frame_indices[{index}]={step} is not on the "
                f"stride-{CACHE_STRIDE} grid; the cache and the serve grid disagree"
            )
        request = {
            "observation/image": blank,
            "observation/wrist_image": blank,
            "observation/right_image": blank,
            "observation/state": (
                np.zeros(16, np.float32) if states is None else np.asarray(states[index], dtype=np.float32)
            ),
            "wsm_env_id": "parity-w0",
            "wsm_task": task,
            "wsm_demo_episode": int(episode),
            "wsm_t": step,
            "wsm_prompt": prompt,
        }
        if request_frames is not None:
            for wire in FRAMES_NPZ_VIEW:
                request[wire] = np.ascontiguousarray(request_frames[wire][index])
        window = sidecar.omega(request)
        # The window's LAST row is omega at this grid frame; the earlier rows are older frames this
        # loop already checked, so checking the newest row at every step covers the whole window.
        online = np.asarray(window[-1], dtype=np.float32)
        reference = np.asarray(cached_w[index], dtype=np.float32)
        diff = float(np.max(np.abs(online - reference)))
        scale = float(np.max(np.abs(reference))) or 1.0
        diffs.append(diff)
        rels.append(diff / scale)
        # The full window must also equal the cache's own causal window over the same prefix: this
        # is what the GR00T train loader (wsm_align.window_at) hands the conditioner.
        from workspace_models.features.wsm_align import window_at

        expected_window = window_at(
            np.asarray(cached_w[: index + 1], dtype=np.float32),
            frame_indices[: index + 1],
            step,
            producer.k,
        )
        window_diff = float(np.max(np.abs(np.asarray(window, np.float32) - expected_window)))
        diffs[-1] = max(diffs[-1], window_diff)
    return {
        "task": task,
        "episode": int(episode),
        "grid_frames_checked": n,
        "max_abs_diff": max(diffs) if diffs else float("nan"),
        "max_rel_diff": max(rels) if rels else float("nan"),
        "per_frame_max_abs": [round(d, 8) for d in diffs],
        "cached_w_absmax": float(np.max(np.abs(np.asarray(cached_w[:n], np.float32)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("encoder", "full"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--demos", default="0,1,2", help="comma-separated episode indices")
    parser.add_argument(
        "--grid-frames",
        type=int,
        default=12,
        help="how many stride-8 grid frames (== policy chunks) to check per demo",
    )
    parser.add_argument("--source-features-root", required=True)
    parser.add_argument("--omega-root", required=True)
    parser.add_argument("--encoder-ckpt", required=True)
    parser.add_argument("--task-lang-table", required=True)
    parser.add_argument("--task-prompt-manifest", required=True)
    parser.add_argument("--expect-encoder-sha256", default=None)
    parser.add_argument("--k-window", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    # full stage only
    parser.add_argument("--frames-dir", default=None, help="extract_frames output (full stage)")
    parser.add_argument("--lerobot-root", default=None, help="<root>/<task>/*/lerobot (full stage)")
    parser.add_argument("--tap-ckpt", default=None)
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument(
        "--tap-batch-size",
        type=int,
        default=None,
        help="rows the frozen tap is called with. Defaults to the sidecar's own "
        "TAP_BATCH_SIZE, which is what makes the tap land on the same XLA "
        "kernel the B=32 cache build used. Pass 1 to MEASURE the small-batch "
        "offset instead of eliminating it.",
    )
    parser.add_argument(
        "--tap-image-size",
        type=int,
        default=0,
        help="0 (default) = feed the cache's native 256 px frames untouched, which "
        "is what the offline pipeline did. Do not change for a parity run.",
    )
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--out", default=None, help="write the report json here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # SAME pin as the sidecar, from the SAME function, before any jax import. The gate that failed
    # on 2026-08-08 differed from the passing run in exactly this and nothing else.
    logging.info("jax memory policy: %s", json.dumps(pin_jax_memory_env(), sort_keys=True))

    prompt = load_canonical_prompt(Path(args.task_prompt_manifest), args.task)
    episodes = [int(x) for x in args.demos.split(",") if x.strip()]
    source_root = Path(args.source_features_root).expanduser()
    omega_root = Path(args.omega_root).expanduser()

    reports = []
    for episode in episodes:
        demo = f"demo_{episode:06d}"
        with np.load(omega_root / args.task / demo / "w.npz") as cache:
            cached_w, frame_indices = cache["w"], cache["frame_indices"].astype(np.int64)
        request_frames = states = None
        if args.stage == "encoder":
            patch = np.load(source_root / args.task / demo / "patch_tokens.npy", mmap_mode="r")
            with np.load(source_root / args.task / demo / "feats.npz") as feats:
                lang_per_frame = feats["lang_per_frame"]
                cached_indices = feats["frame_indices"].astype(np.int64)
            if not np.array_equal(cached_indices, frame_indices):
                raise SystemExit(f"[omega-parity] {demo}: feats/omega frame_indices disagree")
            tap = CachedTapReplay(patch, lang_per_frame)
        else:
            if not (args.frames_dir and args.lerobot_root and args.tap_ckpt):
                raise SystemExit("--stage full needs --frames-dir, --lerobot-root and --tap-ckpt")
            request_frames = demo_frames(Path(args.frames_dir), args.task, episode)
            if not np.array_equal(request_frames["frame_indices"], frame_indices):
                raise SystemExit(f"[omega-parity] {demo}: frames/omega frame_indices disagree")
            # NOT a check: the frames npz records the DEMO's own expanded instruction, which the
            # offline tap deliberately did not use — `stage_s_cache_features.main` passes
            # `prompt_map[task]`, the canonical terse manifest string, into `cache_task` (:365).
            # Serve does the same (the runner's --task-prompt-manifest -> wsm_prompt). Logged so a
            # reader can see the two strings differ ON PURPOSE.
            logging.info(
                "%s: tapping with the CANONICAL prompt %r (demo string %r is unused)",
                demo,
                prompt,
                request_frames["prompt"],
            )
            states = demo_states(Path(args.lerobot_root), args.task, episode, frame_indices)
            tap = None

        producer = PiOmegaProducer(
            tap_ckpt=args.tap_ckpt or "",
            encoder_ckpt=args.encoder_ckpt,
            task_lang_table=args.task_lang_table,
            k_window=args.k_window,
            stride=CACHE_STRIDE,
            device=args.device,
            configs_dir=args.configs_dir,
            tap_image_size=args.tap_image_size,
            **({} if args.tap_batch_size is None else {"tap_batch_size": args.tap_batch_size}),
            expect_encoder_sha256=args.expect_encoder_sha256,
            tap=tap,
        )
        report = compare_demo(
            producer,
            task=args.task,
            episode=episode,
            prompt=prompt,
            grid_frames=args.grid_frames,
            cached_w=cached_w,
            frame_indices=frame_indices,
            request_frames=request_frames,
            states=states,
        )
        report["stage"] = args.stage
        reports.append(report)
        logging.info(
            "%s demo %d: max_abs=%.3e max_rel=%.3e over %d grid frames",
            args.task,
            episode,
            report["max_abs_diff"],
            report["max_rel_diff"],
            report["grid_frames_checked"],
        )

    worst_abs = max(r["max_abs_diff"] for r in reports)
    worst_rel = max(r["max_rel_diff"] for r in reports)
    scale = max(r["cached_w_absmax"] for r in reports)
    summary = {
        "stage": args.stage,
        "task": args.task,
        "demos": episodes,
        "k_window": args.k_window,
        "tap_image_size": args.tap_image_size,
        "tap_batch_size": producer.tap_batch_size,
        "encoder_sha256": producer.encoder_sha256,
        "max_abs_diff": worst_abs,
        "max_rel_diff": worst_rel,
        "cached_w_absmax": scale,
        "rtol": args.rtol,
        "atol": args.atol,
        "reports": reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(summary, indent=2, sort_keys=True))
    tolerance = args.atol + args.rtol * scale
    if not (worst_abs <= tolerance):
        raise SystemExit(
            f"[omega-parity] FAIL: max|online - cached| = {worst_abs:.3e} > {tolerance:.3e} "
            f"(atol {args.atol} + rtol {args.rtol} * |w|max {scale:.3f}). The sidecar does NOT "
            f"reproduce the omega the conditioner was trained on; do not run the dnw8 cells."
        )
    print(f"[omega-parity] PASS stage={args.stage}: max|online - cached| = {worst_abs:.3e} <= {tolerance:.3e}")


if __name__ == "__main__":
    main()
