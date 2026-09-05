"""Contiguous-window SEQUENCE dataset + collator for GR00T N1.7 — the `StageQWindowDataset` analogue.

RoboTTT needs contiguous sub-trajectories: the fast weights are rolled W_0 -> W_1 -> ... -> W_{L-1}
over L chunk-steps of ONE episode, in temporal order. GR00T's stock `ShardedSingleStepDataset` is
single-step by construction and, worse, **deliberately shuffles the step indices inside each shard
split** (`sharded_single_step_dataset.py:197`) — so temporal order is actively destroyed. This module
replaces that sharding while keeping the greedy shard-balancing skeleton intact.

WINDOW ENUMERATION is byte-equivalent to the pi side's `contiguous_episode_windows`
(`openpi/groot_utils/groot_openpi_dataset.py:861-882`):

    steps = range(0, effective_episode_length, chunk_stride)
    for start in range(0, len(steps) - window_len + 1, window_len):   # NON-OVERLAPPING
        yield steps[start : start + window_len]

Non-overlapping (start advances by `window_len`, not 1) so each frame is used at most once per epoch.
**Fail-closed at episode ends**: a trailing partial window is DROPPED and an episode shorter than one
window is skipped entirely. There is no padding, because a padded window would either repeat frames
into the recurrence or splice across an episode boundary, and a fast-weight chain that silently
crosses episodes is the one failure this mechanism cannot tolerate.

BATCH LAYOUT. Each item is one window: every processor key gains a leading `L`, so the collator's
stock `else` branch (`np.stack`) yields `[B, L, ...]`. Downstream the action head reshapes to `B*L`
row-major, i.e. flat index `b*L + t` — **window-major**, matching pi's
`swapaxes(0,1).reshape(b*length, -1)`. Get this order wrong and every observation is paired with
another timestep's conditioning, which trains a plausible-looking but meaningless policy.

`vlm_content` is the one key the stock collator cannot stack: it is a dict of `{"text", "images"}`
that the collator flattens across the batch into a single processor call, and it explicitly refuses
pre-tokenized keys (`processing_gr00t_n1d7.py:129-135`). `Gr00tSequenceCollator` below therefore
flattens the per-item LIST of L contents in the same window-major order, producing `input_ids` /
`pixel_values` at batch `B*L`, consistent with every other key.

GR00T-venv-only: every gr00t/torch import is inside a function or a lazily-built class.
"""

from __future__ import annotations

__all__ = [
    "contiguous_episode_windows",
    "install_sequence_dataset",
    "build_sequence_collator",
    "SEQ_KEYS",
]

# Keys this module adds to every item (all survive the collator's else branch).
SEQ_KEYS = ("seq_len", "loss_mask", "reset")


def contiguous_episode_windows(effective_length: int, window_len: int, chunk_stride: int) -> list:
    """Non-overlapping contiguous windows of `window_len` chunk-steps, `chunk_stride` frames apart.

    Pure function, no gr00t import — this is what the equivalence test against the pi enumerator
    exercises. Returns a list of lists of native step indices, each of length exactly `window_len`.
    """
    if window_len < 1:
        raise ValueError(f"window_len must be >= 1, got {window_len}")
    if chunk_stride < 1:
        raise ValueError(f"chunk_stride must be >= 1, got {chunk_stride}")
    steps = list(range(0, int(effective_length), int(chunk_stride)))
    return [steps[start : start + window_len] for start in range(0, len(steps) - window_len + 1, window_len)]


def install_sequence_dataset(window_len: int = 8, chunk_stride: int = 8, extra_datapoint_fn=None) -> None:
    """Monkeypatch the factory's dataset class to emit contiguous windows instead of single steps.

    `extra_datapoint_fn(dataset, episode_data, step_index, out) -> None` lets another mechanism
    (e.g. the omega window) add per-step keys; it is called for every step of every window, so its
    keys are stacked to `[L, ...]` like everything else.
    """
    import gr00t.data.dataset.factory as factory_module
    import numpy as np
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    if window_len < 1 or chunk_stride < 1:
        raise ValueError(f"window_len/chunk_stride must be >= 1, got {window_len}/{chunk_stride}")

    class WindowShardedSequenceDataset(ShardedSingleStepDataset):
        """One item == one contiguous window of `window_len` chunk-steps from a single episode."""

        def shard_dataset(self):
            """Same greedy balancing as the base class, but the unit of work is a WINDOW.

            Deliberately does NOT shuffle within an episode — the base class's `rng.shuffle` on step
            indices is exactly what breaks contiguity. Episode ORDER is still shuffled, and windows
            are still round-robined into `num_splits` sub-sequences, so shard diversity is preserved.
            """
            shuffled = self.rng.permutation(len(self.episode_loader.episode_lengths))
            num_splits = max(1, int(1 / self.episode_sampling_rate))
            assert len(shuffled) > 0, f"No valid trajectories found for dataset {self.dataset_path}"

            episode_splits = []
            total_windows = 0
            skipped = 0
            for ep_idx in shuffled:
                windows = contiguous_episode_windows(
                    self.get_effective_episode_length(ep_idx), window_len, chunk_stride
                )
                if not windows:
                    skipped += 1
                    continue
                total_windows += len(windows)
                order = self.rng.permutation(len(windows))  # shuffle WINDOWS, never within one
                for i in range(num_splits):
                    sel = order[i::num_splits]
                    if len(sel) > 0:
                        episode_splits.append((ep_idx, [windows[j] for j in sel]))

            assert total_windows > 0 and episode_splits, (
                f"No contiguous windows for dataset {self.dataset_path}: every episode is shorter "
                f"than window_len={window_len} x chunk_stride={chunk_stride} "
                f"(= {window_len * chunk_stride} frames, plus action_horizon {self.action_horizon}). "
                "Lower WSM_SEQ_WINDOW_LEN / WSM_SEQ_CHUNK_STRIDE or use longer episodes."
            )

            num_shards = min(int(np.ceil(total_windows / max(1, self.shard_size))), len(episode_splits))
            num_shards = max(1, num_shards)
            sharded_episodes = [[] for _ in range(num_shards)]
            shard_lengths = np.zeros(num_shards, dtype=int)
            for ep_idx, windows in episode_splits:
                s = int(np.argmin(shard_lengths))
                sharded_episodes[s].append((ep_idx, windows))
                shard_lengths[s] += len(windows)

            assert all(shard_lengths[i] > 0 for i in range(num_shards))
            print(
                f"[seq] {num_shards} shards for {self.dataset_path}: {total_windows} windows "
                f"(L={window_len} stride={chunk_stride}), {skipped} episodes too short, "
                f"avg shard {total_windows / num_shards:.1f}",
                flush=True,
            )
            self.sharded_episodes = sharded_episodes
            self.shard_lengths = shard_lengths

        def get_shard(self, idx: int) -> list:
            """Decode each episode ONCE, then build every window from that one DataFrame.

            All L steps of a window share a single `episode_data`, which is what makes contiguity and
            temporal order structural rather than something a later refactor could break.
            """
            items = []
            for ep_idx, windows in self.sharded_episodes[idx]:
                episode_data = self.episode_loader[ep_idx]
                self._seq_cur_ep_idx = ep_idx
                for steps in windows:
                    items.append(self._build_window(episode_data, steps))
            return items

        def _build_window(self, episode_data, steps) -> dict:
            per_step = []
            for t in steps:
                out = super().get_datapoint(episode_data, int(t))
                if extra_datapoint_fn is not None:
                    extra_datapoint_fn(self, episode_data, int(t), out)
                per_step.append(out)

            keys = set(per_step[0])
            for i, d in enumerate(per_step[1:], 1):
                if set(d) != keys:  # ragged windows would stack into a silently wrong batch
                    raise ValueError(
                        f"[seq] step {steps[i]} has keys {sorted(set(d) ^ keys)} that step "
                        f"{steps[0]} does not — refusing to stack a ragged window"
                    )

            window = {}
            for key in keys:
                vals = [d[key] for d in per_step]
                if key == "vlm_content":
                    window[key] = vals  # list of L; the sequence collator flattens it
                elif isinstance(vals[0], (int, float, bool)):
                    window[key] = np.asarray(vals)
                else:
                    window[key] = np.stack([np.asarray(v) for v in vals])
            window["seq_len"] = np.asarray(len(steps), dtype=np.int64)
            # Reserved per-step fields (07a §3.3). Constant for now, but present so a loss-masked
            # variant is a loader change rather than a model change.
            window["loss_mask"] = np.ones(len(steps), dtype=np.float32)
            window["reset"] = np.asarray(
                [i == 0 for i in range(len(steps))], dtype=np.bool_
            )  # true episode-window start
            return window

    factory_module.ShardedSingleStepDataset = WindowShardedSequenceDataset
    print(f"[seq] dataset patched: contiguous windows L={window_len} stride={chunk_stride}", flush=True)


def build_sequence_collator(base_collator):
    """Wrap the stock GR00T collator so a per-item LIST of L `vlm_content`s flattens to `B*L`.

    Order is WINDOW-MAJOR (`for item: for step:`), i.e. flat index `b*L + t`, which is exactly the
    order a `[B, L, ...]` tensor takes under a row-major `reshape(B*L, ...)`. Every other key goes
    through the base collator untouched and arrives as `[B, L, ...]`.
    """
    import torch

    class Gr00tSequenceCollator(base_collator.__class__):
        def __call__(self, features):
            if not features or not isinstance(features[0].get("vlm_content"), list):
                return base_collator(features)  # not a sequence batch; stock behavior

            lens = {len(f["vlm_content"]) for f in features}
            if len(lens) != 1:
                raise ValueError(f"[seq] ragged window lengths in one batch: {sorted(lens)}")
            length = lens.pop()

            # Window-major expansion: one pseudo-sample per (item, step).
            flat = []
            for f in features:
                for t in range(length):
                    flat.append({"vlm_content": f["vlm_content"][t]})
            vlm_batch = base_collator(flat)["inputs"]

            rest = [{k: v for k, v in f.items() if k != "vlm_content"} for f in features]
            batch = base_collator(rest)["inputs"]
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
                if key in vlm_batch:
                    batch[key] = vlm_batch[key]
            batch["seq_window_len"] = torch.tensor(length, dtype=torch.long)
            return {"inputs": batch}

    collator = Gr00tSequenceCollator.__new__(Gr00tSequenceCollator)
    collator.__dict__.update(base_collator.__dict__)
    return collator
