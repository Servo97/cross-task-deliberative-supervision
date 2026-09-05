"""WSM-base data: cached frozen features + salient labels + cross-demo partners -> training batches.

Reads manifest.parquet (features/build_pairs). Per demo: mmap patch_tokens.npy + feats.npz + the label
npz (salient_global = next-keyframe targets, cumulative_global = cumulative targets). Builds per-frame
language conditioning (the active subgoal embedding, dropout -> global expanded prompt) and, for the
cross-demo alignment objective, a sampled same-task partner. Recon targets are gathered from
patch_tokens[keyframe_pos, global_patch_id] (frozen-VLM feature space). No VLM/sim in the loop.

Hot path: the per-demo work (mmap read + cond_lang + keyframe-target gather) runs INSIDE DataLoader
workers via WSMSampleDataset; wsm_collate right-pads + stacks into a pinnable batch. Tensors stay fp16
through the loader + H2D transfer and are cast to fp32 on the GPU (halves PCIe bandwidth; the heavy
tensor is patch [B,T,192,2048]). The trainer keeps the fp32 forward + Hungarian match unchanged.
Governed by internal_planning_and_todos/07_wsm_preprocessing_and_revised_plan.md + 09 (dataloader plan).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Demo:
    patch: np.ndarray  # [F,192,2048] fp16 (mmap)
    state_emb: np.ndarray  # [F,1536] fp16
    lang_global: np.ndarray  # [2048] fp16 (expanded-prompt embedding)
    subgoal_embs: np.ndarray  # [n_sub,2048] fp16
    frame_indices: np.ndarray  # [F] int64 (episode frame idx of each cached frame)
    keyframes: np.ndarray  # [K] int64 (episode frame idx of subgoal completions)
    salient_next: list  # [K] arrays of GLOBAL patch ids (0..191) — next-keyframe targets
    salient_cumul: list  # [K] arrays — cumulative targets
    task: str
    demo_id: int

    @property
    def F(self) -> int:
        return self.patch.shape[0]

    def kf_positions(self) -> np.ndarray:
        """Position in F (0..F-1) of each keyframe (nearest cached frame)."""
        return np.array(
            [int(np.argmin(np.abs(self.frame_indices - int(kf)))) for kf in self.keyframes], dtype=np.int64
        )

    def per_frame_subgoal_idx(self) -> np.ndarray:
        """For each cached frame, the active subgoal index (the subgoal it is progressing toward =
        the first keyframe >= this frame); clamped to the last subgoal."""
        n = len(self.keyframes)
        if n == 0:
            return np.zeros(self.F, dtype=np.int64)
        idx = np.searchsorted(self.keyframes, self.frame_indices, side="left")
        return np.clip(idx, 0, n - 1)


def load_demo(row: dict) -> Demo:
    fd = Path(row["feature_dir"])
    patch = np.load(fd / "patch_tokens.npy", mmap_mode="r")
    f = np.load(fd / "feats.npz", allow_pickle=True)
    lab = np.load(row["labels_path"], allow_pickle=True)
    return Demo(
        patch=patch,
        state_emb=f["state_emb"],
        lang_global=f["lang_global"],
        subgoal_embs=f["subgoal_embs"],
        frame_indices=f["frame_indices"],
        keyframes=lab["keyframes"],
        salient_next=list(lab["salient_global"]),
        salient_cumul=list(lab["cumulative_global"]),
        task=str(row["task"]),
        demo_id=int(row["demo_id"]),
    )


def load_demo_pi(row: dict) -> Demo:
    """pi0.5 variant: pi discretizes robot state into the prompt (no separate state_emb), so the
    per-frame language embedding `lang_per_frame` [F,2048] fills the encoder's proprio slot (see
    pi_cache_features.py). Labels are the pi-geometry vlm_episode_pi_*.npz (build_pairs --label-suffix _pi)."""
    fd = Path(row["feature_dir"])
    patch = np.load(fd / "patch_tokens.npy", mmap_mode="r")
    f = np.load(fd / "feats.npz", allow_pickle=True)
    lab = np.load(row["labels_path"], allow_pickle=True)
    return Demo(
        patch=patch,
        state_emb=f["lang_per_frame"],
        lang_global=f["lang_global"],
        subgoal_embs=f["subgoal_embs"],
        frame_indices=f["frame_indices"],
        keyframes=lab["keyframes"],
        salient_next=list(lab["salient_global"]),
        salient_cumul=list(lab["cumulative_global"]),
        task=str(row["task"]),
        demo_id=int(row["demo_id"]),
    )


def load_demo_pi_stage_s(row: dict, table: dict) -> Demo:
    """Stage-S pi loader: canonical feats.npz{lang_per_frame, frame_indices} + the SEALED per-task
    language vector (never a per-demo/expanded prompt). `table` maps task -> [lang_dim] fp16.

    The encoder proprio slot is fed `lang_per_frame` (pi bakes robot state into the prompt); the
    modulator/global-language slot is the packet-01 task-language table row. subgoal_embs is empty
    (Stage-S trains at subgoal-dropout 1.0 = global language only). Salient targets come from the
    pi-geometry label npz declared in the pairs manifest."""
    fd = Path(row["feature_dir"])
    patch = np.load(fd / "patch_tokens.npy", mmap_mode="r")
    f = np.load(fd / "feats.npz", allow_pickle=False)
    if set(f.files) != {"lang_per_frame", "frame_indices"}:
        raise ValueError(
            f"Stage-S feats.npz for {fd} must contain exactly {{lang_per_frame, frame_indices}}; "
            f"got {sorted(f.files)} — this looks like a legacy expanded-prompt cache"
        )
    lab = np.load(row["labels_path"], allow_pickle=True)
    task = str(row["task"])
    task_lang = np.asarray(table[task], dtype=np.float16)
    return Demo(
        patch=patch,
        state_emb=f["lang_per_frame"],
        lang_global=task_lang,
        subgoal_embs=np.zeros((0, task_lang.shape[-1]), np.float16),
        frame_indices=f["frame_indices"],
        keyframes=lab["keyframes"],
        salient_next=list(lab["salient_global"]),
        salient_cumul=list(lab["cumulative_global"]),
        task=task,
        demo_id=int(row["demo_id"]),
    )


def cond_lang(demo: Demo, rng: np.random.Generator, dropout: float) -> np.ndarray:
    """Per-frame language condition [F,2048] fp32: the active subgoal emb, with `dropout` prob -> global."""
    F, D = demo.F, demo.lang_global.shape[-1]
    out = np.broadcast_to(demo.lang_global, (F, D)).astype(np.float32).copy()
    if len(demo.subgoal_embs):
        sidx = demo.per_frame_subgoal_idx()  # [F] active subgoal per frame
        use = (rng.random(F) >= dropout) & (sidx < len(demo.subgoal_embs))
        out[use] = np.asarray(demo.subgoal_embs[sidx[use]], dtype=np.float32)
    return out


def _gather_targets(demo: Demo, mode: str) -> list[tuple[int, torch.Tensor]]:
    """Per keyframe with >=1 salient patch: (position-in-F, target frozen-VLM features [m,2048] fp16)."""
    kf_pos = demo.kf_positions()
    sets = demo.salient_next if mode == "next" else demo.salient_cumul
    out = []
    for k, pos in enumerate(kf_pos):
        ids = np.asarray(sets[k], dtype=np.int64)
        if len(ids) == 0:
            continue
        feats = np.asarray(demo.patch[int(pos)])[ids]  # [m,2048] fp16 frozen-VLM features
        out.append((int(pos), torch.from_numpy(np.ascontiguousarray(feats))))
    return out


# --------------------------------------------------------------------------------------------------
# Async pipeline: per-demo CPU work in DataLoader workers -> pinnable padded batch.
# --------------------------------------------------------------------------------------------------
def _sample_demo(demo: Demo, rng: np.random.Generator, dropout: float, mode: str) -> dict:
    """The per-demo CPU bundle (runs in a worker). fp16 throughout; cast to fp32 on the GPU."""
    return {
        "patch": torch.from_numpy(np.array(demo.patch)),  # [F,P,Db] fp16 (mmap->RAM copy)
        "state": torch.from_numpy(np.ascontiguousarray(demo.state_emb)),  # [F,Dp]  fp16
        "lang": torch.from_numpy(cond_lang(demo, rng, dropout)).to(torch.float16),  # [F,Dl] fp16
        "F": int(demo.F),
        "sup": _gather_targets(demo, mode),  # [(pos, feats fp16)]
        "kf_pos": torch.from_numpy(demo.kf_positions()),  # [K] int64
    }


class WSMSampleDataset(torch.utils.data.Dataset):
    """One item == one demo's processed bundle (+ a same-task partner when align). All disk reads and
    numpy work happen here, in worker processes; the trainer just consumes collated GPU-ready batches.

    `load_fn` is injectable so the pi0.5 trainer (different cache schema) can reuse this machinery."""

    def __init__(
        self,
        rows: list[dict],
        *,
        dropout: float,
        mode: str = "next",
        align: bool = False,
        seed: int = 0,
        load_fn=load_demo,
        lut_rows: list[dict] | None = None,
        strict_split: bool = False,
    ):
        self.rows = rows
        self.dropout = dropout
        self.mode = mode
        self.align = align
        self.seed = seed
        self.load_fn = load_fn
        self.strict_split = strict_split
        self.n_self_partner = 0
        # Legacy: partner lookup spans lut_rows (default the full manifest), so a sampled partner is
        # always resolvable. Stage-S strict mode: lut_rows MUST be given (this split's rows only) and
        # each row's partner candidates are filtered to that split at __init__ (structural, not
        # rng-dependent), so training can never draw a validation partner and vice versa.
        if align:
            lut_source = lut_rows if lut_rows is not None else rows
            self.lut = {(r["task"], int(r["demo_id"])): r for r in lut_source}
            if strict_split:
                if lut_rows is None:
                    raise ValueError("strict_split=True requires lut_rows (this split's rows only)")
                self._candidates = self._filter_candidates(rows)
        else:
            self.lut = None
        self.rng = np.random.default_rng(seed)  # replaced per-worker by worker_init_fn

    def _filter_candidates(self, rows: list[dict]) -> dict:
        """Per (task, demo_id): the partner_demo_ids intersected with THIS split's lut; empty -> []."""
        import json

        out: dict[tuple[str, int], list[int]] = {}
        for r in rows:
            pids = json.loads(r["partner_demo_ids"]) or []
            keep = [int(p) for p in pids if (r["task"], int(p)) in self.lut]
            out[(r["task"], int(r["demo_id"]))] = keep
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def _partner_row(self, row: dict) -> dict:
        import json

        if self.strict_split:
            keep = self._candidates.get((row["task"], int(row["demo_id"])), [])
            if not keep:
                self.n_self_partner += 1
                return row
            return self.lut[(row["task"], int(self.rng.choice(keep)))]
        pids = json.loads(row["partner_demo_ids"]) or [row["demo_id"]]
        return self.lut.get((row["task"], int(self.rng.choice(pids))), row)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        item = _sample_demo(self.load_fn(row), self.rng, self.dropout, self.mode)
        if self.align:
            p = _sample_demo(self.load_fn(self._partner_row(row)), self.rng, self.dropout, self.mode)
            item["partner"] = {
                "patch": p["patch"],
                "state": p["state"],
                "lang": p["lang"],
                "F": p["F"],
                "kf_pos": p["kf_pos"],
            }
        else:
            item["partner"] = None
        return item


def worker_init_fn(worker_id: int) -> None:
    """Give each worker an independent, run-reproducible rng for cond_lang dropout + partner choice."""
    info = torch.utils.data.get_worker_info()
    dataset = info.dataset
    while not hasattr(dataset, "seed") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset  # unwrap Subset-style views to the WSMSampleDataset underneath
    dataset.rng = np.random.default_rng(dataset.seed * 100003 + worker_id)


def _pad_stack(items: list[dict], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad each item[key] ([F, ...] fp16) to T=max(F) and stack -> ([B,T,...], t_valid [B])."""
    B = len(items)
    T = max(it["F"] for it in items)
    tail = items[0][key].shape[1:]
    out = torch.zeros((B, T, *tail), dtype=items[0][key].dtype)
    for i, it in enumerate(items):
        out[i, : it["F"]] = it[key]
    t_valid = torch.tensor([it["F"] for it in items], dtype=torch.int64)
    return out, t_valid


def wsm_collate(items: list[dict]) -> dict:
    """Pad+stack a list of per-demo bundles into one batch (fp16 tensors; pinnable by the DataLoader).
    Variable-length supervision (`sup`) and keyframe positions stay as per-sample Python lists."""
    patch, t_valid = _pad_stack(items, "patch")
    state, _ = _pad_stack(items, "state")
    lang, _ = _pad_stack(items, "lang")
    batch = {
        "patch": patch,
        "state": state,
        "lang": lang,
        "t_valid": t_valid,
        "sup": [it["sup"] for it in items],  # list[B] of [(pos, feats)]
        "kf_pos": [it["kf_pos"] for it in items],  # list[B] of [K] int64
    }
    if items[0]["partner"] is not None:
        parts = [it["partner"] for it in items]
        batch["patch_p"], batch["t_valid_p"] = _pad_stack(parts, "patch")
        batch["state_p"], _ = _pad_stack(parts, "state")
        batch["lang_p"], _ = _pad_stack(parts, "lang")
        batch["kf_pos_p"] = [p["kf_pos"] for p in parts]
    return batch
