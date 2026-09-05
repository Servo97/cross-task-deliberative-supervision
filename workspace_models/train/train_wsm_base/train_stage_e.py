#!/usr/bin/env python3
"""H14 Stage-E — cross-domain deliberatively-supervised ω encoder (plan §4, amendments A1-A4).

    L = jepa(EMA, k) + λ_sig·sigreg(rank-capped) + λ_ep·supcon_episode + λ_del·supcon_deliberative

`supcon_deliberative` operates on FRAMES carrying their segment's Qwen edge labels (A2 — a w16 GDN
window spans ~1.1 segments, so a mean-pooled z_seg is invariant to the content the read actually
consumes). Positives = EQUIVALENT ∪ ANALOGOUS, hard negatives = CONTRAST upweighted inside the
denominator, everything confidence-weighted, low-confidence excluded by default.

Batches are composed EDGE-FIRST, not anchor-first (coordinator ruling 2026-08-28 off the pass-2 QA:
28.5% of anchors have NO cross-task positive and 2% have no positive at all, so an anchor-first
sampler would spend a quarter of the corpus on negatives only). Each step samples edges, then the
episodes those edges live in, then fills to the batch size with domain balance (pin D5).

Cells (A4 funnel), all selected by ONE `--cell` field:

    E1            full objective, all loaded domains, Qwen edges
    ctrl-0        λ_del = 0            — same corpus/objective otherwise  (E1-ctrl-0 attribution)
    ctrl-1D       RoboCasa only, full objective                           (domain mixing)
    ctrl-0-seed2  ctrl-0 at the second seed                               (paired spread)
    ctrl-E        positives = top-k descriptor-embedding neighbours, NO Qwen (A1b: the real
                  "is the deliberation worth it" cell, because EQUIVALENT ⊂ embedding-nearest
                  BY CONSTRUCTION of the mining step)
    ctrl-S        type-preserving SHUFFLED edges                          (structure vs regularisation)
    ctrl-T        same-task positives, NO Qwen                            (deliberation vs trivial pairing)
    E1-analog05   E1 with ANALOGOUS positives at weight 0.5               (sensitivity)
    E1-seed2      E1 at a second seed                                     (run-to-run spread)
    E1-noCONTRAST E1 with CONTRAST demoted to an ordinary negative        (do the A9-contested
                  CONTRAST edges carry signal? positives and seed unchanged)
    E1b           full objective on label artifact v2 (binding-aware CONTRAST)
    ctrl-0b       v2 corpus, lambda_del = 0                                (v2's own control)
    E1b-bindingOnly  v2 with hard negatives = binding-corroborated only    (is the binding table,
                  alone, the whole of the usable CONTRAST signal?)

Gates are computed in the eval step and written as JSON, pre-registered before the run:
  * per-domain G1b validity predicate — the `scripts/robocerebra/g1b_verdict.py` bar VERBATIM, with
    a frozen (untrained) encoder negative control that must trip every FAIL trigger;
  * a FRAME-LEVEL cross-task retrieval gate scored ONLY on the A1d DISAGREEMENT subset (pairs where
    the Qwen verdict disagrees with descriptor-cosine ranking) against the chance baseline for its
    own candidate count — G4-class: a term that never beats chance is a HOLD (H13's degenerate aux
    cost 5pp with pristine flow curves);
  * keyframe-patch decode grounding where a label store exists (RoboCasa pi geometry).

A3 (domain bridge), decided by MEASUREMENT on 2026-08-28 and recorded here because it changes what
the funnel can attribute. The adapters below are built and exercised, but only ONE domain's frozen
tap is loadable in this execution:

  robocasa    pooled `p.npz` [F,512], 13 tasks x 150 demos = 1,950 episodes           AVAILABLE
  remembench  no pooled and no raw token store exists locally or in any published S3
              prefix; producing one needs a pi05 checkpoint carrying assets/norm_stats.json
              (only bare `params/` is on disk) plus a frames+subgoals tree that does not
              exist, through ~61 GB of intermediate patch_tokens.npy                   UNAVAILABLE
  robomme     its tap is a DIFFERENT frozen network in a different schema (official SigLIP,
              2 views, 64 tokens x 2048 vs our 3-view 192); no local or published token
              store; the upstream preprocessed cache is not downloaded                 UNAVAILABLE

A3's rule ("if the stats are irreconcilable, RoboMME drops out of the joint encoder and stays in the
deliberation corpus") is therefore applied to RoboMME on a STRONGER ground than statistics — its
tokens are a different encoder world, not a rescalable version of ours — and to ReMemBench on
availability. All three domains remain in the deliberation corpus and in the edge-label artifact;
the trainer filters edges to the loaded taps, so the objective silently and correctly becomes
cross-TASK within RoboCasa (13 tasks), which is what C1/C2 actually rest on. The consequence for
the funnel is explicit: ctrl-1D is definitionally identical to E1 under a single-domain corpus, so
the domain-mixing attribution is UNAVAILABLE here and is deferred to a venue that can produce the
other two taps.

Perf: the whole pooled corpus is uploaded to the GPU once as one flat fp16 tensor and never leaves;
there is no per-step host copy, no DataLoader, and the edge tables are GPU-resident CSR.

    PYTHONPATH=. python workspace_models/train/train_wsm_base/train_stage_e.py \
        --labels ~/Research/TRI/wsm_data/deliberation/stage_e_labels/<label_id> \
        --tap robocasa=~/Research/TRI/wsm_data/wsm_pooled/pi_100k \
        --cell E1 --steps 4000 --out ~/Research/TRI/wsm_data/deliberation/stage_e_runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from workspace_models.networks.omega_objectives import (  # noqa: E402
    jepa_loss,
    sigreg_term,
    supcon_deliberative,
    supcon_discriminative_stat,
)
from workspace_models.networks.stage_e_encoder import StageEEncoder  # noqa: E402

DOMAINS = ("robocasa", "remembench", "robomme", "robocerebra")
EDGE_KINDS = ("EQUIVALENT", "ANALOGOUS", "CONTRAST")
CONFIDENCES = ("high", "med", "low")

#: PRE-REGISTERED per-domain validity bar. Copied verbatim from `scripts/robocerebra/g1b_verdict.py`
#: so the H14 encoder is judged on the bar that already discriminated a collapsed encoder from a
#: working one — no thresholds are chosen here, and the frozen control must trip every FAIL.
G1B_BAR = {
    "temporal_coherence_gap": {"fail_below": 0.15, "pass_at_or_above": 0.40},
    "effective_rank": {"fail_below": 6.5, "pass_at_or_above": 8.0},
    "between_episode_variance_fraction": {"fail_below": 0.08, "pass_at_or_above": 0.20},
}

#: MULTI-DOMAIN G1b RECALIBRATION — pre-registered 2026-08-29, BEFORE any multi-domain cell ran.
#:
#: The single-domain effective-rank line (6.5 fail / 8.0 pass) was calibrated on RoboCerebra and
#: happens to sit sensibly under the RoboCasa tap's own raw rank. It does NOT transfer to a domain
#: whose raw tap is narrower: §17.4 measures the frozen pooled tap at effective rank 10.16 [9.93,
#: 10.35] on RoboCasa and 5.90 [5.66, 6.12] on ReMemBench — rmb frames genuinely occupy a 42%
#: narrower subspace (13 near-identical kitchen layouts vs 13 varied tasks). Judged on the fixed
#: 6.5 line, a PERFECT rmb encoder that preserved every dimension its input carries would still be
#: marked FAIL. The bar therefore becomes relative to what the domain's input supplies:
#:
#:     fail_below = 0.80 x that domain's RAW-TAP effective rank
#:
#: (RoboCasa 10.16 -> 8.13, rmb 5.90 -> 4.72; the coordinator's 8.1 / 4.7.) The PASS line is scaled
#: by the SAME ratio the original bar used (8.0 / 6.5 = 1.2308 x the fail line), because a pass line
#: left at a fixed 8.0 would sit BELOW RoboCasa's new fail line and the predicate would be
#: ill-formed. Coherence and bevf floors are UNCHANGED, and the collapse control must still trip
#: FAIL on every domain — that is what keeps this a recalibration rather than a relaxation.
#:
#: Applied ONLY when more than one tap is loaded, so every single-domain cell (the sealed funnel,
#: the v2 cells, the A14 seed replication) is still judged on the bar it was run under.
#: RoboCerebra's entry is NOT hardcoded: its tap is built by the same cluster pipeline that trains
#: this encoder, so its raw-tap effective rank is measured by `tap_stats_audit.py` in that run and
#: injected here via WSM_RAW_TAP_ERANK_JSON ({"robocerebra": <rank>, ...}). Hardcoding a guess would
#: be choosing a threshold, which is the one thing this bar exists to avoid.
#: robocerebra MEASURED 2026-09-01 by tap_stats_audit over the shipped 994-episode store
#: (stratified, 48 files / 5,610 rows): 4.497 [4.353, 4.619]; the node's own single-tap audit
#: independently gave 4.558 [4.388, 4.718]. Narrowest of the three taps, as a 2-view/128-token
#: LIBERO input should be -- which is precisely why the recalibrated bar matters most here.
RAW_TAP_EFFECTIVE_RANK = {"robocasa": 10.16, "remembench": 5.90, "robocerebra": 4.50}
if os.environ.get("WSM_RAW_TAP_ERANK_JSON"):
    RAW_TAP_EFFECTIVE_RANK = {
        **RAW_TAP_EFFECTIVE_RANK,
        **{
            k: float(v)
            for k, v in json.loads(Path(os.environ["WSM_RAW_TAP_ERANK_JSON"]).expanduser().read_text()).items()
        },
    }
G1B_ERANK_FAIL_FRACTION = 0.80
G1B_ERANK_PASS_OVER_FAIL = 8.0 / 6.5


LANG_MODES = ("episode_mean", "task_mean", "per_frame")

#: The SERVE-CONSISTENT default: for each domain, exactly the statistic the server computes at
#: rollout. Pre-registered in §27; `episode_mean` survives only as an explicit opt-in so the sealed
#: cells still reproduce.
SERVE_CONSISTENT_LANG = {
    "robocasa": "task_mean",
    "remembench": "task_mean",
    "robomme": "task_mean",
    "robocerebra": "per_frame",
}


def parse_lang_modes(spec: str) -> dict:
    """'task_mean' -> every domain; 'robocasa=task_mean,robocerebra=per_frame' -> per-domain.

    'serve' is the shorthand for SERVE_CONSISTENT_LANG. Unlisted domains fall back to
    `episode_mean`, so a typo cannot silently change a domain's conditioning.
    """
    if spec == "serve":
        return dict(SERVE_CONSISTENT_LANG)
    if "=" not in spec:
        if spec not in LANG_MODES:
            raise SystemExit(f"unknown --lang-mode {spec!r}; expected one of {LANG_MODES} or 'serve'")
        return {d: spec for d in DOMAINS}
    out = {d: "episode_mean" for d in DOMAINS}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        dom, _, mode = part.partition("=")
        if dom not in DOMAINS:
            raise SystemExit(f"--lang-mode names unknown domain {dom!r}; expected {DOMAINS}")
        if mode not in LANG_MODES:
            raise SystemExit(f"--lang-mode {dom}={mode!r}; expected one of {LANG_MODES}")
        out[dom] = mode
    return out


def g1b_bar_for(domain: str, multi_domain: bool) -> dict:
    """The bar this domain is judged on. Single-domain runs keep the sealed bar verbatim."""
    if not multi_domain:
        return G1B_BAR
    if domain not in RAW_TAP_EFFECTIVE_RANK:
        # Falling through to the fixed bar here is exactly the ill-formed comparison the
        # recalibration exists to prevent (a narrow-tap domain judged on RoboCasa's line). Fail
        # loud: the raw-tap rank is a measurement, and a missing one is a missing measurement.
        raise SystemExit(
            f"multi-domain cell loaded tap {domain!r} with no raw-tap effective rank. Measure it "
            f"with scripts/deliberation/tap_stats_audit.py and pass it via WSM_RAW_TAP_ERANK_JSON."
        )
    fail = round(G1B_ERANK_FAIL_FRACTION * RAW_TAP_EFFECTIVE_RANK[domain], 2)
    bar = {k: dict(v) for k, v in G1B_BAR.items()}
    bar["effective_rank"] = {
        "fail_below": fail,
        "pass_at_or_above": round(fail * G1B_ERANK_PASS_OVER_FAIL, 2),
        "raw_tap_effective_rank": RAW_TAP_EFFECTIVE_RANK[domain],
        "recalibrated": "multi-domain, 0.80 x raw-tap effective rank",
    }
    return bar


CELLS = (
    "E1",
    "ctrl-0",
    "ctrl-1D",
    "ctrl-E",
    "ctrl-S",
    "ctrl-T",
    "E1-analog05",
    "E1-seed2",
    "ctrl-0-seed2",
    "E1-noCONTRAST",
    # label artifact v2 (binding-aware CONTRAST)
    "E1b",
    "ctrl-0b",
    "E1b-bindingOnly",
    # A14 seed replication of the primary contrast (label artifact v2b)
    "ctrl-Eb",
    "E1b-analog05",
    # multi-domain funnel (robocasa + remembench)
    "ctrl-1Db",
    # A18 second wave: identical objective to E1b/ctrl-0b, but loaded with the RoboMME tap as
    # a FOURTH domain. The cell name is what distinguishes them -- the tap set is chosen by the
    # entry, not by the spec -- so they must be separate names or their run dirs collide.
    "E1b-4tap",
    "ctrl-0b-4tap",
)

#: Per-cell overrides. Everything not named here is identical across cells by construction.
CELL_SPEC = {
    "E1": {"edges": "E1", "lambda_del": 1.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    "ctrl-0": {"edges": "E1", "lambda_del": 0.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    "ctrl-1D": {"edges": "E1", "lambda_del": 1.0, "domains": ("robocasa",), "analogous_weight": 1.0, "seed_offset": 0},
    "ctrl-E": {"edges": "ctrl-E", "lambda_del": 1.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    "ctrl-S": {"edges": "ctrl-S", "lambda_del": 1.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    "ctrl-T": {"edges": "ctrl-T", "lambda_del": 1.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    "E1-analog05": {"edges": "E1", "lambda_del": 1.0, "domains": None, "analogous_weight": 0.5, "seed_offset": 0},
    "E1-seed2": {"edges": "E1", "lambda_del": 1.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 1},
    # Paired second seed on the OTHER arm of the primary contrast. It exists because ctrl-1D is
    # definitionally identical to E1 whenever only one domain's tap is loadable (see the module
    # docstring's A3 note): a bit-identical rerun contributes no attribution, whereas a paired seed
    # makes the E1 - ctrl-0 difference readable against run-to-run spread on BOTH sides.
    "ctrl-0-seed2": {"edges": "E1", "lambda_del": 0.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 1},
    # A9 fallout (coordinator, 2026-08-28): CONTRAST precision adjudicated at 0.172 and planted-probe
    # recovery at 0.533 — 72% of CONTRAST verdicts are really positives under the frozen schema's
    # "may differ in instance" clause. This cell is the discriminating experiment for whether those
    # contested edges carry signal at all: E1 with CONTRAST DEMOTED to an ordinary negative,
    # positives untouched, same seed, so the pair is read paired.
    #
    # `contrast_weight = 1.0`, not 0.0, and the difference is load-bearing. In the SupCon kernel the
    # weight multiplies the pair INSIDE THE DENOMINATOR, so 1.0 makes a CONTRAST pair exactly as
    # repulsive as any other frame pair (the neutralisation this experiment wants), while 0.0 would
    # DELETE it from the denominator — giving those pairs a second, opposite special role rather
    # than none. Neutralise, do not excise: excision changes two things at once.
    "E1-noCONTRAST": {
        "edges": "E1",
        "lambda_del": 1.0,
        "domains": None,
        "analogous_weight": 1.0,
        "seed_offset": 0,
        "contrast_weight": 1.0,
    },
    # --- label artifact v2: hard negatives graded by EVIDENCE, not by Qwen's letter -----------
    # These consume `edges_E1b.npz`, whose `hardneg` in [0,1] is mapped to a denominator
    # multiplier m = 1 + hardneg*(contrast_weight - 1). Same seeds and budget as the v1 funnel, so
    # E1b/ctrl-0b read paired against E1/ctrl-0.
    "E1b": {"edges": "E1b", "lambda_del": 1.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    "ctrl-0b": {"edges": "E1b", "lambda_del": 0.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    # A18: byte-identical objectives to E1b / ctrl-0b above. The ONLY difference is which taps the
    # entry loads, so E1b vs E1b-4tap is a clean paired reading of "does adding RoboMME's frames and
    # its 86,711 edges change retrieval on the other three domains".
    "E1b-4tap": {"edges": "E1b", "lambda_del": 1.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    "ctrl-0b-4tap": {"edges": "E1b", "lambda_del": 0.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    # Hard negatives = binding-corroborated ONLY; a Qwen CONTRAST the binding table cannot confirm
    # is demoted to an ordinary negative (hardneg 0 -> multiplier 1.0), not deleted.
    "E1b-bindingOnly": {
        "edges": "E1b",
        "lambda_del": 1.0,
        "domains": None,
        "analogous_weight": 1.0,
        "seed_offset": 0,
        "binding_only": True,
    },
    # --- A14 seed replication (label artifact v2b, `edges_ctrl-Eb.npz`) -----------------------
    # The funnel left "is Qwen worth it" INDETERMINATE: ctrl-E's lift sat inside E1's own
    # same-config seed spread. `ctrl-Eb` is the control that isolates the POSITIVES: top-k
    # descriptor-embedding neighbours (no Qwen) minus the binding-flagged pairs, carrying the SAME
    # v2 hard negatives as E1b, verbatim rows and strengths. The old `ctrl-E` differed from E1 on
    # two axes at once — mined positives AND no hard negatives at all — so half of its deficit
    # could have been the missing denominator term. Same `seed_offset` 0 on all three arms: the
    # replication varies the seed through `--seed` so the pairing is explicit at the call site.
    "ctrl-Eb": {"edges": "ctrl-Eb", "lambda_del": 1.0, "domains": None, "analogous_weight": 1.0, "seed_offset": 0},
    # The v1 sensitivity cell (E1-analog05, the funnel's top scorer) carried onto v2 labels, so the
    # ANALOGOUS down-weight is read paired against E1b at the same seed rather than across artifacts.
    "E1b-analog05": {"edges": "E1b", "lambda_del": 1.0, "domains": None, "analogous_weight": 0.5, "seed_offset": 0},
    # --- multi-domain funnel (robocasa + remembench), 2026-08-29 ------------------------------
    # `ctrl-1D` above is pinned to the v1 `edges_E1` artifact and must not move; on v2 labels the
    # domain-mixing control is this cell. It is E1b with the corpus RESTRICTED to RoboCasa — same
    # edges, same full objective, same seed — so E1b - ctrl-1Db isolates the second domain and
    # nothing else. (Read against E1b at the same seed; against ctrl-1D it would differ on two axes.)
    "ctrl-1Db": {
        "edges": "E1b",
        "lambda_del": 1.0,
        "domains": ("robocasa",),
        "analogous_weight": 1.0,
        "seed_offset": 0,
    },
}


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def effective_rank_np(x: np.ndarray) -> float:
    centered = x - x.mean(0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(np.cov(centered, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / max((ev**2).sum(), 1e-12))


# =============================================================================================
# Corpus: pooled taps, GPU-resident
# =============================================================================================
class Corpus:
    """Every loaded episode as ONE flat fp16 GPU tensor plus offset/segment index tables.

    Layout: `feat [F_total, feat_dim]`, and episode e owns rows [start[e], start[e] + length[e]).
    Nothing is copied from the host after construction.
    """

    def __init__(
        self,
        labels_dir: Path,
        taps: dict,
        device,
        max_episodes_per_task: int = 0,
        split_seed: int = 20260722,
        heldout_frac: float = 0.1,
        lang_mode: str = "episode_mean",
        task_lang_tables: dict | None = None,
    ) -> None:
        # SERVE-CONSISTENT CONDITIONING (§25.5-25.7 -> §27). The conditioning statistic is now a
        # per-DOMAIN choice, because "what the server can compute at rollout" is a per-domain fact:
        #   task_mean     the per-task vector the sealed serve lane's --task-lang-table provides
        #                 (robocasa, remembench: one instruction per task, constant over the episode)
        #   per_frame     the frame's own instruction (robocerebra: the prompt is re-pinned per
        #                 subtask, so the current instruction IS what the policy is served)
        #   episode_mean  each demo's own mean -- the SEALED behaviour, kept only so existing cells
        #                 reproduce. It is NOT serveable: no causal path can know an episode mean.
        self.lang_modes = parse_lang_modes(lang_mode)
        self.lang_mode = lang_mode
        self.task_lang_tables = dict(task_lang_tables or {})
        segments = np.load(labels_dir / "segments.npz", allow_pickle=True)
        vocab = json.loads((labels_dir / "vocab.json").read_text())
        task_names = vocab["tasks"]
        self.labels_dir, self.device = labels_dir, device
        self.n_segments = int(len(segments["t0"]))

        # (domain, task, episode) -> the segment keys of that episode, ordered by segment index
        by_episode: dict = {}
        for key in range(self.n_segments):
            ident = (int(segments["domain"][key]), int(segments["task"][key]), int(segments["episode"][key]))
            by_episode.setdefault(ident, []).append(key)

        feats, langs, seg_keys, lengths, meta, frame_indices = [], [], [], [], [], []
        lang_frames: list = []
        self.feat_dim, self.lang_dim = None, None
        missing = {"no_tap_file": 0, "no_segments": 0}
        for domain_name, root in sorted(taps.items()):
            domain_index = DOMAINS.index(domain_name)
            per_task: dict = {}
            for (d, t, ep), keys in by_episode.items():
                if d != domain_index:
                    continue
                per_task.setdefault(t, []).append((ep, keys))
            for task_index, episodes in sorted(per_task.items()):
                episodes.sort()
                if max_episodes_per_task:
                    episodes = episodes[:max_episodes_per_task]
                task = task_names[task_index]
                for ep, keys in episodes:
                    path = Path(root) / task / f"demo_{ep:06d}" / "p.npz"
                    if not path.exists():
                        missing["no_tap_file"] += 1
                        continue
                    blob = np.load(path)
                    p = np.asarray(blob["p"])
                    frame_index = np.asarray(blob["frame_indices"], np.int64)
                    lang = np.asarray(blob["lang_global"], np.float32)
                    # §25.3: Stage-E conditions on ONE lang vector per episode, but a live rollout
                    # cannot know an episode mean without reading the future. Where the tap ships a
                    # per-frame vector (RoboCerebra, whose prompt changes at every re-pin), the
                    # per-frame form is the only one a causal serve path can reproduce EXACTLY.
                    # Opt-in: `--lang-mode episode_mean` (the default) leaves every existing cell
                    # bit-identical.
                    lang_pf = None
                    if self.lang_modes.get(domain_name) == "per_frame" and "lang_per_frame" in blob.files:
                        lang_pf = np.asarray(blob["lang_per_frame"], np.float16)
                    key_of_frame = np.full(len(frame_index), -1, np.int32)
                    for key in keys:
                        inside = (frame_index >= int(segments["t0"][key])) & (frame_index < int(segments["t1"][key]))
                        key_of_frame[inside] = key
                    if (key_of_frame < 0).all():
                        missing["no_segments"] += 1
                        continue
                    feats.append(p.astype(np.float16))
                    langs.append(lang.astype(np.float16))
                    lang_frames.append(lang_pf)
                    seg_keys.append(key_of_frame)
                    frame_indices.append(frame_index)
                    lengths.append(len(p))
                    meta.append((domain_index, task_index, ep, task))
                    if self.feat_dim is None:
                        self.feat_dim, self.lang_dim = p.shape[-1], lang.shape[-1]
        if not feats:
            raise SystemExit(f"no episodes loaded from taps {taps} — refusing to train on nothing")

        self.missing = missing
        self.length = torch.tensor(lengths, dtype=torch.int64, device=device)
        self.start = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), self.length.cumsum(0)[:-1]])
        self.feat = torch.from_numpy(np.concatenate(feats)).to(device)
        langs_np = np.stack(langs)
        dom_np = np.asarray([m[0] for m in meta], np.int64)
        tsk_np = np.asarray([m[1] for m in meta], np.int64)
        self.lang_source = {}
        for dname, mode in self.lang_modes.items():
            if mode != "task_mean" or dname not in DOMAINS:
                continue
            di = DOMAINS.index(dname)
            sel = dom_np == di
            if not sel.any():
                continue
            table = self.task_lang_tables.get(dname)
            if table is not None:
                # Prefer the SERVED bytes over a recomputed mean: the server reads a task-lang
                # table, and a mean recomputed over a different demo set is a different vector.
                names = {str(k): np.asarray(v, np.float32) for k, v in table.items()}
                hit = 0
                for i in np.flatnonzero(sel):
                    name = meta[i][3]
                    if name in names:
                        langs_np[i] = names[name].astype(np.float16)
                        hit += 1
                self.lang_source[dname] = f"task_lang_table ({hit}/{int(sel.sum())} matched)"
                if hit != int(sel.sum()):
                    raise SystemExit(
                        f"task-lang table for {dname} covers {hit}/{int(sel.sum())} episodes; "
                        "a partial table would condition some episodes on a different statistic"
                    )
            else:
                for t in np.unique(tsk_np[sel]):
                    rows = np.flatnonzero(sel & (tsk_np == t))
                    langs_np[rows] = langs_np[rows].astype(np.float32).mean(0).astype(np.float16)
                self.lang_source[dname] = "within-task mean of lang_global"
        self.lang = torch.from_numpy(langs_np).to(device)
        # Compact bank: only the episodes that actually carry per-frame lang are materialised, so a
        # 3-domain corpus with one per-frame domain pays for that domain alone (994 x T x 2048 fp16
        # ~ 1.1 GB), not for every episode.
        have = [i for i, lf in enumerate(lang_frames) if lf is not None]
        if have:
            t_max = max(lang_frames[i].shape[0] for i in have)
            bank = np.zeros((len(have), t_max, self.lang_dim), np.float16)
            for row, i in enumerate(have):
                lf = lang_frames[i]
                bank[row, : lf.shape[0]] = lf
            self.lang_frames = torch.from_numpy(bank).to(device)
            row_map = np.full(len(langs), -1, np.int64)
            row_map[np.asarray(have, np.int64)] = np.arange(len(have), dtype=np.int64)
            self.lang_frame_row = torch.from_numpy(row_map).to(device)
        else:
            self.lang_frames, self.lang_frame_row = None, None
        print(
            f"[corpus] lang_modes={self.lang_modes} sources={self.lang_source}; "
            f"per-frame lang on {len(have)}/{len(langs)} episodes",
            flush=True,
        )

        # FAIL-CLOSED on the silent-success class. Passing --tap <domain>=... asserts that domain is
        # part of this cell. The loader walks SEGMENTS, not taps, so a domain absent from the label
        # artifact (or a mis-pointed tap root) contributes zero episodes with `missing` all-zero and
        # no warning: a "3-tap" cell trains as a silent 2-tap one, passes every gate, and exports ω
        # for a domain the encoder never saw. Reproduced 2026-09-02 with the real three taps.
        loaded = {}
        for m in meta:
            loaded[DOMAINS[m[0]]] = loaded.get(DOMAINS[m[0]], 0) + 1
        empty = sorted(set(taps) - set(loaded))
        print(f"[corpus] episodes per domain: {loaded}", flush=True)
        if empty:
            raise SystemExit(
                f"[corpus] FATAL: taps were loaded for {sorted(taps)} but {empty} contributed ZERO "
                f"episodes (per-domain counts {loaded}). Either the label artifact has no segments "
                f"for those domains or the tap root is wrong. Refusing to train a cell that "
                f"silently drops a domain."
            )
        self.episodes_per_domain = loaded
        self.seg_of_frame = torch.from_numpy(np.concatenate(seg_keys).astype(np.int64)).to(device)
        self.domain_of_episode = torch.tensor([m[0] for m in meta], dtype=torch.int64, device=device)
        self.task_of_episode = torch.tensor([m[1] for m in meta], dtype=torch.int64, device=device)
        self.meta = meta
        self.frame_index = frame_indices  # per episode, the EPISODE frame index of each cached frame
        self.n_episodes = len(meta)
        self.max_len = int(self.length.max())
        self.segments = {k: segments[k] for k in segments.files}
        self.task_names = task_names

        # Episode index of each segment key (-1 where the segment's episode never loaded).
        episode_of_segment = np.full(self.n_segments, -1, np.int64)
        for episode_index, (d, t, ep, _task) in enumerate(meta):
            for key in by_episode[(d, t, ep)]:
                episode_of_segment[key] = episode_index
        self.episode_of_segment = torch.from_numpy(episode_of_segment).to(device)

        # Split by EPISODE, per domain, so heldout between-episode variance is meaningful and every
        # domain is represented on both sides.
        rng = np.random.default_rng(split_seed)
        train_mask = np.ones(self.n_episodes, bool)
        for domain_index in range(len(DOMAINS)):
            rows = np.flatnonzero(np.asarray([m[0] for m in meta]) == domain_index)
            if len(rows) == 0:
                continue
            n_heldout = max(4, int(round(heldout_frac * len(rows)))) if len(rows) > 8 else max(1, len(rows) // 5)
            train_mask[rng.permutation(rows)[:n_heldout]] = False
        self.train_episodes = torch.from_numpy(np.flatnonzero(train_mask)).to(device)
        self.heldout_episodes = torch.from_numpy(np.flatnonzero(~train_mask)).to(device)
        self.is_train_episode = torch.from_numpy(train_mask).to(device)

    def gather(self, episodes: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Right-padded [B, T, ...] view of the requested episodes. Pure GPU gather."""
        lengths = self.length[episodes]
        t = int(lengths.max())
        positions = torch.arange(t, device=self.device)
        valid = positions[None, :] < lengths[:, None]
        rows = (self.start[episodes][:, None] + positions[None, :]).clamp_max(len(self.feat) - 1)
        feat = self.feat[rows]
        seg = torch.where(valid, self.seg_of_frame[rows], torch.full_like(rows, -1))
        lang = self.lang[episodes][:, None, :].expand(-1, t, -1)
        if self.lang_frames is not None:
            rows_l = self.lang_frame_row[episodes]
            sel = rows_l >= 0
            if bool(sel.any()):
                lang = lang.clone()
                bank = self.lang_frames[rows_l[sel]]  # [m, T_bank, D]
                k = min(t, bank.shape[1])
                lang[sel, :k] = bank[:, :k]
            # Frames beyond the bank (only possible if a tap shipped a short lang_per_frame) keep
            # the episode vector rather than zeros -- a zero lang is a different condition, not a
            # missing one.
        return feat, lang, seg, valid, self.domain_of_episode[episodes]


# =============================================================================================
# Edges, GPU-resident CSR
# =============================================================================================
class EdgeTable:
    def __init__(
        self,
        path: Path,
        corpus: Corpus,
        device,
        *,
        analogous_weight: float,
        include_low_confidence: bool,
        lambda_xdom: float,
        domain_filter=None,
        contrast_weight: float = 2.0,
        binding_only: bool = False,
    ) -> None:
        blob = np.load(path)
        src, dst = blob["src"].astype(np.int64), blob["dst"].astype(np.int64)
        kind, conf = blob["kind"].astype(np.int64), blob["conf"].astype(np.int64)
        segments = corpus.segments
        episode_of = corpus.episode_of_segment.cpu().numpy()
        keep = (episode_of[src] >= 0) & (episode_of[dst] >= 0)
        if not include_low_confidence:
            keep &= conf < CONFIDENCES.index("low")
        if domain_filter is not None:
            allowed = np.zeros(len(DOMAINS), bool)
            for name in domain_filter:
                allowed[DOMAINS.index(name)] = True
            keep &= allowed[segments["domain"][src]] & allowed[segments["domain"][dst]]
        # Only train-split episodes may shape the objective.
        train = corpus.is_train_episode.cpu().numpy()
        keep &= train[episode_of[src]] & train[episode_of[dst]]
        # v2 (`edges_E1b.npz`) grades each hard negative by evidence in [0,1]; v1 has no such
        # column, so every CONTRAST is full strength, which reproduces the funnel exactly.
        hardneg = (
            blob["hardneg"].astype(np.float32)
            if "hardneg" in blob.files
            else (kind == EDGE_KINDS.index("CONTRAST")).astype(np.float32)
        )
        binding = blob["binding"].astype(bool) if "binding" in blob.files else np.zeros(len(kind), bool)
        if binding_only:
            # Demote every non-corroborated hard negative to an ORDINARY negative (strength 0),
            # which is a multiplier of 1.0 — not a deletion from the denominator.
            hardneg = np.where(binding, hardneg, 0.0).astype(np.float32)
        src, dst, kind, conf = src[keep], dst[keep], kind[keep], conf[keep]
        hardneg, binding = hardneg[keep], binding[keep]

        kind_w = np.where(kind == EDGE_KINDS.index("ANALOGOUS"), analogous_weight, 1.0)
        conf_w = np.asarray([1.0, 0.5, 0.25], np.float32)[conf]
        cross_domain = segments["domain"][src] != segments["domain"][dst]
        weight = (kind_w * conf_w * np.where(cross_domain, lambda_xdom, 1.0)).astype(np.float32)

        positive = kind <= EDGE_KINDS.index("ANALOGOUS")

        # Symmetrise: the store holds one record per unordered pair (edge_schema §1).
        def both_ways(mask):
            s = np.concatenate([src[mask], dst[mask]])
            d = np.concatenate([dst[mask], src[mask]])
            w = np.concatenate([weight[mask], weight[mask]])
            x = np.concatenate([cross_domain[mask], cross_domain[mask]])
            order = np.argsort(s, kind="stable")
            return s[order], d[order], w[order], x[order]

        # Denominator multiplier per hard negative: m = 1 + hardneg*(contrast_weight - 1), so
        # strength 0 -> 1.0 (ordinary negative) and strength 1 -> contrast_weight (the funnel).
        multiplier = (1.0 + hardneg * (contrast_weight - 1.0)).astype(np.float32)

        self.pos_src, self.pos_dst, self.pos_w, self.pos_xdom = both_ways(positive)
        negative = ~positive
        order = np.argsort(np.concatenate([src[negative], dst[negative]]), kind="stable")
        self.neg_src = np.concatenate([src[negative], dst[negative]])[order]
        self.neg_dst = np.concatenate([dst[negative], src[negative]])[order]
        self.neg_m = np.concatenate([multiplier[negative], multiplier[negative]])[order]
        self.n_positive, self.n_contrast = int(positive.sum()), int(negative.sum())
        self.n_binding = int(binding.sum())
        self.mean_hardneg = float(hardneg[negative].mean()) if negative.any() else 0.0
        self.cross_domain_frac = float(cross_domain[positive].mean()) if positive.any() else 0.0

        to = lambda a, dtype: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)  # noqa: E731
        self.g_pos_src = to(self.pos_src, torch.int64)
        self.g_pos_dst = to(self.pos_dst, torch.int64)
        self.g_pos_w = to(self.pos_w, torch.float32)
        self.g_pos_xdom = to(self.pos_xdom, torch.bool)
        self.g_neg_src = to(self.neg_src, torch.int64)
        self.g_neg_dst = to(self.neg_dst, torch.int64)
        self.g_neg_m = to(self.neg_m, torch.float32)
        self.pos_offsets = torch.searchsorted(self.g_pos_src, torch.arange(corpus.n_segments + 1, device=device))
        self.neg_offsets = torch.searchsorted(self.g_neg_src, torch.arange(corpus.n_segments + 1, device=device))
        self.device = device

    def submatrix(self, segment_keys: torch.Tensor, n_segments: int):
        """[S,S] positive-weight and CONTRAST tables for the segments realised in this batch."""
        s = len(segment_keys)
        local = torch.full((n_segments,), -1, dtype=torch.int64, device=self.device)
        local[segment_keys] = torch.arange(s, device=self.device)
        weight = torch.zeros(s, s, device=self.device)
        # Denominator multiplier table: 1.0 everywhere = every pair an ordinary negative; hard
        # negatives overwrite their own cells with their graded multiplier.
        negative_multiplier = torch.ones(s, s, device=self.device)
        contrast = torch.zeros(s, s, dtype=torch.bool, device=self.device)
        xdom_hits = 0
        for offsets, dst, values, out, is_positive in (
            (self.pos_offsets, self.g_pos_dst, self.g_pos_w, weight, True),
            (self.neg_offsets, self.g_neg_dst, self.g_neg_m, negative_multiplier, False),
        ):
            lo, hi = offsets[segment_keys], offsets[segment_keys + 1]
            counts = hi - lo
            total = int(counts.sum())
            if total == 0:
                continue
            # Vectorised ragged range expansion: concat(arange(lo_i, hi_i)) without a host loop.
            rows = torch.repeat_interleave(torch.arange(s, device=self.device), counts)
            starts = (counts.cumsum(0) - counts).repeat_interleave(counts)
            index = lo.repeat_interleave(counts) + (torch.arange(total, device=self.device) - starts)
            columns = local[dst[index]]
            live = columns >= 0
            if is_positive:
                out[rows[live], columns[live]] = values[index][live]
                xdom_hits = int(self.g_pos_xdom[index][live].sum())
            else:
                out[rows[live], columns[live]] = values[index][live]
                contrast[rows[live], columns[live]] = True
        weight = torch.maximum(weight, weight.T)
        contrast = contrast | contrast.T
        negative_multiplier = torch.maximum(negative_multiplier, negative_multiplier.T)
        # A pair cannot be both; the positive edge wins and its multiplier reverts to ordinary.
        contrast &= weight <= 0
        negative_multiplier = torch.where(weight > 0, torch.ones_like(negative_multiplier), negative_multiplier)
        return weight, contrast, negative_multiplier, xdom_hits


# =============================================================================================
# Training
# =============================================================================================
def build_batch(corpus: Corpus, edges: EdgeTable, rng: np.random.Generator, batch_episodes: int, min_edges: int):
    """EDGE-FIRST composition, then domain balance. Returns the episode indices for this step."""
    picked: list[int] = []
    if len(edges.pos_src):
        seeds = rng.integers(0, len(edges.pos_src), size=max(min_edges, batch_episodes))
        episode_of = corpus.episode_of_segment
        for s in seeds:
            for key in (edges.pos_src[s], edges.pos_dst[s]):
                episode = int(episode_of[int(key)])
                if episode >= 0 and episode not in picked:
                    picked.append(episode)
            if len(picked) >= batch_episodes:
                break
    picked = picked[:batch_episodes]
    # Domain balance (pin D5): top up the shortest-represented domain first.
    train = corpus.train_episodes.cpu().numpy()
    domain_of = corpus.domain_of_episode.cpu().numpy()
    present = sorted({int(d) for d in domain_of[train]})
    while len(picked) < batch_episodes:
        counts = {d: sum(1 for e in picked if domain_of[e] == d) for d in present}
        target = min(counts, key=counts.get)
        pool = train[domain_of[train] == target]
        candidate = int(pool[rng.integers(0, len(pool))])
        if candidate not in picked:
            picked.append(candidate)
    return torch.tensor(picked, dtype=torch.int64, device=corpus.device)


@torch.no_grad()
def omega_for_episodes(encoder, corpus: Corpus, episodes: torch.Tensor, chunk: int = 16):
    """ω per episode, as a list of [F, dim] float32 CPU arrays (padding removed)."""
    out = []
    for start in range(0, len(episodes), chunk):
        block = episodes[start : start + chunk]
        feat, lang, _seg, valid, domain = corpus.gather(block)
        omega = encoder(feat, lang, domain)
        for row in range(len(block)):
            out.append(omega[row][valid[row]].float().cpu().numpy())
    return out


@torch.no_grad()
def g1b_metrics(omegas: list, step: int) -> dict:
    from workspace_models.networks.sigreg_loss import sigreg_epps_pulley

    omega = np.concatenate(omegas)
    within = float(np.mean([e.var(0).mean() for e in omegas]))
    between = float(np.stack([e.mean(0) for e in omegas]).var(0).mean())
    units = [e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9) for e in omegas]
    adjacent = np.concatenate([(u[:-1] * u[1:]).sum(1) for u in units if len(u) > 1])
    unit = omega / np.maximum(np.linalg.norm(omega, axis=1, keepdims=True), 1e-9)
    rng = np.random.default_rng(0)
    random_pairs = (unit[rng.integers(0, len(unit), 20000)] * unit[rng.integers(0, len(unit), 20000)]).sum(1)
    return {
        "n_frames": int(len(omega)),
        "n_episodes": int(len(omegas)),
        "finite": bool(np.isfinite(omega).all()),
        "omega_rms": float(np.sqrt((omega**2).mean())),
        "effective_rank": effective_rank_np(omega[:8000]),
        "between_episode_variance_fraction": float(between / max(between + within, 1e-12)),
        "cos_adjacent_mean": float(adjacent.mean()),
        "cos_random_mean": float(random_pairs.mean()),
        "temporal_coherence_gap": float(adjacent.mean() - random_pairs.mean()),
        # `.cuda()` here was unconditional and made every CPU run (a local smoke of a new code
        # path, for one) die inside the gate rather than in training. The statistic is
        # device-agnostic.
        "sigreg_stat": float(
            sigreg_epps_pulley(
                torch.from_numpy(omega[:512]).float().to("cuda" if torch.cuda.is_available() else "cpu"), step
            )
        ),
    }


def g1b_verdict(metrics: dict, bar: dict | None = None) -> dict:
    rows, fails, passes = {}, [], []
    for metric, bounds in (bar or G1B_BAR).items():
        value = float(metrics[metric])
        failed, passed = value < bounds["fail_below"], value >= bounds["pass_at_or_above"]
        rows[metric] = {"value": round(value, 4), **bounds, "fails": failed, "passes": passed}
        fails.append(failed)
        passes.append(passed)
    return {"verdict": "FAIL" if any(fails) else ("PASS" if all(passes) else "INDETERMINATE"), "rows": rows}


@torch.no_grad()
def retrieval_gate(encoder, corpus: Corpus, device, rng: np.random.Generator, max_anchors: int = 400) -> dict:
    """FRAME-LEVEL cross-task retrieval on the A1d DISAGREEMENT subset (A2's go/no-go).

    Pool = every segment appearing in a held-out disagreement pair. For each anchor segment with at
    least one positive partner, rank every pool frame from a DIFFERENT task by ω-cosine and ask
    whether the nearest is a frame of a positive partner. `chance` is that anchor's positive
    density over its own candidate set, so the baseline is the one for this token count — a
    representation carrying no information scores lift 1.0 whatever the loss is doing.
    """
    blob = np.load(corpus.labels_dir / "gate_pairs.npz")
    src, dst, kind = blob["src"].astype(np.int64), blob["dst"].astype(np.int64), blob["kind"]
    episode_of = corpus.episode_of_segment.cpu().numpy()
    heldout = ~corpus.is_train_episode.cpu().numpy()
    live = (episode_of[src] >= 0) & (episode_of[dst] >= 0)
    live &= heldout[episode_of[src]] & heldout[episode_of[dst]]
    src, dst, kind = src[live], dst[live], kind[live]
    if len(src) == 0:
        return {
            "n_anchors": 0,
            "anchors_by_domain": {},
            "n_disagreement_pairs_total": int(len(blob["src"])),
            "note": "no held-out disagreement pairs for the loaded taps",
        }

    pool_segments = np.unique(np.concatenate([src, dst]))
    pool_episodes = np.unique(episode_of[pool_segments])
    omegas = omega_for_episodes(encoder, corpus, torch.tensor(pool_episodes, dtype=torch.int64, device=device))
    row_of_episode = {int(e): i for i, e in enumerate(pool_episodes)}

    # Flatten every pool frame, tagged with its segment key and task.
    frames, frame_segment = [], []
    for segment in pool_segments:
        episode = int(episode_of[segment])
        omega = omegas[row_of_episode[episode]]
        # seg_of_frame is stored flat and already excludes padding, so it aligns 1:1 with omega.
        start = int(corpus.start[episode])
        keys = corpus.seg_of_frame[start : start + len(omega)].cpu().numpy()
        inside = np.flatnonzero(keys == segment)
        if len(inside) == 0:
            continue
        frames.append(omega[inside])
        frame_segment.append(np.full(len(inside), segment, np.int64))
    if not frames:
        return {"n_anchors": 0, "anchors_by_domain": {}, "note": "no pool frames"}
    frames = torch.from_numpy(np.concatenate(frames)).float().to(device)
    frames = F.normalize(frames, dim=-1)
    frame_segment = np.concatenate(frame_segment)
    frame_task = corpus.segments["task"][frame_segment].astype(np.int64)
    # Domain of every pool frame, so the SAME anchors can also be scored with the candidate set
    # restricted to one domain at a time (the multi-domain pair-type split below).
    episode_of_np = episode_of
    domain_of_episode = corpus.domain_of_episode.cpu().numpy()
    frame_domain = domain_of_episode[episode_of_np[frame_segment]]
    domains_present = sorted(int(d) for d in np.unique(frame_domain))

    partners_positive: dict = {}
    for a, b, k in zip(src, dst, kind):
        for x, y in ((a, b), (b, a)):
            partners_positive.setdefault(int(x), set())
            if k <= EDGE_KINDS.index("ANALOGOUS"):
                partners_positive[int(x)].add(int(y))
    anchors = [s for s in pool_segments if partners_positive.get(int(s))]
    rng.shuffle(anchors)
    anchors = anchors[:max_anchors]

    def score(anchor_rows: np.ndarray, candidate: np.ndarray, positive_segments) -> tuple | None:
        """(hits, n_query_frames, chance_mass) for one anchor against one candidate set."""
        if len(candidate) < 8 or len(anchor_rows) == 0:
            return None
        is_positive = np.isin(frame_segment[candidate], list(positive_segments))
        if not is_positive.any() or is_positive.all():
            return None
        similarity = (
            frames[torch.from_numpy(anchor_rows).to(device)] @ frames[torch.from_numpy(candidate).to(device)].T
        )
        top1 = similarity.argmax(1).cpu().numpy()
        return (int(is_positive[top1].sum()), len(anchor_rows), float(is_positive.mean()) * len(anchor_rows))

    def summarise(hits: int, total: int, chance_mass: float) -> dict | None:
        if total == 0:
            return None
        chance = float(chance_mass / total)
        low, high = wilson(hits, total)
        return {
            "n_query_frames": total,
            "top1": round(hits / total, 4),
            "chance": round(chance, 4),
            "lift": round((hits / total) / max(chance, 1e-9), 3),
            "wilson95": [round(low, 4), round(high, 4)],
            "beats_chance": bool(low > chance),
        }

    hits, chances, total = 0, [], 0
    # PAIR-TYPE SPLIT (pre-registered for the multi-domain funnel, 2026-08-29): the identical anchor
    # set is re-scored with the candidate pool restricted to ONE domain at a time, so each stratum
    # carries its own chance baseline computed on its own candidate set. Keys are
    # `<anchor domain index>-><candidate domain index>`; the overall number above is unchanged and
    # remains the pre-registered selection metric.
    strata: dict = {}
    n_anchors_scored = 0
    anchors_by_domain: dict = {}
    for anchor in anchors:
        anchor_rows = np.flatnonzero(frame_segment == anchor)
        if len(anchor_rows) == 0:
            continue
        anchor_task = int(corpus.segments["task"][anchor])
        anchor_domain = int(domain_of_episode[int(episode_of_np[anchor])])
        eligible = (frame_task != anchor_task) & (frame_segment != anchor)
        candidate = np.flatnonzero(eligible)
        positive_segments = partners_positive[int(anchor)]
        scored = score(anchor_rows, candidate, positive_segments)
        if scored is not None:
            hits += scored[0]
            total += scored[1]
            chances.append(scored[2])
            n_anchors_scored += 1
            dn = DOMAINS[anchor_domain]
            anchors_by_domain[dn] = anchors_by_domain.get(dn, 0) + 1
        if len(domains_present) > 1:
            for cand_domain in domains_present:
                sub = np.flatnonzero(eligible & (frame_domain == cand_domain))
                s = score(anchor_rows, sub, positive_segments)
                if s is None:
                    continue
                key = f"{anchor_domain}->{cand_domain}"
                acc = strata.setdefault(key, [0, 0, 0.0, 0])
                acc[0] += s[0]
                acc[1] += s[1]
                acc[2] += s[2]
                acc[3] += 1
    if total == 0:
        return {"n_anchors": 0, "anchors_by_domain": {}, "note": "no qualifying anchors"}
    chance = float(np.sum(chances) / total)
    low, high = wilson(hits, total)
    out = {
        "n_anchors": len(anchors),
        "n_anchors_scored": n_anchors_scored,
        "anchors_by_domain": anchors_by_domain,
        "n_query_frames": total,
        "top1": round(hits / total, 4),
        "chance": round(chance, 4),
        "lift": round((hits / total) / max(chance, 1e-9), 3),
        "wilson95": [round(low, 4), round(high, 4)],
        "beats_chance": bool(low > chance),
        "subset": "A1d disagreement pairs, cross-task, held-out episodes only",
    }
    if strata:
        named = {}
        for key, (h, n, c, na) in sorted(strata.items()):
            a, _, b = key.partition("->")
            label = f"{DOMAINS[int(a)]}->{DOMAINS[int(b)]}"
            row = summarise(h, n, c)
            if row:
                named[label] = {**row, "n_anchors": na}
        # Pooled cross-domain bucket: both directions together, one baseline over their union.
        cross = [(k, v) for k, v in strata.items() if k.split("->")[0] != k.split("->")[1]]
        if cross:
            row = summarise(sum(v[0] for _, v in cross), sum(v[1] for _, v in cross), sum(v[2] for _, v in cross))
            if row:
                named["cross_domain (both directions)"] = {**row, "n_anchors": sum(v[3] for _, v in cross)}
        out["by_pair_type"] = named
        out["pair_type_note"] = (
            "same anchors, candidate pool restricted to one domain; each "
            "stratum's chance is computed on its own candidate set"
        )
    return out


@torch.no_grad()
def decode_grounding(
    encoder, corpus: Corpus, device, label_root: Path, n_patches: int = 192, top_m: int = 8, steps: int = 300
) -> dict:
    """Keyframe-patch decode probe on the domain that has a label store (RoboCasa pi geometry).

    A linear head ω -> 192 patch logits is fit on TRAIN keyframes and scored on HELD-OUT ones:
    recall@m of the salient set against the chance baseline m/192. This is the grounding readout
    (the standing lesson is to select on decode, never on a probe of the inputs).
    """
    if not label_root.exists():
        return {"skipped": "no label store"}
    rows_train, rows_test = [], []
    for episode_index, (domain, _task, ep, task) in enumerate(corpus.meta):
        if domain != DOMAINS.index("robocasa"):
            continue
        path = label_root / task / f"vlm_episode_pi_{ep:06d}.npz"
        if not path.exists():
            continue
        (rows_train if bool(corpus.is_train_episode[episode_index]) else rows_test).append((episode_index, path))
    if len(rows_train) < 16 or len(rows_test) < 4:
        return {"skipped": f"insufficient labelled episodes ({len(rows_train)}/{len(rows_test)})"}
    rows_train = rows_train[:400]

    def featurise(rows):
        x, y = [], []
        for episode_index, path in rows:
            blob = np.load(path, allow_pickle=True)
            keyframes = np.asarray(blob["keyframes"], np.int64)
            salient = list(blob["salient_global"])
            episode = torch.tensor([episode_index], dtype=torch.int64, device=device)
            omega = omega_for_episodes(encoder, corpus, episode)[0]
            frame_index = corpus.frame_index[episode_index]
            for k, keyframe in enumerate(keyframes):
                ids = np.asarray(salient[k], np.int64)
                if len(ids) == 0:
                    continue
                position = int(np.argmin(np.abs(frame_index - int(keyframe))))
                if position >= len(omega):
                    continue
                target = np.zeros(n_patches, np.float32)
                target[ids[ids < n_patches]] = 1.0
                x.append(omega[position])
                y.append(target)
        if not x:
            return None, None
        return (torch.from_numpy(np.stack(x)).float().to(device), torch.from_numpy(np.stack(y)).float().to(device))

    xtr, ytr = featurise(rows_train)
    xte, yte = featurise(rows_test)
    if xtr is None or xte is None:
        return {"skipped": "no keyframe targets"}
    head = torch.nn.Linear(xtr.shape[1], n_patches).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.0)
    with torch.enable_grad():
        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(head(xtr), ytr)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    predicted = head(xte).topk(top_m, dim=1).indices
    hit = yte.gather(1, predicted).sum(1)
    denominator = yte.sum(1).clamp(max=top_m).clamp_min(1)
    recall = float((hit / denominator).mean())
    chance = float((yte.mean(1) * top_m / denominator).mean())
    return {
        "recall_at_m": round(recall, 4),
        "chance": round(chance, 4),
        "lift": round(recall / max(chance, 1e-9), 3),
        "top_m": top_m,
        "n_train": int(len(xtr)),
        "n_test": int(len(xte)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="stage_e_labels/<label_id>")
    ap.add_argument("--tap", action="append", required=True, help="domain=/path/to/pooled/root, repeatable")
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-episodes", type=int, default=8)
    ap.add_argument("--min-edges-per-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--predict-k", type=int, default=6)
    ap.add_argument("--lambda-sigreg", type=float, default=0.05)
    ap.add_argument("--sigreg-rank-cap", type=float, default=15.0)
    ap.add_argument("--lambda-episode", type=float, default=1.0)
    ap.add_argument("--lambda-del", type=float, default=1.0)
    ap.add_argument(
        "--lambda-xdom",
        type=float,
        default=0.5,
        help="weight multiplier on CROSS-DOMAIN positives: the pass-2 QA shows they are "
        "ANALOGOUS-only (568 EQUIVALENT total, none touching RoboMME), so they "
        "must not carry the same authority as a within-domain EQUIVALENT",
    )
    ap.add_argument("--contrast-weight", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--include-low-confidence", action="store_true")
    ap.add_argument("--max-episodes-per-task", type=int, default=0)
    ap.add_argument("--heldout-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--label-store",
        default="~/Research/TRI/wsm_data/wsm_labels_pi_mirror",
        help="RoboCasa pi-geometry keyframe labels for the decode-grounding gate",
    )
    ap.add_argument(
        "--lang-mode",
        default="episode_mean",
        help="conditioning contract (§27). 'serve' = the serve-consistent per-domain "
        "default (robocasa/remembench/robomme=task_mean, robocerebra=per_frame). "
        "Also accepts one mode for all domains, or 'dom=mode,...'. Modes: "
        "episode_mean (SEALED, and NOT serveable), task_mean, per_frame.",
    )
    ap.add_argument(
        "--task-lang-table",
        action="append",
        default=[],
        help="domain=path to an npz of {task: vector}; used VERBATIM for that domain's "
        "task_mean instead of a recomputed mean, so training conditions on the "
        "exact bytes the server reads. Repeatable.",
    )
    ap.add_argument("--export-omega", default="", help="write a w.npz ω store here after training")
    ap.add_argument(
        "--bindings",
        default="~/Research/TRI/wsm_data/deliberation/binding_annotations/597f3ff5e7cbd6ce",
        help="binding annotation store for the binding-decodability gate",
    )
    ap.add_argument(
        "--pass1-robocasa",
        default="~/Research/TRI/wsm_data/deliberation/pass1_store/robocasa",
        help="descriptor store, used to locate each episode's reveal frame",
    )
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    args = ap.parse_args()

    spec = CELL_SPEC[args.cell]
    seed = args.seed + spec["seed_offset"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(args.device)
    labels_dir = Path(args.labels).expanduser()

    taps = {}
    for entry in args.tap:
        name, _, root = entry.partition("=")
        if name not in DOMAINS:
            raise SystemExit(f"unknown domain {name!r}; expected one of {DOMAINS}")
        taps[name] = str(Path(root).expanduser())
    if spec["domains"] is not None:
        taps = {k: v for k, v in taps.items() if k in spec["domains"]}
        if not taps:
            raise SystemExit(f"cell {args.cell} needs taps for {spec['domains']}")

    # Pre-registered: the effective-rank line is recalibrated per domain ONLY when >1 tap is loaded
    # (see RAW_TAP_EFFECTIVE_RANK). Single-domain cells keep the sealed bar so they stay comparable
    # to the funnel and the A14 seed replication.
    multi_domain = len(taps) > 1

    corpus = Corpus(
        labels_dir,
        taps,
        device,
        max_episodes_per_task=args.max_episodes_per_task,
        heldout_frac=args.heldout_frac,
        lang_mode=args.lang_mode,
        task_lang_tables=load_task_lang_tables(args.task_lang_table),
    )
    print(
        f"[corpus] {corpus.n_episodes} episodes / {len(corpus.feat)} frames "
        f"({len(corpus.train_episodes)} train, {len(corpus.heldout_episodes)} heldout); "
        f"feat_dim={corpus.feat_dim} lang_dim={corpus.lang_dim} missing={corpus.missing}",
        flush=True,
    )

    edges = EdgeTable(
        labels_dir / f"edges_{spec['edges']}.npz",
        corpus,
        device,
        analogous_weight=spec["analogous_weight"],
        include_low_confidence=args.include_low_confidence,
        lambda_xdom=args.lambda_xdom,
        domain_filter=tuple(taps),
        contrast_weight=float(spec.get("contrast_weight", args.contrast_weight)),
        binding_only=bool(spec.get("binding_only", False)),
    )
    print(
        f"[edges] {spec['edges']}: {edges.n_positive} positive / {edges.n_contrast} contrast "
        f"({edges.n_binding} binding-corroborated, mean hard-neg strength "
        f"{edges.mean_hardneg:.3f}); cross-domain positives {edges.cross_domain_frac:.3f}",
        flush=True,
    )

    cfg = SimpleNamespace(
        dim=args.dim,
        n_layers=args.n_layers,
        n_dec_layers=2,
        n_heads=args.n_heads,
        k_slots=32,
        backbone_dim=corpus.feat_dim,
        proprio_dim=corpus.lang_dim,
        lang_dim=corpus.lang_dim,
        c_horizon=1000,
        max_t=max(1200, corpus.max_len + 8),
        mlp_ratio=4.0,
        input_norm=False,
    )
    # `index` is the GLOBAL domain id the corpus tags every episode with. It must travel with the
    # spec: the encoder orders its adapters by sorted(name), which for {robocasa, remembench} is the
    # reverse of DOMAINS, so without it each domain would be routed through the other's adapter.
    domain_specs = {
        name: {"feat_dim": corpus.feat_dim, "lang_dim": corpus.lang_dim, "index": DOMAINS.index(name)} for name in taps
    }
    encoder = StageEEncoder(cfg, domain_specs).to(device)
    target = StageEEncoder(cfg, domain_specs).to(device)
    target.load_state_dict(encoder.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    predictor = torch.nn.Sequential(
        torch.nn.LayerNorm(cfg.dim),
        torch.nn.Linear(cfg.dim, cfg.dim * 4),
        torch.nn.GELU(),
        torch.nn.Linear(cfg.dim * 4, cfg.dim),
    ).to(device)

    # ---- frozen negative control: the SAME architecture, never trained. The bar has to trip on it.
    frozen = StageEEncoder(cfg, domain_specs).to(device).eval()

    run_config = {
        "cell": args.cell,
        "cell_spec": spec,
        "label_id": labels_dir.name,
        "taps": taps,
        "steps": args.steps,
        "batch_episodes": args.batch_episodes,
        "lr": args.lr,
        "warmup": args.warmup,
        "predict_k": args.predict_k,
        "lambda_sigreg": args.lambda_sigreg,
        "sigreg_rank_cap": args.sigreg_rank_cap,
        "lambda_episode": args.lambda_episode,
        "lambda_del": spec["lambda_del"] * args.lambda_del,
        "lambda_xdom": args.lambda_xdom,
        "contrast_weight": float(spec.get("contrast_weight", args.contrast_weight)),
        "tau": args.tau,
        "ema": args.ema,
        "include_low_confidence": bool(args.include_low_confidence),
        "heldout_frac": args.heldout_frac,
        "seed": seed,
        "cfg": {k: v for k, v in vars(cfg).items()},
        "code_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
    }
    encoder_id = hashlib.sha256(
        json.dumps(run_config, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]
    out_dir = Path(args.out).expanduser() / f"{args.cell}_{encoder_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps({"encoder_id": encoder_id, **run_config}, indent=1, default=str)
    )
    print(f"[run] cell={args.cell} encoder_id={encoder_id} -> {out_dir}", flush=True)

    lambda_del = spec["lambda_del"] * args.lambda_del
    contrast_weight = float(spec.get("contrast_weight", args.contrast_weight))
    parameters = encoder.trainable_parameters() + list(predictor.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    history, started = [], time.time()
    # Running totals over EVERY step (not just eval steps), so the reported in-batch cross-domain
    # fraction is what the objective actually consumed rather than a 12-sample snapshot.
    edges_realised_total, xdom_realised_total, batch_domain_counts = 0, 0, np.zeros(len(DOMAINS), np.int64)

    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(args.warmup, 1))
        for group in optimiser.param_groups:
            group["lr"] = lr
        episodes = build_batch(corpus, edges, rng, args.batch_episodes, args.min_edges_per_batch)
        feat, lang, seg, valid, domain = corpus.gather(episodes)

        omega = encoder(feat, lang, domain)
        with torch.no_grad():
            omega_target = target(feat, lang, domain)

        k = args.predict_k
        keep = valid[:, k:]
        predicted = predictor(omega[:, :-k][keep])
        wanted = omega_target[:, k:][keep].detach()
        jepa = jepa_loss(predicted, wanted)

        flat = omega[valid]
        sample = flat[torch.randperm(len(flat), device=device)[:512]]
        sigreg, lambda_sigreg, batch_rank = sigreg_term(sample, step, args.lambda_sigreg, args.sigreg_rank_cap)

        episode_of_frame = torch.arange(len(episodes), device=device)[:, None].expand_as(valid)[valid]
        from workspace_models.networks.omega_objectives import supcon as _supcon

        episode_contrast = _supcon(flat, episode_of_frame[:, None] == episode_of_frame[None, :], tau=args.tau)

        deliberative = flat.sum() * 0.0
        stat = {"top1": float("nan"), "chance": float("nan"), "lift": float("nan"), "n_rows": 0}
        realised, xdom_hits = 0, 0
        seg_flat = seg[valid]
        present = torch.unique(seg_flat[seg_flat >= 0])
        if len(present) > 1:
            weight, contrast, negative_multiplier, xdom_hits = edges.submatrix(present, corpus.n_segments)
            realised = int((weight > 0).sum()) // 2
            local = torch.full((corpus.n_segments,), 0, dtype=torch.int64, device=device)
            local[present] = torch.arange(len(present), device=device)
            has_segment = seg_flat >= 0
            segment_of = local[seg_flat.clamp_min(0)]
            if realised > 0:
                masked_weight = weight.clone()
                usable = flat[has_segment]
                if lambda_del > 0:
                    deliberative = supcon_deliberative(
                        usable,
                        segment_of[has_segment],
                        masked_weight,
                        contrast,
                        tau=args.tau,
                        contrast_weight=contrast_weight,
                        negative_multiplier_by_segment=negative_multiplier,
                    )
                positive = masked_weight[segment_of[has_segment]][:, segment_of[has_segment]] > 0
                task_of = corpus.segments["task"]
                tasks = torch.from_numpy(task_of[seg_flat[has_segment].cpu().numpy()].astype(np.int64)).to(device)
                stat = supcon_discriminative_stat(usable, positive, candidate=(tasks[:, None] != tasks[None, :]))

        edges_realised_total += realised
        # `xdom_hits` counts DIRECTED entries (both_ways duplicates every pair), `realised` counts
        # UNDIRECTED pairs — halve so the ratio below is a fraction and not 2x one.
        xdom_realised_total += int(xdom_hits) // 2
        batch_domain_counts += np.bincount(domain.detach().cpu().numpy().astype(np.int64), minlength=len(DOMAINS))

        loss = jepa + lambda_sigreg * sigreg + args.lambda_episode * episode_contrast + lambda_del * deliberative

        optimiser.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
        else:
            print(f"[warn] non-finite loss at step {step} — skipped", flush=True)
        with torch.no_grad():
            for tp, ep in zip(target.parameters(), encoder.parameters()):
                tp.mul_(args.ema).add_(ep, alpha=1 - args.ema)

        if step % 50 == 0 or step == 1:
            print(
                f"step {step} loss {loss.item():.4f} jepa {jepa.item():.4f} "
                f"sig {sigreg.item():.4f}(l{lambda_sigreg:.3f} r{batch_rank:.1f}) "
                f"ep {episode_contrast.item():.4f} del {deliberative.item():.4f} "
                f"| edges {realised} xdom {xdom_hits} "
                f"| del_top1 {stat['top1']:.3f} chance {stat['chance']:.3f} "
                f"lift {stat['lift']:.2f} lr {lr:.2e}",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            encoder.eval()
            entry = {
                "step": step,
                "minutes": round((time.time() - started) / 60, 2),
                "train": {
                    "loss": loss.item(),
                    "jepa": jepa.item(),
                    "sigreg": sigreg.item(),
                    "supcon_episode": episode_contrast.item(),
                    "supcon_deliberative": deliberative.item(),
                    "edges_realised": realised,
                    "xdom_edge_hits": xdom_hits,
                    "del_discriminative": stat,
                },
            }
            per_domain = {}
            for name in taps:
                index = DOMAINS.index(name)
                rows = corpus.heldout_episodes[corpus.domain_of_episode[corpus.heldout_episodes] == index]
                if len(rows) < 2:
                    continue
                metrics = g1b_metrics(omega_for_episodes(encoder, corpus, rows), step)
                per_domain[name] = {**metrics, "g1b": g1b_verdict(metrics, g1b_bar_for(name, multi_domain))}
            entry["g1b_per_domain"] = per_domain
            entry["retrieval_gate"] = retrieval_gate(encoder, corpus, device, rng)
            encoder.train()
            history.append(entry)
            (out_dir / "history.json").write_text(json.dumps(history, indent=1))
            print(
                "[eval] "
                + json.dumps(
                    {
                        "step": step,
                        "g1b": {k: v["g1b"]["verdict"] for k, v in per_domain.items()},
                        "coh": {k: round(v["temporal_coherence_gap"], 3) for k, v in per_domain.items()},
                        "erank": {k: round(v["effective_rank"], 2) for k, v in per_domain.items()},
                        "bevf": {k: round(v["between_episode_variance_fraction"], 3) for k, v in per_domain.items()},
                        "retr": {
                            kk: entry["retrieval_gate"].get(kk) for kk in ("top1", "chance", "lift", "beats_chance")
                        },
                    }
                ),
                flush=True,
            )

            payload = {
                **encoder.state_payload(),
                "cfg": vars(cfg),
                "step": step,
                "feat_scale": 1.0,
                "encoder_id": encoder_id,
                "run_config": run_config,
                "eval": entry,
            }
            torch.save(payload, out_dir / "encoder.pt")
            best = max(
                history,
                key=lambda h: max((d["temporal_coherence_gap"] for d in h["g1b_per_domain"].values()), default=-1),
            )
            if best["step"] == step:
                torch.save(payload, out_dir / "encoder_best.pt")

    # ---- final gates ------------------------------------------------------------------------
    encoder.eval()
    frozen_control, collapse_control, delta = {}, {}, {}
    final_domains = history[-1]["g1b_per_domain"] if history else {}
    for name in taps:
        index = DOMAINS.index(name)
        rows = corpus.heldout_episodes[corpus.domain_of_episode[corpus.heldout_episodes] == index]
        if len(rows) < 2:
            continue
        untrained = omega_for_episodes(frozen, corpus, rows)
        metrics = g1b_metrics(untrained, args.steps)
        frozen_control[name] = {**metrics, "g1b": g1b_verdict(metrics, g1b_bar_for(name, multi_domain))}
        # A COLLAPSE control the predicate must reject: every frame mapped to one vector. The
        # untrained encoder is NOT such a control on pooled-token input (AdaLN-Zero initialises to
        # near-identity, so it inherits the tap's own temporal structure) — this one is, and it is
        # what proves the bar still functions as a collapse detector here.
        mean = np.concatenate(untrained).mean(0, keepdims=True)
        collapsed = [
            np.repeat(mean, len(e), 0)
            + 1e-4 * np.random.default_rng(0).standard_normal((len(e), mean.shape[1])).astype(np.float32)
            for e in untrained
        ]
        collapse_metrics = g1b_metrics(collapsed, args.steps)
        collapse_control[name] = {
            **collapse_metrics,
            "g1b": g1b_verdict(collapse_metrics, g1b_bar_for(name, multi_domain)),
        }
        if name in final_domains:
            delta[name] = {
                key: round(float(final_domains[name][key]) - float(metrics[key]), 4)
                for key in ("temporal_coherence_gap", "effective_rank", "between_episode_variance_fraction")
            }
    gates = {
        "encoder_id": encoder_id,
        "cell": args.cell,
        "label_id": labels_dir.name,
        "final": history[-1] if history else None,
        "frozen_negative_control": frozen_control,
        "collapse_control": collapse_control,
        "delta_vs_untrained": delta,
        # PRE-REGISTERED 2026-08-28, before any full cell ran: on pooled-token input the a2/G1b bar
        # is a COLLAPSE floor, not a discriminator (an untrained AdaLN-Zero encoder already clears
        # 2 of its 3 PASS thresholds), so the discriminating criterion is the DELTA on the axis a2
        # showed SupCon alone moves — between-episode variance fraction — plus the frame-level
        # retrieval gate, which amendment A2 names as the go/no-go.
        "bevf_delta_floor": 0.10,
        "bevf_delta_passes": {k: bool(v["between_episode_variance_fraction"] >= 0.10) for k, v in delta.items()},
        "frozen_control_trips_fail": all(v["g1b"]["verdict"] == "FAIL" for v in frozen_control.values())
        if frozen_control
        else None,
        "collapse_control_trips_fail": all(v["g1b"]["verdict"] == "FAIL" for v in collapse_control.values())
        if collapse_control
        else None,
        "decode_grounding": decode_grounding(encoder, corpus, device, Path(args.label_store).expanduser()),
        "bar": G1B_BAR,
        "bar_per_domain": {name: g1b_bar_for(name, multi_domain) for name in taps},
        "multi_domain": multi_domain,
        "edges": {
            "set": spec["edges"],
            "n_positive": edges.n_positive,
            "n_contrast": edges.n_contrast,
            "n_binding_corroborated": edges.n_binding,
            "mean_hardneg_strength": round(edges.mean_hardneg, 4),
            "binding_only": bool(spec.get("binding_only", False)),
            "contrast_weight": contrast_weight,
            # A9: which cells actually let the contested CONTRAST verdicts act as HARD
            # negatives. n_contrast > 0 alone is not consumption — a weight of 1.0 makes them
            # ordinary negatives, and lambda_del = 0 makes the whole term inert.
            "consumed_contrast_as_hard_negative": bool(
                edges.n_contrast > 0 and contrast_weight > 1.0 and lambda_del > 0
            ),
            "cross_domain_positive_frac": round(edges.cross_domain_frac, 4),
            # What the objective ACTUALLY consumed, summed over every training step: a
            # cross-domain positive only acts if both its segments land in the same batch.
            "edges_realised_total": int(edges_realised_total),
            "xdom_realised_total": int(xdom_realised_total),
            "in_batch_cross_domain_frac": round(xdom_realised_total / max(edges_realised_total, 1), 5),
            "batch_episode_domain_counts": {
                DOMAINS[i]: int(batch_domain_counts[i]) for i in range(len(DOMAINS)) if batch_domain_counts[i]
            },
        },
        "corpus": {
            "episodes": corpus.n_episodes,
            "frames": int(len(corpus.feat)),
            "per_domain_episodes": {
                name: int((corpus.domain_of_episode == DOMAINS.index(name)).sum()) for name in taps
            },
            "missing": corpus.missing,
        },
    }
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=1, default=str))

    # ---- the retrieval gate is the PRE-REGISTERED go/no-go (A2), so it may not no-op ------------
    # n_anchors == 0 previously carried only a "note" while the cell went on to write a gates.json
    # that LOOKS complete -- the silent-success class applied to the one number the cell is selected
    # on. Fatal for any non-smoke cell; smoke runs keep a warning so a 20-step bench check works.
    SMOKE_STEPS = 1000
    retr = gates["final"]["retrieval_gate"] if gates["final"] else None
    if retr is not None:
        n_anchors = int(retr.get("n_anchors", 0) or 0)
        by_dom = retr.get("anchors_by_domain") or {}
        problems = []
        if n_anchors == 0:
            problems.append(
                f"n_anchors == 0 ({retr.get('note', 'no note')}); disagreement pairs in artifact = "
                f"{retr.get('n_disagreement_pairs_total', 'n/a')}"
            )
        # A domain must not pass by contributing zero gate pairs while the others carry the number.
        starved = sorted(d for d in taps if int(by_dom.get(d, 0)) == 0)
        if starved:
            problems.append(
                f"loaded taps {sorted(taps)} but {starved} scored ZERO anchors (anchors_by_domain={by_dom})"
            )
        if problems:
            msg = (
                "[gates] retrieval gate is the pre-registered go/no-go and did not evaluate: "
                + "; ".join(problems)
                + f". label artifact = {corpus.labels_dir}"
            )
            if args.steps >= SMOKE_STEPS:
                raise SystemExit("FATAL " + msg)
            print(f"WARNING (smoke run, steps={args.steps} < {SMOKE_STEPS}) {msg}", flush=True)
    print(
        "[gates] "
        + json.dumps(
            {
                "cell": args.cell,
                "g1b": {k: v["g1b"]["verdict"] for k, v in (history[-1]["g1b_per_domain"].items() if history else [])},
                "delta_vs_untrained": gates["delta_vs_untrained"],
                "bevf_delta_passes": gates["bevf_delta_passes"],
                "collapse_control_trips_fail": gates["collapse_control_trips_fail"],
                "frozen_control_trips_fail": gates["frozen_control_trips_fail"],
                "retrieval": gates["final"]["retrieval_gate"] if gates["final"] else None,
                "decode": gates["decode_grounding"],
            },
            indent=1,
            default=str,
        ),
        flush=True,
    )

    if args.export_omega:
        omega_root = Path(args.export_omega).expanduser()
        export_omega_store(encoder, corpus, omega_root, encoder_id, device, encoder_step=int(args.steps))
        # BINDING DECODABILITY (pre-registered 2026-08-28) — the Markovianization signature. Run on
        # the EXPORTED store rather than in-graph so the identical code path scores every cell, old
        # and new. Report metric and floor; selection stays on retrieval + decode.
        try:
            from scripts.deliberation.binding_decodability import evaluate as binding_evaluate

            gates["binding_decodability"] = binding_evaluate(
                omega_root, Path(args.bindings).expanduser(), Path(args.pass1_robocasa).expanduser()
            )
            (out_dir / "gates.json").write_text(json.dumps(gates, indent=1, default=str))
            print("[binding] " + json.dumps(gates["binding_decodability"]["signature_rows"]), flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"[binding] skipped: {error}", flush=True)
    print(f"done in {(time.time() - started) / 60:.1f} min -> {out_dir}", flush=True)


def load_task_lang_tables(entries: list) -> dict:
    """['robocasa=/path/table.npz', ...] -> {domain: {task: vector}}."""
    out = {}
    for entry in entries or []:
        dom, _, path = str(entry).partition("=")
        if dom not in DOMAINS or not path:
            raise SystemExit(f"--task-lang-table expects domain=path, got {entry!r}")
        blob = np.load(Path(path).expanduser(), allow_pickle=False)
        out[dom] = {k: np.asarray(blob[k], np.float32) for k in blob.files}
    return out


@torch.no_grad()
def export_omega_store(
    encoder, corpus: Corpus, root: Path, encoder_id: str, device, encoder_step: int | None = None
) -> None:
    """Write the ω store in the EXACT schema the GDN read consumes today
    (`wsm_policy_feats/<enc>/<Task>/demo_%06d/w.npz` = {w [F,512] fp16, frame_indices, lang_global}),
    so a Stage-E encoder is a drop-in for the existing post-train path."""
    root.mkdir(parents=True, exist_ok=True)
    written = 0
    for episode_index, (domain, _task, ep, task) in enumerate(corpus.meta):
        episodes = torch.tensor([episode_index], dtype=torch.int64, device=device)
        feat, lang, _seg, valid, domain_index = corpus.gather(episodes)
        omega = encoder(feat, lang, domain_index)[0][valid[0]].half().cpu().numpy()
        frame_indices = np.asarray(corpus.frame_index[episode_index], np.int64)
        out = root / DOMAINS[domain] / task / f"demo_{ep:06d}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "w.npz",
            w=omega,
            frame_indices=frame_indices,
            lang_global=corpus.lang[episode_index].cpu().numpy(),
            encoder_id=np.array(encoder_id),
        )
        written += 1
    (root / "_meta.json").write_text(
        json.dumps(
            {
                "encoder_id": encoder_id,
                "n_episodes": written,
                "omega_dim": 512,
                # The D7 gate must be pointed at the checkpoint that PRODUCED this store. `encoder_best.pt`
                # is the best-eval step, not the final one, so parity against it compares two different
                # models and fails on a correct encoder (measured 2026-09-02: best=750 vs store=1500 gave
                # cos 0.13; the final checkpoint gave PASS).
                "encoder_step": encoder_step,
                "lang_modes": getattr(corpus, "lang_modes", {}),
                "lang_sources": getattr(corpus, "lang_source", {}),
                "schema": "w.npz {w [F,512] fp16, frame_indices [F] int64, lang_global, encoder_id}",
            },
            indent=1,
        )
    )
    print(f"[omega] wrote {written} episodes -> {root}", flush=True)


if __name__ == "__main__":
    main()
