"""WSM-v2 head: predict 3D flow / tracks from the workspace latent ``w_t`` (DynaFLIP target).

Decodes ``w_t`` into future 3D point tracks for a set of query points — the DynaFLIP target (grid
tracks + depth/pose + unprojection with camera motion canceled). This is the *second* decoder that
sits alongside the keyframe-patch head in WSM v2. Governed by
internal_planning_and_todos/04_wsm_roadmap.md ("R3 = E2 — 3D flow (WSM v2)" + "3D-flow / track head").

TODO
----
- [ ] Define the query-point parameterization (grid seeds) and the horizon T of predicted tracks.
- [ ] Implement the decoder ``w_t`` (+ query coords) -> (B, num_tracks, T, 3) world-frame tracks.
- [ ] Define the target format (camera-motion-canceled 3D tracks) + the loss (e.g. masked endpoint L2).
- [ ] Expose track endpoint error / visibility metrics for train/val.
"""

from __future__ import annotations

import torch
from torch import nn


class Flow3DHead(nn.Module):
    """Decode ``w_t`` -> future 3D point tracks for query points (PLACEHOLDER)."""

    def __init__(self, latent_dim: int, num_tracks: int, horizon: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tracks = num_tracks
        self.horizon = horizon
        # TODO: decoder layers producing (B, num_tracks, horizon, 3) tracks (+ visibility logits).

    def forward(self, w_t: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """w_t (B, latent_dim) + query_points (B, num_tracks, 3) -> tracks (B, num_tracks, T, 3)."""
        raise NotImplementedError("TODO: decode w_t into 3D point tracks")

    def loss(self, tracks: torch.Tensor, targets: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        """TODO: masked 3D-track regression loss against the DynaFLIP labels."""
        raise NotImplementedError("TODO: 3D-flow track prediction loss")
