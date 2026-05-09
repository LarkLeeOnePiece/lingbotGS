"""
GaussianMapManager - Three-level Gaussian map mirroring GCA's context structure.

Levels:
    Anchor   — from the first *n* scale frames.  Dense, fixed forever.
    Window   — from the most recent *k* frames.  Actively updated.
    Memory   — from evicted frames, compacted via voxel-hashing.  Budget-limited.

Eviction-triggered compaction:  when a frame exits the sliding window its
Gaussians are merged with the existing memory pool using confidence-weighted
voxel averaging, then low-opacity / over-budget Gaussians are pruned.

Memory budget analysis (default settings, 518×378 input):
    Anchor  3 frames × ~999 surfels  =   ~3 K     144 KB
    Window 64 frames × ~999 surfels  =  ~64 K     3.1 MB
    Memory budget                     = 100 K     4.8 MB
    ──────────────────────────────────────────────────────
    Total (steady state)              ≈ 167 K     ~8 MB

    Contrast: per-pixel over 3 000 frames → 588 M Gaussians / ~33 GB.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional


# =====================================================================
# Data container
# =====================================================================

@dataclass
class GaussianData:
    """Flat tensor container for a set of 2D Gaussian surfels."""

    positions: torch.Tensor      # [N, 3]
    opacities: torch.Tensor      # [N, 1]
    scales: torch.Tensor         # [N, 2]
    normals: torch.Tensor        # [N, 3]
    colors: torch.Tensor         # [N, 3]
    confidences: torch.Tensor    # [N, 1]  (used as merge weight)

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    # ---- helpers ----

    def to(self, device: torch.device) -> GaussianData:
        return GaussianData(
            positions=self.positions.to(device),
            opacities=self.opacities.to(device),
            scales=self.scales.to(device),
            normals=self.normals.to(device),
            colors=self.colors.to(device),
            confidences=self.confidences.to(device),
        )

    def filter(self, mask: torch.Tensor) -> GaussianData:
        """Keep only rows where *mask* is True."""
        return GaussianData(
            positions=self.positions[mask],
            opacities=self.opacities[mask],
            scales=self.scales[mask],
            normals=self.normals[mask],
            colors=self.colors[mask],
            confidences=self.confidences[mask],
        )

    @staticmethod
    def cat(parts: List[GaussianData]) -> Optional[GaussianData]:
        """Concatenate a list of GaussianData along dim-0."""
        parts = [p for p in parts if p is not None and p.num_gaussians > 0]
        if not parts:
            return None
        return GaussianData(
            positions=torch.cat([p.positions for p in parts]),
            opacities=torch.cat([p.opacities for p in parts]),
            scales=torch.cat([p.scales for p in parts]),
            normals=torch.cat([p.normals for p in parts]),
            colors=torch.cat([p.colors for p in parts]),
            confidences=torch.cat([p.confidences for p in parts]),
        )

    @staticmethod
    def empty(device: torch.device, dtype: torch.dtype = torch.float32) -> GaussianData:
        """Return an empty GaussianData on *device*."""
        z = lambda c: torch.zeros(0, c, device=device, dtype=dtype)
        return GaussianData(
            positions=z(3), opacities=z(1), scales=z(2),
            normals=z(3), colors=z(3), confidences=z(1),
        )


# =====================================================================
# Map manager
# =====================================================================

class GaussianMapManager:
    """Three-level Gaussian map aligned with GCA's context structure.

    Args:
        num_anchor_frames: Number of anchor / scale frames whose Gaussians
            are kept forever at full density.
        window_size: Maximum number of frames in the active window pool.
        memory_budget: Hard cap on the number of memory-pool Gaussians.
        voxel_size: Edge length (metres) for spatial hashing during compaction.
        opacity_threshold: Gaussians below this opacity are discarded on entry.
        device: Storage device.
    """

    def __init__(
        self,
        num_anchor_frames: int = 3,
        window_size: int = 64,
        memory_budget: int = 100_000,
        voxel_size: float = 0.05,
        opacity_threshold: float = 0.1,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_anchor_frames = num_anchor_frames
        self.window_size = window_size
        self.memory_budget = memory_budget
        self.voxel_size = voxel_size
        self.opacity_threshold = opacity_threshold
        self.device = device

        # pools
        self.anchor_gaussians: Optional[GaussianData] = None
        self.window_frames: List[tuple] = []        # [(frame_idx, GaussianData), ...]
        self.memory_gaussians: Optional[GaussianData] = None
        self.frame_count: int = 0

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.anchor_gaussians = None
        self.window_frames.clear()
        self.memory_gaussians = None
        self.frame_count = 0

    # ------------------------------------------------------------- add_frame
    def add_frame(
        self,
        frame_idx: int,
        positions: torch.Tensor,     # [M, 3]
        opacities: torch.Tensor,     # [M, 1]
        scales: torch.Tensor,        # [M, 2]
        normals: torch.Tensor,       # [M, 3]
        colors: torch.Tensor,        # [M, 3]
        depth_conf: Optional[torch.Tensor] = None,  # [M] or [M,1]
    ):
        """Register Gaussians produced by a single frame."""
        valid = opacities.squeeze(-1) > self.opacity_threshold

        conf = (
            depth_conf.reshape(-1, 1).to(opacities.dtype)
            if depth_conf is not None
            else opacities.detach().clone()
        )

        gs = GaussianData(
            positions=positions[valid].detach(),
            opacities=opacities[valid].detach(),
            scales=scales[valid].detach(),
            normals=normals[valid].detach(),
            colors=colors[valid].detach(),
            confidences=conf[valid].detach(),
        )

        if gs.num_gaussians == 0:
            self.frame_count += 1
            return

        if frame_idx < self.num_anchor_frames:
            self.anchor_gaussians = GaussianData.cat(
                [self.anchor_gaussians, gs]
            )
        else:
            self.window_frames.append((frame_idx, gs))
            while len(self.window_frames) > self.window_size:
                _, evicted = self.window_frames.pop(0)
                self._compact_to_memory(evicted)

        self.frame_count += 1

    # ------------------------------------------------ compaction / merging
    def _compact_to_memory(self, evicted: GaussianData):
        """Merge *evicted* Gaussians into the memory pool."""
        if evicted.num_gaussians == 0:
            return

        if self.memory_gaussians is None or self.memory_gaussians.num_gaussians == 0:
            self.memory_gaussians = evicted
        else:
            combined = GaussianData.cat([self.memory_gaussians, evicted])
            self.memory_gaussians = self._voxel_merge(combined)

        self._enforce_budget()

    def _voxel_merge(self, data: GaussianData) -> GaussianData:
        """Confidence-weighted merge of Gaussians sharing the same voxel."""
        coords = torch.floor(data.positions / self.voxel_size).long()

        # Spatial hash → 1-D key per Gaussian
        keys = (
            coords[:, 0] * 73856093
            + coords[:, 1] * 19349663
            + coords[:, 2] * 83492791
        )
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        n_voxels = unique_keys.shape[0]

        if n_voxels == data.num_gaussians:
            return data  # nothing to merge

        dev, dt = data.positions.device, data.positions.dtype
        idx = inverse.unsqueeze(-1)          # [N, 1]
        w = data.confidences                 # [N, 1]

        def _scatter_weighted(vals: torch.Tensor, dim: int) -> torch.Tensor:
            buf = torch.zeros(n_voxels, dim, device=dev, dtype=dt)
            buf.scatter_add_(0, idx.expand(-1, dim), vals * w)
            return buf

        w_sum = torch.zeros(n_voxels, 1, device=dev, dtype=dt)
        w_sum.scatter_add_(0, idx, w)
        w_sum = w_sum.clamp(min=1e-8)

        m_pos = _scatter_weighted(data.positions, 3) / w_sum
        m_opa = _scatter_weighted(data.opacities, 1) / w_sum
        m_sca = _scatter_weighted(data.scales, 2) / w_sum
        m_nor = F.normalize(
            _scatter_weighted(data.normals, 3), dim=-1, eps=1e-6
        )
        m_col = _scatter_weighted(data.colors, 3) / w_sum

        # Confidence: sum within voxel (more observations ⇒ higher confidence)
        m_conf = torch.zeros(n_voxels, 1, device=dev, dtype=dt)
        m_conf.scatter_add_(0, idx, data.confidences)

        return GaussianData(
            positions=m_pos, opacities=m_opa, scales=m_sca,
            normals=m_nor, colors=m_col, confidences=m_conf,
        )

    def _enforce_budget(self):
        """Prune lowest-confidence memory Gaussians to stay within budget."""
        if self.memory_gaussians is None:
            return
        n = self.memory_gaussians.num_gaussians
        if n <= self.memory_budget:
            return
        conf = self.memory_gaussians.confidences.squeeze(-1)
        _, keep = conf.topk(self.memory_budget)
        mask = torch.zeros(n, dtype=torch.bool, device=conf.device)
        mask[keep] = True
        self.memory_gaussians = self.memory_gaussians.filter(mask)

    # -------------------------------------------------------- accessors
    def get_all_gaussians(self) -> Optional[GaussianData]:
        """Return the full Gaussian map (anchor + window + memory)."""
        parts: list = []
        if self.anchor_gaussians is not None:
            parts.append(self.anchor_gaussians)
        for _, gs in self.window_frames:
            parts.append(gs)
        if self.memory_gaussians is not None:
            parts.append(self.memory_gaussians)
        return GaussianData.cat(parts)

    def get_stats(self) -> Dict[str, int]:
        a = self.anchor_gaussians.num_gaussians if self.anchor_gaussians else 0
        w = sum(g.num_gaussians for _, g in self.window_frames)
        m = self.memory_gaussians.num_gaussians if self.memory_gaussians else 0
        return {
            "anchor_gaussians": a,
            "window_gaussians": w,
            "window_frames": len(self.window_frames),
            "memory_gaussians": m,
            "total_gaussians": a + w + m,
            "frames_processed": self.frame_count,
        }
