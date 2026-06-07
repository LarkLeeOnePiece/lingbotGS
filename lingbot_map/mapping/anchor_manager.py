"""
AnchorManager - Incremental 3D anchor point cloud for streaming Gaussian Splatting.

Anchors are persistent 3D reference points derived from depth predictions.
Each anchor accumulates multi-view features via EMA and serves as the origin
for K Gaussian surfels predicted by AnchorGaussianDecoder.

Key operations:
    init_from_depth:    Phase 1 — create initial anchors from scale frame depth maps
    update_and_expand:  Phase 2 — update visible anchors + add new ones from each frame
    enforce_budget:     Prune lowest-confidence anchors when over budget
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class AnchorData:
    """Flat tensor container for 3D anchor points."""

    positions: torch.Tensor       # [A, 3]  world-space coordinates
    features: torch.Tensor        # [A, C]  accumulated feature vectors
    obs_counts: torch.Tensor      # [A]     number of observations
    confidences: torch.Tensor     # [A]     running average depth confidence
    frame_created: torch.Tensor   # [A]     frame index when created

    @property
    def num_anchors(self) -> int:
        return self.positions.shape[0]

    def to(self, device: torch.device) -> AnchorData:
        return AnchorData(
            positions=self.positions.to(device),
            features=self.features.to(device),
            obs_counts=self.obs_counts.to(device),
            confidences=self.confidences.to(device),
            frame_created=self.frame_created.to(device),
        )

    def filter(self, mask: torch.Tensor) -> AnchorData:
        return AnchorData(
            positions=self.positions[mask],
            features=self.features[mask],
            obs_counts=self.obs_counts[mask],
            confidences=self.confidences[mask],
            frame_created=self.frame_created[mask],
        )

    @staticmethod
    def cat(parts: list) -> Optional[AnchorData]:
        parts = [p for p in parts if p is not None and p.num_anchors > 0]
        if not parts:
            return None
        return AnchorData(
            positions=torch.cat([p.positions for p in parts]),
            features=torch.cat([p.features for p in parts]),
            obs_counts=torch.cat([p.obs_counts for p in parts]),
            confidences=torch.cat([p.confidences for p in parts]),
            frame_created=torch.cat([p.frame_created for p in parts]),
        )

    @staticmethod
    def empty(device: torch.device, feature_dim: int = 256) -> AnchorData:
        return AnchorData(
            positions=torch.zeros(0, 3, device=device),
            features=torch.zeros(0, feature_dim, device=device),
            obs_counts=torch.zeros(0, device=device),
            confidences=torch.zeros(0, device=device),
            frame_created=torch.zeros(0, dtype=torch.long, device=device),
        )


class AnchorManager:
    """Incremental 3D anchor set for streaming anchor-aligned Gaussian Splatting.

    Args:
        voxel_size: Edge length (metres) for spatial deduplication.
        anchor_budget: Maximum number of anchors.
        feature_dim: Dimensionality of per-anchor feature vectors.
        depth_consistency_thresh: Relative depth threshold for visibility check.
        max_obs_count: Cap on observation count (for EMA momentum control).
        min_depth_conf: Minimum depth confidence to create an anchor.
        novelty_radius_px: Pixel radius for novelty detection.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        anchor_budget: int = 80_000,
        feature_dim: int = 256,
        depth_consistency_thresh: float = 0.2,
        max_obs_count: int = 20,
        min_depth_conf: float = 1.5,
        novelty_radius_px: float = 21.0,
        device: torch.device = torch.device("cuda"),
    ):
        self.voxel_size = voxel_size
        self.anchor_budget = anchor_budget
        self.feature_dim = feature_dim
        self.depth_consistency_thresh = depth_consistency_thresh
        self.max_obs_count = max_obs_count
        self.min_depth_conf = min_depth_conf
        self.novelty_radius_px = novelty_radius_px
        self.device = device

        self.anchors: Optional[AnchorData] = None
        # Cache existing voxel keys as a tensor for fast dedup
        self._existing_voxel_keys: Optional[torch.Tensor] = None

    def reset(self):
        self.anchors = None
        self._existing_voxel_keys = None

    def _invalidate_voxel_cache(self):
        self._existing_voxel_keys = None

    def _get_existing_voxel_keys(self) -> torch.Tensor:
        if self._existing_voxel_keys is None and self.anchors is not None:
            self._existing_voxel_keys = self._voxel_hash(self.anchors.positions)
        return self._existing_voxel_keys

    # ------------------------------------------------------------------
    # Initialization from depth maps (Phase 1)
    # ------------------------------------------------------------------

    def init_from_depth(
        self,
        world_points: torch.Tensor,   # [N, 3]
        features: torch.Tensor,       # [N, C]
        confidences: torch.Tensor,    # [N]
        frame_indices: torch.Tensor,  # [N]
    ):
        """Create initial anchor set from unprojected depth points.

        Applies voxel-grid downsampling to avoid redundancy.
        """
        # Filter by confidence
        valid = confidences > self.min_depth_conf
        if not valid.any():
            self.anchors = AnchorData.empty(self.device, self.feature_dim)
            return

        pts = world_points[valid]
        feats = features[valid]
        confs = confidences[valid]
        frames = frame_indices[valid]

        # Voxel-grid downsampling: keep one anchor per voxel (highest confidence)
        positions, feat_out, conf_out, frame_out = self._voxel_downsample(
            pts, feats, confs, frames
        )

        self.anchors = AnchorData(
            positions=positions,
            features=feat_out,
            obs_counts=torch.ones(len(positions), device=self.device),
            confidences=conf_out,
            frame_created=frame_out,
        )

        self._invalidate_voxel_cache()
        self.enforce_budget()

    # ------------------------------------------------------------------
    # Projection and visibility
    # ------------------------------------------------------------------

    def project_anchors(
        self,
        c2w: torch.Tensor,       # [4, 4] camera-to-world
        intrinsics: torch.Tensor, # [3, 3]
        H: int,
        W: int,
        depth_map: torch.Tensor,  # [H, W] predicted depth for consistency check
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project all anchors into the given camera frame.

        Returns:
            pixel_coords: [A, 2] (u, v) pixel coordinates
            proj_depths: [A] depth of each anchor in camera frame
            visible_mask: [A] bool - True if anchor is visible
        """
        if self.anchors is None or self.anchors.num_anchors == 0:
            empty = torch.zeros(0, device=self.device)
            return empty.reshape(0, 2), empty, empty.bool()

        # World-to-camera transform
        w2c = torch.inverse(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # Transform to camera coordinates
        cam_pts = (self.anchors.positions @ R.T) + t  # [A, 3]
        z = cam_pts[:, 2]

        # Project to pixels
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        u = fx * cam_pts[:, 0] / z.clamp(min=1e-6) + cx
        v = fy * cam_pts[:, 1] / z.clamp(min=1e-6) + cy

        pixel_coords = torch.stack([u, v], dim=-1)

        # Visibility: in front of camera + within image bounds
        in_front = z > 0.1
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # Depth consistency: projected depth vs predicted depth at that pixel
        depth_consistent = torch.zeros_like(in_front)
        check_mask = in_front & in_bounds
        if check_mask.any():
            ui = u[check_mask].long().clamp(0, W - 1)
            vi = v[check_mask].long().clamp(0, H - 1)
            pred_d = depth_map[vi, ui]
            proj_d = z[check_mask]
            rel_diff = (proj_d - pred_d).abs() / pred_d.clamp(min=0.1)
            depth_consistent[check_mask] = rel_diff < self.depth_consistency_thresh

        visible_mask = in_front & in_bounds & depth_consistent

        return pixel_coords, z, visible_mask

    # ------------------------------------------------------------------
    # Feature update (EMA)
    # ------------------------------------------------------------------

    def update_visible(
        self,
        visible_mask: torch.Tensor,  # [A] bool
        new_features: torch.Tensor,  # [A_vis, C]
        new_confs: torch.Tensor,     # [A_vis]
    ):
        """EMA update features and confidence for visible anchors."""
        if self.anchors is None or not visible_mask.any():
            return

        idx = visible_mask.nonzero(as_tuple=True)[0]
        counts = self.anchors.obs_counts[idx]
        alpha = 1.0 / (1.0 + counts).unsqueeze(-1)  # [A_vis, 1]

        self.anchors.features[idx] = (
            (1.0 - alpha) * self.anchors.features[idx] + alpha * new_features
        )
        self.anchors.confidences[idx] = (
            (1.0 - alpha.squeeze(-1)) * self.anchors.confidences[idx]
            + alpha.squeeze(-1) * new_confs
        )
        self.anchors.obs_counts[idx] = (counts + 1).clamp(max=self.max_obs_count)

    # ------------------------------------------------------------------
    # Novel region detection + new anchor creation
    # ------------------------------------------------------------------

    def compute_novelty_mask(
        self,
        pixel_coords: torch.Tensor,  # [A, 2]
        visible_mask: torch.Tensor,   # [A]
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Compute a binary novelty mask [H, W] — True where no anchor projects nearby.

        Uses a scatter-based approach: rasterize visible anchors to a coverage map,
        then dilate by novelty_radius_px. Uncovered pixels are novel.
        """
        coverage = torch.zeros(H, W, device=self.device, dtype=torch.bool)

        if visible_mask.any():
            vis_coords = pixel_coords[visible_mask]
            u = vis_coords[:, 0].long().clamp(0, W - 1)
            v = vis_coords[:, 1].long().clamp(0, H - 1)
            coverage[v, u] = True

            # Dilate using max_pool2d
            r = int(self.novelty_radius_px)
            k = 2 * r + 1
            coverage_f = coverage.float().unsqueeze(0).unsqueeze(0)
            dilated = F.max_pool2d(coverage_f, kernel_size=k, stride=1, padding=r)
            coverage = dilated.squeeze() > 0.5

        return ~coverage  # True = novel

    def add_new_anchors(
        self,
        new_positions: torch.Tensor,   # [N, 3]
        new_features: torch.Tensor,    # [N, C]
        new_confs: torch.Tensor,       # [N]
        frame_idx: int,
    ) -> int:
        """Add new anchors from novel regions, deduplicating via voxel grid.

        Returns number of anchors actually added.
        """
        if len(new_positions) == 0:
            return 0

        # Filter by confidence
        valid = new_confs > self.min_depth_conf
        if not valid.any():
            return 0

        pts = new_positions[valid]
        feats = new_features[valid]
        confs = new_confs[valid]

        # Voxel-dedup against existing anchors
        if self.anchors is not None and self.anchors.num_anchors > 0:
            pts, feats, confs = self._dedup_against_existing(pts, feats, confs)
            if len(pts) == 0:
                return 0

        # Self-dedup among new points
        frames = torch.full((len(pts),), frame_idx, dtype=torch.long, device=self.device)
        pts, feats, confs, frames = self._voxel_downsample(pts, feats, confs, frames)

        new_anchors = AnchorData(
            positions=pts,
            features=feats,
            obs_counts=torch.ones(len(pts), device=self.device),
            confidences=confs,
            frame_created=frames,
        )

        self.anchors = AnchorData.cat([self.anchors, new_anchors])
        self._invalidate_voxel_cache()
        n_added = new_anchors.num_anchors
        self.enforce_budget()
        return n_added

    # ------------------------------------------------------------------
    # Convenience: update + expand in one call
    # ------------------------------------------------------------------

    def update_and_expand(
        self,
        c2w: torch.Tensor,           # [4, 4]
        intrinsics: torch.Tensor,     # [3, 3]
        H: int, W: int,
        depth_map: torch.Tensor,      # [H, W]
        depth_conf: torch.Tensor,     # [H, W]
        dense_features: torch.Tensor, # [C, H, W]
        frame_idx: int,
    ) -> Dict[str, int]:
        """Combined update + expand for a single streaming frame.

        Returns stats dict with keys: visible, updated, novel_pixels, added.
        """
        # 1. Project existing anchors
        pixel_coords, proj_depths, visible_mask = self.project_anchors(
            c2w, intrinsics, H, W, depth_map
        )

        # 2. Sample features for visible anchors
        n_vis = int(visible_mask.sum())
        if n_vis > 0:
            vis_coords = pixel_coords[visible_mask]  # [V, 2]
            sampled_feats = self._sample_features(dense_features, vis_coords, H, W)
            vis_confs = self._sample_scalar(depth_conf, vis_coords, H, W)
            self.update_visible(visible_mask, sampled_feats, vis_confs)

        # 3. Detect novel regions
        novelty_mask = self.compute_novelty_mask(pixel_coords, visible_mask, H, W)

        # 4. Add new anchors from novel pixels (subsampled)
        n_added = 0
        novel_pixels = int(novelty_mask.sum())
        if novel_pixels > 0:
            new_pts, new_feats, new_confs = self._extract_novel_anchors(
                novelty_mask, depth_map, depth_conf, dense_features, c2w, intrinsics, H, W
            )
            n_added = self.add_new_anchors(new_pts, new_feats, new_confs, frame_idx)

        return {
            "visible": n_vis,
            "updated": n_vis,
            "novel_pixels": novel_pixels,
            "added": n_added,
            "total": self.anchors.num_anchors if self.anchors else 0,
        }

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def enforce_budget(self):
        """Prune lowest-confidence anchors to stay within budget."""
        if self.anchors is None:
            return
        n = self.anchors.num_anchors
        if n <= self.anchor_budget:
            return

        # Score: confidence * sqrt(obs_count) — reward well-observed anchors
        score = self.anchors.confidences * self.anchors.obs_counts.sqrt()
        _, keep_idx = score.topk(self.anchor_budget)
        mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask[keep_idx] = True
        self.anchors = self.anchors.filter(mask)
        self._invalidate_voxel_cache()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        if self.anchors is None:
            return {"num_anchors": 0, "avg_obs": 0, "avg_conf": 0}
        return {
            "num_anchors": self.anchors.num_anchors,
            "avg_obs": float(self.anchors.obs_counts.mean()),
            "avg_conf": float(self.anchors.confidences.mean()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _voxel_hash(self, positions: torch.Tensor) -> torch.Tensor:
        """Spatial hash of positions into 1D keys."""
        coords = torch.floor(positions / self.voxel_size).long()
        return coords[:, 0] * 73856093 + coords[:, 1] * 19349663 + coords[:, 2] * 83492791

    def _voxel_downsample(
        self,
        positions: torch.Tensor,
        features: torch.Tensor,
        confidences: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep one point per voxel (highest confidence). Fully vectorized."""
        keys = self._voxel_hash(positions)
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        n_voxels = unique_keys.shape[0]

        if n_voxels == len(positions):
            return positions, features, confidences, frame_indices

        # Vectorized: for each voxel, pick the point with highest confidence
        # Use scatter_max-like approach: sort by (voxel, -confidence), take first per voxel
        # Scatter approach: assign each point's confidence to its voxel, find max
        best_conf = torch.full((n_voxels,), -float('inf'), device=self.device)
        best_conf.scatter_reduce_(0, inverse, confidences, reduce='amax')

        # Find which points match their voxel's best confidence
        is_best = confidences == best_conf[inverse]
        # Break ties: among equals, take the first one per voxel
        # Use cumcount per voxel to pick first
        if is_best.sum() > n_voxels:
            # Multiple ties - pick first per voxel
            order = torch.arange(len(positions), device=self.device)
            # For each voxel, find min index among best
            best_idx = torch.full((n_voxels,), len(positions), dtype=torch.long, device=self.device)
            # Only consider "is_best" points
            cand_voxels = inverse[is_best]
            cand_order = order[is_best]
            best_idx.scatter_reduce_(0, cand_voxels, cand_order, reduce='amin')
            idx = best_idx[best_idx < len(positions)]
        else:
            idx = is_best.nonzero(as_tuple=True)[0]

        return positions[idx], features[idx], confidences[idx], frame_indices[idx]

    def _dedup_against_existing(
        self,
        new_pts: torch.Tensor,
        new_feats: torch.Tensor,
        new_confs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove new points that fall in already-occupied voxels. Fully vectorized."""
        existing_keys = self._get_existing_voxel_keys()
        new_keys = self._voxel_hash(new_pts)

        # Vectorized set membership: check if new_keys exist in existing_keys
        # Use searchsorted on sorted existing keys
        sorted_existing, _ = existing_keys.sort()
        insert_pos = torch.searchsorted(sorted_existing, new_keys)
        insert_pos = insert_pos.clamp(max=len(sorted_existing) - 1)
        found = sorted_existing[insert_pos] == new_keys

        keep = ~found
        return new_pts[keep], new_feats[keep], new_confs[keep]

    def _sample_features(
        self,
        dense_features: torch.Tensor,  # [C, H, W]
        pixel_coords: torch.Tensor,    # [N, 2] (u, v)
        H: int, W: int,
    ) -> torch.Tensor:
        """Bilinear sample dense features at pixel coordinates. Returns [N, C]."""
        C = dense_features.shape[0]
        u_norm = 2.0 * pixel_coords[:, 0] / (W - 1) - 1.0
        v_norm = 2.0 * pixel_coords[:, 1] / (H - 1) - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).reshape(1, 1, -1, 2)

        sampled = F.grid_sample(
            dense_features.unsqueeze(0),  # [1, C, H, W]
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [1, C, 1, N]
        return sampled.reshape(C, -1).T  # [N, C]

    def _sample_scalar(
        self,
        scalar_map: torch.Tensor,   # [H, W]
        pixel_coords: torch.Tensor,  # [N, 2]
        H: int, W: int,
    ) -> torch.Tensor:
        """Bilinear sample a scalar map. Returns [N]."""
        u_norm = 2.0 * pixel_coords[:, 0] / (W - 1) - 1.0
        v_norm = 2.0 * pixel_coords[:, 1] / (H - 1) - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).reshape(1, 1, -1, 2)

        sampled = F.grid_sample(
            scalar_map.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(-1)

    def _extract_novel_anchors(
        self,
        novelty_mask: torch.Tensor,     # [H, W] bool
        depth_map: torch.Tensor,         # [H, W]
        depth_conf: torch.Tensor,        # [H, W]
        dense_features: torch.Tensor,    # [C, H, W]
        c2w: torch.Tensor,              # [4, 4]
        intrinsics: torch.Tensor,        # [3, 3]
        H: int, W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unproject novel pixels to 3D and sample their features.

        Subsamples to at most ~1000 pixels per frame to control anchor growth rate.
        """
        # Get novel pixel coordinates
        v_idx, u_idx = novelty_mask.nonzero(as_tuple=True)

        # Subsample if too many
        max_novel = 1000
        if len(v_idx) > max_novel:
            perm = torch.randperm(len(v_idx), device=self.device)[:max_novel]
            v_idx, u_idx = v_idx[perm], u_idx[perm]

        # Sample depth and confidence
        d = depth_map[v_idx, u_idx]
        conf = depth_conf[v_idx, u_idx]

        # Unproject to 3D
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        x_cam = (u_idx.float() - cx) / fx * d
        y_cam = (v_idx.float() - cy) / fy * d
        z_cam = d

        cam_pts = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(d)], dim=-1)
        world_pts = (c2w @ cam_pts.T).T[:, :3]  # [N, 3]

        # Sample features
        pixel_coords = torch.stack([u_idx.float(), v_idx.float()], dim=-1)
        feats = self._sample_features(dense_features, pixel_coords, H, W)

        return world_pts, feats, conf
