"""
AnchorGaussianDecoder - Predict K 2D Gaussian surfels per anchor point.

Each anchor's aggregated feature vector is decoded into K sets of Gaussian
attributes. Positions are computed as anchor_position + learned_offset.

Reuses the same activation patterns as CompactGaussianHead:
    - Position offset: tanh * max_offset
    - Opacity: sigmoid
    - Scale: depth-adaptive base * exp(learned_correction)
    - Normal: L2 normalized
    - Color: sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class AnchorGaussianDecoder(nn.Module):
    """Predict K Gaussian surfels per 3D anchor from its feature vector.

    Per-surfel attributes (12 params):
        position_offset [3]: constrained refinement from anchor position
        opacity [1]: sigmoid activated
        scale [2]: 2D tangent-plane scales (depth-adaptive)
        normal [3]: surface normal (L2 normalized)
        color [3]: RGB (sigmoid activated)

    Args:
        feature_dim: Input feature dimension (from AnchorFeatureHead).
        hidden_dim: MLP hidden dimension.
        K: Number of Gaussians per anchor.
        max_pos_offset: Maximum position offset in metres.
        opacity_bias: Initial bias for opacity logit.
        scale_range: Clamp range for log-scale correction.
    """

    ATTR_DIM = 12  # offset(3) + opacity(1) + scale(2) + normal(3) + color(3)

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        K: int = 4,
        max_pos_offset: float = 0.05,
        opacity_bias: float = -1.0,
        scale_range: Tuple[float, float] = (-1.5, 1.5),
    ):
        super().__init__()
        self.K = K
        self.max_pos_offset = max_pos_offset
        self.scale_range = scale_range

        out_dim = K * self.ATTR_DIM
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Initialize last layer
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
        for k in range(K):
            base = k * self.ATTR_DIM
            self.mlp[-1].bias.data[base + 3] = opacity_bias  # start with low opacity

    def forward(
        self,
        anchor_positions: torch.Tensor,  # [A, 3]
        anchor_features: torch.Tensor,   # [A, C]
        anchor_depths: torch.Tensor,     # [A] representative depth
        focal_length: float,             # average focal length
        patch_size: int = 14,            # for scale computation
    ) -> Dict[str, torch.Tensor]:
        """Predict K Gaussian surfels per anchor.

        Returns dict with keys:
            positions:  [A, K, 3]  world coordinates
            opacities:  [A, K, 1]
            scales:     [A, K, 2]  tangent-plane scales
            normals:    [A, K, 3]
            colors:     [A, K, 3]
        """
        A = anchor_positions.shape[0]
        if A == 0:
            dev = anchor_positions.device
            return {
                "positions": torch.zeros(0, self.K, 3, device=dev),
                "opacities": torch.zeros(0, self.K, 1, device=dev),
                "scales": torch.zeros(0, self.K, 2, device=dev),
                "normals": torch.zeros(0, self.K, 3, device=dev),
                "colors": torch.zeros(0, self.K, 3, device=dev),
            }

        raw = self.mlp(anchor_features)  # [A, K*12]
        raw = raw.reshape(A, self.K, self.ATTR_DIM)

        # Parse and activate
        pos_offset = self.max_pos_offset * torch.tanh(raw[..., 0:3])
        opacity = torch.sigmoid(raw[..., 3:4])
        raw_scale = raw[..., 4:6]
        normal = F.normalize(raw[..., 6:9], dim=-1, eps=1e-6)
        color = torch.sigmoid(raw[..., 9:12])

        # Positions = anchor + offset
        positions = anchor_positions.unsqueeze(1) + pos_offset  # [A, K, 3]

        # Depth-adaptive scales
        k_sqrt = max(1, int(self.K ** 0.5 + 0.5))
        target_sigma = patch_size / (2.0 * k_sqrt)
        depth_clamped = anchor_depths.clamp(min=0.1).unsqueeze(-1).unsqueeze(-1)  # [A, 1, 1]
        base_scale = target_sigma * depth_clamped / max(focal_length, 1.0)  # [A, 1, 1]
        correction = torch.exp(raw_scale.clamp(*self.scale_range))  # [A, K, 2]
        scales = base_scale * correction  # [A, K, 2]

        return {
            "positions": positions,
            "opacities": opacity,
            "scales": scales,
            "normals": normal,
            "colors": color,
        }
