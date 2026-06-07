"""
AnchorFeatureHead - Dense feature map extraction via DPT (feature_only mode).

Wraps DPTHead with feature_only=True to produce dense feature maps [B, S, C, H, W]
from aggregated transformer tokens. These features are sampled at anchor locations
for multi-view feature aggregation.
"""

from lingbot_map.heads.dpt_head import DPTHead


class AnchorFeatureHead(DPTHead):
    """DPT head that outputs dense feature maps instead of depth/points.

    Output shape: [B, S, features, H, W] where features=256 by default.
    Uses the same multi-scale fusion as the depth head but stops before
    the final prediction layers.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        features: int = 256,
    ):
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=1,  # unused in feature_only mode
            activation="exp",
            conf_activation="expp1",
            features=features,
            feature_only=True,
        )
