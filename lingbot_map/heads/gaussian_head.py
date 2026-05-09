"""
CompactGaussianHead - Patch-level 2D Gaussian surfel prediction.

Predicts K 2D Gaussian surfels per patch token, where K is configurable
(default 4 for a 2x2 sub-grid). With ~999 patches per frame (518x378),
this yields ~4000 Gaussians per frame — compact enough for streaming
yet dense enough for sharp rendering.

Design rationale:
- Each DINOv2 patch (14x14 pixels) covers a coherent surface region
- Position is derived from depth at K sub-patch sample points
- A small learned offset refines sub-patch positioning
- 2D surfels (vs 3D Gaussians) are surface-aligned and more parameter-efficient
- Total: K*12 learnable params per patch, K*~999 Gaussians per frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CompactGaussianHead(nn.Module):
    """Predict K 2D Gaussian surfels per patch token.

    Per-surfel attributes (12 params):
        position_offset: [3] - small correction to depth-derived position
        opacity: [1] - sigmoid activated
        scale: [2] - 2D tangent plane scales (exp activated)
        normal: [3] - surface normal (L2 normalized)
        color: [3] - direct RGB (sigmoid activated, SH degree 0)

    Args:
        dim_in: Input dimension from aggregated tokens (2 * embed_dim = 2048)
        hidden_dim: Hidden dimension for attribute MLP
        patch_size: Patch size for computing patch center coordinates
        gaussians_per_patch: K sub-Gaussians per patch (1, 4, or 9)
        max_pos_offset: Maximum position offset magnitude (meters)
        opacity_bias: Initial bias for opacity logit (negative = start sparse)
        scale_range: Clamp range for log-scale before exp activation
    """

    ATTR_DIM = 12  # offset(3) + opacity(1) + scale(2) + normal(3) + color(3)

    def __init__(
        self,
        dim_in: int = 2048,
        hidden_dim: int = 512,
        patch_size: int = 14,
        gaussians_per_patch: int = 4,
        max_pos_offset: float = 0.1,
        opacity_bias: float = -2.0,
        scale_range: Tuple[float, float] = (-7.0, 1.0),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.K = gaussians_per_patch
        self.max_pos_offset = max_pos_offset
        self.scale_range = scale_range

        self.norm = nn.LayerNorm(dim_in)

        out_dim = self.K * self.ATTR_DIM
        self.attr_mlp = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Initialize last layer per sub-Gaussian
        nn.init.normal_(self.attr_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.attr_mlp[-1].bias)
        for k in range(self.K):
            base = k * self.ATTR_DIM
            self.attr_mlp[-1].bias.data[base + 3] = opacity_bias
            # Scale dims (4,5): initialized to 0 since depth-adaptive base
            # handles the absolute scale — MLP learns correction factor only
            self.attr_mlp[-1].bias.data[base + 4] = 0.0
            self.attr_mlp[-1].bias.data[base + 5] = 0.0

        # Cache for pixel coordinates (built lazily)
        self._patch_grid_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._subpatch_grid_cache: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        aggregated_tokens_list: list,
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
        patch_start_idx: int,
        image_size_hw: Tuple[int, int],
        novelty_gate: Optional[torch.Tensor] = None,
        input_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict compact Gaussian surfels from patch tokens.

        Args:
            aggregated_tokens_list: Token tensors from transformer layers.
                Last element has shape [B, S, num_tokens, C].
            depth: Predicted depth [B, S, H, W, 1]
            pose_enc: Camera pose encoding [B, S, 9]  (absT_quaR_FoV)
            patch_start_idx: Index where patch tokens start in the token dim
            image_size_hw: (H, W) of the input images
            novelty_gate: Optional per-patch gating [B, S, M] from overlap
                detection.  Values in [0, 1]; multiplied into opacity.
            input_image: Optional denormalized input image [B, S, 3, H, W]
                in [0, 1] range.  When provided, colors are sampled from the
                image at sub-patch locations instead of predicted by the MLP.

        Returns:
            Dict with keys (N = M * K where K = gaussians_per_patch):
                positions  [B, S, N, 3]  world coordinates
                opacities  [B, S, N, 1]
                scales     [B, S, N, 2]  tangent-plane scales
                normals    [B, S, N, 3]
                colors     [B, S, N, 3]
                valid_mask [B, S, N]     True where opacity > threshold
        """
        tokens = aggregated_tokens_list[-1]
        patch_tokens = tokens[:, :, patch_start_idx:]  # [B, S, M, C]
        B, S, M, C = patch_tokens.shape

        # Predict raw attributes
        normed = self.norm(patch_tokens.float())
        raw = self.attr_mlp(normed)  # [B, S, M, K*12]

        # Reshape to [B, S, M, K, 12] for multi-Gaussian
        raw = raw.reshape(B, S, M, self.K, self.ATTR_DIM)

        # Parse & activate
        pos_offset = self.max_pos_offset * torch.tanh(raw[..., 0:3])
        opacity = torch.sigmoid(raw[..., 3:4])
        normal = F.normalize(raw[..., 6:9], dim=-1, eps=1e-6)

        # Color: sample from input image if available, else predict from MLP
        if input_image is not None:
            color = self._sample_image_colors(input_image, image_size_hw)  # [B, S, M, K, 3]
            # Add small learned residual for view-dependent correction
            color_residual = 0.1 * torch.tanh(raw[..., 9:12])
            color = (color + color_residual).clamp(0, 1)
        else:
            color = torch.sigmoid(raw[..., 9:12])

        if novelty_gate is not None:
            # novelty_gate [B, S, M] → [B, S, M, 1, 1]
            opacity = opacity * novelty_gate.unsqueeze(-1).unsqueeze(-1)

        # 3D positions from depth at sub-patch sample points
        H, W = image_size_hw
        base_pos, depth_at_subpatch = self._unproject_subpatch(
            depth, pose_enc, H, W, M, return_depth=True
        )  # [B, S, M, K, 3], [B, S, M, K]
        positions = base_pos + pos_offset

        # Depth-adaptive scales: base scale ensures ~target_sigma projection
        # target_sigma ≈ patch_size / (2 * sqrt(K)) pixels
        scale = self._compute_adaptive_scale(
            raw[..., 4:6], depth_at_subpatch, pose_enc, H, W
        )  # [B, S, M, K, 2]

        # Flatten M*K → N
        N = M * self.K
        positions = positions.reshape(B, S, N, 3)
        opacity = opacity.reshape(B, S, N, 1)
        scale = scale.reshape(B, S, N, 2)
        normal = normal.reshape(B, S, N, 3)
        color = color.reshape(B, S, N, 3)

        valid_mask = opacity.squeeze(-1) > 0.01

        return {
            "positions": positions,
            "opacities": opacity,
            "scales": scale,
            "normals": normal,
            "colors": color,
            "valid_mask": valid_mask,
        }

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _get_subpatch_grid(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached (uu, vv) sub-patch pixel coordinates.

        For K=1: returns patch centers [ph, pw].
        For K>1: returns sub-grid points [ph, pw, K].
        """
        key = (H, W, self.K)
        if key not in self._subpatch_grid_cache:
            ph = H // self.patch_size
            pw = W // self.patch_size
            off = self.patch_size / 2.0

            u_c = torch.arange(pw, device=device, dtype=dtype) * self.patch_size + off
            v_c = torch.arange(ph, device=device, dtype=dtype) * self.patch_size + off
            vv_c, uu_c = torch.meshgrid(v_c, u_c, indexing="ij")  # [ph, pw]

            if self.K == 1:
                uu_sub = uu_c.unsqueeze(-1)  # [ph, pw, 1]
                vv_sub = vv_c.unsqueeze(-1)
            else:
                # Build sub-grid within each patch (e.g., 2x2 for K=4)
                k_sqrt = int(self.K ** 0.5 + 0.5)
                assert k_sqrt * k_sqrt == self.K, f"K={self.K} must be a perfect square"
                sub_off = self.patch_size / (2 * k_sqrt)
                su = torch.linspace(
                    -sub_off * (k_sqrt - 1), sub_off * (k_sqrt - 1),
                    k_sqrt, device=device, dtype=dtype,
                )
                sv = torch.linspace(
                    -sub_off * (k_sqrt - 1), sub_off * (k_sqrt - 1),
                    k_sqrt, device=device, dtype=dtype,
                )
                sv_g, su_g = torch.meshgrid(sv, su, indexing="ij")
                su_flat = su_g.reshape(-1)  # [K]
                sv_flat = sv_g.reshape(-1)  # [K]
                uu_sub = uu_c.unsqueeze(-1) + su_flat  # [ph, pw, K]
                vv_sub = vv_c.unsqueeze(-1) + sv_flat

            self._subpatch_grid_cache[key] = (uu_sub, vv_sub)
        uu, vv = self._subpatch_grid_cache[key]
        return uu.to(device=device, dtype=dtype), vv.to(device=device, dtype=dtype)

    def _sample_image_colors(
        self,
        input_image: torch.Tensor,
        image_size_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Sample RGB colors from the input image at sub-patch grid points.

        Args:
            input_image: [B, S, 3, H, W] denormalized image in [0, 1]
            image_size_hw: (H, W)

        Returns:
            [B, S, M, K, 3] sampled colors
        """
        H, W = image_size_hw
        B, S = input_image.shape[:2]
        device = input_image.device
        dtype = input_image.dtype

        uu, vv = self._get_subpatch_grid(H, W, device, dtype)  # [ph, pw, K]
        ph, pw, K = uu.shape

        # Normalize to [-1, 1] for grid_sample
        u_norm = 2.0 * uu / (W - 1) - 1.0
        v_norm = 2.0 * vv / (H - 1) - 1.0

        # Flatten to [ph*pw*K, 2]
        grid = torch.stack([u_norm.reshape(-1), v_norm.reshape(-1)], dim=-1)
        n_pts = grid.shape[0]
        grid = grid.reshape(1, 1, n_pts, 2).expand(B * S, -1, -1, -1)

        # Sample: input [B*S, 3, H, W], grid [B*S, 1, n_pts, 2]
        sampled = F.grid_sample(
            input_image.reshape(B * S, 3, H, W),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B*S, 3, 1, n_pts]

        # Reshape to [B, S, M, K, 3]
        sampled = sampled.reshape(B, S, 3, n_pts)
        sampled = sampled.permute(0, 1, 3, 2)  # [B, S, n_pts, 3]
        return sampled.reshape(B, S, ph * pw, K, 3)

    def _compute_adaptive_scale(
        self,
        raw_scale: torch.Tensor,       # [B, S, M, K, 2]
        depth_at_subpatch: torch.Tensor, # [B, S, M, K]
        pose_enc: torch.Tensor,         # [B, S, 9]
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Compute depth-adaptive Gaussian scales.

        Base scale is chosen so each Gaussian projects to ~target_sigma pixels
        on screen. The MLP learns a correction factor around this base.

        target_sigma = patch_size / (2 * sqrt(K))
        base_scale_3d = target_sigma * depth / focal_length
        scale = base_scale_3d * exp(raw_correction.clamp(-2, 2))
        """
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

        k_sqrt = max(1, int(self.K ** 0.5 + 0.5))
        target_sigma = self.patch_size / (2.0 * k_sqrt)  # ~3.5 for K=4

        # Get focal length from pose encoding
        _, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(H, W), build_intrinsics=True
        )
        fx = intrinsics[:, :, 0, 0]  # [B, S]
        # Average focal length
        f_avg = fx.unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]

        # Base scale: target_sigma * depth / focal_length
        depth_clamped = depth_at_subpatch.clamp(min=0.1)  # [B, S, M, K]
        base_scale = target_sigma * depth_clamped / f_avg  # [B, S, M, K]

        # Learned correction factor (allow 0.25x to 4x adjustment)
        correction = torch.exp(raw_scale.clamp(-1.5, 1.5))  # [B, S, M, K, 2]
        scale = base_scale.unsqueeze(-1) * correction  # [B, S, M, K, 2]

        return scale

    @torch.no_grad()
    def _unproject_subpatch(
        self,
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
        H: int,
        W: int,
        M: int,
        return_depth: bool = False,
    ):
        """Sample depth at sub-patch points and unproject to world coordinates.

        Args:
            depth: [B, S, H, W, 1]
            pose_enc: [B, S, 9]
            H, W: image dimensions
            M: expected number of patch tokens
            return_depth: if True, also return sampled depth values

        Returns:
            positions: [B, S, M, K, 3] world-space positions
            depth_sampled (optional): [B, S, M, K] depth at sub-patch points
        """
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        from lingbot_map.utils.geometry import closed_form_inverse_se3_general

        B, S = depth.shape[:2]
        device = depth.device
        dtype = depth.dtype

        uu, vv = self._get_subpatch_grid(H, W, device, dtype)  # [ph, pw, K]
        ph, pw, K = uu.shape

        # ---- sample depth at sub-patch points via grid_sample ----
        depth_2d = depth[..., 0]  # [B, S, H, W]
        u_norm = 2.0 * uu / (W - 1) - 1.0
        v_norm = 2.0 * vv / (H - 1) - 1.0

        # Flatten to [ph*pw*K] for efficient grid_sample
        u_flat = u_norm.reshape(-1)  # [ph*pw*K]
        v_flat = v_norm.reshape(-1)
        n_pts = u_flat.shape[0]

        grid = torch.stack([u_flat, v_flat], dim=-1)  # [n_pts, 2]
        grid = grid.reshape(1, 1, n_pts, 2).expand(B * S, -1, -1, -1)

        depth_sampled = F.grid_sample(
            depth_2d.reshape(B * S, 1, H, W),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B*S, 1, 1, n_pts]
        depth_sampled = depth_sampled.reshape(B, S, n_pts)  # [B, S, ph*pw*K]

        # ---- camera parameters from pose encoding ----
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(H, W), build_intrinsics=True
        )

        ext_4x4 = torch.zeros(B, S, 4, 4, device=device, dtype=dtype)
        ext_4x4[:, :, :3, :] = extrinsics
        ext_4x4[:, :, 3, 3] = 1.0
        c2w = closed_form_inverse_se3_general(ext_4x4)  # [B, S, 4, 4]

        fx = intrinsics[:, :, 0, 0]
        fy = intrinsics[:, :, 1, 1]
        cx = intrinsics[:, :, 0, 2]
        cy = intrinsics[:, :, 1, 2]

        # Pixel coordinates (un-normalized)
        pixel_u = uu.reshape(-1)  # [ph*pw*K]
        pixel_v = vv.reshape(-1)

        # Camera coords
        x_cam = (pixel_u[None, None] - cx[..., None]) / fx[..., None] * depth_sampled
        y_cam = (pixel_v[None, None] - cy[..., None]) / fy[..., None] * depth_sampled
        z_cam = depth_sampled

        cam_pts = torch.stack(
            [x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1
        )  # [B, S, ph*pw*K, 4]

        world_pts = torch.einsum("bsij,bsmj->bsmi", c2w, cam_pts)
        # Reshape to [B, S, M, K, 3]
        positions = world_pts[..., :3].reshape(B, S, ph * pw, K, 3)
        if return_depth:
            depth_per_subpatch = depth_sampled.reshape(B, S, ph * pw, K)
            return positions, depth_per_subpatch
        return positions
