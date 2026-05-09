"""
GaussianRenderer - Differentiable rendering of compact 2D Gaussian surfel maps.

Supports two backends:
- gsplat (fast, requires CUDA JIT) — used when available
- Pure PyTorch (portable, no CUDA JIT) — fallback for Windows / training

Also provides L1 + SSIM rendering losses for training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from lingbot_map.mapping.gaussian_map import GaussianData


# ---------------------------------------------------------------------------
# 2D surfel → full 3D covariance helpers
# ---------------------------------------------------------------------------

def surfel_to_3d_params(
    scales_2d: torch.Tensor,      # [N, 2]
    normals: torch.Tensor,         # [N, 3]
    min_z_scale: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert 2D surfel (tangent scales + normal) to 3D scales + quaternions.

    The surfel is a flat ellipse on the tangent plane defined by the normal.
    We construct a local frame (t1, t2, n) and express the Gaussian as an
    anisotropic 3D Gaussian with the third scale set to a very small value
    (effectively flat).

    Returns:
        scales_3d: [N, 3]
        quats:     [N, 4]  (wxyz convention)
    """
    N = normals.shape[0]
    device, dtype = normals.device, normals.dtype

    # Build local frame from normal using Gram-Schmidt
    n = F.normalize(normals, dim=-1, eps=1e-8)

    # Pick a reference vector that is not parallel to the normal
    ref = torch.zeros(N, 3, device=device, dtype=dtype)
    ref[:, 0] = 1.0  # default X axis
    # Where normal is nearly parallel to X, use Y instead
    parallel_mask = n[:, 0].abs() > 0.9
    ref[parallel_mask, 0] = 0.0
    ref[parallel_mask, 1] = 1.0

    t1 = F.normalize(ref - (ref * n).sum(-1, keepdim=True) * n, dim=-1, eps=1e-8)
    t2 = torch.cross(n, t1, dim=-1)

    # Rotation matrix: columns are (t1, t2, n) — local-to-world
    R = torch.stack([t1, t2, n], dim=-1)  # [N, 3, 3]

    # Convert rotation matrix to quaternion (wxyz)
    quats = _rotation_matrix_to_quaternion(R)

    # 3D scales: tangent-plane scales + tiny z-scale
    z_scale = torch.full((N, 1), min_z_scale, device=device, dtype=dtype)
    scales_3d = torch.cat([scales_2d, z_scale], dim=-1)  # [N, 3]

    return scales_3d, quats


def _build_surfel_rotation_matrix(normals: torch.Tensor) -> torch.Tensor:
    """Build local frame rotation matrix from normals. Returns [N, 3, 3]."""
    N = normals.shape[0]
    device, dtype = normals.device, normals.dtype
    n = F.normalize(normals, dim=-1, eps=1e-8)
    ref = torch.zeros(N, 3, device=device, dtype=dtype)
    ref[:, 0] = 1.0
    parallel_mask = n[:, 0].abs() > 0.9
    ref[parallel_mask, 0] = 0.0
    ref[parallel_mask, 1] = 1.0
    t1 = F.normalize(ref - (ref * n).sum(-1, keepdim=True) * n, dim=-1, eps=1e-8)
    t2 = torch.cross(n, t1, dim=-1)
    return torch.stack([t1, t2, n], dim=-1)  # [N, 3, 3]


def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [N, 3, 3] → quaternion [N, 4] in (w, x, y, z) format."""
    # Shepperd's method
    batch = R.shape[0]
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    trace = m00 + m11 + m22
    q = torch.zeros(batch, 4, device=R.device, dtype=R.dtype)

    s = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2  # 4w
    q[:, 0] = 0.25 * s
    q[:, 1] = (m21 - m12) / s.clamp(min=1e-10)
    q[:, 2] = (m02 - m20) / s.clamp(min=1e-10)
    q[:, 3] = (m10 - m01) / s.clamp(min=1e-10)

    # For numerical stability, handle cases where trace is small
    # (This simplified version works for well-conditioned rotations)
    q = F.normalize(q, dim=-1)
    # Ensure w > 0 for canonical form
    q = torch.where(q[:, :1] < 0, -q, q)
    return q


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class GaussianRenderer(nn.Module):
    """Render a compact Gaussian surfel map from an arbitrary viewpoint.

    Uses gsplat's rasterization backend.  The renderer is stateless — call
    ``render()`` with positions, attributes and camera parameters to get an
    image.
    """

    def __init__(
        self,
        sh_degree: int = 0,
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.register_buffer(
            "bg_color",
            torch.tensor(bg_color, dtype=torch.float32),
            persistent=False,
        )
        self.near_plane = near_plane
        self.far_plane = far_plane

    def render(
        self,
        positions: torch.Tensor,     # [N, 3]
        opacities: torch.Tensor,     # [N, 1]
        scales_2d: torch.Tensor,     # [N, 2]
        normals: torch.Tensor,       # [N, 3]
        colors: torch.Tensor,        # [N, 3]
        viewmat: torch.Tensor,       # [4, 4] world-to-camera
        K: torch.Tensor,             # [3, 3] intrinsics
        width: int,
        height: int,
    ) -> Dict[str, torch.Tensor]:
        """Render the Gaussian map from a given viewpoint.

        Returns:
            Dict with:
                image:  [H, W, 3] rendered RGB in [0, 1]
                depth:  [H, W, 1] rendered depth
                alpha:  [H, W, 1] accumulated opacity
        """
        from gsplat import rasterization

        N = positions.shape[0]
        if N == 0:
            dev = viewmat.device
            return {
                "image": torch.zeros(height, width, 3, device=dev),
                "depth": torch.zeros(height, width, 1, device=dev),
                "alpha": torch.zeros(height, width, 1, device=dev),
            }

        # Convert 2D surfel params to 3D Gaussian params for gsplat
        scales_3d, quats = surfel_to_3d_params(scales_2d, normals)

        # gsplat expects colors as [N, K, 3] for SH degree K; for degree 0
        # just [N, 1, 3] (DC component only)
        if colors.dim() == 2:
            colors_sh = colors.unsqueeze(1)  # [N, 1, 3]
        else:
            colors_sh = colors

        # gsplat rasterization
        renders, alphas, meta = rasterization(
            means=positions,                    # [N, 3]
            quats=quats,                        # [N, 4]
            scales=scales_3d,                   # [N, 3]
            opacities=opacities.squeeze(-1),    # [N]
            colors=colors_sh,                   # [N, K, 3]
            viewmats=viewmat.unsqueeze(0),      # [1, 4, 4]
            Ks=K.unsqueeze(0),                  # [1, 3, 3]
            width=width,
            height=height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            sh_degree=self.sh_degree,
            backgrounds=self.bg_color.unsqueeze(0),  # [1, 3]
            render_mode="RGB+ED",  # RGB + expected depth
        )
        # renders: [1, H, W, 4] (RGB + depth)
        # alphas:  [1, H, W, 1]

        rgb = renders[0, :, :, :3]    # [H, W, 3]
        depth = renders[0, :, :, 3:]  # [H, W, 1]
        alpha = alphas[0]             # [H, W, 1]

        return {"image": rgb, "depth": depth, "alpha": alpha}

    def render_from_gaussian_data(
        self,
        data: GaussianData,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
    ) -> Dict[str, torch.Tensor]:
        """Convenience: render directly from a GaussianData container."""
        return self.render(
            positions=data.positions,
            opacities=data.opacities,
            scales_2d=data.scales,
            normals=data.normals,
            colors=data.colors,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
        )


# ---------------------------------------------------------------------------
# Pure-PyTorch differentiable renderer (no CUDA JIT needed)
# ---------------------------------------------------------------------------

class PureTorchRenderer(nn.Module):
    """Differentiable Gaussian surfel renderer using only PyTorch operations.

    Designed for training the GS head on Windows or systems without gsplat.
    Handles ~1000 Gaussians per frame efficiently via chunked pixel evaluation.
    """

    def __init__(
        self,
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
        pixel_chunk_size: int = 16384,
        min_z_scale: float = 1e-4,
    ):
        super().__init__()
        self.register_buffer(
            "bg_color",
            torch.tensor(bg_color, dtype=torch.float32),
            persistent=False,
        )
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.pixel_chunk_size = pixel_chunk_size
        self.min_z_scale = min_z_scale

    def render(
        self,
        positions: torch.Tensor,     # [N, 3]
        opacities: torch.Tensor,     # [N, 1]
        scales_2d: torch.Tensor,     # [N, 2]
        normals: torch.Tensor,       # [N, 3]
        colors: torch.Tensor,        # [N, 3]
        viewmat: torch.Tensor,       # [4, 4] world-to-camera
        K: torch.Tensor,             # [3, 3] intrinsics
        width: int,
        height: int,
    ) -> Dict[str, torch.Tensor]:
        """Render surfel Gaussians from a given viewpoint.

        Returns:
            Dict with:
                image: [H, W, 3] rendered RGB in [0, 1]
                depth: [H, W, 1] rendered depth
                alpha: [H, W, 1] accumulated opacity
        """
        dev = positions.device
        N = positions.shape[0]
        if N == 0:
            return {
                "image": self.bg_color.expand(height, width, 3).clone(),
                "depth": torch.zeros(height, width, 1, device=dev),
                "alpha": torch.zeros(height, width, 1, device=dev),
            }

        # --- 1. Transform to camera space ---
        R_w2c = viewmat[:3, :3]    # [3, 3]
        t_w2c = viewmat[:3, 3]     # [3]
        cam_pos = positions @ R_w2c.T + t_w2c  # [N, 3]

        # Depth filter
        z = cam_pos[:, 2]
        visible = (z > self.near_plane) & (z < self.far_plane)
        if not visible.any():
            return {
                "image": self.bg_color.expand(height, width, 3).clone(),
                "depth": torch.zeros(height, width, 1, device=dev),
                "alpha": torch.zeros(height, width, 1, device=dev),
            }

        cam_pos = cam_pos[visible]
        opa = opacities[visible].squeeze(-1)  # [N']
        s2d = scales_2d[visible]               # [N', 2]
        nrm = normals[visible]                 # [N', 3]
        col = colors[visible]                  # [N', 3]
        z = cam_pos[:, 2]                      # [N']
        N_vis = cam_pos.shape[0]

        # --- 2. Project to 2D screen coordinates ---
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_2d = cam_pos[:, 0] / z * fx + cx  # [N']
        y_2d = cam_pos[:, 1] / z * fy + cy  # [N']

        # --- 3. Compute 2D covariance from surfel params ---
        # Build 3D covariance: Σ_3d = R_local @ diag(s1², s2², ε²) @ R_local.T
        R_local = _build_surfel_rotation_matrix(nrm)  # [N', 3, 3]
        # Rotate to camera space: R_cam = R_w2c @ R_local
        R_cam = R_w2c.unsqueeze(0) @ R_local  # [N', 3, 3]

        # Scale matrix (squared scales as variances)
        S_sq = torch.zeros(N_vis, 3, device=dev, dtype=positions.dtype)
        S_sq[:, 0] = s2d[:, 0] ** 2
        S_sq[:, 1] = s2d[:, 1] ** 2
        S_sq[:, 2] = self.min_z_scale ** 2
        # Σ_cam = R_cam @ diag(S_sq) @ R_cam.T
        RS = R_cam * S_sq.unsqueeze(1)  # [N', 3, 3] broadcasting scales
        cov3d_cam = RS @ R_cam.transpose(-1, -2)  # [N', 3, 3]

        # Project to 2D using Jacobian: J = [[fx/z, 0, -fx*x/z²], [0, fy/z, -fy*y/z²]]
        inv_z = 1.0 / z
        inv_z2 = inv_z * inv_z
        J = torch.zeros(N_vis, 2, 3, device=dev, dtype=positions.dtype)
        J[:, 0, 0] = fx * inv_z
        J[:, 0, 2] = -fx * cam_pos[:, 0] * inv_z2
        J[:, 1, 1] = fy * inv_z
        J[:, 1, 2] = -fy * cam_pos[:, 1] * inv_z2

        # Σ_2d = J @ Σ_cam @ J.T  → [N', 2, 2]
        cov2d = J @ cov3d_cam @ J.transpose(-1, -2)

        # Add small regularization for numerical stability
        cov2d[:, 0, 0] = cov2d[:, 0, 0] + 0.3
        cov2d[:, 1, 1] = cov2d[:, 1, 1] + 0.3

        # --- 4. Compute inverse covariance and determinant ---
        a = cov2d[:, 0, 0]  # [N']
        b = cov2d[:, 0, 1]  # [N']
        d = cov2d[:, 1, 1]  # [N']
        det = a * d - b * b  # [N']
        det = det.clamp(min=1e-8)
        inv_det = 1.0 / det
        # Inverse of 2x2: [[d, -b], [-b, a]] / det
        inv_a = d * inv_det   # [N']
        inv_b = -b * inv_det  # [N']
        inv_d = a * inv_det   # [N']

        # Compute effective radius (3σ extent in pixels)
        radius = 3.0 * torch.sqrt(torch.max(a, d))  # [N']

        # --- 5. Sort by depth (front to back) ---
        depth_order = z.argsort()
        x_2d = x_2d[depth_order]
        y_2d = y_2d[depth_order]
        z_sorted = z[depth_order]
        opa = opa[depth_order]
        col = col[depth_order]
        inv_a = inv_a[depth_order]
        inv_b = inv_b[depth_order]
        inv_d = inv_d[depth_order]
        radius = radius[depth_order]

        # --- 6. Render via chunked pixel evaluation ---
        # Create pixel grid
        py, px = torch.meshgrid(
            torch.arange(height, device=dev, dtype=positions.dtype),
            torch.arange(width, device=dev, dtype=positions.dtype),
            indexing="ij",
        )
        pixels = torch.stack([px.reshape(-1), py.reshape(-1)], dim=-1)  # [H*W, 2]
        total_pixels = pixels.shape[0]

        chunk = self.pixel_chunk_size
        chunks_out = []
        for start in range(0, total_pixels, chunk):
            end = min(start + chunk, total_pixels)
            pix = pixels[start:end]  # [C, 2]

            # Use gradient checkpointing to avoid storing intermediates
            # for all chunks simultaneously
            out = torch.utils.checkpoint.checkpoint(
                self._render_chunk,
                pix, x_2d, y_2d, inv_a, inv_b, inv_d, opa, col, z_sorted, radius,
                use_reentrant=False,
            )
            chunks_out.append(out)

        # Concatenate all chunks
        image_flat = torch.cat([c[0] for c in chunks_out], dim=0)
        depth_flat = torch.cat([c[1] for c in chunks_out], dim=0)
        alpha_flat = torch.cat([c[2] for c in chunks_out], dim=0)

        # Reshape to image
        image = image_flat.reshape(height, width, 3)
        depth = depth_flat.reshape(height, width, 1)
        alpha = alpha_flat.reshape(height, width, 1)

        # Apply background
        bg = self.bg_color.to(dev, dtype=positions.dtype)
        image = image + (1.0 - alpha) * bg

        return {"image": image, "depth": depth, "alpha": alpha}

    @staticmethod
    def _render_chunk(pix, x_2d, y_2d, inv_a, inv_b, inv_d, opa, col, z_sorted, radius):
        """Render a chunk of pixels. Separated for gradient checkpointing."""
        C = pix.shape[0]
        dev = pix.device
        dtype = pix.dtype

        dx = pix[:, 0].unsqueeze(1) - x_2d.unsqueeze(0)
        dy = pix[:, 1].unsqueeze(1) - y_2d.unsqueeze(0)

        maha = (
            dx * dx * inv_a.unsqueeze(0)
            + 2.0 * dx * dy * inv_b.unsqueeze(0)
            + dy * dy * inv_d.unsqueeze(0)
        )

        gauss_weight = torch.exp(-0.5 * maha.clamp(min=0.0, max=20.0))

        dist_sq = dx * dx + dy * dy
        radius_mask = dist_sq < (radius.unsqueeze(0) ** 2)
        gauss_weight = gauss_weight * radius_mask.float()

        alpha_per_gs = (opa.unsqueeze(0) * gauss_weight).clamp(max=0.99)

        one_minus_alpha = 1.0 - alpha_per_gs
        T = torch.cat([
            torch.ones(C, 1, device=dev, dtype=dtype),
            one_minus_alpha[:, :-1].cumprod(dim=1),
        ], dim=1)

        w = alpha_per_gs * T

        img = (w.unsqueeze(-1) * col.unsqueeze(0)).sum(dim=1)
        dep = (w * z_sorted.unsqueeze(0)).sum(dim=1, keepdim=True)
        alp = 1.0 - T[:, -1:] * one_minus_alpha[:, -1:]

        return img, dep, alp

    def render_from_gaussian_data(
        self,
        data: GaussianData,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
    ) -> Dict[str, torch.Tensor]:
        """Convenience: render directly from a GaussianData container."""
        return self.render(
            positions=data.positions,
            opacities=data.opacities,
            scales_2d=data.scales,
            normals=data.normals,
            colors=data.colors,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
        )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute 1 - SSIM between pred and target images [H, W, 3]."""
    # Rearrange to [1, 3, H, W] for conv2d
    pred = pred.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)

    # Gaussian window
    ch = pred.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5, ch, pred.device, pred.dtype)

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=ch)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=ch)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = (F.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=ch) - mu1_sq).clamp(min=0)
    sigma2_sq = (F.conv2d(target ** 2, kernel, padding=window_size // 2, groups=ch) - mu2_sq).clamp(min=0)
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=ch) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
    )
    return 1.0 - ssim_map.mean()


def rendering_loss(
    rendered: torch.Tensor,   # [H, W, 3]
    target: torch.Tensor,     # [H, W, 3]
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Combined L1 + SSIM rendering loss."""
    loss_l1 = l1_loss(rendered, target)
    loss_ssim = ssim_loss(rendered, target)
    total = lambda_l1 * loss_l1 + lambda_ssim * loss_ssim
    return {"loss": total, "l1": loss_l1, "ssim": loss_ssim}


def _gaussian_kernel(size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)  # [size, size]
    kernel = kernel_2d.expand(channels, 1, size, size).contiguous()
    return kernel
