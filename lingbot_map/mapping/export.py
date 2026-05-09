"""
Export utilities for compact Gaussian surfel maps.

Supports:
- PLY export (viewable in CloudCompare, MeshLab, etc.)
- Gaussian Splatting PLY export (compatible with GS viewers)
"""

from __future__ import annotations

import struct
import numpy as np
from pathlib import Path
from typing import Optional

from lingbot_map.mapping.gaussian_map import GaussianData


def export_ply_pointcloud(
    data: GaussianData,
    path: str,
    opacity_threshold: float = 0.05,
) -> int:
    """Export Gaussians as a simple coloured point cloud PLY.

    Args:
        data: GaussianData with positions and colors.
        path: Output .ply file path.
        opacity_threshold: Only export Gaussians above this opacity.

    Returns:
        Number of points exported.
    """
    pos = data.positions.cpu().numpy()
    col = (data.colors.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    opa = data.opacities.squeeze(-1).cpu().numpy()

    mask = opa > opacity_threshold
    pos, col = pos[mask], col[mask]
    n = pos.shape[0]

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", *pos[i]))
            f.write(struct.pack("<BBB", *col[i]))

    return n


def export_splat_ply(
    data: GaussianData,
    path: str,
    opacity_threshold: float = 0.05,
) -> int:
    """Export as a Gaussian Splatting-compatible PLY (3DGS viewer format).

    Each Gaussian has: xyz, normals, SH DC color, opacity, scale(3), rot(4).

    Args:
        data: GaussianData with surfel attributes.
        path: Output .ply file path.
        opacity_threshold: Min opacity to export.

    Returns:
        Number of Gaussians exported.
    """
    from lingbot_map.mapping.renderer import surfel_to_3d_params

    pos = data.positions
    opa = data.opacities.squeeze(-1)
    mask = opa > opacity_threshold

    pos_np = pos[mask].cpu().numpy()
    opa_np = opa[mask].cpu().numpy()
    col_np = data.colors[mask].cpu().numpy()
    nor_np = data.normals[mask].cpu().numpy()

    scales_3d, quats = surfel_to_3d_params(
        data.scales[mask].cpu(), data.normals[mask].cpu()
    )
    scales_np = np.log(scales_3d.numpy().clip(min=1e-7))  # store as log-scale
    quats_np = quats.numpy()

    n = pos_np.shape[0]

    # SH DC coefficient: color * C0 inverse
    # For SH degree 0: f_dc = (color - 0.5) / C0 where C0 = 0.28209479
    C0 = 0.28209479177387814
    sh_dc = (col_np - 0.5) / C0

    # Inverse sigmoid for opacity
    opa_raw = np.log(opa_np.clip(1e-6, 1 - 1e-6) / (1 - opa_np.clip(1e-6, 1 - 1e-6)))

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))

        for i in range(n):
            # xyz
            f.write(struct.pack("<fff", *pos_np[i]))
            # normals
            f.write(struct.pack("<fff", *nor_np[i]))
            # SH DC
            f.write(struct.pack("<fff", *sh_dc[i]))
            # opacity (raw logit)
            f.write(struct.pack("<f", opa_raw[i]))
            # scales (log)
            f.write(struct.pack("<fff", *scales_np[i]))
            # rotation (wxyz)
            f.write(struct.pack("<ffff", *quats_np[i]))

    return n
