"""Training script for Streaming Anchor-GS.

Trains anchor_token_proj + anchor_decoder with frozen GCT backbone.
Uses differentiable Gaussian rendering (gsplat) with L1 + SSIM loss.

Usage:
    python train_anchor_gs.py --model_path /path/to/checkpoint.pt \
        --data_dir data/re10k --num_epochs 10 --lr 1e-4

    # Quick feasibility test (1 epoch, small batch)
    python train_anchor_gs.py --model_path /path/to/checkpoint.pt \
        --data_dir data/re10k --num_epochs 1 --log_interval 10
"""

import argparse
import io
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as TF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RE10KDataset(Dataset):
    """RealEstate10K dataset in pixelsplat .torch format.

    Each .torch file contains ~16 scenes. Each scene has N frames with
    JPEG-encoded images and camera parameters [fx_norm, fy_norm, cx, cy, near, far, R(9), t(3)].
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_context: int = 5,
        num_target: int = 2,
        image_size: int = 256,
        patch_size: int = 14,
        max_scenes: int = -1,
    ):
        self.split = split
        self.num_context = num_context
        self.num_target = num_target
        self.image_size = image_size
        self.patch_size = patch_size

        split_dir = Path(data_dir) / split
        torch_files = sorted(split_dir.glob("*.torch"))
        if not torch_files:
            raise FileNotFoundError(f"No .torch files found in {split_dir}")

        logger.info(f"Loading {split} data from {len(torch_files)} files...")
        self.scenes = []
        for f in torch_files:
            scenes = torch.load(f, map_location="cpu", weights_only=False)
            for s in scenes:
                n_frames = len(s["images"])
                if n_frames >= num_context + num_target:
                    self.scenes.append(s)
            if max_scenes > 0 and len(self.scenes) >= max_scenes:
                self.scenes = self.scenes[:max_scenes]
                break

        logger.info(f"Loaded {len(self.scenes)} {split} scenes")

        # Image preprocessing
        self.normalize = TF.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.scenes)

    def _decode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Decode JPEG bytes tensor to RGB float tensor."""
        img = Image.open(io.BytesIO(img_tensor.numpy().tobytes()))
        return TF.functional.to_tensor(img)  # [3, H, W] float [0, 1]

    def _resize_image(self, img: torch.Tensor) -> torch.Tensor:
        """Resize to model input size, maintaining patch alignment."""
        _, h, w = img.shape
        # Resize longest side to image_size, then ensure divisible by patch_size
        scale = self.image_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        # Align to patch_size
        new_h = (new_h // self.patch_size) * self.patch_size
        new_w = (new_w // self.patch_size) * self.patch_size
        new_h = max(new_h, self.patch_size)
        new_w = max(new_w, self.patch_size)
        return F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)

    def _parse_camera(self, cam: torch.Tensor, orig_h: int, orig_w: int, new_h: int, new_w: int):
        """Parse pixelsplat camera format to w2c [3, 4] and intrinsics [3, 3].

        Camera format: [fx_norm, fy_norm, cx_norm, cy_norm, near, far, R(9), t(3)]
        Normalized intrinsics (divided by image dimensions).
        """
        fx_norm, fy_norm = cam[0].item(), cam[1].item()
        cx_norm, cy_norm = cam[2].item(), cam[3].item()
        R = cam[6:15].reshape(3, 3)  # world-to-camera rotation
        t = cam[15:18]               # world-to-camera translation

        # Build intrinsics for resized image
        fx = fx_norm * new_w
        fy = fy_norm * new_h
        cx = cx_norm * new_w
        cy = cy_norm * new_h

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=torch.float32)

        # w2c as 4x4
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        return w2c, K

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        n_frames = len(scene["images"])
        total_needed = self.num_context + self.num_target

        # Sample frame indices: spread context frames, pick nearby targets
        all_idx = np.linspace(0, n_frames - 1, total_needed, dtype=int)
        # Add randomness
        rng = np.random.default_rng()
        jitter = rng.integers(-2, 3, size=total_needed)
        all_idx = np.clip(all_idx + jitter, 0, n_frames - 1)
        all_idx = np.unique(all_idx)
        # Ensure we have enough frames
        while len(all_idx) < total_needed:
            extra = rng.integers(0, n_frames, size=total_needed - len(all_idx))
            all_idx = np.unique(np.concatenate([all_idx, extra]))
        all_idx = np.sort(all_idx[:total_needed])

        context_idx = all_idx[:self.num_context]
        target_idx = all_idx[self.num_context:]

        # Decode and resize images
        context_imgs = []
        context_imgs_raw = []  # for GT rendering comparison
        target_imgs_raw = []
        context_w2c = []
        context_K = []
        target_w2c = []
        target_K = []

        # Get original image size from first frame
        first_img = self._decode_image(scene["images"][context_idx[0]])
        _, orig_h, orig_w = first_img.shape
        resized = self._resize_image(first_img)
        _, new_h, new_w = resized.shape

        for i in context_idx:
            img = self._decode_image(scene["images"][i])
            img_resized = self._resize_image(img)
            context_imgs.append(self.normalize(img_resized))
            context_imgs_raw.append(img_resized)  # unnormalized for loss
            w2c, K = self._parse_camera(scene["cameras"][i], orig_h, orig_w, new_h, new_w)
            context_w2c.append(w2c)
            context_K.append(K)

        for i in target_idx:
            img = self._decode_image(scene["images"][i])
            img_resized = self._resize_image(img)
            target_imgs_raw.append(img_resized)
            w2c, K = self._parse_camera(scene["cameras"][i], orig_h, orig_w, new_h, new_w)
            target_w2c.append(w2c)
            target_K.append(K)

        return {
            "context_images": torch.stack(context_imgs),       # [Nc, 3, H, W] normalized
            "context_images_raw": torch.stack(context_imgs_raw), # [Nc, 3, H, W] [0,1]
            "target_images_raw": torch.stack(target_imgs_raw),   # [Nt, 3, H, W] [0,1]
            "context_w2c": torch.stack(context_w2c),             # [Nc, 4, 4]
            "context_K": torch.stack(context_K),                 # [Nc, 3, 3]
            "target_w2c": torch.stack(target_w2c),               # [Nt, 4, 4]
            "target_K": torch.stack(target_K),                   # [Nt, 3, 3]
            "image_size": torch.tensor([new_h, new_w]),
        }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute 1 - SSIM between two images [B, C, H, W]."""
    C = pred.shape[1]
    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)

    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu12

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


def rendering_loss(pred: torch.Tensor, target: torch.Tensor, lambda_ssim: float = 0.2) -> torch.Tensor:
    """Combined L1 + SSIM loss."""
    l1 = F.l1_loss(pred, target)
    ss = ssim_loss(pred, target) if pred.shape[-1] >= 11 and pred.shape[-2] >= 11 else torch.tensor(0.0, device=pred.device)
    return (1 - lambda_ssim) * l1 + lambda_ssim * ss


# ---------------------------------------------------------------------------
# Surfel-to-3DGS conversion for gsplat
# ---------------------------------------------------------------------------

def normal_to_quaternion(normals: torch.Tensor) -> torch.Tensor:
    """Convert unit normals to quaternions (rotation aligning z-axis to normal).

    Args:
        normals: [N, 3] unit normals
    Returns:
        quats: [N, 4] quaternions (w, x, y, z)
    """
    n = F.normalize(normals, dim=-1, eps=1e-8)
    dot = n[:, 2].clamp(-1.0, 1.0)  # dot with z-axis

    cross = torch.stack([-n[:, 1], n[:, 0], torch.zeros_like(n[:, 0])], dim=-1)
    cross_norm = cross.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    half_angle = torch.acos(dot) / 2.0
    w = torch.cos(half_angle)
    axis = cross / cross_norm
    s = torch.sin(half_angle)

    quats = torch.stack([w, axis[:, 0] * s, axis[:, 1] * s, axis[:, 2] * s], dim=-1)

    # Handle degenerate cases
    degenerate = cross_norm.squeeze(-1) < 1e-6
    identity = torch.tensor([1, 0, 0, 0], dtype=quats.dtype, device=quats.device)
    flip = torch.tensor([0, 1, 0, 0], dtype=quats.dtype, device=quats.device)
    quats[degenerate & (dot >= 0)] = identity
    quats[degenerate & (dot < 0)] = flip

    return quats


def surfels_to_3dgs(gs_dict):
    """Convert 2D surfel format to 3DGS format for gsplat rendering.

    Input gs_dict keys: positions [A,K,3], opacities [A,K,1], scales [A,K,2], normals [A,K,3], colors [A,K,3]
    Returns: means [N,3], quats [N,4], scales [N,3], opacities [N], colors [N,3]
    """
    A, K = gs_dict["positions"].shape[:2]
    N = A * K

    means = gs_dict["positions"].reshape(N, 3)
    opacities = gs_dict["opacities"].reshape(N)
    scales_2d = gs_dict["scales"].reshape(N, 2)
    normals = gs_dict["normals"].reshape(N, 3)
    colors = gs_dict["colors"].reshape(N, 3)

    # Convert to 3DGS: add near-zero thickness along normal
    log_scales_2d = torch.log(scales_2d.clamp(min=1e-7))
    log_scale_z = torch.full((N, 1), math.log(1e-5), device=means.device, dtype=means.dtype)
    scales_3d = torch.exp(torch.cat([log_scales_2d, log_scale_z], dim=-1))

    quats = normal_to_quaternion(normals)

    return means, quats, scales_3d, opacities, colors


# ---------------------------------------------------------------------------
# Rendering with gsplat
# ---------------------------------------------------------------------------

def render_gaussians(
    gs_dict,
    viewmats: torch.Tensor,   # [C, 4, 4] world-to-camera
    Ks: torch.Tensor,         # [C, 3, 3] intrinsics
    height: int,
    width: int,
) -> torch.Tensor:
    """Render Gaussians to images. Tries gsplat, falls back to pure PyTorch splatting.

    Returns: rendered images [C, 3, H, W] in [0, 1].
    """
    means, quats, scales, opacities, colors = surfels_to_3dgs(gs_dict)

    if means.shape[0] == 0:
        return torch.zeros(viewmats.shape[0], 3, height, width, device=viewmats.device)

    try:
        from gsplat import rasterization
        rendered, alphas, info = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks,
            width=width, height=height,
            near_plane=0.01, far_plane=1000.0,
            render_mode="RGB", packed=True,
        )
        return rendered.permute(0, 3, 1, 2).clamp(0, 1)
    except (ImportError, OSError, RuntimeError):
        pass

    # Fallback: pure PyTorch point splatting (differentiable, no CUDA extensions)
    return _render_point_splatting(means, opacities, colors, viewmats, Ks, height, width)


def _render_point_splatting(
    means: torch.Tensor,     # [N, 3]
    opacities: torch.Tensor, # [N]
    colors: torch.Tensor,    # [N, 3]
    viewmats: torch.Tensor,  # [C, 4, 4]
    Ks: torch.Tensor,        # [C, 3, 3]
    height: int,
    width: int,
    splat_size: int = 3,     # radius in pixels for splatting kernel
) -> torch.Tensor:
    """Pure PyTorch differentiable point splatting renderer.

    Projects 3D Gaussians to 2D, splatters with a small Gaussian kernel.
    Slower than gsplat but works without CUDA extensions.
    """
    C = viewmats.shape[0]
    N = means.shape[0]
    device = means.device
    rendered = torch.zeros(C, 3, height, width, device=device)

    for c in range(C):
        w2c = viewmats[c]  # [4, 4]
        K = Ks[c]          # [3, 3]

        # Transform to camera space
        pts_h = torch.cat([means, torch.ones(N, 1, device=device)], dim=-1)  # [N, 4]
        cam_pts = (w2c @ pts_h.T).T[:, :3]  # [N, 3]
        z = cam_pts[:, 2]

        # Filter: in front of camera
        valid = z > 0.01
        if not valid.any():
            continue

        cam_pts = cam_pts[valid]
        z = z[valid]
        valid_colors = colors[valid]      # [V, 3]
        valid_opacities = opacities[valid] # [V]

        # Project to pixel coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = fx * cam_pts[:, 0] / z + cx  # [V]
        v = fy * cam_pts[:, 1] / z + cy  # [V]

        # Filter: in image bounds (with margin)
        margin = splat_size + 1
        in_bounds = (u >= -margin) & (u < width + margin) & \
                    (v >= -margin) & (v < height + margin)
        if not in_bounds.any():
            continue

        u = u[in_bounds]
        v = v[in_bounds]
        z_valid = z[in_bounds]
        c_valid = valid_colors[in_bounds]
        o_valid = valid_opacities[in_bounds]
        V = u.shape[0]

        # Sort by depth (back to front for alpha compositing)
        sort_idx = z_valid.argsort(descending=True)
        u, v, c_valid, o_valid = u[sort_idx], v[sort_idx], c_valid[sort_idx], o_valid[sort_idx]

        # Limit number of splats for memory
        max_splats = 20000
        if V > max_splats:
            u, v, c_valid, o_valid = u[:max_splats], v[:max_splats], c_valid[:max_splats], o_valid[:max_splats]
            V = max_splats

        # Create pixel grid offsets for splatting kernel
        offsets = torch.arange(-splat_size, splat_size + 1, device=device, dtype=torch.float32)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dx = dx.reshape(-1)  # [K]
        dy = dy.reshape(-1)
        K_kernel = dx.shape[0]

        # Gaussian kernel weights
        sigma = splat_size / 2.0
        kernel_weights = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))  # [K]

        # Compute pixel positions for all splats
        # u, v: [V], dx, dy: [K] -> pixel_u, pixel_v: [V, K]
        pixel_u = u.unsqueeze(1) + dx.unsqueeze(0)  # [V, K]
        pixel_v = v.unsqueeze(1) + dy.unsqueeze(0)

        # Sub-pixel distance for Gaussian weighting
        dist_u = pixel_u - u.unsqueeze(1)  # Already equals dx broadcast
        dist_v = pixel_v - v.unsqueeze(1)

        # Round to integer pixel coordinates
        pu = pixel_u.long()
        pv = pixel_v.long()

        # Mask valid pixels
        valid_px = (pu >= 0) & (pu < width) & (pv >= 0) & (pv < height)

        # Compute alpha per splat point
        gauss_w = kernel_weights.unsqueeze(0).expand(V, -1)  # [V, K]
        alpha = (o_valid.unsqueeze(1) * gauss_w).clamp(0, 0.99)  # [V, K]

        # Flatten and scatter (back-to-front alpha blending via accumulation)
        pu_flat = pu[valid_px]
        pv_flat = pv[valid_px]
        alpha_flat = alpha[valid_px]
        color_flat = c_valid.unsqueeze(1).expand(-1, K_kernel, -1)[valid_px]  # [M, 3]

        # Use weighted accumulation (approximate alpha blending)
        pixel_idx = pv_flat * width + pu_flat  # [M]
        weighted_color = color_flat * alpha_flat.unsqueeze(-1)  # [M, 3]

        img_flat = torch.zeros(height * width, 3, device=device)
        weight_flat = torch.zeros(height * width, device=device)

        img_flat.scatter_add_(0, pixel_idx.unsqueeze(-1).expand(-1, 3), weighted_color)
        weight_flat.scatter_add_(0, pixel_idx, alpha_flat)

        # Normalize
        weight_flat = weight_flat.clamp(min=1e-6)
        img_flat = img_flat / weight_flat.unsqueeze(-1)

        rendered[c] = img_flat.reshape(height, width, 3).permute(2, 0, 1).clamp(0, 1)

    return rendered


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

class AnchorGSTrainer:
    """Manages the training of anchor_token_proj + anchor_decoder."""

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = next(model.parameters()).device

        # Freeze backbone, unfreeze anchor heads
        for name, param in model.named_parameters():
            if "anchor_token_proj" in name or "anchor_decoder" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable)
        logger.info(f"Trainable parameters: {total_params:,}")

        self.optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01
        )

    @torch.no_grad()
    def _run_backbone(self, images: torch.Tensor):
        """Run frozen backbone to get pose, depth, tokens.

        Args:
            images: [1, S, 3, H, W] normalized images
        Returns:
            pose_enc, depth, depth_conf, aggregated_tokens_list, patch_start_idx
        """
        B, S, C, H, W = images.shape
        n_scale = min(S, self.args.num_scale_frames)

        self.model.clean_kv_cache()

        # Phase 1: scale frames
        scale_imgs = images[:, :n_scale]
        agg, ps = self.model._aggregate_features(
            scale_imgs,
            num_frame_for_scale=n_scale,
            num_frame_per_block=n_scale,
        )
        preds = {}
        preds.update(self.model._predict_camera(
            agg, causal_inference=True,
            num_frame_for_scale=n_scale, num_frame_per_block=n_scale,
        ))
        preds.update(self.model._predict_depth(agg, scale_imgs, ps))

        all_pose = [preds["pose_enc"]]
        all_depth = [preds["depth"]]
        all_dconf = [preds["depth_conf"]]
        all_agg = [agg]

        # Phase 2: causal streaming for remaining frames
        for i in range(n_scale, S):
            frame = images[:, i:i+1]
            agg_f, ps_f = self.model._aggregate_features(
                frame, num_frame_for_scale=n_scale, num_frame_per_block=1,
            )
            fp = {}
            fp.update(self.model._predict_camera(
                agg_f, causal_inference=True,
                num_frame_for_scale=n_scale, num_frame_per_block=1,
            ))
            fp.update(self.model._predict_depth(agg_f, frame, ps_f))
            all_pose.append(fp["pose_enc"])
            all_depth.append(fp["depth"])
            all_dconf.append(fp["depth_conf"])
            all_agg.append(agg_f)

        pose_enc = torch.cat(all_pose, dim=1)     # [1, S, 9]
        depth = torch.cat(all_depth, dim=1)        # [1, S, H, W, 1]
        depth_conf = torch.cat(all_dconf, dim=1)   # [1, S, H, W]

        return pose_enc, depth, depth_conf, all_agg, ps

    def _build_anchors_and_decode(
        self,
        depth: torch.Tensor,        # [1, S, H, W, 1]
        depth_conf: torch.Tensor,   # [1, S, H, W]
        all_agg: list,               # list of agg token tensors
        ps: int,                     # patch_start_idx
        c2w: torch.Tensor,          # [1, S, 4, 4]
        intrinsics: torch.Tensor,   # [1, S, 3, 3]
        H: int, W: int,
        n_context: int,
    ):
        """Build anchors from context frames and decode to Gaussians.

        This is the differentiable part: anchor_token_proj + anchor_decoder are trainable.
        """
        # Extract features from context frame tokens (differentiable!)
        # all_agg[0] covers scale frames (n_scale frames), subsequent entries are 1 frame each
        n_scale = min(n_context, self.args.num_scale_frames)

        all_features = []
        # Scale frames: all_agg[0] has n_scale frames
        feat_scale = self.model._tokens_to_feature_map(
            all_agg[0], ps, H, W, num_frames=n_scale,
        )  # [n_scale, C, ph, pw]
        all_features.append(feat_scale)

        # Remaining context frames: all_agg[1], all_agg[2], ...
        for i in range(n_scale, n_context):
            agg_idx = i - n_scale + 1  # index into all_agg
            if agg_idx < len(all_agg):
                feat = self.model._tokens_to_feature_map(
                    all_agg[agg_idx], ps, H, W, num_frames=1,
                )
                all_features.append(feat)

        dense_features = torch.cat(all_features, dim=0)  # [Nc_actual, C, ph, pw]
        n_context_actual = dense_features.shape[0]

        # Create anchor manager and init from depth
        self.model.anchor_manager = self.model._create_anchor_manager(depth.device)

        # Build anchors from context frames
        self.model._init_anchors_from_scale_frames(
            depth[:, :n_context_actual], depth_conf[:, :n_context_actual],
            dense_features[:n_context_actual], c2w[:, :n_context_actual],
            intrinsics[:, :n_context_actual], n_context_actual,
        )

        if self.model.anchor_manager.anchors is None or self.model.anchor_manager.anchors.num_anchors == 0:
            return None

        # Now re-extract features for all anchors (differentiable path)
        # Project anchors to each context frame and sample features
        anchors = self.model.anchor_manager.anchors
        A = anchors.num_anchors

        # Average focal length for decoder
        avg_focal = float((intrinsics[0, :n_context_actual, 0, 0].mean() + intrinsics[0, :n_context_actual, 1, 1].mean()) / 2)

        # Re-sample anchor features from dense_features (differentiable)
        all_feats = []
        all_counts = []
        for s in range(n_context_actual):
            feat_map = dense_features[s]  # [C, ph, pw]
            cam = c2w[0, s]
            intr = intrinsics[0, s]
            d_map = depth[0, s, :, :, 0]

            # Project anchors
            pixel_coords, proj_d, vis = self.model.anchor_manager.project_anchors(
                cam, intr, H, W, d_map
            )
            if vis.any():
                vis_coords = pixel_coords[vis]
                sampled = self.model.anchor_manager._sample_features(feat_map, vis_coords, H, W)
                # Accumulate
                feat_accum = torch.zeros(A, sampled.shape[-1], device=depth.device)
                count = torch.zeros(A, device=depth.device)
                feat_accum[vis] = sampled
                count[vis] = 1.0
                all_feats.append(feat_accum)
                all_counts.append(count)

        if not all_feats:
            return None

        # Average features across views
        total_feats = torch.stack(all_feats).sum(dim=0)  # [A, C]
        total_counts = torch.stack(all_counts).sum(dim=0).clamp(min=1.0)  # [A]
        avg_feats = total_feats / total_counts.unsqueeze(-1)

        # Override anchor features with differentiable ones
        anchors.features = avg_feats

        # Decode Gaussians (differentiable!)
        anchor_depths = anchors.positions.norm(dim=-1)
        gs_dict = self.model.anchor_decoder(
            anchor_positions=anchors.positions.float(),
            anchor_features=anchors.features.float(),
            anchor_depths=anchor_depths.float(),
            focal_length=avg_focal,
            patch_size=self.model.patch_size,
        )
        return gs_dict

    def train_step(self, batch):
        """One training step on a single scene.

        Returns loss dict.
        """
        context_imgs = batch["context_images"].to(self.device)     # [B, Nc, 3, H, W]
        context_raw = batch["context_images_raw"].to(self.device)   # [B, Nc, 3, H, W]
        target_raw = batch["target_images_raw"].to(self.device)     # [B, Nt, 3, H, W]
        context_w2c = batch["context_w2c"].to(self.device)          # [B, Nc, 4, 4]
        context_K = batch["context_K"].to(self.device)              # [B, Nc, 3, 3]
        target_w2c = batch["target_w2c"].to(self.device)            # [B, Nt, 4, 4]
        target_K = batch["target_K"].to(self.device)                # [B, Nt, 3, 3]
        H, W = batch["image_size"][0].tolist()

        B = context_imgs.shape[0]
        assert B == 1, "Batch size must be 1 for streaming"
        Nc = context_imgs.shape[1]
        Nt = target_raw.shape[1]

        # Run frozen backbone on context frames
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pose_enc, depth, depth_conf, all_agg, ps = self._run_backbone(context_imgs)

        # Use GT cameras (from dataset) instead of predicted poses for training
        # This isolates the Gaussian quality from pose estimation errors
        c2w = torch.inverse(context_w2c)  # [B, Nc, 4, 4]
        intrinsics = context_K             # [B, Nc, 3, 3]

        # Build anchors and decode Gaussians (differentiable)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            gs_dict = self._build_anchors_and_decode(
                depth, depth_conf, all_agg, ps, c2w, intrinsics, H, W, Nc,
            )

        if gs_dict is None:
            logger.warning("No anchors created, skipping step")
            return {"loss": torch.tensor(0.0, device=self.device), "num_gaussians": 0,
                    "target_loss": 0.0, "context_loss": 0.0, "num_anchors": 0}

        # Render target views
        target_c2w = torch.inverse(target_w2c)
        rendered_targets = render_gaussians(
            gs_dict, target_w2c[0], target_K[0], H, W,
        )  # [Nt, 3, H, W]

        # Also render context views for self-supervision
        rendered_context = render_gaussians(
            gs_dict, context_w2c[0], context_K[0], H, W,
        )  # [Nc, 3, H, W]

        # Compute loss
        target_loss = rendering_loss(rendered_targets, target_raw[0])
        context_loss = rendering_loss(rendered_context, context_raw[0])
        loss = 0.5 * target_loss + 0.5 * context_loss

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()

        n_gs = gs_dict["positions"].shape[0] * gs_dict["positions"].shape[1]
        return {
            "loss": loss.item(),
            "target_loss": target_loss.item(),
            "context_loss": context_loss.item(),
            "num_gaussians": n_gs,
            "num_anchors": self.model.anchor_manager.anchors.num_anchors if self.model.anchor_manager.anchors else 0,
        }

    @torch.no_grad()
    def eval_step(self, batch):
        """Evaluation step. Returns metrics dict."""
        self.model.eval()

        context_imgs = batch["context_images"].to(self.device)
        context_raw = batch["context_images_raw"].to(self.device)
        target_raw = batch["target_images_raw"].to(self.device)
        context_w2c = batch["context_w2c"].to(self.device)
        context_K = batch["context_K"].to(self.device)
        target_w2c = batch["target_w2c"].to(self.device)
        target_K = batch["target_K"].to(self.device)
        H, W = batch["image_size"][0].tolist()

        Nc = context_imgs.shape[1]

        with torch.amp.autocast("cuda", dtype=torch.float16):
            pose_enc, depth, depth_conf, all_agg, ps = self._run_backbone(context_imgs)

        c2w = torch.inverse(context_w2c)
        intrinsics = context_K

        with torch.amp.autocast("cuda", dtype=torch.float16):
            gs_dict = self._build_anchors_and_decode(
                depth, depth_conf, all_agg, ps, c2w, intrinsics, H, W, Nc,
            )

        if gs_dict is None:
            return {"psnr": 0.0, "loss": 1.0}

        rendered = render_gaussians(gs_dict, target_w2c[0], target_K[0], H, W)
        gt = target_raw[0]

        mse = F.mse_loss(rendered, gt)
        psnr = -10 * torch.log10(mse.clamp(min=1e-8))
        loss = rendering_loss(rendered, gt)

        self.model.train()
        return {"psnr": psnr.item(), "loss": loss.item()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Streaming Anchor-GS")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/re10k")
    parser.add_argument("--output_dir", type=str, default="output/anchor_gs_train")
    # Training
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_context", type=int, default=5,
                        help="Number of context frames per scene")
    parser.add_argument("--num_target", type=int, default=2,
                        help="Number of target frames for rendering loss")
    parser.add_argument("--max_train_scenes", type=int, default=-1,
                        help="Limit number of training scenes (-1 = all)")
    parser.add_argument("--max_eval_scenes", type=int, default=16)
    # Model
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for training (smaller = faster)")
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--num_scale_frames", type=int, default=2)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=16)
    parser.add_argument("--anchor_voxel_size", type=float, default=0.1)
    parser.add_argument("--anchor_budget", type=int, default=20000)
    parser.add_argument("--gaussians_per_anchor", type=int, default=4)
    parser.add_argument("--use_sdpa", action="store_true")
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from lingbot_map.models.gct_stream_anchor_gs import GCTStreamAnchorGS

    logger.info("Loading model...")
    model = GCTStreamAnchorGS.from_pretrained_lingbot(
        args.model_path,
        device=device,
        img_size=args.image_size,
        patch_size=args.patch_size,
        max_frame_num=64,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        anchor_voxel_size=args.anchor_voxel_size,
        anchor_budget=args.anchor_budget,
        gaussians_per_anchor=args.gaussians_per_anchor,
    )
    model.train()

    # Load data
    logger.info("Loading dataset...")
    train_dataset = RE10KDataset(
        args.data_dir, split="train",
        num_context=args.num_context, num_target=args.num_target,
        image_size=args.image_size, patch_size=args.patch_size,
        max_scenes=args.max_train_scenes,
    )
    eval_dataset = RE10KDataset(
        args.data_dir, split="test",
        num_context=args.num_context, num_target=args.num_target,
        image_size=args.image_size, patch_size=args.patch_size,
        max_scenes=args.max_eval_scenes,
    )

    # No DataLoader shuffle needed since we sample frames randomly within each scene
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Trainer
    trainer = AnchorGSTrainer(model, args)

    # Training log
    log_file = os.path.join(args.output_dir, "train_log.jsonl")
    log_fh = open(log_file, "w")

    global_step = 0
    best_eval_psnr = 0.0

    logger.info(f"Starting training: {args.num_epochs} epochs, {len(train_dataset)} scenes")

    for epoch in range(args.num_epochs):
        epoch_losses = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            metrics = trainer.train_step(batch)
            step_time = time.time() - step_start

            epoch_losses.append(metrics["loss"])
            global_step += 1

            # Log
            if global_step % args.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-args.log_interval:])
                log_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": metrics["loss"],
                    "target_loss": metrics.get("target_loss", 0),
                    "context_loss": metrics.get("context_loss", 0),
                    "avg_loss": avg_loss,
                    "num_gaussians": metrics["num_gaussians"],
                    "num_anchors": metrics["num_anchors"],
                    "lr": trainer.optimizer.param_groups[0]["lr"],
                    "step_time": step_time,
                }
                log_fh.write(json.dumps(log_entry) + "\n")
                log_fh.flush()
                logger.info(
                    f"[E{epoch} S{global_step}] loss={metrics['loss']:.4f} "
                    f"avg={avg_loss:.4f} gs={metrics['num_gaussians']} "
                    f"anchors={metrics['num_anchors']} t={step_time:.1f}s"
                )

            # Eval
            if global_step % args.eval_interval == 0:
                eval_metrics = []
                for eval_batch in eval_loader:
                    em = trainer.eval_step(eval_batch)
                    eval_metrics.append(em)
                avg_psnr = np.mean([m["psnr"] for m in eval_metrics])
                avg_eval_loss = np.mean([m["loss"] for m in eval_metrics])
                logger.info(
                    f"[EVAL S{global_step}] PSNR={avg_psnr:.2f}dB loss={avg_eval_loss:.4f}"
                )
                eval_entry = {
                    "step": global_step,
                    "eval_psnr": avg_psnr,
                    "eval_loss": avg_eval_loss,
                    "type": "eval",
                }
                log_fh.write(json.dumps(eval_entry) + "\n")
                log_fh.flush()

                if avg_psnr > best_eval_psnr:
                    best_eval_psnr = avg_psnr
                    save_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save({
                        "anchor_token_proj": model.anchor_token_proj.state_dict(),
                        "anchor_decoder": model.anchor_decoder.state_dict(),
                        "step": global_step,
                        "psnr": avg_psnr,
                    }, save_path)
                    logger.info(f"Saved best model (PSNR={avg_psnr:.2f}dB) to {save_path}")

            # Save checkpoint
            if global_step % args.save_interval == 0:
                save_path = os.path.join(args.output_dir, f"ckpt_step{global_step}.pt")
                torch.save({
                    "anchor_token_proj": model.anchor_token_proj.state_dict(),
                    "anchor_decoder": model.anchor_decoder.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "scheduler": trainer.scheduler.state_dict(),
                    "step": global_step,
                    "epoch": epoch,
                }, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

        trainer.scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
        logger.info(
            f"Epoch {epoch} done: avg_loss={avg_epoch_loss:.4f} time={epoch_time:.0f}s"
        )

    log_fh.close()
    logger.info(f"Training complete. Best eval PSNR: {best_eval_psnr:.2f}dB")
    logger.info(f"Logs saved to {log_file}")


if __name__ == "__main__":
    main()
