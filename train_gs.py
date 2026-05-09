"""
Train the CompactGaussianHead on top of a frozen LingBot-Map backbone.

Strategy:
- Freeze backbone (DINOv2 + GCA + camera head + depth head)
- Train only the GS head (~600K params) with differentiable rendering loss
- Use PureTorchRenderer (no CUDA JIT needed, portable)
- Self-supervised: render from same viewpoint, compare to input image

Usage:
    python train_gs.py --image_folder example/courthouse_small --epochs 50

    # Resume from checkpoint
    python train_gs.py --image_folder example/courthouse_small --resume output_train/gs_head.pt
"""

import argparse
import glob
import os
import random
import sys
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.models.gct_stream_gs import GCTStreamGS
from lingbot_map.mapping.renderer import PureTorchRenderer, rendering_loss
from lingbot_map.mapping.export import export_ply_pointcloud, export_splat_ply
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general


def parse_args():
    p = argparse.ArgumentParser(description="Train GCA-Splat GS head")
    # Input
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    # Model
    p.add_argument("--model_path", type=str,
                   default="C:/Users/LID0E/.cache/huggingface/hub/models--robbyant--lingbot-map"
                           "/snapshots/756d3c0b6d431cb95357228e3dc66ed8316772ef/lingbot-map-long.pt")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--patch_size", type=int, default=14)
    # Inference config
    p.add_argument("--kv_cache_sliding_window", type=int, default=16)
    p.add_argument("--num_scale_frames", type=int, default=2)
    p.add_argument("--camera_num_iterations", type=int, default=1)
    p.add_argument("--gaussians_per_patch", type=int, default=4,
                   help="K sub-Gaussians per patch (1, 4, or 9)")
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_l1", type=float, default=0.8)
    p.add_argument("--lambda_ssim", type=float, default=0.2)
    p.add_argument("--lambda_opacity_reg", type=float, default=0.001,
                   help="Opacity regularization to encourage sparsity")
    p.add_argument("--lambda_scale_reg", type=float, default=0.0,
                   help="Scale regularization to keep Gaussians compact")
    p.add_argument("--lambda_cross_view", type=float, default=0.0,
                   help="Cross-view rendering loss weight (0 = disabled)")
    p.add_argument("--cross_view_window", type=int, default=3,
                   help="Max frame distance for cross-view pairs")
    p.add_argument("--cross_view_mode", type=str, default="reproject",
                   choices=["render", "reproject"],
                   help="Cross-view loss mode: render (full re-render, slow) or reproject (fast)")
    p.add_argument("--scale_range_max", type=float, default=None,
                   help="Override max log-scale (default uses head's default)")
    p.add_argument("--render_chunk_size", type=int, default=16384)
    # Output
    p.add_argument("--output_dir", type=str, default="output_train")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def load_images(args):
    exts = [".jpg", ".png", ".JPG", ".PNG"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(args.image_folder, f"*{ext}")))
    paths = sorted(paths)
    if args.first_k:
        paths = paths[:args.first_k]
    if args.stride > 1:
        paths = paths[::args.stride]
    print(f"Loading {len(paths)} images from {args.image_folder}...")
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=args.image_size, patch_size=args.patch_size,
    )
    h, w = images.shape[-2:]
    print(f"  Preprocessed to {w}x{h}")
    return images, paths


def build_model(args, device):
    print("Building GCTStreamGS model...")
    model = GCTStreamGS(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=False,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=args.camera_num_iterations,
        gaussians_per_patch=args.gaussians_per_patch,
        gaussian_memory_budget=100_000,
        gaussian_voxel_size=0.05,
        gaussian_opacity_threshold=0.1,
    )

    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    gs_missing = [k for k in missing if "gaussian_head" in k]
    print(f"  GS head: {len(gs_missing)} new params")

    if args.resume:
        print(f"  Resuming GS head from: {args.resume}")
        gs_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.gaussian_head.load_state_dict(gs_ckpt["gs_head"])

    # Override scale range if specified
    if args.scale_range_max is not None:
        old_max = model.gaussian_head.scale_range[1]
        model.gaussian_head.scale_range = (model.gaussian_head.scale_range[0], args.scale_range_max)
        print(f"  Scale range: max {old_max} → {args.scale_range_max}")

    model = model.to(device)

    # Freeze everything except gaussian_head
    for name, param in model.named_parameters():
        if "gaussian_head" not in name:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def precompute_features(model, images, args, device):
    """Run frozen backbone once to cache per-frame features, depth, and poses.

    This avoids re-running the expensive backbone on every training epoch.
    Returns a list of per-frame dicts with cached intermediate tensors.
    """
    print("Precomputing backbone features (one-time)...")
    model.eval()

    if images.dim() == 4:
        images = images.unsqueeze(0)
    B, S, C, H, W = images.shape
    n_scale = min(args.num_scale_frames, S)

    model.clean_kv_cache()

    frame_cache = []

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # Phase 1: scale frames
        scale_imgs = images[:, :n_scale].to(device, non_blocking=True)
        torch.compiler.cudagraph_mark_step_begin()
        agg, ps = model._aggregate_features(
            scale_imgs, num_frame_for_scale=n_scale, num_frame_per_block=n_scale,
        )
        preds = {}
        preds.update(model._predict_camera(
            agg, causal_inference=True,
            num_frame_for_scale=n_scale, num_frame_per_block=n_scale,
        ))
        preds.update(model._predict_depth(agg, scale_imgs, ps))

        # Cache per-frame tokens and predictions
        for si in range(n_scale):
            tokens = [t[:, si:si+1].float().cpu() for t in agg]
            frame_cache.append({
                "tokens": tokens,
                "depth": preds["depth"][:, si:si+1].float().cpu(),
                "pose_enc": preds["pose_enc"][:, si:si+1].float().cpu(),
                "patch_start_idx": ps,
                "image": images[:, si:si+1].cpu(),
            })

        del preds, agg

        # Phase 2: causal streaming
        for i in tqdm(range(n_scale, S), desc="Caching features"):
            frame = images[:, i:i+1].to(device, non_blocking=True)
            torch.compiler.cudagraph_mark_step_begin()
            agg, ps = model._aggregate_features(
                frame, num_frame_for_scale=n_scale, num_frame_per_block=1,
            )
            fp = {}
            fp.update(model._predict_camera(
                agg, causal_inference=True,
                num_frame_for_scale=n_scale, num_frame_per_block=1,
            ))
            fp.update(model._predict_depth(agg, frame, ps))

            tokens = [t.float().cpu() for t in agg]
            frame_cache.append({
                "tokens": tokens,
                "depth": fp["depth"].float().cpu(),
                "pose_enc": fp["pose_enc"].float().cpu(),
                "patch_start_idx": ps,
                "image": images[:, i:i+1].cpu(),
            })
            del fp, agg

    print(f"  Cached {len(frame_cache)} frames")
    torch.cuda.empty_cache()
    return frame_cache


def get_viewmat_and_K(pose_enc, H, W, device):
    """Convert pose encoding to viewmat (w2c) and intrinsics."""
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(H, W), build_intrinsics=True
    )
    # extrinsics: [B, S, 3, 4] (world-to-camera)
    ext = extrinsics[0, 0]  # [3, 4]
    viewmat = torch.eye(4, device=device, dtype=ext.dtype)
    viewmat[:3, :] = ext
    K = intrinsics[0, 0]  # [3, 3]
    return viewmat, K


def cross_view_reprojection_loss(positions, colors, opacities, viewmat_j, K_j, target_j_chw, H, W):
    """Fast cross-view consistency via reprojection (no full render needed).

    Project frame i's Gaussian centers to frame j's image, sample GT color,
    and compare. Weighted by opacity so transparent Gaussians don't contribute.

    Args:
        positions: [N, 3] Gaussian positions in world space
        colors: [N, 3] Gaussian colors
        opacities: [N, 1] Gaussian opacities
        viewmat_j: [4, 4] world-to-camera for frame j
        K_j: [3, 3] intrinsics for frame j
        target_j_chw: [3, H, W] GT image for frame j (in [0, 1])
        H, W: image dimensions

    Returns:
        scalar loss
    """
    N = positions.shape[0]
    dev = positions.device

    # Project to frame j's camera space
    R = viewmat_j[:3, :3]
    t = viewmat_j[:3, 3]
    cam = positions @ R.T + t  # [N, 3]
    z = cam[:, 2]

    # Keep only points in front of camera and in-bounds
    valid = z > 0.01
    if valid.sum() < 10:
        return torch.tensor(0.0, device=dev, requires_grad=True)

    fx, fy = K_j[0, 0], K_j[1, 1]
    cx, cy = K_j[0, 2], K_j[1, 2]
    x_2d = cam[:, 0] / z.clamp(min=0.01) * fx + cx
    y_2d = cam[:, 1] / z.clamp(min=0.01) * fy + cy

    # Check screen bounds
    in_bounds = valid & (x_2d >= 0) & (x_2d < W) & (y_2d >= 0) & (y_2d < H)
    if in_bounds.sum() < 10:
        return torch.tensor(0.0, device=dev, requires_grad=True)

    # Normalize to [-1, 1] for grid_sample
    grid_x = 2.0 * x_2d[in_bounds] / (W - 1) - 1.0
    grid_y = 2.0 * y_2d[in_bounds] / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, M, 2]

    # Sample GT color at projected positions
    gt_sampled = F.grid_sample(
        target_j_chw.unsqueeze(0),  # [1, 3, H, W]
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # [1, 3, 1, M]
    gt_colors = gt_sampled[0, :, 0, :].T  # [M, 3]

    # Opacity-weighted L1 loss
    opa_weights = opacities[in_bounds].squeeze(-1).detach()  # [M] - detach opacity weights
    color_diff = (colors[in_bounds] - gt_colors).abs().mean(dim=-1)  # [M]
    loss = (opa_weights * color_diff).sum() / (opa_weights.sum() + 1e-8)

    return loss


def denormalize_image(img_tensor):
    """Undo ImageNet normalization. Input [1,1,3,H,W] or [3,H,W]."""
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])

    if img_tensor.dim() == 5:
        img = img_tensor[0, 0]  # [3, H, W]
    elif img_tensor.dim() == 4:
        img = img_tensor[0]
    else:
        img = img_tensor

    dev = img.device
    MEAN = MEAN.to(dev)
    STD = STD.to(dev)
    img = img * STD[:, None, None] + MEAN[:, None, None]
    return img.clamp(0, 1).permute(1, 2, 0)  # [H, W, 3]


def denormalize_image_bschw(img_tensor):
    """Undo ImageNet normalization, keep [B, S, 3, H, W] shape."""
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])

    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)
    # img_tensor: [B, S, 3, H, W]
    dev = img_tensor.device
    MEAN = MEAN.to(dev).reshape(1, 1, 3, 1, 1)
    STD = STD.to(dev).reshape(1, 1, 3, 1, 1)
    return (img_tensor * STD + MEAN).clamp(0, 1)


def train_epoch(model, renderer, frame_cache, optimizer, args, device, epoch):
    """Train GS head for one epoch over all cached frames."""
    model.gaussian_head.train()

    H, W = frame_cache[0]["image"].shape[-2:]
    n_frames = len(frame_cache)

    # Shuffle frame order each epoch
    indices = torch.randperm(n_frames).tolist()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0

    for idx in indices:
        fc = frame_cache[idx]

        # Move cached data to GPU
        tokens = [t.to(device) for t in fc["tokens"]]
        depth = fc["depth"].to(device)
        pose_enc = fc["pose_enc"].to(device)
        ps = fc["patch_start_idx"]
        target_img = denormalize_image(fc["image"]).to(device)  # [H, W, 3]

        # Denormalized image in [B, S, 3, H, W] for color sampling
        input_img_bschw = denormalize_image_bschw(fc["image"]).to(device)

        # Forward through GS head (trainable)
        gs = model.gaussian_head(
            tokens,
            depth=depth,
            pose_enc=pose_enc,
            patch_start_idx=ps,
            image_size_hw=(H, W),
            input_image=input_img_bschw,
        )

        # Extract per-frame Gaussians
        positions = gs["positions"][0, 0]   # [M, 3]
        opacities = gs["opacities"][0, 0]   # [M, 1]
        scales = gs["scales"][0, 0]         # [M, 2]
        normals = gs["normals"][0, 0]       # [M, 3]
        colors = gs["colors"][0, 0]         # [M, 3]

        # Get viewmat and intrinsics
        viewmat, K = get_viewmat_and_K(pose_enc, H, W, device)

        # Render
        rendered = renderer.render(
            positions, opacities, scales, normals, colors,
            viewmat, K, W, H,
        )
        rendered_img = rendered["image"]  # [H, W, 3]

        # Rendering loss
        losses = rendering_loss(
            rendered_img, target_img,
            lambda_l1=args.lambda_l1, lambda_ssim=args.lambda_ssim,
        )
        loss = losses["loss"]

        # Cross-view consistency loss
        if args.lambda_cross_view > 0 and n_frames > 1:
            # Pick a nearby frame j (different from i)
            w = args.cross_view_window
            lo = max(0, idx - w)
            hi = min(n_frames - 1, idx + w)
            candidates = [j for j in range(lo, hi + 1) if j != idx]
            if candidates:
                j = random.choice(candidates)
                fc_j = frame_cache[j]
                pose_enc_j = fc_j["pose_enc"].to(device)
                viewmat_j, K_j = get_viewmat_and_K(pose_enc_j, H, W, device)

                if args.cross_view_mode == "reproject":
                    # Fast: project Gaussian centers to frame j, compare colors
                    target_j_chw = denormalize_image_bschw(fc_j["image"]).to(device)[0, 0]  # [3, H, W]
                    cv_loss = cross_view_reprojection_loss(
                        positions, colors, opacities,
                        viewmat_j, K_j, target_j_chw, H, W,
                    )
                    loss = loss + args.lambda_cross_view * cv_loss
                    del target_j_chw, cv_loss
                else:
                    # Slow: full re-render from frame j's viewpoint (20% probabilistic)
                    if random.random() < 0.2:
                        target_j = denormalize_image(fc_j["image"]).to(device)
                        rendered_j = renderer.render(
                            positions, opacities, scales, normals, colors,
                            viewmat_j, K_j, W, H,
                        )
                        cross_losses = rendering_loss(
                            rendered_j["image"], target_j,
                            lambda_l1=args.lambda_l1, lambda_ssim=args.lambda_ssim,
                        )
                        loss = loss + args.lambda_cross_view * cross_losses["loss"]
                        del target_j, rendered_j, cross_losses

                del pose_enc_j, viewmat_j, K_j

        # Opacity regularization (encourage sparsity)
        if args.lambda_opacity_reg > 0:
            opa_reg = opacities.mean()
            loss = loss + args.lambda_opacity_reg * opa_reg

        # Scale regularization (keep Gaussians compact)
        if args.lambda_scale_reg > 0:
            scale_reg = scales.mean()
            loss = loss + args.lambda_scale_reg * scale_reg

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # NaN gradient guard
        for p in model.gaussian_head.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                p.grad.zero_()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.gaussian_head.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        total_l1 += losses["l1"].item()
        total_ssim += losses["ssim"].item()

        # Explicit cleanup to prevent GPU memory accumulation
        del tokens, depth, pose_enc, target_img, input_img_bschw
        del gs, positions, opacities, scales, normals, colors
        del viewmat, K, rendered, rendered_img, losses, loss

    avg_loss = total_loss / n_frames
    avg_l1 = total_l1 / n_frames
    avg_ssim = total_ssim / n_frames
    return {"loss": avg_loss, "l1": avg_l1, "ssim": avg_ssim}


def save_sample_renders(model, renderer, frame_cache, args, device, epoch):
    """Save a few sample renderings for visual inspection."""
    model.gaussian_head.eval()
    render_dir = os.path.join(args.output_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    H, W = frame_cache[0]["image"].shape[-2:]
    n = len(frame_cache)
    sample_indices = np.linspace(0, n - 1, min(5, n), dtype=int)

    from PIL import Image

    with torch.no_grad():
        for idx in sample_indices:
            fc = frame_cache[idx]
            tokens = [t.to(device) for t in fc["tokens"]]
            depth = fc["depth"].to(device)
            pose_enc = fc["pose_enc"].to(device)
            ps = fc["patch_start_idx"]
            target_img = denormalize_image(fc["image"]).to(device)
            input_img_bschw = denormalize_image_bschw(fc["image"]).to(device)

            gs = model.gaussian_head(
                tokens, depth=depth, pose_enc=pose_enc,
                patch_start_idx=ps, image_size_hw=(H, W),
                input_image=input_img_bschw,
            )
            positions = gs["positions"][0, 0]
            opacities = gs["opacities"][0, 0]
            scales = gs["scales"][0, 0]
            normals = gs["normals"][0, 0]
            colors = gs["colors"][0, 0]

            viewmat, K = get_viewmat_and_K(pose_enc, H, W, device)
            rendered = renderer.render(
                positions, opacities, scales, normals, colors,
                viewmat, K, W, H,
            )

            # Save rendered vs target side by side
            rendered_np = (rendered["image"].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            target_np = (target_img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            combined = np.concatenate([target_np, rendered_np], axis=1)

            Image.fromarray(combined).save(
                os.path.join(render_dir, f"epoch{epoch:03d}_frame{idx:03d}.png")
            )


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load images
    images, paths = load_images(args)

    # Build model
    model = build_model(args, device)
    renderer = PureTorchRenderer(
        pixel_chunk_size=args.render_chunk_size,
    ).to(device)

    # Precompute backbone features (frozen, one-time cost)
    frame_cache = precompute_features(model, images, args, device)
    del images  # free memory

    # Offload frozen backbone to CPU — only the GS head needs GPU for training
    import gc
    print(f"  GPU before offload: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    # Extract GS head, delete full model to free GPU memory
    gs_head = model.gaussian_head
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU after offload: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    gs_head = gs_head.to(device)

    # Create a lightweight wrapper so train_epoch can use gs_head directly
    class _GSHeadWrapper:
        def __init__(self, head):
            self.gaussian_head = head
    model = _GSHeadWrapper(gs_head)

    # Optimizer (only GS head params)
    optimizer = torch.optim.AdamW(
        model.gaussian_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # Training log
    log_path = os.path.join(args.output_dir, "train_log.json")
    train_log = []

    print(f"\n{'='*60}")
    print(f"Training GS head on {len(frame_cache)} frames for {args.epochs} epochs")
    print(f"  LR: {args.lr}, Weight Decay: {args.weight_decay}")
    xv = f" + {args.lambda_cross_view}*cross_view({args.cross_view_mode})" if args.lambda_cross_view > 0 else ""
    print(f"  Loss: {args.lambda_l1}*L1 + {args.lambda_ssim}*SSIM + {args.lambda_opacity_reg}*opa_reg{xv}")
    print(f"{'='*60}\n")

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        metrics = train_epoch(model, renderer, frame_cache, optimizer, args, device, epoch)
        scheduler.step()
        ep_time = time.time() - ep_t0

        lr = optimizer.param_groups[0]["lr"]
        log_entry = {"epoch": epoch, "lr": lr, "time": ep_time, **metrics}
        train_log.append(log_entry)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={metrics['loss']:.4f} l1={metrics['l1']:.4f} ssim={metrics['ssim']:.4f} | "
              f"lr={lr:.2e} | {ep_time:.1f}s")

        # Save best
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save({
                "gs_head": model.gaussian_head.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
            }, os.path.join(args.output_dir, "gs_head_best.pt"))

        # Periodic save
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({
                "gs_head": model.gaussian_head.state_dict(),
                "epoch": epoch,
                "loss": metrics["loss"],
            }, os.path.join(args.output_dir, "gs_head.pt"))

            save_sample_renders(model, renderer, frame_cache, args, device, epoch)

            # Save log
            with open(log_path, "w") as f:
                json.dump(train_log, f, indent=2)

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Best loss: {best_loss:.4f}")

    # Final export: rebuild full model and run streaming inference
    print("\nRunning final inference with trained GS head...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    full_model = build_model(args, device)
    full_model.gaussian_head.load_state_dict(gs_head.state_dict())
    full_model = full_model.eval()
    images_reload, _ = load_images(args)

    full_model._gs_opacity_threshold = 0.01
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = full_model.inference_streaming_gs(
            images_reload, num_scale_frames=args.num_scale_frames,
            keyframe_interval=1, output_device=torch.device("cpu"),
        )

    gmap = result["gaussian_map"]
    stats = gmap.get_stats()
    print(f"  Total Gaussians: {stats['total_gaussians']:,}")

    full_map = gmap.get_all_gaussians()
    if full_map and full_map.num_gaussians > 0:
        ply_path = os.path.join(args.output_dir, "pointcloud.ply")
        n = export_ply_pointcloud(full_map, ply_path, opacity_threshold=0.05)
        print(f"  Exported {n:,} points to {ply_path}")

        splat_path = os.path.join(args.output_dir, "gaussians.ply")
        n_gs = export_splat_ply(full_map, splat_path, opacity_threshold=0.05)
        print(f"  Exported {n_gs:,} Gaussians to {splat_path}")

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
