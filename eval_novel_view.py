"""
Novel-view evaluation for GCA-Splat.

Protocol: leave-one-out with even/odd split.
- Training frames: even indices (0, 2, 4, ...) — Gaussians accumulated into map
- Test frames: odd indices (1, 3, 5, ...) — render map from their viewpoints, compare to GT

This tests whether the predicted 3D Gaussians have correct geometry for
synthesizing unseen viewpoints, not just reproducing the input view.

Usage:
    python eval_novel_view.py --image_folder example/courthouse_small \
        --gs_head output_train_v6_50f/gs_head_best.pt \
        --output_dir output_eval_novel_view
"""

import argparse
import glob
import os
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
import lpips
from PIL import Image
from tqdm.auto import tqdm

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.models.gct_stream_gs import GCTStreamGS
from lingbot_map.mapping.renderer import PureTorchRenderer, ssim_loss
from lingbot_map.mapping.gaussian_map import GaussianData
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--gs_head", type=str, required=True)
    p.add_argument("--model_path", type=str,
                   default="C:/Users/LID0E/.cache/huggingface/hub/models--robbyant--lingbot-map"
                           "/snapshots/756d3c0b6d431cb95357228e3dc66ed8316772ef/lingbot-map-long.pt")
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--gaussians_per_patch", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="output_eval_novel_view")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hold_out", type=str, default="odd",
                   choices=["odd", "even"],
                   help="Which frames to hold out for testing (default: odd)")
    p.add_argument("--max_render_gaussians", type=int, default=200_000,
                   help="Max Gaussians to render at once (memory limit)")
    p.add_argument("--nearest_k", type=int, default=0,
                   help="Use K nearest training frames per viewpoint (0 = full map)")
    return p.parse_args()


def psnr(pred, target):
    mse = (pred - target).pow(2).mean()
    if mse < 1e-10:
        return 100.0
    return -10 * torch.log10(mse).item()


def ssim_metric(pred, target):
    return 1.0 - ssim_loss(pred, target).item()


def l1_metric(pred, target):
    return (pred - target).abs().mean().item()


def lpips_metric(pred, target, lpips_fn):
    """Compute LPIPS. Input: [H, W, 3] in [0,1]."""
    # LPIPS expects [B, C, H, W] in [-1, 1]
    p = pred.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    t = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return lpips_fn(p, t).item()


def denormalize_image(img_tensor):
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    if img_tensor.dim() == 5:
        img = img_tensor[0, 0]
    elif img_tensor.dim() == 4:
        img = img_tensor[0]
    else:
        img = img_tensor
    dev = img.device
    img = img * STD[:, None, None].to(dev) + MEAN[:, None, None].to(dev)
    return img.clamp(0, 1).permute(1, 2, 0)


def denormalize_image_bschw(img_tensor):
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)
    dev = img_tensor.device
    MEAN = MEAN.to(dev).reshape(1, 1, 3, 1, 1)
    STD = STD.to(dev).reshape(1, 1, 3, 1, 1)
    return (img_tensor * STD + MEAN).clamp(0, 1)


def frustum_cull(gd, viewmat, K, width, height, margin=100, near=0.01, far=1000.0):
    """Cull Gaussians outside the camera frustum. Returns filtered GaussianData."""
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    cam_pos = gd.positions @ R.T + t  # [N, 3]
    z = cam_pos[:, 2]

    # Depth filter
    in_depth = (z > near) & (z < far)

    # Project to 2D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_2d = cam_pos[:, 0] / z.clamp(min=0.01) * fx + cx
    y_2d = cam_pos[:, 1] / z.clamp(min=0.01) * fy + cy

    # Screen bounds with margin
    in_screen = (
        (x_2d > -margin) & (x_2d < width + margin) &
        (y_2d > -margin) & (y_2d < height + margin)
    )

    mask = in_depth & in_screen
    return gd.filter(mask), mask.sum().item()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load images
    exts = [".jpg", ".png", ".JPG", ".PNG"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(args.image_folder, f"*{ext}")))
    paths = sorted(paths)
    if args.first_k:
        paths = paths[:args.first_k]
    S = len(paths)
    print(f"Loading {S} images...")
    images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    H, W = images.shape[-2:]

    # Define train/test split
    if args.hold_out == "odd":
        train_indices = set(range(0, S, 2))  # even
        test_indices = sorted(set(range(1, S, 2)))  # odd
    else:
        train_indices = set(range(1, S, 2))  # odd
        test_indices = sorted(set(range(0, S, 2)))  # even
    print(f"Train frames: {len(train_indices)}, Test frames: {len(test_indices)}")

    # Build model
    model = GCTStreamGS(
        img_size=518, patch_size=14, enable_3d_rope=False,
        kv_cache_sliding_window=16, kv_cache_scale_frames=2,
        kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
        use_sdpa=True, camera_num_iterations=1,
        gaussians_per_patch=args.gaussians_per_patch,
        gaussian_memory_budget=100_000, gaussian_voxel_size=0.05,
        gaussian_opacity_threshold=0.01,
    )
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    gs_ckpt = torch.load(args.gs_head, map_location="cpu", weights_only=False)
    model.gaussian_head.load_state_dict(gs_ckpt["gs_head"])
    print(f"Loaded GS head from {args.gs_head} (epoch {gs_ckpt.get('epoch', '?')})")
    model = model.to(device).eval()

    renderer = PureTorchRenderer(pixel_chunk_size=2048).to(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    # ======================================================================
    # Phase 1: Run streaming inference on ALL frames, collect per-frame data
    # ======================================================================
    print("\nPhase 1: Streaming inference (all frames)...")
    model.clean_kv_cache()
    if images.dim() == 4:
        images = images.unsqueeze(0)

    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 1, 3, 1, 1)
    images_denorm = (images.to(device) * _STD + _MEAN).clamp(0, 1)

    n_scale = 2
    per_frame = []  # list of dicts: {pose_enc, gs_data, is_train}

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # Scale frames
        scale_imgs = images[:, :n_scale].to(device)
        agg, ps = model._aggregate_features(
            scale_imgs, num_frame_for_scale=n_scale, num_frame_per_block=n_scale
        )
        preds = {}
        preds.update(model._predict_camera(
            agg, causal_inference=True,
            num_frame_for_scale=n_scale, num_frame_per_block=n_scale
        ))
        preds.update(model._predict_depth(agg, scale_imgs, ps))

        # Predict Gaussians for scale frames
        with torch.amp.autocast("cuda", enabled=False):
            tokens_fp32 = [t.float() for t in agg]
            gs = model.gaussian_head(
                tokens_fp32,
                depth=preds["depth"].float(),
                pose_enc=preds["pose_enc"].float(),
                patch_start_idx=ps,
                image_size_hw=(H, W),
                input_image=images_denorm[:, :n_scale].float(),
            )

        for si in range(n_scale):
            gd = GaussianData(
                positions=gs["positions"][0, si].cpu(),
                opacities=gs["opacities"][0, si].cpu(),
                scales=gs["scales"][0, si].cpu(),
                normals=gs["normals"][0, si].cpu(),
                colors=gs["colors"][0, si].cpu(),
                confidences=gs["opacities"][0, si].cpu(),
            )
            per_frame.append({
                "pose_enc": preds["pose_enc"][:, si:si+1].float().cpu(),
                "gs_data": gd,
                "is_train": si in train_indices,
            })
        del preds, gs, agg

        # Causal frames
        for i in tqdm(range(n_scale, S), desc="Inference"):
            frame = images[:, i:i+1].to(device)
            agg, ps = model._aggregate_features(
                frame, num_frame_for_scale=n_scale, num_frame_per_block=1
            )
            fp = {}
            fp.update(model._predict_camera(
                agg, causal_inference=True,
                num_frame_for_scale=n_scale, num_frame_per_block=1
            ))
            fp.update(model._predict_depth(agg, frame, ps))

            with torch.amp.autocast("cuda", enabled=False):
                tokens_fp32 = [t.float() for t in agg]
                gs = model.gaussian_head(
                    tokens_fp32,
                    depth=fp["depth"].float(),
                    pose_enc=fp["pose_enc"].float(),
                    patch_start_idx=ps,
                    image_size_hw=(H, W),
                    input_image=images_denorm[:, i:i+1].float(),
                )

            gd = GaussianData(
                positions=gs["positions"][0, 0].cpu(),
                opacities=gs["opacities"][0, 0].cpu(),
                scales=gs["scales"][0, 0].cpu(),
                normals=gs["normals"][0, 0].cpu(),
                colors=gs["colors"][0, 0].cpu(),
                confidences=gs["opacities"][0, 0].cpu(),
            )
            per_frame.append({
                "pose_enc": fp["pose_enc"].float().cpu(),
                "gs_data": gd,
                "is_train": i in train_indices,
            })
            del fp, gs, agg

    inference_time = time.time() - t0
    print(f"  Inference: {S} frames in {inference_time:.1f}s ({S/inference_time:.1f} FPS)")

    # Free model GPU memory — critical for WDDM systems
    del model, images_denorm
    torch.cuda.empty_cache()
    print(f"  GPU after model free: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    # ======================================================================
    # Phase 2: Prepare per-frame Gaussians and camera positions
    # ======================================================================
    print("\nPhase 2: Preparing per-frame Gaussians...")

    # Extract camera positions for all frames (for nearest-K selection)
    cam_positions = []
    for i, pf in enumerate(per_frame):
        # pose_enc: [1, 1, 9] — first 3 dims are absolute translation
        cam_pos = pf["pose_enc"][0, 0, :3]
        cam_positions.append(cam_pos)
    cam_positions = torch.stack(cam_positions)  # [S, 3]

    # Build per-frame filtered Gaussians for training frames
    train_frame_gs = {}  # frame_idx -> GaussianData
    for i, pf in enumerate(per_frame):
        if pf["is_train"]:
            opa = pf["gs_data"].opacities.squeeze(-1)
            mask = opa > 0.01
            if mask.any():
                train_frame_gs[i] = pf["gs_data"].filter(mask)

    train_frame_indices = sorted(train_frame_gs.keys())
    total_train_gs = sum(gd.num_gaussians for gd in train_frame_gs.values())
    print(f"  Train frames: {len(train_frame_indices)}, total GS: {total_train_gs:,}")

    use_nearest_k = args.nearest_k > 0
    if use_nearest_k:
        print(f"  Mode: nearest-{args.nearest_k} frames per viewpoint")
    else:
        print(f"  Mode: full accumulated map")

    # Build full accumulated map for non-nearest-K mode
    if not use_nearest_k:
        train_map = GaussianData.cat([train_frame_gs[i] for i in train_frame_indices])
        if train_map is None or train_map.num_gaussians == 0:
            print("  ERROR: No training Gaussians available!")
            return
        if train_map.num_gaussians > args.max_render_gaussians:
            print(f"  Subsampling {train_map.num_gaussians} -> {args.max_render_gaussians} Gaussians")
            perm = torch.randperm(train_map.num_gaussians)[:args.max_render_gaussians]
            mask = torch.zeros(train_map.num_gaussians, dtype=torch.bool)
            mask[perm] = True
            train_map = train_map.filter(mask)
        train_map_gpu = train_map.to(device)

    def get_gaussians_for_viewpoint(query_idx):
        """Get Gaussians to render for a given viewpoint index."""
        if not use_nearest_k:
            return train_map_gpu

        # Find K nearest training frames by camera distance
        query_pos = cam_positions[query_idx]
        train_pos = cam_positions[train_frame_indices]
        dists = (train_pos - query_pos.unsqueeze(0)).norm(dim=-1)
        k = min(args.nearest_k, len(train_frame_indices))
        _, topk_idx = dists.topk(k, largest=False)
        selected = [train_frame_indices[ti] for ti in topk_idx.tolist()]

        parts = [train_frame_gs[fi] for fi in selected]
        combined = GaussianData.cat(parts)
        return combined.to(device)

    # ======================================================================
    # Phase 3: Same-view eval (sanity check) + Novel-view eval
    # ======================================================================
    render_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(render_dir, exist_ok=True)

    # Same-view evaluation (per-frame, sanity check)
    print("\nPhase 3a: Same-view per-frame sanity check (5 samples)...")
    same_view_psnr = []
    same_view_ssim = []

    for i in sorted(train_indices)[:5]:
        pf = per_frame[i]
        gd = pf["gs_data"].to(device)
        pe = pf["pose_enc"].to(device)

        extr, intr = pose_encoding_to_extri_intri(pe, image_size_hw=(H, W), build_intrinsics=True)
        viewmat = torch.eye(4, device=device, dtype=torch.float32)
        viewmat[:3, :] = extr[0, 0].float()
        K = intr[0, 0].float()

        target = denormalize_image(images[:, i:i+1]).to(device)
        rendered = renderer.render(
            gd.positions.float(), gd.opacities.float(), gd.scales.float(),
            gd.normals.float(), gd.colors.float(), viewmat, K, W, H
        )
        same_view_psnr.append(psnr(rendered["image"], target))
        same_view_ssim.append(ssim_metric(rendered["image"], target))
        del gd

    print(f"  Per-frame (5 samples): PSNR={np.mean(same_view_psnr):.2f}, SSIM={np.mean(same_view_ssim):.4f}")

    # Novel-view evaluation
    mode_str = f"nearest-{args.nearest_k}" if use_nearest_k else "full-map"
    print(f"\nPhase 3b: Novel-view evaluation ({mode_str})...")

    novel_psnr = []
    novel_ssim = []
    novel_l1 = []
    novel_lpips = []

    for i in tqdm(test_indices, desc="Novel-view render"):
        pf = per_frame[i]
        pe = pf["pose_enc"].to(device)

        extr, intr = pose_encoding_to_extri_intri(pe, image_size_hw=(H, W), build_intrinsics=True)
        viewmat = torch.eye(4, device=device, dtype=torch.float32)
        viewmat[:3, :] = extr[0, 0].float()
        K = intr[0, 0].float()

        target = denormalize_image(images[:, i:i+1]).to(device)

        gs_for_view = get_gaussians_for_viewpoint(i)
        culled_map, n_culled = frustum_cull(gs_for_view, viewmat, K, W, H)

        with torch.no_grad():
            rendered = renderer.render(
                culled_map.positions.float(),
                culled_map.opacities.float(),
                culled_map.scales.float(),
                culled_map.normals.float(),
                culled_map.colors.float(),
                viewmat, K, W, H,
            )

        p_val = psnr(rendered["image"], target)
        s_val = ssim_metric(rendered["image"], target)
        l_val = l1_metric(rendered["image"], target)
        lp_val = lpips_metric(rendered["image"], target, lpips_fn)
        novel_psnr.append(p_val)
        novel_ssim.append(s_val)
        novel_l1.append(l_val)
        novel_lpips.append(lp_val)

        r_np = (rendered["image"].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        t_np = (target.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        combined = np.concatenate([t_np, r_np], axis=1)
        Image.fromarray(combined).save(os.path.join(render_dir, f"novel_{i:03d}.png"))

        if use_nearest_k:
            del gs_for_view

    # Train-view eval (from map/nearest-K)
    print(f"\nPhase 3c: Train-view eval ({mode_str})...")
    train_psnr = []
    train_ssim = []
    train_l1 = []
    train_lpips = []

    for i in tqdm(sorted(train_indices), desc="Train-view render"):
        pf = per_frame[i]
        pe = pf["pose_enc"].to(device)

        extr, intr = pose_encoding_to_extri_intri(pe, image_size_hw=(H, W), build_intrinsics=True)
        viewmat = torch.eye(4, device=device, dtype=torch.float32)
        viewmat[:3, :] = extr[0, 0].float()
        K = intr[0, 0].float()

        target = denormalize_image(images[:, i:i+1]).to(device)

        gs_for_view = get_gaussians_for_viewpoint(i)
        culled_map, _ = frustum_cull(gs_for_view, viewmat, K, W, H)

        with torch.no_grad():
            rendered = renderer.render(
                culled_map.positions.float(),
                culled_map.opacities.float(),
                culled_map.scales.float(),
                culled_map.normals.float(),
                culled_map.colors.float(),
                viewmat, K, W, H,
            )

        train_psnr.append(psnr(rendered["image"], target))
        train_ssim.append(ssim_metric(rendered["image"], target))
        train_l1.append(l1_metric(rendered["image"], target))
        train_lpips.append(lpips_metric(rendered["image"], target, lpips_fn))

        if i % max(1, len(train_indices) // 5) == 0:
            r_np = (rendered["image"].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            t_np = (target.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            combined = np.concatenate([t_np, r_np], axis=1)
            Image.fromarray(combined).save(os.path.join(render_dir, f"train_{i:03d}.png"))

        if use_nearest_k:
            del gs_for_view

    # ======================================================================
    # Report
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"Novel-View Evaluation Results")
    print(f"{'='*60}")
    print(f"  Scene: {args.image_folder}")
    print(f"  Total frames: {S}")
    print(f"  Train/Test split: {len(train_indices)}/{len(test_indices)}")
    print(f"  Total train Gaussians: {total_train_gs:,}")
    print(f"  Render mode: {mode_str}")
    print(f"")
    print(f"  Train views (from map):")
    print(f"    PSNR:  {np.mean(train_psnr):.2f} dB")
    print(f"    SSIM:  {np.mean(train_ssim):.4f}")
    print(f"    LPIPS: {np.mean(train_lpips):.4f}")
    print(f"    L1:    {np.mean(train_l1):.4f}")
    print(f"")
    print(f"  Novel views (held-out):")
    print(f"    PSNR:  {np.mean(novel_psnr):.2f} dB")
    print(f"    SSIM:  {np.mean(novel_ssim):.4f}")
    print(f"    LPIPS: {np.mean(novel_lpips):.4f}")
    print(f"    L1:    {np.mean(novel_l1):.4f}")
    print(f"")
    print(f"  Inference: {S/inference_time:.1f} FPS")
    print(f"{'='*60}")

    # Save results
    results = {
        "scene": args.image_folder,
        "num_frames": S,
        "num_train": len(train_indices),
        "num_test": len(test_indices),
        "hold_out": args.hold_out,
        "train_map_gaussians": total_train_gs,
        "render_mode": mode_str,
        "nearest_k": args.nearest_k,
        "inference_time": inference_time,
        "fps": S / inference_time,
        "train_view": {
            "psnr": float(np.mean(train_psnr)),
            "ssim": float(np.mean(train_ssim)),
            "lpips": float(np.mean(train_lpips)),
            "l1": float(np.mean(train_l1)),
            "psnr_per_frame": train_psnr,
            "ssim_per_frame": train_ssim,
            "lpips_per_frame": train_lpips,
        },
        "novel_view": {
            "psnr": float(np.mean(novel_psnr)),
            "ssim": float(np.mean(novel_ssim)),
            "lpips": float(np.mean(novel_lpips)),
            "l1": float(np.mean(novel_l1)),
            "psnr_per_frame": novel_psnr,
            "ssim_per_frame": novel_ssim,
            "lpips_per_frame": novel_lpips,
        },
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
