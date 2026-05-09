"""
Evaluate trained GCA-Splat model: compute PSNR, SSIM, and produce comparison figures.

Usage:
    python eval_gs.py --image_folder example/courthouse_small --gs_head output_train_v2/gs_head_best.pt
"""

import argparse
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
import lpips
from pathlib import Path
from PIL import Image

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.models.gct_stream_gs import GCTStreamGS
from lingbot_map.mapping.renderer import PureTorchRenderer, ssim_loss
from lingbot_map.mapping.export import export_ply_pointcloud, export_splat_ply
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--gs_head", type=str, default="output_train_v2/gs_head_best.pt")
    p.add_argument("--model_path", type=str,
                   default="C:/Users/LID0E/.cache/huggingface/hub/models--robbyant--lingbot-map"
                           "/snapshots/756d3c0b6d431cb95357228e3dc66ed8316772ef/lingbot-map-long.pt")
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--gaussians_per_patch", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="output_eval")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def psnr(pred, target):
    mse = (pred - target).pow(2).mean()
    if mse < 1e-10:
        return 100.0
    return -10 * torch.log10(mse).item()


def ssim_metric(pred, target):
    """Compute SSIM using the same function as training (1 - ssim_loss)."""
    return 1.0 - ssim_loss(pred, target).item()


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
    """Undo ImageNet normalization, keep [B, S, 3, H, W] shape."""
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)
    dev = img_tensor.device
    MEAN = MEAN.to(dev).reshape(1, 1, 3, 1, 1)
    STD = STD.to(dev).reshape(1, 1, 3, 1, 1)
    return (img_tensor * STD + MEAN).clamp(0, 1)


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
    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    H, W = images.shape[-2:]

    # Build model with trained GS head
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
    renderer = PureTorchRenderer().to(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    # Run streaming inference
    print("Running streaming GS inference...")
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = model.inference_streaming_gs(
            images, num_scale_frames=2, keyframe_interval=1,
            output_device=torch.device("cpu"),
        )
    elapsed = time.time() - t0
    S = len(paths)
    print(f"  {S} frames in {elapsed:.1f}s ({S/elapsed:.1f} FPS)")

    gmap = result["gaussian_map"]
    stats = gmap.get_stats()
    print(f"  Gaussians: {stats['total_gaussians']:,} total "
          f"(anchor={stats['anchor_gaussians']}, window={stats['window_gaussians']}, memory={stats['memory_gaussians']})")

    # Export PLY
    full_map = gmap.get_all_gaussians()
    if full_map and full_map.num_gaussians > 0:
        ply_path = os.path.join(args.output_dir, "pointcloud.ply")
        n = export_ply_pointcloud(full_map, ply_path, opacity_threshold=0.01)
        print(f"  Exported {n:,} points to {ply_path}")

        splat_path = os.path.join(args.output_dir, "gaussians.ply")
        n_gs = export_splat_ply(full_map, splat_path, opacity_threshold=0.01)
        print(f"  Exported {n_gs:,} Gaussians to {splat_path}")

    # Per-frame rendering evaluation
    print("\nPer-frame rendering evaluation...")

    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    render_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(render_dir, exist_ok=True)

    model.clean_kv_cache()
    if images.dim() == 4:
        images = images.unsqueeze(0)

    # Precompute features for rendering
    n_scale = 2
    with torch.no_grad():
        # Use autocast ONLY for backbone feature extraction
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            scale_imgs = images[:, :n_scale].to(device)
            agg, ps = model._aggregate_features(scale_imgs, num_frame_for_scale=n_scale, num_frame_per_block=n_scale)
            preds = {}
            preds.update(model._predict_camera(agg, causal_inference=True, num_frame_for_scale=n_scale, num_frame_per_block=n_scale))
            preds.update(model._predict_depth(agg, scale_imgs, ps))

        # Evaluate scale frames — NO autocast for GS head, renderer, and metrics
        for si in range(n_scale):
            tokens = [t[:, si:si+1].float() for t in agg]
            d = preds["depth"][:, si:si+1].float()
            pe = preds["pose_enc"][:, si:si+1].float()
            target = denormalize_image(images[:, si:si+1]).to(device)
            input_img_bschw = denormalize_image_bschw(images[:, si:si+1]).to(device)

            with torch.amp.autocast("cuda", enabled=False):
                gs = model.gaussian_head(tokens, depth=d, pose_enc=pe, patch_start_idx=ps, image_size_hw=(H, W), input_image=input_img_bschw)
            pos = gs["positions"][0, 0]
            opa = gs["opacities"][0, 0]
            sca = gs["scales"][0, 0]
            nrm = gs["normals"][0, 0]
            col = gs["colors"][0, 0]

            extr, intr = pose_encoding_to_extri_intri(pe, image_size_hw=(H, W), build_intrinsics=True)
            viewmat = torch.eye(4, device=device, dtype=torch.float32)
            viewmat[:3, :] = extr[0, 0].float()
            K = intr[0, 0].float()

            rendered = renderer.render(pos.float(), opa.float(), sca.float(), nrm.float(), col.float(), viewmat, K, W, H)
            p_val = psnr(rendered["image"], target)
            s_val = ssim_metric(rendered["image"], target)
            # LPIPS expects [B,C,H,W] in [-1,1]
            with torch.no_grad():
                ri = rendered["image"].permute(2, 0, 1).unsqueeze(0) * 2 - 1
                ti = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                lp_val = lpips_fn(ri, ti).item()
            psnr_vals.append(p_val)
            ssim_vals.append(s_val)
            lpips_vals.append(lp_val)

            # Save comparison
            r_np = (rendered["image"].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            t_np = (target.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            combined = np.concatenate([t_np, r_np], axis=1)
            Image.fromarray(combined).save(os.path.join(render_dir, f"frame_{si:03d}.png"))

        del preds, agg

        # Evaluate causal frames
        for i in range(n_scale, S):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                frame = images[:, i:i+1].to(device)
                agg, ps = model._aggregate_features(frame, num_frame_for_scale=n_scale, num_frame_per_block=1)
                fp = {}
                fp.update(model._predict_camera(agg, causal_inference=True, num_frame_for_scale=n_scale, num_frame_per_block=1))
                fp.update(model._predict_depth(agg, frame, ps))

            tokens = [t.float() for t in agg]
            d = fp["depth"].float()
            pe = fp["pose_enc"].float()
            target = denormalize_image(images[:, i:i+1]).to(device)
            input_img_bschw = denormalize_image_bschw(images[:, i:i+1]).to(device)

            with torch.amp.autocast("cuda", enabled=False):
                gs = model.gaussian_head(tokens, depth=d, pose_enc=pe, patch_start_idx=ps, image_size_hw=(H, W), input_image=input_img_bschw)
            pos = gs["positions"][0, 0]
            opa = gs["opacities"][0, 0]
            sca = gs["scales"][0, 0]
            nrm = gs["normals"][0, 0]
            col = gs["colors"][0, 0]

            extr, intr = pose_encoding_to_extri_intri(pe, image_size_hw=(H, W), build_intrinsics=True)
            viewmat = torch.eye(4, device=device, dtype=torch.float32)
            viewmat[:3, :] = extr[0, 0].float()
            K = intr[0, 0].float()

            rendered = renderer.render(pos.float(), opa.float(), sca.float(), nrm.float(), col.float(), viewmat, K, W, H)
            p_val = psnr(rendered["image"], target)
            s_val = ssim_metric(rendered["image"], target)
            with torch.no_grad():
                ri = rendered["image"].permute(2, 0, 1).unsqueeze(0) * 2 - 1
                ti = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                lp_val = lpips_fn(ri, ti).item()
            psnr_vals.append(p_val)
            ssim_vals.append(s_val)
            lpips_vals.append(lp_val)

            # Save comparison for select frames
            if i % max(1, S // 10) == 0 or i == S - 1:
                r_np = (rendered["image"].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                t_np = (target.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                combined = np.concatenate([t_np, r_np], axis=1)
                Image.fromarray(combined).save(os.path.join(render_dir, f"frame_{i:03d}.png"))

            del fp, agg

    # Report
    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    avg_lpips = np.mean(lpips_vals)
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({S} frames)")
    print(f"{'='*60}")
    print(f"  PSNR:  {avg_psnr:.2f} dB (per-frame range: {min(psnr_vals):.2f} - {max(psnr_vals):.2f})")
    print(f"  SSIM:  {avg_ssim:.4f} (per-frame range: {min(ssim_vals):.4f} - {max(ssim_vals):.4f})")
    print(f"  LPIPS: {avg_lpips:.4f} (per-frame range: {min(lpips_vals):.4f} - {max(lpips_vals):.4f})")
    print(f"  Total Gaussians: {stats['total_gaussians']:,}")
    print(f"  Inference FPS: {S/elapsed:.1f}")
    gs_per_frame = stats['total_gaussians'] / S
    mem_mb = stats['total_gaussians'] * 13 * 4 / 1024**2
    print(f"  Gaussians/frame: {gs_per_frame:.0f}")
    print(f"  Map size: {mem_mb:.2f} MB")
    print(f"{'='*60}")

    # Save results
    results = {
        "num_frames": S,
        "inference_time": elapsed,
        "fps": S / elapsed,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "avg_lpips": avg_lpips,
        "total_gaussians": stats['total_gaussians'],
        "gs_per_frame": gs_per_frame,
        "map_size_mb": mem_mb,
        "psnr_per_frame": psnr_vals,
        "ssim_per_frame": ssim_vals,
        "lpips_per_frame": lpips_vals,
    }
    import json
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
