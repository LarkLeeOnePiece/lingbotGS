"""
Multi-scene joint training for CompactGaussianHead.

Trains a single unified GS head across multiple scenes with balanced scene sampling.
This is Phase 1 of the large-scale training plan.

Key differences from train_gs.py (single scene):
- Loads and caches features for multiple scenes
- Balanced sampling: each iteration picks a random scene, then a random frame
- Iteration-based (not epoch-based) with warmup + cosine decay
- Leave-one-out evaluation for generalization testing

Usage:
    python train_gs_multi.py \
        --scenes courthouse:example/courthouse_small:50 \
                 university:example/university:50 \
                 oxford:example/oxford:50 \
                 loop:example/loop:50 \
        --iterations 50000 --lr 5e-4 \
        --output_dir output_train_multi

    # Hold out one scene for zero-shot eval
    python train_gs_multi.py \
        --scenes university:example/university:50 \
                 oxford:example/oxford:50 \
                 loop:example/loop:50 \
        --holdout_scene courthouse:example/courthouse_small:50 \
        --iterations 50000 --lr 5e-4 \
        --output_dir output_train_multi_no_courthouse
"""

import argparse
import glob
import gc
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.models.gct_stream_gs import GCTStreamGS
from lingbot_map.mapping.renderer import PureTorchRenderer, rendering_loss, ssim_loss
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    p = argparse.ArgumentParser(description="Multi-scene joint training for GCA-Splat GS head")
    # Scenes: name:folder:first_k
    p.add_argument("--scenes", nargs="+", required=True,
                   help="Scene specs: name:folder:first_k (e.g., courthouse:example/courthouse_small:50)")
    p.add_argument("--holdout_scene", type=str, default=None,
                   help="Holdout scene spec for zero-shot eval: name:folder:first_k")
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
    p.add_argument("--gaussians_per_patch", type=int, default=4)
    # Training
    p.add_argument("--iterations", type=int, default=50000,
                   help="Total training iterations")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--lambda_l1", type=float, default=0.8)
    p.add_argument("--lambda_ssim", type=float, default=0.2)
    p.add_argument("--lambda_opacity_reg", type=float, default=0.001)
    p.add_argument("--render_chunk_size", type=int, default=16384)
    # Output
    p.add_argument("--output_dir", type=str, default="output_train_multi")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def parse_scene_spec(spec):
    """Parse 'name:folder:first_k' into dict."""
    parts = spec.split(":")
    if len(parts) == 3:
        return {"name": parts[0], "folder": parts[1], "first_k": int(parts[2])}
    elif len(parts) == 2:
        return {"name": parts[0], "folder": parts[1], "first_k": None}
    else:
        raise ValueError(f"Invalid scene spec: {spec}. Expected name:folder:first_k")


def load_scene_images(folder, first_k, image_size, patch_size):
    """Load and preprocess images from a scene folder."""
    exts = [".jpg", ".png", ".JPG", ".PNG"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    paths = sorted(paths)
    if first_k:
        paths = paths[:first_k]
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=image_size, patch_size=patch_size,
    )
    return images, paths


def build_model(args, device):
    """Build model, load backbone weights, return model with frozen backbone."""
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

    model = model.to(device)

    # Freeze everything except gaussian_head
    for name, param in model.named_parameters():
        if "gaussian_head" not in name:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def precompute_scene_features(model, images, args, device):
    """Run frozen backbone once to cache per-frame features for one scene."""
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
        for i in tqdm(range(n_scale, S), desc="  Caching features", leave=False):
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

    torch.cuda.empty_cache()
    return frame_cache


def denormalize_image(img_tensor):
    """Undo ImageNet normalization. Input [1,1,3,H,W] or [3,H,W]."""
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    if img_tensor.dim() == 5:
        img = img_tensor[0, 0]
    elif img_tensor.dim() == 4:
        img = img_tensor[0]
    else:
        img = img_tensor
    dev = img.device
    MEAN = MEAN.to(dev)
    STD = STD.to(dev)
    img = img * STD[:, None, None] + MEAN[:, None, None]
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


def get_viewmat_and_K(pose_enc, H, W, device):
    """Convert pose encoding to viewmat (w2c) and intrinsics."""
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(H, W), build_intrinsics=True
    )
    ext = extrinsics[0, 0]
    viewmat = torch.eye(4, device=device, dtype=ext.dtype)
    viewmat[:3, :] = ext
    K = intrinsics[0, 0]
    return viewmat, K


def get_lr(step, warmup_iters, total_iters, base_lr):
    """Warmup + cosine annealing schedule."""
    if step < warmup_iters:
        return base_lr * step / max(warmup_iters, 1)
    progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
    return base_lr * 0.01 + 0.5 * (base_lr - base_lr * 0.01) * (1 + math.cos(math.pi * progress))


def train_step(gs_head, renderer, frame_cache, scene_name, optimizer, args, device):
    """Single training step: pick random frame, render, compute loss."""
    gs_head.train()

    # Pick random frame from this scene
    idx = random.randint(0, len(frame_cache) - 1)
    fc = frame_cache[idx]

    H, W = fc["image"].shape[-2:]

    # Move cached data to GPU
    tokens = [t.to(device) for t in fc["tokens"]]
    depth = fc["depth"].to(device)
    pose_enc = fc["pose_enc"].to(device)
    ps = fc["patch_start_idx"]
    target_img = denormalize_image(fc["image"]).to(device)
    input_img_bschw = denormalize_image_bschw(fc["image"]).to(device)

    # Forward through GS head
    gs = gs_head(
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
    rendered_img = rendered["image"]

    losses = rendering_loss(
        rendered_img, target_img,
        lambda_l1=args.lambda_l1, lambda_ssim=args.lambda_ssim,
    )
    loss = losses["loss"]

    if args.lambda_opacity_reg > 0:
        loss = loss + args.lambda_opacity_reg * opacities.mean()

    if torch.isnan(loss) or torch.isinf(loss):
        optimizer.zero_grad()
        return {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "scene": scene_name, "skipped": True}

    optimizer.zero_grad()
    loss.backward()

    for p in gs_head.parameters():
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            p.grad.zero_()

    torch.nn.utils.clip_grad_norm_(gs_head.parameters(), max_norm=0.5)
    optimizer.step()

    result = {
        "loss": loss.item(),
        "l1": losses["l1"].item(),
        "ssim": losses["ssim"].item(),
        "scene": scene_name,
        "skipped": False,
    }

    del tokens, depth, pose_enc, target_img, input_img_bschw
    del gs, positions, opacities, scales, normals, colors
    del viewmat, K, rendered, rendered_img, losses, loss

    return result


def evaluate_scene(gs_head, renderer, frame_cache, device):
    """Evaluate on a scene's frames. Returns avg PSNR and SSIM."""
    gs_head.eval()
    psnrs = []
    ssims = []

    with torch.no_grad():
        for idx in range(len(frame_cache)):
            fc = frame_cache[idx]
            H, W = fc["image"].shape[-2:]

            tokens = [t.to(device) for t in fc["tokens"]]
            depth = fc["depth"].to(device)
            pose_enc = fc["pose_enc"].to(device)
            ps = fc["patch_start_idx"]
            target_img = denormalize_image(fc["image"]).to(device)
            input_img_bschw = denormalize_image_bschw(fc["image"]).to(device)

            gs = gs_head(
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

            rendered_img = rendered["image"]

            mse = ((rendered_img - target_img) ** 2).mean().item()
            psnr = -10 * np.log10(mse + 1e-10)
            psnrs.append(psnr)

            ssim_val = 1.0 - ssim_loss(rendered_img, target_img).item()
            ssims.append(ssim_val)

            del tokens, depth, pose_enc, target_img, input_img_bschw
            del gs, positions, opacities, scales, normals, colors

    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "n_frames": len(frame_cache)}


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse scene specs
    scenes = [parse_scene_spec(s) for s in args.scenes]
    holdout = parse_scene_spec(args.holdout_scene) if args.holdout_scene else None

    print(f"Training scenes: {[s['name'] for s in scenes]}")
    if holdout:
        print(f"Holdout scene: {holdout['name']}")

    # Build model
    model = build_model(args, device)

    # Precompute features for each scene
    scene_caches = {}
    for scene in scenes + ([holdout] if holdout else []):
        name = scene["name"]
        print(f"\nPrecomputing features for '{name}' ({scene['folder']})...")
        images, paths = load_scene_images(
            scene["folder"], scene["first_k"], args.image_size, args.patch_size,
        )
        print(f"  Loaded {len(paths)} images")
        cache = precompute_scene_features(model, images, args, device)
        scene_caches[name] = cache
        print(f"  Cached {len(cache)} frames")
        del images

    total_frames = sum(len(c) for name, c in scene_caches.items() if name != (holdout["name"] if holdout else None))
    print(f"\nTotal training frames across all scenes: {total_frames}")

    # Offload frozen backbone to CPU
    gs_head = model.gaussian_head
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU after offload: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    gs_head = gs_head.to(device)

    renderer = PureTorchRenderer(pixel_chunk_size=args.render_chunk_size).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        gs_head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Training scene names (exclude holdout)
    train_scene_names = [s["name"] for s in scenes]

    # Training log
    log_path = os.path.join(args.output_dir, "train_log.json")
    train_log = []

    print(f"\n{'='*60}")
    print(f"Multi-scene joint training")
    print(f"  Scenes: {train_scene_names}")
    print(f"  Total frames: {total_frames}")
    print(f"  Iterations: {args.iterations}")
    print(f"  LR: {args.lr} (warmup: {args.warmup_iters})")
    print(f"  Loss: {args.lambda_l1}*L1 + {args.lambda_ssim}*SSIM + {args.lambda_opacity_reg}*opa_reg")
    print(f"{'='*60}\n")

    best_loss = float("inf")
    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_count = 0
    scene_losses = {name: [] for name in train_scene_names}
    t0 = time.time()

    for step in range(1, args.iterations + 1):
        # Update learning rate
        lr = get_lr(step, args.warmup_iters, args.iterations, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Balanced scene sampling: pick random scene, then random frame
        scene_name = random.choice(train_scene_names)
        frame_cache = scene_caches[scene_name]

        result = train_step(gs_head, renderer, frame_cache, scene_name, optimizer, args, device)

        if not result["skipped"]:
            running_loss += result["loss"]
            running_l1 += result["l1"]
            running_ssim += result["ssim"]
            running_count += 1
            scene_losses[scene_name].append(result["loss"])

        # Logging
        if step % args.log_every == 0 and running_count > 0:
            avg_loss = running_loss / running_count
            avg_l1 = running_l1 / running_count
            avg_ssim = running_ssim / running_count
            elapsed = time.time() - t0
            its_per_sec = step / elapsed

            # Per-scene recent loss
            scene_info = ""
            for sn in train_scene_names:
                recent = scene_losses[sn][-50:] if scene_losses[sn] else []
                if recent:
                    scene_info += f" {sn}={np.mean(recent):.4f}"

            log_entry = {
                "step": step, "lr": lr,
                "loss": avg_loss, "l1": avg_l1, "ssim": avg_ssim,
                "it/s": its_per_sec,
            }
            train_log.append(log_entry)

            print(f"Step {step:6d}/{args.iterations} | "
                  f"loss={avg_loss:.4f} l1={avg_l1:.4f} ssim={avg_ssim:.4f} | "
                  f"lr={lr:.2e} | {its_per_sec:.1f} it/s |{scene_info}")

            running_loss = 0.0
            running_l1 = 0.0
            running_ssim = 0.0
            running_count = 0

        # Save checkpoint
        if step % args.save_every == 0:
            avg_recent = np.mean([e["loss"] for e in train_log[-10:]]) if train_log else float("inf")

            torch.save({
                "gs_head": gs_head.state_dict(),
                "step": step,
                "loss": avg_recent,
                "scenes": train_scene_names,
            }, os.path.join(args.output_dir, "gs_head.pt"))

            if avg_recent < best_loss:
                best_loss = avg_recent
                torch.save({
                    "gs_head": gs_head.state_dict(),
                    "step": step,
                    "loss": best_loss,
                    "scenes": train_scene_names,
                }, os.path.join(args.output_dir, "gs_head_best.pt"))

            with open(log_path, "w") as f:
                json.dump(train_log, f, indent=2)

        # Evaluation
        if step % args.eval_every == 0:
            print(f"\n--- Evaluation at step {step} ---")
            eval_results = {}
            for name in train_scene_names:
                r = evaluate_scene(gs_head, renderer, scene_caches[name], device)
                eval_results[name] = r
                print(f"  {name}: PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} ({r['n_frames']} frames)")

            if holdout:
                r = evaluate_scene(gs_head, renderer, scene_caches[holdout["name"]], device)
                eval_results[holdout["name"] + " (holdout)"] = r
                print(f"  {holdout['name']} (HOLDOUT): PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f}")

            # Save eval results
            eval_path = os.path.join(args.output_dir, f"eval_step{step:06d}.json")
            with open(eval_path, "w") as f:
                json.dump(eval_results, f, indent=2)
            print()

            gs_head.train()

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Best loss: {best_loss:.4f}")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    final_results = {}
    for name in train_scene_names:
        r = evaluate_scene(gs_head, renderer, scene_caches[name], device)
        final_results[name] = r
        print(f"  {name}: PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f}")

    if holdout:
        r = evaluate_scene(gs_head, renderer, scene_caches[holdout["name"]], device)
        final_results[holdout["name"] + " (holdout)"] = r
        print(f"  {holdout['name']} (HOLDOUT): PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f}")

    with open(os.path.join(args.output_dir, "final_eval.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
