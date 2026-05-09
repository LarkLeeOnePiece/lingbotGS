"""
GCA-Splat Demo — Streaming 3D Gaussian Map from Monocular Video.

Usage:
    python demo_gs.py --image_folder example/courthouse_small --output_dir output_gs

Produces:
    output_gs/
        pointcloud.ply          — coloured point cloud (viewable in CloudCompare)
        gaussians.ply           — GS-viewer compatible PLY
        rendered_frame_*.png    — sample rendered views from the Gaussian map
        stats.txt               — per-frame Gaussian map statistics
"""

import argparse
import glob
import os
import sys
import time

import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.models.gct_stream_gs import GCTStreamGS
from lingbot_map.mapping.export import export_ply_pointcloud, export_splat_ply
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3


def parse_args():
    p = argparse.ArgumentParser(description="GCA-Splat streaming demo")
    # Input
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--video_path", type=str, default=None)
    p.add_argument("--video_fps", type=int, default=15)
    p.add_argument("--first_k", type=int, default=None,
                   help="Use only first K frames")
    p.add_argument("--stride", type=int, default=1)
    # Model
    p.add_argument("--model_path", type=str,
                   default="C:/Users/LID0E/.cache/huggingface/hub/models--robbyant--lingbot-map"
                           "/snapshots/756d3c0b6d431cb95357228e3dc66ed8316772ef/lingbot-map-long.pt")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--patch_size", type=int, default=14)
    # Inference
    p.add_argument("--use_sdpa", action="store_true", default=True,
                   help="Use SDPA backend (default on Windows)")
    p.add_argument("--kv_cache_sliding_window", type=int, default=16)
    p.add_argument("--num_scale_frames", type=int, default=2)
    p.add_argument("--camera_num_iterations", type=int, default=1)
    p.add_argument("--keyframe_interval", type=int, default=1)
    # Gaussian map
    p.add_argument("--gaussian_memory_budget", type=int, default=100_000)
    p.add_argument("--gaussian_voxel_size", type=float, default=0.05)
    p.add_argument("--gaussian_opacity_threshold", type=float, default=0.1)
    # Output
    p.add_argument("--output_dir", type=str, default="output_gs")
    p.add_argument("--render_samples", type=int, default=5,
                   help="Number of sample rendered views to save")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def load_images(args):
    """Load and preprocess images from folder."""
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
    """Build GCTStreamGS and load checkpoint."""
    print("Building GCTStreamGS model...")
    model = GCTStreamGS(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=False,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        gaussian_memory_budget=args.gaussian_memory_budget,
        gaussian_voxel_size=args.gaussian_voxel_size,
        gaussian_opacity_threshold=args.gaussian_opacity_threshold,
    )

    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    gs_missing = [k for k in missing if "gaussian_head" in k]
    other_missing = [k for k in missing if "gaussian_head" not in k]
    print(f"  Loaded. GS head: {len(gs_missing)} new params (randomly initialised)")
    if other_missing:
        print(f"  Warning: {len(other_missing)} non-GS keys missing")

    gs_params = sum(p.numel() for p in model.gaussian_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,} total, {gs_params:,} GS head ({100*gs_params/total_params:.1f}%)")

    return model.to(device).eval()


def render_sample_views(result, args, device):
    """Render a few sample views from the Gaussian map."""
    try:
        from lingbot_map.mapping.renderer import GaussianRenderer
    except ImportError as e:
        print(f"  Skipping rendering: {e}")
        return

    gmap = result["gaussian_map"]
    full_map = gmap.get_all_gaussians()
    if full_map is None or full_map.num_gaussians == 0:
        print("  No Gaussians to render")
        return

    full_map = full_map.to(device)
    renderer = GaussianRenderer(sh_degree=0).to(device)

    pose_enc = result["pose_enc"]
    B, S = pose_enc.shape[:2]
    images = result["images"]
    H, W = images.shape[-2:]

    # Pick evenly-spaced frames to render from
    indices = np.linspace(0, S - 1, min(args.render_samples, S), dtype=int)
    render_dir = os.path.join(args.output_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    for idx in indices:
        pe = pose_enc[0, idx:idx+1]  # [1, 9]
        extr, intr = pose_encoding_to_extri_intri(
            pe.unsqueeze(0), image_size_hw=(H, W), build_intrinsics=True,
        )
        # extr [1, 1, 3, 4], intr [1, 1, 3, 3]
        extr_44 = torch.eye(4, device=device, dtype=extr.dtype)
        extr_44[:3, :] = extr[0, 0]

        try:
            rendered = renderer.render_from_gaussian_data(
                full_map, extr_44, intr[0, 0], W, H,
            )
            img_np = (rendered["image"].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            from PIL import Image
            Image.fromarray(img_np).save(
                os.path.join(render_dir, f"rendered_frame_{idx:04d}.png")
            )
        except Exception as e:
            print(f"  Render frame {idx} failed: {e}")
            continue

    print(f"  Saved {len(indices)} rendered views to {render_dir}/")


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load images
    images, paths = load_images(args)

    # Build model
    model = build_model(args, device)

    # Run streaming GS inference
    print(f"\nRunning streaming GS inference on {len(paths)} frames...")
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = model.inference_streaming_gs(
            images,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=torch.device("cpu"),
        )

    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    fps = len(paths) / elapsed

    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"  Frames: {len(paths)}, Time: {elapsed:.1f}s, FPS: {fps:.1f}")
    print(f"  Peak GPU: {peak_mem:.2f} GB")

    # Map stats
    gmap = result["gaussian_map"]
    stats = gmap.get_stats()
    print(f"\n  Gaussian Map:")
    print(f"    Anchor:  {stats['anchor_gaussians']:>7,}")
    print(f"    Window:  {stats['window_gaussians']:>7,} ({stats['window_frames']} frames)")
    print(f"    Memory:  {stats['memory_gaussians']:>7,}")
    print(f"    Total:   {stats['total_gaussians']:>7,}")

    full_map = gmap.get_all_gaussians()
    if full_map:
        bytes_per_gs = 13 * 4
        mem_mb = full_map.num_gaussians * bytes_per_gs / 1024**2
        print(f"    Map size: {mem_mb:.2f} MB")

    # Export PLY
    print(f"\nExporting...")
    ply_path = os.path.join(args.output_dir, "pointcloud.ply")
    n_exported = export_ply_pointcloud(full_map, ply_path)
    print(f"  Point cloud: {ply_path} ({n_exported:,} points)")

    splat_path = os.path.join(args.output_dir, "gaussians.ply")
    n_gs = export_splat_ply(full_map, splat_path)
    print(f"  GS PLY:      {splat_path} ({n_gs:,} Gaussians)")

    # Save stats
    stats_path = os.path.join(args.output_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Frames: {len(paths)}\n")
        f.write(f"Time: {elapsed:.1f}s\n")
        f.write(f"FPS: {fps:.1f}\n")
        f.write(f"Peak GPU: {peak_mem:.2f} GB\n")
        f.write(f"Anchor: {stats['anchor_gaussians']}\n")
        f.write(f"Window: {stats['window_gaussians']}\n")
        f.write(f"Memory: {stats['memory_gaussians']}\n")
        f.write(f"Total: {stats['total_gaussians']}\n")
        f.write(f"\nPer-frame stats:\n")
        for i, s in enumerate(result["gaussian_stats"]):
            f.write(f"  {i:4d}: a={s['anchor_gaussians']:5d} "
                    f"w={s['window_gaussians']:5d} "
                    f"m={s['memory_gaussians']:5d} "
                    f"t={s['total_gaussians']:5d}\n")

    # Render sample views
    print(f"\nRendering sample views from Gaussian map...")
    render_sample_views(result, args, device)

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
