"""
Analyze scalability of GCA-Splat Gaussian map construction.

Runs streaming inference on varying sequence lengths and records:
- Total Gaussians vs number of frames
- Memory usage
- Inference time / FPS
- Per-partition Gaussian counts (anchor, window, memory)

Demonstrates bounded memory growth via map management.

Usage:
    python analyze_scalability.py --image_folder example/courthouse_small --gs_head output_train_v5/gs_head_best.pt
"""

import argparse
import glob
import os
import time
import json

import torch
import numpy as np

from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.models.gct_stream_gs import GCTStreamGS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--gs_head", type=str, required=True)
    p.add_argument("--model_path", type=str,
                   default="C:/Users/LID0E/.cache/huggingface/hub/models--robbyant--lingbot-map"
                           "/snapshots/756d3c0b6d431cb95357228e3dc66ed8316772ef/lingbot-map-long.pt")
    p.add_argument("--gaussians_per_patch", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="output_scalability")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def run_experiment(model, images, n_frames, device):
    """Run streaming inference on first n_frames and return stats."""
    model.clean_kv_cache()
    imgs = images[:, :n_frames] if images.dim() == 5 else images[:n_frames]

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = model.inference_streaming_gs(
            imgs, num_scale_frames=2, keyframe_interval=1,
            output_device=torch.device("cpu"),
        )
    elapsed = time.time() - t0

    gmap = result["gaussian_map"]
    stats = gmap.get_stats()
    gs_stats = result["gaussian_stats"]

    return {
        "n_frames": n_frames,
        "total_gaussians": stats["total_gaussians"],
        "anchor_gaussians": stats["anchor_gaussians"],
        "window_gaussians": stats["window_gaussians"],
        "memory_gaussians": stats["memory_gaussians"],
        "inference_time": elapsed,
        "fps": n_frames / elapsed,
        "gs_per_frame_history": [s["total_gaussians"] for s in gs_stats],
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all images
    exts = [".jpg", ".png", ".JPG", ".PNG"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(args.image_folder, f"*{ext}")))
    paths = sorted(paths)
    print(f"Found {len(paths)} images")

    images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    if images.dim() == 4:
        images = images.unsqueeze(0)
    S = images.shape[1]

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
    print(f"Loaded GS head from {args.gs_head}")
    model = model.to(device).eval()

    # Run experiments at different sequence lengths
    frame_counts = sorted(set(
        [3, 5, 10, 15, 20, 25, 30, 40, 50] +
        list(range(10, S + 1, 10)) +
        [S]
    ))
    frame_counts = [n for n in frame_counts if n <= S]

    results = []
    for n in frame_counts:
        print(f"\nRunning with {n} frames...")
        torch.cuda.empty_cache()
        r = run_experiment(model, images, n, device)
        results.append(r)
        print(f"  Gaussians: {r['total_gaussians']:,} "
              f"(A={r['anchor_gaussians']}, W={r['window_gaussians']}, M={r['memory_gaussians']})")
        print(f"  Time: {r['inference_time']:.1f}s, FPS: {r['fps']:.1f}")

    # Save results
    output = {
        "total_images": S,
        "gaussians_per_patch": args.gaussians_per_patch,
        "experiments": results,
    }
    with open(os.path.join(args.output_dir, "scalability.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Frames':>8} {'Total GS':>10} {'Anchor':>8} {'Window':>8} {'Memory':>8} {'FPS':>6}")
    print(f"{'='*70}")
    for r in results:
        print(f"{r['n_frames']:>8} {r['total_gaussians']:>10,} "
              f"{r['anchor_gaussians']:>8,} {r['window_gaussians']:>8,} "
              f"{r['memory_gaussians']:>8,} {r['fps']:>6.1f}")
    print(f"{'='*70}")

    # Compute growth rate
    if len(results) >= 2:
        n1, g1 = results[0]["n_frames"], results[0]["total_gaussians"]
        n2, g2 = results[-1]["n_frames"], results[-1]["total_gaussians"]
        if g1 > 0 and g2 > 0 and n1 > 0 and n2 > 0:
            growth_rate = (g2 / g1) / (n2 / n1)
            print(f"\nGrowth ratio: {growth_rate:.2f}x "
                  f"({g1:,} GS @ {n1} frames → {g2:,} GS @ {n2} frames)")
            if growth_rate < 1.0:
                print("  → Sub-linear growth (bounded map!)")
            else:
                print("  → Linear or super-linear growth")

    print(f"\nResults saved to {args.output_dir}/scalability.json")


if __name__ == "__main__":
    main()
