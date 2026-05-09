"""
End-to-end test: load LingBot-Map checkpoint, attach CompactGaussianHead,
run streaming GS inference on courthouse_small (50 frames).
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

CKPT = "C:/Users/LID0E/.cache/huggingface/hub/models--robbyant--lingbot-map/snapshots/756d3c0b6d431cb95357228e3dc66ed8316772ef/lingbot-map-long.pt"
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "example", "courthouse_small")
DEVICE = torch.device("cuda:0")


def main():
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.models.gct_stream_gs import GCTStreamGS

    # ---- Load images ----
    import glob
    paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
    # Use first 30 frames for faster test
    paths = paths[:30]
    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    h, w = images.shape[-2:]
    print(f"  Shape: {images.shape} ({w}x{h})")

    # ---- Build model (GCTStream + fresh Gaussian head) ----
    print("Building GCTStreamGS model...")
    model = GCTStreamGS(
        img_size=518,
        patch_size=14,
        enable_3d_rope=False,
        kv_cache_sliding_window=16,
        kv_cache_scale_frames=2,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,  # Windows: no FlashInfer
        camera_num_iterations=1,
        # GS-specific
        gaussian_hidden_dim=512,
        gaussian_memory_budget=50_000,
        gaussian_voxel_size=0.05,
        gaussian_opacity_threshold=0.1,
    )

    # Load LingBot-Map checkpoint (backbone, GCA, camera, depth)
    print(f"Loading checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    gs_missing = [k for k in missing if "gaussian_head" in k]
    other_missing = [k for k in missing if "gaussian_head" not in k]
    print(f"  Loaded. GS head params to train: {len(gs_missing)}")
    if other_missing:
        print(f"  WARNING: other missing keys: {other_missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model = model.to(DEVICE).eval()

    # Count params
    total = sum(p.numel() for p in model.parameters())
    gs_params = sum(p.numel() for p in model.gaussian_head.parameters())
    print(f"  Total params: {total:,}  GS head: {gs_params:,} ({100*gs_params/total:.1f}%)")

    # ---- Run streaming GS inference ----
    print(f"\nRunning inference_streaming_gs on {len(paths)} frames...")
    torch.cuda.reset_peak_memory_stats(DEVICE)
    t0 = time.time()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = model.inference_streaming_gs(
            images,
            num_scale_frames=2,
            keyframe_interval=1,
            output_device=torch.device("cpu"),
        )

    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3

    # ---- Results ----
    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"  Frames: {len(paths)}")
    print(f"  Time: {elapsed:.1f}s ({len(paths)/elapsed:.1f} FPS)")
    print(f"  Peak GPU: {peak_mem:.2f} GB")
    print(f"  Pose shape: {result['pose_enc'].shape}")
    print(f"  Depth shape: {result['depth'].shape}")

    # Gaussian map stats
    gmap = result["gaussian_map"]
    stats = gmap.get_stats()
    print(f"\n  Gaussian Map:")
    print(f"    Anchor:  {stats['anchor_gaussians']:,} Gaussians")
    print(f"    Window:  {stats['window_gaussians']:,} Gaussians ({stats['window_frames']} frames)")
    print(f"    Memory:  {stats['memory_gaussians']:,} Gaussians")
    print(f"    Total:   {stats['total_gaussians']:,} Gaussians")

    # Get full map and show memory footprint
    full_map = gmap.get_all_gaussians()
    if full_map is not None:
        n = full_map.num_gaussians
        bytes_per_gs = (3+1+2+3+3+1) * 4  # 13 floats * 4 bytes
        mem_mb = n * bytes_per_gs / 1024**2
        print(f"    Map memory: {mem_mb:.2f} MB ({n:,} × {bytes_per_gs} bytes)")

    # Per-frame stats evolution
    gs_stats = result["gaussian_stats"]
    print(f"\n  Per-frame total Gaussians:")
    for i in [0, len(gs_stats)//4, len(gs_stats)//2, 3*len(gs_stats)//4, -1]:
        s = gs_stats[i]
        print(f"    Frame {s['frames_processed']:3d}: "
              f"anchor={s['anchor_gaussians']:5d} "
              f"window={s['window_gaussians']:5d} "
              f"memory={s['memory_gaussians']:5d} "
              f"total={s['total_gaussians']:5d}")

    print(f"\n[PASS] End-to-end streaming GS inference successful!")


if __name__ == "__main__":
    main()
