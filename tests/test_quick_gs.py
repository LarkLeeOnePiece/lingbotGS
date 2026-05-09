"""Quick smoke test: 10 frames through GCTStreamGS pipeline."""
import sys, os, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

CKPT = "C:/Users/LID0E/.cache/huggingface/hub/models--robbyant--lingbot-map/snapshots/756d3c0b6d431cb95357228e3dc66ed8316772ef/lingbot-map-long.pt"
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "example", "courthouse_small")

def main():
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.models.gct_stream_gs import GCTStreamGS
    from lingbot_map.mapping.export import export_ply_pointcloud

    paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))[:10]
    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    print(f"  Shape: {images.shape}")

    print("Building model...")
    model = GCTStreamGS(
        img_size=518, patch_size=14, enable_3d_rope=False,
        kv_cache_sliding_window=16, kv_cache_scale_frames=2,
        kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
        use_sdpa=True, camera_num_iterations=1,
        gaussian_memory_budget=50_000, gaussian_voxel_size=0.05,
        gaussian_opacity_threshold=0.1,
    )

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    gs_missing = [k for k in missing if "gaussian_head" in k]
    print(f"  Loaded. GS head new params: {len(gs_missing)}")

    DEVICE = torch.device("cuda:0")
    model = model.to(DEVICE).eval()

    print(f"Running inference on {len(paths)} frames...")
    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = model.inference_streaming_gs(
            images, num_scale_frames=2, keyframe_interval=1,
            output_device=torch.device("cpu"),
        )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    gmap = result["gaussian_map"]
    stats = gmap.get_stats()
    print(f"  Anchor: {stats['anchor_gaussians']}, Window: {stats['window_gaussians']}, Memory: {stats['memory_gaussians']}, Total: {stats['total_gaussians']}")

    full_map = gmap.get_all_gaussians()
    if full_map and full_map.num_gaussians > 0:
        os.makedirs("output_test", exist_ok=True)
        n = export_ply_pointcloud(full_map, "output_test/test_pointcloud.ply", opacity_threshold=0.01)
        print(f"  Exported {n} points to output_test/test_pointcloud.ply")
    else:
        print("  WARNING: No Gaussians produced!")

    print("[PASS] Quick GS test successful!")

if __name__ == "__main__":
    main()
