"""Demo: Streaming Anchor-GS inference with PLY export.

Usage:
    python demo_anchor_gs.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/

    # With video input
    python demo_anchor_gs.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --video_fps 15

    # Export PLY
    python demo_anchor_gs.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/ --export_ply output.ply
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from lingbot_map.utils.load_fn import load_and_preprocess_images


def load_images(image_folder=None, video_path=None, fps=10,
                first_k=None, stride=1, image_size=518, patch_size=14):
    """Load and preprocess images from folder or video."""
    import cv2
    import glob
    from tqdm.auto import tqdm

    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                p = os.path.join(out_dir, f"{idx:06d}.jpg")
                if not os.path.exists(p):
                    cv2.imwrite(p, frame)
                saved.append(p)
            idx += 1
        cap.release()
        image_folder = out_dir
        paths = saved
        print(f"Extracted {len(paths)} frames from video")
    else:
        exts = (".jpg", ".png", ".JPG", ".PNG", ".jpeg", ".JPEG")
        paths = sorted([
            p for p in glob.glob(os.path.join(image_folder, "*"))
            if os.path.splitext(p)[1] in exts
        ])

    if stride > 1:
        paths = paths[::stride]
    if first_k:
        paths = paths[:first_k]

    print(f"Loading {len(paths)} images from {image_folder}")
    images = load_and_preprocess_images(paths, image_size=image_size, patch_size=patch_size)
    return images, paths, image_folder


def export_anchor_ply(filepath, positions, colors=None):
    """Export anchor positions as simple point cloud PLY."""
    n = len(positions)
    if colors is None:
        colors = np.full((n, 3), 128, dtype=np.uint8)
    elif colors.dtype == np.float32 or colors.dtype == np.float64:
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")


def _normal_to_quaternion(normals):
    """Convert surface normals to quaternions (rotation aligning z-axis to normal).

    Args:
        normals: [N, 3] unit normals
    Returns:
        quats: [N, 4] quaternions (w, x, y, z)
    """
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    n = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)

    # Rotation axis = cross(z, normal), angle = acos(dot(z, normal))
    dot = np.clip(n[:, 2], -1.0, 1.0)  # dot with z-axis
    cross = np.stack([-n[:, 1], n[:, 0], np.zeros(len(n), dtype=np.float32)], axis=-1)
    cross_norm = np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-8

    # Half-angle formula: q = [cos(theta/2), sin(theta/2) * axis]
    half_angle = np.arccos(dot) / 2.0
    w = np.cos(half_angle)
    axis = cross / cross_norm
    s = np.sin(half_angle)

    quats = np.stack([w, axis[:, 0] * s, axis[:, 1] * s, axis[:, 2] * s], axis=-1)

    # Handle degenerate case (normal ≈ z or ≈ -z)
    degenerate = cross_norm.squeeze() < 1e-6
    quats[degenerate & (dot >= 0)] = [1, 0, 0, 0]   # identity
    quats[degenerate & (dot < 0)] = [0, 1, 0, 0]     # 180° around x

    return quats.astype(np.float32)


def export_gaussians_3dgs_ply(filepath, gs_dict):
    """Export Gaussians in 3DGS-compatible PLY format (loadable by SuperSplat, etc.).

    Converts 2D surfels (scale[2] + normal) to 3DGS format (scale[3] + quaternion).
    """
    import struct

    pos = gs_dict["positions"].reshape(-1, 3).cpu().numpy().astype(np.float32)
    opacities = gs_dict["opacities"].reshape(-1, 1).cpu().numpy().astype(np.float32)
    scales_2d = gs_dict["scales"].reshape(-1, 2).cpu().numpy().astype(np.float32)
    normals = gs_dict["normals"].reshape(-1, 3).cpu().numpy().astype(np.float32)
    colors = gs_dict["colors"].reshape(-1, 3).cpu().numpy().astype(np.float32)
    N = len(pos)

    # Convert to 3DGS format
    # Scale: 2D tangent scales + near-zero normal scale (log-space)
    log_scales = np.log(np.clip(scales_2d, 1e-7, None))
    # Third scale (along normal) is very thin
    log_scale_z = np.full((N, 1), np.log(1e-5), dtype=np.float32)
    log_scales_3d = np.concatenate([log_scales, log_scale_z], axis=-1)

    # Rotation: normal → quaternion
    quats = _normal_to_quaternion(normals)

    # Opacity: inverse sigmoid (3DGS stores raw logit)
    opacity_logit = np.log(np.clip(opacities, 1e-6, 1 - 1e-6) /
                           (1 - np.clip(opacities, 1e-6, 1 - 1e-6)))

    # Color: convert RGB [0,1] to SH DC coefficient
    # SH DC = (color - 0.5) / 0.28209479177387814 (C0 constant)
    C0 = 0.28209479177387814
    sh_dc = (colors - 0.5) / C0

    # Write binary PLY
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(N):
            f.write(struct.pack('<3f', *pos[i]))                  # xyz
            f.write(struct.pack('<3f', *normals[i]))              # normals
            f.write(struct.pack('<3f', *sh_dc[i]))                # f_dc_0/1/2
            f.write(struct.pack('<f', opacity_logit[i, 0]))       # opacity
            f.write(struct.pack('<3f', *log_scales_3d[i]))        # scale_0/1/2
            f.write(struct.pack('<4f', *quats[i]))                # rot_0/1/2/3


def main():
    parser = argparse.ArgumentParser(description="Streaming Anchor-GS demo")
    # Input
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    # Model
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--num_scale_frames", type=int, default=2)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=16)
    parser.add_argument("--keyframe_interval", type=int, default=1)
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    # Anchor config
    parser.add_argument("--anchor_voxel_size", type=float, default=0.1)
    parser.add_argument("--anchor_budget", type=int, default=80000)
    parser.add_argument("--gaussians_per_anchor", type=int, default=4)
    parser.add_argument("--anchor_feature_dim", type=int, default=256)
    # Output
    parser.add_argument("--export_ply", type=str, default=None,
                        help="Export anchor point cloud to PLY file")
    parser.add_argument("--export_gaussians_ply", type=str, default=None,
                        help="Export decoded Gaussian positions to PLY file")
    args = parser.parse_args()

    assert args.image_folder or args.video_path, "Provide --image_folder or --video_path"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load images
    images, paths, image_folder = load_images(
        image_folder=args.image_folder,
        video_path=args.video_path,
        fps=args.video_fps,
        first_k=args.first_k,
        stride=args.stride,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    print(f"Images tensor: {images.shape}")

    # Load model
    from lingbot_map.models.gct_stream_anchor_gs import GCTStreamAnchorGS

    print("Building Anchor-GS model...")
    model = GCTStreamAnchorGS.from_pretrained_lingbot(
        args.model_path,
        device=device,
        img_size=args.image_size,
        patch_size=args.patch_size,
        max_frame_num=1024,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        anchor_feature_dim=args.anchor_feature_dim,
        anchor_voxel_size=args.anchor_voxel_size,
        anchor_budget=args.anchor_budget,
        gaussians_per_anchor=args.gaussians_per_anchor,
    )
    model.eval()
    print(f"Model loaded. Anchor heads: feature_head + decoder")

    # Run inference
    print("\nRunning streaming Anchor-GS inference...")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        result = model.inference_streaming_anchor_gs(
            images.to(device),
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
        )

    # Print stats
    anchor_mgr = result["anchor_manager"]
    stats = anchor_mgr.get_stats()
    print(f"\nFinal anchor stats:")
    print(f"  Anchors: {stats['num_anchors']}")
    print(f"  Avg observations: {stats['avg_obs']:.1f}")
    print(f"  Avg confidence: {stats['avg_conf']:.2f}")

    if "gaussians" in result:
        gs = result["gaussians"]
        K = gs["positions"].shape[1]
        total_gs = gs["positions"].shape[0] * K
        print(f"  Gaussians: {total_gs} ({stats['num_anchors']} anchors x {K})")

    # Print per-frame stats summary
    all_stats = result["anchor_stats"]
    print(f"\nAnchor growth over {len(all_stats)} frames:")
    for i, s in enumerate(all_stats):
        if i == 0 or i == len(all_stats) - 1 or i % max(1, len(all_stats) // 5) == 0:
            print(f"  Frame {i:4d}: anchors={s['num_anchors']:6d}, "
                  f"avg_obs={s['avg_obs']:.1f}, avg_conf={s['avg_conf']:.2f}")

    # Export PLY
    if args.export_ply and anchor_mgr.anchors is not None:
        positions = anchor_mgr.anchors.positions.cpu().numpy()
        confs = anchor_mgr.anchors.confidences.cpu().numpy()
        # Color by confidence (blue=low, red=high)
        conf_norm = np.clip((confs - confs.min()) / (confs.max() - confs.min() + 1e-6), 0, 1)
        colors = np.zeros((len(positions), 3), dtype=np.float32)
        colors[:, 0] = conf_norm       # red = high conf
        colors[:, 2] = 1 - conf_norm   # blue = low conf
        export_anchor_ply(args.export_ply, positions, colors)
        print(f"\nExported {len(positions)} anchors to {args.export_ply}")

    if args.export_gaussians_ply and "gaussians" in result:
        gs = result["gaussians"]
        export_gaussians_3dgs_ply(args.export_gaussians_ply, gs)
        total = gs["positions"].reshape(-1, 3).shape[0]
        print(f"Exported {total} Gaussians (3DGS format) to {args.export_gaussians_ply}")

    print("\nDone.")


if __name__ == "__main__":
    main()
