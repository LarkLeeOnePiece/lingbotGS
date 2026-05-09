"""Smoke test for CompactGaussianHead and GaussianMapManager."""

import torch
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_gaussian_head_shapes():
    """Verify CompactGaussianHead produces correct output shapes."""
    from lingbot_map.heads.gaussian_head import CompactGaussianHead

    B, S, M, C = 1, 2, 999, 2048
    H, W = 378, 518
    patch_start_idx = 6  # camera(1) + register(4) + scale(1)

    head = CompactGaussianHead(dim_in=C, hidden_dim=256, patch_size=14)
    head.eval()

    # Fake inputs
    total_tokens = patch_start_idx + M
    tokens = torch.randn(B, S, total_tokens, C)
    depth = torch.ones(B, S, H, W, 1) * 2.0  # 2m depth everywhere
    pose_enc = torch.zeros(B, S, 9)
    pose_enc[..., 3] = 1.0   # identity quaternion (w=1)
    pose_enc[..., 7:] = 1.0  # ~57 degree FOV

    with torch.no_grad():
        out = head(
            aggregated_tokens_list=[tokens],
            depth=depth,
            pose_enc=pose_enc,
            patch_start_idx=patch_start_idx,
            image_size_hw=(H, W),
        )

    assert out["positions"].shape == (B, S, M, 3), f"positions: {out['positions'].shape}"
    assert out["opacities"].shape == (B, S, M, 1), f"opacities: {out['opacities'].shape}"
    assert out["scales"].shape == (B, S, M, 2), f"scales: {out['scales'].shape}"
    assert out["normals"].shape == (B, S, M, 3), f"normals: {out['normals'].shape}"
    assert out["colors"].shape == (B, S, M, 3), f"colors: {out['colors'].shape}"
    assert out["valid_mask"].shape == (B, S, M), f"valid_mask: {out['valid_mask'].shape}"

    # Check activations are in expected ranges
    assert (out["opacities"] >= 0).all() and (out["opacities"] <= 1).all()
    assert (out["colors"] >= 0).all() and (out["colors"] <= 1).all()
    norm_lengths = out["normals"].norm(dim=-1)
    assert torch.allclose(norm_lengths, torch.ones(B, S, M), atol=1e-4), \
        f"Normal norms not unit: min={norm_lengths.min():.6f} max={norm_lengths.max():.6f}"

    print(f"[PASS] CompactGaussianHead output shapes correct")
    print(f"       positions: {out['positions'].shape}")
    print(f"       params per surfel: 12, total surfels/frame: {M}")
    print(f"       param count: {sum(p.numel() for p in head.parameters()):,}")
    return True


def test_gaussian_map_manager():
    """Verify GaussianMapManager three-level lifecycle."""
    from lingbot_map.mapping.gaussian_map import GaussianMapManager

    mgr = GaussianMapManager(
        num_anchor_frames=2,
        window_size=4,
        memory_budget=500,
        voxel_size=0.1,
        opacity_threshold=0.05,
        device=torch.device("cpu"),
    )

    M = 100  # surfels per frame

    for i in range(10):
        pos = torch.randn(M, 3) + i * 0.5  # slight translation per frame
        opa = torch.rand(M, 1) * 0.5 + 0.3
        sca = torch.rand(M, 2) * 0.1
        nor = torch.randn(M, 3)
        nor = nor / nor.norm(dim=-1, keepdim=True)
        col = torch.rand(M, 3)

        mgr.add_frame(i, pos, opa, sca, nor, col)
        stats = mgr.get_stats()

        if i < 2:
            # Should be in anchor pool
            assert stats["anchor_gaussians"] > 0
            assert stats["window_frames"] == 0
        elif i < 6:
            # Window not full yet
            assert stats["window_frames"] == i - 1  # frames 2..i in window
        else:
            # Window full, evictions happening
            assert stats["window_frames"] == 4
            assert stats["memory_gaussians"] > 0

    final = mgr.get_stats()
    assert final["total_gaussians"] <= (
        final["anchor_gaussians"] + 4 * M + 500
    ), "Total should be bounded"

    full_map = mgr.get_all_gaussians()
    assert full_map is not None
    assert full_map.num_gaussians == final["total_gaussians"]

    print(f"[PASS] GaussianMapManager lifecycle correct")
    print(f"       Final stats: {final}")
    return True


def test_novelty_gate():
    """Verify novelty gating reduces effective opacity."""
    from lingbot_map.heads.gaussian_head import CompactGaussianHead

    B, S, C = 1, 1, 2048
    H, W = 378, 518
    ps = 6
    M = (H // 14) * (W // 14)  # 27 * 37 = 999

    head = CompactGaussianHead(dim_in=C, hidden_dim=128, patch_size=14)
    head.eval()

    tokens = torch.randn(B, S, ps + M, C)
    depth = torch.ones(B, S, H, W, 1) * 3.0
    pose_enc = torch.zeros(B, S, 9)
    pose_enc[..., 3] = 1.0
    pose_enc[..., 7:] = 1.0

    with torch.no_grad():
        out_no_gate = head([tokens], depth, pose_enc, ps, (H, W))
        gate = torch.zeros(B, S, M)  # fully gated = suppress all
        out_gated = head([tokens], depth, pose_enc, ps, (H, W), novelty_gate=gate)

    # Gated output should have zero opacity
    assert (out_gated["opacities"] == 0).all(), "Full gate should zero opacity"
    # Ungated should have non-zero opacity
    assert (out_no_gate["opacities"] > 0).any(), "Ungated should have some opacity"
    print("[PASS] Novelty gating works correctly")
    return True


if __name__ == "__main__":
    test_gaussian_head_shapes()
    test_gaussian_map_manager()
    test_novelty_gate()
    print("\nAll tests passed!")
