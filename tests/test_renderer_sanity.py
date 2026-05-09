"""Sanity test: render known Gaussians and verify the image makes sense."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from lingbot_map.mapping.renderer import PureTorchRenderer
from PIL import Image
import numpy as np

def main():
    dev = torch.device("cuda")
    renderer = PureTorchRenderer(bg_color=(0, 0, 0)).to(dev)

    H, W = 128, 192
    K = torch.tensor([[200, 0, 96], [0, 200, 64], [0, 0, 1]], dtype=torch.float32, device=dev)
    viewmat = torch.eye(4, device=dev)

    # Place 4 colored Gaussians at known locations
    positions = torch.tensor([
        [-0.3, -0.2, 3.0],   # top-left, red
        [0.3, -0.2, 3.0],    # top-right, green
        [-0.3, 0.2, 3.0],    # bottom-left, blue
        [0.3, 0.2, 3.5],     # bottom-right, yellow (farther)
    ], device=dev)

    opacities = torch.tensor([[0.95], [0.95], [0.95], [0.95]], device=dev)
    scales = torch.tensor([[0.05, 0.05], [0.08, 0.03], [0.03, 0.08], [0.06, 0.06]], device=dev)
    normals = torch.tensor([[0, 0, -1.0]] * 4, device=dev)  # facing camera
    colors = torch.tensor([
        [1.0, 0.0, 0.0],   # red
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.0, 1.0],   # blue
        [1.0, 1.0, 0.0],   # yellow
    ], device=dev)

    out = renderer.render(positions, opacities, scales, normals, colors, viewmat, K, W, H)
    img = out["image"]
    alpha = out["alpha"]

    print(f"Image: {img.shape}, range [{img.min():.3f}, {img.max():.3f}]")
    print(f"Alpha: max={alpha.max():.3f}, mean={alpha.mean():.3f}")

    # Check that each quadrant has the correct dominant color
    mid_h, mid_w = H // 2, W // 2
    tl = img[:mid_h, :mid_w].mean(dim=(0, 1))
    tr = img[:mid_h, mid_w:].mean(dim=(0, 1))
    bl = img[mid_h:, :mid_w].mean(dim=(0, 1))
    br = img[mid_h:, mid_w:].mean(dim=(0, 1))

    print(f"  Top-left (should be red):    [{tl[0]:.3f}, {tl[1]:.3f}, {tl[2]:.3f}]")
    print(f"  Top-right (should be green): [{tr[0]:.3f}, {tr[1]:.3f}, {tr[2]:.3f}]")
    print(f"  Bottom-left (should be blue):[{bl[0]:.3f}, {bl[1]:.3f}, {bl[2]:.3f}]")
    print(f"  Bottom-right (should be yellow): [{br[0]:.3f}, {br[1]:.3f}, {br[2]:.3f}]")

    # Save image
    os.makedirs("output_test", exist_ok=True)
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save("output_test/renderer_sanity.png")
    print("Saved to output_test/renderer_sanity.png")

    # Test gradient flow
    pos = positions.clone().requires_grad_(True)
    col = colors.clone().requires_grad_(True)
    opa = opacities.clone().requires_grad_(True)
    sca = scales.clone().requires_grad_(True)
    out2 = renderer.render(pos, opa, sca, normals, col, viewmat, K, W, H)
    target = torch.ones(H, W, 3, device=dev) * 0.5  # gray target
    loss = (out2["image"] - target).abs().mean()
    loss.backward()
    print(f"\nGradient test: loss={loss.item():.4f}")
    print(f"  pos.grad norm: {pos.grad.norm():.6f}")
    print(f"  col.grad norm: {col.grad.norm():.6f}")
    print(f"  opa.grad norm: {opa.grad.norm():.6f}")
    print(f"  sca.grad norm: {sca.grad.norm():.6f}")
    has_nan = any(
        torch.isnan(p.grad).any()
        for p in [pos, col, opa, sca]
    )
    print(f"  NaN in grads: {has_nan}")
    print(f"\n[{'PASS' if not has_nan else 'FAIL'}] Renderer sanity test")


if __name__ == "__main__":
    main()
