"""Generate side-by-side comparison: GT | Full-Map | Nearest-1 for the paper."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

base = "D:/Projects/SLAM-GS/lingbotMap/lingbot-map"
output_dir = os.path.join(base, "figures")
os.makedirs(output_dir, exist_ok=True)

# Frames to show
frames = [1, 13, 25, 49]

rows = []
for fidx in frames:
    fname = f"novel_{fidx:03d}.png"

    # Full-map comparison (GT | fullmap)
    fullmap_path = os.path.join(base, "output_eval_novel_v8_50f", "comparisons", fname)
    nearest1_path = os.path.join(base, "output_eval_novel_v6_full_nearest1", "comparisons", fname)

    if not os.path.exists(fullmap_path) or not os.path.exists(nearest1_path):
        print(f"  Skipping frame {fidx}: missing files")
        continue

    fullmap_img = np.array(Image.open(fullmap_path))
    nearest1_img = np.array(Image.open(nearest1_path))

    h, w_total = fullmap_img.shape[:2]
    w = w_total // 2

    gt = fullmap_img[:, :w]
    render_fullmap = fullmap_img[:, w:]
    render_nearest1 = nearest1_img[:, w:]

    # Create separator
    sep = np.ones((h, 2, 3), dtype=np.uint8) * 200

    row = np.concatenate([gt, sep, render_fullmap, sep, render_nearest1], axis=1)
    rows.append(row)

if rows:
    # Add horizontal separator between rows
    sep_h = np.ones((2, rows[0].shape[1], 3), dtype=np.uint8) * 200
    all_rows = []
    for i, row in enumerate(rows):
        if i > 0:
            all_rows.append(sep_h)
        all_rows.append(row)

    grid = np.concatenate(all_rows, axis=0)

    # Add labels at top
    label_h = 25
    label_bar = np.ones((label_h, grid.shape[1], 3), dtype=np.uint8) * 255
    final = np.concatenate([label_bar, grid], axis=0)

    img = Image.fromarray(final)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    # Label positions (center of each column)
    col_w = rows[0].shape[1] // 3
    for label, offset in [("Ground Truth", 0), ("Full Map (22.5 dB)", col_w), ("Nearest-1 (35.2 dB)", 2 * col_w)]:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        x = offset + (col_w - tw) // 2
        draw.text((x, 4), label, fill=(0, 0, 0), font=font)

    out_path = os.path.join(output_dir, "comparison_fullmap_vs_nearest1.png")
    img.save(out_path, quality=95)
    print(f"Saved comparison figure: {out_path} ({len(rows)} rows)")
