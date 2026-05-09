#!/bin/bash
# Full training + evaluation pipeline for GCA-Splat paper
# Run sequentially (one GPU)
set -e
export PYTHONUNBUFFERED=1
cd D:/Projects/SLAM-GS/lingbotMap/lingbot-map

echo "============================================"
echo "GCA-Splat Full Pipeline"
echo "============================================"

# --- Step 1: v6 baseline (no cross-view, 50 frames, 300 epochs) ---
echo ""
echo "=== Step 1: v6 baseline training (no cross-view) ==="
D:/conda_envs/lingbot-map/python.exe -u train_gs.py \
    --image_folder example/courthouse_small --first_k 50 \
    --epochs 300 --lr 5e-4 \
    --output_dir output_train_v6_50f_full \
    --render_chunk_size 16384

# --- Step 2: v6 same-view evaluation ---
echo ""
echo "=== Step 2: v6 same-view eval ==="
D:/conda_envs/lingbot-map/python.exe -u eval_gs.py \
    --image_folder example/courthouse_small --first_k 50 \
    --gs_head output_train_v6_50f_full/gs_head_best.pt \
    --output_dir output_eval_v6_50f_full

# --- Step 3: v6 novel-view evaluation ---
echo ""
echo "=== Step 3: v6 novel-view eval ==="
D:/conda_envs/lingbot-map/python.exe -u eval_novel_view.py \
    --image_folder example/courthouse_small --first_k 50 \
    --gs_head output_train_v6_50f_full/gs_head_best.pt \
    --output_dir output_eval_novel_v6_50f_full

# --- Step 4: v8 same-view evaluation ---
echo ""
echo "=== Step 4: v8 same-view eval ==="
D:/conda_envs/lingbot-map/python.exe -u eval_gs.py \
    --image_folder example/courthouse_small --first_k 50 \
    --gs_head output_train_v8_reproject/gs_head_best.pt \
    --output_dir output_eval_v8_50f

# --- Step 5: v8 novel-view evaluation ---
echo ""
echo "=== Step 5: v8 novel-view eval ==="
D:/conda_envs/lingbot-map/python.exe -u eval_novel_view.py \
    --image_folder example/courthouse_small --first_k 50 \
    --gs_head output_train_v8_reproject/gs_head_best.pt \
    --output_dir output_eval_novel_v8_50f

# --- Step 6: Generate figures ---
echo ""
echo "=== Step 6: Generate figures ==="
D:/conda_envs/lingbot-map/python.exe make_figures.py --output_dir figures

echo ""
echo "============================================"
echo "Full pipeline complete!"
echo "============================================"

# Results summary
echo ""
echo "--- Results Summary ---"
for d in output_eval_v6_50f_full output_eval_novel_v6_50f_full output_eval_v8_50f output_eval_novel_v8_50f; do
    if [ -f "${d}/metrics.json" ]; then
        echo ""
        echo "$d:"
        D:/conda_envs/lingbot-map/python.exe -c "
import json
with open('${d}/metrics.json') as f:
    m = json.load(f)
if 'novel_view' in m:
    nv = m['novel_view']
    tv = m['train_view']
    print(f'  Train-view: PSNR={tv[\"psnr\"]:.2f} SSIM={tv[\"ssim\"]:.4f} LPIPS={tv.get(\"lpips\",0):.4f}')
    print(f'  Novel-view: PSNR={nv[\"psnr\"]:.2f} SSIM={nv[\"ssim\"]:.4f} LPIPS={nv.get(\"lpips\",0):.4f}')
else:
    print(f'  PSNR={m[\"avg_psnr\"]:.2f} SSIM={m[\"avg_ssim\"]:.4f} LPIPS={m.get(\"avg_lpips\",0):.4f}')
    print(f'  GS={m[\"total_gaussians\"]:,} FPS={m[\"fps\"]:.1f}')
"
    fi
done
