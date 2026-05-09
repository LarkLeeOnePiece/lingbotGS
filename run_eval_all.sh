#!/bin/bash
# Comprehensive evaluation of GCA-Splat models
# Run after training completes
set -e
export PYTHONUNBUFFERED=1
cd D:/Projects/SLAM-GS/lingbotMap/lingbot-map

echo "============================================"
echo "GCA-Splat Comprehensive Evaluation"
echo "============================================"

# --- v6 baseline (same-viewpoint only training, 20 frames) ---
if [ -f "output_train_v6/gs_head_best.pt" ]; then
    echo ""
    echo "=== v6 Baseline: Courthouse 20f, same-view eval ==="
    D:/conda_envs/lingbot-map/python.exe -u eval_gs.py \
        --image_folder example/courthouse_small \
        --first_k 20 \
        --gs_head output_train_v6/gs_head_best.pt \
        --output_dir output_eval_v6_20f_lpips

    echo ""
    echo "=== v6 Baseline: Courthouse 20f, novel-view eval ==="
    D:/conda_envs/lingbot-map/python.exe -u eval_novel_view.py \
        --image_folder example/courthouse_small \
        --first_k 20 \
        --gs_head output_train_v6/gs_head_best.pt \
        --output_dir output_eval_novel_v6_20f
fi

# --- v6 50-frame (same-viewpoint only training, 50 frames) ---
if [ -f "output_train_v6_50f/gs_head_best.pt" ]; then
    echo ""
    echo "=== v6 50f: Courthouse 50f, same-view eval ==="
    D:/conda_envs/lingbot-map/python.exe -u eval_gs.py \
        --image_folder example/courthouse_small \
        --first_k 50 \
        --gs_head output_train_v6_50f/gs_head_best.pt \
        --output_dir output_eval_v6_50f

    echo ""
    echo "=== v6 50f: Courthouse 50f, novel-view eval ==="
    D:/conda_envs/lingbot-map/python.exe -u eval_novel_view.py \
        --image_folder example/courthouse_small \
        --first_k 50 \
        --gs_head output_train_v6_50f/gs_head_best.pt \
        --output_dir output_eval_novel_v6_50f
fi

# --- v7 cross-view (cross-view training, 50 frames) ---
if [ -f "output_train_v7_crossview/gs_head_best.pt" ]; then
    echo ""
    echo "=== v7 Cross-View: Courthouse 50f, same-view eval ==="
    D:/conda_envs/lingbot-map/python.exe -u eval_gs.py \
        --image_folder example/courthouse_small \
        --first_k 50 \
        --gs_head output_train_v7_crossview/gs_head_best.pt \
        --output_dir output_eval_v7_50f

    echo ""
    echo "=== v7 Cross-View: Courthouse 50f, novel-view eval ==="
    D:/conda_envs/lingbot-map/python.exe -u eval_novel_view.py \
        --image_folder example/courthouse_small \
        --first_k 50 \
        --gs_head output_train_v7_crossview/gs_head_best.pt \
        --output_dir output_eval_novel_v7_50f
fi

# --- Multi-scene evaluation (v7 on other scenes) ---
for scene in university oxford loop; do
    if [ -f "output_train_v7_crossview/gs_head_best.pt" ] && [ -d "example/$scene" ]; then
        echo ""
        echo "=== v7 Cross-View: $scene, novel-view eval ==="
        D:/conda_envs/lingbot-map/python.exe -u eval_novel_view.py \
            --image_folder "example/$scene" \
            --first_k 50 \
            --gs_head output_train_v7_crossview/gs_head_best.pt \
            --output_dir "output_eval_novel_v7_${scene}"
    fi
done

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "============================================"

# Summarize results
echo ""
echo "--- Results Summary ---"
for d in output_eval_*/; do
    if [ -f "${d}metrics.json" ]; then
        echo ""
        echo "$d:"
        D:/conda_envs/lingbot-map/python.exe -c "
import json
with open('${d}metrics.json') as f:
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
