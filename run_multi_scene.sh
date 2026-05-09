#!/bin/bash
# Multi-scene training + evaluation pipeline for GCA-Splat
# Trains v8 (cross-view reproject) GS head per scene, then evaluates with nearest-1
set -e
export PYTHONUNBUFFERED=1
cd D:/Projects/SLAM-GS/lingbotMap/lingbot-map

PYTHON=D:/conda_envs/lingbot-map/python.exe

echo "============================================"
echo "GCA-Splat Multi-Scene Pipeline"
echo "============================================"

# Scenes to train (courthouse already trained as v8_reproject)
declare -A SCENES
SCENES[university]="example/university:50"
SCENES[oxford]="example/oxford:50"
SCENES[loop]="example/loop:50"

for name in university oxford loop; do
    IFS=':' read -r folder first_k <<< "${SCENES[$name]}"
    train_dir="output_train_${name}_v8"
    eval_dir="output_eval_novel_${name}_v8_trained_nearest1"

    echo ""
    echo "=== Scene: ${name} (${first_k} frames) ==="

    # Train
    if [ -f "${train_dir}/gs_head_best.pt" ]; then
        epoch=$($PYTHON -c "import torch; c=torch.load('${train_dir}/gs_head_best.pt',map_location='cpu',weights_only=False); print(c.get('epoch',0))" 2>/dev/null || echo "0")
        if [ "$epoch" -ge 200 ]; then
            echo "  Already trained (epoch ${epoch}), skipping..."
        else
            echo "  Resuming from epoch ${epoch}..."
            $PYTHON -u train_gs.py \
                --image_folder "$folder" --first_k "$first_k" \
                --epochs 300 --lr 5e-4 \
                --lambda_cross_view 0.5 --cross_view_mode reproject \
                --output_dir "$train_dir" \
                --render_chunk_size 16384 \
                --resume "${train_dir}/gs_head_best.pt"
        fi
    else
        echo "  Training from scratch..."
        $PYTHON -u train_gs.py \
            --image_folder "$folder" --first_k "$first_k" \
            --epochs 300 --lr 5e-4 \
            --lambda_cross_view 0.5 --cross_view_mode reproject \
            --output_dir "$train_dir" \
            --render_chunk_size 16384
    fi

    # Evaluate with nearest-1
    if [ -f "${train_dir}/gs_head_best.pt" ]; then
        echo "  Evaluating ${name} (nearest-1)..."
        $PYTHON -u eval_novel_view.py \
            --image_folder "$folder" --first_k "$first_k" \
            --gs_head "${train_dir}/gs_head_best.pt" \
            --output_dir "$eval_dir" \
            --nearest_k 1
    fi
done

# Summary
echo ""
echo "============================================"
echo "Multi-Scene Results Summary"
echo "============================================"

# Include courthouse (already trained)
for name in courthouse university oxford loop; do
    if [ "$name" = "courthouse" ]; then
        eval_dir="output_eval_novel_v8_nearest1"
    else
        eval_dir="output_eval_novel_${name}_v8_trained_nearest1"
    fi
    if [ -f "${eval_dir}/metrics.json" ]; then
        echo ""
        echo "${name}:"
        $PYTHON -c "
import json
with open('${eval_dir}/metrics.json') as f:
    m = json.load(f)
tv = m.get('train_view', {})
nv = m.get('novel_view', {})
print(f'  Train: PSNR={tv.get(\"psnr\",0):.2f} SSIM={tv.get(\"ssim\",0):.4f} LPIPS={tv.get(\"lpips\",0):.4f}')
print(f'  Novel: PSNR={nv.get(\"psnr\",0):.2f} SSIM={nv.get(\"ssim\",0):.4f} LPIPS={nv.get(\"lpips\",0):.4f}')
"
    fi
done

echo ""
echo "All done!"
