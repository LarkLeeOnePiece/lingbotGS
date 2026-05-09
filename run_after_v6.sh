#!/bin/bash
# Run after v6 50f 300-epoch training completes
# 1. Evaluate v6 with nearest-1
# 2. Start multi-scene training
set -e
export PYTHONUNBUFFERED=1
cd D:/Projects/SLAM-GS/lingbotMap/lingbot-map

PYTHON=D:/conda_envs/lingbot-map/python.exe

# Wait for v6 training to complete
echo "Waiting for v6 50f training to complete..."
while true; do
    if [ -f "output_train_v6_50f_full/train_log.json" ]; then
        n=$($PYTHON -c "import json; print(len(json.load(open('output_train_v6_50f_full/train_log.json'))))" 2>/dev/null || echo "0")
        if [ "$n" -ge 290 ]; then
            echo "  v6 training complete (${n} epochs logged)"
            break
        fi
        echo "  v6 at epoch ${n}/300..."
    fi
    sleep 120
done

# Evaluate v6 with nearest-1
echo ""
echo "=== Evaluating v6 50f (nearest-1) ==="
$PYTHON -u eval_novel_view.py \
    --image_folder example/courthouse_small --first_k 50 \
    --gs_head output_train_v6_50f_full/gs_head_best.pt \
    --output_dir output_eval_novel_v6_full_nearest1 \
    --nearest_k 1

# Now run multi-scene training
echo ""
echo "=== Starting multi-scene training ==="
bash run_multi_scene.sh
