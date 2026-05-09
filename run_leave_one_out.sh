#!/bin/bash
# Leave-one-out multi-scene joint training experiment
# For each scene, train on the other 3, evaluate zero-shot on the held-out one
# This validates whether multi-scene joint training improves generalization
set -e
export PYTHONUNBUFFERED=1
cd D:/Projects/SLAM-GS/lingbotMap/lingbot-map

PYTHON=D:/conda_envs/lingbot-map/python.exe

# Scene definitions: name:folder:first_k
COURTHOUSE="courthouse:example/courthouse_small:50"
UNIVERSITY="university:example/university:50"
OXFORD="oxford:example/oxford:50"
LOOP="loop:example/loop:50"

echo "============================================"
echo "Leave-One-Out Multi-Scene Training"
echo "============================================"

# Experiment 1: Hold out courthouse (train on university+oxford+loop)
echo ""
echo "=== Holding out: courthouse ==="
$PYTHON -u train_gs_multi.py \
    --scenes "$UNIVERSITY" "$OXFORD" "$LOOP" \
    --holdout_scene "$COURTHOUSE" \
    --iterations 30000 --lr 5e-4 --warmup_iters 500 \
    --log_every 100 --save_every 5000 --eval_every 5000 \
    --output_dir output_multi_holdout_courthouse \
    --render_chunk_size 16384

# Experiment 2: Hold out university (train on courthouse+oxford+loop)
echo ""
echo "=== Holding out: university ==="
$PYTHON -u train_gs_multi.py \
    --scenes "$COURTHOUSE" "$OXFORD" "$LOOP" \
    --holdout_scene "$UNIVERSITY" \
    --iterations 30000 --lr 5e-4 --warmup_iters 500 \
    --log_every 100 --save_every 5000 --eval_every 5000 \
    --output_dir output_multi_holdout_university \
    --render_chunk_size 16384

# Experiment 3: All 4 scenes jointly
echo ""
echo "=== All 4 scenes joint training ==="
$PYTHON -u train_gs_multi.py \
    --scenes "$COURTHOUSE" "$UNIVERSITY" "$OXFORD" "$LOOP" \
    --iterations 40000 --lr 5e-4 --warmup_iters 500 \
    --log_every 100 --save_every 5000 --eval_every 5000 \
    --output_dir output_multi_all4 \
    --render_chunk_size 16384

# Summary
echo ""
echo "============================================"
echo "Leave-One-Out Summary"
echo "============================================"

for dir in output_multi_holdout_courthouse output_multi_holdout_university output_multi_all4; do
    if [ -f "${dir}/final_eval.json" ]; then
        echo ""
        echo "${dir}:"
        $PYTHON -c "
import json
with open('${dir}/final_eval.json') as f:
    r = json.load(f)
for name, m in r.items():
    print(f'  {name}: PSNR={m[\"psnr\"]:.2f} SSIM={m[\"ssim\"]:.4f}')
"
    fi
done

echo ""
echo "All done!"
