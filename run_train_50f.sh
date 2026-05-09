#!/bin/bash
# Train v6 GS head on all 50 courthouse frames
cd D:/Projects/SLAM-GS/lingbotMap/lingbot-map
export PYTHONUNBUFFERED=1
conda run -n lingbot-map python -u train_gs.py \
    --image_folder example/courthouse_small \
    --first_k 50 \
    --epochs 300 \
    --lr 5e-4 \
    --output_dir output_train_v6_50f \
    --render_chunk_size 16384
