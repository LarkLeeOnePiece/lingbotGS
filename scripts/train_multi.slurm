#!/bin/bash
#SBATCH --job-name=lingbotGS-multi
#SBATCH --output=logs/%j_multi.out
#SBATCH --error=logs/%j_multi.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=batch

# Adjust modules for your cluster
module load cuda/12.8

source activate lingbot-gs
mkdir -p logs

MODEL_PATH="checkpoints/lingbot-map-long.pt"

# Multi-scene joint training with leave-one-out
python -u train_gs_multi.py \
    --scenes \
        courthouse:example/courthouse_small:50 \
        university:example/university:50 \
        oxford:example/oxford:50 \
        loop:example/loop:50 \
    --model_path "$MODEL_PATH" \
    --iterations 50000 --lr 5e-4 --warmup_iters 1000 \
    --log_every 100 --save_every 5000 --eval_every 5000 \
    --output_dir output_train_multi_all4 \
    --render_chunk_size 16384
