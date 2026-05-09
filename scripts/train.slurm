#!/bin/bash
#SBATCH --job-name=lingbotGS-train
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=batch

# Adjust modules for your cluster
module load cuda/12.8

# Activate environment
source activate lingbot-gs

# Create log directory
mkdir -p logs

# Set paths
MODEL_PATH="checkpoints/lingbot-map-long.pt"
IMAGE_FOLDER="example/courthouse_small"

# Single-scene training
python -u train_gs.py \
    --image_folder "$IMAGE_FOLDER" --first_k 50 \
    --model_path "$MODEL_PATH" \
    --epochs 300 --lr 5e-4 \
    --output_dir output_train_courthouse \
    --render_chunk_size 16384

# Evaluate with nearest-1
python -u eval_novel_view.py \
    --image_folder "$IMAGE_FOLDER" --first_k 50 \
    --model_path "$MODEL_PATH" \
    --gs_head output_train_courthouse/gs_head_best.pt \
    --output_dir output_eval_courthouse \
    --nearest_k 1
