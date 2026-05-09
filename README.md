# LingbotGS (GCA-Splat)

**Feed-Forward Gaussian Map Construction from a Streaming 3D Foundation Model**

LingbotGS extends [LingBot-Map](https://github.com/Robbyant/lingbot-map) with a compact Gaussian splatting head that converts streaming 3D reconstructions into renderable Gaussian surfel maps — no per-scene optimization needed.

## Key Results

| Scene | Train PSNR | Novel PSNR | Novel SSIM | Method |
|-------|-----------|-----------|-----------|--------|
| Courthouse | 36.70 | 35.21 | 0.901 | Per-scene (v6, 50f) |
| Oxford | 37.94 | 37.54 | 0.935 | Per-scene (v8, 50f) |
| Loop | 44.64 | 43.88 | 0.981 | Per-scene (v8, 50f) |
| University | 32.74 | 32.25 | 0.816 | Per-scene (v8, 50f) |

All evaluated with **nearest-1 frame rendering** (even/odd train/test split).

## Architecture

```
Video Frame → Frozen LingBot-Map Backbone (1.16B params)
           → DINOv2 ViT-L features + Predicted Depth + Camera Pose
           → CompactGaussianHead (1.3M params, trainable)
           → K=4 Gaussian surfels per patch (~3K Gaussians/frame)
           → PureTorchRenderer (differentiable, no CUDA JIT)
```

### Key Design Choices

1. **Depth-adaptive scales**: `base_scale = target_sigma × depth / focal_length` with learned correction. Ensures consistent ~3.5px screen footprint regardless of depth (+5.4 dB improvement).

2. **Image-sampled colors**: RGB directly sampled from input image via bilinear grid_sample, plus a small learned residual (0.1 × tanh). Captures high-frequency texture that MLPs cannot.

3. **Nearest-frame rendering**: At inference, select only the single closest training frame's Gaussians per viewpoint. This eliminates multi-frame Gaussian interference (+12.6 dB over full accumulated map).

## Project Structure

```
lingbotGS/
├── lingbot_map/                 # Core package
│   ├── heads/
│   │   └── gaussian_head.py     # CompactGaussianHead (our addition)
│   ├── mapping/
│   │   ├── gaussian_map.py      # GaussianData + GaussianMapManager
│   │   ├── renderer.py          # PureTorchRenderer (differentiable splatting)
│   │   └── export.py            # PLY export
│   ├── models/
│   │   └── gct_stream_gs.py     # GCTStreamGS (extends GCTStream)
│   ├── aggregator/              # Upstream: patch embedding, GCA
│   ├── layers/                  # Upstream: attention, blocks, KV cache
│   ├── utils/                   # Upstream: pose encoding, geometry
│   └── vis/                     # Upstream: visualization
├── train_gs.py                  # Single-scene GS head training
├── train_gs_multi.py            # Multi-scene joint training
├── eval_novel_view.py           # Novel-view evaluation (nearest-K)
├── eval_gs.py                   # Same-view evaluation
├── demo_gs.py                   # Demo: inference + export + render
├── demo.py                      # Upstream interactive demo
├── paper/                       # Paper draft (LaTeX)
├── example/courthouse_small/    # Small test scene (50 frames)
├── RESULTS.md                   # Detailed experimental results
└── PLAN_large_scale_training.md # Scaling plan for large datasets
```

**Files we added** (GCA-Splat specific):
- `lingbot_map/heads/gaussian_head.py` — CompactGaussianHead
- `lingbot_map/mapping/` — GaussianData, GaussianMapManager, PureTorchRenderer, export
- `lingbot_map/models/gct_stream_gs.py` — GCTStreamGS
- `train_gs.py`, `train_gs_multi.py` — Training scripts
- `eval_novel_view.py`, `eval_gs.py` — Evaluation scripts
- `demo_gs.py` — Demo script
- `paper/` — Paper draft

## Setup

### Prerequisites

- Python >= 3.10
- PyTorch 2.8.0 + CUDA 12.8
- LingBot-Map checkpoint

### Installation

```bash
# Create environment
conda create -n lingbot-gs python=3.10
conda activate lingbot-gs

# Install PyTorch
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Install package
pip install -e .

# Additional dependencies for training/eval
pip install lpips matplotlib
```

### Download LingBot-Map Checkpoint

```bash
# Option 1: HuggingFace CLI
huggingface-cli download robbyant/lingbot-map-long --local-dir checkpoints/

# Option 2: Python
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('robbyant/lingbot-map', 'lingbot-map-long.pt', local_dir='checkpoints/')"
```

### Download Example Data

The `example/courthouse_small/` (50 frames) is included in the repo. For full scenes:

```bash
# TODO: Add download link for full scene data
# Full scenes: courthouse (286f), university (324f), oxford (320f), loop (237f)
```

## Usage

### Training (Single Scene)

```bash
python train_gs.py \
    --image_folder example/courthouse_small --first_k 50 \
    --model_path checkpoints/lingbot-map-long.pt \
    --epochs 300 --lr 5e-4 \
    --output_dir output_train

# Key flags:
#   --use_sdpa              Use SDPA attention (no FlashInfer needed)
#   --render_chunk_size N   Pixels per rendering chunk (16384 default)
#   --gaussians_per_patch K Sub-Gaussians per patch (4 default)
```

### Training (Multi-Scene Joint)

```bash
python train_gs_multi.py \
    --scenes courthouse:example/courthouse_small:50 \
             university:path/to/university:50 \
    --iterations 50000 --lr 5e-4 \
    --output_dir output_train_multi

# Leave-one-out validation:
python train_gs_multi.py \
    --scenes university:path/to/university:50 \
             oxford:path/to/oxford:50 \
    --holdout_scene courthouse:example/courthouse_small:50 \
    --iterations 30000
```

### Evaluation

```bash
# Novel-view evaluation with nearest-1 rendering
python eval_novel_view.py \
    --image_folder example/courthouse_small --first_k 50 \
    --gs_head output_train/gs_head_best.pt \
    --output_dir output_eval \
    --nearest_k 1

# Same-view evaluation
python eval_gs.py \
    --image_folder example/courthouse_small \
    --gs_head output_train/gs_head_best.pt \
    --output_dir output_eval_same
```

### Demo (Inference + Visualization)

```bash
python demo_gs.py \
    --image_folder example/courthouse_small --first_k 50 \
    --gs_head output_train/gs_head_best.pt \
    --output_dir output_demo
```

## IBEX (HPC) Setup

For running on IBEX or similar SLURM-based clusters:

```bash
# Load modules (adjust for your cluster)
module load cuda/12.8
module load python/3.10

# Create conda environment
conda create -n lingbot-gs python=3.10
conda activate lingbot-gs
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# On Linux with FlashInfer (recommended, faster):
pip install flashinfer-python
# Remove --use_sdpa from commands (FlashInfer is default)

# SLURM job example:
sbatch scripts/train.slurm
```

### Key Differences from Windows Setup

- **FlashInfer works on Linux** — no need for `--use_sdpa` fallback
- **gsplat may work on Linux** — can try `pip install gsplat` for faster rendering
- **Multi-GPU**: Can precompute features on one GPU, train on another
- **No WDDM issues**: GPU memory management is clean on Linux

## Training Notes

- **Backbone is frozen** — only the 1.3M GS head is trained
- Features are precomputed once and cached in CPU RAM (~20 MB/frame)
- Training: ~36s/epoch on 50 frames (RTX 4090)
- **Cross-view loss is NOT needed** — pure self-supervised rendering loss is sufficient
- **NaN prevention**: LR ≤ 5e-4, grad clip 0.5, variance clamping in SSIM
- **Memory**: Backbone offloaded to CPU after precomputation, training uses ~8 GB VRAM

## Citation

```bibtex
@article{gcasplat2026,
  title={GCA-Splat: Feed-Forward Gaussian Map Construction from a Streaming 3D Foundation Model},
  author={Anonymous},
  year={2026}
}

@article{lingbotmap2026,
  title={Geometric Context Transformer for Streaming 3D Reconstruction},
  author={Chen, Haolin and others},
  journal={arXiv preprint arXiv:2604.14141},
  year={2026}
}
```

## License

This project builds on [LingBot-Map](https://github.com/Robbyant/lingbot-map). See [LICENSE.txt](LICENSE.txt).
