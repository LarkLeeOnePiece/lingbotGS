# GCA-Splat Results Summary

## Main Results: Novel-View Synthesis (Courthouse, 50 frames)

### Nearest-K Frame Rendering (Key Finding)

The accumulated Gaussian map suffers from multi-frame interference (~22 dB).
**Nearest-K rendering** selects only the K closest training frames' Gaussians
for each viewpoint, eliminating overlap artifacts.

| Model | Render Mode | Train PSNR | Novel PSNR | Novel SSIM | Novel LPIPS |
|-------|-------------|-----------|-----------|-----------|------------|
| v8 (cross-view) | full-map | 22.53 | 22.53 | 0.804 | 0.558 |
| v8 (cross-view) | nearest-3 | 24.83 | 24.79 | 0.810 | 0.512 |
| v6 (baseline, ep44) | nearest-1 | 35.07 | 34.37 | 0.885 | 0.399 |
| v8 (cross-view) | nearest-1 | 36.47 | 35.10 | 0.901 | 0.348 |
| **v6 (300 epochs)** | **nearest-1** | **36.70** | **35.21** | **0.901** | **0.349** |

**Key findings:**
- Nearest-1 gives +12.7 dB over full-map on novel views (35.21 vs 22.53)
- Train→novel gap is only 1.49 dB with nearest-1 (vs ~0 with full-map)
- K=1 > K=3 > K=all — more frames means more interference
- Cross-view training (v8) provides NO benefit when v6 is fully trained (v6 35.21 > v8 35.10)
- Simple self-supervised rendering loss is sufficient with depth-adaptive scales + image-sampled colors

### Per-Frame Rendering Quality (Same-View)

| Version | PSNR (dB) | SSIM | LPIPS | Epochs | GS/frame |
|---------|-----------|------|-------|--------|----------|
| v6 (20f) | 36.02 | 0.913 | — | 300 | ~3000 |
| v8 (50f, cross-view) | 36.47 | 0.919 | 0.341 | 300 | ~3000 |
| **v6 (50f, no cross-view)** | **36.70** | **0.920** | **0.342** | **300** | **~3000** |

## Rendering Quality History (Courthouse, 20 frames, same-view)

| Version | Description | PSNR (dB) | SSIM | L1 | Epochs | GS/frame |
|---------|-------------|-----------|------|------|--------|----------|
| v2 | K=1, MLP colors | 29.21 | 0.822 | 0.022 | 300 | ~500 |
| v3 | K=4, MLP colors | 30.58 | 0.834 | 0.020 | 200 | ~4000 |
| v4 | K=4, tight scales | — | — | — | 40 (stopped) | ~4000 |
| v5 | K=4, image-sampled colors | — | 0.818 | 0.026 | 100 (stopped) | ~4000 |
| v6 (20f) | K=4, depth-adaptive + image colors | 36.02 | 0.913 | 0.009 | 300 | ~3000 |
| **v6 (50f)** | **Same, 50 frames** | **36.70** | **0.920** | **0.009** | **300** | **~3000** |

### Key Improvements (v3 → v6)
- **PSNR**: 30.58 → 36.02 (+5.4 dB)
- **SSIM**: 0.834 → 0.913 (+0.079)
- **L1**: 0.020 → 0.009 (2.2× reduction)

### Design Innovations
1. **Depth-adaptive Gaussian scales**: `base_scale = target_sigma × depth / focal_length` with learned correction. Each Gaussian projects to ~3.5 pixels on screen regardless of depth.
2. **Image-sampled colors**: RGB directly sampled from input image at sub-patch grid points via bilinear grid_sample. Small MLP residual (0.1 × tanh) for fine adjustment.
3. **Nearest-K rendering**: At inference, select only the K nearest training frames' Gaussians per viewpoint. K=1 eliminates multi-frame interference while maintaining 35+ dB quality.

## Cross-View Training (v8)

- **Mode**: Reprojection-based consistency loss (O(N) per frame, no full re-render)
- Projects frame i's Gaussian centers to frame j's image, samples GT color, computes opacity-weighted L1
- Loss: base_loss + 0.5 * cross_view_reprojection_loss
- Per-epoch time: ~36s (vs ~43s baseline — only 0.85× overhead)
- Training: 300 epochs on 50 frames

## Scalability Analysis (Courthouse Scene, all frames)

| Frames | Total GS | Anchor | Window | Memory | FPS |
|--------|----------|--------|--------|--------|-----|
| 3 | 9,324 | 6,216 | 3,108 | 0 | 1.4 |
| 5 | 15,540 | 6,216 | 9,324 | 0 | 7.5 |
| 10 | 31,080 | 6,216 | 24,864 | 0 | 6.8 |
| 20 | 58,256 | 6,216 | 49,728 | 2,312 | 6.9 |
| 50 | 93,419 | 6,216 | 49,726 | 37,477 | 5.8 |
| 100 | 135,317 | 6,216 | 49,724 | 79,377 | 6.1 |

## System Specs
- **Backbone**: LingBot-Map (1.16B params, frozen)
- **GS Head**: CompactGaussianHead (~1.3M params, trained)
- **Renderer**: PureTorchRenderer (pure PyTorch, no CUDA JIT)
- **GPU**: RTX 4090 (24GB)
- **Resolution**: 518 × 294 (courthouse)
- **Map size**: ~2.9 MB (20 frames) to ~6.7 MB (100 frames)

## Evaluation Protocol
- **Novel-view**: Even/odd frame split — even = train map, odd = test viewpoints
- **Metrics**: PSNR, SSIM, LPIPS (VGG), L1
- **Rendering**: Nearest-1 frame selection + frustum culling
- **Inference**: Streaming causal backbone (frozen) → per-frame Gaussian prediction

## Ablation Study (Courthouse Scene, 20 frames)

| Component | PSNR | SSIM | Description |
|-----------|------|------|-------------|
| Base (K=1, MLP colors) | 29.21 | 0.822 | 1 Gaussian per patch, MLP-predicted RGB |
| + K=4 Gaussians | 30.58 | 0.834 | 4 Gaussians per patch on sub-grid |
| + Image-sampled colors | ~30.0 | 0.818 | Grid-sample from input image (no depth-adaptive) |
| + Depth-adaptive scales | **36.02** | **0.913** | Scales = target_sigma * depth / focal |

## Training Configuration
- Epochs: 300
- LR: 5e-4 with cosine annealing
- Loss: 0.8×L1 + 0.2×SSIM + 0.001×opacity_reg [+ 0.5×cross_view_reproject for v8]
- Optimizer: AdamW (weight_decay=1e-5)
- Gradient clipping: max_norm=0.5
- Backbone: offloaded to CPU after feature precomputation
- Renderer: gradient checkpointing for memory efficiency

## Cross-Scene Generalization (Zero-Shot)

GS head trained on courthouse, evaluated on unseen scenes with nearest-1:

| Scene | Novel PSNR | Novel SSIM | Novel LPIPS |
|-------|-----------|-----------|------------|
| Courthouse (trained) | 35.10 | 0.901 | 0.348 |
| Loop (unseen) | 25.41 | 0.850 | 0.464 |
| Oxford (unseen) | 24.88 | 0.786 | 0.487 |
| University (unseen) | 23.64 | 0.650 | 0.582 |

Gap: ~10 dB between trained and unseen scenes → per-scene or multi-scene joint training needed.

## Multi-Scene Per-Scene Training (Completed)

Training v8 (cross-view reproject) per scene, 50 frames each, 300 epochs, nearest-1 eval:

| Scene | Train PSNR | Novel PSNR | Novel SSIM | Novel LPIPS |
|-------|-----------|-----------|-----------|------------|
| Courthouse (v6) | 36.70 | 35.21 | 0.901 | 0.349 |
| Courthouse (v8) | 36.47 | 35.10 | 0.901 | 0.348 |
| **Oxford** | **37.94** | **37.54** | **0.935** | **0.337** |
| **Loop** | **44.64** | **43.88** | **0.983** | **0.233** |
| **University** | **32.74** | **32.25** | **0.816** | **0.511** |

Per-scene training gives strong results across all scenes. Loop scene achieves 43.88 dB novel-view!
Train→novel gap is consistently small (0.49-1.49 dB), confirming nearest-1 generalizes well.

## Multi-Scene Joint Training (Planned)

`train_gs_multi.py` — trains a single unified GS head across all scenes simultaneously.
Leave-one-out experiments to validate generalization improvement.
See `PLAN_large_scale_training.md` for the full scaling plan.
