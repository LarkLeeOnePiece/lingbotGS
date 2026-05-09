# LingBot-Map

## Project Overview

LingBot-Map is a **feed-forward 3D foundation model** for streaming 3D reconstruction from video, built on a **Geometric Context Transformer (GCT)** architecture. It processes video frames causally (one at a time, no future access) and predicts per-frame camera poses and dense depth maps in real-time (~20 FPS at 518x378 resolution).

**Paper:** "Geometric Context Transformer for Streaming 3D Reconstruction" (Chen et al., 2026, arXiv:2604.14141v2)

**Key innovation:** Geometric Context Attention (GCA) — a structured attention mechanism that maintains three complementary context types to balance long-range consistency with bounded memory/compute:

1. **Anchor Context** — First n frames (n<<N) with full tokens + learnable anchor token for coordinate grounding and scale establishment
2. **Local Pose-Reference Window** — Sliding window of k most recent frames with full image tokens for dense local geometry estimation
3. **Trajectory Memory** — Compact 6-token summaries (camera + 4 register + scale) of all past frames outside the window, for long-range drift correction

This design yields nearly constant memory/compute per frame regardless of sequence length.

## Architecture

```
Video Frame → DINOv2 ViT-L Backbone (patch=14, dim=1024)
           → Augment with special tokens (camera, 4x register, scale, anchor)
           → 24 alternating layers of:
               - Frame Attention (within-frame self-attention)
               - GCA (cross-frame attention with structured mask)
           → Camera Head (camera token → pose: translation + quaternion + FOV)
           → DPT Depth Head (image tokens → dense depth map)
```

### Token Structure Per Frame
- 1 camera token
- 4 register tokens
- 1 scale token (for anchor frames)
- M image tokens (patch embeddings from DINOv2, ~1369 for 518x378)
- 1 learnable anchor token (for anchor frames)

### KV Cache Design (FlashInfer)
Two-stream paged layout:
- **Patch stream**: 1 page/frame, recycled via sliding window eviction
- **Special stream**: 6 tokens/frame (camera + registers + scale), append-only, never evicted

## Repository Structure

```
lingbot-map/
├── demo.py                    # Main interactive demo (streaming + windowed modes)
├── gct_profile.py             # FPS benchmarking script
├── pyproject.toml             # Package config (pip install -e .)
├── lingbot_map/               # Core Python package
│   ├── models/
│   │   ├── gct_base.py        # Abstract base: prediction heads, forward pass
│   │   ├── gct_stream.py      # Streaming inference with KV cache
│   │   ├── gct_stream_window.py    # Windowed inference for very long sequences
│   │   └── gct_stream_window_v2.py # Windowed variant v2
│   ├── aggregator/
│   │   ├── base.py            # Patch embedding, special tokens, RoPE, block construction
│   │   └── stream.py          # Streaming causal aggregator with FlashInfer KV cache
│   ├── heads/
│   │   ├── camera_head.py     # CameraHead + CameraCausalHead (iterative refinement, 4 iters)
│   │   ├── dpt_head.py        # DPT dense prediction head (depth/points)
│   │   ├── head_act.py        # Activation functions
│   │   └── utils.py           # UV grid, position embeddings
│   ├── layers/
│   │   ├── attention.py       # Attention, CausalAttention, FlashInferAttention, SDPAAttention
│   │   ├── block.py           # Transformer blocks (standard, FlashInfer, SDPA, Camera variants)
│   │   ├── flashinfer_cache.py # FlashInferKVCacheManager (paged two-stream design)
│   │   ├── rope.py            # 2D RoPE (spatial) + 3D RoPE (temporal)
│   │   ├── vision_transformer.py # DINOv2 ViT backbone
│   │   ├── patch_embed.py     # Patch embedding layer
│   │   ├── mlp.py             # MLP
│   │   ├── swiglu_ffn.py      # SwiGLU FFN
│   │   ├── layer_scale.py     # LayerScale
│   │   └── drop_path.py       # DropPath
│   ├── utils/
│   │   ├── pose_enc.py        # Pose encoding/decoding (absT_quaR_FoV, 9-dim)
│   │   ├── geometry.py        # 3D geometry (depth unprojection, SE(3) ops)
│   │   ├── rotation.py        # Quaternion/rotation matrix conversions
│   │   └── load_fn.py         # Image/video loading and preprocessing
│   └── vis/
│       ├── viser_wrapper.py   # Viser 3D visualization server
│       ├── point_cloud_viewer.py # Interactive point cloud viewer
│       ├── sky_segmentation.py   # ONNX sky masking
│       ├── glb_export.py      # GLB 3D model export
│       └── utils.py           # Visualization utilities
├── demo_render/               # Offline rendering pipeline
│   ├── batch_demo.py          # Batch processing for long sequences → MP4/NPZ
│   ├── demo.py                # Legacy demo
│   └── interactive_viewer/    # Web-based GLB/NPZ viewer
└── example/                   # Example scene configs (courthouse, university, etc.)
```

## Key Concepts

### Inference Modes
- **Direct Output Mode** (default): Causal frame-by-frame, three-level GCA context accumulates without reset. Best for sequences ≤ ~3000 frames.
- **Visual Odometry (VO) Mode**: Partitions into overlapping windows, fuses via Sim(3) alignment at boundaries. For arbitrarily long sequences (10,000+ frames).

### Streaming Inference Pipeline (`GCTStream.inference_streaming`)
1. **Phase 1** (Scale frames): Process first n frames with bidirectional attention to establish coordinate system and scale
2. **Phase 2** (Causal streaming): Process remaining frames one-by-one with KV cache
   - Keyframe selection: Based on optical flow magnitude threshold
   - Non-keyframes: Predict but don't persist in KV cache

### Camera Pose Representation
9-dimensional encoding (`absT_quaR_FoV`):
- `[0:3]` — Absolute translation (camera-to-world)
- `[3:7]` — Quaternion rotation
- `[7:9]` — Field of view (2D)

### Loss Function (Training)
```
L = λ_depth * L_depth + λ_abs-pose * L_abs-pose + λ_rel-pose * L_rel-pose
```
- Depth loss: Scale-invariant with gradient and uncertainty terms
- Absolute pose loss: L2 on camera-to-world transformations
- Relative pose loss: Geodesic rotation + L1 translation over all frame pairs in local window

### Two-Stage Training
1. **Base model** (Stage 1): Offline with global attention on 2-24 views, 160K iterations, 29 datasets
2. **Streaming model** (Stage 2): Replace global attention with GCA, progressive view curriculum (24→320 views), 160K iterations, context parallelism (Ulysses)

## Development Setup

```bash
conda create -n lingbot-map python=3.10
conda activate lingbot-map
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install flashinfer-python              # For paged KV cache (recommended)
pip install -e ".[vis]"                    # For interactive demo (viser)
pip install -e ".[vis,render]"             # For offline rendering (open3d, kaolin)
```

**Requirements:** Python >= 3.10, PyTorch 2.8.0, CUDA 12.8

## Running Demos

```bash
# Interactive streaming demo (opens viser at http://localhost:8080)
python demo.py --model_path robbyant/lingbot-map-long --image_folder <path>

# With video input
python demo.py --model_path robbyant/lingbot-map-long --video_path <video> --video_fps 15

# Windowed mode for very long sequences
python demo.py --model_path robbyant/lingbot-map-long --image_folder <path> --mode windowed

# SDPA fallback (no FlashInfer needed)
python demo.py --model_path robbyant/lingbot-map-long --image_folder <path> --use_sdpa

# Offline batch rendering
python demo_render/batch_demo.py --model_path robbyant/lingbot-map-long --video_path <video>
```

### Key CLI Options
- `--keyframe_interval N` — Cache every N-th frame (reduces memory)
- `--camera_num_iterations K` — Camera head refinement iterations (default: 4)
- `--mask_sky` — Apply sky segmentation to filter outdoor sky points
- `--offload_to_cpu` — Offload KV cache to CPU for long sequences
- `--compile` — Enable torch.compile() optimization
- `--use_sdpa` — Use SDPA backend instead of FlashInfer

## Model Checkpoints

Available on HuggingFace (`robbyant/`):
- `lingbot-map-long` — Best for long sequences (recommended)
- `lingbot-map` — Balanced version
- `lingbot-map-stage1` — Stage 1 only (bidirectional, supports c2w)

## Evaluation Benchmarks

| Benchmark | Type | Key Metric |
|-----------|------|------------|
| Oxford Spires | Large-scale indoor/outdoor trajectories | ATE, AUC@15 |
| ETH3D | Indoor/outdoor with laser-scanned depth GT | F1 (threshold 0.25, voxel 0.039m) |
| 7-Scenes | Indoor RGB-D, textureless surfaces | ATE, F1 |
| Tanks and Temples | Outdoor multi-view, large structures | ATE, AUC@30 |
| NRGBD | Indoor RGB-D with fine geometric details | F1 |

## Code Conventions

- Coordinate convention: OpenCV (camera-to-world transformations)
- Image preprocessing: ImageNet normalization (RESNET_MEAN/STD), resize to max dim 518
- Patch size: 14 (DINOv2 ViT-L)
- Embedding dimension: 1024
- Number of transformer blocks: 24 (alternating Frame Attention + GCA)
- Default sliding window size: 64 frames
- Default anchor frames: 3
- Compact trajectory token count: 6 per frame (camera + 4 register + scale)
