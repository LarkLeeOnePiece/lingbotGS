# Feed-Forward Gaussian Splatting Methods: Comprehensive Survey

> Compiled: 2026-05-09 | Purpose: Compare GCA-Splat GS Head design against existing feed-forward GS methods

---

## Our Method: GCA-Splat (Reference Point)

| Component | Details |
|-----------|---------|
| Backbone | GCT (LingBot-Map), 1.16B params, **frozen** |
| GS Head | `CompactGaussianHead`, **1.34M params** |
| Input | Streaming monocular video, causal (1 frame at a time) |
| Backbone output | Per-frame depth map + camera pose (absT_quaR_FoV) |
| Token input to GS head | Last-layer patch tokens [B, S, M, 2048] from GCT |
| Position | Depth-unprojected 3D world position + small learned offset (max 0.1 m × tanh) |
| Scale | **Depth-adaptive**: `base_scale = target_sigma × depth / focal_length`, learned ×exp(correction.clamp(−1.5,1.5)) |
| Color | **Bilinear image sampling** at sub-patch grid points + 0.1×tanh MLP residual; SH degree 0 |
| Opacity | sigmoid(MLP output), bias init −2.0 |
| Normal | L2-normalized MLP output |
| Representation | **2D Gaussian surfels** (surface-aligned, 2 scale params, not full 3D) |
| Gaussians/frame | K=4 per DINOv2 patch → **~3,000/frame** at 518×378 |
| MLP | 3-layer: 2048 → 512 → 512 → 48 (K×12 attrs) |
| Rendering | **Nearest-1 frame selection** per test viewpoint (frustum culling) |
| Training | Per-scene, 50 frames, 300 epochs, ~3 h on **1× RTX 4090** |
| Best results | 35.21 dB novel-view PSNR (per-scene); **~24 dB zero-shot** |
| Training datasets | 4 scenes (Courthouse, Oxford, Loop, University) |
| Test datasets | Same 4 scenes (even/odd frame split) |

---

## 1. pixelSplat (Charatan et al., CVPR 2024 Oral — Best Paper Runner-Up)

- **Paper**: "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction"
  [arXiv:2312.12337](https://arxiv.org/abs/2312.12337)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | ResNet-50 + DINOv1 ViT-B/8 (both DINO pretrained); outputs summed. Two rounds of **epipolar cross-attention** (cross-view, positionally-encoded depth). |
| Position | **Probabilistic**: predicts discrete Z-depth probability distribution per ray. 3 Gaussians **sampled** (reparameterization trick) from this distribution and unprojected to 3D. |
| Scale/Rotation | Covariance matrix predicted per pixel (full 3D covariance). |
| Color | **Spherical Harmonics (SH) coefficients** predicted per pixel. |
| Opacity | Set equal to sampled depth bucket probability (α = φ_z / 3). |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | **3 per pixel** (2 views → 2 × H × W × 3 total) |
| Special design | Probabilistic depth sampling with reparameterization handles scale ambiguity. Epipolar geometry enforces cross-view consistency. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | RealEstate10K (home walkthroughs, YouTube) + ACID (aerial landscapes). Resolution 256×256 |
| **Test/Eval data** | RealEstate10K, ACID — wide-baseline NVS. PSNR 26.09, SSIM 0.863, LPIPS 0.136 (RE10K) |
| **Hardware** | ~80 GB VRAM GPU (equiv. 1 large GPU) |
| **Training time** | ~4 days on **2× A40 GPUs**, 300K steps, batch size 7 |
| **Inference speed** | ~10 FPS (encoding 0.102 s + rendering 0.002 s) |

### Comparison to GCA-Splat
- **Input**: Exactly 2 wide-baseline **posed** images → GCA-Splat: streaming N frames, self-estimated poses
- **Representation**: Full 3DGS (probabilistic) → GCA-Splat: 2D surfels (deterministic, depth-adaptive)
- **Color**: SH coefficients → GCA-Splat: bilinear image sampling
- **Scale**: ~195K Gaussians/view → GCA-Splat: ~3K/frame (65× fewer)
- **Streaming**: None → GCA-Splat: causal KV cache, arbitrary-length sequences
- **Training scale**: RealEstate10K 67K scenes → GCA-Splat: 4 scenes (per-scene)

---

## 2. MVSplat (Chen et al., ECCV 2024 Oral)

- **Paper**: "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images"
  [arXiv:2403.14627](https://arxiv.org/abs/2403.14627)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | Shallow ResNet-like CNN (6 residual blocks) + multi-view Transformer (6 stacked self+cross-attention with Swin local window). **Cost volume** via plane sweeping for geometry. 2D U-Net with cross-view attention for final decoding. |
| Position | 1 Gaussian per pixel — depth predicted from cost volume, unprojected to 3D world. |
| Scale/Rotation | Covariance Σ = R(q) diag(exp ŝ)² R(q)ᵀ; quaternion q + log-scale ŝ predicted. |
| Color | **SH coefficients** from convolutional layers. |
| Opacity | Derived from matching confidence (max of softmax output), refined by 2 conv layers. |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | **1 per pixel** |
| Special design | Cost volume replaces epipolar transformer → lower compute. Deterministic depth (not probabilistic). |

### Training & Evaluation
| | |
|--|--|
| **Training data** | RealEstate10K (67,477 training scenes) + ACID (11,075 training scenes). Batch 14 (2 input + 4 target views each) |
| **Test/Eval data** | RealEstate10K, ACID, DTU. Metrics: PSNR, SSIM, LPIPS |
| **Hardware** | **1× NVIDIA A100 (80 GB)** |
| **Training time** | 300K iterations (450K with random init) |
| **Inference speed** | **22 FPS** (0.043 s encoding + 0.0015 s rendering at 256×256) |

### Comparison to GCA-Splat
- **Input**: 2-view posed → GCA-Splat: streaming monocular (no poses needed)
- **Backbone**: Lightweight CNN+Transformer → GCA-Splat: 1.16B GCT
- **Representation**: 3DGS, 1/pixel → GCA-Splat: 2D surfels, 4/patch
- **No temporal streaming** → GCA-Splat: causal KV cache with trajectory memory
- **Training scale**: 78K scenes → GCA-Splat: 4 scenes

---

## 3. latentSplat (Wewer et al., ECCV 2024)

- **Paper**: "latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction"
  [arXiv:2403.16292](https://arxiv.org/abs/2403.16292)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | Frozen DINOv1 ViT-B/8 + 2 epipolar transformer blocks (4-head attention, 32 points per epipolar line). Adapted from pixelSplat. |
| Position | From predicted depth distributions (sampled probabilistically, similar to pixelSplat). |
| Scale/Rotation | Normalized quaternions + scaled sigmoid, via linear projections. |
| Color | SH degree 4 (deterministic) + **variational SH degree 2** (stochastic latent appearance). VAE-GAN decoder renders latent Gaussians. |
| Opacity | From sampling probability. |
| Representation | Full **3D Gaussians** with **variational latent space** for appearance |
| Gaussians/frame | Multiple per ray |
| Special design | CVAE-like formulation for stochastic appearance. Handles uncertainty in appearance. Light-weight VAE-GAN decoder. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | CO3Dv2 (hydrants, teddybears — object-centric) + RealEstate10K. Progressive training curriculum. |
| **Test/Eval data** | CO3Dv2, RealEstate10K. Metrics: FID, KID, LPIPS, DISTS, PSNR, SSIM |
| **Hardware** | **2× NVIDIA A40** |
| **Training time** | ~4 days, 200K iterations |
| **Inference speed** | Encoding ~0.080 s + rendering ~0.003 s; GPU memory 3.161 GB |

### Comparison to GCA-Splat
- **Color**: Variational SH degree 4 → GCA-Splat: deterministic image sampling
- **Appearance**: Stochastic (generative) → GCA-Splat: deterministic (reconstructive)
- **No streaming** → GCA-Splat: causal KV cache

---

## 4. GPS-Gaussian (Zheng et al., CVPR 2024 Highlight)

- **Paper**: "GPS-Gaussian: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis"
  [arXiv:2312.02155](https://arxiv.org/abs/2312.02155)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | RAFT-Stereo–inspired iterative stereo depth estimator (correlation volume, T=3 iters). Separate image encoder (E_depth) + U-Net Gaussian parameter decoder (D_parm) with skip connections. |
| Position | Each foreground pixel unprojected to 3D using camera projection + predicted stereo depth. |
| Color | **Directly copied from source RGB** (justified by diffuse human appearance). |
| Rotation | 2-layer conv head, quaternion output. |
| Scale | Softplus-activated head. |
| Opacity | Sigmoid-activated head. |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | **1 per foreground pixel** |
| Special design | Domain-specific to humans. Stereo input (not monocular). |

### Training & Evaluation
| | |
|--|--|
| **Training data** | Twindom (1,700 scans), THuman2.0 (526 scans), 4 real-world characters. 16-camera rig. |
| **Test/Eval data** | THuman2.0, Twindom, real-world. PSNR 25.57, SSIM 0.898, LPIPS 0.112 |
| **Hardware** | **1× RTX 3090** |
| **Training time** | ~15 h (40K pre-training + 100K joint), batch size 2 |
| **Inference speed** | **25 FPS at 2K resolution** (novel view: 0.8 ms; full pipeline: 40 ms) |

### Comparison to GCA-Splat
- **Domain**: Human-only → GCA-Splat: general scenes
- **Color**: Direct pixel copy → GCA-Splat: bilinear sampling + residual (similar philosophy, more principled)
- **Input**: Stereo rig (calibrated) → GCA-Splat: monocular video
- **Training data**: 2.2K human scans → GCA-Splat: video sequences

---

## 5. Splatter Image (Szymanowicz et al., CVPR 2024)

- **Paper**: "Splatter Image: Ultra-Fast Single-View 3D Reconstruction"
  [arXiv:2312.13150](https://arxiv.org/abs/2312.13150)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | Modified **SongUNet** (U-Net from score-based diffusion models). Final layer replaced by 1×1 conv → 12–15 output channels. |
| Position | Depth d + 3D offset (Δx, Δy, Δz) predicted per pixel. |
| Scale/Rotation | Σ = R(q) diag(exp ŝ)² R(q)ᵀ; predicted quaternion q + log-scale ŝ. |
| Color | Lambertian (3 ch) or SH degree 1 (12 ch). |
| Opacity | Sigmoid-activated. |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | **1 per pixel** |
| Special design | Object-centric, single-image input. 2D UNet operators only. Extended to multi-view via cross-view attention. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | ShapeNet-SRN, CO3D, Objaverse-LVIS, Google Scanned Objects. Object-centric. |
| **Test/Eval data** | ShapeNet-SRN Cars (PSNR 24.00), multi-class ShapeNet (29.38), GSO (21.06) |
| **Hardware** | **1–2× NVIDIA A6000** |
| **Training time** | ~7 days single-view (800K + 100–200K iters); ~3.5 days Objaverse |
| **Inference speed** | **38 FPS** reconstruction + **588 FPS** rendering |

### Comparison to GCA-Splat
- **Domain**: Object-centric 360° → GCA-Splat: outdoor/indoor scenes
- **Input**: Single image → GCA-Splat: streaming video
- **Backbone**: Lightweight UNet → GCA-Splat: 1.16B GCT
- **Scale**: 1 Gaussian/pixel vs GCA-Splat's 4/patch (~60× fewer Gaussians)

---

## 6. Flash3D (Szymanowicz et al., 2024)

- **Paper**: "Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image"
  [arXiv:2406.04343](https://arxiv.org/abs/2406.04343)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | ResNet50 U-Net encoder-decoder. Extends a **monocular depth foundation model** (pre-trained). Input: RGB + predicted depth. |
| Position | μᵢ = (u_x·d_i/f, u_y·d_i/f, d_i) + Δᵢ, where d_i = d + Σδⱼ (iterated depth offsets per layer). |
| Scale/Rotation | Predicted by decoder; rotation as unit quaternions. |
| Color | **SH coefficients** (up to degree L) from decoder. |
| Opacity | σ predicted per Gaussian. |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | **2 layers × H × W** per image (layer 2 handles occlusions) |
| Special design | Multi-layer (2) Gaussian prediction for scene completion behind occlusions. Leverages foundation depth model. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | RealEstate10K (67,477 training scenes). Resolution 256×384 |
| **Test/Eval data** | RealEstate10K, NYU, KITTI. PSNR/SSIM/LPIPS at 5-, 10-, ±30-frame offsets |
| **Hardware** | **1× NVIDIA A6000** |
| **Training time** | **16 h** (with pre-extracted depth, 40K iters, batch 16) |
| **Inference speed** | ~10 FPS |

### Comparison to GCA-Splat
- **Input**: Single image → GCA-Splat: streaming N frames
- **Multi-layer depth**: 2 layers per pixel for occlusion → GCA-Splat: K=4 per patch for density
- **Both use depth to anchor positions**: Flash3D from pretrained depth net; GCA-Splat from GCT depth head
- **Both depth-adaptive**: Flash3D via depth offsets; GCA-Splat via explicit depth×focal formula
- **No cross-frame accumulation** → GCA-Splat: causal KV cache

---

## 7. GS-LRM (Zhang et al., ECCV 2024)

- **Paper**: "GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting"
  [arXiv:2404.19702](https://arxiv.org/abs/2404.19702)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | Large transformer (LRM-style) — patchifies multi-view images → tokens → N transformer blocks → decoding heads per pixel. |
| Position | Depth back-projected per pixel to 3D world (given known camera parameters). |
| Scale/Rotation/Opacity/Color | Decoded from transformer output tokens via dedicated heads. SH for color. |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | **1 per pixel** |
| Special design | Same architecture scales to objects (Objaverse) and scenes (RE10K) by changing training data. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | Objaverse (objects) OR RealEstate10K (scenes). Two separate models. |
| **Test/Eval data** | GSO, OmniObject3D (objects); RE10K (scenes). Scene PSNR ~+2.2 dB over prior art. |
| **Hardware** | A100 (inference); training hardware not disclosed |
| **Training time** | Not reported |
| **Inference speed** | **0.23 s** for 2–4 posed views on single A100 |

### Comparison to GCA-Splat
- **Input**: 2–4 posed images → GCA-Splat: streaming monocular video
- **Large transformer backbone** (similar in spirit to GCT) but not causal streaming
- **No temporal accumulation** → GCA-Splat: causal KV cache

---

## 8. FreeSplat (Wang et al., NeurIPS 2024)

- **Paper**: "FreeSplat: Generalizable 3D Gaussian Splatting Towards Free-View Synthesis of Indoor Scenes"
  [arXiv:2405.17958](https://arxiv.org/abs/2405.17958)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | CNN (EfficientNet/ResNet) for low-cost features. Adaptive cost volumes for multi-view aggregation. **Pixel-wise Triplet Fusion (PTF)** module — GRU-based aggregation of local/global Gaussian triplets. |
| Position | Depth-unprojected from predicted depth maps. |
| Scale/Rotation/Opacity/Color | MLP decoder from global Gaussian features. |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | 1/pixel initially → PTF **prunes ~55% redundant Gaussians** via pixel alignment |
| Special design | PTF reduces multi-view redundancy. Free-view training (2–8 views). Very **data-efficient**: only 100 training scenes. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | **ScanNet (100 training scenes, 8 test scenes)**. Batch 1, 384×512 |
| **Test/Eval data** | ScanNet (3-view: PSNR 27.34, SSIM 0.826, LPIPS 0.226; 10-view extrapolation: 24.64), Replica |
| **Hardware** | **1× NVIDIA RTX A6000 (42.2 GB)** |
| **Training time** | Not reported |
| **Inference speed** | Up to 72 FPS (with PTF) |

### Comparison to GCA-Splat
- Trained on only 100 scenes (very data-efficient CNN) vs GCA-Splat's per-scene approach
- PTF redundancy removal conceptually similar to GCA-Splat's compact Gaussian design
- Multi-view posed input, indoor only → GCA-Splat: streaming monocular, general

---

## 9. DepthSplat (Xu et al., CVPR 2025)

- **Paper**: "DepthSplat: Connecting Gaussian Splatting and Depth"
  [arXiv:2410.13862](https://arxiv.org/abs/2410.13862)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | Dual-branch: **Depth Anything V2** (ViT-S/B/L) for monocular depth + ResNet cost volume for multi-view matching. Three model sizes (Small/Base/Large). |
| Position | Per-pixel depth back-projected to 3D world. |
| Scale/Rotation/Opacity/Color | Predicted by lightweight DPT head. |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | **1 per pixel per input view** |
| Special design | Explicit bridge between monocular depth priors and multi-view matching. GS training acts as unsupervised pretraining for the depth model. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | TartanAir + ScanNet + **RealEstate10K (67K)** + **DL3DV (9,896 scenes)**. Batch 32. |
| **Test/Eval data** | RE10K (PSNR 27.47, SSIM 0.889, LPIPS 0.114 at 2-view 256×256), ScanNet, DL3DV (PSNR 23.12, SSIM 0.780 at 4-view) |
| **Hardware** | **4× GH200 (96 GB each)**. Also validated on 4× RTX 4090 with ≤0.1 dB PSNR difference. |
| **Training time** | **1 day** (small) to **2 days** (large) on 4× GH200; 100K–150K iterations |
| **Inference speed** | 0.05–0.08 s at 256×256; 0.6 s for 12 views at 512×960 |

### Comparison to GCA-Splat
- Multi-view posed input → GCA-Splat: streaming monocular
- Both leverage strong depth priors (Depth Anything V2 vs GCT depth head)
- 4 GH200 GPUs vs GCA-Splat 1× RTX 4090
- No causal streaming → GCA-Splat: unlimited-length KV cache

---

## 10. NoPoSplat (Chen et al., ICLR 2025 Oral)

- **Paper**: "No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images"
  [OpenReview](https://openreview.net/forum?id=P4o9akekdf)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | **ViT-Large encoder** (init from MASt3R) + ViT-Base decoder. Camera intrinsics embedded as a token and concatenated with image tokens for scale disambiguation. |
| Position | MASt3R-style 3D point cloud prediction in canonical (first-view) coordinate space. |
| Scale/Rotation/Opacity/Color | Predicted alongside positions from decoder tokens (details not fully disclosed). |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | **1 per pixel per view** |
| Special design | **No camera poses required at test time**. Intrinsic embedding resolves scale ambiguity. Outperforms pose-required methods at wide baselines. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | RealEstate10K (primary) + ACID + DL3DV. Photometric loss only. |
| **Test/Eval data** | RE10K, ACID, DTU, ScanNet++, ScanNet-1500, Tanks & Temples |
| **Hardware** | **8× GPUs (≥80 GB, e.g., 8× A100)** |
| **Training time** | ~6 hours (from repository README) |
| **Inference speed** | Real-time (not benchmarked in ms) |

### Comparison to GCA-Splat
- Both pose-free at inference (NoPoSplat: no pose; GCA-Splat: pose from GCT backbone)
- 2–3 unordered images → GCA-Splat: streaming video sequence
- 8× A100 → GCA-Splat: 1× RTX 4090
- No causal streaming → GCA-Splat: KV cache for arbitrary-length sequences
- Both trained purely with photometric/rendering loss

---

## 11. Splatt3R (Smart et al., 2024)

- **Paper**: "Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs"
  [arXiv:2408.13912](https://arxiv.org/abs/2408.13912)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | **Frozen pre-trained MASt3R** (ViT-based encoder + cross-attention transformer decoder). Added parallel Gaussian head alongside MASt3R's existing point+descriptor heads. |
| Position | μ = x + Δ (MASt3R 3D point + learned offset Δ). |
| Scale/Rotation | Quaternion q ∈ ℝ⁴ + scale s ∈ ℝ³ from head. |
| Color | **Constant RGB per Gaussian** (SH tested but underperformed). |
| Opacity | Sigmoid-activated scalar. |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | **1 per pixel per input image** |
| Special design | Two-stage training: geometry loss first, then NVS loss. Custom loss masking for extrapolated viewpoints. Frozen backbone (like GCA-Splat). |

### Training & Evaluation
| | |
|--|--|
| **Training data** | **ScanNet++ (450+ indoor scenes)**. 2000 epochs ≈ 500K iterations, 512×512. |
| **Test/Eval data** | ScanNet++ (Very Wide baseline: PSNR 19.18, LPIPS 0.209) |
| **Hardware** | Not reported |
| **Training time** | Not reported (500K iterations) |
| **Inference speed** | **4 FPS** at 512×512 on RTX 2080 Ti |

### Comparison to GCA-Splat
- **Most similar design philosophy**: frozen backbone (MASt3R) + lightweight Gaussian head (like GCA-Splat's frozen GCT + CompactGaussianHead)
- Key difference: 2-view unordered images vs GCA-Splat's streaming N frames
- Constant RGB color → GCA-Splat: bilinear image sampling (related concept, more general)
- Indoor only (ScanNet++) → GCA-Splat: general
- 4 FPS → GCA-Splat: ~20 FPS streaming

---

## 12. AnySplat (Jiang et al., SIGGRAPH Asia 2025 / ACM ToG)

- **Paper**: "AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views"
  [arXiv:2505.23716](https://arxiv.org/abs/2505.23716)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | L=24 **Alternating-Attention Transformer** (frame attn + global attn), init from **VGGT** (DINOv2-based, patch=14, dim=1024). **886M params total**. |
| Position | Depth back-projection from predicted camera poses. |
| Scale/Rotation/Opacity/Color | CNN regression heads (DPT features + appearance features). **Differentiable voxelization** merges pixel-wise → voxel-wise Gaussians. |
| Representation | Full **3D Gaussians** (voxelized for efficiency) |
| Gaussians/frame | Per-pixel initially → consolidated via voxelization |
| Special design | Self-supervised knowledge distillation from VGGT. No SfM/MVS supervision. Simultaneous Gaussian + camera pose prediction. Handles 3–64+ views. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | **9 datasets, 23.5M+ frames**: ARKitScenes (9.2M, 4,406 scenes), CO3D-v2 (5.5M, 27,520 scenes), DL3DV (3.4M, 9,894 scenes), WildRGBD (3.9M, 11,050 scenes), Objaverse (8M, 199K scenes), BlendedMVS, ScanNet++, Hypersim, Unreal4K |
| **Test/Eval data** | Deep-Blending, VRNeRF (PSNR/SSIM/LPIPS); CO3D-v2 (pose AUC at 5°/10°/20°/30°) |
| **Hardware** | **16× NVIDIA A800** |
| **Training time** | ~1 day (15K iterations) |
| **Inference speed** | 0.17 s (3 views) → 0.77 s (16 views) → 4.1 s (64 views) |

### Comparison to GCA-Splat
- **Massive training scale** (23.5M frames, 9 datasets) vs GCA-Splat's per-scene
- 886M param transformer backbone vs GCA-Splat's frozen 1.16B GCT (similar in scale)
- Simultaneous pose + Gaussian (one model) vs GCA-Splat's decoupled backbone + head (1.34M)
- Multi-view (not causal) → GCA-Splat: online streaming causal
- Voxel Gaussian merging vs GCA-Splat's nearest-frame selection
- 16× A800 → GCA-Splat: 1× RTX 4090

---

## 13. StreamGS (Li et al., ICCV 2025)

- **Paper**: "StreamGS: Online Generalizable Gaussian Splatting Reconstruction for Unposed Image Streams"
  [arXiv:2503.06235](https://arxiv.org/abs/2503.06235)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | Frozen pretrained **DUSt3R** as coarse 3D predictor. Additional lightweight: 2D image feature extractor (φ₂D), matching head (φ_match), MergeNet (2 conv layers). Content-adaptive refinement on DUSt3R. |
| Position | DUSt3R-predicted 3D points (refined by content-adaptive matching). |
| Scale/Rotation/Opacity/Color | Lightweight decoder (φ_GS) from concatenated multi-modal features (2D image + 3D coords + 3D matching feats). SH for color. |
| Representation | Full **3D Gaussians** |
| Gaussians/frame | H×W per frame initially → **Adaptive Density Control prunes ~37–40%** across frames |
| Special design | **First generalizable pose-free 3DGS for online image streams**. Content-adaptive descriptors for cross-frame consistency. Density-controlled pruning. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | **RealEstate10K (training split)**. 30K steps, batch 14, 224×224 |
| **Test/Eval data** | RE10K, DL3DV, MVImgNet, ScanNet, ACID. PSNR ~23.1 dB |
| **Hardware** | **1× NVIDIA Tesla A100 (80 GB)** |
| **Training time** | Not reported explicitly |
| **Inference speed** | **~9 FPS** (150× faster than CF-3DGS baseline) |

### Comparison to GCA-Splat
**Most direct competitor — both are online streaming feed-forward GS from monocular video.**

| Aspect | StreamGS | GCA-Splat |
|--------|----------|-----------|
| Backbone | Frozen DUSt3R (pairwise) | Frozen GCT (causal, KV cache) |
| Streaming mechanism | Pairwise + merge buffer | Causal KV cache (bounded compute) |
| Memory per frame | Growing unless pruned | Constant (paged KV cache) |
| Gaussian merging | Density Control (ADC) | Nearest-frame selection |
| Color | SH | Bilinear image sampling |
| PSNR (zero-shot) | ~23.1 dB | ~24 dB (cross-scene) / 35 dB (per-scene) |
| Training data | RE10K | 4 scenes (per-scene) |
| GPU (train) | A100 | RTX 4090 |

**Key GCA-Splat advantages over StreamGS**:
1. GCT's causal KV cache → strictly bounded per-frame compute regardless of sequence length
2. Depth-adaptive scale formula → more principled than pure regression
3. Image-sampled color → simpler and potentially more accurate for high-PSNR scenarios
4. 2D surfels → surface-aligned, fewer artifacts

---

## 14. StreamSplat (Ye et al., ICLR 2026)

- **Paper**: "StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams"
  [arXiv:2506.08862](https://arxiv.org/abs/2506.08862)

### GS Head Design
| Attribute | Details |
|-----------|---------|
| Backbone | Two-stage: frozen **static encoder** (transformer) + **dynamic decoder** (self-attn + cross-attn with DINOv2 + FFN). |
| Position | **Probabilistic**: predicts truncated normal distribution for 3D offset (robust to noise). |
| Scale/Rotation/Opacity/Color | Linear heads from decoder tokens + SH color coefficients. |
| Representation | Full **3D Gaussians** + **bidirectional deformation field** for dynamics |
| Gaussians/frame | Per-frame tokens |
| Special design | **Bidirectional deformation field** for smooth dynamic Gaussian transitions. Adaptive opacity-based Gaussian fusion propagates persistent Gaussians. **Handles dynamic scenes** (not just static). |

### Training & Evaluation
| | |
|--|--|
| **Training data** | CO3Dv2 + RealEstate10K (static); DAVIS + YouTube-VOS (dynamic) |
| **Test/Eval data** | RE10K, CO3Dv2, DAVIS, YouTube-VOS. Metrics: PSNR, SSIM, LPIPS |
| **Hardware** | **8× NVIDIA A100** |
| **Training time** | ~3 days (FlashAttention-2 + gradient checkpointing + BF16) |
| **Inference speed** | ~Near real-time (~1.48 s per DAVIS sequence) |

### Comparison to GCA-Splat
- **Dynamic scenes** (deformation field) → GCA-Splat: static scenes only
- DINOv2 cross-attention decoder → GCA-Splat: GCT causal KV cache
- Probabilistic position → GCA-Splat: deterministic depth-unproject
- 8× A100 → GCA-Splat: 1× RTX 4090
- Both handle uncalibrated streaming video

---

## 15. SLAM3R (Liu et al., CVPR 2025 Highlight)

- **Paper**: "SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos"
  [arXiv:2412.09401](https://arxiv.org/abs/2412.09401)

> Note: SLAM3R outputs **point clouds, not Gaussian splats** — included for context as a direct streaming reconstruction competitor.

### Design
| Attribute | Details |
|-----------|---------|
| Backbone | Multi-branch ViT with shared encoder, separate decoders for keyframes/supporting frames. Two networks: Image-to-Points (I2P) + Local-to-World (L2W). |
| Output | Dense 3D **pointmaps** only — not renderable Gaussians |
| Special design | No explicit camera estimation — geometry inferred end-to-end. Causal/streaming for real-time. |

### Training & Evaluation
| | |
|--|--|
| **Training data** | ~880K clips: ScanNet++ (iPhone+DSLR), Aria Synthetic Environments (450 scenes), CO3D-v2 (41 categories) |
| **Test/Eval data** | 7-Scenes, Replica, Tanks and Temples, BlendedMVS, LLFF, ETH3D, DTU |
| **Hardware** | **8× NVIDIA RTX 4090D (24 GB each)** |
| **Training time** | I2P ~6 h + L2W ~15 h = **~21 h total** |
| **Inference speed** | **20–25 FPS** on single RTX 4090D |

### Comparison to GCA-Splat
- Both: monocular streaming video, ~20 FPS, causal
- SLAM3R: no rendering capability → GCA-Splat: adds Gaussian rendering on top
- SLAM3R: 880K clips training → GCA-Splat: 4 scenes (GS head only)

---

## 16. CAT3D (Gao et al., NeurIPS 2024)

- **Paper**: "CAT3D: Create Anything in 3D with Multi-View Diffusion Models"
  [arXiv:2405.10314](https://arxiv.org/abs/2405.10314)

> Note: Hybrid diffusion + per-scene optimization. Not a true feed-forward GS method — included for completeness.

### Design
- **Pipeline**: Input views → multi-view diffusion (generates novel views) → per-scene 3DGS optimization → real-time rendering
- Not zero-shot; requires per-scene optimization (diffusion inference + 3DGS fine-tuning)
- Handles 1/few/many input images

### Training & Evaluation
| | |
|--|--|
| **Training data** | Large-scale diffusion training (not disclosed) |
| **Test/Eval data** | NeRF benchmarks, CO3D, RE10K |
| **Hardware** | Not disclosed |
| **Scene creation time** | ~1 minute per scene (inference time) |

### Comparison to GCA-Splat
- Hybrid (generative diffusion + optimization) vs GCA-Splat's pure feed-forward
- Per-scene fine-tuning needed → GCA-Splat: zero-shot inference from backbone features
- Generative (can hallucinate) → GCA-Splat: geometric grounding from real depth

---

## Summary Comparison Table

| Method | Venue | Input | Backbone | GS Repr. | Gaussians/frame | Color | Training Data | GPU (Train) | Train Time | FPS |
|--------|-------|-------|----------|-----------|----------------|-------|---------------|-------------|------------|-----|
| **GCA-Splat** | — | Streaming mono | GCT 1.16B (frozen) | **2D surfels** | **~3K** (K=4/patch) | **Bilinear image + residual** | 4 scenes (per-scene) | **1× RTX 4090** | **~3 h/scene** | ~20 |
| pixelSplat | CVPR'24 Oral | 2 images (posed) | ResNet50+DINOv1 ViT-B | 3D Gaussian | ~600K (3/px, 2 views) | SH | RE10K+ACID | ~80GB GPU | ~4 days | ~10 |
| MVSplat | ECCV'24 Oral | 2–N (posed) | CNN+MultiView Tx | 3D Gaussian | ~65K (1/px) | SH | RE10K+ACID (78K) | 1× A100 | 300K iters | 22 |
| latentSplat | ECCV'24 | 2 (posed) | DINOv1 ViT-B+Epipolar | 3D Gaussian (VAE) | Multi/ray | SH+VAE-GAN | CO3D+RE10K | 2× A40 | ~4 days | ~12 |
| GPS-Gaussian | CVPR'24 HL | Stereo (humans) | RAFT-Stereo+U-Net | 3D Gaussian | 1/fg px | Direct RGB copy | THuman+Twindom (2.2K) | 1× RTX 3090 | ~15 h | 25 |
| Splatter Image | CVPR'24 | 1 image (object) | SongUNet (diffusion UNet) | 3D Gaussian | 1/px | SH degree 1 | ShapeNet+CO3D+Objaverse | 1–2× A6000 | 3.5–7 days | 38 |
| Flash3D | 2024 | 1 image | ResNet50+depth foundation | 3D Gaussian | 2 layers/px | SH | RE10K (67K) | 1× A6000 | 16 h | ~10 |
| GS-LRM | ECCV'24 | 2–4 (posed) | Large Transformer (LRM) | 3D Gaussian | 1/px | SH | Objaverse+RE10K | A100 | N/R | 4 FPS |
| FreeSplat | NeurIPS'24 | 2–8 (posed) | CNN+cost vol+PTF | 3D Gaussian | 1/px (−55%) | MLP | ScanNet (100 scenes) | 1× A6000 | N/R | 72 |
| DepthSplat | CVPR'25 | 2–12 (posed) | DepthAnything V2 ViT-L | 3D Gaussian | 1/px | DPT head | RE10K+ScanNet+DL3DV+TartanAir | 4× GH200 | 1–2 days | ~17 |
| NoPoSplat | ICLR'25 Oral | 2–3 (unposed) | MASt3R ViT-L | 3D Gaussian | 1/px | N/R | RE10K+ACID+DL3DV | 8× A100 | ~6 h | RT |
| Splatt3R | 2024 | 2 (unposed) | Frozen MASt3R ViT | 3D Gaussian | 1/px | **Constant RGB** | ScanNet++ (450 scenes) | N/R | 500K iters | 4 |
| AnySplat | SIGGRAPH Asia'25 | 3–64+ (unposed) | VGGT Alternating-Attn Tx 886M | 3D Gaussian (voxelized) | 1/px→voxel | CNN regression | **9 datasets, 23.5M frames** | 16× A800 | ~1 day | 0.2–6s |
| StreamGS | ICCV'25 | Online stream | Frozen DUSt3R+decoder | 3D Gaussian | 1/px (−38%) | SH | RE10K | 1× A100 | 30K iters | ~9 |
| StreamSplat | ICLR'26 | Online video | Static Tx+DINOv2 decoder | 3D Gaussian+deform | Per-frame tokens | SH | CO3D+RE10K+DAVIS+YT-VOS | 8× A100 | ~3 days | ~RT |
| SLAM3R | CVPR'25 HL | Online mono | Multi-branch ViT | **Points (no GS)** | N/A | N/A | 880K clips | 8× RTX 4090D | ~21 h | 20–25 |
| CAT3D | NeurIPS'24 | 1–N | Diffusion model | 3DGS (per-scene) | N/A (opt) | Per-scene | Large (undisclosed) | Undisclosed | N/R | ~1 min/scene |

*N/R = not reported. RT = real-time (not benchmarked in ms). HL = Highlight.*

---

## Key Differentiators of GCA-Splat

### 1. Causal Streaming with Bounded-Cost KV Cache
GCA-Splat is the **only method with a structured three-level causal context** (Anchor + Sliding Window + Trajectory Memory) for unlimited-length sequences, where per-frame compute is **strictly constant** regardless of sequence length. StreamGS uses pairwise DUSt3R matching without a paged cache (cost grows unless pruned). StreamSplat uses a separate dynamic decoder without the paged KV cache design.

### 2. Extreme Separation of Backbone and GS Head
GCA-Splat's GS head has **only 1.34M parameters** (vs. hundreds of millions in competing methods that fuse backbone and Gaussian prediction). The frozen 1.16B GCT backbone handles all geometry estimation; the GS head learns appearance only. This enables:
- Training on a single RTX 4090 (vs. 8× A100 or 16× A800 for competing methods)
- Fast iteration: ~3 hours per scene
- Modular design: backbone and head can be updated independently

The closest analog is Splatt3R (frozen MASt3R + lightweight Gaussian head), but Splatt3R is limited to 2-view input and achieves only 4 FPS.

### 3. Depth-Adaptive Scale Formula
Most methods directly regress scale from MLP outputs (unbounded, dataset-dependent). GCA-Splat uses:
```
base_scale = target_sigma × depth / focal_length
scale = base_scale × exp(MLP_correction.clamp(−1.5, 1.5))
```
This physically anchors the Gaussian's apparent projected size (~3.5 pixels) regardless of depth, focal length, or scene scale — the MLP only learns a correction factor. This is more generalizable across scenes with different depth ranges.

### 4. Bilinear Image-Sampled Color (Not SH)
All competing methods (except GPS-Gaussian and Splatt3R) use SH coefficients for color. GCA-Splat directly bilinear-samples the input image at sub-patch positions, ensuring per-pixel accurate color at the training viewpoint. This directly drives high same-view PSNR (36.7 dB) without any view-dependent overhead. The SH degree-0 limitation (no specular highlights) is the trade-off.

### 5. 2D Gaussian Surfels (Surface-Aligned)
Every other method uses full 3D Gaussians (6-DOF covariance = 4 rotation + 3 scale params). GCA-Splat uses **2D surfels**: surface-aligned, 2 tangent-plane scale parameters, normals predicted explicitly. Benefits:
- Fewer parameters per Gaussian (12 total vs. ~14–18 for full 3DGS)
- Better models smooth surfaces (avoids the "needle" artifacts of elongated 3D Gaussians)
- Consistent with the underlying depth-map geometry

### 6. K=4 Gaussians Per DINOv2 Patch vs. 1 Per Pixel
| Method | Gaussians/frame | At 518×378 |
|--------|----------------|-----------|
| GCA-Splat | 4 per 14×14 patch | **~3,000** |
| Most others | 1 per pixel | **~195,000** |

GCA-Splat uses ~65× fewer Gaussians per frame, making streaming accumulation tractable. The spatial coverage per Gaussian is larger but the depth-adaptive scale compensates. The nearest-1 rendering strategy means only ~3,000 Gaussians need to be rasterized per viewpoint.

### 7. Nearest-Frame Rendering as a Design Choice
GCA-Splat deliberately uses only Gaussians from the nearest training frame for each test viewpoint — relying on temporal density of the streaming video rather than long-range Gaussian fusion. This avoids multi-frame interference (which costs ~12 dB: full-map 22 dB vs. nearest-1 35 dB) but requires sufficient frame density. This is a unique design choice not used by any other method.

---

## Limitations vs. Prior Work

| Limitation | Details | Relevant Competitors |
|------------|---------|---------------------|
| **Per-scene training** | Current: 300 epochs ~3 h/scene; zero-shot 24 dB (11 dB gap) | All zero-shot methods (MVSplat: 27 dB, DepthSplat: 27.5 dB, StreamGS: 23 dB) |
| **No view-dependent color** | SH degree 0 only — no specular highlights | pixelSplat, latentSplat, MVSplat use SH degree 2–4 |
| **Static scenes only** | No dynamic object handling | StreamSplat handles dynamic scenes |
| **Nearest-frame only** | Cannot synthesize views far from captured frames | Full-map methods can extrapolate further |
| **Patch-level granularity** | 14×14 px per Gaussian — fine detail may be missed | 1/pixel methods (pixelSplat, MVSplat) have finer granularity |
| **Per-scene MLP residual** | 0.1×tanh residual may overfit to scene-specific appearance | Not applicable to zero-shot methods |

---

## Sources

- pixelSplat: [arXiv:2312.12337](https://arxiv.org/abs/2312.12337) | [GitHub](https://github.com/dcharatan/pixelsplat)
- MVSplat: [arXiv:2403.14627](https://arxiv.org/abs/2403.14627)
- latentSplat: [arXiv:2403.16292](https://arxiv.org/abs/2403.16292) | [Project page](https://geometric-rl.mpi-inf.mpg.de/latentsplat/)
- GPS-Gaussian: [arXiv:2312.02155](https://arxiv.org/abs/2312.02155)
- Splatter Image: [arXiv:2312.13150](https://arxiv.org/abs/2312.13150)
- Flash3D: [arXiv:2406.04343](https://arxiv.org/abs/2406.04343)
- GS-LRM: [arXiv:2404.19702](https://arxiv.org/abs/2404.19702)
- FreeSplat: [arXiv:2405.17958](https://arxiv.org/abs/2405.17958)
- DepthSplat: [arXiv:2410.13862](https://arxiv.org/abs/2410.13862) | [GitHub](https://github.com/cvg/depthsplat)
- NoPoSplat: [OpenReview](https://openreview.net/forum?id=P4o9akekdf) | [GitHub](https://github.com/cvg/NoPoSplat)
- Splatt3R: [arXiv:2408.13912](https://arxiv.org/abs/2408.13912)
- AnySplat: [arXiv:2505.23716](https://arxiv.org/abs/2505.23716)
- StreamGS: [arXiv:2503.06235](https://arxiv.org/abs/2503.06235)
- StreamSplat: [arXiv:2506.08862](https://arxiv.org/abs/2506.08862)
- SLAM3R: [arXiv:2412.09401](https://arxiv.org/abs/2412.09401)
- CAT3D: [arXiv:2405.10314](https://arxiv.org/abs/2405.10314)
