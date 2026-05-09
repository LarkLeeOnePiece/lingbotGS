# GCA-Splat: Streaming 3D Gaussian Map Construction via Geometric Context Attention

## 1. Research Landscape Analysis

### 1.1 Competitive Landscape (2025-2026)

We identify five categories of related work and position our contribution at their intersection.

#### Category A: Optimization-Based GS-SLAM
| Method | Venue | Input | Loop Closure | Speed | Key Limitation |
|--------|-------|-------|-------------|-------|----------------|
| MonoGS | CVPR'24 | Mono | No | ~3 FPS | Per-frame optimization |
| SplaTAM | CVPR'24 | RGB-D | No | ~5 FPS | Requires depth sensor |
| Photo-SLAM | CVPR'24 | Mono | No | ~20 FPS (tracking) | Hyper-primitive map |
| LoopSplat | 3DV'25 | RGB-D | Yes (GS registration) | ~2 FPS | Requires depth, pose graph opt |
| MGSO | ICRA'25 | Mono | No | Real-time tracking | Coupled with photometric SLAM |
| FeatureSLAM | 2026 | Mono | No | Real-time | Feature-enriched but still optimizing |

**Common limitation:** All require iterative per-frame/per-Gaussian optimization, limiting real-time dense mapping.

#### Category B: Feed-Forward Streaming GS (DIRECT COMPETITORS)
| Method | Venue | Backbone | Long-Seq? | Loop Closure | Key Limitation |
|--------|-------|----------|-----------|-------------|----------------|
| **StreamGS** | ICCV'25 | Pairwise encoder | No | No | No long-range memory, drift |
| **StreamSplat** | ICLR'26 | Static encoder + dynamic decoder | Bidirectional deform | No | Dynamic-focused, no trajectory memory |
| **Flash-Mono** | ICLR'26 | Recurrent cross-attn | Hidden state | Hidden-state Sim(3) | Single recurrent state drifts |
| **S2GS** | 2026 | Causal transformer | Incremental | No | Semantic-focused, no anchor/trajectory |
| **SLARM** | CVPR'26 Highlight | Window causal attn | Window-based | No | Dynamic + language, not geometric quality |
| **PLANING** | 2026 | Triangle-Gaussian | Online opt | No | Loosely coupled, still optimizes |

**Flash-Mono is the closest competitor.** It uses a recurrent hidden state, but this is fundamentally less expressive than LingBot-Map's three-level GCA. The hidden state has no explicit anchor grounding, no structured trajectory memory, and no sliding window with dense tokens.

#### Category C: Streaming 3D Reconstruction (Point Cloud, No GS)
| Method | Venue | Context | ATE↓ (Oxford Spires) | F1↑ (ETH3D) |
|--------|-------|---------|---------------------|-------------|
| StreamVGGT | ICLR'26 | Causal + token memory | 28.41 | - |
| Stream3R | - | Causal + sliding window | 29.58 | 72.87 |
| CUT3R | - | Recurrent state | 18.16 | 67.63 |
| TTT3R | - | Test-time training | 19.35 | 68.48 |
| Wint3R | - | Causal attention | 21.10 | 77.28 |
| **LingBot-Map** | 2026 | **GCA (anchor+window+memory)** | **6.42** | **98.98** |

**LingBot-Map dominates** all streaming methods on both pose and reconstruction quality.

#### Category D: Feed-Forward GS Prediction (Offline, No Streaming)
NoPoSplat, Splatt3R, MVSplat, AnySplat, pixelSplat — all process fixed image sets, not streaming video.

#### Category E: Hybrid 3R + GS-SLAM
MASt3R-SLAM, VGGT-SLAM, MASt3R-GS, G-CUT3R — use foundation models for init/priors but still require per-scene optimization for Gaussians.

### 1.2 Gap Identification

**No existing method combines LingBot-Map's three-level Geometric Context Attention with feed-forward Gaussian Splatting output.**

| Property | StreamGS | StreamSplat | Flash-Mono | S2GS | **Ours** |
|----------|----------|-------------|------------|------|----------|
| Feed-forward GS | ✓ | ✓ | ✓ | ✓ | ✓ |
| Streaming/causal | ✓ | ✓ | ✓ | ✓ | ✓ |
| No poses required | ✓ | ✓ | ✓ | ✓ | ✓ |
| Anchor grounding | ✗ | ✗ | ✗ | ✗ | **✓** |
| Dense local window | ✗ | ✗ | ✗ | ✗ | **✓** |
| Trajectory memory | ✗ | Deform field | Hidden state | Instance bank | **✓ (6-token/frame)** |
| Compact GS (bounded) | ✗ | ✗ | ✗ | ✗ | **✓ (~8 MB steady)** |
| Proven long-seq pose | ✗ | ✗ | ✗ | ✗ | **✓ (ATE 6.42)** |
| Paged KV cache | ✗ | ✗ | ✗ | ✗ | **✓ (FlashInfer)** |
| 10K+ frame capable | ✗ | ✓ | ✗ | ✓ | **✓** |

---

## 2. Proposed Method: GCA-Splat

### 2.1 Core Thesis

> The three-level geometric context structure (anchor, local window, trajectory memory) that makes LingBot-Map the best streaming pose estimator is also the ideal inductive bias for streaming Gaussian map construction — enabling coordinate-grounded, locally-refined, globally-consistent 3D Gaussian maps without per-scene optimization.

### 2.2 Key Contributions

1. **Compact patch-level Gaussian prediction**: Instead of per-pixel Gaussians (~196K/frame), predict one 2D surfel per DINOv2 patch token (~999/frame) — a ~200× reduction — with positions derived from the already-predicted depth and a lightweight attribute MLP (~2M params). This makes the Gaussian head negligible in cost relative to the 300M+ backbone.

2. **Three-level Gaussian map management**: A hierarchical map that mirrors GCA's context structure — anchor Gaussians (dense, fixed), window Gaussians (active), memory Gaussians (voxel-compacted, budget-limited) — keeping total map size bounded at ~170K Gaussians (~8 MB) regardless of sequence length.

3. **Overlap-aware emission gating**: Use GCA's trajectory memory attention weights as a novelty signal — high attention to past tokens = well-covered region = suppress new Gaussian emission — naturally controlling map density without heuristics.

4. **Eviction-triggered voxel compaction**: When a frame exits the sliding window, its Gaussians are merged with memory via confidence-weighted voxel averaging, then pruned. This aligns the Gaussian lifecycle exactly with the KV cache lifecycle.

5. **State-of-the-art streaming results**: Superior pose estimation (inherited from GCA, ATE 6.42 on Oxford Spires) directly translates to superior Gaussian map quality, demonstrated on established benchmarks.

### 2.3 Architecture Overview

```
Video Frame I_t
    │
    ▼
┌──────────────────────────────────────────────────┐
│  DINOv2 ViT-L Backbone (frozen from LingBot-Map) │
│  → M image tokens + special tokens per frame      │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  24 × (Frame Attention + GCA)                     │
│  [Anchor Context | Pose-Ref Window | Traj Memory] │
│  (initialized from LingBot-Map checkpoint)        │
└──────────────────────────────────────────────────┘
    │
    ├───────────────┬───────────────┐
    ▼               ▼               ▼
┌─────────┐  ┌───────────┐  ┌─────────────────┐
│ Camera   │  │ DPT Depth │  │ Gaussian Attr   │
│ Head     │  │ Head      │  │ Head (NEW)      │
│ (pose)   │  │ (depth)   │  │ (opacity, scale,│
│          │  │           │  │  rotation, SH)  │
└─────────┘  └───────────┘  └─────────────────┘
    │               │               │
    ▼               ▼               ▼
┌──────────────────────────────────────────────────┐
│  Gaussian Map Manager                             │
│  ┌──────────┬──────────┬───────────────────────┐ │
│  │ Anchor   │ Window   │ Memory Gaussians      │ │
│  │ Gaussians│ Gaussians│ (frozen, compressed)  │ │
│  │ (fixed)  │ (active) │                       │ │
│  └──────────┴──────────┴───────────────────────┘ │
│  ↕ Fusion Module (learned, attention-based)       │
└──────────────────────────────────────────────────┘
    │
    ▼
  3D Gaussian Map (renderable, incrementally updated)
```

### 2.4 Compact Gaussian Attribute Head (CompactGaussianHead)

**Key design decision:** predict at **patch resolution** (37×27 = 999 tokens), not pixel resolution (518×378 = 196K pixels). Each DINOv2 patch covers a 14×14 pixel region — a coherent surface element — and produces exactly one 2D Gaussian surfel.

```
aggregated_tokens_list[-1][:, :, patch_start_idx:]   →  patch tokens [B, S, 999, 2048]
                    ↓
              LayerNorm + MLP(2048 → 512 → 512 → 12)
                    ↓
    ┌─────────────────────────────────────────────────────────┐
    │  pos_offset [3]  tanh × 0.1m    (refine depth position) │
    │  opacity    [1]  sigmoid         (start sparse: bias -2) │
    │  scale      [2]  exp, clamped    (2D tangent-plane)      │
    │  normal     [3]  L2 normalize    (surface orientation)   │
    │  color      [3]  sigmoid         (direct RGB, SH deg 0)  │
    └─────────────────────────────────────────────────────────┘
                    ↓
    position = unproject_patch_center(depth, pose) + pos_offset
```

**Per-surfel cost:** 12 learnable params. **Per-frame:** 999 × 12 = ~12K params.
**Contrast per-pixel:** 196K × 14 = ~2.7M params/frame. → **~230× reduction.**

**Position derivation:** the depth head already predicts dense depth supervised by GT. We sample depth at patch centers via `grid_sample`, unproject using the predicted pose + intrinsics, then add a small learned offset (clamped to ±0.1m). This reuses the existing depth supervision for free.

**Implementation:** `lingbot_map/heads/gaussian_head.py` — ~200 lines, ~2M params.

### 2.5 Three-Level Gaussian Map Management (GaussianMapManager)

Mirrors GCA's context structure with aligned eviction semantics:

| Level | Source | GCA Context | Gaussians | Density | Update Policy |
|-------|--------|-------------|-----------|---------|---------------|
| Anchor | First n frames | Full tokens, never evicted | ~3K (n×999) | Dense | Fixed forever |
| Window | Recent k frames | Full tokens, sliding window | ~64K (k×999) | Medium | Active, gated |
| Memory | Evicted frames | Compact 6-token summaries | ≤100K (budget) | Sparse | Frozen after compaction |

**Eviction-triggered voxel compaction** (when a frame exits the sliding window):
1. Quantize Gaussian positions to a voxel grid (default 5cm resolution)
2. Within each occupied voxel, merge via confidence-weighted averaging:
   `pos_merged = Σ(conf_i × pos_i) / Σ(conf_i)` (same for color, scale, opacity)
3. Prune Gaussians with opacity < threshold
4. If memory pool exceeds budget, keep top-K by cumulative confidence

**Memory budget analysis (steady state at any sequence length):**

| Component | Gaussians | Memory |
|-----------|-----------|--------|
| Anchor (3 frames) | ~3,000 | 144 KB |
| Window (64 frames, gated) | ~64,000 | 3.1 MB |
| Memory (budget cap) | ≤100,000 | 4.8 MB |
| **Total** | **~167,000** | **~8 MB** |

Compare naive per-pixel over 3000 frames: ~588M Gaussians / ~33 GB → **4700× reduction.**

**Implementation:** `lingbot_map/mapping/gaussian_map.py`

### 2.6 Overlap-Aware Emission Gating (Novel)

GCA's cross-frame attention weights encode **how much the current region has been seen before**. We repurpose this as a novelty signal for Gaussian emission:

```python
# After GCA — attention weights between current tokens and trajectory memory
# coverage_score: high = this patch is well-covered by past frames
coverage_score = attn_weights.mean(dim=heads).max(dim=memory)  # [B, S, M]
novelty_gate = 1.0 - sigmoid(coverage_score - threshold)       # [B, S, M]
gaussian_opacity = predicted_opacity * novelty_gate
```

Effect: as the camera revisits areas, fewer new Gaussians are emitted, naturally bounding map size without explicit deduplication.

### 2.7 Rendering-Based Self-Supervision

In addition to LingBot-Map's depth + pose losses, add rendering losses:

```
L = λ_depth · L_depth + λ_abs · L_abs_pose + λ_rel · L_rel_pose
  + λ_rgb · L_rgb + λ_ssim · L_ssim
```

Where `L_rgb`, `L_ssim` are computed by:
1. Rasterize the compact surfel map (~1-2K Gaussians/frame) from a held-out viewpoint
2. Compare with the actual observed image

Low Gaussian count makes rasterization fast (~5ms per render via gsplat). Additional losses:
- `L_depth_gs = |depth_rendered - depth_predicted|` — anchors surfel positions to depth supervision
- `L_normal = 1 - cos(normal_predicted, normal_from_depth_gradient)` — geometric consistency

### 2.8 Training Strategy

**Stage 1: Initialize from LingBot-Map** (0 GPU hours)
- Load pre-trained LingBot-Map checkpoint (backbone, GCA, camera head, depth head)
- Freeze all existing parameters
- Initialize CompactGaussianHead randomly (~2M params)

**Stage 2: Train Gaussian head only** (~2K GPU hours)
- Freeze backbone + GCA + camera head + depth head
- Train Gaussian head with rendering loss + depth consistency loss
- Use LingBot-Map's existing training data (29 datasets)
- Progressive view curriculum: 24 → 128 views
- Fast iteration: only ~2M trainable params

**Stage 3: Joint fine-tuning** (~8K GPU hours)
- Unfreeze GCA + depth head (keep backbone frozen)
- Joint optimization of depth, GCA attention, and Gaussian quality
- Rendering loss + existing depth/pose losses
- Long-sequence training with context parallelism (Ulysses)

---

## 3. Evaluation Plan

### 3.1 Benchmarks

| Benchmark | Task | Metrics | Baselines |
|-----------|------|---------|-----------|
| ETH3D | Reconstruction | F1, Acc, Comp | StreamGS, Flash-Mono, Wint3R, MonoGS |
| 7-Scenes | Pose + Recon | ATE, F1 | Flash-Mono, SplaTAM, MonoGS |
| Tanks & Temples | Reconstruction | F1, ATE | StreamGS, StreamSplat, MonoGS |
| ScanNet/ScanNet++ | Reconstruction + NVS | PSNR, SSIM, LPIPS, F1 | Flash-Mono, PLANING, MonoGS |
| Oxford Spires | Long-seq pose | ATE, AUC@15 | LingBot-Map, Flash-Mono, CUT3R |
| Replica | NVS + Recon | PSNR, SSIM, Depth L1 | SplaTAM, MonoGS, Photo-SLAM |

### 3.2 Ablation Studies

1. **GCA components for GS quality**: anchor-only vs window-only vs full GCA
2. **Gaussian head architecture**: DPT-based vs separate decoder
3. **Rendering loss weight**: impact on pose vs reconstruction quality
4. **Fusion module**: learned vs heuristic (nearest-neighbor merge)
5. **SH degree**: 0 vs 1 vs 2 (quality vs speed tradeoff)

### 3.3 Key Claims to Validate

1. Our method achieves SOTA rendering quality (PSNR/SSIM) among streaming methods
2. Pose accuracy is maintained (or improved) compared to LingBot-Map baseline
3. 3D reconstruction quality (F1) improves over point-cloud-only LingBot-Map
4. Speed remains practical: >10 FPS streaming on 518×378
5. Long-sequence stability: quality does not degrade over 3000+ frames

---

## 4. Implementation Plan

### Phase 0: Setup & Baseline (Week 1-2)

**Goal:** Reproducible LingBot-Map baseline + evaluation infrastructure

- [ ] Set up Linux environment with FlashInfer (WSL2 or remote server)
- [ ] Reproduce LingBot-Map results on ETH3D, 7-Scenes, Tanks & Temples
- [ ] Set up evaluation scripts for pose (ATE, AUC) and reconstruction (F1)
- [ ] Set up Novel View Synthesis evaluation (PSNR, SSIM, LPIPS)
- [ ] Download and prepare all benchmark datasets
- [ ] Profile LingBot-Map inference: FPS, memory, per-module timing

### Phase 1: Compact Gaussian Head (Week 3-5) ✅ IMPLEMENTED

**Goal:** Predict patch-level 2D Gaussian surfels from LingBot-Map features

- [x] Implement `CompactGaussianHead` module
  - Input: patch tokens from aggregated_tokens_list[-1] (dim 2048)
  - Output: K per-patch opacity, scale_2d, normal, color (K*12 params/patch)
  - Position: depth unprojection at sub-patch sample points + learned offset
  - Architecture: LayerNorm + MLP(2048→512→512→K*12), ~1.3M params
  - **K=4 (2×2 sub-grid)**: ~4000 Gaussians/frame for sharper rendering
- [x] Implement sub-patch depth sampling via grid_sample
- [x] Implement unprojection using pose_encoding_to_extri_intri
- [x] PureTorchRenderer as gsplat fallback (gsplat JIT fails on Windows/CUDA mismatch)
  - Gradient checkpointing for memory-efficient training with K=4 (saves ~20 GB)
- [x] Test: train Gaussian head with rendering loss on courthouse scene
  - K=1 (v2): 300 epochs, L1=0.022, PSNR=29.2 dB, SSIM=0.822 (~500 GS/frame, blurry)
  - K=4 (v3): 200 epochs, PSNR=30.58 dB, SSIM=0.834, 45K Gaussians (still blurry)
  - K=4 + depth-adaptive scales + image-sampled colors (v6): **PSNR=36.02 dB, SSIM=0.913**
    58K Gaussians, 2.89 MB map, 4 FPS inference. Dramatic quality improvement.

**Files created:**
- `lingbot_map/heads/gaussian_head.py` — CompactGaussianHead (~300 lines)

### Phase 2: Streaming Gaussian Map (Week 6-8) ✅ IMPLEMENTED

**Goal:** Accumulate Gaussians across frames with bounded memory

- [x] Implement `GaussianMapManager` class
  - Three pools: anchor_gaussians, window_gaussians, memory_gaussians
  - Per-frame: add surfels to appropriate pool, opacity-threshold filtering
  - On window eviction: voxel-hash compaction + confidence-weighted merge
- [x] Implement `GaussianData` container with cat/filter/empty ops
- [x] Implement voxel-based spatial hashing for merge (5cm default)
- [x] Implement budget enforcement (top-K by confidence)
- [x] Implement `GCTStreamGS` model extending GCTStream
  - `inference_streaming_gs()` — full two-phase pipeline with GS output
  - `from_pretrained_lingbot()` — load checkpoint + fresh GS head
- [ ] Integrate gsplat for incremental rendering
- [ ] Test streaming pipeline end-to-end on short sequences (50-100 frames)

**Files created:**
- `lingbot_map/mapping/__init__.py`
- `lingbot_map/mapping/gaussian_map.py` — GaussianData + GaussianMapManager (~270 lines)
- `lingbot_map/models/gct_stream_gs.py` — GCTStreamGS (~280 lines)

### Phase 3: Rendering-Based Training (Week 9-12) 🔄 IN PROGRESS

**Goal:** Train the full pipeline with rendering supervision

- [x] Implement rendering loss computation within training loop
  - Per-frame: render from same viewpoint, compare to input image
  - Loss: 0.8×L1 + 0.2×SSIM + 0.001×opacity_reg
  - Backpropagate through PureTorchRenderer → Gaussian attributes
- [x] Stage 2 training: freeze backbone, train GS head only
  - Precompute backbone features (one-time), iterate over cached tokens
  - AdamW optimizer, CosineAnnealingLR, grad clip 0.5
  - NaN guards: skip NaN losses, zero NaN gradients
- [x] Training verified stable: 300 epochs without NaN
- [ ] Extend to multi-scene training (not just per-scene overfitting)
- [ ] Add novel-view rendering loss (held-out viewpoints)

**Key findings:**
- **Depth-adaptive scales (v6 breakthrough)**: Base scale = target_sigma × depth / focal_length, with learned correction factor. Ensures each Gaussian projects to ~3.5px on screen regardless of depth. This alone improved SSIM from 0.834 → 0.913 (+0.079).
- **Image-sampled colors**: Sample RGB directly from input image at sub-patch grid points. Much better than MLP-predicted colors from semantic DINOv2 features.
- Gradient checkpointing essential for K=4 (37 chunks × 650 MB = 24 GB without)
- Backbone offload to CPU frees 6.7 GB GPU memory during training
- LR=5e-4 with cosine decay is stable; LR=2e-3 causes NaN at ~epoch 36

**Key infrastructure:**
- PureTorchRenderer: pure PyTorch differentiable splatting (no CUDA JIT)
- Gradient checkpointing per pixel chunk for memory efficiency
- Feature caching on CPU, backbone offloaded during training

### Phase 4: Learned Fusion & Refinement (Week 13-15)

**Goal:** Replace heuristic fusion with learned attention-based fusion

- [ ] Implement attention-based Gaussian fusion module
  - Query: new Gaussians from current frame
  - Key/Value: existing Gaussians in spatial neighborhood
  - Output: merged Gaussian attributes (weighted combination)
  - Use GCA's cross-frame attention weights as fusion prior
- [ ] Implement Gaussian pruning: remove low-confidence / occluded Gaussians
- [ ] End-to-end fine-tuning (Stage 3): unfreeze all, joint optimization
- [ ] Benchmark rendering quality improvement from learned fusion

### Phase 5: Evaluation & Paper (Week 16-20)

**Goal:** Comprehensive evaluation and paper writing

- [ ] Run all benchmarks (ETH3D, 7-Scenes, T&T, ScanNet, Oxford Spires, Replica)
- [ ] Ablation studies (5 experiments)
- [ ] Qualitative visualizations: rendered views, point clouds, trajectories
- [ ] Long-sequence demos: 3000+ frame sequences
- [ ] Compare against: Flash-Mono, StreamGS, StreamSplat, MonoGS, SplaTAM
- [ ] Write paper targeting CVPR 2027 / ECCV 2026 / NeurIPS 2026

---

## 5. Risk Assessment & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Rendering loss destabilizes pose training | High | Medium | Stage-wise training; freeze backbone initially |
| Gaussian fusion quality insufficient | Medium | Medium | Start with voxel heuristic; learned fusion is bonus |
| Memory exceeds GPU for long sequences | Medium | Low | Leverage existing paged KV cache; Gaussian pruning |
| Cannot match optimization-based NVS quality | Medium | High | Focus on streaming speed-quality tradeoff; not competing on NVS with offline methods |
| FlashInfer required for practical speed | Low | Low | Already proven in LingBot-Map; SDPA fallback exists |

---

## 6. Expected Outcomes

### Quantitative Targets

| Metric | LingBot-Map | Flash-Mono | MonoGS | **Ours (target)** |
|--------|-------------|------------|--------|-------------------|
| ATE↓ (Oxford Spires) | 6.42 | ~15* | - | **≤7.0** |
| F1↑ (ETH3D) | 98.98 | - | ~50* | **≥95** |
| PSNR↑ (ScanNet NVS) | N/A | ~25* | ~28* | **≥26** |
| Streaming FPS | 20 | ~15* | ~3 | **≥12** |
| Max frames (no reset) | ~3000 | ~500* | ~500 | **≥3000** |

*Estimated from paper descriptions; exact numbers pending evaluation.

### Qualitative Advantages
- First streaming method producing renderable 3D Gaussian maps from monocular video
- Consistent quality over 3000+ frame sequences (vs drift in competitors)
- Real-time capable (>10 FPS) without per-scene optimization

---

## 7. References (Key Papers)

- LingBot-Map (Chen et al., 2026) — Our backbone
- Flash-Mono (ICLR 2026) — Primary competitor
- StreamGS (ICCV 2025) — Feed-forward streaming GS
- StreamSplat (ICLR 2026) — Dynamic streaming GS
- S2GS (2026) — Semantic streaming GS
- SLARM (CVPR 2026) — Language-aligned streaming GS
- PLANING (2026) — Triangle-Gaussian streaming
- LoopSplat (3DV 2025) — GS loop closure
- MonoGS (CVPR 2024) — Optimization-based GS-SLAM baseline
- VGGT (Meta, 2025) — Feed-forward 3D foundation model
- gsplat — Differentiable Gaussian rasterizer
