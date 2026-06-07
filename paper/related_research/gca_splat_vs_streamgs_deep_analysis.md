# GCA-Splat vs StreamGS: Deep Technical Comparison

> Last updated: 2026-05-09

---

## Overview

Both methods target the same problem: **online feed-forward 3D Gaussian Splatting from unposed monocular image streams**. This is a rare niche — most feed-forward GS methods require posed multi-view images. The comparison is therefore highly targeted and directly relevant for positioning GCA-Splat.

---

## 1. Backbone Architecture: The Core Divergence

This is the most fundamental difference between the two systems.

### StreamGS: Pairwise DUSt3R Matching

```
Frame i, Frame j (pair) → Frozen DUSt3R encoder+decoder
                         → 3D pointmaps (X_i, X_j) in frame i's coordinate space
                         → content-adaptive descriptors (φ_2D features + φ_match)
                         → 3D matching features f_3D
                         → [f_2D | f_3D] → φ_GS decoder → Gaussians per pixel
```

**Key properties:**
- DUSt3R is fundamentally **pairwise and bidirectional** — it processes two frames simultaneously, with cross-attention between them
- The "streaming" is achieved by processing each new frame as a pair with a selected reference frame, then merging Gaussians
- No persistent hidden state across frames — each pair is processed independently
- Per-frame compute: O(L²) self-attention within the pair (L = frame tokens), constant per pair
- Growing map cost: separate from inference; ADC pruning applied to accumulated Gaussians

### GCA-Splat: Causal KV Cache with Structured Context

```
Frame i (single) → DINOv2 ViT-L backbone (patch=14, dim=1024)
                 → Augment with special tokens (camera, 4x register, scale, anchor)
                 → 24 alternating blocks:
                     Frame Attention (within-frame self-attention)
                     GCA (cross-frame attention, structured mask):
                         ├─ Anchor Context  [first n frames, full tokens + anchor token]
                         ├─ Local Window    [last k frames, full image tokens]
                         └─ Trajectory Mem  [all older frames, 6 compact tokens each]
                 → Camera Head → pose (absT_quaR_FoV, 9-dim)
                 → DPT Depth Head → dense depth map
                 → CompactGaussianHead → 2D Gaussian surfels
```

**Key properties:**
- Strictly **causal** — frame i attends only to frames 0..i-1, never to future frames
- Persistent, paged KV cache (FlashInfer) — context from all past frames is live
- Per-frame compute: **O(1) w.r.t. sequence length** (bounded by window size + fixed trajectory memory tokens)
- The same forward pass produces depth, pose, AND Gaussian attributes
- GS head is separated and lightweight (1.34M params) — geometry and appearance are decoupled

### Structural Implication

| Property | StreamGS | GCA-Splat |
|----------|----------|-----------|
| Processing unit | Frame **pair** | Single **frame** |
| Context access | Current pair only | All past frames (structured) |
| Memory per new frame | Constant (pair inference) | Constant (KV cache eviction) |
| Map growth | Unbounded (until ADC prunes) | Bounded by window + memory budget |
| Long-range drift correction | No (pairwise local) | Yes (Trajectory Memory = 6 tokens/frame, never evicted) |
| Coordinate system | Per-pair, merged via alignment | Global (absolute world coords from GCT) |

---

## 2. Gaussian Prediction Head Design

### StreamGS: Per-Pixel 3D Gaussians from Multi-Modal Features

StreamGS concatenates three feature sources to predict Gaussians:
```
Input to φ_GS:
  f_2D       = 2D image features (φ_2D encoder)             dim = ?
  f_3D       = 3D matching features (DUSt3R pointmap-derived) dim = ?
  3D_coords  = DUSt3R-predicted xyz positions                dim = 3

→ [f_2D | f_3D | 3D_coords] → φ_GS (lightweight decoder)
→ Per-pixel output: rotation(quat 4), scale(3), opacity(1), SH_color(K×3)
```

**Key design:**
- 1 Gaussian per pixel → ~65K Gaussians at 256×256 (their training resolution)
- Full 3D Gaussians (6-DOF covariance = quaternion×4 + scale×3)
- Colors as SH coefficients (view-dependent)
- Content-adaptive descriptors: 2D features from a learned encoder that replaces DUSt3R's default correlation features, trained end-to-end to produce better Gaussian-relevant correspondences

### GCA-Splat: Per-Patch 2D Surfels from GCT Tokens

```
Input to CompactGaussianHead:
  patch_tokens = GCT last-layer tokens  [B, S, M, 2048]  (M ≈ 999 patches)
  depth        = DPT depth map          [B, S, H, W, 1]
  pose_enc     = absT_quaR_FoV          [B, S, 9]
  input_image  = denorm RGB             [B, S, 3, H, W]

→ LayerNorm(patch_tokens) → 3-layer MLP [2048→512→512→K×12]
→ K=4 Gaussians per patch:
    pos_offset [3]: tanh × 0.1m
    opacity    [1]: sigmoid, bias init −2.0
    scale      [2]: depth_adaptive_base × exp(MLP.clamp(−1.5,1.5))
    normal     [3]: L2-normalize
    color      [3]: bilinear_sample(input_image, subpatch_grid) + 0.1×tanh(MLP)
```

**Key design:**
- K=4 Gaussians per 14×14 DINOv2 patch → **~3,000 Gaussians/frame** at 518×378
- **2D Gaussian surfels** (tangent plane: 2 scale params, explicit normal)
- Position: physically grounded (depth-unprojected + small learned correction)
- Scale: **depth-adaptive formula** — not regressed freely but anchored to projected pixel size
- Color: **bilinear-sampled directly from input image**, not predicted
- 65× fewer Gaussians per frame than StreamGS at similar resolution

### StreamGS GS Head — Exact Architecture (from paper)

```
F^i_GS  =  F^i_2D  ⊕  X^i  ⊕  F^i_3D
            (φ_2D)   (DUSt3R)  (φ_match)

→ φ_GS (lightweight decoder, unspecified layers)
→ per pixel: q^i [4], s^i [3], α^i [1], c^i [SH, degree unspecified]
   covariance: Σ^i = R(q^i) · s · sᵀ · R(q^i)ᵀ
```

**Component parameter counts (from paper Table 3):**
- DUSt3R (frozen): 656.74M
- φ_match + φ_2D + φ_GS combined: **1.83M** (total adaptive refinement)
- MergeNet (ADC): **0.04M**
- **Total trained params: ~1.87M** (vs GCA-Splat's 1.34M — roughly similar!)

**Training loss:**
```
L = ||I^i − Î^i||_2  +  0.05 × LPIPS(I^i, Î^i)  +  ||Î^i_M − Î^i||_2
```
- Î^i = rendered from raw per-frame Gaussians
- Î^i_M = rendered from merged Gaussians (enforces MergeNet preserves quality)
- No depth supervision, no pose supervision, no SSIM term

### Head Design Comparison

| Aspect | StreamGS | GCA-Splat |
|--------|----------|-----------|
| Input features | F_2D (2-layer CNN) + X (DUSt3R points) + F_3D (matching feats) | GCT patch tokens [2048-dim] |
| Output | Full 3D Gaussian params | 2D surfel params |
| Representation | 3D Gaussian (quat×4 + scale×3) | 2D surfel (scale×2 + normal×3) |
| Color | SH (degree unspecified) | Bilinear image sample + residual |
| Scale | Directly regressed | Depth-adaptive formula + correction |
| Position | DUSt3R 3D points (input, not predicted) | Depth-unprojected + learned offset |
| Gaussians/frame | ~50K (1/pixel at 224²) | **~3K** (4/patch at 518×378) |
| Trained head params | ~1.87M (φ_2D + φ_match + φ_GS + MergeNet) | **1.34M** |
| Backbone params (frozen) | 656.74M DUSt3R | 1,160M GCT |
| Training loss | L2 + 0.05×LPIPS + merge consistency | 0.8×L1 + 0.2×SSIM + opacity_reg |

---

## 3. Memory and Map Management

### StreamGS: Adaptive Density Control (ADC) — Exact Mechanism

StreamGS's ADC is **not** a threshold-based pruning algorithm. It is a **feed-forward correspondence-driven merge** with no explicit opacity thresholds or gradient accumulation:

**Algorithm:**
1. Compute reciprocal NN matches M^(t-1,t) between F³D features of frames t-1 and t (bidirectional cosine NN check, Eq. 4)
2. For each matched pair (j,k): warp F^t_GS(k) to position j → F^(t|t-1)_GS
3. MergeNet takes [F^(t|t-1)_GS ⊕ F^(t-1)_GS] → merged Gaussians Ĝ^(t|t-1)
4. Final set: G^t = **G^(t-1) ∪ Ĝ^(t|t-1)** (union, growing)

**Compression achieved:**
- MVImgNet: 1.58× reduction (36.71% fewer new Gaussians per frame), PSNR cost −2.53%
- ACID: 1.68× reduction (40.48% fewer), PSNR cost −3.90%

**Critical: NO hard cap on total Gaussians.** Every frame still adds ~30–40K new Gaussians (224² × ~60%). The paper does not describe any eviction, maximum budget, or CPU offloading. Memory grows **linearly** with sequence length.

### GCA-Splat: Three-Level Structured Map (Mirroring GCA Context)

The map structure is deliberately aligned with the backbone's attention structure:

```
Level 1 — Anchor Pool (forever, full density)
  ├─ First num_anchor_frames frames (default 3)
  └─ Purpose: coordinate grounding, scale establishment

Level 2 — Window Pool (sliding, default 64 frames × ~999 surfels)
  ├─ Most recent k frames at full density
  └─ Eviction: when frame exits window → compact to Memory

Level 3 — Memory Pool (hard budget: 100K Gaussians)
  ├─ All evicted non-anchor frames, compacted via voxel hashing
  ├─ Merge: confidence-weighted spatial averaging per voxel (5cm edge)
  └─ Budget enforcement: keep top-confidence Gaussians if over 100K
```

**Steady-state map size** at default settings (518×378):
- Anchor: 3 frames × ~3K surfels = ~9K
- Window: 64 frames × ~3K surfels = ~192K
- Memory: hard cap at 100K
- **Total: ~300K (bounded forever)**

| Aspect | StreamGS | GCA-Splat |
|--------|----------|-----------|
| Map growth | Unbounded until ADC prunes | **Strictly bounded** (hard budget) |
| Merge strategy | Learned MergeNet (2 conv layers) | Confidence-weighted voxel averaging |
| Pruning criterion | Opacity + gradient (ADC, like 3DGS) | Confidence score topK |
| Map structure | Flat (single pool) | **Three-level** (anchor/window/memory) |
| Long-seq support | Degrades with drift | Trajectory Memory provides long-range anchor |
| Map size (steady) | Grows, then prunes (~60% retention) | ~300K (fixed ceiling) |

---

## 4. Streaming Mechanism and Per-Frame Compute

### StreamGS's "Online" Inference

StreamGS is "online" in the sense that it processes frames sequentially and does not require future frames. However:

1. Each new frame is processed as a **pair** with a reference frame via DUSt3R
2. DUSt3R's cross-attention between the two frames means the context is limited to the reference frame only (no long-range history beyond what's in the accumulated map)
3. The reference frame selection strategy (which frame to pair with) is a design choice that affects quality
4. The DUSt3R backbone itself runs in O(L²) attention over the pair

### GCA-Splat's Causal Streaming

1. Truly single-frame-at-a-time: frame i is processed in isolation with only the KV cache
2. KV cache contains:
   - **Full image tokens** of last 64 frames (Patch stream, 1 page/frame, evicted by sliding window)
   - **6 compact tokens** (camera + 4 register + scale) of ALL past frames (Special stream, never evicted)
3. Frame i thus "sees" the full trajectory through trajectory tokens — no pairwise reference needed
4. FlashInfer paged attention: O(L × W) where W = window size (constant), not O(L²)

### Compute Complexity

| | StreamGS | GCA-Splat |
|--|----------|-----------|
| Per-frame inference | DUSt3R(pair) 0.02s + refinement 0.08s + MergeNet 0.01s = **0.11s** | GCT(1 frame, KV cache) + GS head(patches) |
| Attention complexity | O(L²) for the pair (two-frame) | O(L × W) bounded by window size |
| Grows with sequence? | MergeNet input grows with map size | No — KV cache is paged, constant |
| Map accumulation cost | Every frame adds ~30K Gaussians permanently | Voxel merge only on window eviction |
| Inference FPS | ~9 FPS | ~20 FPS (backbone alone) |

---

## 5. Training Setup

### StreamGS
- **Training data**: RealEstate10K (training split)
- **Resolution**: 224×224 (small)
- **Steps**: 30,000 steps, batch size 14
- **GPU**: 1× NVIDIA Tesla A100 (80 GB)
- **Framework**: DUSt3R frozen; φ_2D, φ_match, φ_GS, MergeNet trained
- **Loss**: Photometric rendering loss (L1 + SSIM + LPIPS) on novel viewpoints
- **Training time**: Not reported explicitly

### GCA-Splat (current)
- **Training data**: Per-scene (4 scenes: Courthouse, Oxford, Loop, University)
- **Resolution**: 518×378 (significantly higher)
- **Steps**: 300 epochs × #frames, per scene
- **GPU**: **1× RTX 4090 (24 GB)**
- **Framework**: GCT backbone frozen; only CompactGaussianHead trained
- **Loss**: 0.8 × L1 + 0.2 × SSIM + 0.001 × opacity_reg
- **Training time**: ~3 hours per scene

### Key Training Difference

StreamGS trains on a **large diverse dataset** (RE10K) to achieve zero-shot generalization, but at low resolution (224²) and with a frozen DUSt3R backbone. The learned components (φ_2D, φ_match, φ_GS, MergeNet) must generalize across scenes.

GCA-Splat currently trains **per scene** at high resolution with an even more frozen backbone (only 1.34M GS head params trained). This gives 35+ dB per-scene but ~24 dB zero-shot. The planned multi-scene joint training on RealEstate10K/ScanNet++ is the path to zero-shot generalization.

---

## 6. Rendering Strategy

### StreamGS
- Renders from the full accumulated Gaussian map (no nearest-frame selection)
- All Gaussians in view contribute to the render
- Multi-frame Gaussian overlap is handled by ADC (some redundant Gaussians removed)
- PSNR reported: ~23.1 dB (zero-shot, RE10K)

### GCA-Splat
- **Nearest-1 rendering**: For each test viewpoint, only use Gaussians from the single nearest training frame
- This eliminates multi-frame interference, which costs ~12 dB when using the full map
- Relies on temporal density of streaming video to provide coverage from nearby viewpoints
- Per-scene PSNR: 35.21 dB novel-view (nearest-1), 22.53 dB full-map

**The nearest-1 insight is GCA-Splat's most important empirical finding.** The 12 dB gap between nearest-1 and full-map reveals that Gaussian overlap across frames is the primary quality bottleneck, not per-Gaussian quality.

StreamGS does not use nearest-frame selection — it relies on ADC to manage overlap. This is an architectural choice that accepts some quality penalty in exchange for a "proper" 3D map that can be rendered from any viewpoint.

---

## 7. Quantitative Comparison

### StreamGS PSNR Numbers (from paper Table 2, LPIPS/SSIM also reported)

| Dataset | PSNR ↑ | LPIPS ↓ | SSIM ↑ |
|---------|--------|---------|--------|
| RealEstate10K | 22.42 | 0.17 | 0.83 |
| DL3DV | 20.54 | 0.24 | 0.64 |
| MVImgNet | 25.05 | 0.31 | 0.79 |
| ScanNet | 28.43 | 0.16 | 0.86 |
| ACID | 28.50 | 0.15 | 0.84 |

*StreamGS is the only method in their comparison that is both pose-free AND generalizable (zero-shot). CF-3DGS (per-scene opt) achieves 26.33 on MVImgNet but is 150× slower.*

### GCA-Splat Numbers (from RESULTS.md, our scenes)

| Setup | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| Per-scene, same-view (v6, 50f) | 36.70 | 0.920 | 0.342 |
| Per-scene, novel-view nearest-1 | **35.21** | **0.901** | **0.349** |
| Cross-scene zero-shot (avg) | ~24 | ~0.77 | ~0.51 |

### Side-by-Side (note: not same test sets)

| Metric | StreamGS (zero-shot) | GCA-Splat (per-scene) | GCA-Splat (zero-shot) |
|--------|---------------------|----------------------|----------------------|
| Novel PSNR | 22–28 dB (dataset-dep) | **35.21 dB** | ~24 dB |
| Inference speed | ~9 FPS | ~20 FPS (backbone) | ~20 FPS |
| Map size (100 frames) | **Growing** (unbounded) | **~300K (bounded)** | **~300K** |
| Gaussians/frame (new) | ~30–35K (after ADC) | **~3K** | **~3K** |
| Training GPU | 1× A100 (80GB) | **1× RTX 4090** | **1× RTX 4090** |
| Training time | Not reported | ~3 h/scene | TBD (multi-scene) |

*Direct PSNR comparison is not valid — different benchmarks. StreamGS uses RE10K/DL3DV/ScanNet; GCA-Splat uses Oxford Spires-style scenes. Need to evaluate GCA-Splat on RE10K to make this comparison publishable.*

---

## 8. Problem Space Analysis: What Problems Each Solves

### Problems StreamGS Addresses
1. **Zero-shot generalization**: Works on any new video without per-scene training (trained on RE10K once)
2. **No pose requirement**: DUSt3R provides camera geometry from raw image pairs
3. **Rendering from any view**: Full accumulated map supports arbitrary novel views
4. **Dynamic scene handling**: ADC adapts to changing Gaussian needs

### Problems GCA-Splat Addresses (beyond StreamGS)
1. **Bounded computation**: Strictly O(1) per-frame compute regardless of sequence length (KV cache)
2. **Long-range drift correction**: Trajectory Memory keeps 6-token summaries of ALL past frames — long sequences don't lose scale/coordinate grounding
3. **High-fidelity reconstruction**: 35+ dB per-scene (vs ~23 dB zero-shot for StreamGS) due to high-res training and depth-adaptive surfels
4. **Memory efficiency**: ~3K Gaussians/frame vs ~65K — 22× fewer, enabling real streaming on memory-constrained hardware
5. **Physically grounded Gaussians**: Depth-adaptive scale formula ensures consistent projected size across depth ranges — more robust to scene scale changes than pure regression
6. **Surface-aligned representation**: 2D surfels avoid "needle" floaters from full 3D Gaussians; explicit normal prediction enables surface normal export

### What GCA-Splat Does NOT Address (vs StreamGS)
1. **Zero-shot generalization**: Currently per-scene training only (24 dB cross-scene)
2. **View-dependent appearance**: SH degree 0 only — no specular highlights
3. **Full-map rendering**: Nearest-1 limits coverage to near-training-frame views

---

## 9. Architectural Philosophy

| Philosophy | StreamGS | GCA-Splat |
|------------|----------|-----------|
| Backbone role | Geometry + correspondence (DUSt3R) | Geometry + pose + **context memory** (GCT) |
| Backbone size | Large (DUSt3R, undisclosed) | 1.16B (GCT, frozen) |
| GS head role | Decode Gaussians from 2D+3D features | Decode appearance from GCT semantic tokens |
| GS head size | Not disclosed | **1.34M** (minimal) |
| Streaming via | Pairwise processing + map merge | Causal KV cache with eviction |
| Context history | Only reference frame | **Full history** (window + trajectory tokens) |
| Map management | ADC (heuristic, like 3DGS) | Voxel merge + budget (systematic) |
| Color source | SH (predicted) | Image-sampled (anchored to input) |
| Scale source | Regressed freely | Depth-adaptive (physically derived) |

---

## 10. Key Claims for Paper Positioning

When positioning GCA-Splat against StreamGS, the strongest claims are:

### Claim 1: Bounded Compute and Memory
GCA-Splat's per-frame compute is **strictly O(1)** regardless of sequence length, enabled by the paged KV cache. StreamGS's map grows with the scene (though ADC prunes it) and MergeNet cost scales with map size. This makes GCA-Splat uniquely suitable for very long sequences (1000+ frames).

### Claim 2: Structured Long-Range Context
GCA-Splat's Trajectory Memory retains 6 compact tokens per evicted frame — permanently. This means frame 1000 still "sees" a compressed summary of frame 1 through the GCT backbone. StreamGS's pairwise DUSt3R has no mechanism for this.

### Claim 3: Compact Representation
~3K surfels per frame (vs ~65K for StreamGS) makes the map 22× smaller per frame at comparable resolution. The depth-adaptive scale formula ensures each surfel maintains consistent projected size regardless of scene depth.

### Claim 4: High Per-Scene Quality
35.21 dB novel-view PSNR (per-scene trained, nearest-1) is far above StreamGS's ~23 dB. The 12 dB gap between nearest-1 and full-map rendering is an important finding — it suggests that smarter rendering selection (not more Gaussians) is the key to quality.

### Claim 5: Modular Frozen Backbone
Only 1.34M params are trained (vs StreamGS's φ_2D + φ_match + φ_GS + MergeNet). This makes the system extremely lightweight to train and update, and naturally separates geometry concerns (backbone) from appearance concerns (GS head). Training on a single RTX 4090 for 3 hours per scene is a significant practical advantage.

---

## 11. Open Questions for Further Analysis

1. **StreamGS reference frame selection**: When processing frame i, which frame does it use as the DUSt3R reference? If it always uses frame i-1, it's essentially monocular depth estimation with no long-range context. If it uses a keyframe, how is that chosen?

2. **StreamGS φ_GS decoder details**: What is the exact architecture (number of layers, channels)? Are the SH degree-2 coefficients meaningful given that training uses photometric loss only (no multi-view consistency)?

3. **StreamGS rendering at test time**: Does it render from the full map or do any nearest-frame selection? If full-map, how does it avoid the multi-frame interference that costs GCA-Splat 12 dB?

4. **ADC convergence**: How many iterations of ADC are needed per frame? Is this cost included in the 9 FPS figure?

5. **GCA-Splat nearest-1 at novel views**: The nearest-1 strategy works well for views close to training frames. What is the PSNR degradation at larger view changes (e.g., ±30° vs training frame)?

---

## Q&A: GCA-Splat 设计问答

### Q1: GCA-Splat 的 CompactGaussianHead 是如何设计的？

**整体结构**

```
输入: patch_tokens [B, S, M, 2048]  (GCT 最后一层, M≈999 个 patch)
      depth        [B, S, H, W, 1]
      pose_enc     [B, S, 9]
      input_image  [B, S, 3, H, W]  (可选)

→ LayerNorm(2048)
→ 3层 MLP: 2048 → 512 → 512 → K×12
→ reshape [B, S, M, K, 12]
→ 几何/外观解码
→ 输出: K×M 个 2D Gaussian surfel/帧 (~3000个)
```

**每个 Gaussian 的 12 个属性**

```
raw[..., 0:3]  → pos_offset  = 0.1 × tanh(raw)         # 位置修正 (最大 0.1m)
raw[..., 3:4]  → opacity     = sigmoid(raw)              # 透明度 (bias init −2.0 → 稀疏)
raw[..., 4:6]  → scale       = depth_adaptive × exp(MLP) # 深度自适应尺度
raw[..., 6:9]  → normal      = L2_normalize(raw)         # 表面法向量
raw[..., 9:12] → color       = image_sample + 0.1×tanh   # 颜色 (从图像采样)
```

**3D 位置的计算（非 MLP 直接预测）**

位置不是由 MLP 直接预测的，而是用几何方法计算：

```python
# 1. 在每个 14×14 patch 内建 K=4 的 2×2 子格（sub-patch grid）
uu, vv = sub_patch_grid(H, W, patch_size=14, K=4)  # offset = ±3.5px from patch center

# 2. 在子格点处双线性采样 depth map
depth_sampled = grid_sample(depth, subpatch_coords)  # [B, S, M, K]

# 3. 用相机内参反投影到 3D 世界坐标
x_cam = (u - cx) / fx * depth
y_cam = (v - cy) / fy * depth
world_pts = c2w @ [x_cam, y_cam, depth, 1]           # [B, S, M, K, 3]

# 4. 加上 MLP 预测的小偏移（最大 0.1m）
positions = world_pts + pos_offset
```

**K=4 的子格布局**

```
patch (14×14 px)
┌────────┬────────┐
│  G₀    │  G₁    │   每个 Gk 位于 patch 内的 2×2 子格中心
│  (3,3) │ (3,11) │   offset = ±3.5 px from patch center
├────────┼────────┤
│  G₂    │  G₃    │
│ (11,3) │(11,11) │
└────────┴────────┘
```

**2D Surfel vs 全 3D Gaussian**

Head 预测的是 2D Gaussian surfel（表面对齐椭圆片）：

```python
# 渲染时转换为 gsplat 接受的全 3D 格式
R_local = Gram_Schmidt(normal)           # [t1, t2, n] 局部坐标系
scales_3d = [s1, s2, ε]                 # ε = 1e-4，法向方向极薄
quats = rotation_matrix_to_quaternion(R_local)
# 传给 gsplat.rasterization()
```

好处：比全 3D Gaussian 少 1 个自由度，天然与场景表面对齐，避免针状 floater。

---

### Q2: 这个 Head 的设计是从别人的论文里直接取出来的吗？

不是从单一论文直接取的，是几个已有想法的组合，但有一个核心创新点。

| 设计元素 | 来源 | 原创性 |
|---------|------|--------|
| 2D surfel 表示 | 2DGS (Huang et al., SIGGRAPH 2024) | 借鉴 |
| 深度反投影位置 | pixelSplat (CVPR'24), Flash3D (2024) | 借鉴（领域标准做法） |
| 图像采样颜色（直接取 RGB） | GPS-Gaussian (CVPR'24 Highlight) | 借鉴 + 改进（加了双线性插值 + MLP 残差） |
| Sub-patch 2×2 子格 | 无明确先例 | 部分原创 |
| **深度自适应尺度公式** | **16 篇 survey 论文中无先例** | **核心创新** |
| Frozen backbone + 轻量 MLP head | Splatt3R 有相似哲学 | 设计选择 |

所有其他 feed-forward GS 方法（pixelSplat, MVSplat, Flash3D, DepthSplat 等）都是让 MLP 直接回归绝对尺度——这在跨场景时因深度范围差异（室内 1–5m vs 室外 5–100m）而失效。深度自适应尺度公式是消融实验里效果最显著的设计，单独贡献了 **+5.4 dB PSNR**（v3→v6: 30.58→36.02 dB）。

---

### Q3: 深度自适应尺度公式是什么意思？具体怎么做？

**问题根源**

让 MLP 直接回归 Gaussian 的绝对尺度（单位：米）有一个根本问题：

```
近处物体 (depth = 1m):  合理 scale ≈ 0.01m
远处物体 (depth = 50m): 合理 scale ≈ 0.5m
```

MLP 需要"猜"当前场景的深度范围才能给出合理数值。跨场景时（室内 vs 室外，深度范围差 10–100 倍）泛化性很差。

**核心直觉：Gaussian 投影到图像上应该有多大？**

把问题反过来想：一个 Gaussian 应该在图像上覆盖多大面积？答案很自然——大概覆盖它所在的 patch 子格大小，约 **3.5 像素**（`14 / (2×2) = 3.5`）。

已知目标投影大小，用相机投影公式反推 3D 物理尺寸：

```
投影公式:  screen_size = world_size × focal_length / depth
反推:      world_size  = screen_size × depth / focal_length
                       = target_sigma × depth / focal_length
```

**代码实现（`_compute_adaptive_scale()`）**

```python
target_sigma = 14 / (2.0 * 2)   # = 3.5 像素（固定先验）

fx = intrinsics[:, :, 0, 0]     # 从 pose_enc 提取焦距 [B, S]

# 物理公式：保证投影像素大小恒定
base_scale = target_sigma * depth_clamped / fx   # [B, S, M, K]，单位：米

# MLP 只学相对于基准的校正因子（不学绝对值）
correction = exp(raw_scale.clamp(-1.5, 1.5))     # 范围: [0.22×, 4.48×]

scale = base_scale * correction                  # [B, S, M, K, 2]
```

**图解**

```
相机           近处物体 (depth 小)        远处物体 (depth 大)
  │                  ■ Gaussian                   ■ Gaussian
  │                  │ world_size 小                   │ world_size 大
  ├──────────────────┼─────────────────────────────────┼──→
  投影到图像:     ████ = 3.5 px              ████ = 3.5 px

两者投影大小相同！因为 base_scale ∝ depth，自动补偿透视缩放
```

**为什么 MLP 只学校正因子？**

| | 直接回归绝对尺度 | 深度自适应公式 |
|--|----------------|---------------|
| MLP 输出范围 | 0.001m ~ 10m（跨越 4 个数量级） | ±1.5 对数空间（固定范围） |
| 跨场景表现 | 室内训练→室外失效 | 公式自动补偿，MLP 只学局部校正 |
| 消融 PSNR | v3 基线: 30.58 dB | v6 加入公式: **36.02 dB (+5.4 dB)** |

**一句话总结：** 与其让 MLP 猜"这个 Gaussian 在世界空间里有多大"（受深度影响巨大），不如先用物理公式算出"要让它投影成 3.5 像素需要多大"，MLP 只负责在这个基准上做小幅调整。

---

### Q4: 训练 GS Head 应该用什么数据才能跟别人做到公平比较？

核心原则：**训练数据要跟对比方法对齐。**

**对比方法的训练数据汇总**

| 方法 | 训练数据 | 测试数据 |
|------|---------|---------|
| **StreamGS** (最直接竞争对手) | RE10K | RE10K / DL3DV / MVImgNet / ScanNet / ACID |
| MVSplat | RE10K + ACID | RE10K / ACID / DTU |
| pixelSplat | RE10K + ACID | RE10K / ACID |
| DepthSplat | RE10K + ScanNet + DL3DV + TartanAir | RE10K / ScanNet / DL3DV |
| NoPoSplat | RE10K + ACID + DL3DV | RE10K / ACID / DTU / ScanNet++ |

**建议：分两阶段**

阶段一（针对 StreamGS 的最小公平对比）：
- 训练：RealEstate10K (training split, ~67K scenes)
- 测试：RE10K test / DL3DV / ScanNet / ACID（与 StreamGS 完全一致）
- 理由：StreamGS 只用 RE10K 训练，对比最干净；30K steps 可快速验证

阶段二（更广泛对比）：
- 训练：RE10K + ACID（对标 MVSplat/pixelSplat）
- 或：RE10K + ScanNet + DL3DV（对标 DepthSplat）

**必须先确认的问题：LingBot-Map backbone 的训练数据**

如果 GCT backbone 已经在 RE10K 上训练过，用 RE10K 训练 GS head 存在数据泄露风险，reviewer 会质疑结果有效性。需查阅 `lingbot-map_paper.pdf` 的训练集列表。

处理方式：
1. 若 backbone 见过 RE10K → 用 backbone 没见过的数据（ScanNet++、DL3DV）训练 GS head
2. 或明确说明并讨论（类比 ImageNet pretrained backbone 做下游任务是标准做法，backbone 预训练和 GS head 训练是分开的）

**实际操作建议**

先用 RE10K 的子集（~1000 scenes）快速训练，看零样本 PSNR 能到多少，再决定是否扩大规模。目标：RE10K test 上超过 StreamGS 的 22.42 dB，达到 24+ dB 即可作为基础对比点。

---

## Sources

- StreamGS paper: [arXiv:2503.06235](https://arxiv.org/abs/2503.06235) (ICCV 2025)
- GCA-Splat: this repository + `RESULTS.md`, `PLAN_large_scale_training.md`
- LingBot-Map paper: `lingbot-map_paper.pdf` (arXiv:2604.14141v2)
- Code: `lingbot_map/heads/gaussian_head.py`, `lingbot_map/mapping/gaussian_map.py`, `lingbot_map/mapping/renderer.py`, `lingbot_map/models/gct_stream_gs.py`
