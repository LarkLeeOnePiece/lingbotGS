# Streaming Anchor-GS: Anchor-Aligned Gaussian Splatting for Streaming 3D Reconstruction

## Problem

GCA-Splat uses patch-aligned surfels: each frame independently generates ~4K Gaussians, causing multi-view inconsistency and heavy redundancy in overlapping regions. AnchorSplat proposes anchor-aligned Gaussians (247K vs 5.5M), but operates in batch mode (all views available), incompatible with streaming.

## Goal

Adapt AnchorSplat's anchor-aligned idea to LingBot-Map's streaming inference framework: incremental anchor construction + anchor-aligned Gaussian prediction.

**Priority**: Inference framework first, using LingBot-Map depth as anchor geometry source.

## Architecture

```
Frame t -> [GCT Backbone (frozen)] -> pose, depth, depth_conf, patch_tokens
                                          |
                              [AnchorFeatureHead] -> dense_features [C, H, W]
                                          |
                              [AnchorManager]
                                |-- project existing anchors -> sample features
                                |-- EMA update visible anchor features
                                |-- detect novel regions -> add new anchors
                                |-- enforce budget
                                          |
                              [AnchorGaussianDecoder] -> K Gaussians per anchor
                                          |
                              Anchor-indexed Gaussian Map
```

### Comparison with GCA-Splat

| Dimension | GCA-Splat (current) | Streaming Anchor-GS |
|-----------|-------------------|---------------------|
| Gaussian source | Per-frame patch-aligned | Cross-frame anchor-aligned |
| Per-frame new | ~4K (fixed) | ~0-500 new anchors + update existing |
| Cross-view fusion | Post-hoc voxel merge (lossy) | Feature-level EMA aggregation |
| Redundancy | High in overlap regions | Voxel-dedup, low redundancy |
| Steady-state size | ~167K (budget-capped) | ~50-80K anchors x K |

## Streaming Algorithm

### Phase 1: Scale Frames (frames 0..n-1, bidirectional attention)

1. GCT backbone processes n scale frames -> pose, depth, depth_conf
2. AnchorFeatureHead extracts dense features [B, n, C_feat, H, W]
3. All scale frame depths unprojected to world coordinates -> merged point cloud
4. Voxel-grid downsample (voxel_size=0.05m) -> initial anchor set
5. Each anchor projects back to scale frames, sample features -> average as initial feature
6. AnchorGaussianDecoder predicts K Gaussians per anchor

### Phase 2: Causal Streaming (frames n..S-1, per-frame)

For each frame t:
1. GCT forward (KV cache) -> pose, depth, depth_conf
2. AnchorFeatureHead -> dense_features_t
3. **Projection + visibility**: existing anchors project to frame t, filter visible
   - Visible condition: in image bounds AND projected depth consistent with predicted depth (+-20%)
4. **Update visible anchors**: sample feature -> EMA update
   - `f = (1 - alpha) * f_old + alpha * f_new`, alpha = 1/(1+obs_count)
5. **Add new anchors**: detect uncovered regions (novelty mask)
   - Unproject novel pixels -> voxel dedup -> add to anchor set
6. **Predict Gaussians**: AnchorGaussianDecoder on all anchors
7. **Budget management**: prune low-confidence anchors when over budget

## Module Specifications

### 1. AnchorManager (`lingbot_map/mapping/anchor_manager.py`)

Core data structure managing the incremental anchor set.

**AnchorData fields:**
- `positions: [A, 3]` — world-space coordinates
- `features: [A, C]` — accumulated feature vectors (EMA)
- `obs_counts: [A]` — observation count (capped at 20)
- `confidences: [A]` — running average depth confidence
- `frame_created: [A]` — creation frame index

**Key methods:**
- `init_from_depth()` — Phase 1 initialization from scale frame depth maps
- `project_anchors()` — Project to camera frame, depth consistency check
- `update_visible()` — EMA feature update for visible anchors
- `compute_novelty_mask()` — Scatter + dilate coverage map, invert
- `add_new_anchors()` — Unproject novel pixels, voxel dedup, add
- `update_and_expand()` — Combined update + expand per frame
- `enforce_budget()` — Score = confidence * sqrt(obs_count), keep top-K

**Reuse:** Voxel hash from `gaussian_map.py:205-213`; projection via `pose_enc.py`.

### 2. AnchorFeatureHead (`lingbot_map/heads/anchor_feature_head.py`)

DPTHead wrapper with `feature_only=True`. Output: `[B, S, 256, H, W]`.

### 3. AnchorGaussianDecoder (`lingbot_map/heads/anchor_gaussian_decoder.py`)

MLP: anchor feature -> K Gaussian attributes (12 params per surfel).

Per-surfel attributes:
- position_offset [3]: tanh * max_offset
- opacity [1]: sigmoid
- scale [2]: depth-adaptive base * exp(correction)
- normal [3]: L2 normalized
- color [3]: sigmoid

**Reuse:** Same activation patterns as CompactGaussianHead.

### 4. GCTStreamAnchorGS (`lingbot_map/models/gct_stream_anchor_gs.py`)

Top-level model extending GCTStream. Orchestrates the two-phase streaming pipeline.

**Key method:** `inference_streaming_anchor_gs(images, ...)` -> Dict with pose, depth, anchor_manager, gaussians.

## File Structure

### New files
```
lingbot_map/mapping/anchor_manager.py
lingbot_map/heads/anchor_feature_head.py
lingbot_map/heads/anchor_gaussian_decoder.py
lingbot_map/models/gct_stream_anchor_gs.py
demo_anchor_gs.py
paper/streaming_anchor_gs/PLAN.md
```

### Modified files
- `lingbot_map/mapping/__init__.py` — register AnchorManager

## Implementation Steps

1. **Infrastructure** (AnchorManager) — DONE
2. **Feature Head + Decoder** — DONE
3. **Streaming inference integration** (GCTStreamAnchorGS) — DONE
4. **Demo + visualization** (demo_anchor_gs.py) — TODO
5. **Training** (future): frozen backbone, train FeatureHead + Decoder with rendering loss

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| 80K anchor projection cost | Batch matrix ops ~0.1ms, can add frustum culling |
| Feature EMA drift | obs_count weighted, capped at 20, slow adaptation |
| Anchor unbounded growth | Hard budget + confidence-based pruning |
| Untrained decoder random output | Verify anchor management logic first, evaluate after training |
| Depth quality affects anchors | LingBot-Map depth is good (F1 98.98), depth_conf filters low quality |

## Validation

1. **Anchor management correctness**: Print stats (count, obs_count distribution), verify budget
2. **Projection consistency**: Visualize anchor projections on frames, check depth consistency
3. **PLY export**: Export anchor + Gaussian point clouds, inspect in MeshLab/CloudCompare
4. **Comparison**: Same scene with GCA-Splat vs Anchor-GS, compare Gaussian count and distribution
