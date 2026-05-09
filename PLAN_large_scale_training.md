# Plan: Extending GCA-Splat Training to Large Datasets

## Current State

- **Training setup**: Per-scene, 50 frames, 300 epochs, ~3 hours on 1x RTX 4090 (24 GB)
- **GS head**: 1.34M params (LayerNorm + 3-layer MLP, 2048->512->512->48)
- **Backbone**: Frozen LingBot-Map (1.16B params), features precomputed once and cached on CPU
- **Per-frame cache**: ~20 MB (4 DPT feature levels x [1,1,~1005,1024] float32 + depth + pose + image)
- **Best result**: 35.2 dB novel-view PSNR on courthouse (50 frames, nearest-1 rendering)
- **Cross-scene zero-shot**: ~24 dB (trained on courthouse, tested on university/oxford/loop)
- **System**: 256 GB RAM, 24 GB VRAM (primary), 8 GB VRAM (secondary)

## Goal

Train a **single unified GS head** that generalizes across diverse scenes, instead of per-scene fine-tuning. This would make GCA-Splat a true feed-forward system: given any new video stream from LingBot-Map, the GS head produces high-quality Gaussians without retraining.

## Scaling Dimensions

### A. More Frames Per Scene (50 -> 200-300)

Currently we train on 50 frames from each scene. Each scene has 237-324 frames total.

**What changes:**
- Feature cache: 50 frames = ~1 GB -> 300 frames = ~6 GB per scene (easily fits in 256 GB RAM)
- Backbone precomputation: 50 frames = ~35s -> 300 frames = ~210s (one-time cost)
- Epoch time: scales linearly. 50f = ~36s/epoch -> 300f = ~216s/epoch
- With 300 epochs: ~18 hours per scene

**Things to be careful about:**
1. **Even/odd split at scale**: With 300 frames, we train on 150 even frames. The training set becomes very dense (nearly continuous video). The GS head might overfit to interpolation and not learn to generalize.
2. **Diminishing returns**: With nearest-1 rendering, additional frames mainly add coverage. After sufficient coverage of the scene, more frames won't improve PSNR much. 50 frames may already be near-optimal for small scenes.
3. **KV cache sliding window**: The backbone uses a 16-frame sliding window. For frames far apart in the sequence, the backbone features may be less coherent. This is fine — it matches how LingBot-Map actually works.

**Recommendation**: Start with full scenes (all available frames) but measure PSNR vs frame count to find the sweet spot.

### B. Multi-Scene Joint Training (4 scenes -> 10-100+ scenes)

This is the critical step for generalization. The current 24 dB cross-scene gap (vs 35 dB in-scene) shows the GS head is heavily overfitting to scene-specific geometry/appearance.

**What changes:**
- Feature cache: 4 scenes x 300 frames x 20 MB = ~24 GB RAM (fine)
- Backbone precomputation: 4 scenes x 210s = ~14 minutes total (one-time)
- Need to sample across scenes during training, not just within one scene
- GS head architecture may need to grow to handle scene diversity

**Things to be careful about:**

1. **Scene-level batch sampling strategy**:
   - Naive: shuffle all frames from all scenes together -> problem: gradients dominated by larger scenes
   - Better: **balanced scene sampling** — each training step, pick a random scene, then a random frame from that scene. This ensures equal contribution regardless of scene size.
   - Even better: **curriculum** — start with easier indoor scenes, gradually add harder outdoor ones

2. **Depth scale variance across scenes**:
   - LingBot-Map predicts metric depth, but scale can vary (indoor 0.5-5m vs outdoor 5-100m)
   - The `base_scale = target_sigma * depth / focal_length` formula handles this adaptively
   - BUT: the learned MLP correction might overfit to one depth range
   - **Mitigation**: Normalize depth statistics per scene, or add depth range as a conditioning signal to the GS head

3. **Appearance diversity**:
   - Different scenes have very different lighting, textures, sky/ground ratios
   - Image-sampled colors (grid_sample from input) are scene-agnostic by design — this is good
   - The learned 0.1*tanh residual should be small enough to not overfit
   - **Risk**: If MLP residual grows too large, it becomes scene-specific. Monitor residual magnitude during training.

4. **Resolution/aspect ratio differences**:
   - All images are currently cropped to 518x378. If source images vary in aspect ratio, the crop removes different amounts of content.
   - For large datasets with varying resolutions, need consistent preprocessing.
   - **Mitigation**: Always use the same crop mode and target size.

5. **Camera trajectory diversity**:
   - Some scenes are forward-facing, others are 360-degree
   - The nearest-1 rendering naturally handles this — it only cares about distance to nearest training viewpoint
   - BUT: training the GS head on forward-facing scenes might not teach it to produce Gaussians visible from behind
   - **Mitigation**: Include diverse trajectory types in training data

### C. Large-Scale Datasets (100+ scenes, 10K+ frames)

**Candidate datasets:**
- **ScanNet/ScanNet++**: 1500+ indoor scenes with RGB-D, diverse rooms and furniture
- **MegaDepth**: 100K+ internet photos of landmarks, with SfM poses
- **RealEstate10K**: 80K video clips of real estate walkthroughs
- **Tanks & Temples**: 14 scenes, high-quality outdoor
- **ETH3D**: 25 scenes, indoor/outdoor with laser-scanned GT depth
- **7-Scenes**: 7 indoor scenes, RGB-D

**Memory budget at scale:**
| Frames | Cache Size | Backbone Precompute | Notes |
|--------|-----------|---------------------|-------|
| 1,000  | ~20 GB    | ~12 min             | 4 scenes, full |
| 5,000  | ~100 GB   | ~60 min             | ~20 scenes |
| 10,000 | ~200 GB   | ~2 hours            | Fits in 256 GB RAM |
| 50,000 | ~1 TB     | ~10 hours           | Needs disk-based caching |

**Things to be careful about:**

1. **Memory wall at ~10K frames**:
   - At 20 MB/frame, 10K frames = 200 GB — just barely fits in our 256 GB RAM
   - Beyond this, **disk-based feature caching** is mandatory
   - Solution: Save precomputed features as `.pt` files per frame, load on-demand during training
   - This adds I/O overhead but is necessary. With NVMe SSD, loading a 20 MB tensor takes <1ms.

2. **Implementation: Lazy feature loading**:
   ```python
   class LazyFrameCache:
       """Load precomputed features from disk on demand."""
       def __init__(self, cache_dir):
           self.cache_dir = cache_dir
           self.index = json.load(open(f"{cache_dir}/index.json"))

       def __getitem__(self, idx):
           scene, frame = self.index[idx]
           return torch.load(f"{self.cache_dir}/{scene}/frame_{frame:04d}.pt")

       def __len__(self):
           return len(self.index)
   ```

3. **Backbone precomputation becomes a bottleneck**:
   - At ~1.2 frames/sec, 50K frames = 11.5 hours just for feature extraction
   - Can parallelize across GPUs or run as a batch preprocessing step
   - Features only need to be computed once and saved to disk
   - **Critical**: Must ensure consistent backbone state (same checkpoint, same preprocessing)

4. **GS head capacity**:
   - Current: 1.34M params (3-layer MLP). This may be too small for 100+ diverse scenes.
   - Options to scale:
     - Wider: 512 -> 1024 hidden dim (~5M params)
     - Deeper: 3 -> 6 layers (~3M params)
     - Add scene-level conditioning (scene embedding or statistics)
     - Add a lightweight cross-attention layer to attend to the DPT features more selectively
   - **Be careful**: Larger head = slower training per step AND risk of overfitting with insufficient data
   - **Recommendation**: Only increase capacity if training loss plateaus while validation loss is still high

5. **Evaluation protocol for generalization**:
   - Hold out entire scenes (not just frames) for testing
   - Report both in-distribution (trained scenes, held-out frames) and out-of-distribution (unseen scenes)
   - The key metric is: **zero-shot novel scene PSNR**
   - Currently 24 dB zero-shot. Target: >30 dB would be publishable.

6. **Training dynamics at scale**:
   - With 50 frames, 300 epochs = 15K frame updates. This works.
   - With 10K frames, 300 epochs = 3M frame updates. Way too many.
   - **Scale epochs inversely**: More frames -> fewer epochs needed
   - Rule of thumb: ~50K-100K total frame updates is probably sufficient
   - With 10K frames, that's 5-10 epochs
   - Use **iteration-based** training instead of epoch-based

7. **Learning rate schedule**:
   - Current: CosineAnnealing over 300 epochs
   - At scale: Use **warmup + cosine decay** over total iterations
   - Warmup: 1000 iterations (important when seeing diverse scenes early)
   - Total: 50K-100K iterations with batch_size=1 (or equivalent)

8. **Cross-view loss: skip it**:
   - Our experiments show cross-view reprojection loss doesn't help when the GS head is well-trained
   - At large scale, the GS head will see enough viewpoint diversity naturally
   - Dropping it saves 0.85x training overhead per step
   - **Conclusion**: Do not use cross-view loss for large-scale training

## Recommended Implementation Plan

### Phase 1: Full-Scene Training (immediate, ~1 day)
- Train on all 4 scenes with full frames (237-324 each)
- Use balanced scene sampling
- 100 epochs (since we have ~6x more frames per scene)
- Validate on held-out scenes (leave-one-out)
- **Goal**: Establish multi-scene baseline, see if gap closes from 24 to >28 dB

### Phase 2: Disk-Based Feature Cache (1-2 days engineering)
- Implement `precompute_and_save.py`: Run backbone over any scene, save features to disk
- Implement `LazyFrameCache` in train_gs.py
- Implement scene-balanced DataLoader
- Switch from epoch-based to iteration-based training
- **Goal**: Remove memory constraint, enable arbitrary scale

### Phase 3: Dataset Expansion (1 week)
- Download and preprocess RealEstate10K or ScanNet++ (most diverse)
- Precompute features for all scenes (parallelize across GPUs if available)
- Train unified GS head on 50-100 scenes
- Evaluate zero-shot on held-out scenes
- **Goal**: >30 dB zero-shot novel-view PSNR

### Phase 4: Architecture Scaling (if needed)
- If Phase 3 shows capacity bottleneck (train loss >> val loss of per-scene models)
- Try wider MLP (1024 hidden) or add lightweight conditioning
- Use depth statistics or scene embedding as extra input
- **Goal**: Close the gap between per-scene (35 dB) and unified (target: 32+ dB)

## Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU RAM overflow at >10K frames | Training crashes | Disk-based lazy loading |
| Backbone precompute too slow | Days of preprocessing | Parallelize, cache to SSD |
| GS head underfits diverse scenes | Low quality everywhere | Increase model capacity carefully |
| GS head overfits to scene distribution | Poor on unseen scene types | Diverse training data, held-out eval |
| Depth scale mismatch across scenes | Gaussian sizes wrong for some scenes | Depth-adaptive scaling already handles this; monitor per-scene metrics |
| Training instability with diverse data | NaN/divergence | Gradient clipping (already have), warmup, lower LR |
| Diminishing returns from scale | Wasted compute | Track zero-shot PSNR at checkpoints, stop when plateaus |

## What NOT to Change

- **Nearest-1 rendering**: This is our key insight. Keep it regardless of scale.
- **Image-sampled colors**: Already scene-agnostic by design. Don't replace with learned colors.
- **Frozen backbone**: The 1.16B backbone is too expensive to fine-tune. Keep it frozen.
- **Surfel representation**: 2D surfels with depth-adaptive scales work well. Don't switch to 3D Gaussians.
- **Patch-level prediction**: One GS head forward pass per frame, not per pixel. This keeps inference fast.

## Quick Wins Before Scaling

Before investing in large-scale training, these experiments can validate the approach:

1. **4-scene joint training** (current scenes, ~1 day): If the unified head reaches 30+ dB on held-out frames of trained scenes AND >28 dB on the held-out scene, scaling is worth pursuing.

2. **Leave-one-out**: Train on 3 scenes, eval on 4th. If zero-shot improves from 24 to 28+ dB, the approach generalizes.

3. **Frame count ablation**: Train on 20/50/100/200 frames of courthouse, plot PSNR vs frames. Find the saturation point.
