"""
GCTStreamAnchorGS - Streaming GCT with anchor-aligned Gaussian Splatting.

Extends GCTStream with:
- Lightweight token-based feature extraction (no extra DPT pass)
- AnchorManager for incremental 3D anchor management (voxel-dedup, EMA update)
- AnchorGaussianDecoder for per-anchor Gaussian surfel prediction

Unlike GCTStreamGS (patch-aligned, ~4K Gaussians/frame), this approach:
- Shares anchors across views (anchor-aligned, not pixel-aligned)
- Grows anchors incrementally via novelty detection
- Updates anchor features via EMA across observations
- Produces K Gaussians per anchor (e.g., 4), giving ~50-80K anchors total
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from tqdm.auto import tqdm

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.heads.anchor_gaussian_decoder import AnchorGaussianDecoder
from lingbot_map.mapping.anchor_manager import AnchorManager
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3

logger = logging.getLogger(__name__)


class GCTStreamAnchorGS(GCTStream):
    """Streaming GCT with anchor-aligned Gaussian map construction.

    Uses a lightweight linear projection of transformer patch tokens as
    dense features (no extra DPT pass), keeping overhead minimal.
    """

    def __init__(
        self,
        *,
        anchor_feature_dim: int = 256,
        anchor_voxel_size: float = 0.05,
        anchor_budget: int = 80_000,
        gaussians_per_anchor: int = 4,
        anchor_decoder_hidden: int = 256,
        anchor_max_pos_offset: float = 0.05,
        anchor_depth_thresh: float = 0.2,
        anchor_min_depth_conf: float = 1.5,
        **kwargs,
    ):
        self._anchor_feature_dim = anchor_feature_dim
        self._anchor_voxel_size = anchor_voxel_size
        self._anchor_budget = anchor_budget
        self._anchor_K = gaussians_per_anchor
        self._anchor_decoder_hidden = anchor_decoder_hidden
        self._anchor_max_pos_offset = anchor_max_pos_offset
        self._anchor_depth_thresh = anchor_depth_thresh
        self._anchor_min_depth_conf = anchor_min_depth_conf

        super().__init__(**kwargs)

        # Lightweight feature projector: project last-layer tokens to anchor features
        # Input: 2 * embed_dim (concatenated tokens from DPT-style layers)
        # This replaces the expensive AnchorFeatureHead (full DPT forward)
        self.anchor_token_proj = nn.Sequential(
            nn.LayerNorm(2 * self.embed_dim),
            nn.Linear(2 * self.embed_dim, anchor_feature_dim),
        )

        # ------ Anchor Gaussian decoder (new, randomly initialised) ------
        self.anchor_decoder = AnchorGaussianDecoder(
            feature_dim=anchor_feature_dim,
            hidden_dim=anchor_decoder_hidden,
            K=gaussians_per_anchor,
            max_pos_offset=anchor_max_pos_offset,
        )

        # Created per-sequence in inference; not a module parameter
        self.anchor_manager: Optional[AnchorManager] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_lingbot(
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cuda"),
        **anchor_kwargs,
    ) -> "GCTStreamAnchorGS":
        """Load a LingBot-Map checkpoint and attach fresh anchor heads."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "config" in ckpt:
            init_kwargs = ckpt["config"]
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
            init_kwargs = {}

        init_kwargs.update(anchor_kwargs)

        model = cls(**init_kwargs)

        # Handle pos_embed size mismatch when training at different resolution
        pe_key = "aggregator.patch_embed.pos_embed"
        if pe_key in state_dict:
            model_pe = model.state_dict().get(pe_key)
            if model_pe is not None and state_dict[pe_key].shape != model_pe.shape:
                logger.info(
                    "Interpolating pos_embed: %s -> %s",
                    state_dict[pe_key].shape, model_pe.shape,
                )
                old_pe = state_dict[pe_key]  # [1, N_old, D]
                # Separate cls/register tokens from patch tokens
                num_prefix = model.aggregator.patch_embed.num_tokens  # cls + register tokens
                prefix = old_pe[:, :num_prefix]
                patch_pe = old_pe[:, num_prefix:]  # [1, P_old, D]
                # Infer old grid size
                old_gs = int(patch_pe.shape[1] ** 0.5)
                D = patch_pe.shape[-1]
                patch_pe = patch_pe.reshape(1, old_gs, old_gs, D).permute(0, 3, 1, 2)
                # New grid size
                new_n = model_pe.shape[1] - num_prefix
                new_gs = int(new_n ** 0.5)
                patch_pe = torch.nn.functional.interpolate(
                    patch_pe, size=(new_gs, new_gs), mode="bicubic", align_corners=False,
                )
                patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, -1, D)
                state_dict[pe_key] = torch.cat([prefix, patch_pe[:, :new_n]], dim=1)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        anchor_keys = [k for k in missing
                       if "anchor_token_proj" in k or "anchor_decoder" in k]
        other_missing = [k for k in missing if k not in anchor_keys]
        if other_missing:
            logger.warning("Missing non-anchor keys: %s", other_missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)
        logger.info(
            "Loaded checkpoint. %d anchor-head params randomly initialised.",
            len(anchor_keys),
        )
        return model.to(device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_anchor_manager(self, device: torch.device) -> AnchorManager:
        return AnchorManager(
            voxel_size=self._anchor_voxel_size,
            anchor_budget=self._anchor_budget,
            feature_dim=self._anchor_feature_dim,
            depth_consistency_thresh=self._anchor_depth_thresh,
            min_depth_conf=self._anchor_min_depth_conf,
            device=device,
        )

    def _tokens_to_feature_map(
        self,
        aggregated_tokens_list: list,
        patch_start_idx: int,
        H: int, W: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Extract dense features from last-layer tokens via linear projection.

        Returns features at **patch resolution** [num_frames, C_feat, patch_h, patch_w].
        Callers use grid_sample to query at arbitrary pixel coordinates, so
        full-resolution upsampling is unnecessary and wasteful.
        """
        # Last layer tokens: [B, S, num_tokens, 2*embed_dim]
        tokens = aggregated_tokens_list[-1]
        B, S_tok = tokens.shape[:2]
        patch_tokens = tokens[:, :, patch_start_idx:]  # [B, S, num_patches, 2*D]

        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # Flatten B*S for projection
        flat = patch_tokens.reshape(B * S_tok, -1, patch_tokens.shape[-1])

        # Project to anchor feature dim
        with torch.amp.autocast("cuda", enabled=False):
            feat = self.anchor_token_proj(flat.float())  # [B*S, P, C_feat]

        # Reshape to spatial (patch resolution only, no upsample)
        feat = feat.permute(0, 2, 1).reshape(-1, self._anchor_feature_dim, patch_h, patch_w)

        return feat[:num_frames]  # [S, C_feat, patch_h, patch_w]

    def _pose_enc_to_c2w_and_intrinsics(
        self,
        pose_enc: torch.Tensor,  # [B, S, 9]
        H: int, W: int,
    ):
        """Convert pose encoding to c2w [B, S, 4, 4] and intrinsics [B, S, 3, 3]."""
        device = pose_enc.device
        dtype = pose_enc.dtype

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(H, W), build_intrinsics=True,
        )

        B, S = extrinsics.shape[:2]
        ext_flat = extrinsics.view(B * S, 3, 4)
        ext_4x4 = torch.zeros(B * S, 4, 4, device=device, dtype=dtype)
        ext_4x4[:, :3, :] = ext_flat
        ext_4x4[:, 3, 3] = 1.0
        c2w = closed_form_inverse_se3(ext_4x4).view(B, S, 4, 4)

        return c2w, intrinsics

    def _unproject_pixels(
        self,
        depth_map: torch.Tensor,  # [H, W]
        c2w: torch.Tensor,       # [4, 4]
        intrinsics: torch.Tensor, # [3, 3]
        H: int, W: int,
        mask: Optional[torch.Tensor] = None,  # [H, W] bool
    ) -> torch.Tensor:
        """Unproject depth pixels to world coordinates. Returns [N, 3]."""
        device = depth_map.device

        v_idx, u_idx = torch.meshgrid(
            torch.arange(H, device=device, dtype=depth_map.dtype),
            torch.arange(W, device=device, dtype=depth_map.dtype),
            indexing="ij",
        )

        if mask is not None:
            v_idx = v_idx[mask]
            u_idx = u_idx[mask]
            d = depth_map[mask]
        else:
            v_idx = v_idx.reshape(-1)
            u_idx = u_idx.reshape(-1)
            d = depth_map.reshape(-1)

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x_cam = (u_idx - cx) / fx * d
        y_cam = (v_idx - cy) / fy * d
        z_cam = d

        cam_pts = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(d)], dim=-1)
        world_pts = (c2w @ cam_pts.T).T[:, :3]
        return world_pts

    def _init_anchors_from_scale_frames(
        self,
        depth: torch.Tensor,        # [1, S, H, W, 1]
        depth_conf: torch.Tensor,    # [1, S, H, W]
        dense_features: torch.Tensor, # [S, C, H, W]
        c2w: torch.Tensor,           # [1, S, 4, 4]
        intrinsics: torch.Tensor,     # [1, S, 3, 3]
        n_scale: int,
    ):
        """Phase 1: Create initial anchors from scale frame depth maps.

        Subsamples pixels (stride) to avoid creating hundreds of thousands of
        candidate points from high-res depth maps.
        """
        _, S, H, W, _ = depth.shape
        device = depth.device

        # Subsample grid for efficiency (e.g. every 4th pixel)
        stride = max(1, self.patch_size // 2)
        v_grid = torch.arange(0, H, stride, device=device)
        u_grid = torch.arange(0, W, stride, device=device)
        vv, uu = torch.meshgrid(v_grid, u_grid, indexing="ij")
        vv = vv.reshape(-1)
        uu = uu.reshape(-1)

        all_pts = []
        all_feats = []
        all_confs = []
        all_frames = []

        for s in range(n_scale):
            d_full = depth[0, s, :, :, 0]    # [H, W]
            dc_full = depth_conf[0, s]        # [H, W]
            feat = dense_features[s]           # [C, H, W]
            cam = c2w[0, s]                    # [4, 4]
            intr = intrinsics[0, s]            # [3, 3]

            # Sample at subsampled grid
            d = d_full[vv, uu]
            dc = dc_full[vv, uu]
            valid = (d > 0.01) & (dc > self._anchor_min_depth_conf)
            if not valid.any():
                continue

            d_v = d[valid]
            v_sel = vv[valid]
            u_sel = uu[valid]

            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            x_cam = (u_sel.float() - cx) / fx * d_v
            y_cam = (v_sel.float() - cy) / fy * d_v
            cam_pts = torch.stack([x_cam, y_cam, d_v, torch.ones_like(d_v)], dim=-1)
            pts = (cam @ cam_pts.T).T[:, :3]

            # Sample features
            pixel_coords = torch.stack([u_sel.float(), v_sel.float()], dim=-1)
            feats = self.anchor_manager._sample_features(feat, pixel_coords, H, W)
            confs = dc[valid]

            all_pts.append(pts)
            all_feats.append(feats)
            all_confs.append(confs)
            all_frames.append(torch.full((len(pts),), s, dtype=torch.long, device=device))

        if all_pts:
            self.anchor_manager.init_from_depth(
                world_points=torch.cat(all_pts),
                features=torch.cat(all_feats),
                confidences=torch.cat(all_confs),
                frame_indices=torch.cat(all_frames),
            )
            n_anchors = self.anchor_manager.anchors.num_anchors if self.anchor_manager.anchors else 0
            logger.info("Initialized %d anchors from %d scale frames", n_anchors, n_scale)

    def _decode_all_anchors(
        self,
        focal_length: float,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Run AnchorGaussianDecoder on all current anchors."""
        if self.anchor_manager.anchors is None or self.anchor_manager.anchors.num_anchors == 0:
            return None

        anchors = self.anchor_manager.anchors
        anchor_depths = anchors.positions.norm(dim=-1)

        with torch.amp.autocast("cuda", enabled=False):
            return self.anchor_decoder(
                anchor_positions=anchors.positions.float(),
                anchor_features=anchors.features.float(),
                anchor_depths=anchor_depths.float(),
                focal_length=focal_length,
                patch_size=self.patch_size,
            )

    # ------------------------------------------------------------------
    # Streaming inference with anchor-aligned Gaussian construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference_streaming_anchor_gs(
        self,
        images: torch.Tensor,
        num_scale_frames: Optional[int] = None,
        keyframe_interval: int = 1,
        output_device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Streaming inference that builds an anchor-aligned Gaussian map.

        Two-phase protocol:
        Phase 1 (scale frames): Bidirectional attention -> depth + token features -> init anchors
        Phase 2 (causal streaming): Per-frame anchor update + expansion

        Returns dict with pose_enc, depth, anchor_manager, anchor_stats, gaussians.
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, S, C, H, W = images.shape
        assert B == 1, "Anchor GS streaming supports batch_size=1 only"

        n_scale = num_scale_frames or self.num_frame_for_scale
        n_scale = min(n_scale, S)
        dev = next(self.parameters()).device

        self.anchor_manager = self._create_anchor_manager(dev)
        self.clean_kv_cache()

        def _out(t: torch.Tensor) -> torch.Tensor:
            return t.to(output_device) if output_device is not None else t

        # ==============================================================
        # Phase 1 — scale frames (bidirectional attention)
        # ==============================================================
        logger.info("Phase 1: %d scale frames", n_scale)
        scale_imgs = images[:, :n_scale].to(dev, non_blocking=True)
        torch.compiler.cudagraph_mark_step_begin()

        agg, ps = self._aggregate_features(
            scale_imgs,
            num_frame_for_scale=n_scale,
            num_frame_per_block=n_scale,
        )

        preds: Dict[str, torch.Tensor] = {}
        preds.update(self._predict_camera(
            agg, causal_inference=True,
            num_frame_for_scale=n_scale, num_frame_per_block=n_scale,
        ))
        preds.update(self._predict_depth(agg, scale_imgs, ps))

        # Lightweight feature extraction from tokens (no extra DPT)
        dense_features = self._tokens_to_feature_map(agg, ps, H, W, n_scale)

        c2w, intrinsics = self._pose_enc_to_c2w_and_intrinsics(preds["pose_enc"], H, W)

        self._init_anchors_from_scale_frames(
            preds["depth"], preds["depth_conf"],
            dense_features, c2w, intrinsics, n_scale,
        )

        avg_focal = float((intrinsics[0, :, 0, 0].mean() + intrinsics[0, :, 1, 1].mean()) / 2)

        all_pose = [_out(preds["pose_enc"])]
        all_depth = [_out(preds["depth"])]
        all_dconf = [_out(preds["depth_conf"])] if "depth_conf" in preds else []
        anchor_stats = [self.anchor_manager.get_stats()]
        del preds, agg, dense_features

        # ==============================================================
        # Phase 2 — causal streaming, one frame at a time
        # ==============================================================
        pbar = tqdm(
            range(n_scale, S),
            desc="Streaming Anchor-GS",
            initial=n_scale,
            total=S,
        )
        for i in pbar:
            frame = images[:, i : i + 1].to(dev, non_blocking=True)
            is_kf = (keyframe_interval <= 1) or ((i - n_scale) % keyframe_interval == 0)

            if not is_kf:
                self._set_skip_append(True)

            torch.compiler.cudagraph_mark_step_begin()
            agg, ps = self._aggregate_features(
                frame, num_frame_for_scale=n_scale, num_frame_per_block=1,
            )
            fp: Dict[str, torch.Tensor] = {}
            fp.update(self._predict_camera(
                agg, causal_inference=True,
                num_frame_for_scale=n_scale, num_frame_per_block=1,
            ))
            fp.update(self._predict_depth(agg, frame, ps))

            if is_kf:
                frame_feat = self._tokens_to_feature_map(agg, ps, H, W, 1)

                frame_c2w, frame_intr = self._pose_enc_to_c2w_and_intrinsics(
                    fp["pose_enc"], H, W,
                )

                depth_2d = fp["depth"][0, 0, :, :, 0]
                dconf_2d = fp["depth_conf"][0, 0]

                self.anchor_manager.update_and_expand(
                    c2w=frame_c2w[0, 0],
                    intrinsics=frame_intr[0, 0],
                    H=H, W=W,
                    depth_map=depth_2d,
                    depth_conf=dconf_2d,
                    dense_features=frame_feat[0],
                    frame_idx=i,
                )

                avg_focal = float(
                    (frame_intr[0, 0, 0, 0] + frame_intr[0, 0, 1, 1]) / 2
                )

            if not is_kf:
                self._set_skip_append(False)

            all_pose.append(_out(fp["pose_enc"]))
            all_depth.append(_out(fp["depth"]))
            if "depth_conf" in fp:
                all_dconf.append(_out(fp["depth_conf"]))

            stats = self.anchor_manager.get_stats()
            anchor_stats.append(stats)
            pbar.set_postfix(
                anchors=stats["num_anchors"],
                obs=f"{stats['avg_obs']:.1f}",
            )
            del fp, agg

        # ==============================================================
        # Final Gaussian decoding from all anchors
        # ==============================================================
        gaussians = self._decode_all_anchors(avg_focal)

        # ==============================================================
        # Collect outputs
        # ==============================================================
        result: Dict[str, Any] = {
            "pose_enc": torch.cat(all_pose, dim=1),
            "depth": torch.cat(all_depth, dim=1),
            "anchor_manager": self.anchor_manager,
            "anchor_stats": anchor_stats,
            "images": images,
        }
        if all_dconf:
            result["depth_conf"] = torch.cat(all_dconf, dim=1)
        if gaussians is not None:
            result["gaussians"] = gaussians
        return result
