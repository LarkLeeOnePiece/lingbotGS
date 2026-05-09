"""
GCTStreamGS - Streaming GCT with compact Gaussian Splatting output.

Extends GCTStream with:
- CompactGaussianHead for patch-level 2D surfel prediction (~999/frame)
- GaussianMapManager for three-level map (anchor/window/memory)
- Streaming Gaussian map construction during inference

The Gaussian head is designed to be trained on top of a frozen or fine-tuned
LingBot-Map checkpoint using differentiable rendering losses (gsplat).
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from tqdm.auto import tqdm

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.heads.gaussian_head import CompactGaussianHead
from lingbot_map.mapping.gaussian_map import GaussianMapManager

logger = logging.getLogger(__name__)


class GCTStreamGS(GCTStream):
    """Streaming GCT with compact Gaussian map construction.

    Inherits the full LingBot-Map pipeline (backbone, GCA, camera head,
    depth head) and adds a lightweight CompactGaussianHead (~2 M params)
    that predicts one 2D surfel per patch token.

    During streaming inference the GaussianMapManager accumulates surfels
    into a bounded three-level map that mirrors GCA's attention structure.

    Extra constructor args (on top of GCTStream):
        gaussian_hidden_dim:       Hidden dim for the attribute MLP.
        gaussian_memory_budget:    Hard cap on memory-pool Gaussians.
        gaussian_voxel_size:       Voxel edge (m) for compaction hashing.
        gaussian_opacity_threshold: Min opacity to materialise a surfel.
    """

    def __init__(
        self,
        *,
        gaussian_hidden_dim: int = 512,
        gaussians_per_patch: int = 4,
        gaussian_memory_budget: int = 100_000,
        gaussian_voxel_size: float = 0.05,
        gaussian_opacity_threshold: float = 0.1,
        **kwargs,
    ):
        # Store GS config before super().__init__ (which calls _build_*)
        self._gs_hidden_dim = gaussian_hidden_dim
        self._gs_gaussians_per_patch = gaussians_per_patch
        self._gs_memory_budget = gaussian_memory_budget
        self._gs_voxel_size = gaussian_voxel_size
        self._gs_opacity_threshold = gaussian_opacity_threshold

        super().__init__(**kwargs)

        # ------ Gaussian head (new, randomly initialised) ------
        self.gaussian_head = CompactGaussianHead(
            dim_in=2 * self.embed_dim,
            hidden_dim=gaussian_hidden_dim,
            patch_size=self.patch_size,
            gaussians_per_patch=gaussians_per_patch,
        )

        # Created per-sequence in inference; not a module parameter
        self.map_manager: Optional[GaussianMapManager] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_lingbot(
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cuda"),
        **gs_kwargs,
    ) -> "GCTStreamGS":
        """Load a LingBot-Map checkpoint and attach a fresh Gaussian head.

        Args:
            checkpoint_path: Path to a ``lingbot-map-long.pt`` checkpoint.
            device: Target device.
            **gs_kwargs: Extra keyword args forwarded to the constructor
                (gaussian_hidden_dim, gaussian_memory_budget, …).

        Returns:
            GCTStreamGS model with backbone/GCA/heads loaded and
            Gaussian head randomly initialised.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract constructor kwargs stored alongside the checkpoint
        if "config" in ckpt:
            init_kwargs = ckpt["config"]
            state_dict = ckpt["model"]
        else:
            # Flat checkpoint (state-dict only) — use sensible defaults
            state_dict = ckpt
            init_kwargs = {}

        # Merge GS-specific kwargs
        init_kwargs.update(gs_kwargs)

        model = cls(**init_kwargs)
        # Load only the keys that exist in both
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        gs_keys = [k for k in missing if "gaussian_head" in k]
        other_missing = [k for k in missing if "gaussian_head" not in k]
        if other_missing:
            logger.warning("Missing non-GS keys: %s", other_missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)
        logger.info(
            "Loaded checkpoint. %d GS-head params randomly initialised.",
            len(gs_keys),
        )
        return model.to(device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_map_manager(self, device: torch.device) -> GaussianMapManager:
        return GaussianMapManager(
            num_anchor_frames=self.kv_cache_scale_frames,
            window_size=self.kv_cache_sliding_window,
            memory_budget=self._gs_memory_budget,
            voxel_size=self._gs_voxel_size,
            opacity_threshold=self._gs_opacity_threshold,
            device=device,
        )

    def _predict_gaussians(
        self,
        aggregated_tokens_list: list,
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
        patch_start_idx: int,
        image_size_hw: tuple,
        input_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the Gaussian head in fp32."""
        tokens_fp32 = [t.float() for t in aggregated_tokens_list]
        with torch.amp.autocast("cuda", enabled=False):
            return self.gaussian_head(
                tokens_fp32,
                depth=depth.float(),
                pose_enc=pose_enc.float(),
                patch_start_idx=patch_start_idx,
                image_size_hw=image_size_hw,
                input_image=input_image.float() if input_image is not None else None,
            )

    def _add_gs_to_map(
        self, frame_idx: int, gs: Dict[str, torch.Tensor], batch: int = 0, seq: int = 0
    ):
        """Extract tensors from the head output and push them to the map."""
        self.map_manager.add_frame(
            frame_idx=frame_idx,
            positions=gs["positions"][batch, seq],
            opacities=gs["opacities"][batch, seq],
            scales=gs["scales"][batch, seq],
            normals=gs["normals"][batch, seq],
            colors=gs["colors"][batch, seq],
        )

    # ------------------------------------------------------------------
    # Streaming inference with Gaussian map construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference_streaming_gs(
        self,
        images: torch.Tensor,
        num_scale_frames: Optional[int] = None,
        keyframe_interval: int = 1,
        output_device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Streaming inference that also builds a compact Gaussian map.

        Follows the same two-phase protocol as ``inference_streaming``
        (scale-frame init → causal frame-by-frame) but additionally:
        1. Runs the CompactGaussianHead after each forward pass.
        2. Pushes resulting surfels into the GaussianMapManager.

        Args:
            images: [S, 3, H, W] or [B, S, 3, H, W], values in [0, 1].
            num_scale_frames: Override for number of anchor frames.
            keyframe_interval: Only keyframes persist in the KV cache *and*
                contribute Gaussians to the map.
            output_device: Offload per-frame predictions here (e.g. CPU).

        Returns:
            Dict with standard predictions plus:
                gaussian_map   — the GaussianMapManager instance
                gaussian_stats — list of per-frame map statistics dicts
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, S, C, H, W = images.shape
        assert B == 1, "Gaussian streaming supports batch_size=1 only"

        n_scale = num_scale_frames or self.num_frame_for_scale
        n_scale = min(n_scale, S)
        dev = next(self.parameters()).device

        self.map_manager = self._create_map_manager(dev)
        self.clean_kv_cache()

        # Denormalize images for color sampling in the GS head
        _MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).reshape(1, 1, 3, 1, 1)
        _STD = torch.tensor([0.229, 0.224, 0.225], device=dev).reshape(1, 1, 3, 1, 1)
        images_denorm = (images.to(dev) * _STD + _MEAN).clamp(0, 1)

        def _out(t: torch.Tensor) -> torch.Tensor:
            return t.to(output_device) if output_device is not None else t

        # ==============================================================
        # Phase 1 — scale frames (bidirectional attention among them)
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

        gs = self._predict_gaussians(agg, preds["depth"], preds["pose_enc"], ps, (H, W),
                                      input_image=images_denorm[:, :n_scale])
        for si in range(n_scale):
            self._add_gs_to_map(si, gs, seq=si)

        all_pose = [_out(preds["pose_enc"])]
        all_depth = [_out(preds["depth"])]
        all_dconf = [_out(preds["depth_conf"])] if "depth_conf" in preds else []
        gs_stats = [self.map_manager.get_stats()]
        del preds, gs, agg

        # ==============================================================
        # Phase 2 — causal streaming, one frame at a time
        # ==============================================================
        pbar = tqdm(
            range(n_scale, S),
            desc="Streaming GS",
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

            gs = self._predict_gaussians(agg, fp["depth"], fp["pose_enc"], ps, (H, W),
                                          input_image=images_denorm[:, i:i+1])

            if is_kf:
                self._add_gs_to_map(i, gs)

            if not is_kf:
                self._set_skip_append(False)

            all_pose.append(_out(fp["pose_enc"]))
            all_depth.append(_out(fp["depth"]))
            if "depth_conf" in fp:
                all_dconf.append(_out(fp["depth_conf"]))

            stats = self.map_manager.get_stats()
            gs_stats.append(stats)
            pbar.set_postfix(
                gs=stats["total_gaussians"], mem=stats["memory_gaussians"],
            )
            del fp, gs, agg

        # ==============================================================
        # Collect outputs
        # ==============================================================
        result: Dict[str, Any] = {
            "pose_enc": torch.cat(all_pose, dim=1),
            "depth": torch.cat(all_depth, dim=1),
            "gaussian_map": self.map_manager,
            "gaussian_stats": gs_stats,
            "images": images,
        }
        if all_dconf:
            result["depth_conf"] = torch.cat(all_dconf, dim=1)
        return result
