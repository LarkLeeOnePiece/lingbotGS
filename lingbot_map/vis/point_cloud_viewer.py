# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Interactive 3D Point Cloud Viewer using Viser.

This module provides the PointCloudViewer class for visualizing 3D reconstruction results,
including point clouds, camera poses, and animated playback.
"""

import os
import time
import threading
import subprocess
import tempfile
import shutil
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import cv2
import matplotlib.cm as cm
from tqdm.auto import tqdm

import viser
import viser.transforms as tf

from lingbot_map.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from lingbot_map.vis.utils import CameraState
from lingbot_map.vis.sky_segmentation import apply_sky_segmentation

try:
    import moderngl
    _HAS_MODERNGL = True
except ImportError:
    _HAS_MODERNGL = False


class _GLPointRenderer:
    """OpenGL offscreen point cloud renderer using moderngl.

    Uses hardware rasterization + z-test for ~200x speedup over PyTorch z-buffer.
    Renders color + point index in a single pass via MRT (Multiple Render Targets).
    """

    _VERT = """
    #version 330
    in vec3 in_pos;
    in vec3 in_color;
    in float in_id;
    uniform mat4 mvp;
    uniform float point_size;
    out vec3 v_color;
    flat out float v_id;
    void main() {
        gl_Position = mvp * vec4(in_pos, 1.0);
        gl_PointSize = point_size;
        v_color = in_color;
        v_id = in_id;
    }
    """

    _FRAG = """
    #version 330
    in vec3 v_color;
    flat in float v_id;
    layout(location=0) out vec4 frag_color;
    layout(location=1) out vec4 frag_id;
    void main() {
        frag_color = vec4(v_color, 1.0);
        frag_id = vec4(v_id, 0.0, 0.0, 1.0);
    }
    """

    def __init__(self):
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(
            vertex_shader=self._VERT, fragment_shader=self._FRAG
        )
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self._fbo = None
        self._fbo_size = None
        self._vbo = None
        self._vao = None
        self._vbo_pos = None
        self._vbo_col = None
        self._vbo_id = None

    def _ensure_fbo(self, W, H):
        if self._fbo_size == (W, H):
            return
        if self._fbo is not None:
            self._fbo.release()
            self._fbo_color.release()
            self._fbo_id.release()
            self._fbo_depth.release()
        self._fbo_color = self.ctx.texture((W, H), 4, dtype='f4')
        self._fbo_id = self.ctx.texture((W, H), 4, dtype='f4')
        self._fbo_depth = self.ctx.depth_renderbuffer((W, H))
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self._fbo_color, self._fbo_id],
            depth_attachment=self._fbo_depth,
        )
        self._fbo_size = (W, H)

    def upload_points(self, pts, cols, point_indices):
        """Upload point cloud using separate VBOs (no interleaving needed).

        Args:
            pts: (N, 3) float32 numpy — contiguous
            cols: (N, 3) float32 numpy — contiguous
            point_indices: (N,) int64 numpy — global point IDs
        """
        N = len(pts)
        self._full_N = N
        if N == 0:
            return

        pts_f32 = np.ascontiguousarray(pts, dtype=np.float32)
        cols_f32 = np.ascontiguousarray(cols, dtype=np.float32)
        ids_f32 = np.ascontiguousarray(point_indices.astype(np.float32))

        pos_bytes = pts_f32.nbytes
        col_bytes = cols_f32.nbytes
        id_bytes = ids_f32.nbytes

        def _ensure_buf(old_buf, size, data):
            if old_buf is not None and old_buf.size >= size:
                old_buf.write(data)
                return old_buf
            if old_buf is not None:
                old_buf.release()
            buf = self.ctx.buffer(reserve=int(size * 1.2))
            buf.write(data)
            return buf

        self._vbo_pos = _ensure_buf(self._vbo_pos, pos_bytes, pts_f32)
        self._vbo_col = _ensure_buf(self._vbo_col, col_bytes, cols_f32)
        self._vbo_id = _ensure_buf(self._vbo_id, id_bytes, ids_f32)

        # Recreate VAO (lightweight, just binds existing buffers)
        if self._vao is not None:
            self._vao.release()
        self._vao = self.ctx.vertex_array(
            self.prog,
            [
                (self._vbo_pos, '3f', 'in_pos'),
                (self._vbo_col, '3f', 'in_color'),
                (self._vbo_id, '1f', 'in_id'),
            ],
        )

    def kill_points(self, indices):
        """Move points behind camera (z=-1e6) so OpenGL clips them.

        Uses batched writes for contiguous runs, falls back to individual
        writes for scattered indices. Typically ~0.1ms for 1000 points.
        """
        if self._vbo_pos is None or len(indices) == 0:
            return
        indices = np.asarray(indices, dtype=np.int64)
        indices.sort()
        dead_pos = np.array([0.0, 0.0, -1e6], dtype=np.float32)

        # Find contiguous runs for batched writes
        if len(indices) == 0:
            return
        breaks = np.where(np.diff(indices) != 1)[0] + 1
        runs = np.split(indices, breaks)

        for run in runs:
            start = int(run[0])
            count = len(run)
            dead_block = np.tile(dead_pos, count).astype(np.float32)
            self._vbo_pos.write(dead_block.tobytes(), offset=start * 12)

    def render_view(self, R_w2c, t_w2c, K, H, W, point_size=1.0,
                    num_points=None, first=0):
        """Render from a viewpoint using pre-uploaded points.

        Args:
            num_points: Render N points starting from `first`.
                        If None, render all uploaded points.
            first: Index of first point to render (default 0).
        """
        N = num_points if num_points is not None else self._full_N
        vertices = N - first
        if vertices <= 0:
            return (np.zeros((H, W, 3), dtype=np.float32),
                    np.zeros((H, W), dtype=bool),
                    np.full((H, W), -1, dtype=np.int64))

        self._ensure_fbo(W, H)
        mvp = self._build_mvp(R_w2c, t_w2c, K, H, W)

        self.prog['mvp'].write(mvp.T.tobytes())
        self.prog['point_size'].value = point_size

        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._vao.render(moderngl.POINTS, vertices=vertices, first=first)

        return self._read_fbo(H, W)

    def render(self, pts, cols, point_indices, R_w2c, t_w2c, K, H, W, point_size=1.0):
        """One-shot render: upload + render. Use upload_points + render_view for batches."""
        N = len(pts)
        if N == 0:
            return (np.zeros((H, W, 3), dtype=np.float32),
                    np.zeros((H, W), dtype=bool),
                    np.full((H, W), -1, dtype=np.int64))

        self.upload_points(pts, cols, point_indices)
        return self.render_view(R_w2c, t_w2c, K, H, W, point_size=point_size)

    @staticmethod
    def _build_mvp(R_w2c, t_w2c, K, H, W):
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        near, far = 0.01, 1000.0

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2 * fx / W
        proj[1, 1] = 2 * fy / H
        proj[0, 2] = 1.0 - 2 * cx / W
        proj[1, 2] = 2 * cy / H - 1.0
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1.0

        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R_w2c.astype(np.float32)
        view[:3, 3] = t_w2c.astype(np.float32)
        flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        view = flip @ view

        return (proj @ view).astype(np.float32)

    def _read_fbo(self, H, W):
        raw_color = self._fbo_color.read()
        raw_id = self._fbo_id.read()

        # Color is now f4 (float32) — no quantization
        color_img = np.frombuffer(raw_color, dtype=np.float32).reshape(H, W, 4)
        color_img = color_img[::-1].copy()
        rendered = color_img[:, :, :3]
        coverage = color_img[:, :, 3] > 0

        id_img = np.frombuffer(raw_id, dtype=np.float32).reshape(H, W, 4)
        id_img = id_img[::-1].copy()
        index_buf = np.round(id_img[:, :, 0]).astype(np.int64)
        index_buf[~coverage] = -1

        return (rendered, coverage, index_buf)

    def release(self):
        if self._vao is not None:
            self._vao.release()
        for buf in (self._vbo, self._vbo_pos, self._vbo_col, self._vbo_id):
            if buf is not None:
                buf.release()
        if self._fbo is not None:
            self._fbo.release()
            self._fbo_color.release()
            self._fbo_id.release()
            self._fbo_depth.release()
        self.ctx.release()


# Thread-local GL renderer (each thread gets its own OpenGL context)
import threading as _threading
_gl_thread_local = _threading.local()


def _get_gl_renderer():
    if not hasattr(_gl_thread_local, 'renderer'):
        _gl_thread_local.renderer = _GLPointRenderer()
    return _gl_thread_local.renderer


class PointCloudViewer:
    """
    Interactive 3D point cloud viewer with camera visualization.

    Features:
    - Point cloud visualization with confidence-based filtering
    - Camera frustum visualization with gradient colors
    - Frame-by-frame playback animation (3D/4D modes)
    - Range-based and recent-N-frames visualization modes
    - Video export with FFmpeg

    Args:
        model: Optional model for interactive inference
        state_args: Optional state arguments
        pc_list: List of point clouds per frame
        color_list: List of colors per frame
        conf_list: List of confidence scores per frame
        cam_dict: Camera dictionary with focal, pp, R, t
        image_mask: Optional image mask
        edge_color_list: Optional edge colors
        device: Device for computation
        port: Viser server port
        show_camera: Whether to show camera frustums
        vis_threshold: Visibility threshold for filtering
        size: Image size
        downsample_factor: Point cloud downsample factor
        point_size: Initial point size
        pred_dict: Prediction dictionary (alternative to pc_list/color_list/conf_list)
        init_conf_threshold: Initial confidence threshold percentage
        use_point_map: Use point map instead of depth-based points
        mask_sky: Apply sky segmentation
        image_folder: Path to image folder (for sky segmentation)
    """

    def __init__(
        self,
        model=None,
        state_args=None,
        pc_list=None,
        color_list=None,
        conf_list=None,
        cam_dict=None,
        image_mask=None,
        edge_color_list=None,
        device: str = "cpu",
        port: int = 8080,
        show_camera: bool = True,
        vis_threshold: float = 1.0,
        size: int = 512,
        downsample_factor: int = 10,
        point_size: float = 0.00001,
        pred_dict: Optional[Dict] = None,
        init_conf_threshold: float = 50.0,
        use_point_map: bool = False,
        mask_sky: bool = False,
        image_folder: Optional[str] = None,
        sky_mask_dir: Optional[str] = None,
        sky_mask_visualization_dir: Optional[str] = None,
        depth_stride: int = 1,
    ):
        self.model = model
        self.size = size
        self.state_args = state_args
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self.device = device
        self.conf_list = conf_list
        self.vis_threshold = vis_threshold
        self.point_size = point_size
        self.tt = lambda x: torch.from_numpy(x).float().to(device)

        # Process the prediction dictionary to create pc_list, color_list, conf_list
        if pred_dict is not None:
            pc_list, color_list, conf_list, cam_dict = self._process_pred_dict(
                pred_dict, use_point_map, mask_sky, image_folder,
                sky_mask_dir=sky_mask_dir,
                sky_mask_visualization_dir=sky_mask_visualization_dir,
                depth_stride=depth_stride,
            )
        else:
            self.original_images = []

        self.pcs, self.all_steps = self.read_data(
            pc_list, color_list, conf_list, edge_color_list
        )
        self.cam_dict = cam_dict
        self.num_frames = len(self.all_steps)
        self.image_mask = image_mask
        self.show_camera = show_camera
        self.on_replay = False
        self.vis_pts_list = []
        self.traj_list = []
        self.orig_img_list = [x[0] for x in color_list if len(x) > 0] if color_list else []
        self.via_points = []

        self._setup_gui()
        self.server.on_client_connect(self._connect_client)

    def _process_pred_dict(
        self,
        pred_dict: Dict,
        use_point_map: bool,
        mask_sky: bool,
        image_folder: Optional[str],
        sky_mask_dir: Optional[str] = None,
        sky_mask_visualization_dir: Optional[str] = None,
        depth_stride: int = 1,
    ) -> Tuple[List, List, List, Dict]:
        """Process prediction dictionary to extract visualization data.

        Args:
            pred_dict: Model prediction dictionary.
            use_point_map: Use point map instead of depth-based projection.
            mask_sky: Apply sky segmentation to filter sky points.
            image_folder: Path to images for sky segmentation.
            sky_mask_dir: Directory for cached sky masks.
            sky_mask_visualization_dir: Directory for sky mask visualization images.
            depth_stride: Only project depth to point cloud every N frames.
                Frames not projected will have empty point clouds but still
                show camera frustums and images. 1 = every frame (default).
        """
        images = pred_dict["images"]  # (S, 3, H, W)

        depth_map = pred_dict.get("depth")  # (S, H, W, 1)
        depth_conf = pred_dict.get("depth_conf")  # (S, H, W)

        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

        # Compute world points from depth if not using the precomputed point map
        if not use_point_map:
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
            conf = depth_conf
        else:
            world_points = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict.get("world_points_conf", depth_conf)  # (S, H, W)

        # Apply sky segmentation if enabled
        if mask_sky:
            conf = apply_sky_segmentation(
                conf, image_folder=image_folder, images=images,
                sky_mask_dir=sky_mask_dir,
                sky_mask_visualization_dir=sky_mask_visualization_dir,
            )

        # Convert images from (S, 3, H, W) to (S, H, W, 3)
        colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
        S = world_points.shape[0]

        # Store original images for camera frustum display
        self.original_images = []
        for i in range(S):
            img = images[i]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            self.original_images.append(img)

        # Create lists - apply depth_stride to skip frames for point projection
        H, W = world_points.shape[1], world_points.shape[2]
        pc_list = []
        color_list = []
        conf_list = []
        skipped = 0
        for i in range(S):
            if depth_stride > 1 and i % depth_stride != 0:
                # Empty point cloud for skipped frames
                pc_list.append(np.zeros((0, 0, 3), dtype=np.float32))
                color_list.append(np.zeros((0, 0, 3), dtype=np.float32))
                conf_list.append(np.zeros((0, 0), dtype=np.float32))
                skipped += 1
            else:
                pc_list.append(world_points[i])
                color_list.append(colors[i])
                if conf is not None:
                    conf_list.append(conf[i])
                else:
                    conf_list.append(np.ones(world_points[i].shape[:2], dtype=np.float32))

        if depth_stride > 1:
            print(f'  depth_stride={depth_stride}: projecting {S - skipped}/{S} frames, skipping {skipped}')

        # Create camera dictionary (all frames keep cameras)
        cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
        cam_dict = {
            "focal": [intrinsics_cam[i, 0, 0] for i in range(S)],
            "pp": [(intrinsics_cam[i, 0, 2], intrinsics_cam[i, 1, 2]) for i in range(S)],
            "R": [cam_to_world_mat[i, :3, :3] for i in range(S)],
            "t": [cam_to_world_mat[i, :3, 3] for i in range(S)],
        }

        return pc_list, color_list, conf_list, cam_dict

    def _compute_scene_center_and_scale(self) -> Tuple[np.ndarray, float]:
        """Compute scene center and scale from camera positions and point clouds.

        Returns:
            Tuple of (center as 3D array, scale as float distance).
        """
        # Use camera positions as primary reference (more reliable than noisy points)
        if self.cam_dict is not None and "t" in self.cam_dict:
            cam_positions = np.array([self.cam_dict["t"][s] for s in self.all_steps])
            center = np.mean(cam_positions, axis=0)
            if len(cam_positions) > 1:
                extent = np.ptp(cam_positions, axis=0)  # range per axis
                scale = np.linalg.norm(extent)
            else:
                scale = 1.0
        else:
            # Fallback: use point cloud data
            all_pts = []
            for step in self.all_steps:
                pc = self.pcs[step]["pc"].reshape(-1, 3)
                # subsample for speed
                if len(pc) > 1000:
                    pc = pc[::len(pc) // 1000]
                all_pts.append(pc)
            all_pts = np.concatenate(all_pts, axis=0)
            center = np.median(all_pts, axis=0)
            extent = np.percentile(all_pts, 95, axis=0) - np.percentile(all_pts, 5, axis=0)
            scale = np.linalg.norm(extent)

        return center, max(scale, 0.1)

    def _reset_view_to_direction(
        self,
        direction: np.ndarray,
        up: np.ndarray = np.array([0.0, -1.0, 0.0]),
        distance_scale: float = 1.5,
        smooth: bool = True,
    ):
        """Reset the viewer camera to look at scene center from a given direction.

        Args:
            direction: Unit vector pointing FROM the scene center TO the camera.
            up: Up vector for the camera.
            distance_scale: Multiplier on scene scale for camera distance.
            smooth: Whether to smoothly transition.
        """
        center, scale = self._compute_scene_center_and_scale()
        distance = scale * distance_scale
        position = center + direction * distance

        for client in self.server.get_clients().values():
            if smooth:
                self._smooth_camera_transition(
                    client,
                    target_position=position,
                    target_look_at=center,
                    target_up=up,
                    duration=0.4,
                )
            else:
                client.camera.up_direction = tuple(up)
                client.camera.position = tuple(position)
                client.camera.look_at = tuple(center)

    def _setup_gui(self):
        """Setup GUI controls."""
        gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )

        # Video frame display controls — kept at top so the current frame is always visible
        with self.server.gui.add_folder("Video Display"):
            self.show_video_checkbox = self.server.gui.add_checkbox("Show Current Frame", initial_value=True)
            if hasattr(self, 'original_images') and len(self.original_images) > 0:
                self.current_frame_image = self.server.gui.add_image(
                    self.original_images[0], label="Current Frame"
                )
            else:
                self.current_frame_image = None

        # Preset view direction buttons
        with self.server.gui.add_folder("Reset View Direction"):
            btn_look_at_center = self.server.gui.add_button(
                "Look At Scene Center",
                hint="Reset orbit center to the scene center (fixes orbit after dragging).",
            )
            btn_overview = self.server.gui.add_button(
                "Overview",
                hint="Reset to a 3/4 overview of the scene.",
            )
            btn_front = self.server.gui.add_button(
                "Front (+Z)",
                hint="View scene from the front.",
            )
            btn_back = self.server.gui.add_button(
                "Back (-Z)",
                hint="View scene from the back.",
            )
            btn_top = self.server.gui.add_button(
                "Top (-Y)",
                hint="View scene from above (bird's eye).",
            )
            btn_left = self.server.gui.add_button(
                "Left (-X)",
                hint="View scene from the left.",
            )
            btn_right = self.server.gui.add_button(
                "Right (+X)",
                hint="View scene from the right.",
            )
            btn_first_cam = self.server.gui.add_button(
                "First Camera",
                hint="Reset to the first camera's viewpoint.",
            )

        @btn_look_at_center.on_click
        def _(_) -> None:
            center, _ = self._compute_scene_center_and_scale()
            for client in self.server.get_clients().values():
                client.camera.look_at = tuple(center)

        @btn_overview.on_click
        def _(_) -> None:
            d = np.array([0.5, -0.6, 0.6])
            self._reset_view_to_direction(d / np.linalg.norm(d))

        @btn_front.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([0.0, 0.0, 1.0]))

        @btn_back.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([0.0, 0.0, -1.0]))

        @btn_top.on_click
        def _(_) -> None:
            self._reset_view_to_direction(
                np.array([0.0, -1.0, 0.0]),
                up=np.array([0.0, 0.0, 1.0]),
            )

        @btn_left.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([-1.0, 0.0, 0.0]))

        @btn_right.on_click
        def _(_) -> None:
            self._reset_view_to_direction(np.array([1.0, 0.0, 0.0]))

        @btn_first_cam.on_click
        def _(_) -> None:
            self._move_to_camera(0, smooth=True)

        button3 = self.server.gui.add_button("4D (Only Show Current Frame)")
        button4 = self.server.gui.add_button("3D (Show All Frames)")
        self.is_render = False
        self.fourd = False

        @button3.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = True

        @button4.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = False

        self.focal_slider = self.server.gui.add_slider(
            "Focal Length", min=0.1, max=99999, step=1, initial_value=533
        )
        self.psize_slider = self.server.gui.add_slider(
            "Point Size", min=0.00001, max=0.1, step=0.00001, initial_value=self.point_size
        )
        self.camsize_slider = self.server.gui.add_slider(
            "Camera Size", min=0.01, max=0.5, step=0.01, initial_value=0.1
        )
        self.downsample_slider = self.server.gui.add_slider(
            "Downsample Factor", min=1, max=1000, step=1, initial_value=10
        )
        self.show_camera_checkbox = self.server.gui.add_checkbox(
            "Show Camera", initial_value=self.show_camera
        )
        self.vis_threshold_slider = self.server.gui.add_slider(
            "Visibility Threshold", min=1.0, max=5.0, step=0.01,
            initial_value=self.vis_threshold,
        )
        self.camera_downsample_slider = self.server.gui.add_slider(
            "Camera Downsample Factor", min=1, max=50, step=1, initial_value=1
        )

        # Screenshot controls
        with self.server.gui.add_folder("Screenshot"):
            self.screenshot_button = self.server.gui.add_button("Take Screenshot")
            self.screenshot_resolution = self.server.gui.add_dropdown(
                "Resolution",
                options=["1920x1080", "2560x1440", "3840x2160", "Current"],
                initial_value="1920x1080",
            )
            self.screenshot_path = self.server.gui.add_text(
                "Save Path", initial_value="screenshot.png"
            )
            self.screenshot_status = self.server.gui.add_text(
                "Status", initial_value="Ready"
            )

        @self.screenshot_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self._take_screenshot(event.client)

        # GLB export controls
        with self.server.gui.add_folder("Export GLB"):
            self.glb_output_path = self.server.gui.add_text(
                "Output Path", initial_value="export.glb"
            )
            self.glb_show_cam_checkbox = self.server.gui.add_checkbox(
                "Include Cameras", initial_value=True,
            )
            self.glb_cam_scale_slider = self.server.gui.add_slider(
                "Camera Scale", min=0.01, max=5.0, step=0.01, initial_value=1.0,
                hint="Scale factor for camera size in GLB.",
            )
            self.glb_frustum_thickness_slider = self.server.gui.add_slider(
                "Frustum Thickness", min=1.0, max=10.0, step=0.5, initial_value=3.0,
                hint="Thickness multiplier for camera frustum edges.",
            )
            self.glb_trajectory_checkbox = self.server.gui.add_checkbox(
                "Show Trajectory", initial_value=True,
                hint="Connect cameras with a trajectory line.",
            )
            self.glb_trajectory_radius_slider = self.server.gui.add_slider(
                "Trajectory Radius", min=0.001, max=0.05, step=0.001, initial_value=0.005,
                hint="Radius of the trajectory tube.",
            )
            self.glb_mode_dropdown = self.server.gui.add_dropdown(
                "Export Mode",
                options=["Points", "Spheres"],
                initial_value="Points",
                hint="Points: raw (fast). Spheres: each point becomes a small sphere (prettier, slower).",
            )
            self.glb_sphere_radius_slider = self.server.gui.add_slider(
                "Sphere Radius", min=0.001, max=0.1, step=0.001, initial_value=0.005,
                hint="Radius of each sphere in Spheres mode.",
                disabled=True,
            )
            self.glb_max_sphere_pts_slider = self.server.gui.add_slider(
                "Max Sphere Points", min=10000, max=500000, step=10000, initial_value=100000,
                hint="Cap point count for Spheres mode to keep file size manageable.",
                disabled=True,
            )
            self.glb_opacity_slider = self.server.gui.add_slider(
                "Opacity", min=0.0, max=1.0, step=0.05, initial_value=1.0,
                hint="Point/sphere opacity (alpha). <1.0 = semi-transparent.",
            )
            self.glb_saturation_slider = self.server.gui.add_slider(
                "Saturation Boost", min=0.0, max=2.0, step=0.1, initial_value=1.0,
                hint="Color saturation multiplier. >1 = more vivid, <1 = washed out.",
            )
            self.glb_brightness_slider = self.server.gui.add_slider(
                "Brightness Boost", min=0.5, max=2.0, step=0.1, initial_value=1.0,
                hint="Color brightness multiplier.",
            )
            self.glb_export_button = self.server.gui.add_button(
                "Export GLB",
                hint="Export current filtered point clouds and cameras as GLB.",
            )
            self.glb_status = self.server.gui.add_text("Status", initial_value="Ready")

        @self.glb_mode_dropdown.on_update
        def _(_) -> None:
            is_sphere = self.glb_mode_dropdown.value == "Spheres"
            self.glb_sphere_radius_slider.disabled = not is_sphere
            self.glb_max_sphere_pts_slider.disabled = not is_sphere

        @self.glb_export_button.on_click
        def _(_) -> None:
            self._export_glb()

        # Video saving controls
        with self.server.gui.add_folder("Video Saving"):
            self.save_video_button = self.server.gui.add_button("Save Video", disabled=False)
            self.video_output_path = self.server.gui.add_text("Output Path", initial_value="output_pointcloud.mp4")
            self.video_save_fps = self.server.gui.add_slider("Video FPS", min=10, max=60, step=1, initial_value=30)
            self.video_resolution = self.server.gui.add_dropdown(
                "Resolution", options=["1920x1080", "1280x720", "3840x2160"], initial_value="1920x1080"
            )
            self.save_original_video_checkbox = self.server.gui.add_checkbox("Also Save Original Video", initial_value=True)
            self.video_status = self.server.gui.add_text("Status", initial_value="Ready to save")

        @self.save_video_button.on_click
        def _(_) -> None:
            self.save_video(
                output_path=self.video_output_path.value,
                fps=self.video_save_fps.value,
                resolution=self.video_resolution.value,
                save_original_video=self.save_original_video_checkbox.value
            )

        # ── Rendering Comparison panel ────────────────────────────────────
        with self.server.gui.add_folder("Rendering Comparison"):
            self.render_mode_dropdown = self.server.gui.add_dropdown(
                "Mode", options=["All Frames", "Accumulated"],
                initial_value="All Frames",
                hint="All Frames: full point cloud. Accumulated: only frames 0..i.",
            )
            self.render_stride_slider = self.server.gui.add_slider(
                "Point Stride", min=1, max=8, step=1, initial_value=3,
                hint="Spatial downsample stride (higher = faster, sparser)",
            )
            self.compute_render_btn = self.server.gui.add_button(
                "Compute Renderings",
                hint="Pre-compute z-buffer renderings from each viewpoint",
            )
            self.render_status_text = self.server.gui.add_text(
                "Status", initial_value="Click 'Compute Renderings' to start"
            )
            self.render_metrics_text = self.server.gui.add_text("Metrics", initial_value="—")
            placeholder = np.zeros((10, 30, 3), dtype=np.uint8)
            self.render_comparison_image = self.server.gui.add_image(
                placeholder, label="GT | Rendered | Error"
            )

        @self.compute_render_btn.on_click
        def _(_) -> None:
            self.compute_render_btn.disabled = True
            accumulated = self.render_mode_dropdown.value == "Accumulated"
            stride = int(self.render_stride_slider.value)
            threading.Thread(
                target=self._precompute_renderings,
                args=(stride, accumulated),
                daemon=True,
            ).start()

        # ── Error-based point filtering ───────────────────────────────
        with self.server.gui.add_folder("Error-Based Filtering"):
            self.error_threshold_slider = self.server.gui.add_slider(
                "Error Threshold", min=0.01, max=0.5, step=0.01, initial_value=0.1,
                hint="Mean RGB error above this marks a pixel as 'bad'",
            )
            self.min_bad_views_slider = self.server.gui.add_slider(
                "Min Bad Views", min=1, max=20, step=1, initial_value=2,
                hint="A point must be bad in >= this many views to be removed",
            )
            self.compute_filter_btn = self.server.gui.add_button(
                "Compute Filtered Renderings",
                hint="Identify bad points via error map and re-render without them",
            )
            self.filter_status_text = self.server.gui.add_text(
                "Filter Status", initial_value="Run 'Compute Renderings' first"
            )
            self.show_filtered_checkbox = self.server.gui.add_checkbox(
                "Show Filtered", initial_value=False,
                hint="Toggle between original and filtered renderings",
            )
            self.filter_comparison_image = self.server.gui.add_image(
                np.zeros((10, 30, 3), dtype=np.uint8),
                label="Original | Filtered | Improvement",
            )

        @self.compute_filter_btn.on_click
        def _(_) -> None:
            if not hasattr(self, '_pts_per_frame'):
                self.filter_status_text.value = "Run 'Compute Renderings' first!"
                return
            self.compute_filter_btn.disabled = True
            error_thresh = self.error_threshold_slider.value
            min_views = int(self.min_bad_views_slider.value)
            threading.Thread(
                target=self._compute_error_filter,
                args=(error_thresh, min_views),
                daemon=True,
            ).start()

        @self.show_filtered_checkbox.on_update
        def _(_) -> None:
            if hasattr(self, 'gui_timestep'):
                self._update_render_display(self.gui_timestep.value)

        @self.show_video_checkbox.on_update
        def _(_) -> None:
            if self.current_frame_image is not None:
                self.current_frame_image.visible = self.show_video_checkbox.value

        self.pc_handles = []
        self.cam_handles = []

        @self.psize_slider.on_update
        def _(_) -> None:
            for handle in self.pc_handles:
                handle.point_size = self.psize_slider.value

        @self.camsize_slider.on_update
        def _(_) -> None:
            for handle in self.cam_handles:
                handle.scale = self.camsize_slider.value
                handle.line_thickness = 0.03 * handle.scale

        @self.downsample_slider.on_update
        def _(_) -> None:
            self._regenerate_point_clouds()

        @self.show_camera_checkbox.on_update
        def _(_) -> None:
            self.show_camera = self.show_camera_checkbox.value
            if self.show_camera:
                self._regenerate_cameras()
            else:
                for handle in self.cam_handles:
                    handle.visible = False

        @self.vis_threshold_slider.on_update
        def _(_) -> None:
            self.vis_threshold = self.vis_threshold_slider.value
            self._regenerate_point_clouds()

        @self.camera_downsample_slider.on_update
        def _(_) -> None:
            self._regenerate_cameras()

    def _regenerate_point_clouds(self):
        """Regenerate all point clouds with current settings."""
        if not hasattr(self, 'frame_nodes'):
            return

        for handle in self.pc_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        self.pc_handles.clear()
        self.vis_pts_list.clear()

        for i, step in enumerate(self.all_steps):
            pc = self.pcs[step]["pc"]
            color = self.pcs[step]["color"]
            conf = self.pcs[step]["conf"]
            edge_color = self.pcs[step].get("edge_color", None)

            pred_pts, pc_color = self.parse_pc_data(
                pc, color, conf, edge_color, set_border_color=True,
                downsample_factor=self.downsample_slider.value
            )

            self.vis_pts_list.append(pred_pts)
            handle = self.server.scene.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=pc_color,
                point_size=self.psize_slider.value,
            )
            self.pc_handles.append(handle)

    def _regenerate_cameras(self):
        """Regenerate camera visualizations with current settings."""
        if not hasattr(self, 'frame_nodes'):
            return

        for handle in self.cam_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        self.cam_handles.clear()

        if self.show_camera:
            downsample_factor = int(self.camera_downsample_slider.value)
            for i, step in enumerate(self.all_steps):
                if i % downsample_factor == 0:
                    self.add_camera(step)

    def _export_glb(self):
        """Export current filtered point clouds and cameras as a GLB file."""
        try:
            import trimesh
        except ImportError:
            self.glb_status.value = "Error: pip install trimesh"
            return

        self.glb_status.value = "Collecting points..."
        print("Exporting GLB...")

        # Collect all currently visible, filtered points and colors
        all_points = []
        all_colors = []
        for step in self.all_steps:
            pc = self.pcs[step]["pc"]
            color = self.pcs[step]["color"]
            conf = self.pcs[step]["conf"]
            edge_color = self.pcs[step].get("edge_color", None)

            pts, cols = self.parse_pc_data(
                pc, color, conf, edge_color, set_border_color=False,
                downsample_factor=self.downsample_slider.value,
            )
            if len(pts) > 0:
                all_points.append(pts)
                if cols.dtype != np.uint8:
                    cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
                all_colors.append(cols)

        if not all_points:
            self.glb_status.value = "Error: no points to export"
            return

        vertices = np.concatenate(all_points, axis=0)
        colors_rgb = np.concatenate(all_colors, axis=0)

        # --- Color enhancement ---
        colors_float = colors_rgb.astype(np.float32) / 255.0

        sat_boost = self.glb_saturation_slider.value
        if sat_boost != 1.0:
            gray = colors_float.mean(axis=1, keepdims=True)
            colors_float = gray + sat_boost * (colors_float - gray)

        bri_boost = self.glb_brightness_slider.value
        if bri_boost != 1.0:
            colors_float = colors_float * bri_boost

        colors_float = np.clip(colors_float, 0.0, 1.0)

        # --- Opacity ---
        # Simulate opacity by blending colors toward white (works in all viewers).
        # For Spheres mode, also set true alpha for viewers that support it.
        alpha = self.glb_opacity_slider.value
        if alpha < 1.0:
            bg = np.ones_like(colors_float)  # white background
            colors_float = colors_float * alpha + bg * (1.0 - alpha)
            colors_float = np.clip(colors_float, 0.0, 1.0)

        colors_u8 = (colors_float * 255).astype(np.uint8)
        colors_rgba = np.concatenate([
            colors_u8,
            np.full((len(colors_u8), 1), int(alpha * 255), dtype=np.uint8),
        ], axis=1)  # (N, 4)

        # Compute scene scale for camera sizing
        lo = np.percentile(vertices, 5, axis=0)
        hi = np.percentile(vertices, 95, axis=0)
        scene_scale = max(np.linalg.norm(hi - lo), 0.1)

        scene_3d = trimesh.Scene()

        # --- Export mode ---
        export_mode = self.glb_mode_dropdown.value
        if export_mode == "Spheres":
            self.glb_status.value = "Building spheres..."
            max_pts = int(self.glb_max_sphere_pts_slider.value)
            radius = self.glb_sphere_radius_slider.value

            # Subsample if too many points
            if len(vertices) > max_pts:
                idx = np.random.choice(len(vertices), max_pts, replace=False)
                idx.sort()
                vertices = vertices[idx]
                colors_rgba = colors_rgba[idx]

            sphere_template = trimesh.creation.icosphere(subdivisions=1, radius=radius)
            n_verts_per = len(sphere_template.vertices)
            n_faces_per = len(sphere_template.faces)

            all_verts = np.empty((len(vertices) * n_verts_per, 3), dtype=np.float32)
            all_faces = np.empty((len(vertices) * n_faces_per, 3), dtype=np.int64)
            all_face_colors = np.empty((len(vertices) * n_faces_per, 4), dtype=np.uint8)

            for i, (pt, rgba) in enumerate(zip(vertices, colors_rgba)):
                v_off = i * n_verts_per
                f_off = i * n_faces_per
                all_verts[v_off:v_off + n_verts_per] = sphere_template.vertices + pt
                all_faces[f_off:f_off + n_faces_per] = sphere_template.faces + v_off
                all_face_colors[f_off:f_off + n_faces_per] = rgba

            mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
            mesh.visual.face_colors = all_face_colors
            # Enable alpha blending in glTF material for true transparency
            if alpha < 1.0:
                mesh.visual.material.alphaMode = 'BLEND'
            scene_3d.add_geometry(mesh)
            print(f"Spheres mode: {len(vertices):,} spheres, {len(all_faces):,} faces")
        else:
            # Points mode (GLB viewers ignore alpha on points, so use blended RGB)
            scene_3d.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors_u8))

        # Add cameras and trajectory
        if self.glb_show_cam_checkbox.value and self.cam_dict is not None:
            from lingbot_map.vis.glb_export import integrate_camera_into_scene
            import matplotlib
            colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
            num_cameras = len(self.all_steps)
            cam_positions = []

            frustum_thickness = self.glb_frustum_thickness_slider.value
            effective_cam_scale = scene_scale * self.glb_cam_scale_slider.value

            for i, step in enumerate(self.all_steps):
                R = self.cam_dict["R"][step] if "R" in self.cam_dict else np.eye(3)
                t = self.cam_dict["t"][step] if "t" in self.cam_dict else np.zeros(3)

                c2w = np.eye(4)
                c2w[:3, :3] = R
                c2w[:3, 3] = t
                cam_positions.append(np.array(t, dtype=np.float64))

                rgba_c = colormap(i / max(num_cameras - 1, 1))
                cam_color = tuple(int(255 * x) for x in rgba_c[:3])
                integrate_camera_into_scene(
                    scene_3d, c2w, cam_color,
                    effective_cam_scale,
                    frustum_thickness=frustum_thickness,
                )

            # Add trajectory line as a tube connecting camera positions
            if self.glb_trajectory_checkbox.value and len(cam_positions) >= 2:
                traj_pts = np.array(cam_positions)
                traj_radius = self.glb_trajectory_radius_slider.value * self.glb_cam_scale_slider.value
                traj_mesh = self._build_trajectory_tube(
                    traj_pts, traj_radius, colormap, num_cameras
                )
                if traj_mesh is not None:
                    scene_3d.add_geometry(traj_mesh)

        # Align scene using first camera extrinsic
        if self.cam_dict is not None and len(self.all_steps) > 0:
            from lingbot_map.vis.glb_export import apply_scene_alignment
            step0 = self.all_steps[0]
            R0 = self.cam_dict["R"][step0] if "R" in self.cam_dict else np.eye(3)
            t0 = self.cam_dict["t"][step0] if "t" in self.cam_dict else np.zeros(3)
            c2w_0 = np.eye(4)
            c2w_0[:3, :3] = R0
            c2w_0[:3, 3] = t0
            w2c_0 = np.linalg.inv(c2w_0)
            extrinsics = np.expand_dims(w2c_0, 0)
            scene_3d = apply_scene_alignment(scene_3d, extrinsics)

        output_path = self.glb_output_path.value
        scene_3d.export(output_path)

        n_pts = len(vertices)
        mode_str = f"spheres r={self.glb_sphere_radius_slider.value}" if export_mode == "Spheres" else "points"
        self.glb_status.value = f"Saved: {output_path} ({n_pts:,} {mode_str})"
        print(f"GLB exported to {output_path} ({n_pts:,} {mode_str})")

    @staticmethod
    def _build_trajectory_tube(positions, radius, colormap, num_cameras):
        """Build a tube mesh following camera trajectory with per-segment color.

        Args:
            positions: (N, 3) camera positions.
            radius: Tube radius.
            colormap: Matplotlib colormap for gradient coloring.
            num_cameras: Total number of cameras (for color normalization).

        Returns:
            trimesh.Trimesh or None.
        """
        import trimesh

        segments = []
        for i in range(len(positions) - 1):
            p0, p1 = positions[i], positions[i + 1]
            seg_len = np.linalg.norm(p1 - p0)
            if seg_len < 1e-8:
                continue

            # Create cylinder along Z, then transform
            cyl = trimesh.creation.cylinder(radius=radius, height=seg_len, sections=8)

            # Direction vector
            direction = (p1 - p0) / seg_len
            mid = (p0 + p1) / 2.0

            # Build rotation: default cylinder is along Z
            z_axis = np.array([0.0, 0.0, 1.0])
            v = np.cross(z_axis, direction)
            c = np.dot(z_axis, direction)

            if np.linalg.norm(v) < 1e-8:
                rot = np.eye(3) if c > 0 else np.diag([1, -1, -1])
            else:
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                rot = np.eye(3) + vx + vx @ vx / (1.0 + c)

            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = mid
            cyl.apply_transform(transform)

            # Color: midpoint index
            t_color = (i + 0.5) / max(num_cameras - 1, 1)
            rgba = colormap(t_color)
            color_rgb = tuple(int(255 * x) for x in rgba[:3])
            cyl.visual.face_colors[:, :3] = color_rgb
            segments.append(cyl)

        if not segments:
            return None
        return trimesh.util.concatenate(segments)

    def update_frame_visibility(self):
        """Show all frames up to the current timestep (or only the current one in 4D mode)."""
        if not hasattr(self, 'frame_nodes') or not hasattr(self, 'gui_timestep'):
            return

        current_timestep = self.gui_timestep.value
        for i, frame_node in enumerate(self.frame_nodes):
            frame_node.visible = (
                i <= current_timestep if not self.fourd else i == current_timestep
            )

    # ── Rendering Comparison methods ────────────────────────────────────────

    def _precompute_renderings(self, stride=3, accumulated=False):
        """Pre-compute z-buffer renderings from each camera viewpoint."""
        if not hasattr(self, 'original_images') or len(self.original_images) == 0:
            self.render_status_text.value = "No images available"
            self.compute_render_btn.disabled = False
            return

        H, W = self.original_images[0].shape[:2]
        S = len(self.all_steps)

        # Collect points per frame
        self.render_status_text.value = "Collecting points..."
        pts_per_frame = []
        cols_per_frame = []

        for step in self.all_steps:
            pc = self.pcs[step]["pc"]
            color = self.pcs[step]["color"]
            conf = self.pcs[step]["conf"]

            if pc.size == 0:
                pts_per_frame.append(np.zeros((0, 3), dtype=np.float32))
                cols_per_frame.append(np.zeros((0, 3), dtype=np.float32))
                continue

            pc_ds = pc[::stride, ::stride]
            color_ds = color[::stride, ::stride]
            conf_ds = conf[::stride, ::stride]

            pts = pc_ds.reshape(-1, 3)
            cols = color_ds.reshape(-1, 3)
            conf_flat = conf_ds.reshape(-1)

            valid = (conf_flat > self.vis_threshold) & np.isfinite(pts).all(axis=1)
            pts_per_frame.append(pts[valid].astype(np.float32))
            cols_per_frame.append(cols[valid].astype(np.float32))

        # Save for error-based filtering
        self._pts_per_frame = pts_per_frame
        self._cols_per_frame = cols_per_frame
        self._render_stride = stride

        total_pts = sum(len(p) for p in pts_per_frame)
        self.render_status_text.value = f"{total_pts:,} pts. Rendering {S} frames..."

        # Pre-upload full point cloud once for OpenGL rendering
        all_pts_cat = np.concatenate(
            [p for p in pts_per_frame if len(p) > 0] or [np.zeros((0, 3), dtype=np.float32)],
            axis=0,
        )
        all_cols_cat = np.concatenate(
            [c for c in cols_per_frame if len(c) > 0] or [np.zeros((0, 3), dtype=np.float32)],
            axis=0,
        )
        all_idx_cat = np.arange(len(all_pts_cat), dtype=np.int64)

        # Frame boundary offsets for accumulated mode
        offsets = [0]
        for p in pts_per_frame:
            offsets.append(offsets[-1] + len(p))

        splat_radius = stride // 2 if stride > 1 else 0
        point_size = max(1.0, float(2 * splat_radius + 1))

        gl = _get_gl_renderer()
        gl.upload_points(all_pts_cat, all_cols_cat, all_idx_cat)

        self._rendered_frames = []

        for i, step in enumerate(self.all_steps):
            R_c2w = np.asarray(self.cam_dict["R"][step], dtype=np.float64)
            t_c2w = np.asarray(self.cam_dict["t"][step], dtype=np.float64)
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            focal = float(self.cam_dict["focal"][step])
            ppx, ppy = self.cam_dict["pp"][step]
            K = np.array([[focal, 0, float(ppx)],
                          [0, focal, float(ppy)],
                          [0, 0, 1]], dtype=np.float64)

            # Render: accumulated uses first offsets[i+1] points, all-frames uses all
            num_pts = offsets[i + 1] if accumulated else len(all_pts_cat)
            rendered, coverage, _ = gl.render_view(
                R_w2c, t_w2c, K, H, W, point_size=point_size,
                num_points=num_pts,
            )

            # GT as float [0, 1]
            gt_float = self.original_images[i].astype(np.float32) / 255.0

            # Metrics (on covered pixels only for PSNR)
            n_covered = int(coverage.sum())
            coverage_pct = n_covered / (H * W) * 100.0

            if n_covered > 100:
                diff_sq = ((gt_float - rendered) ** 2)[coverage]
                mse = float(diff_sq.mean())
                psnr = float(-10.0 * np.log10(mse + 1e-10))
                ssim = self._compute_ssim(gt_float, rendered)
            else:
                psnr, ssim = 0.0, 0.0

            # Error map (colored)
            error = np.mean(np.abs(gt_float - rendered), axis=2)
            error[~coverage] = 0.0
            error_norm = np.clip(error / 0.25, 0, 1)
            error_colored = (cm.jet(error_norm)[:, :, :3] * 255).astype(np.uint8)
            error_colored[~coverage] = [40, 40, 40]

            # Build comparison: [GT | Rendered | Error Map]
            gt_u8 = self.original_images[i]
            ren_u8 = (np.clip(rendered, 0, 1) * 255).astype(np.uint8)

            gt_lbl = gt_u8.copy()
            ren_lbl = ren_u8.copy()
            err_lbl = error_colored.copy()
            cv2.putText(gt_lbl, "GT", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ren_lbl, f"Rendered  PSNR:{psnr:.1f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(err_lbl, f"Error  Cov:{coverage_pct:.0f}%", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            sep = np.full((H, 2, 3), 255, dtype=np.uint8)
            combined = np.concatenate([gt_lbl, sep, ren_lbl, sep, err_lbl], axis=1)

            self._rendered_frames.append({
                'combined': combined,
                'psnr': psnr,
                'ssim': ssim,
                'coverage': coverage_pct,
            })

            if (i + 1) % 20 == 0 or i == S - 1:
                self.render_status_text.value = f"Rendered {i + 1}/{S}..."
                self._update_render_display(
                    self.gui_timestep.value if hasattr(self, 'gui_timestep') else 0
                )

        valid_r = [r for r in self._rendered_frames if r['psnr'] > 0]
        if valid_r:
            avg_psnr = np.mean([r['psnr'] for r in valid_r])
            avg_ssim = np.mean([r['ssim'] for r in valid_r])
            avg_cov = np.mean([r['coverage'] for r in valid_r])
            self.render_status_text.value = (
                f"Done! Avg PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | Cov: {avg_cov:.0f}%"
            )
        else:
            self.render_status_text.value = "Done (no valid renderings)"
        self.compute_render_btn.disabled = False
        self._update_render_display(
            self.gui_timestep.value if hasattr(self, 'gui_timestep') else 0
        )

    @staticmethod
    def _render_zbuffer(pts, cols, R_w2c, t_w2c, K, H, W, splat_radius=0,
                        point_indices=None):
        """Z-buffer render using OpenGL hardware rasterization (moderngl).

        Falls back to PyTorch if moderngl is not available.
        """
        if point_indices is None:
            point_indices = np.arange(len(pts), dtype=np.int64)

        point_size = max(1.0, float(2 * splat_radius + 1))

        renderer = _get_gl_renderer()
        return renderer.render(
            pts, cols, point_indices,
            R_w2c, t_w2c, K, H, W,
            point_size=point_size,
        )

    @staticmethod
    def _render_zbuffer_gpu(all_pts_np, all_cols_np, all_idx_np, active_idx,
                            R_w2c, t_w2c, K, H, W, splat_radius=0):
        """Z-buffer render using OpenGL. active_idx selects which points to render.

        Args:
            all_pts_np: (N, 3) float32 numpy
            all_cols_np: (N, 3) float32 numpy
            all_idx_np: (N,) int64 numpy — global point indices
            active_idx: numpy int array or torch tensor — indices of active points
            R_w2c, t_w2c: numpy arrays — camera extrinsics
            K: (3,3) numpy — intrinsics

        Returns: (rendered, coverage, index_buf)
        """
        if isinstance(active_idx, torch.Tensor):
            active_idx = active_idx.cpu().numpy()
        active_idx = active_idx.astype(np.int64)

        if len(active_idx) == 0:
            return (np.zeros((H, W, 3), dtype=np.float32),
                    np.zeros((H, W), dtype=bool),
                    np.full((H, W), -1, dtype=np.int64))

        pts = all_pts_np[active_idx]
        cols = all_cols_np[active_idx]
        pidx = all_idx_np[active_idx]

        point_size = max(1.0, float(2 * splat_radius + 1))
        renderer = _get_gl_renderer()
        return renderer.render(pts, cols, pidx, R_w2c, t_w2c, K, H, W,
                               point_size=point_size)

    @staticmethod
    def _compute_ssim(img1, img2):
        """Compute mean SSIM between two (H, W, 3) float images in [0, 1]."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_per_ch = []
        for c in range(3):
            a = img1[:, :, c].astype(np.float64)
            b = img2[:, :, c].astype(np.float64)
            mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
            mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
            s_aa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
            s_bb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
            s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
            ssim_map = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / \
                       ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
            ssim_per_ch.append(float(np.mean(ssim_map)))
        return float(np.mean(ssim_per_ch))

    def _compute_error_filter(self, error_threshold=0.1, min_bad_views=2):
        """Incremental map building with error-based pruning.

        Simulates streaming: for each frame i, accumulates points 0..i into
        the map, renders from viewpoint i, identifies high-error points via
        index buffer, and prunes them. The map grows incrementally with
        continuous quality control.

        Also renders the unpruned accumulated map for side-by-side comparison.
        """
        if not hasattr(self, '_pts_per_frame'):
            self.filter_status_text.value = "No point data. Run 'Compute Renderings' first."
            self.compute_filter_btn.disabled = False
            return

        H, W = self.original_images[0].shape[:2]
        S = len(self.all_steps)
        stride = self._render_stride
        splat_radius = stride // 2 if stride > 1 else 0

        # Flatten all points with global indices and frame boundaries
        all_pts = np.concatenate(
            [p for p in self._pts_per_frame if len(p) > 0] or [np.zeros((0, 3), dtype=np.float32)],
            axis=0,
        )
        all_cols = np.concatenate(
            [c for c in self._cols_per_frame if len(c) > 0] or [np.zeros((0, 3), dtype=np.float32)],
            axis=0,
        )
        N = len(all_pts)

        # Frame boundary offsets: frame i owns points [offsets[i], offsets[i+1])
        offsets = [0]
        for p in self._pts_per_frame:
            offsets.append(offsets[-1] + len(p))

        # Per-point state
        good_mask = np.ones(N, dtype=bool)          # still in the map?
        bad_count = np.zeros(N, dtype=np.int32)      # how many views flagged this point
        total_removed = 0

        self.filter_status_text.value = f"Incremental filtering {S} frames ({N:,} pts)..."
        self._filtered_frames = []
        self._unfiltered_accum_frames = []

        import time as _time
        _t_total_start = _time.perf_counter()

        all_idx = np.arange(N, dtype=np.int64)
        point_size = max(1.0, float(2 * splat_radius + 1))

        # Two GL contexts: one for filtered (with kill_points), one for unfiltered
        gl_filt = _GLPointRenderer()
        gl_unfilt = _GLPointRenderer()

        # Upload ALL points once to both renderers
        gl_filt.upload_points(all_pts, all_cols, all_idx)
        gl_unfilt.upload_points(all_pts, all_cols, all_idx)

        already_killed = set()  # track which points have been killed in gl_filt

        for i, step in enumerate(self.all_steps):
            _t_frame_start = _time.perf_counter()
            R_c2w = np.asarray(self.cam_dict["R"][step], dtype=np.float64)
            t_c2w = np.asarray(self.cam_dict["t"][step], dtype=np.float64)
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            focal = float(self.cam_dict["focal"][step])
            ppx, ppy = self.cam_dict["pp"][step]
            K = np.array([[focal, 0, float(ppx)],
                          [0, focal, float(ppy)],
                          [0, 0, 1]], dtype=np.float64)

            gt_float = self.original_images[i].astype(np.float32) / 255.0
            end = offsets[i + 1]

            # ── Render accumulated map (frames 0..i) ──
            rendered, coverage, index_buf = gl_filt.render_view(
                R_w2c, t_w2c, K, H, W, point_size=point_size,
                num_points=end)

            # Error analysis: L1 → flag bad points from all frames
            error = np.mean(np.abs(gt_float - rendered), axis=2)
            high_error = (error > error_threshold) & coverage
            bad_ids = index_buf[high_error].ravel()
            bad_ids = np.unique(bad_ids[bad_ids >= 0])

            if len(bad_ids) > 0:
                np.add.at(bad_count, bad_ids, 1)
                to_kill = bad_ids[
                    (bad_count[bad_ids] >= min_bad_views) & good_mask[bad_ids]]
                if len(to_kill) > 0:
                    good_mask[to_kill] = False
                    gl_filt.kill_points(to_kill.tolist())
                total_removed = int((~good_mask[:end]).sum())

            _t_frame_end = _time.perf_counter()
            _dt_frame = _t_frame_end - _t_frame_start
            _n_active = int(good_mask[:end].sum())

            if (i + 1) % 50 == 0 or i == S - 1:
                _elapsed = _t_frame_end - _t_total_start
                _fps = (i + 1) / _elapsed
                pct = total_removed / max(offsets[i + 1], 1) * 100
                self.filter_status_text.value = (
                    f"Pruning {i + 1}/{S} | {_n_active:,} pts | "
                    f"{_dt_frame*1000:.0f}ms | {_fps:.1f} FPS | "
                    f"Rm: {total_removed:,} ({pct:.1f}%)"
                )
                print(
                    f"[ErrorFilter] Frame {i+1}/{S}: "
                    f"{_n_active:,} active pts, "
                    f"{_dt_frame*1000:.1f}ms, "
                    f"avg {_fps:.2f} FPS, "
                    f"removed {total_removed:,}"
                )

        # ── Phase 1 done: timing ──
        _t_prune = _time.perf_counter() - _t_total_start
        pct_removed = total_removed / max(N, 1) * 100
        print(
            f"[ErrorFilter] Pruning done: {S} frames, {N:,} pts, "
            f"{_t_prune:.1f}s ({_t_prune/S*1000:.0f}ms/f), "
            f"removed {total_removed:,} ({pct_removed:.1f}%)"
        )

        # ══════════════════════════════════════════════════════════════════
        # Phase 2: Generate visualizations (render filtered + unfiltered)
        # ══════════════════════════════════════════════════════════════════
        self.filter_status_text.value = f"Generating visualizations..."
        self._filtered_frames = []
        self._unfiltered_accum_frames = []

        sep = np.full((H, 2, 3), 255, dtype=np.uint8)

        for i, step in enumerate(self.all_steps):
            R_c2w = np.asarray(self.cam_dict["R"][step], dtype=np.float64)
            t_c2w = np.asarray(self.cam_dict["t"][step], dtype=np.float64)
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            focal = float(self.cam_dict["focal"][step])
            ppx, ppy = self.cam_dict["pp"][step]
            K = np.array([[focal, 0, float(ppx)],
                          [0, focal, float(ppy)],
                          [0, 0, 1]], dtype=np.float64)

            gt_float = self.original_images[i].astype(np.float32) / 255.0
            end = offsets[i + 1]

            # Filtered render (pruned points already killed)
            ren_filt, cov_filt, _ = gl_filt.render_view(
                R_w2c, t_w2c, K, H, W, point_size=point_size,
                num_points=end)
            # Unfiltered render
            ren_unfilt, cov_unfilt, _ = gl_unfilt.render_view(
                R_w2c, t_w2c, K, H, W, point_size=point_size,
                num_points=end)

            # PSNR
            n_cov_f = int(cov_filt.sum())
            n_cov_u = int(cov_unfilt.sum())
            filt_psnr = unfilt_psnr = 0.0
            if n_cov_f > 100:
                mse_f = float(((gt_float - ren_filt) ** 2)[cov_filt].mean())
                filt_psnr = float(-10.0 * np.log10(mse_f + 1e-10))
            if n_cov_u > 100:
                mse_u = float(((gt_float - ren_unfilt) ** 2)[cov_unfilt].mean())
                unfilt_psnr = float(-10.0 * np.log10(mse_u + 1e-10))
            psnr_delta = filt_psnr - unfilt_psnr

            # [GT | Filtered | Error]
            err_filt = np.mean(np.abs(gt_float - ren_filt), axis=2)
            err_filt[~cov_filt] = 0.0
            err_u8 = np.clip(err_filt * (255.0 / 0.25), 0, 255).astype(np.uint8)
            err_colored = np.zeros((H, W, 3), dtype=np.uint8)
            err_colored[:, :, 2] = err_u8
            err_colored[:, :, 1] = 255 - err_u8
            err_colored[~cov_filt] = [40, 40, 40]

            gt_u8 = self.original_images[i].copy()
            filt_u8 = (np.clip(ren_filt, 0, 1) * 255).astype(np.uint8)
            n_in_map = int(good_mask[:end].sum())
            cv2.putText(gt_u8, "GT", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(filt_u8, f"Filtered PSNR:{filt_psnr:.1f} ({psnr_delta:+.1f})", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(err_colored, f"Map:{n_in_map:,}pts Rm:{total_removed:,}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            combined = np.concatenate([gt_u8, sep, filt_u8, sep, err_colored], axis=1)
            self._filtered_frames.append({
                'combined': combined, 'psnr': filt_psnr, 'ssim': 0.0,
                'coverage': n_cov_f / (H * W) * 100.0, 'psnr_delta': psnr_delta,
            })

            # [Unfiltered | Filtered | Improvement]
            unfilt_u8 = (np.clip(ren_unfilt, 0, 1) * 255).astype(np.uint8)
            err_unfilt = np.mean(np.abs(gt_float - ren_unfilt), axis=2)
            improvement = err_unfilt - err_filt
            imp_vis = np.zeros((H, W, 3), dtype=np.uint8)
            better = improvement > 0.01
            worse = improvement < -0.01
            if better.any():
                imp_vis[better, 1] = (np.clip(improvement[better] / 0.15, 0, 1) * 255).astype(np.uint8)
            if worse.any():
                imp_vis[worse, 0] = (np.clip(-improvement[worse] / 0.15, 0, 1) * 255).astype(np.uint8)

            cv2.putText(unfilt_u8, f"No filter PSNR:{unfilt_psnr:.1f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            filt_u8_2 = (np.clip(ren_filt, 0, 1) * 255).astype(np.uint8)
            cv2.putText(filt_u8_2, f"Filtered PSNR:{filt_psnr:.1f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(imp_vis, f"dPSNR:{psnr_delta:+.1f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            filter_cmp = np.concatenate([unfilt_u8, sep, filt_u8_2, sep, imp_vis], axis=1)
            self._unfiltered_accum_frames.append({
                'filter_cmp': filter_cmp,
                'psnr_unfilt': unfilt_psnr, 'psnr_filt': filt_psnr,
            })

        # ── Final summary ──
        _t_total = _time.perf_counter() - _t_total_start
        print(
            f"[ErrorFilter] TOTAL: {_t_total:.1f}s "
            f"(prune {_t_prune:.1f}s + vis {_t_total - _t_prune:.1f}s)"
        )

        valid_f = [r for r in self._filtered_frames if r['psnr'] > 0]
        valid_u = [r for r in self._unfiltered_accum_frames if r['psnr_unfilt'] > 0]
        if valid_f and valid_u:
            avg_filt = np.mean([r['psnr'] for r in valid_f])
            avg_unfilt = np.mean([r['psnr_unfilt'] for r in valid_u])
            self.filter_status_text.value = (
                f"Done! {_t_total:.1f}s (prune {_t_prune:.1f}s) | "
                f"Rm {pct_removed:.1f}% | "
                f"PSNR: {avg_unfilt:.2f}→{avg_filt:.2f} ({avg_filt - avg_unfilt:+.2f})"
            )
        else:
            self.filter_status_text.value = f"Done. {_t_total:.1f}s. Removed {pct_removed:.1f}%."

        # Release dedicated GL contexts
        gl_filt.release()
        gl_unfilt.release()

        self.show_filtered_checkbox.value = True
        self.compute_filter_btn.disabled = False
        self._update_render_display(
            self.gui_timestep.value if hasattr(self, 'gui_timestep') else 0
        )

    def _update_render_display(self, frame_idx):
        """Update the rendering comparison panel for the given frame."""
        # Choose between original and filtered
        show_filtered = (hasattr(self, 'show_filtered_checkbox')
                         and self.show_filtered_checkbox.value
                         and hasattr(self, '_filtered_frames')
                         and self._filtered_frames)

        if show_filtered:
            frames = self._filtered_frames
        elif hasattr(self, '_rendered_frames') and self._rendered_frames:
            frames = self._rendered_frames
        else:
            return

        idx = min(frame_idx, len(frames) - 1)
        data = frames[idx]
        self.render_comparison_image.image = data['combined']

        delta_str = ""
        if show_filtered and 'psnr_delta' in data:
            delta_str = f" ({data['psnr_delta']:+.2f})"

        self.render_metrics_text.value = (
            f"Frame {idx}: PSNR {data['psnr']:.2f} dB{delta_str} | "
            f"SSIM {data['ssim']:.4f} | Cov {data['coverage']:.1f}%"
        )

        # Update filter comparison panel (Unfiltered | Filtered | Improvement)
        if (hasattr(self, '_unfiltered_accum_frames') and self._unfiltered_accum_frames
                and idx < len(self._unfiltered_accum_frames)):
            self.filter_comparison_image.image = self._unfiltered_accum_frames[idx]['filter_cmp']

    def _move_to_camera(self, frame_idx: int, smooth: bool = True):
        """Move viewer camera to match reconstructed camera at given frame."""
        if self.cam_dict is None:
            return

        step = self.all_steps[frame_idx] if frame_idx < len(self.all_steps) else self.all_steps[-1]

        R = self.cam_dict["R"][step] if "R" in self.cam_dict else np.eye(3)
        t = self.cam_dict["t"][step] if "t" in self.cam_dict else np.zeros(3)
        focal = self.cam_dict["focal"][step] if "focal" in self.cam_dict else 1.0
        pp = self.cam_dict["pp"][step] if "pp" in self.cam_dict else (1.0, 1.0)

        offset = 0.5
        viewing_dir = R[:, 2]  # camera Z axis in world frame
        position = t - viewing_dir * offset
        look_at = t + viewing_dir * 0.5  # look slightly ahead of camera

        fov = 2 * np.arctan(pp[0] / focal)
        up = -R[:, 1]  # camera -Y axis in world frame

        for client in self.server.get_clients().values():
            if smooth:
                self._smooth_camera_transition(
                    client,
                    target_position=position,
                    target_look_at=look_at,
                    target_up=up,
                    target_fov=fov,
                    duration=0.3,
                )
            else:
                client.camera.up_direction = tuple(up)
                client.camera.position = tuple(position)
                client.camera.look_at = tuple(look_at)
                if fov is not None:
                    client.camera.fov = fov

    def _smooth_camera_transition(
        self,
        client,
        target_position,
        target_look_at=None,
        target_up=None,
        target_fov=None,
        duration=0.3,
    ):
        """Smoothly transition camera to target pose using look_at based control.

        Args:
            client: Viser client handle.
            target_position: Target camera position (3,).
            target_look_at: Target look-at point (3,). If None, keeps current.
            target_up: Target up direction (3,). If None, keeps current.
            target_fov: Target FOV. If None, keeps current.
            duration: Transition duration in seconds.
        """
        def interpolate():
            num_steps = 15
            dt = duration / num_steps

            start_position = np.array(client.camera.position, dtype=np.float64)
            start_look_at = np.array(client.camera.look_at, dtype=np.float64)
            start_fov = client.camera.fov

            end_position = np.asarray(target_position, dtype=np.float64)
            end_look_at = np.asarray(target_look_at, dtype=np.float64) if target_look_at is not None else start_look_at

            # Set up direction once at the start (not interpolated to avoid flicker)
            if target_up is not None:
                client.camera.up_direction = tuple(np.asarray(target_up, dtype=np.float64))

            for i in range(num_steps + 1):
                alpha = i / num_steps
                # Smooth ease-in-out
                alpha_smooth = alpha * alpha * (3 - 2 * alpha)

                interp_pos = start_position + (end_position - start_position) * alpha_smooth
                interp_look = start_look_at + (end_look_at - start_look_at) * alpha_smooth

                # Set position first (this auto-moves look_at), then override look_at
                client.camera.position = tuple(interp_pos)
                client.camera.look_at = tuple(interp_look)

                if target_fov is not None:
                    interp_fov = start_fov + (target_fov - start_fov) * alpha_smooth
                    client.camera.fov = interp_fov

                time.sleep(dt)

        thread = threading.Thread(target=interpolate, daemon=True)
        thread.start()

    def _slerp(self, q1, q2, t):
        """Spherical linear interpolation between quaternions."""
        dot = np.dot(q1, q2)

        if abs(dot) > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        dot = np.clip(dot, -1.0, 1.0)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        q2_orthogonal = q2 - q1 * dot
        q2_orthogonal = q2_orthogonal / np.linalg.norm(q2_orthogonal)

        return q1 * np.cos(theta) + q2_orthogonal * np.sin(theta)

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        """Get current camera state from client."""
        camera = client.camera
        c2w = np.concatenate([
            np.concatenate([tf.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1),
            [[0, 0, 0, 1]],
        ], 0)
        return CameraState(fov=camera.fov, aspect=camera.aspect, c2w=c2w)

    @staticmethod
    def generate_pseudo_intrinsics(h: int, w: int) -> np.ndarray:
        """Generate pseudo intrinsics from image size."""
        focal = (h**2 + w**2) ** 0.5
        return np.array([[focal, 0, w // 2], [0, focal, h // 2], [0, 0, 1]]).astype(np.float32)

    def _connect_client(self, client: viser.ClientHandle):
        """Setup client connection callbacks."""
        wxyz_panel = client.gui.add_text("wxyz:", f"{client.camera.wxyz}")
        position_panel = client.gui.add_text("position:", f"{client.camera.position}")
        fov_panel = client.gui.add_text(
            "fov:", f"{2 * np.arctan(self.size/self.focal_slider.value) * 180 / np.pi}"
        )
        aspect_panel = client.gui.add_text("aspect:", "1.0")

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            with self.server.atomic():
                wxyz_panel.value = f"{client.camera.wxyz}"
                position_panel.value = f"{client.camera.position}"
                fov_panel.value = f"{2 * np.arctan(self.size/self.focal_slider.value) * 180 / np.pi}"
                aspect_panel.value = "1.0"

    @staticmethod
    def set_color_border(image, border_width=5, color=[1, 0, 0]):
        """Add colored border to image."""
        image[:border_width, :, 0] = color[0]
        image[:border_width, :, 1] = color[1]
        image[:border_width, :, 2] = color[2]
        image[-border_width:, :, 0] = color[0]
        image[-border_width:, :, 1] = color[1]
        image[-border_width:, :, 2] = color[2]
        image[:, :border_width, 0] = color[0]
        image[:, :border_width, 1] = color[1]
        image[:, :border_width, 2] = color[2]
        image[:, -border_width:, 0] = color[0]
        image[:, -border_width:, 1] = color[1]
        image[:, -border_width:, 2] = color[2]
        return image

    def read_data(self, pc_list, color_list, conf_list, edge_color_list=None):
        """Read and organize point cloud data."""
        pcs = {}
        step_list = []
        for i, pc in enumerate(pc_list):
            step = i
            pcs.update({
                step: {
                    "pc": pc,
                    "color": color_list[i],
                    "conf": conf_list[i],
                    "edge_color": (
                        None if edge_color_list is None or edge_color_list[i] is None
                        else edge_color_list[i]
                    ),
                }
            })
            step_list.append(step)

        # Generate camera gradient colors
        num_cameras = len(pc_list)
        if num_cameras > 1:
            normalized_indices = np.array(list(range(num_cameras))) / (num_cameras - 1)
        else:
            normalized_indices = np.array([0.0])
        cmap = cm.get_cmap('viridis')
        self.camera_colors = cmap(normalized_indices)
        return pcs, step_list

    def parse_pc_data(
        self,
        pc,
        color,
        conf=None,
        edge_color=[0.251, 0.702, 0.902],
        set_border_color=False,
        downsample_factor=1,
    ):
        """Parse and filter point cloud data."""
        pred_pts = pc.reshape(-1, 3)

        if set_border_color and edge_color is not None:
            color = self.set_color_border(color[0], color=edge_color)
        if np.isnan(color).any():
            color = np.zeros((pred_pts.shape[0], 3))
            color[:, 2] = 1
        else:
            color = color.reshape(-1, 3)

        # Remove NaN / Inf points
        valid = np.isfinite(pred_pts).all(axis=1)
        if not valid.all():
            pred_pts = pred_pts[valid]
            color = color[valid]
            if conf is not None:
                conf = conf.reshape(-1)[valid]

        # Confidence threshold filter
        if conf is not None:
            conf_flat = conf.reshape(-1) if conf.ndim > 1 else conf
            mask = conf_flat > self.vis_threshold
            pred_pts = pred_pts[mask]
            color = color[mask]

        if len(pred_pts) == 0:
            return pred_pts, color

        # Downsample
        if downsample_factor > 1 and len(pred_pts) > 0:
            indices = np.arange(0, len(pred_pts), downsample_factor)
            pred_pts = pred_pts[indices]
            color = color[indices]

        return pred_pts, color

    def add_pc(self, step):
        """Add point cloud for a frame."""
        pc = self.pcs[step]["pc"]
        color = self.pcs[step]["color"]
        conf = self.pcs[step]["conf"]
        edge_color = self.pcs[step].get("edge_color", None)

        pred_pts, color = self.parse_pc_data(
            pc, color, conf, edge_color, set_border_color=True,
            downsample_factor=self.downsample_slider.value
        )

        self.vis_pts_list.append(pred_pts)
        self.pc_handles.append(
            self.server.scene.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=color,
                point_size=self.psize_slider.value,
            )
        )

    def add_camera(self, step):
        """Add camera visualization for a frame."""
        cam = self.cam_dict
        focal = cam["focal"][step] if cam and "focal" in cam else 1.0
        pp = cam["pp"][step] if cam and "pp" in cam else (1.0, 1.0)
        R = cam["R"][step] if cam and "R" in cam else np.eye(3)
        t = cam["t"][step] if cam and "t" in cam else np.zeros(3)

        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(pp[0] / focal)
        aspect = pp[0] / pp[1]
        self.traj_list.append((q, t))

        step_index = self.all_steps.index(step) if step in self.all_steps else 0
        camera_color = self.camera_colors[step_index]
        camera_color_rgb = tuple((camera_color[:3] * 255).astype(int))

        self.server.scene.add_frame(
            f"/frames/{step}/camera_frame",
            wxyz=q,
            position=t,
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
        )

        frustum_handle = self.server.scene.add_camera_frustum(
            name=f"/frames/{step}/camera",
            fov=fov,
            aspect=aspect,
            wxyz=q,
            position=t,
            scale=0.03,
            color=camera_color_rgb,
        )

        @frustum_handle.on_click
        def _(event) -> None:
            look_at_pt = t + R[:, 2] * 0.5  # look ahead along camera Z
            up_dir = -R[:, 1]
            for client in self.server.get_clients().values():
                client.camera.up_direction = tuple(up_dir)
                client.camera.position = tuple(t)
                client.camera.look_at = tuple(look_at_pt)

        self.cam_handles.append(frustum_handle)

    def animate(self):
        """Setup and run animation controls."""
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Train Step", min=0, max=self.num_frames - 1, step=1, initial_value=0, disabled=False
            )
            gui_next_frame = self.server.gui.add_button("Next Step", disabled=False)
            gui_prev_frame = self.server.gui.add_button("Prev Step", disabled=False)
            gui_playing = self.server.gui.add_checkbox("Playing", True)
            gui_framerate = self.server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=20)
            gui_framerate_options = self.server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))

        @gui_next_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % self.num_frames

        @gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        prev_timestep = self.gui_timestep.value

        @self.gui_timestep.on_update
        def _(_) -> None:
            nonlocal prev_timestep
            current_timestep = self.gui_timestep.value

            if self.current_frame_image is not None and hasattr(self, 'original_images'):
                if current_timestep < len(self.original_images):
                    self.current_frame_image.image = self.original_images[current_timestep]

            with self.server.atomic():
                self.frame_nodes[current_timestep].visible = True
                self.frame_nodes[prev_timestep].visible = False
            self.server.flush()

            self._update_render_display(current_timestep)

            prev_timestep = current_timestep

        self.server.scene.add_frame("/frames", show_axes=False)
        self.frame_nodes = []
        for i in range(self.num_frames):
            step = self.all_steps[i]
            self.frame_nodes.append(
                self.server.scene.add_frame(f"/frames/{step}", show_axes=False)
            )
            self.add_pc(step)
            if self.show_camera:
                downsample_factor = int(self.camera_downsample_slider.value)
                if i % downsample_factor == 0:
                    self.add_camera(step)

        prev_timestep = self.gui_timestep.value
        while True:
            if self.on_replay:
                pass
            else:
                if gui_playing.value:
                    self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames
                self.update_frame_visibility()

            time.sleep(1.0 / gui_framerate.value)

    def _take_screenshot(self, client: Optional[Any] = None):
        """Capture a screenshot from the current view and save to file.

        Args:
            client: The viser client that triggered the action. If None,
                    uses the first connected client.
        """
        output_path = self.screenshot_path.value
        res_str = self.screenshot_resolution.value

        # Resolve client
        if client is None:
            clients = list(self.server.get_clients().values())
            if not clients:
                self.screenshot_status.value = "Error: no client connected"
                return
            client = clients[0]

        try:
            self.screenshot_status.value = "Capturing..."

            if res_str == "Current":
                # Use default render size
                width, height = 1920, 1080
            else:
                width, height = map(int, res_str.split("x"))

            render = client.camera.get_render(height=height, width=width)

            if render is not None:
                frame = np.array(render)
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, frame_bgr)
                self.screenshot_status.value = f"Saved: {output_path}"
                print(f"Screenshot saved to {output_path} ({width}x{height})")
            else:
                self.screenshot_status.value = "Error: render returned None"
                print("Screenshot failed: render returned None")

        except Exception as e:
            self.screenshot_status.value = f"Error: {e}"
            print(f"Screenshot error: {e}")

    def save_video(
        self,
        output_path: str = "output_pointcloud.mp4",
        fps: int = 30,
        resolution: str = "1920x1080",
        save_original_video: bool = True
    ):
        """Save point cloud animation as video."""
        try:
            if hasattr(self, 'video_status'):
                self.video_status.value = "Saving video..."
            print(f"Saving video to {output_path}...")

            width, height = map(int, resolution.split('x'))
            temp_dir = tempfile.mkdtemp(prefix="viser_video_")
            print(f"Temporary directory: {temp_dir}")

            print("Waiting for client connection...")
            timeout = 10
            start_time = time.time()
            while len(self.server.get_clients()) == 0:
                time.sleep(0.1)
                if time.time() - start_time > timeout:
                    raise RuntimeError("No client connected. Please open the visualization in a browser first.")

            print("Client connected. Starting to render frames...")
            clients = list(self.server.get_clients().values())
            client = clients[0]

            if not hasattr(self, 'gui_timestep'):
                raise RuntimeError("Animation not initialized. Please ensure animate() is called before save_video().")

            for i in tqdm(range(self.num_frames), desc="Rendering frames"):
                self.gui_timestep.value = i
                time.sleep(0.1)

                try:
                    screenshot = client.camera.get_render(height=height, width=width)
                    if screenshot is not None:
                        frame = np.array(screenshot)
                        if frame.shape[2] == 4:
                            frame = frame[:, :, :3]
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                        cv2.imwrite(frame_path, frame)
                    else:
                        frame = self._render_frame_fallback(i, width, height)
                        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                        cv2.imwrite(frame_path, frame)
                except Exception as e:
                    print(f"Warning: Error capturing frame {i}: {e}, using fallback")
                    frame = self._render_frame_fallback(i, width, height)
                    frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    cv2.imwrite(frame_path, frame)

            print("Encoding video with ffmpeg...")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                output_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Point cloud video saved successfully to {output_path}")
                if hasattr(self, 'video_status'):
                    self.video_status.value = f"Saved to {output_path}"
            else:
                print(f"FFmpeg error: {result.stderr}")
                if hasattr(self, 'video_status'):
                    self.video_status.value = "Error: FFmpeg failed"

            if save_original_video and hasattr(self, 'original_images') and len(self.original_images) > 0:
                self._save_original_video(output_path, fps, width, height)

            shutil.rmtree(temp_dir)
            print("Temporary files cleaned up")

        except Exception as e:
            print(f"Error saving video: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'video_status'):
                self.video_status.value = f"Error: {str(e)}"

    def _save_original_video(self, pointcloud_video_path: str, fps: int, width: int, height: int):
        """Save original images as video."""
        base_path = os.path.splitext(pointcloud_video_path)[0]
        original_video_path = f"{base_path}_original.mp4"

        print(f"Saving original images video to {original_video_path}...")

        try:
            temp_dir = tempfile.mkdtemp(prefix="original_video_")

            for i, img in enumerate(tqdm(self.original_images, desc="Saving original frames")):
                frame = cv2.resize(img, (width, height))
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, frame)

            print("Encoding original video with ffmpeg...")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                original_video_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Original video saved successfully to {original_video_path}")
            else:
                print(f"FFmpeg error for original video: {result.stderr}")

            shutil.rmtree(temp_dir)

        except Exception as e:
            print(f"Error saving original video: {e}")
            import traceback
            traceback.print_exc()

    def _render_frame_fallback(self, frame_idx: int, width: int, height: int) -> np.ndarray:
        """Fallback rendering when screenshot capture fails."""
        if hasattr(self, 'original_images') and frame_idx < len(self.original_images):
            frame = self.original_images[frame_idx].copy()
            frame = cv2.resize(frame, (width, height))
            cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Frame {frame_idx} - No render available",
                        (width//4, height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

    def run(self, background_mode: bool = False):
        """Run the viewer."""
        self.animate()
        if background_mode:
            def server_loop():
                while True:
                    time.sleep(0.001)

            thread = threading.Thread(target=server_loop, daemon=True)
            thread.start()
        else:
            while True:
                time.sleep(10.0)
