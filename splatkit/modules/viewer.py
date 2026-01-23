"""
Viewer module for visualizing 3D Gaussian Splatting training progress.

Based on gsplat's viewer implementation with viser and nerfview.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence, Tuple

import numpy as np
import torch
import viser
from nerfview import CameraState, RenderTabState, Viewer


from nerfview._renderer import InterruptRenderException

from .base import SplatBaseModule
from .frame import SplatRenderPayload
from ..splat.training_state import SplatTrainingState

if TYPE_CHECKING:
    from ..logger.base import SplatLogger
    from ..renderer.base import SplatRenderer
    from ..data_provider.base import SplatDataProvider, SplatDataItem
    from ..loss_fn.base import SplatLossFn
    from ..densification.base import SplatDensification


class SplatViewerTabState(RenderTabState):
    """Extended render tab state with gsplat-specific controls."""
    
    # Non-controllable parameters (stats)
    total_gs_count: int = 0
    rendered_gs_count: int = 0
    
    # Controllable parameters
    max_sh_degree: int = 3
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[int, int, int] = (0, 0, 0)  # RGB values 0-255
    render_mode: Literal[
        "rgb", "depth(accumulated)", "depth(expected)", "alpha"
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"


class SplatViewer(SplatBaseModule[SplatRenderPayload]):
    """
    Interactive viewer module for training visualization.
    
    Integrates viser + nerfview for real-time 3D visualization during training.
    Provides controls for SH degree, rendering modes, depth visualization, etc.
    
    Example:
        viewer = SplatViewer(
            port=8080,
            output_dir="results/viewer",
            update_interval=1,
        )
    """
    
    def __init__(
        self,
        port: int = 8080,
        output_dir: str | None = None,
        update_interval: int = 1,
        mode: Literal["training", "rendering"] = "training",
    ):
        """
        Initialize the viewer module.
        
        Args:
            port: Port number for the viser server
            output_dir: Directory to save viewer outputs (screenshots, etc.)
            update_interval: Update the viewer every N steps
            mode: Viewer mode ("training" or "rendering")
        """
        # Configuration (assigned in __init__)
        self._port = port
        self._output_dir = Path(output_dir) if output_dir else Path("viewer_output")
        self._update_interval = update_interval
        self._mode: Literal["training", "rendering"] = mode
        
        # Runtime state (initialized in on_setup)
        self._server: viser.ViserServer | None = None
        self._viewer: Viewer | None = None
        self._lock = None
        self._render_tab_state: SplatViewerTabState | None = None
        
        # References to training components (set in on_setup/hooks)
        self._renderer: SplatRenderer[SplatRenderPayload] | None = None
        self._training_state: SplatTrainingState | None = None
        
        # Timing for rays per second calculation
        self._step_start_time: float | None = None
        
    @property
    def module_name(self) -> str:
        return "SplatViewer"
    
    def on_setup(
        self,
        logger: "SplatLogger",
        renderer: "SplatRenderer[SplatRenderPayload]",
        data_provider: "SplatDataProvider[SplatRenderPayload, SplatDataItem]",
        loss_fn: "SplatLossFn[SplatRenderPayload]",
        densification: "SplatDensification[SplatRenderPayload]",
        modules: Sequence[SplatBaseModule[SplatRenderPayload]],
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        """Setup the viewer server and initialize all components."""
        # Only run viewer on rank 0 in distributed training
        if world_rank != 0:
            logger.info("Viewer disabled on non-zero ranks", module=self.module_name)
            return
        
        if world_size > 1:
            logger.warning(
                "Viewer in distributed training may have limited functionality",
                module=self.module_name,
            )
        
        # Store renderer reference (needed by render function)
        self._renderer = renderer
        
        # Initialize viser server (capture npm/node output)
        try:
            import sys
            from io import StringIO
            
            # Capture stdout/stderr to buffer
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_buffer = StringIO()
            stderr_buffer = StringIO()
            
            try:
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                self._server = viser.ViserServer(port=self._port, verbose=False)
                self._server.gui.set_panel_label("splatkit viewer")
            except Exception as e:
                # If error, print captured output for debugging
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                captured_out = stdout_buffer.getvalue()
                captured_err = stderr_buffer.getvalue()
                if captured_out:
                    print(captured_out, end='')
                if captured_err:
                    print(captured_err, end='', file=sys.stderr)
                raise e
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            logger.info(f"Viser server initialized successfully on port {self._port}", module=self.module_name)
            
            # Create output directory
            self._output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize render tab state
            self._render_tab_state = SplatViewerTabState()
            
            # Create internal render function (closure with access to self)
            def _internal_render_fn(camera_state: CameraState, render_tab_state: RenderTabState) -> np.ndarray:
                """Internal render function with access to module state."""
                return self._render_fn(camera_state, render_tab_state)
            
            # Initialize basic viewer
            self._viewer = Viewer(
                server=self._server,
                render_fn=_internal_render_fn,
                output_dir=self._output_dir,
                mode=self._mode
            )
            
            # Override viewer's render_tab_state with our custom one
            self._viewer.render_tab_state = self._render_tab_state
            
            # Add custom GUI controls
            self._setup_gui_controls()
            
            # Create lock for thread-safe updates
            import threading
            self._lock = threading.Lock()
            
            logger.info(
                f"Viewer initialized. Open http://localhost:{self._port} to view training progress",
                module=self.module_name,
            )
            
        except Exception as e:
            logger.error(f"Failed to start viewer: {e}", module=self.module_name)
            self._server = None
            self._viewer = None
    
    def _setup_gui_controls(self):
        """Setup custom GUI controls for splat rendering."""
        if self._server is None or self._render_tab_state is None:
            return
        
        server = self._server
        tab_state = self._render_tab_state
        
        # Create rendering folder
        with server.gui.add_folder("Gaussian Splatting"):
            # Stats (non-editable)
            total_gs_count_number = server.gui.add_number(
                "Total Gaussians",
                initial_value=tab_state.total_gs_count,
                disabled=True,
                hint="Total number of Gaussians in the scene.",
            )
            rendered_gs_count_number = server.gui.add_number(
                "Rendered Gaussians",
                initial_value=tab_state.rendered_gs_count,
                disabled=True,
                hint="Number of Gaussians rendered.",
            )
            
            # SH degree control
            max_sh_degree_number = server.gui.add_number(
                "Max SH Degree",
                initial_value=tab_state.max_sh_degree,
                min=0,
                max=5,
                step=1,
                hint="Maximum spherical harmonics degree",
            )
            
            @max_sh_degree_number.on_update
            def _(_) -> None:
                tab_state.max_sh_degree = int(max_sh_degree_number.value)
                if self._viewer:
                    self._viewer.rerender(_)
            
            # Near/Far plane
            near_far_plane_vec2 = server.gui.add_vector2(
                "Near/Far Plane",
                initial_value=(tab_state.near_plane, tab_state.far_plane),
                min=(1e-3, 1e1),
                max=(1e1, 1e3),
                step=1e-3,
                hint="Near and far plane for rendering.",
            )
            
            @near_far_plane_vec2.on_update
            def _(_) -> None:
                tab_state.near_plane = near_far_plane_vec2.value[0]
                tab_state.far_plane = near_far_plane_vec2.value[1]
                if self._viewer:
                    self._viewer.rerender(_)
            
            # Radius clip
            radius_clip_slider = server.gui.add_number(
                "Radius Clip",
                initial_value=tab_state.radius_clip,
                min=0.0,
                max=100.0,
                step=1.0,
                hint="2D radius clip for rendering.",
            )
            
            @radius_clip_slider.on_update
            def _(_) -> None:
                tab_state.radius_clip = radius_clip_slider.value
                if self._viewer:
                    self._viewer.rerender(_)
            
            # 2D Epsilon
            eps2d_slider = server.gui.add_number(
                "2D Epsilon",
                initial_value=tab_state.eps2d,
                min=0.0,
                max=1.0,
                step=0.01,
                hint="Epsilon added to eigenvalues of 2D covariance.",
            )
            
            @eps2d_slider.on_update
            def _(_) -> None:
                tab_state.eps2d = eps2d_slider.value
                if self._viewer:
                    self._viewer.rerender(_)
            
            # Background color
            backgrounds_slider = server.gui.add_rgb(
                "Background",
                initial_value=tab_state.backgrounds,
                hint="Background color for rendering.",
            )
            
            @backgrounds_slider.on_update
            def _(_) -> None:
                tab_state.backgrounds = backgrounds_slider.value
                if self._viewer:
                    self._viewer.rerender(_)
            
            # Render mode
            render_mode_dropdown = server.gui.add_dropdown(
                "Render Mode",
                ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                initial_value=tab_state.render_mode,
                hint="Render mode to use.",
            )
            
            @render_mode_dropdown.on_update
            def _(_) -> None:
                if "depth" in render_mode_dropdown.value:
                    normalize_nearfar_checkbox.disabled = False
                    inverse_checkbox.disabled = False
                    colormap_dropdown.disabled = False
                else:
                    normalize_nearfar_checkbox.disabled = True
                    inverse_checkbox.disabled = True
                    colormap_dropdown.disabled = True
                tab_state.render_mode = render_mode_dropdown.value
                if self._viewer:
                    self._viewer.rerender(_)
            
            # Depth visualization controls
            normalize_nearfar_checkbox = server.gui.add_checkbox(
                "Normalize Near/Far",
                initial_value=tab_state.normalize_nearfar,
                disabled=True,
                hint="Normalize depth with near/far plane.",
            )
            
            @normalize_nearfar_checkbox.on_update
            def _(_) -> None:
                tab_state.normalize_nearfar = normalize_nearfar_checkbox.value
                if self._viewer:
                    self._viewer.rerender(_)
            
            inverse_checkbox = server.gui.add_checkbox(
                "Inverse",
                initial_value=tab_state.inverse,
                disabled=True,
                hint="Inverse the depth/alpha values.",
            )
            
            @inverse_checkbox.on_update
            def _(_) -> None:
                tab_state.inverse = inverse_checkbox.value
                if self._viewer:
                    self._viewer.rerender(_)
            
            colormap_dropdown = server.gui.add_dropdown(
                "Colormap",
                ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                initial_value=tab_state.colormap,
                hint="Colormap for depth/alpha visualization.",
                disabled=True,
            )
            
            @colormap_dropdown.on_update
            def _(_) -> None:
                tab_state.colormap = colormap_dropdown.value
                if self._viewer:
                    self._viewer.rerender(_)
            
            # Rasterization mode
            rasterize_mode_dropdown = server.gui.add_dropdown(
                "Anti-Aliasing",
                ("classic", "antialiased"),
                initial_value=tab_state.rasterize_mode,
                hint="Rasterization mode.",
            )
            
            @rasterize_mode_dropdown.on_update
            def _(_) -> None:
                tab_state.rasterize_mode = rasterize_mode_dropdown.value
                if self._viewer:
                    self._viewer.rerender(_)
            
            # Camera model
            camera_model_dropdown = server.gui.add_dropdown(
                "Camera Model",
                ("pinhole", "ortho", "fisheye"),
                initial_value=tab_state.camera_model,
                hint="Camera model for rendering.",
            )
            
            @camera_model_dropdown.on_update
            def _(_) -> None:
                tab_state.camera_model = camera_model_dropdown.value
                if self._viewer:
                    self._viewer.rerender(_)
        
        # Store handles for updating stats
        self._gui_handles = {
            "total_gs_count": total_gs_count_number,
            "rendered_gs_count": rendered_gs_count_number,
        }
    
    def pre_step(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        target_frames: torch.Tensor,
        training_state: SplatTrainingState,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """Update viewer state before training step."""
        if self._viewer is None or world_rank != 0:
            return
        
        # Check if paused
        while self._viewer.state == "paused":
            time.sleep(0.01)
        
        # Acquire lock for thread safety
        if self._lock:
            self._lock.acquire()
        
        # Update training state reference
        self._training_state = training_state
        self._step_start_time = time.time()
    
    def post_step(
        self,
        logger: "SplatLogger",
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor,
        target_frames: torch.Tensor,
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayload,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """Update viewer after training step."""
        if self._viewer is None or world_rank != 0:
            return
        
        # Calculate timing stats
        if self._step_start_time is not None:
            step_time = time.time() - self._step_start_time
            num_train_rays_per_step = (
                rendered_frames.shape[0] * rendered_frames.shape[1] * rendered_frames.shape[2]
            )
            num_train_rays_per_sec = num_train_rays_per_step / max(step_time, 1e-10)
            
            # Set rays per second in render_tab_state (required by nerfview)
            self._viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
        
        # Release lock
        if self._lock:
            self._lock.release()
        
        # Update viewer every N steps
        if step % self._update_interval == 0:
            # Calculate number of training rays per step
            num_train_rays_per_step = (
                rendered_frames.shape[0] * rendered_frames.shape[1] * rendered_frames.shape[2]
            )
            
            # Update stats in GUI
            if self._render_tab_state and hasattr(self, '_gui_handles'):
                self._gui_handles["total_gs_count"].value = self._render_tab_state.total_gs_count
                self._gui_handles["rendered_gs_count"].value = self._render_tab_state.rendered_gs_count
            
            # Update viewer (num_train_rays_per_sec must be set in render_tab_state first)
            if hasattr(self._viewer, 'update'):
                self._viewer.update(step, num_train_rays_per_step)
    
    def on_cleanup(
        self,
        logger: "SplatLogger",
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """Cleanup viewer resources."""
        if self._viewer is None or world_rank != 0:
            return
        
        try:
            if hasattr(self._viewer, 'complete'):
                self._viewer.complete()
            logger.info("Viewer training complete", module=self.module_name)
        except Exception as e:
            logger.warning(f"Error during viewer cleanup: {e}", module=self.module_name)
    
    @torch.no_grad()
    def _render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState) -> np.ndarray:
        """
        Internal render function called by the viewer.
        
        Has access to self._renderer and self._training_state.
        
        Args:
            camera_state: Camera state from the viewer
            render_tab_state: Render tab state with parameters
            
        Returns:
            Rendered image as numpy array [H, W, 3] in range [0, 1]
        """
        if self._training_state is None or self._renderer is None:
            # Return black image if not ready
            return np.zeros((600, 800, 3), dtype=np.float32)
        
        # Cast to our custom state type
        if not isinstance(render_tab_state, SplatViewerTabState):
            # Fallback for basic RenderTabState
            return np.zeros((600, 800, 3), dtype=np.float32)
        
        tab_state = render_tab_state
        
        # Get render dimensions from viewer
        width = tab_state.viewer_width
        height = tab_state.viewer_height
        
        # Get camera parameters
        c2w = camera_state.c2w  # [4, 4]
        K = camera_state.get_K((width, height))  # [3, 3]
        
        # Convert to torch tensors
        device = self._training_state.device
        c2w = torch.from_numpy(c2w).float().to(device).unsqueeze(0)  # [1, 4, 4]
        K = torch.from_numpy(K).float().to(device).unsqueeze(0)  # [1, 3, 3]
        
        # Map render mode to gsplat render mode
        RENDER_MODE_MAP: dict[str, Literal["RGB", "D", "ED"]] = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",  # Render RGB and extract alpha later
        }
        render_mode = RENDER_MODE_MAP.get(tab_state.render_mode, "RGB")
        
        # Get rendering parameters
        sh_degree = min(tab_state.max_sh_degree, self._training_state.sh_degree)
        
        # Get background color (convert from RGB 0-255 to 0-1)
        backgrounds = torch.tensor(
            [[c / 255.0 for c in tab_state.backgrounds]], device=device, dtype=torch.float32
        )
        
        # Render
        try:
            renders, info = self._renderer.render(
                splat_state=self._training_state,
                cam_to_worlds=c2w,
                Ks=K,
                width=width,
                height=height,
                sh_degree=sh_degree,
                render_mode=render_mode,
                backgrounds=backgrounds,
                camera_model=tab_state.camera_model,
            )
            
            # Update stats
            tab_state.total_gs_count = len(self._training_state.params["means"])
            if hasattr(info, 'radii') and info.radii is not None:
                tab_state.rendered_gs_count = int((info.radii > 0).sum().item())
            
            # Process output based on render mode
            if tab_state.render_mode == "rgb":
                # RGB rendering
                output = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
                
            elif tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
                # Depth rendering
                depth = renders[0, ..., 0]
                
                # Normalize depth
                if tab_state.normalize_nearfar:
                    depth_norm = (depth - tab_state.near_plane) / (
                        tab_state.far_plane - tab_state.near_plane + 1e-10
                    )
                else:
                    depth_min = depth.min()
                    depth_max = depth.max()
                    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-10)
                
                depth_norm = depth_norm.clamp(0, 1)
                
                # Apply inverse if requested
                if tab_state.inverse:
                    depth_norm = 1 - depth_norm
                
                # Apply colormap
                from nerfview import apply_float_colormap
                output = apply_float_colormap(
                    depth_norm.unsqueeze(-1),
                    tab_state.colormap  # type: ignore
                ).cpu().numpy()
                
            elif tab_state.render_mode == "alpha":
                # Alpha rendering - extract from info
                if hasattr(info, 'alphas') and info.alphas is not None:
                    alpha = info.alphas[0, ..., 0]
                else:
                    # Fallback: compute from rendered colors
                    alpha = renders[0, ..., :3].mean(dim=-1)
                
                # Apply inverse if requested
                if tab_state.inverse:
                    alpha = 1 - alpha
                
                # Apply colormap
                from nerfview import apply_float_colormap
                output = apply_float_colormap(
                    alpha.unsqueeze(-1),
                    tab_state.colormap  # type: ignore
                ).cpu().numpy()
                
            else:
                # Unknown mode: return RGB
                output = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
            
            return output
        
        except InterruptRenderException:
            # This is normal - nerfview interrupts rendering when user interacts
            # Return black image without error message
            return np.zeros((height, width, 3), dtype=np.float32)
            
        except Exception as e:
            # For unexpected exceptions, log the error
            import traceback
            print(f"Error in viewer render: {e}")
            traceback.print_exc()
            return np.zeros((height, width, 3), dtype=np.float32)
