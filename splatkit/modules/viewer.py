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
    """Extended render tab state with gsplat-specific controls.
    
    NOTE: 
        Only includes viewer-controllable parameters. 
        Renderer configuration (like anti-aliasing, distortion loss, etc.) 
        is fixed at renderer construction and cannot be changed here.
    """
    
    # Non-controllable parameters (stats)
    total_gs_count: int = 0
    rendered_gs_count: int = 0
    
    # Controllable viewer parameters
    max_sh_degree: int = 3
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[int, int, int] = (0, 0, 0)  # RGB values 0-255
    render_mode: str = "rgb"  # Dynamic based on detected payload capabilities
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    
    # Camera model (only used if renderer's render() method accepts it)
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"


class SplatViewer(SplatBaseModule[SplatRenderPayload]):
    """
    Interactive real-time viewer for visualizing training progress.
    
    Example:
        >>> from splatkit.modules import SplatViewer
        >>> viewer = SplatViewer(
        ...     port=8080,
        ...     update_interval=5,  # Update every 5 steps
        ...     mode="training",
        ... )
        >>> # Add to trainer's modules list
        >>> # Then open http://localhost:8080 in browser
    
    NOTE:
        - Viewer only runs on rank 0 in distributed training
        - High update frequency (interval=1) may slow down training
        - Rendering happens on GPU; may compete with training for memory
    """
    
    def __init__(
        self,
        port: int = 8080,
        output_dir: str | None = None,
        update_interval: int = 1,
        mode: Literal["training", "rendering"] = "training",
        verbose: bool = False,
    ):
        """
        Initialize the viewer module.
        
        Args:
            port: Web server port number (default: 8080)
            output_dir: Directory to save screenshots and exports; if None, nothing will be saved
            update_interval: Update viewer every N training steps (default: 1)
            mode: Viewer mode:
                  - "training": Updates during training, shows live progress
                  - "rendering": Static viewing, no training updates
            verbose: Print viser server logs (default: False)
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
        
        # Available render modes (populated in on_setup from renderer)
        self._available_render_modes: tuple[str, ...] = ("rgb", "depth(accumulated)", "depth(expected)", "alpha")
        
        # Timing for rays per second calculation
        self._step_start_time: float | None = None

        self._verbose = verbose
        
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
        
        # Get available visualization options from renderer
        self._available_render_modes = renderer.get_visualization_options()
        logger.info(f"Setting up viewer with available visualization modes: {', '.join(self._available_render_modes)}", module=self.module_name)
        
        # Initialize viser server (silence npm/node output)
        try:
            import sys
            import os
            import logging
            
            # Suppress websockets.server error logs for normal connection closures
            # These are harmless ConnectionClosedOK exceptions that occur when clients disconnect
            websockets_logger = logging.getLogger("websockets.server")
            websockets_logger.setLevel(logging.CRITICAL)  # Only show critical errors, not normal disconnects
            
            # Save original file descriptors
            stdout_fd = sys.stdout.fileno()
            stderr_fd = sys.stderr.fileno()
            
            # Duplicate original file descriptors for later restoration
            stdout_dup = os.dup(stdout_fd)
            stderr_dup = os.dup(stderr_fd)
            
            try:
                if not self._verbose:
                    logger.info("Initializing viser server with npm/node output silenced", module=self.module_name)
                    # Redirect stdout and stderr to devnull
                    devnull_fd = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull_fd, stdout_fd)
                    os.dup2(devnull_fd, stderr_fd)
                    os.close(devnull_fd)
                else:
                    logger.info("Initializing viser server with npm/node output enabled", module=self.module_name)
                
                # Initialize viser server (npm/node output now silenced)
                self._server = viser.ViserServer(port=self._port, verbose=False)
                self._server.gui.set_panel_label("splatkit viewer")
                
            finally:
                if not self._verbose:
                    # Restore original file descriptors
                    os.dup2(stdout_dup, stdout_fd)
                    os.dup2(stderr_dup, stderr_fd)
                    os.close(stdout_dup)
                    os.close(stderr_dup)
            
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
            
            # Render mode (dynamic based on detected capabilities)
            render_mode_dropdown = server.gui.add_dropdown(
                "Render Mode",
                self._available_render_modes,
                initial_value=tab_state.render_mode if tab_state.render_mode in self._available_render_modes else "rgb",
                hint="Render mode to use.",
            )
            
            @render_mode_dropdown.on_update
            def _(_) -> None:
                mode = render_mode_dropdown.value
                
                # Default all to disabled
                normalize_nearfar_checkbox.disabled = True
                inverse_checkbox.disabled = True
                colormap_dropdown.disabled = True
                
                # Enable controls based on render mode
                if "depth" in mode or mode == "distortion":
                    normalize_nearfar_checkbox.disabled = False
                    inverse_checkbox.disabled = False
                    colormap_dropdown.disabled = False
                elif mode == "alpha":
                    inverse_checkbox.disabled = False
                    colormap_dropdown.disabled = False
                # normal mode and rgb mode keep all disabled
                
                tab_state.render_mode = mode
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
            
            # Camera model control (all renderers support this parameter)
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
        
        # Get rendering parameters
        sh_degree = min(tab_state.max_sh_degree, self._training_state.sh_degree)
        
        # Get background color (convert from RGB 0-255 to 0-1)
        device = self._training_state.device
        backgrounds = torch.tensor(
            [[c / 255.0 for c in tab_state.backgrounds]], device=device, dtype=torch.float32
        )
        
        # Delegate to renderer's visualize method
        try:
            output, rendered_gaussians = self._renderer.visualize(
                splat_state=self._training_state,
                camera_state=camera_state,
                width=width,
                height=height,
                visualization_mode=tab_state.render_mode,
                sh_degree=sh_degree,
                backgrounds=backgrounds,
                camera_model=tab_state.camera_model,
                normalize_nearfar=tab_state.normalize_nearfar,
                near_plane=tab_state.near_plane,
                far_plane=tab_state.far_plane,
                inverse=tab_state.inverse,
                colormap=tab_state.colormap,
            )
            
            # Update stats
            tab_state.total_gs_count = len(self._training_state.params["means"])
            tab_state.rendered_gs_count = rendered_gaussians
            
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
