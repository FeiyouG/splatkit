from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Literal
from typing_extensions import override

import torch
import numpy as np

from .base import SplatBaseModule
from .frame import SplatRenderPayload
from ..splat.training_state import SplatTrainingState
from torch.utils.tensorboard.writer import SummaryWriter
from ..logger import SplatLogger


class SplatTensorboard(SplatBaseModule[SplatRenderPayload]):
    """
    Tensorboard logging module for splatkit.
    
    Logs training metrics, memory usage, and optionally images during training.
    
    Example:
        >>> from splatkit.modules import SplatTensorboard
        >>> tensorboard = SplatTensorboard(
        ...     output_dir="results/garden/tb",
        ...     log_interval=100,       # Log metrics every 100 steps
        ...     log_images=False,       # Don't log images by default
        ...     log_lr=True,            # Log learning rates
        ...     enable_server=True,     # Auto-start TensorBoard server
        ...     port=6060,              # Access at http://localhost:6060
        ... )
        >>> # Add to trainer's modules list
        >>> # Then open http://localhost:6006 in browser
    """
    
    def __init__(
        self,
        output_dir: str,
        log_interval: int = 100,
        log_images: bool = False,
        log_lr: bool = True,
        log_memory: bool = True,
        comment: str = "",
        enable_server: bool = False,
        port: int = 6060,
    ):
        """
        Initialize tensorboard logger.
        
        Args:
            log_dir: Directory to save tensorboard logs (required)
            log_interval: Log metrics every N steps (default: 100)
            log_images: Whether to log rendered images (GT | Rendered comparison)
            log_lr: Whether to log learning rates for each optimizer
            log_memory: Whether to log GPU memory usage
            comment: Optional comment to append to log directory name
            enable_server: Whether to auto-start TensorBoard server (default: False)
            port: Port for TensorBoard server if enabled (default: 6060)
        """
        super().__init__()
        self._log_path = output_dir + "/tb_logs"
        self._log_interval = log_interval
        self._log_images = log_images
        self._log_lr = log_lr
        self._log_memory = log_memory
        self._comment = comment
        self._enable_server = enable_server
        self._port = port
        
        # Will be initialized in on_setup
        self._writer: "SummaryWriter | None" = None
        self._device = None
        self._training_state = None
        self._server_process = None
        
        # Track last loss for logging
        self._last_loss: float | None = None
        
    @override
    def on_setup(
        self,
        logger: SplatLogger,
        renderer: "SplatBaseModule[SplatRenderPayload]",
        data_provider: "SplatBaseModule[SplatRenderPayload]",
        loss_fn: "SplatBaseModule[SplatRenderPayload]",
        densification: "SplatBaseModule[SplatRenderPayload]",
        modules: Sequence["SplatBaseModule[SplatRenderPayload]"], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        """Initialize tensorboard writer."""
        # Only run on rank 0 for distributed training
        if world_rank != 0:
            return
        
        # Setup device
        self._device = f"cuda:{world_rank}" if torch.cuda.is_available() else "cpu"
        
        # Create log directory
        log_path = Path(self._log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        self._writer = SummaryWriter(
            log_dir=str(log_path),
            comment=self._comment,
        )
        
        logger.info(f"Successfully set up tensorboard writer at: {log_path}, logging every {self._log_interval} steps", module=self.module_name)
        
        # Start TensorBoard server if enabled
        if self._enable_server and world_rank == 0:
            self._start_tensorboard_server(logger)
        else:
            logger.info(f"View logs with: tensorboard --logdir {log_path}", module=self.module_name)
    
    @override
    def post_compute_loss(
        self,
        logger: SplatLogger,
        step: int,
        max_steps: int,
        loss: torch.Tensor,
        training_state: SplatTrainingState,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """Store loss for logging in post_step."""
        if world_rank != 0 or self._writer is None:
            return
        
        self._last_loss = loss.item()
        self._training_state = training_state
    
    @override
    def post_step(
        self,
        logger: SplatLogger,
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
        """Log metrics to tensorboard after each training step."""
        # Only run on rank 0
        if world_rank != 0 or self._writer is None:
            return
        
        # Only log at specified intervals
        if step % self._log_interval != 0:
            return
        
        # Log total loss
        if self._last_loss is not None:
            self._writer.add_scalar("train/loss", self._last_loss, step)
        
        # Log number of Gaussians
        num_gaussians = training_state.num_gaussians
        self._writer.add_scalar("train/num_GS", num_gaussians, step)
        
        # Log memory usage
        if self._log_memory and torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated(self._device) / (1024 ** 3)
            self._writer.add_scalar("train/memory_gb", mem_gb, step)
        
        # Log learning rates
        if self._log_lr and hasattr(training_state, 'optimizers'):
            for name, optimizer in training_state.optimizers.items():
                if hasattr(optimizer, 'param_groups'):
                    lr = optimizer.param_groups[0]['lr']
                    self._writer.add_scalar(f"train/lr_{name}", lr, step)
        
        # Log images (GT | Rendered)
        if self._log_images:
            # Create side-by-side comparison: GT | Rendered
            # rendered_frames: [B, H, W, 3], target_frames: [B, H, W, 3]
            canvas = torch.cat([target_frames, rendered_frames], dim=2)  # [B, H, W*2, 3]
            
            # Take first image from batch and clamp to [0, 1]
            canvas = torch.clamp(canvas[0], 0.0, 1.0)  # [H, W*2, 3]
            
            # Convert to CHW format for tensorboard
            canvas = canvas.permute(2, 0, 1)  # [3, H, W*2]
            
            self._writer.add_image("train/render", canvas, step)
        
        # Flush writer
        self._writer.flush()
    
    def _start_tensorboard_server(self, logger: SplatLogger):
        """Start TensorBoard server in a subprocess."""
        import subprocess
        import sys
        
        try:
            # Start tensorboard server
            self._server_process = subprocess.Popen(
                [
                    sys.executable, "-m", "tensorboard.main",
                    "--logdir", self._log_path,
                    "--port", str(self._port),
                    "--bind_all",  # Allow connections from any IP
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            logger.info(f"TensorBoard server started at: http://localhost:{self._port}", module=self.module_name)
            
        except Exception as e:
            logger.warning(
                f"Failed to start TensorBoard server: {e}. "
                f"You can manually run: tensorboard --logdir {self._log_path} --port {self._port}",
                module=self.module_name
            )
            self._server_process = None
    
    def _stop_tensorboard_server(self, logger: SplatLogger):
        """Stop TensorBoard server if it's running."""
        if self._server_process is not None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
                logger.info("TensorBoard server stopped", module=self.module_name)
            except Exception as e:
                logger.warning(f"Failed to stop TensorBoard server gracefully: {e}", module=self.module_name)
                try:
                    self._server_process.kill()
                except:
                    pass
            finally:
                self._server_process = None
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
    ):
        """
        Log a custom scalar metric.
        
        Args:
            tag: Metric name (e.g., "custom/my_metric")
            value: Metric value
            step: Current training step
        """
        if self._writer is None:
            return
        
        self._writer.add_scalar(tag, value, step)
        self._writer.flush()
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor | np.ndarray,
        step: int,
    ):
        """
        Log a histogram of values.
        
        Args:
            tag: Histogram name (e.g., "gaussian_params/scales")
            values: Values to histogram
            step: Current training step
        """
        if self._writer is None:
            return
        
        self._writer.add_histogram(tag, values, step)
        self._writer.flush()
    
    def log_image(
        self,
        tag: str,
        image: torch.Tensor | np.ndarray,
        step: int,
    ):
        """
        Log an image.
        
        Args:
            tag: Image name (e.g., "debug/depth_map")
            image: Image tensor in [C, H, W] or [H, W, C] format
            step: Current training step
        """
        if self._writer is None:
            return
        
        # Convert to torch tensor if numpy
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        # Convert [H, W, C] to [C, H, W] if needed
        if image.ndim == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)
        
        self._writer.add_image(tag, image, step)
        self._writer.flush()
    
    @override
    def on_cleanup(
        self,
        logger: "SplatLogger",
        world_rank: int = 0,
        world_size: int = 1,
    ):
        """Close tensorboard writer after training."""
        if world_rank != 0:
            return
        
        # Stop TensorBoard server if it was started
        if self._enable_server:
            self._stop_tensorboard_server(logger)
        
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.info("Tensorboard writer closed", module=self.module_name)