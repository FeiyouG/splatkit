import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence
from collections import defaultdict

import torch
import numpy as np
import imageio
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import override

from .base import SplatBaseModule
from .frame import SplatRenderPayload
from ..splat.training_state import SplatTrainingState

from ..renderer import SplatRenderer    
from ..logger import SplatLogger


class SplatEvaluator(SplatBaseModule[SplatRenderPayload]):
    """
    Module for evaluating 3D Gaussian Splatting on validation/test set.
    
    Computes PSNR, SSIM, LPIPS metrics and saves results to disk.
    
    Example:
        evaluator = SplatEvaluator(
            output_dir="results/eval",
            eval_steps=[7_000, 30_000],
            save_images=False,  # Default: False, saves to output_dir/images/{step}.png
            save_stats=True,    # Default: True, saves to output_dir/stats/{step}.json
        )
    """
    
    def __init__(
        self,
        output_dir: str,
        eval_steps: list[int] = [],
        save_images: bool = False,
        save_stats: bool = True,
        lpips_net: Literal["alex", "vgg"] = "alex",
        log_to_console: bool = True,
        ckpt_path: str | None = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save results (required).
            eval_steps: Steps at which to run evaluation (e.g., [7000, 30000])
            save_images: Whether to save rendered images (GT | Rendered comparison)
            save_stats: Whether to save metrics as JSON
            lpips_net: LPIPS network type ("alex" or "vgg")
            log_to_console: Whether to print metrics to console
        """
        self._eval_steps = set(eval_steps)
        self._output_dir = output_dir
        self._save_images = save_images
        self._save_stats = save_stats
        self._lpips_net = lpips_net
        self._log_to_console = log_to_console

        # Will be initialized in on_setup
        self._data_provider = None
        self._renderer = None
        self._device = None
        self._train_start_time = None
        
        # Metrics (initialized in on_setup)
        self.psnr = None
        self.ssim = None
        self.lpips = None
    
    @override
    def on_setup(
        self,
        logger: "SplatLogger",
        renderer: SplatBaseModule[SplatRenderPayload],
        data_provider: SplatBaseModule[SplatRenderPayload],
        loss_fn: SplatBaseModule[SplatRenderPayload],
        densification: SplatBaseModule[SplatRenderPayload],
        modules: Sequence[SplatBaseModule[SplatRenderPayload]], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        """Initialize metrics with explicit renderer and data provider."""
        from ..data_provider.base import SplatDataProvider

        # Validate types
        if not isinstance(data_provider, SplatDataProvider):
            raise TypeError(f"data_provider must be SplatDataProvider, got {type(data_provider)}")
        if not isinstance(renderer, SplatRenderer):
            raise TypeError(f"renderer must be SplatRenderer, got {type(renderer)}")
        
        self._data_provider = data_provider
        self._renderer = renderer
        
        # Setup device
        self._device = f"cuda:{world_rank}" if torch.cuda.is_available() else "cpu"
        
        # Track training start time
        self._train_start_time = time.time()
        
        # Initialize metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self._device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self._device)
        
        if self._lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self._device)
        elif self._lpips_net == "vgg":
            # 3DGS official repo uses lpips vgg without normalization
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self._device)
        else:
            raise ValueError(f"Unknown LPIPS network: {self._lpips_net}")
        
        # Create output directories
        if world_rank == 0:
            if self._save_images:
                images_dir = Path(self._output_dir) / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
            
            if self._save_stats:
                stats_dir = Path(self._output_dir) / "stats"
                stats_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initialized. Will evaluate at steps: {sorted(self._eval_steps)}", module=self.module_name)
            logger.info(f"Output dir: {self._output_dir}", module=self.module_name)
    
    @override
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
        """Run evaluation if we're at an eval step."""

        # Only run on rank 0 for single-GPU eval
        if world_rank != 0:
            return

        # Only evaluate if step is in eval_steps (not on last step unless explicitly listed)
        if step in self._eval_steps:
            logger.info(f"Evaluating at step {step}...", module=self.module_name)            
            self._run_eval(step, training_state, world_rank, world_size, logger)
    
    @torch.no_grad()
    def _run_eval(
        self,
        step: int,
        training_state: SplatTrainingState,
        world_rank: int,
        world_size: int,
        logger: "SplatLogger",
    ):
        """Run full evaluation on test set."""
        metrics = defaultdict(list)
        total_render_time = 0.0
        num_images = 0

        if self._data_provider is None:
            raise ValueError("SplatEvaluator requires a SplatDataProvider in modules")
        if self._renderer is None:
            raise ValueError("SplatEvaluator requires a SplatRenderer in modules")
        if self._device is None:
            raise ValueError("SplatEvaluator requires a device")
        if self.psnr is None:
            raise ValueError("PSNR is not initialized")
        if self.ssim is None:
            raise ValueError("SSIM is not initialized")
        if self.lpips is None:
            raise ValueError("LPIPS is not initialized")

        if self._train_start_time is None:
            raise ValueError("Train start time is not set")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        
        # Iterate through test dataset
        test_size = self._data_provider.get_test_data_size(world_rank, world_size)
        
        for i in range(test_size):
            # Get test data
            data = self._data_provider.next_test_data(step=i, world_rank=world_rank, world_size=world_size)
            data = data.to(self._device)
            
            # Extract data
            camtoworlds = data["cam_to_world"]
            Ks = data["K"]
            pixels_gt = data.image  # [B, H, W, 3]
            masks = data.mask if "mask" in data else None
            height, width = pixels_gt.shape[1:3]
            
            # Render with timing
            torch.cuda.synchronize()
            tic = time.time()
            
            renders, _ = self._renderer.render( # type: ignore
                splat_state=training_state,
                cam_to_worlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=training_state.sh_degree,
            )
            
            torch.cuda.synchronize()
            elapsed = time.time() - tic
            total_render_time += elapsed
            num_images += 1
            
            # Clamp renders to [0, 1]
            renders = torch.clamp(renders, 0.0, 1.0)
            
            # Apply masks if present (to match training loss computation)
            if masks is not None:
                mask_3ch = masks.unsqueeze(-1)  # [B, H, W] -> [B, H, W, 1]
                renders = renders * mask_3ch
                pixels_gt = pixels_gt * mask_3ch
            
            # Compute metrics
            pixels_gt_p = pixels_gt.permute(0, 3, 1, 2)  # [B, 3, H, W]
            renders_p = renders.permute(0, 3, 1, 2)  # [B, 3, H, W]
            
            metrics["psnr"].append(self.psnr(renders_p, pixels_gt_p).item())
            metrics["ssim"].append(self.ssim(renders_p, pixels_gt_p).item())
            metrics["lpips"].append(self.lpips(renders_p, pixels_gt_p).item())
            
            # Save images if requested
            if self._save_images:
                # Create side-by-side comparison: GT | Rendered
                canvas = torch.cat([pixels_gt, renders], dim=2)  # [B, H, W*2, 3]
                canvas = canvas.squeeze(0).cpu().numpy()  # [H, W*2, 3]
                canvas = (canvas * 255).astype(np.uint8)
                
                images_dir = Path(self._output_dir) / "images"
                image_path = images_dir / f"{step}_{i:04d}.png"
                imageio.imwrite(image_path, canvas)
        
        # Aggregate metrics (matching gsplat's implementation)
        stats: dict[str, float] = {k: float(np.mean(v)) for k, v in metrics.items()}
        stats["avg_render_time"] = total_render_time / max(num_images, 1)
        stats["num_GS"] = training_state.num_gaussians
        stats["num_images"] = num_images
        
        # Add training time if available
        total_train_time = time.time() - self._train_start_time
        stats["total_train_time"] = total_train_time

        # Peak memory allocated since start (in GB)
        peak_memory_gb = torch.cuda.max_memory_allocated(self._device) / (1024 ** 3)
        # Current memory allocated (in GB)
        current_memory_gb = torch.cuda.memory_allocated(self._device) / (1024 ** 3)
        stats["peak_memory_gb"] = peak_memory_gb
        stats["current_memory_gb"] = current_memory_gb
        
        # Print to console
        if self._log_to_console:
            logger.info("=" * 60, module=self.module_name)
            logger.info(f"Evaluation Results at Step {step}", module=self.module_name)
            logger.info("=" * 60, module=self.module_name)
            logger.info(f"PSNR:  {stats['psnr']:.3f}", module=self.module_name)
            logger.info(f"SSIM:  {stats['ssim']:.4f}", module=self.module_name)
            logger.info(f"LPIPS: {stats['lpips']:.4f}", module=self.module_name)
            
            logger.info(f"Num Gaussians: {stats['num_GS']:,}", module=self.module_name)
            logger.info(f"Num Images: {stats['num_images']}", module=self.module_name)

            logger.info(f"Total Train Time: {stats['total_train_time']:.1f}s ({stats['total_train_time'] / 60:.1f}min)", module=self.module_name)
            logger.info(f"Avg Render time: {stats['avg_render_time']:.3f}s/image", module=self.module_name)

            logger.info(f"Peak Memory: {stats['peak_memory_gb']:.2f}GB", module=self.module_name)
            logger.info(f"Current Memory: {stats['current_memory_gb']:.2f}GB", module=self.module_name)

            logger.info("=" * 60, module=self.module_name)
        
        # Save stats to JSON
        if self._save_stats:
            stats_dir = Path(self._output_dir) / "stats"
            stats_path = stats_dir / f"{step}.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Stats saved to: {stats_path}", module=self.module_name)

