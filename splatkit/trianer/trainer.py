from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from ..splat.training_state import SplatTrainingState
from ..renderer.renderer import SplatRenderer
from ..loss import SplatLoss

class SplatTrainer:
    """Main trainer for 3D Gaussian Splatting."""
    
    def __init__(
        self,
        splat_state: SplatTrainingState,
        renderer: SplatRenderer,
        loss_fn: SplatLoss,
        strategy: Union[DefaultStrategy, MCMCStrategy],
        evaluator: Optional[SplatEvaluator] = None,
        optional_modules: Optional[List[OptionalModule]] = None,
        random_background: bool = False,
        sh_degree_interval: int = 1000,  # Increase SH degree every N steps
        device: str = "cuda",
    ):
        """
        Args:
            splat_state: SplatTrainingState managing gaussians and optimizers
            renderer: Renderer for rasterization
            loss_fn: Loss function
            strategy: Densification strategy (from gsplat)
            evaluator: Optional evaluator for validation
            optional_modules: List of optional modules (pose opt, etc.)
            random_background: Use random background during training
            sh_degree_interval: Steps between SH degree increases
            device: Device to train on
        """
        self.splat_state = splat_state
        self.renderer = renderer
        self.loss_fn = loss_fn
        self.strategy = strategy
        self.evaluator = evaluator
        self.optional_modules = optional_modules or []
        self.random_background = random_background
        self.sh_degree_interval = sh_degree_interval
        self.device = device
        
        # Initialize strategy state
        if isinstance(strategy, DefaultStrategy):
            self.strategy_state = strategy.initialize_state(
                scene_scale=splat_state.scene_scale
            )
        elif isinstance(strategy, MCMCStrategy):
            self.strategy_state = strategy.initialize_state()
        
        # Validate strategy
        strategy.check_sanity(splat_state.params, splat_state.optimizers)
    
    def train_step(
        self,
        batch: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Training batch with keys:
                - cam_to_world: [B, 4, 4]
                - K: [B, 3, 3]
                - image: [B, H, W, 3]
                - id: [B] image indices
                - mask (optional): [B, H, W]
            step: Current training step
        
        Returns:
            Dictionary of metrics/losses for logging
        """
        # 1. Preprocess batch (optional modules can modify)
        for module in self.optional_modules:
            batch = module.preprocess_batch(batch)
        
        # 2. Extract batch data
        camtoworlds = batch["cam_to_world"]
        Ks = batch["K"]
        targets = batch["image"]
        height, width = targets.shape[1:3]
        
        # 3. Determine SH degree for this step
        sh_degree = min(step // self.sh_degree_interval, self.splat_state.sh_degree)
        
        # 4. Render
        renders, alphas, info = self.renderer.render(
            splat_state=self.splat_state,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
        )
        
        # 5. Postprocess renders (optional modules)
        for module in self.optional_modules:
            renders = module.postprocess_renders(renders, batch)
        
        # 6. Add random background if enabled
        if self.random_background:
            bkgd = torch.rand(1, 3, device=self.device)
            renders = renders + bkgd * (1.0 - alphas)
        
        # 7. Pre-backward strategy hook
        self.strategy.step_pre_backward(
            params=self.splat_state.params,
            optimizers=self.splat_state.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )
        
        # 8. Compute loss
        loss, loss_dict = self.loss_fn.compute(
            renders=renders,
            targets=targets,
            splat_state=self.splat_state,
            info=info,
            batch=batch,
        )
        
        # 9. Optional module losses
        for module in self.optional_modules:
            module_loss, module_loss_dict = module.compute_loss(
                renders, targets, batch
            )
            loss += module_loss
            loss_dict.update(module_loss_dict)
        
        # 10. Backward
        loss.backward()
        
        # 11. Handle sparse gradients if needed
        if self.renderer.sparse_grad:
            self._convert_to_sparse_gradients(info)
        
        # 12. Optimizer steps
        for optimizer in self.splat_state.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        for module in self.optional_modules:
            for optimizer in module.get_optimizers():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        # 13. Post-backward strategy hook
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_post_backward(
                params=self.splat_state.params,
                optimizers=self.splat_state.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=self.renderer.packed,
            )
        elif isinstance(self.strategy, MCMCStrategy):
            # MCMC needs current LR
            current_lr = self.schedulers[0].get_last_lr()[0]
            self.strategy.step_post_backward(
                params=self.splat_state.params,
                optimizers=self.splat_state.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                lr=current_lr,
            )
        
        # Add metadata
        loss_dict["sh_degree"] = sh_degree
        loss_dict["num_gaussians"] = len(self.splat_state.params["means"])
        
        return loss_dict
    
    def train(
        self,
        dataloader: DataLoader,
        max_steps: int,
        eval_dataset: Optional[Dataset] = None,
        eval_steps: Optional[List[int]] = None,
        save_ckpt_steps: Optional[List[int]] = None,
        save_ply_steps: Optional[List[int]] = None,
        ckpt_dir: Optional[str] = None,
        ply_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        log_every: int = 100,
    ):
        """
        Main training loop.
        
        Args:
            dataloader: Training data loader
            max_steps: Maximum training steps
            eval_dataset: Optional validation dataset
            eval_steps: Steps at which to run evaluation
            save_ckpt_steps: Steps at which to save checkpoints
            save_ply_steps: Steps at which to save PLY files
            ckpt_dir: Directory to save checkpoints
            ply_dir: Directory to save PLY files
            log_dir: Directory for tensorboard logs
            log_every: Log metrics every N steps
        """
        # Setup directories
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        if ply_dir:
            os.makedirs(ply_dir, exist_ok=True)
        
        # Setup tensorboard
        writer = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
        
        # Setup schedulers
        self.schedulers = self._create_schedulers(max_steps)
        
        # Setup optional modules
        for module in self.optional_modules:
            module.setup(num_images=len(dataloader.dataset), device=self.device)
        
        # Training loop
        dataloader_iter = iter(dataloader)
        pbar = tqdm.tqdm(range(max_steps))
        
        for step in pbar:
            # Get batch
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, Tensor) else v 
                    for k, v in batch.items()}
            
            # Train step
            metrics = self.train_step(batch, step)
            
            # Update schedulers
            for scheduler in self.schedulers:
                scheduler.step()
            
            # Logging
            if writer and step % log_every == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.flush()
            
            # Progress bar
            pbar.set_description(
                f"loss={metrics['total']:.3f} | "
                f"sh_deg={metrics['sh_degree']} | "
                f"#GS={metrics['num_gaussians']}"
            )
            
            # Evaluation
            if eval_steps and step in eval_steps and self.evaluator and eval_dataset:
                eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
                eval_metrics = self.evaluator.evaluate(
                    self.splat_state, eval_loader, step
                )
                
                if writer:
                    for k, v in eval_metrics.items():
                        writer.add_scalar(f"eval/{k}", v, step)
            
            # Save checkpoint
            if save_ckpt_steps and step in save_ckpt_steps and ckpt_dir:
                self.splat_state.save_ckpt(
                    f"{ckpt_dir}/ckpt_{step}.pt",
                    step=step,
                    include_optimizers=True,
                )
            
            # Save PLY
            if save_ply_steps and step in save_ply_steps and ply_dir:
                splat_model = self.splat_state.to_splat_model()
                if splat_model is not None:
                    splat_model.save_ply(f"{ply_dir}/point_cloud_{step}.ply")
        
        if writer:
            writer.close()
    
    def _create_schedulers(self, max_steps: int) -> List:
        """Create LR schedulers for main optimizers and optional modules."""
        schedulers = []
        
        # Main means scheduler (exponential decay to 1%)
        schedulers.append(
            torch.optim.lr_scheduler.ExponentialLR(
                self.splat_state.optimizers["means"],
                gamma=0.01 ** (1.0 / max_steps),
            )
        )
        
        # Optional module schedulers
        for module in self.optional_modules:
            schedulers.extend(module.get_schedulers(max_steps))
        
        return schedulers
    
    def _convert_to_sparse_gradients(self, info: Dict):
        """Convert dense gradients to sparse (for packed mode)."""
        gaussian_ids = info["gaussian_ids"]
        for k in self.splat_state.params.keys():
            grad = self.splat_state.params[k].grad
            if grad is None or grad.is_sparse:
                continue
            self.splat_state.params[k].grad = torch.sparse_coo_tensor(
                indices=gaussian_ids[None],
                values=grad[gaussian_ids],
                size=self.splat_state.params[k].size(),
                is_coalesced=True,
            )