import os
from typing import Any


from splatkit.trainer import SplatTrainer, SplatTrainerConfig
from splatkit.data_provider import SplatColmapDataProvider, SplatColmapDataProviderConfig, ColmapDataItem
from splatkit.renderer import Splat2DGSRenderer, Splat2dgsRenderPayload
from splatkit.loss_fn import Splat2DGSLossFn
from splatkit.modules import SplatExporter, SplatProgressTracker, SplatEvaluator, SplatViewer
from splatkit.densification import SplatDefaultDensification
    
if __name__ == "__main__":
    work_dir = "/Users/feiyouguo/Downloads/test/crossbag2/new"

    trainer_config = SplatTrainerConfig(
        max_steps=30000,
    )
    
    data_provider = SplatColmapDataProvider(
        config = SplatColmapDataProviderConfig(
            colmap_dir=os.path.join(work_dir, "undistorted/sparse/0"),
            images_dir=os.path.join(work_dir, "undistorted/images"),
            masks_dir=os.path.join(work_dir, "undistorted/masks/object_0"),
            factor=1,
            normalize=True,
            load_depth=False,  # 2DGS typically doesn't use depth supervision
            train_test_ratio=0.8,
        )
    )

    # 2DGS renderer with appropriate settings
    renderer = Splat2DGSRenderer(
        near_plane=0.2,
        far_plane=200.0,
        distloss=True,  # Enable distortion loss computation
    )
    
    # 2DGS loss function with normal consistency and distortion losses
    loss_func = Splat2DGSLossFn(
        ssim_lambda=0.2,
        normal_lambda=0.05,  # Weight for normal consistency loss
        normal_start_iter=7000,  # Start normal loss after warmup
        dist_lambda=0.01,  # Weight for distortion loss
        dist_start_iter=3000,  # Start distortion loss earlier
    )
    
    # Default densification works for both 3DGS and 2DGS (auto-detects gradient key)
    densification = SplatDefaultDensification(
        prune_opa=0.05,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_scale3d=0.1,
        refine_start_iter=500,
        refine_stop_iter=15000,
        reset_every=3000,
        refine_every=100,
    )

    modules = [
        SplatProgressTracker(
            update_every=10,  # Update progress bar every step
        ),
        SplatExporter(
            output_dir=os.path.join(work_dir, "export_2dgs"),
            export_steps=[trainer_config.max_steps],
        ),
        SplatEvaluator(
            output_dir=os.path.join(work_dir, "eval_2dgs"),
            eval_steps=[trainer_config.max_steps],
        ),
        # Uncomment to enable viewer
        # SplatViewer(    
        #     port=8080,
        #     output_dir=os.path.join(work_dir, "viewer_2dgs"),
        #     update_interval=1,
        #     mode="training",
        # ),
    ]
    
    # Filter out None modules
    modules = [m for m in modules if m is not None]

    trainer = SplatTrainer[ColmapDataItem, Splat2dgsRenderPayload](
        config=trainer_config,
        renderer=renderer,
        loss_fn=loss_func,
        data_provider=data_provider,
        densification=densification,
        modules=modules,
    )
    trainer.run()

