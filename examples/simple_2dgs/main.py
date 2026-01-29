import os
import argparse
from typing import Any


from splatkit.trainer import SplatTrainer, SplatTrainerConfig
from splatkit.data_provider import SplatColmapDataProvider, SplatColmapDataProviderConfig, ColmapDataItem
from splatkit.renderer import Splat2DGSRenderer, Splat2dgsRenderPayload
from splatkit.loss_fn import Splat2DGSLossFn
from splatkit.modules import SplatExporter, SplatProgressor, SplatEvaluator, SplatViewer
from splatkit.densification import SplatDefaultDensification
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 2D Gaussian Splatting model")
    parser.add_argument("--colmap_dir", type=str, required=True, help="Path to COLMAP sparse reconstruction directory")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--masks_dir", type=str, default=None, help="Path to masks directory (optional)")
    parser.add_argument("--output_dir", type=str, default="./output_2dgs", help="Path to output directory (default: ./output_2dgs)")
    parser.add_argument("--max_steps", type=int, default=30000, help="Maximum number of steps (default: 30000)")
    args = parser.parse_args()

    trainer_config = SplatTrainerConfig(
        max_steps=args.max_steps,
    )
    
    data_provider = SplatColmapDataProvider(
        config = SplatColmapDataProviderConfig(
            colmap_dir=args.colmap_dir,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
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
    loss_func = Splat2DGSLossFn()
    
    # Default densification works for both 3DGS and 2DGS (auto-detects gradient key)
    densification = SplatDefaultDensification()

    modules = [
        SplatProgressor(
            update_every=10,  # Update progress bar every step
        ),
        SplatExporter(
            output_dir=os.path.join(args.output_dir, "export"),
            export_steps=[trainer_config.max_steps],
        ),
        SplatEvaluator(
            output_dir=os.path.join(args.output_dir, "eval"),
            eval_steps=[trainer_config.max_steps],
        ),
        # Uncomment to enable viewer
        # Make sure viewer dependency is installed: pip install -e ".[viewer]"
        # SplatViewer(    
        #     port=8080,
        #     output_dir=os.path.join(args.output_dir, "viewer"),
        #     update_interval=1,
        #     mode="training",
        # ),
    ]

    trainer = SplatTrainer[ColmapDataItem, Splat2dgsRenderPayload](
        config=trainer_config,
        renderer=renderer,
        loss_fn=loss_func,
        data_provider=data_provider,
        densification=densification,
        modules=modules,
    )
    trainer.run()

