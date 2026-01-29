import os
import argparse
from typing import Any


from splatkit.trainer import SplatTrainer, SplatTrainerConfig
from splatkit.data_provider import SplatColmapDataProvider, SplatColmapDataProviderConfig, ColmapDataItem
from splatkit.renderer import Splat3DGSRenderer, Splat3dgsRenderPayload
from splatkit.loss_fn import Splat3DGSLossFn
from splatkit.modules import SplatExporter, SplatProgressor, SplatEvaluator, SplatViewer
from splatkit.densification import SplatDefaultDensification
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D Gaussian Splatting model")
    parser.add_argument("--colmap_dir", type=str, required=True, help="Path to COLMAP sparse reconstruction directory")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--masks_dir", type=str, default=None, help="Path to masks directory (optional)")
    parser.add_argument("--output_dir", type=str, default="./output_3dgs", help="Path to output directory (default: ./output_3dgs)")
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
            load_depth=True,
            train_test_ratio=0.8,
        )
    )

    renderer = Splat3DGSRenderer()
    loss_func = Splat3DGSLossFn()
    
    # Default densification works for both 3DGS and 2DGS (auto-detects gradient key)
    densification = SplatDefaultDensification(
        prune_opa=0.005,  # Default 3DGS setting
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_scale3d=0.1,
        refine_start_iter=500,
        refine_stop_iter=15000,
        reset_every=3000,
        refine_every=100,
    )

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

    trainer = SplatTrainer[ColmapDataItem, Splat3dgsRenderPayload](
        config=trainer_config,
        renderer=renderer,
        loss_fn=loss_func,
        data_provider=data_provider,
        densification=densification,
        modules=modules,
    )
    trainer.run()