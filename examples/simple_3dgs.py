import os
from typing import Any


from splatkit.trainer import SplatTrainer, SplatTrainerConfig
from splatkit.data_provider import SplatColmapDataProvider, SplatColmapDataProviderConfig, ColmapDataItem
from splatkit.renderer import Splat3DGSRenderer, Splat3dgsRenderPayload
from splatkit.loss_fn import SplatDefaultLossFn
from splatkit.modules import SplatExporter, SplatProgressTracker, SplatEvaluator
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
            load_depth=True,
            train_test_ratio=0.8,
        )
    )

    renderer = Splat3DGSRenderer()
    loss_func = SplatDefaultLossFn()
    densification = SplatDefaultDensification()

    modules = [
        SplatProgressTracker(
            update_every=10,  # Update progress bar every step
        ),
        SplatExporter(
            output_dir=os.path.join(work_dir, "export"),
            export_steps=[trainer_config.max_steps],
        ),
        SplatEvaluator(
        output_dir=os.path.join(work_dir, "eval"),
        eval_steps=[trainer_config.max_steps],
    )
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