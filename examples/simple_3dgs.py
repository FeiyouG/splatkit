
from typing import Any


from splatkit.trianer import SplatTrainer
from splatkit.data_provider import SplatColmapDataProvider, SplatColmapDataProviderConfig
from splatkit.renderer import Splat3DGSRenderer, Splat3DGSFrame
from splatkit.loss import Splat3dgsSimpleLossFn
    
if __name__ == "__main__":
    data_provider = SplatColmapDataProvider(
        config = SplatColmapDataProviderConfig(
            colmap_dir="/Users/feiyouguo/Downloads/test/crossbag2/new/undistorted/sparse/0",
            images_dir="/Users/feiyouguo/Downloads/test/crossbag2/new/undistorted/images",
            masks_dir="/Users/feiyouguo/Downloads/test/crossbag2/new/undistorted/masks",
            factor=1,
            normalize=True,
            load_depth=True,
            train_test_ratio=0.8,
        )
    )

    renderer = Splat3DGSRenderer()
    loss_func = Splat3dgsSimpleLossFn()

    trainer = SplatTrainer[Splat3DGSFrame](
        renderer=renderer,
        loss_fn=loss_func,
        train_data_provider=data_provider,
        test_data_provider=data_provider,
    )
    trainer.train()