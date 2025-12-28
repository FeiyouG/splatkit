from splatkit.dataset import ColmapDataset

loader = ColmapDataset(
    dir="/Users/feiyouguo/Downloads/test/videos/iron-patriont/sfm/undistorted",
    load_masks=True,
    load_depth=True,
    masks_dir="/Users/feiyouguo/Downloads/test/videos/iron-patriont/sfm/undistorted/masks/object_0",
)

print("total length:", len(loader))
# for item in loader:
#     print("all: ", item.id, item.image_name)

train = loader.split(lambda index, image_name: index % 8 != 0)
test = loader.split(lambda index, image_name: index % 8 == 0)

print("train length:", len(train))
# for item in train:
#     print("train: ", item.id, item.image_name)
print("test length:", len(test))
# for item in test:
#     print("test: ", item.id, item.image_name)