import os
import polars as pl
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from config import Config
import glob

DATA_TRANSFORMS = {
    "train": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7485, 0.7274, 0.7051], std=[0.2481, 0.2566, 0.2747]
            ),
        ]
    ),
    "validation": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7485, 0.7274, 0.7051], std=[0.2481, 0.2566, 0.2747]
            ),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7485, 0.7274, 0.7051], std=[0.2481, 0.2566, 0.2747]
            ),
        ]
    ),
}


class ImageLoader(Dataset):
    def __init__(self, path: str, data_transform: transforms, split: str):
        super().__init__()
        self.path = path
        img_files = glob.glob(self.path + "/*")

        image_paths, class_names = [], []

        for classes in img_files:
            class_name = classes.split("\\")[-1]
            images = glob.glob(classes + "/*")
            num_images = len(images)

            if split == "train":
                start_idx, end_idx = 0, int(0.6 * num_images)
            elif split == "validation":
                start_idx, end_idx = int(0.6 * num_images), int(0.9 * num_images)
            else:
                start_idx, end_idx = int(0.9 * num_images), num_images

            for idx in range(start_idx, end_idx):
                image_paths.append(images[idx])
                class_names.append(class_name)

        self.image_paths = image_paths
        self.class_names = class_names

        self.class_map = {
            class_name: idx for idx, class_name in enumerate(os.listdir(self.path))
        }

        self.data = {"Image_paths": self.image_paths, "Class_names": self.class_names}
        self.df = pl.DataFrame(self.data)

        # print(self.df)

        self.df = self.df.with_columns(
            self.df["Class_names"]
            .apply(lambda x: self.class_map[x], return_dtype=int)
            .alias("Class_indexes")
            .cast(pl.Int32)
        )

        # self.df.write_json("data_snapshot.json", row_oriented=True)

        self.transform = data_transform

        # print(f"Class mappings -> {self.class_map}")
        # print(f"Image size -> {self.img_size}")

    def __getitem__(self, idx: int):
        image_path, label, label_idx = self.df.row(idx)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label_idx

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    path = Config.DATA_BASE_DIR

    train_dataset = ImageLoader(
        path=path, data_transform=DATA_TRANSFORMS["train"], split="train"
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    for images, labels in train_loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print("Labels:", labels)
        break

    val_dataset = ImageLoader(
        path=path, data_transform=DATA_TRANSFORMS["validation"], split="validation"
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    for images, labels in val_loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print("Labels:", labels)
        break

    test_dataset = ImageLoader(
        path=path, data_transform=DATA_TRANSFORMS["test"], split="test"
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    for images, labels in test_loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print("Labels:", labels)
        break
