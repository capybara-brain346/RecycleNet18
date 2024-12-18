import os
import polars as pl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from config import Config
import glob


class ImageLoader(Dataset):
    """
    A custom Dataset class for loading and transforming images.

    Attributes:
        path (str): The directory path containing the dataset.
        split (str): The dataset split to use ('train', 'validation', or 'test'). Default is None.
        data_transform (transforms): The transformations to apply to the images. Default is None.

    Methods:
        __getitem__:
            Retrieve an image and its corresponding class index by index.

        __len__:
            Return the number of samples in the dataset.
    """

    def __init__(
        self, path: str, split: str = None, data_transform: transforms = None
    ) -> None:
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

        self.transform = (
            data_transform if data_transform is not None else transforms.ToTensor()
        )

        # print(f"Class mappings -> {self.class_map}")
        # print(f"Image size -> {self.img_size}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label, label_idx = self.df.row(idx)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label_idx

    def __len__(self) -> int:
        return len(self.df)


if __name__ == "__main__":
    path = Config.DATA_BASE_DIR
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageLoader(path=path, data_transform=transform, split="train")
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    for images, labels in train_loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print("Labels:", labels)
        break
