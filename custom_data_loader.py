import os
import polars as pl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import config
import glob


class ImageLoader(Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        img_files = glob.glob(self.path + "/*")

        self.data = {"Image_paths": [], "Class_names": []}
        for classes in img_files:
            class_name = classes.split("\\")[-1]
            for images in glob.glob(classes + "/*"):
                self.data["Image_paths"].append(images)
                self.data["Class_names"].append(class_name)

        print(self.data)

        self.class_map = {
            class_name: idx for idx, class_name in enumerate(os.listdir(self.path))
        }

        print(self.class_map)

        self.df = pl.DataFrame(self.data)

        print(self.df)

        self.df = self.df.with_columns(
            self.df["Class_names"]
            .apply(lambda x: self.class_map[x])
            .alias("Class_indexes")
            .cast(pl.Int32)
        )

        self.img_size = (256, 256)

        self.df.write_json("./data_snapshot.json", row_oriented=True)

        self.transform = transforms.Compose(
            [transforms.Resize(self.img_size), transforms.ToTensor()]
        )

        print(f"Class mappings -> {self.class_map}")
        print(f"Image size -> {self.img_size}")

    def __getitem__(self, idx: int):
        image_path, label, label_idx = self.df.row(idx)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label_idx

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    path = config.DATA_BASE_DIR
    dataset = ImageLoader(path=path)
    loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    for images, labels in loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print("Labels:", labels)
        break
