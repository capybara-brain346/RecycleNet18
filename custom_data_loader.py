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
    "validate": transforms.Compose(
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
    def __init__(self, path: str, data_transform: transforms):
        super().__init__()
        self.path = path
        img_files = glob.glob(self.path + "/*")

        self.data = {"Image_paths": [], "Class_names": []}
        for classes in img_files:
            class_name = classes.split("\\")[-1]
            for images in glob.glob(classes + "/*"):
                self.data["Image_paths"].append(images)
                self.data["Class_names"].append(class_name)

        # print(self.data)

        self.class_map = {
            class_name: idx for idx, class_name in enumerate(os.listdir(self.path))
        }

        # print(self.class_map)

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
    dataset = ImageLoader(path=path, data_transform=DATA_TRANSFORMS["train"])
    loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    for images, labels in loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print("Labels:", labels)
        break
