import os
import config
from pydantic import BaseModel


class Modifier(BaseModel):
    """
    Modify and log changes to images in the dataset.

    Attributes:
        path (str): The directory path containing the dataset classes.

    Methods:
        rename_img:
            Rename images in each class directory and log the changes to a specified log file.

    Args:
        log_file: The file to which renaming logs will be written.
    """

    path: str

    def rename_img(self, log_file):
        with open(log_file, "a") as file:
            for classes in os.listdir(self.path):
                class_dir = os.path.join(self.path, classes)
                if os.path.isdir(class_dir):
                    img_counter = 0
                    for img in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img)
                        if os.path.isfile(img_path):
                            new_name = (
                                f"{classes}_{img_counter}{os.path.splitext(img)[1]}"
                            )
                            new_img_path = os.path.join(class_dir, new_name)
                            os.rename(img_path, new_img_path)
                            file.write(f"Renaming {img_path} to {new_img_path}\n")
                            img_counter += 1


if __name__ == "__main__":
    data_path = config.DATA_BASE_DIR
    data_modifier = Modifier(path=data_path)
    data_modifier.rename_img(log_file="modification_logs.txt")
