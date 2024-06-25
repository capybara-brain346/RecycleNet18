import os
import config
import pandas as pd
from pydantic import BaseModel


class Validator(BaseModel):
    path: str

    def log_classes(self):
        classes_list: list[str] = os.listdir(self.path)
        classes_df = pd.DataFrame(classes_list, columns=["Class_Names"])
        return classes_df

    def log_image_count(self):
        classes_list: list[str] = os.listdir(self.path)
        img_counts = []
        for classes in os.listdir(self.path):
            img_counts.append(len(os.listdir(os.path.join(self.path, classes))))
        # print(lens(classes_list), len(img_counts))
        img_counts_df = pd.DataFrame()
        img_counts_df["Class_Names"] = classes_list
        img_counts_df["Class_Counts"] = img_counts
        return img_counts_df


if __name__ == "__main__":
    data_path = config.DATA_BASE_DIR
    data_validation = Validator(path=data_path)
    data_validation.log_classes()
    data_validation.log_image_count()
