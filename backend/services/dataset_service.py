import os
from werkzeug.utils import secure_filename
from backend.utils.aws import AWSManager
from config import Config


class DatasetService:
    def __init__(self):
        self.aws = AWSManager()

    def upload_dataset(self, file):
        filename = secure_filename(file.filename)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)

        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

        file.save(temp_path)

        s3_path = self.aws.upload_file_to_s3(temp_path, Config.S3_DATASET_BUCKET)

        os.remove(temp_path)

        return s3_path

    def list_datasets(self):
        response = self.aws.s3.list_objects_v2(Bucket=Config.S3_DATASET_BUCKET)
        datasets = []

        for obj in response.get("Contents", []):
            datasets.append(
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                }
            )

        return datasets

    def get_dataset(self, dataset_id):
        response = self.aws.s3.head_object(
            Bucket=Config.S3_DATASET_BUCKET, Key=dataset_id
        )

        return {
            "key": dataset_id,
            "size": response["ContentLength"],
            "last_modified": response["LastModified"].isoformat(),
            "metadata": response.get("Metadata", {}),
        }
