import os
from werkzeug.utils import secure_filename
from api.utils.aws import AWSManager
from config import Config


class DatasetService:
    def __init__(self):
        self.aws = AWSManager()

    def upload_dataset(self, file, is_annotated=False):
        filename = secure_filename(file.filename)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)

        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

        file.save(temp_path)

        prefix = "annotated-data" if is_annotated else "raw-data"
        s3_path = self.aws.upload_file_to_s3(
            temp_path, Config.S3_BUCKET, f"{prefix}/{filename}"
        )

        os.remove(temp_path)

        return s3_path

    def list_datasets(self, data_type="raw"):
        prefix = "annotated-data" if data_type == "annotated" else "raw-data"
        response = self.aws.s3.list_objects_v2(
            Bucket=Config.S3_BUCKET, Prefix=f"{prefix}/"
        )
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

    def get_dataset(self, dataset_id, data_type="annotated"):
        prefix = "annotated-data" if data_type == "annotated" else "raw-data"
        response = self.aws.s3.head_object(
            Bucket=Config.S3_BUCKET, Key=f"{prefix}/{dataset_id}"
        )

        return {
            "key": dataset_id,
            "size": response["ContentLength"],
            "last_modified": response["LastModified"].isoformat(),
            "metadata": response.get("Metadata", {}),
        }

    def _is_dataset_downloaded(self, dataset_id):
        dataset_path = os.path.join(Config.DATASET_FOLDER, dataset_id)
        if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
            return False

        files = [
            f
            for f in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, f))
        ]
        return len(files) > 0

    def download_dataset(self, dataset_id):
        if self._is_dataset_downloaded(dataset_id):
            dataset_path = os.path.join(Config.DATASET_FOLDER, dataset_id)
            return dataset_path

        dataset_path = os.path.join(Config.DATASET_FOLDER, dataset_id)
        os.makedirs(dataset_path, exist_ok=True)

        try:
            response = self.aws.s3.list_objects_v2(
                Bucket=Config.S3_BUCKET, Prefix=f"annotated-data/{dataset_id}/"
            )

            if not response.get("Contents"):
                raise Exception(f"No files found in S3 for dataset: {dataset_id}")

            downloaded_files = 0
            for obj in response.get("Contents", []):
                s3_key = obj["Key"]

                if s3_key.endswith("/"):
                    continue

                relative_path = s3_key.replace(f"annotated-data/{dataset_id}/", "")

                if relative_path:
                    local_path = os.path.join(dataset_path, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    self.aws.s3.download_file(Config.S3_BUCKET, s3_key, local_path)
                    downloaded_files += 1

            if downloaded_files == 0:
                raise Exception(f"No files were downloaded for dataset: {dataset_id}")

            return dataset_path

        except Exception as e:
            raise Exception(f"Failed to download dataset: {str(e)}")
