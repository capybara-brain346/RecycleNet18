import os
import json
from datetime import datetime
from ultralytics import YOLO
import shutil
from decimal import Decimal

from config import Config
from api.utils.aws import AWSManager
from api.services.dataset_service import DatasetService


class TrainingService:
    def __init__(self):
        self.aws = AWSManager()
        self.dataset_service = DatasetService()
        os.makedirs(Config.TRAINING_OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)

    def _convert_to_decimal(self, value):
        if isinstance(value, float):
            return Decimal(str(value))
        elif isinstance(value, dict):
            return {k: self._convert_to_decimal(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._convert_to_decimal(item) for item in value]
        return value

    def start_training(self, dataset_id, hyperparameters={}):
        hyperparameters = self._convert_to_decimal(hyperparameters)

        job_id = self.aws.create_training_job(
            {
                "dataset_id": dataset_id,
                "hyperparameters": hyperparameters,
                "status": "running",
            }
        )

        try:
            self.dataset_service.download_dataset(dataset_id)

            model = YOLO("yolov8n.pt")

            dataset_yaml = os.path.abspath(
                os.path.join(Config.DATASET_FOLDER, dataset_id, "data.yaml")
            )

            training_args = {
                "data": dataset_yaml,
                "epochs": int(hyperparameters.get("epochs", 1)),
                "batch": int(hyperparameters.get("batch_size", 16)),
                "imgsz": int(hyperparameters.get("image_size", 640)),
                "patience": int(hyperparameters.get("patience", 50)),
                "device": hyperparameters.get("device", "cuda"),
                "project": Config.TRAINING_OUTPUT_FOLDER,
                "name": job_id,
                "exist_ok": True,
            }
            if "learning_rate" in hyperparameters:
                training_args["lr0"] = float(hyperparameters["learning_rate"])

            results = model.train(**training_args)

            run_folder = os.path.join(Config.TRAINING_OUTPUT_FOLDER, job_id)
            model_path = os.path.join(run_folder, "weights", "best.pt")
            final_model_path = os.path.join(Config.MODEL_FOLDER, f"{job_id}.pt")

            shutil.copy2(model_path, final_model_path)

            s3_result = self.aws.upload_file_to_s3(
                final_model_path, Config.AWS_S3_BUCKET, f"model-files/{job_id}.pt"
            )

            metrics_path = os.path.join(
                Config.TRAINING_OUTPUT_FOLDER, f"{job_id}_metrics.json"
            )

            metrics = []
            precision_data = results.results_dict["metrics/precision(B)"]
            recall_data = results.results_dict["metrics/recall(B)"]
            map50_data = results.results_dict["metrics/mAP50(B)"]
            map50_95_data = results.results_dict["metrics/mAP50-95(B)"]

            if hasattr(precision_data, "__len__") and len(precision_data) > 0:
                for epoch in range(len(precision_data)):
                    metrics.append(
                        {
                            "epoch": epoch,
                            "precision": self._convert_to_decimal(
                                float(precision_data[epoch])
                            ),
                            "recall": self._convert_to_decimal(
                                float(recall_data[epoch])
                            ),
                            "mAP50": self._convert_to_decimal(float(map50_data[epoch])),
                            "mAP50-95": self._convert_to_decimal(
                                float(map50_95_data[epoch])
                            ),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
            else:
                metrics.append(
                    {
                        "epoch": 0,
                        "precision": self._convert_to_decimal(float(precision_data)),
                        "recall": self._convert_to_decimal(float(recall_data)),
                        "mAP50": self._convert_to_decimal(float(map50_data)),
                        "mAP50-95": self._convert_to_decimal(float(map50_95_data)),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

            model_data = {
                "job_id": job_id,
                "model_path": final_model_path,
                "s3_model_path": s3_result["s3_url"] if s3_result else None,
                "hyperparameters": hyperparameters,
                "metrics": metrics,
            }
            self.aws.create_model_record(model_data)

            self.aws.training_jobs_table.update_item(
                Key={"training_jobs_partition": job_id},
                UpdateExpression="SET #status = :status, model_path = :model_path, s3_model_path = :s3_model_path, metrics = :metrics",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":status": "completed",
                    ":model_path": final_model_path,
                    ":s3_model_path": s3_result["s3_url"] if s3_result else None,
                    ":metrics": metrics,
                },
            )

            return job_id

        except Exception as e:
            self.aws.training_jobs_table.update_item(
                Key={"training_jobs_partition": job_id},
                UpdateExpression="SET #status = :status, #error = :error",
                ExpressionAttributeNames={"#status": "status", "#error": "error"},
                ExpressionAttributeValues={":status": "failed", ":error": str(e)},
            )
            raise e

    def list_jobs(self):
        response = self.aws.training_jobs_table.scan()
        return response.get("Items", [])

    def get_job_status(self, job_id):
        response = self.aws.training_jobs_table.get_item(
            Key={"training_jobs_partition": job_id}
        )
        job = response.get("Item")

        if not job:
            return None

        metrics_path = os.path.join(
            Config.TRAINING_OUTPUT_FOLDER, f"{job_id}_metrics.json"
        )
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                job["metrics"] = json.load(f)

        return job

    def get_training_metrics(self, job_id):
        metrics_path = os.path.join(
            Config.TRAINING_OUTPUT_FOLDER, f"{job_id}_metrics.json"
        )
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                return json.load(f)
        return []
