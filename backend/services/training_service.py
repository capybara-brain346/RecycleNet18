import os
import json
from datetime import datetime
from backend.utils.aws import AWSManager
from backend.services.dataset_service import DatasetService
from config import Config
from ultralytics import YOLO
import shutil


class TrainingService:
    def __init__(self):
        self.aws = AWSManager()
        self.dataset_service = DatasetService()
        os.makedirs(Config.TRAINING_OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)

    def start_training(self, dataset_id, hyperparameters):
        job_data = {
            "dataset_id": dataset_id,
            "hyperparameters": hyperparameters,
            "status": "running",
        }

        job_id = self.aws.create_training_job(job_data)

        try:
            self.dataset_service.download_dataset(dataset_id)

            model = YOLO("yolov8n.pt")

            dataset_yaml = os.path.join(
                Config.DATASET_FOLDER, dataset_id, "dataset.yaml"
            )

            training_args = {
                "data": dataset_yaml,
                "epochs": hyperparameters.get("epochs", 100),
                "batch": hyperparameters.get("batch_size", 16),
                "imgsz": hyperparameters.get("image_size", 640),
                "patience": hyperparameters.get("patience", 50),
                "device": hyperparameters.get("device", "cuda"),
                "project": Config.TRAINING_OUTPUT_FOLDER,
                "name": job_id,
                "exist_ok": True,
            }

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
            for epoch in range(len(results.results_dict["metrics/precision(B)"])):
                metrics.append(
                    {
                        "epoch": epoch,
                        "precision": results.results_dict["metrics/precision(B)"][
                            epoch
                        ],
                        "recall": results.results_dict["metrics/recall(B)"][epoch],
                        "mAP50": results.results_dict["metrics/mAP50(B)"][epoch],
                        "mAP50-95": results.results_dict["metrics/mAP50-95(B)"][epoch],
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

            self.aws.training_jobs_table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET #status = :status, model_path = :model_path, s3_model_path = :s3_model_path",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":status": "completed",
                    ":model_path": final_model_path,
                    ":s3_model_path": s3_result["s3_url"] if s3_result else None,
                },
            )

            return job_id

        except Exception as e:
            self.aws.training_jobs_table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET #status = :status, error = :error",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={":status": "failed", ":error": str(e)},
            )
            raise e

    def list_jobs(self):
        response = self.aws.training_jobs_table.scan()
        return response.get("Items", [])

    def get_job_status(self, job_id):
        response = self.aws.training_jobs_table.get_item(Key={"job_id": job_id})
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
