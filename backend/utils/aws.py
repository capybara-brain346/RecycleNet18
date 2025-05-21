import boto3
from botocore.exceptions import ClientError
from config import Config
import uuid
from datetime import datetime


class AWSManager:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION,
        )

        self.dynamodb = boto3.resource(
            "dynamodb",
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION,
        )

        self.models_table = self.dynamodb.Table(Config.DYNAMODB_MODELS_TABLE)
        self.training_jobs_table = self.dynamodb.Table(
            Config.DYNAMODB_TRAINING_JOBS_TABLE
        )
        self.inference_logs_table = self.dynamodb.Table(
            Config.DYNAMODB_INFERENCE_LOGS_TABLE
        )

    def upload_file_to_s3(self, file_path, bucket, object_name=None):
        if object_name is None:
            object_name = str(uuid.uuid4()) + "_" + file_path.split("/")[-1]

        try:
            self.s3.upload_file(file_path, bucket, object_name)
            s3_url = f"s3://{bucket}/{object_name}"
            return {"s3_url": s3_url, "object_name": object_name}
        except ClientError as e:
            return None

    def create_model_record(self, model_data):
        model_id = str(uuid.uuid4())
        item = {
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "created",
            **model_data,
        }
        self.models_table.put_item(Item=item)
        return model_id

    def create_training_job(self, job_data):
        job_id = str(uuid.uuid4())
        item = {
            "job_id": job_id,
            "start_time": datetime.utcnow().isoformat(),
            **job_data,
        }
        self.training_jobs_table.put_item(Item=item)
        return job_id

    def log_inference(self, inference_data):
        inference_id = str(uuid.uuid4())
        item = {
            "inference_id": inference_id,
            "timestamp": datetime.utcnow().isoformat(),
            **inference_data,
        }
        self.inference_logs_table.put_item(Item=item)
        return inference_id

    def get_production_model(self):
        response = self.models_table.scan(
            FilterExpression="#status = :status",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":status": "production"},
        )
        items = response.get("Items", [])
        return items[0] if items else None

    def promote_model_to_production(self, model_id):
        current_prod = self.get_production_model()
        if current_prod:
            self.models_table.update_item(
                Key={"model_id": current_prod["model_id"]},
                UpdateExpression="SET #status = :status",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={":status": "archived"},
            )

        self.models_table.update_item(
            Key={"model_id": model_id},
            UpdateExpression="SET #status = :status",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":status": "production"},
        )
