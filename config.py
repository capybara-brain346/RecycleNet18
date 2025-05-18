import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key"

    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

    S3_DATASET_BUCKET = os.environ.get("S3_DATASET_BUCKET", "recyclenet-datasets")
    S3_MODEL_BUCKET = os.environ.get("S3_MODEL_BUCKET", "recyclenet-models")

    DYNAMODB_MODELS_TABLE = "models"
    DYNAMODB_TRAINING_JOBS_TABLE = "training_jobs"
    DYNAMODB_INFERENCE_LOGS_TABLE = "inference_logs"

    SAGEMAKER_ROLE = os.environ.get("SAGEMAKER_ROLE")
    SAGEMAKER_INSTANCE_TYPE = "ml.g5.2xlarge"

    UPLOAD_FOLDER = "/tmp/uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
