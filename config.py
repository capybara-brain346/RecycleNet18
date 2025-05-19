import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key"

    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.environ.get("AWS_REGION")

    S3_BUCKET = os.environ.get("S3_BUCKET")

    DYNAMODB_MODELS_TABLE = "models"
    DYNAMODB_TRAINING_JOBS_TABLE = "training_jobs"
    DYNAMODB_INFERENCE_LOGS_TABLE = "inference_logs"

    UPLOAD_FOLDER = "uploads"
    MODEL_FOLDER = "models"
    DATASET_FOLDER = "datasets"
    TRAINING_OUTPUT_FOLDER = "training_output"

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
