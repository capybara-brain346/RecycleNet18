import os
from werkzeug.utils import secure_filename
import json
from api.utils.aws import AWSManager
from config import Config


class InferenceService:
    def __init__(self):
        self.aws = AWSManager()

    def predict(self, image_file):
        model = self.aws.get_production_model()
        if not model:
            return None

        filename = secure_filename(image_file.filename)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        image_file.save(temp_path)

        s3_path = self.aws.upload_file_to_s3(
            temp_path, Config.S3_DATASET_BUCKET, f"inference/{filename}"
        )

        os.remove(temp_path)

        endpoint_name = "recyclenet-production"
        response = self.aws.sagemaker.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/x-image",
            Body=image_file.read(),
        )

        predictions = json.loads(response["Body"].read().decode())

        inference_data = {
            "model_id": model["model_id"],
            "image_s3_path": s3_path,
            "predictions": predictions,
        }
        inference_id = self.aws.log_inference(inference_data)

        return {"inference_id": inference_id, "predictions": predictions}

    def get_inference_logs(self):
        response = self.aws.inference_logs_table.scan()
        return response.get("Items", [])
