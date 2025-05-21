import os
from werkzeug.utils import secure_filename
from api.utils.aws import AWSManager
from config import Config
from ultralytics import YOLO


class InferenceService:
    def __init__(self):
        self.aws = AWSManager()
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    def predict(self, image_file):
        model_info = self.aws.get_production_model()
        if not model_info:
            return None

        filename = secure_filename(image_file.filename)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        image_file.save(temp_path)

        try:
            model_path = model_info.get("model_path")
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            model = YOLO(model_path)
            results = model.predict(temp_path, conf=0.25)

            predictions = []
            for r in results:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    predictions.append(
                        {
                            "bbox": box.tolist(),
                            "confidence": float(conf),
                            "class": int(cls),
                            "class_name": r.names[int(cls)],
                        }
                    )

            annotated_filename = f"annotated_{filename}"
            annotated_path = os.path.join(Config.UPLOAD_FOLDER, annotated_filename)

            for r in results:
                im_array = r.plot()
                r.save(annotated_path)

            s3_path = self.aws.upload_file_to_s3(
                annotated_path, Config.S3_BUCKET, f"inference/{annotated_filename}"
            )

            os.remove(temp_path)
            os.remove(annotated_path)

            inference_data = {
                "model_id": model_info["model_id"],
                "image_path": temp_path,
                "annotated_image_url": s3_path,
                "predictions": predictions,
            }
            inference_id = self.aws.log_inference(inference_data)

            return {
                "inference_id": inference_id,
                "predictions": predictions,
                "model_id": model_info["model_id"],
                "annotated_image_url": s3_path,
            }

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(annotated_path):
                os.remove(annotated_path)
            raise e

    def get_inference_logs(self):
        response = self.aws.inference_logs_table.scan()
        return response.get("Items", [])
