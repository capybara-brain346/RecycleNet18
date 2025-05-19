import os
import torch
from werkzeug.utils import secure_filename
import json
from backend.utils.aws import AWSManager
from config import Config
from backend.models.recyclenet import RecycleNet
from PIL import Image
import torchvision.transforms as transforms


class InferenceService:
    def __init__(self):
        self.aws = AWSManager()
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    def predict(self, image_file):
        model = self.aws.get_production_model()
        if not model:
            return None

        filename = secure_filename(image_file.filename)
        temp_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        image_file.save(temp_path)

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = model.get("model_path")

            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            net = RecycleNet()
            net.load_state_dict(torch.load(model_path, map_location=device))
            net = net.to(device)
            net.eval()

            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            image = Image.open(temp_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = net(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predictions = probabilities[0].tolist()

            os.remove(temp_path)

            inference_data = {
                "model_id": model["model_id"],
                "image_path": temp_path,
                "predictions": predictions,
            }
            inference_id = self.aws.log_inference(inference_data)

            return {"inference_id": inference_id, "predictions": predictions}

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def get_inference_logs(self):
        response = self.aws.inference_logs_table.scan()
        return response.get("Items", [])
