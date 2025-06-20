import os
import shutil
from api.utils.aws import AWSManager
from config import Config


class ModelService:
    def __init__(self):
        self.aws = AWSManager()
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)

    def list_models(self):
        response = self.aws.models_table.scan()
        return response.get("Items", [])

    def get_model(self, model_id):
        response = self.aws.models_table.get_item(Key={"models_partition": model_id})
        return response.get("Item")

    def promote_model(self, model_id):
        model = self.get_model(model_id)
        if not model:
            return None

        model_path = model.get("model_path")
        if not model_path or not os.path.exists(model_path):
            return None

        self.aws.promote_model_to_production(model_id)
        return model_id

    def get_production_model(self):
        return self.aws.get_production_model()
