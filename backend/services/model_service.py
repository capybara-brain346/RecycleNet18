from backend.utils.aws import AWSManager
from config import Config


class ModelService:
    def __init__(self):
        self.aws = AWSManager()

    def list_models(self):
        response = self.aws.models_table.scan()
        return response.get("Items", [])

    def get_model(self, model_id):
        response = self.aws.models_table.get_item(Key={"model_id": model_id})
        return response.get("Item")

    def promote_model(self, model_id):
        model = self.get_model(model_id)
        if not model:
            return None

        self.aws.promote_model_to_production(model_id)

        endpoint_name = "recyclenet-production"
        model_path = model["s3_path"]

        self.aws.sagemaker.create_model(
            ModelName=f"recyclenet-model-{model_id}",
            PrimaryContainer={
                "Image": f"{Config.AWS_REGION}.amazonaws.com/pytorch-inference:1.8.1-gpu-py36",
                "ModelDataUrl": model_path,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": Config.AWS_REGION,
                },
            },
            ExecutionRoleArn=Config.SAGEMAKER_ROLE,
        )

        self.aws.sagemaker.create_endpoint_config(
            EndpointConfigName=f"recyclenet-config-{model_id}",
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": f"recyclenet-model-{model_id}",
                    "InstanceType": "ml.g4dn.xlarge",
                    "InitialInstanceCount": 1,
                    "InitialVariantWeight": 1,
                }
            ],
        )

        try:
            self.aws.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=f"recyclenet-config-{model_id}",
            )
        except self.aws.sagemaker.exceptions.ClientError:
            self.aws.sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=f"recyclenet-config-{model_id}",
            )

        return model_id

    def get_production_model(self):
        return self.aws.get_production_model()
