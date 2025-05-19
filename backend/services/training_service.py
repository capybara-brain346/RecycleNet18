from backend.utils.aws import AWSManager
from config import Config
import boto3


class TrainingService:
    def __init__(self):
        self.aws = AWSManager()

    def start_training(self, dataset_id, hyperparameters):
        job_data = {
            "dataset_id": dataset_id,
            "hyperparameters": hyperparameters,
            "instance_type": Config.SAGEMAKER_INSTANCE_TYPE,
        }

        job_id = self.aws.create_training_job(job_data)

        training_params = {
            "TrainingJobName": f"recyclenet-training-{job_id}",
            "AlgorithmSpecification": {
                "TrainingImage": f"{Config.AWS_REGION}.amazonaws.com/pytorch-training:1.8.1-gpu-py36",
                "TrainingInputMode": "File",
            },
            "RoleArn": Config.SAGEMAKER_ROLE,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{Config.S3_DATASET_BUCKET}/{dataset_id}",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": f"s3://{Config.S3_MODEL_BUCKET}/training-output"
            },
            "ResourceConfig": {
                "InstanceType": Config.SAGEMAKER_INSTANCE_TYPE,
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            "HyperParameters": hyperparameters,
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        }

        self.aws.sagemaker.create_training_job(**training_params)
        return job_id

    def list_jobs(self):
        response = self.aws.training_jobs_table.scan()
        return response.get("Items", [])

    def get_job_status(self, job_id):
        response = self.aws.training_jobs_table.get_item(Key={"job_id": job_id})
        job = response.get("Item")

        if not job:
            return None

        sagemaker_job_name = f"recyclenet-training-{job_id}"
        try:
            sagemaker_response = self.aws.sagemaker.describe_training_job(
                TrainingJobName=sagemaker_job_name
            )
            job["sagemaker_status"] = sagemaker_response["TrainingJobStatus"]
            job["metrics"] = sagemaker_response.get("FinalMetricDataList", [])
        except self.aws.sagemaker.exceptions.ResourceNotFound:
            job["sagemaker_status"] = "NotFound"

        return job

    def get_training_metrics(self, job_id):
        cloudwatch = boto3.client(
            "cloudwatch",
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION,
        )

        metrics = cloudwatch.get_metric_data(
            MetricDataQueries=[
                {
                    "Id": "training_loss",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": "AWS/SageMaker",
                            "MetricName": "loss",
                            "Dimensions": [
                                {
                                    "Name": "TrainingJobName",
                                    "Value": f"recyclenet-training-{job_id}",
                                }
                            ],
                        },
                        "Period": 60,
                        "Stat": "Average",
                    },
                }
            ],
            StartTime="-1H",
            EndTime="0H",
        )

        return metrics["MetricDataResults"]
