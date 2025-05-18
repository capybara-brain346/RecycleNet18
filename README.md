# RecycleNet: Recycling Object Detection API

A Flask-based API for recycling object detection using YOLOv8 and AWS SageMaker, with a Vision-Language Model assistant for recycling guidance.

## Features

- Dataset Management

  - Upload and organize recycling image datasets
  - Store datasets in S3
  - Track dataset metadata in DynamoDB

- Model Training

  - Train YOLOv8 models on AWS SageMaker
  - Configure training hyperparameters
  - Monitor training progress and metrics
  - Store model artifacts in S3

- Model Management

  - List and view trained models
  - Promote models to production
  - Automatic SageMaker endpoint deployment
  - Model versioning and A/B testing

- Inference

  - Real-time object detection using production model
  - Inference logging and tracking
  - Batch prediction support

- VLM Assistant
  - Ask recycling-related questions
  - Support for image and text queries
  - Powered by LLaVA 1.5 7B model

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/recyclenet.git
cd recyclenet
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Set up AWS resources:

- Create S3 buckets for datasets and models
- Create DynamoDB tables:
  - models
  - training_jobs
  - inference_logs
- Configure SageMaker role with necessary permissions

## Running the API

Development:

```bash
flask run
```

Production:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 'app:create_app()'
```

## API Endpoints

### Dataset Management

- `POST /api/v1/datasets/upload` - Upload dataset
- `GET /api/v1/datasets` - List datasets
- `GET /api/v1/datasets/{id}` - Get dataset details

### Model Training

- `POST /api/v1/training/start` - Start training job
- `GET /api/v1/training/jobs` - List training jobs
- `GET /api/v1/training/jobs/{id}` - Get job status
- `GET /api/v1/training/jobs/{id}/metrics` - Get training metrics

### Model Management

- `GET /api/v1/models` - List models
- `GET /api/v1/models/{id}` - Get model details
- `POST /api/v1/models/{id}/promote` - Promote to production
- `GET /api/v1/models/production` - Get production model

### Inference

- `POST /api/v1/predict` - Get predictions
- `GET /api/v1/inference/logs` - Get inference history

### VLM Assistant

- `POST /api/v1/assistant/query` - Submit question

## Example Usage

1. Upload a dataset:

```bash
curl -X POST -F "file=@dataset.zip" http://localhost:5000/api/v1/datasets/upload
```

2. Start model training:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{
       "dataset_id": "your-dataset-id",
       "hyperparameters": {
         "learning_rate": 0.001,
         "batch_size": 16,
         "epochs": 100
       }
     }' \
     http://localhost:5000/api/v1/training/start
```

3. Get predictions:

```bash
curl -X POST -F "image=@recycling.jpg" http://localhost:5000/api/v1/predict
```

4. Ask the VLM Assistant:

```bash
curl -X POST \
     -F "question=How should I recycle this?" \
     -F "image=@item.jpg" \
     http://localhost:5000/api/v1/assistant/query
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
