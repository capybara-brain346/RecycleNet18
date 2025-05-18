# ♻️ RecycleNet: Product Requirements Document (PRD)

## 1. Overview

**RecycleNet is a cloud-based, end-to-end object detection platform for classifying recyclable materials in user-supplied images and answering recycling questions using a Vision-Language Model assistant.**

The platform provides an intuitive upload, training, deployment, and inference workflow, paired with detailed model metrics, logs, and best-model selection—fully orchestrated from Flask and relying on AWS S3 and DynamoDB as persistent stores.

---

## 2. Goals & Success Criteria

- Enable rapid, reliable, and secure uploading of recycling image datasets.
- Support annotation, format conversion, and model training using YOLOv8.
- Log all model training metrics, hyperparameters, logs, and artifacts reliably.
- Always serve predictions using the best-performing (highest mAP) model, auto-selected at inference time.
- Provide a text/chat interface for educational recycling Q&A via a vision-language model (VLM) assistant.
- Ensure all workflows are accessible through RESTful API endpoints.

---

## 3. Non-Goals

- In-browser annotation or visualization of bounding boxes.
- Non-recycling classification or support for other datasets.
- Real-time data labeling (out of scope; upstream tools like Roboflow possible).
- Multi-tenant support (single-organization for now).
- Hybrid cloud or on-prem deployments.
- Admin dashboard UI (API-only interface).

---

## 4. Stakeholders

- **End Users**: Upload images and receive recycling classification via API.
- **ML Engineers**: Upload and organize datasets, launch and tune model training, promote models via API endpoints.
- **QA/Sustainability Staff**: Use VLM Assistant for recycling guidance or technical support.

---

## 5. Functional Requirements

### 5.1 Dataset Management

- [ ] Users can upload single or bulk image datasets via API endpoints.
- [ ] Uploaded image/data files are directly persisted to AWS S3 under structured prefixes.
- [ ] Metadata for datasets (uploader, description, S3 location, upload date) is stored in DynamoDB using boto3.

### 5.2 Model Training Pipeline

- [ ] API endpoint to select dataset and supply YOLOv8 hyperparameters:
  - Training hyperparameters:
    - Learning rate
    - Batch size
    - Number of epochs
    - Image size
    - Optimizer settings (SGD, Adam)
    - Data augmentation options
    - Model architecture selection (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x)
  - Training configuration:
    - GPU instance type selection
    - Multi-GPU training options
    - Early stopping criteria
    - Model checkpoint frequency
    - Validation interval
- [ ] RESTful API endpoints provide:
  - Dataset selection and preview
  - Hyperparameter configuration
  - Training job status monitoring
  - Real-time training metrics retrieval:
    - Loss curves (training/validation)
    - mAP metrics
    - Learning rate schedule
    - GPU utilization
  - Model version management
  - A/B testing between model versions
- [ ] Launching training creates a ModelRun record and background task for model training
- [ ] Training is executed on Amazon SageMaker using ml.g5.2xlarge instances for GPU acceleration with the following workflow:
  - SageMaker training job automatically pulls dataset from designated S3 bucket
  - Uses PyTorch-based YOLOv8 implementation for multi-object detection
  - Training script handles data loading, augmentation, and model training
  - Model checkpoints saved periodically to S3 during training
  - Final model artifacts and config files saved in PyTorch format (.pt)
- [ ] Training metrics and artifacts:
  - Per-class detection metrics (mAP, precision, recall)
  - Multi-object detection performance metrics
  - Loss curves for objectness and classification
  - Model checkpoints and final weights
  - Training/validation splits and class distribution
  - Hardware utilization metrics (GPU/CPU usage, memory)
- [ ] All metrics, hyperparameters, and metadata reliably written to DynamoDB using boto3
- [ ] Training and debug logs uploaded to S3 and referenced from DynamoDB

### 5.3 Model Metadata & Logs

- [ ] Each model training run produces a record in DynamoDB, with:
  - S3 model path and logs path
  - mAP, accuracy, loss, and all hyperparameters
  - Run timestamps (start/end)
  - Status (training, complete, production, failed)
- [ ] All logs and experiment metadata are accessible via API endpoints.

### 5.4 Model Selection and Promotion

- [ ] API endpoints to:
  - List completed model runs with metrics and logs
  - Promote a model to "production" status
  - Get current production model status
- [ ] Only one model at a time can be in "production" status.

### 5.5 Inference API

- [ ] A `/predict` endpoint is provided.
- [ ] On POST (image upload), the endpoint:
  - Looks up the "production" model with best mAP from DynamoDB
  - Retrieves model artifact from S3 using stored s3_path
  - Creates/updates SageMaker endpoint deployment with the model
  - Runs inference using SageMaker endpoint and returns bounding box predictions as JSON
  - Stores inference data in DynamoDB InferenceLog table including:
    - Input image S3 path
    - Model ID used
    - Prediction results
    - Inference timestamp
    - Inference latency
    - User ID (if authenticated)
- [ ] On promotion of a new model, the endpoint automatically triggers a new SageMaker deployment

### 5.6 VLM Assistant

- [ ] Users can access a Q&A endpoint (`/assistant`).
- [ ] The endpoint accepts image or text questions about recycling.
- [ ] Query is routed to a local VLM assistant (Llava, Gemma, or similar), which returns a relevant answer.
- [ ] The VLM can (optionally) retrieve knowledge from uploaded recycling documents to ground its answers.

---

## 6. Non-Functional Requirements

- **Security:** All endpoints require JWT authentication.
- **Storage Scale:** S3 bucket scales with user uploads and number of models.
- **Reliability:** All metadata and logs are written to DynamoDB (strongly consistent). Model artifacts are redundantly stored in S3.
- **Performance:** Inference is optimized for batch or single-image upload. Production model is cached in memory where possible.
- **Extensibility:** New model types, metrics, or VLM capabilities can be added with minimal schema/API changes.

---

## 7. System Architecture

```plaintext
[Users/ML Engineers]
   │
   ▼
[Flask API Server (EC2)]
   │    • Dataset upload endpoints
   │    • Model training & orchestration
   │    • Model/metrics API
   │    • Inference API
   │    • VLM Q&A assistant
   │
   ├───────────────┬─────────────────┬───────────────┬───────────────┐
   ▼               ▼                 ▼               ▼               ▼
[S3: images] [S3: models/logs] [DynamoDB] [Local VLM API] [SageMaker]
                                                                     • PyTorch container
                                                                     • Multi-object detection
                                                                     • GPU acceleration
```

### Data Lifecycle

- Images—> upload to S3 —> SageMaker training job (fetch from S3) —> `.pt` model to S3—> ModelMeta (metrics, logs) to DynamoDB —> Model retrieved from S3 and deployed to SageMaker endpoint —> Inference
- VLM Assistant runs on EC2, serviced via API endpoint.

---

## 8. DynamoDB Schema

### models Table

Primary Key: model_id (UUID)
Attributes:

- s3_path: String
- hyperparameters: Map
- mAP: Number
- accuracy: Number
- loss: Number
- timestamp: String (ISO format)
- status: String
- logs_path: String

### training_jobs Table

Primary Key: job_id (UUID)
Attributes:

- model_id: UUID
- dataset_id: UUID
- hyperparameters: Map
- start_time: String (ISO format)
- end_time: String (ISO format)
- status: String
- metrics: Map
- checkpoints: List

### inference_logs Table

Primary Key: inference_id (UUID)
Attributes:

- model_id: UUID
- image_s3_path: String
- predictions: Map
- timestamp: String (ISO format)
- latency_ms: Number
- user_id: UUID

---

## 9. API Endpoints

### Dataset Management

- POST `/api/v1/datasets/upload` - Upload dataset
- GET `/api/v1/datasets` - List datasets
- GET `/api/v1/datasets/{id}` - Get dataset details

### Model Training

- POST `/api/v1/training/start` - Start training job
- GET `/api/v1/training/jobs` - List training jobs
- GET `/api/v1/training/jobs/{id}` - Get job status
- GET `/api/v1/training/jobs/{id}/metrics` - Get training metrics

### Model Management

- GET `/api/v1/models` - List models
- GET `/api/v1/models/{id}` - Get model details
- POST `/api/v1/models/{id}/promote` - Promote to production
- GET `/api/v1/models/production` - Get production model

### Inference

- POST `/api/v1/predict` - Get predictions
- GET `/api/v1/inference/logs` - Get inference history

### VLM Assistant

- POST `/api/v1/assistant/query` - Submit question

---

## 10. Acceptance Criteria

- [ ] All API endpoints are documented with OpenAPI/Swagger.
- [ ] Each model training run is recorded in DynamoDB, with complete metrics, S3 paths, and logs.
- [ ] Only "production" model is used for serving predictions, auto-selected by lookup in DynamoDB.
- [ ] Model logs (training/debug) are accessible via API endpoints.
- [ ] Each inference request is logged to DynamoDB with complete metadata and predictions.
- [ ] VLM Assistant responds to queries and retrieves context from uploaded documents or configured sources.
- [ ] Secure, authenticated access for all endpoints using JWT.

---

## 11. Out of Scope

- Web UI/Admin dashboard
- Mobile app UI
- BYO annotation (external tools recommended)
- Real-time streaming inference
- Multi-organization context

---

## 12. Links / References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [AWS DynamoDB Python SDK](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [AWS S3 Python SDK](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [VLM models (e.g., LLaVA, Gemma)](https://llava-vl.github.io/)
- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)

---

**End of Document**
