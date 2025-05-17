# ♻️ RecycleNet: Product Requirements Document (PRD)

## 1. Overview

**RecycleNet is a cloud-based, end-to-end object detection platform for classifying recyclable materials in user-supplied images and answering recycling questions using a Vision-Language Model assistant.**

The platform provides an intuitive upload, training, deployment, and inference workflow, paired with detailed model metrics, logs, and best-model selection—fully orchestrated from Django and relying on AWS S3 and DynamoDB as persistent stores.

---

## 2. Goals & Success Criteria

- Enable rapid, reliable, and secure uploading of recycling image datasets.
- Support annotation, format conversion, and model training using YOLOv8.
- Log all model training metrics, hyperparameters, logs, and artifacts reliably.
- Always serve predictions using the best-performing (highest mAP) model, auto-selected at inference time.
- Provide a text/chat interface for educational recycling Q&A via a vision-language model (VLM) assistant.
- Ensure all workflows are usable by both technical and non-technical users through an admin dashboard and web UI.

---

## 3. Non-Goals

- In-browser annotation or visualization of bounding boxes.
- Non-recycling classification or support for other datasets.
- Real-time data labeling (out of scope; upstream tools like Roboflow possible).
- Multi-tenant support (single-organization for now).
- Hybrid cloud or on-prem deployments.

---

## 4. Stakeholders

- **End Users**: Upload images and receive recycling classification.
- **Admins/ML Engineers**: Upload and organize datasets, launch and tune model training, promote models, and monitor system.
- **QA/Sustainability Staff**: Use VLM Assistant for recycling guidance or technical support.

---

## 5. Functional Requirements

### 5.1 Dataset Management

- [ ] Users/Admins can upload single or bulk image datasets via a Django web interface.
- [ ] Uploaded image/data files are directly persisted to AWS S3 under structured prefixes.
- [ ] Metadata for datasets (uploader, description, S3 location, upload date) is visible and queryable in Django Admin.

### 5.2 Model Training Pipeline

- [ ] Admin can select an existing dataset and supply YOLOv8 hyperparameters via web UI.
- [ ] Launching training creates a ModelRun record and background task for model training.
- [ ] Training is executed on Amazon SageMaker using ml.g5.2xlarge instances for GPU acceleration.
- [ ] Training downloads images from S3, runs YOLOv8 in a SageMaker training job, and uploads model artifact to S3.
- [ ] Training status, metrics (mAP, loss, accuracy, precision, recall, F1-score, confusion matrix), hyperparameters (learning rate, batch size, epochs, image size, augmentation settings), training/validation splits, class distribution, timestamps (start, end, checkpoints), hardware utilization metrics (GPU/CPU usage, memory), and S3 paths are reliably written to DynamoDB.
- [ ] Training and debug logs are uploaded to S3 and referenced from DynamoDB.

### 5.3 Model Metadata & Logs

- [ ] Each model training run produces an entry in DynamoDB, keyed by unique model_id, with:
  - S3 model path and logs path
  - mAP, accuracy, loss, and all hyperparameters
  - Run timestamps (start/end)
  - Status (training, complete, production, failed)
- [ ] All logs and experiment metadata are accessible to admins via Django Admin.

### 5.4 Model Selection and Promotion

- [ ] Admins can browse completed model runs and review metrics and logs from Django Admin.
- [ ] Admins can promote a model to "production" via the UI, marking it in DynamoDB.
- [ ] Only one model at a time can be in "production" status.

### 5.5 Inference API

- [ ] A Django endpoint `/predict/` is provided.
- [ ] On POST (image upload), the endpoint:
  - Looks up the "production" model with best mAP from DynamoDB.
  - If not cached, downloads and loads the model from SageMaker endpoint.
  - Runs inference using SageMaker endpoint and returns bounding box predictions as JSON.
- [ ] On promotion of a new model, the endpoint automatically updates the SageMaker deployment.

### 5.6 VLM Assistant

- [ ] Users can access a Q&A endpoint (`/assistant/`) in the Django UI.
- [ ] The endpoint accepts image or text questions about recycling.
- [ ] Query is routed to a local VLM assistant (Llava, Gemma, or similar), which returns a relevant answer.
- [ ] The VLM can (optionally) retrieve knowledge from uploaded recycling documents to ground its answers.

---

## 6. Non-Functional Requirements

- **Security:** All endpoints require Django authentication. Admin permissions restrict upload, training, promotion.
- **Storage Scale:** S3 bucket scales with user uploads and number of models.
- **Reliability:** All metadata and logs are written to DynamoDB (strongly consistent). Model artifacts are redundantly stored in S3.
- **Performance:** Inference is optimized for batch or single-image upload. Production model is cached in memory where possible.
- **Extensibility:** New model types, metrics, or VLM capabilities can be added with minimal schema/API changes.

---

## 7. System Architecture

```plaintext
[Users/Admins]
   │
   ▼
[Django Web/Admin (EC2)]
   │    • Dataset upload
   │    • Model training & orchestration
   │    • Model/metrics dashboard
   │    • Inference API
   │    • VLM Q&A assistant
   │
   ├───────────────┬─────────────────┬───────────────┬───────────────┐
   ▼               ▼                 ▼               ▼               ▼
[S3: images] [S3: models/logs] [DynamoDB: ModelMeta] [Local VLM API] [SageMaker]
```

### Data Lifecycle

- Images—> upload to S3 —> SageMaker training job (fetch from S3) —> `.pt` model to S3—> ModelMeta (metrics, logs) to DynamoDB —> SageMaker endpoint uses best model for inference.
- VLM Assistant runs on EC2, serviced via UI endpoint.

---

## 8. DynamoDB ModelMetadata Table

| model_id | s3_path | hyperparameters | mAP | accuracy | loss | timestamp | status | logs_path |
| -------- | ------- | --------------- | --- | -------- | ---- | --------- | ------ | --------- |

---

## 9. User Flows

### Dataset Upload & Registration

1. User logs in → uploads images.
2. Images stored in S3, dataset metadata tracked by Django.

### Launch Model Training

1. Admin selects dataset/hyperparams.
2. Django triggers background training, saving ModelRun to DynamoDB.

### Model Metrics & Promotion

1. Admin browses ModelRun in Django, reviews metrics/logs (from DynamoDB).
2. "Promote" marks chosen model as production (`status=production`) in DynamoDB.

### Inference (Prediction)

1. User posts image to `/predict/`.
2. Django looks up "production" model/metrics in DynamoDB, loads checkpoint from S3, returns predictions.

### VLM Assistant

1. User submits image/question to `/assistant/`, receives recycling-focused answer.

---

## 10. Acceptance Criteria

- [ ] Django admin supports all dataset management and model promotion workflows.
- [ ] Each model training run is recorded in DynamoDB, with complete metrics, S3 paths, and logs.
- [ ] Only "production" model is used for serving predictions, auto-selected by lookup in DynamoDB.
- [ ] Model logs (training/debug) are accessible/downloadable per run.
- [ ] VLM Assistant responds to queries and retrieves context from uploaded documents or configured sources.
- [ ] Secure, authenticated access for all endpoints.

---

## 11. Out of Scope

- Mobile app UI
- BYO annotation (external tools recommended)
- Real-time streaming inference
- Multi-organization context

---

## 12. Links / References

- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [django-storages (S3)](https://django-storages.readthedocs.io)
- [AWS DynamoDB Python SDK](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html)
- [VLM models (e.g., LLaVA, Gemma)](https://llava-vl.github.io/)
- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)

---

**End of Document**
