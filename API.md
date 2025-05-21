# RecycleNet18 API Documentation

This document provides detailed information about the RecycleNet18 API endpoints.

## Table of Contents

- [Models](#models)
- [Inference](#inference)
- [Training](#training)
- [Datasets](#datasets)

## Models

### List Models

Retrieves a list of all available models.

**Endpoint:** `GET /api/v1/models`

### Get Model

Retrieves information about a specific model.

**Endpoint:** `GET /api/v1/models/{model_id}`

### Promote Model

Promotes a model to production status.

**Endpoint:** `POST /api/v1/models/{model_id}/promote`

### Get Production Model

Retrieves the currently active production model.

**Endpoint:** `GET /api/v1/models/production`

## Inference

### Predict

Performs inference using the production model on an uploaded image.

**Endpoint:** `POST /api/v1/predict`

**Request Format:**

- Content-Type: multipart/form-data
- Body:
  - `image` (file): The image file to perform inference on

### Get Inference Logs

Retrieves logs of past inference requests.

**Endpoint:** `GET /api/v1/inference/logs`

## Training

### Start Training Job

Initiates a new training job.

**Endpoint:** `POST /api/v1/training/start`

**Request Format:**

```json
{
  "dataset_id": "string",
  "hyperparameters": {
    "epochs": number,
    "batch_size": number,
    "image_size": number,
    "patience": number,
    "device": string
  }
}
```

### List Training Jobs

Retrieves a list of all training jobs.

**Endpoint:** `GET /api/v1/training/jobs`

### Get Training Job Status

Retrieves the status of a specific training job.

**Endpoint:** `GET /api/v1/training/jobs/{job_id}`

### Get Training Metrics

Retrieves metrics for a specific training job.

**Endpoint:** `GET /api/v1/training/jobs/{job_id}/metrics`

## Datasets

### Upload Dataset

Uploads a new dataset.

**Endpoint:** `POST /api/v1/datasets/upload`

**Request Format:**

- Content-Type: multipart/form-data
- Body:
  - `file` (file): The dataset file to upload

### List Datasets

Retrieves a list of all available datasets.

**Endpoint:** `GET /api/v1/datasets`

### Get Dataset

Retrieves information about a specific dataset.

**Endpoint:** `GET /api/v1/datasets/{dataset_id}`

## Response Status Codes

All endpoints may return the following status codes:

- 200: Success
- 400: Bad Request (missing or invalid parameters)
- 404: Resource Not Found
- 500: Internal Server Error
