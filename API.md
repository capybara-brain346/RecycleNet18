# RecycleNet18 API Documentation

This document provides detailed information about the RecycleNet18 API endpoints, their functionality, request/response formats, and examples.

## Table of Contents

- [Models](#models)
- [Inference](#inference)
- [Training](#training)
- [Datasets](#datasets)
- [Assistant](#assistant)

## Models

### List Models

Retrieves a list of all available models.

**Endpoint:** `GET /model`

**Response Format:**

```json
{
  "Items": [
    {
      "model_id": "string",
      "timestamp": "string",
      "status": "string",
      "s3_path": "string"
    }
  ]
}
```

**Status Codes:**

- 200: Success
- 500: Internal Server Error

### Get Model

Retrieves information about a specific model.

**Endpoint:** `GET /model/{model_id}`

**Parameters:**

- `model_id` (path): The ID of the model to retrieve

**Response Format:**

```json
{
  "model_id": "string",
  "timestamp": "string",
  "status": "string",
  "s3_path": "string"
}
```

**Status Codes:**

- 200: Success
- 404: Model not found
- 500: Internal Server Error

### Promote Model

Promotes a model to production status and deploys it to the SageMaker endpoint.

**Endpoint:** `POST /model/{model_id}/promote`

**Parameters:**

- `model_id` (path): The ID of the model to promote

**Response Format:**

```json
{
  "message": "Model promoted to production successfully",
  "model_id": "string"
}
```

**Status Codes:**

- 200: Success
- 404: Model not found
- 500: Internal Server Error

### Get Production Model

Retrieves the currently active production model.

**Endpoint:** `GET /model/production`

**Response Format:**

```json
{
  "model_id": "string",
  "timestamp": "string",
  "status": "production",
  "s3_path": "string"
}
```

**Status Codes:**

- 200: Success
- 404: No production model found
- 500: Internal Server Error

## Inference

### Predict

Performs inference using the production model on an uploaded image.

**Endpoint:** `POST /inference`

**Request Format:**

- Content-Type: multipart/form-data
- Body:
  - `image` (file): The image file to perform inference on

**Response Format:**

```json
{
  "inference_id": "string",
  "predictions": {
    // Model-specific prediction output
  }
}
```

**Status Codes:**

- 200: Success
- 400: No image file provided or invalid file
- 404: No production model available
- 500: Internal Server Error

### Get Inference Logs

Retrieves logs of past inference requests.

**Endpoint:** `GET /inference/logs`

**Response Format:**

```json
{
  "Items": [
    {
      "inference_id": "string",
      "timestamp": "string",
      "model_id": "string",
      "image_s3_path": "string",
      "predictions": {
        // Model-specific prediction output
      }
    }
  ]
}
```

**Status Codes:**

- 200: Success
- 500: Internal Server Error

## Training

### Start Training

Initiates a new training job.

**Endpoint:** `POST /training/start`

**Request Format:**

```json
{
  "dataset_id": "string",
  "hyperparameters": {
    // Training-specific hyperparameters
  }
}
```

**Response Format:**

```json
{
  "message": "Training job started successfully",
  "job_id": "string"
}
```

**Status Codes:**

- 200: Success
- 400: Missing required fields
- 500: Internal Server Error

**Authentication Required:** Yes (JWT)

### List Training Jobs

Retrieves a list of all training jobs.

**Endpoint:** `GET /training/jobs`

**Response Format:**

```json
{
  "Items": [
    {
      "job_id": "string",
      "start_time": "string",
      "status": "string",
      "dataset_id": "string",
      "hyperparameters": {
        // Training-specific hyperparameters
      }
    }
  ]
}
```

**Status Codes:**

- 200: Success
- 500: Internal Server Error

**Authentication Required:** Yes (JWT)

### Get Training Job Status

Retrieves the status and details of a specific training job.

**Endpoint:** `GET /training/jobs/{job_id}`

**Parameters:**

- `job_id` (path): The ID of the training job

**Response Format:**

```json
{
  "job_id": "string",
  "start_time": "string",
  "status": "string",
  "dataset_id": "string",
  "hyperparameters": {
    // Training-specific hyperparameters
  },
  "sagemaker_status": "string",
  "metrics": [
    // Training metrics
  ]
}
```

**Status Codes:**

- 200: Success
- 404: Job not found
- 500: Internal Server Error

**Authentication Required:** Yes (JWT)

### Get Training Metrics

Retrieves metrics for a specific training job.

**Endpoint:** `GET /training/jobs/{job_id}/metrics`

**Parameters:**

- `job_id` (path): The ID of the training job

**Response Format:**

```json
[
    {
        "Id": "training_loss",
        "Label": "loss",
        "Timestamps": ["string"],
        "Values": [number]
    }
]
```

**Status Codes:**

- 200: Success
- 500: Internal Server Error

**Authentication Required:** Yes (JWT)

## Datasets

### Upload Dataset

Uploads a new dataset to the system.

**Endpoint:** `POST /dataset/upload`

**Request Format:**

- Content-Type: multipart/form-data
- Body:
  - `file` (file): The dataset file to upload

**Response Format:**

```json
{
  "message": "Dataset uploaded successfully",
  "s3_path": "string"
}
```

**Status Codes:**

- 200: Success
- 400: No file provided or invalid file
- 500: Internal Server Error

### List Datasets

Retrieves a list of all available datasets.

**Endpoint:** `GET /dataset`

**Response Format:**

```json
{
    "datasets": [
        {
            "key": "string",
            "size": number,
            "last_modified": "string"
        }
    ]
}
```

**Status Codes:**

- 200: Success
- 500: Internal Server Error

### Get Dataset

Retrieves information about a specific dataset.

**Endpoint:** `GET /dataset/{dataset_id}`

**Parameters:**

- `dataset_id` (path): The ID of the dataset

**Response Format:**

```json
{
    "key": "string",
    "size": number,
    "last_modified": "string",
    "metadata": {
        // Dataset-specific metadata
    }
}
```

**Status Codes:**

- 200: Success
- 404: Dataset not found
- 500: Internal Server Error

## Assistant

### Query Assistant

Queries the AI assistant with a question and optionally an image.

**Endpoint:** `POST /assistant/query`

**Request Format:**

- Content-Type: multipart/form-data
- Body:
  - `question` (text): The question to ask the assistant
  - `image` (file, optional): An image file related to the question

**Response Format:**

```json
{
    "question": "string",
    "response": "string",
    "has_image": boolean
}
```

**Status Codes:**

- 200: Success
- 400: No question provided
- 500: Internal Server Error
