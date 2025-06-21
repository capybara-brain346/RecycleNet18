# MLOps Automation with Agents

## Current Workflow Analysis

The project enables a standard MLOps lifecycle for an object detection model (YOLOv8):

1. **Data Management**: `DatasetService` handles uploading raw and annotated datasets to S3 bucket. Can list and download datasets for local use.
2. **Model Training**: `TrainingService` can be manually triggered to start training on specific annotated dataset. Handles training, saves model and metrics, logs to DynamoDB.
3. **Model Management**: `ModelService` tracks trained models. Uses `promote_model` for manual selection of "production" version.
4. **Inference**: `InferenceService` uses designated "production" model for predictions on new images. Saves annotated images and logs inference details.

Key observation: While components are well-defined, end-to-end process requires manual intervention for training decisions, hyperparameter tuning, and model promotion.

## Proposed Agent-Based Architecture

### 1. Training-Trigger Agent

**Responsibility**: Automatically start new training jobs when sufficient new annotated data is ready.

**How it works**:

1. Periodically monitors S3 bucket for new annotated datasets
2. Uses simple rules (e.g., "if new .zip dataset appears, trigger training")
3. Calls `training_service.start_training()` with dataset ID and default hyperparameters

**Benefit**: Eliminates manual training initiation. System self-improves when new data arrives.

### 2. Model-Evaluator-and-Promoter Agent

**Responsibility**: Evaluate new models and auto-promote if they outperform current production model.

**How it works**:

1. Monitors DynamoDB training jobs table for "completed" status
2. Fetches performance metrics for new model
3. Compares with current production model metrics
4. Uses predefined promotion strategy (e.g., "promote if mAP50-95 is 1% higher")
5. Calls `model_service.promote_model()` for better models

**Benefit**: Creates true CI/CD pipeline for models. Better models deploy automatically.

### 3. Active-Learning Agent

**Responsibility**: Identify uncertain predictions and flag for re-annotation.

**How it works**:

1. Periodically scans inference logs
2. Identifies low-confidence predictions
3. Moves "hard" images to re-annotation queue in S3
4. Creates focused dataset of valuable examples

**Benefit**: Optimizes data annotation efforts by focusing on challenging cases.

## Human-in-the-Loop Implementation

To maintain control and visibility while automating:

### 1. Centralized Approval System

- Add `_get_human_approval` method to `AWSManager` class
- Prints proposed action and reasoning
- Requires explicit y/n approval
- Halts workflow on denial

### 2. AWS Interaction Centralization

1. Move all direct AWS SDK calls to `AWSManager`
2. Refactor service files to use new approval-gated methods
3. Ensure all AWS interactions pass through HITL checkpoint

## Implementation Priority

1. Set up HITL mechanism in `api/utils/aws.py`
2. Implement Training-Trigger and Model-Evaluator agents
3. Add Active-Learning agent after initial automation pipeline is stable

## Technology Stack

Can be implemented using:

- Scheduled scripts (cron jobs)
- Event-driven architecture (AWS Lambda)
- S3 Event Notifications
- DynamoDB Streams
