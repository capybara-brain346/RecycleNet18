# RecycleNet: Recyclable Items Classification and Chatbot Guide
![diagram-export-5-17-2025-10_13_05-PM](https://github.com/user-attachments/assets/8ef414ef-da44-4d7e-bae4-2c35dc1ed5ad)

---

## Overview

RecycleNet is a cloud-based, end-to-end object detection platform for classifying recyclable materials in user-supplied images and answering recycling questions using a Vision-Language Model assistant. The platform provides an intuitive upload, training, deployment, and inference workflow, paired with detailed model metrics, logs, and best-model selection—fully orchestrated from Django and relying on AWS S3 and DynamoDB as persistent stores.

## Features

### Object Detection & Classification

- Upload single or bulk image datasets via Django web interface
- YOLOv8-based model training on AWS SageMaker using ml.g5.2xlarge instances
- Comprehensive model metrics tracking including mAP, accuracy, precision, recall, and F1-score
- Automatic model selection and deployment based on best performance
- Real-time inference via SageMaker endpoints

### VLM-Powered Recycling Assistant

- Interactive Q&A interface for recycling guidance
- Support for both image and text-based queries
- Knowledge-grounded responses using recycling documentation
- Educational content and best practices sharing

### Admin Dashboard

- Dataset management and organization
- Model training orchestration and monitoring
- Performance metrics visualization
- Model promotion and deployment controls

## System Architecture

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

## Technical Details

### Model Training

- YOLOv8 object detection model trained on AWS SageMaker
- GPU-accelerated training using ml.g5.2xlarge instances
- Automated metrics logging and model artifact storage
- Best model selection based on mAP performance

### Infrastructure

- **Storage**: AWS S3 for datasets, model artifacts, and logs
- **Database**: DynamoDB for model metadata and metrics
- **Compute**: AWS SageMaker for training and inference
- **Web Interface**: Django-based admin dashboard and API
- **VLM Assistant**: Local deployment of vision-language model

### API Endpoints

- `/predict/`: Object detection inference endpoint
- `/assistant/`: VLM-based recycling Q&A endpoint
- Secure authentication required for all endpoints

## Installation & Setup

1. **Configure AWS Credentials**

   ```bash
   aws configure
   ```

2. **Clone Repository**

   ```bash
   git clone https://github.com/yourusername/recyclenet.git
   cd recyclenet
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   ```bash
   cp .env.example .env
   # Edit .env with your AWS and Django settings
   ```

5. **Run Development Server**
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

## Usage

### For Users

1. Upload images of recyclable items through the web interface
2. Receive object detection results with bounding boxes and classifications
3. Use the VLM Assistant for recycling guidance and questions

### For Administrators

1. Manage datasets and launch training jobs
2. Monitor model performance and training metrics
3. Review and promote best-performing models to production
4. Access comprehensive logs and debugging information

## Future Enhancements

- Multi-tenant support
- Real-time data labeling integration
- Hybrid cloud deployment options
- Mobile application interface
- Enhanced VLM capabilities with more knowledge sources

## License

This project is licensed under the [MIT License](LICENSE).
