from fastapi import FastAPI, File, UploadFile, Response, status, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime
from uuid import uuid4
from .inference_utils import classify

app = FastAPI(debug=True)

# generates a unique request id using timestamp and uuid
def generate_request_id() -> str:
    return f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


class PredictionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PROCESSING = "processing"


class PredictionDetail(BaseModel):
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score between 0 and 1"
    )
    class_index: Optional[int] = Field(None, description="Index of the predicted class")


class MetaData(BaseModel):
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="File MIME type")
    processed_at: datetime = Field(default_factory=datetime.now)


class PredictionResponse(BaseModel):
    status: PredictionStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Human-readable status message")
    prediction: Optional[PredictionDetail] = None
    metadata: MetaData
    request_id: str = Field(..., description="Unique request identifier")


# generates a standardized json response containing prediction results and metadata
def generate_response(
    predicted_class: str,
    probability: float,
    predicted_class_index: Optional[int],
    file,
):
    response = PredictionResponse(
        status=PredictionStatus.SUCCESS,
        message="Image processed successfully",
        prediction=PredictionDetail(
            class_name=predicted_class,
            confidence=probability,
            class_index=predicted_class_index,
        ),
        metadata=MetaData(
            filename=file.filename,
            file_size=file.size,
            mime_type=file.content_type,
        ),
        request_id=generate_request_id(),
    )
    return response.model_dump_json()


# health check endpoint to verify api status
@app.get("/health")
def health_check(response: Response):
    try:
        response.status_code = status.HTTP_200_OK
        return {"status": "Healthy"}
    except Exception as e:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"404": f"Something Went Wrong: {e}"}


# main endpoint for image upload and classification
# accepts image file, returns prediction results with metadata
@app.post("/upload")
async def upload_image(response: Response, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        predicted_class_index, predicted_class, probability = classify(
            image_bytes=contents
        )
        response.status_code = status.HTTP_200_OK
        response_body = PredictionResponse(
            status=PredictionStatus.SUCCESS,
            message="Image processed successfully",
            prediction=PredictionDetail(
                class_name=predicted_class,
                confidence=probability,
                class_index=predicted_class_index,
            ),
            metadata=MetaData(
                filename=file.filename,
                file_size=file.size,
                mime_type=file.content_type,
            ),
            request_id=generate_request_id(),
        )
        return response_body.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": PredictionStatus.FAILED,
                "message": str(e),
                "metadata": {
                    "filename": file.filename,
                    "processed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                },
                "request_id": generate_request_id(),
            },
        )
