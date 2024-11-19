from fastapi import FastAPI, File, UploadFile, Response, status, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import datetime
from uuid import uuid4
import inference_utils

app = FastAPI()


def generate_request_id() -> str:
    return f"pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


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
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class PredictionResponse(BaseModel):
    status: PredictionStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Human-readable status message")
    prediction: Optional[PredictionDetail] = None
    metadata: MetaData
    request_id: str = Field(..., description="Unique request identifier")


@app.get("/health")
def health_check(response: Response):
    try:
        response.status_code = status.HTTP_200_OK
        return {"status": "Healthy"}
    except Exception as e:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"404": f"Something Went Wrong: {e}"}


@app.post("/upload")
async def upload_image(response: Response, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        prepredicted_class_index, predicted_class, probablility = (
            inference_utils.classify(image_bytes=contents)
        )

        response.status_code = status.HTTP_200_OK

        return {
            PredictionResponse(
                status=PredictionStatus.SUCCESS,
                message="Image processed successfully",
                prediction=PredictionDetail(
                    class_name=predicted_class,
                    confidence=probablility,
                    class_index=prepredicted_class_index,
                ),
                metadata=MetaData(
                    filename=file.filename,
                    file_size=file.size,
                    mime_type=file.content_type,
                ),
                request_id=generate_request_id(),
            )
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": PredictionStatus.FAILED,
                "message": str(e),
                "metadata": {
                    "filename": file.filename,
                    "processed_at": datetime.utcnow(),
                },
                "request_id": generate_request_id(),
            },
        )
