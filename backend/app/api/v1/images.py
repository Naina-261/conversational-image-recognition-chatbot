from fastapi import APIRouter, UploadFile, File, Form, Depends, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID, uuid4
from typing import Optional

from app.core.database import get_db
from app.services.image_service import get_image_service, ImageService
from app.schemas.image import ImageResponse

router = APIRouter()


@router.post("/upload", response_model=ImageResponse, status_code=status.HTTP_201_CREATED)
async def upload_image_endpoint(
    file: UploadFile = File(..., description="Image binary file (JPEG, PNG, WebP)"),
    session_id: UUID = Form(default_factory=uuid4, description="Target conversation session UUID"),
    db: AsyncSession = Depends(get_db),
    image_service: ImageService = Depends(get_image_service),
):
    """
    Ingests an uploaded image with memory-safe streaming verification,
    stores it with a unique UUID, and runs deep-learning YOLO object detection.
    """
    return await image_service.upload_and_process_image(
        db=db,
        file=file,
        session_id=session_id,
    )


@router.post("/{image_id}/detect", response_model=ImageResponse, status_code=status.HTTP_200_OK)
async def rerun_detection_endpoint(
    image_id: UUID,
    confidence_threshold: float = Query(
        default=0.35, ge=0.01, le=1.0, description="Detection confidence threshold"
    ),
    db: AsyncSession = Depends(get_db),
    image_service: ImageService = Depends(get_image_service),
):
    """
    Re-runs YOLO deep-learning object detection on an existing image
    with a modified confidence slider threshold.
    """
    return await image_service.rerun_detection(
        db=db,
        image_id=image_id,
        confidence_threshold=confidence_threshold,
    )


@router.get("/{filename}/file")
async def get_image_file(
    filename: str,
    image_service: ImageService = Depends(get_image_service),
):
    """Streams the raw binary image from local persistent storage."""
    file_path = image_service.get_image_file_path(filename)
    return FileResponse(path=str(file_path))
