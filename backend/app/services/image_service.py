from typing import Optional, Tuple
from uuid import UUID, uuid4
from pathlib import Path
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.config import get_settings
from app.core.exceptions import EntityNotFoundError
from app.repositories.session_repo import SessionRepository
from app.repositories.image_repo import ImageRepository
from app.services.image_processing import validate_and_save_image
from app.utils.file_validation import validate_and_read_upload
from app.clients.yolo_client import get_yolo_client, YOLOClient
from app.schemas.image import ImageResponse, DetectedObjectSchema, BoundingBox
from app.utils.memory import trim_process_memory

logger = logging.getLogger("bel_chatbot.image_service")
settings = get_settings()


class ImageService:
    """
    Image Service Layer.
    Orchestrates streaming upload validation, image downscaling, disk persistence,
    and deep-learning object detection.
    """

    def __init__(self, yolo_client: Optional[YOLOClient] = None):
        self.yolo_client = yolo_client or get_yolo_client()

    @staticmethod
    def _map_detection_schemas(detections: list) -> list:
        """Converts raw YOLO detection dictionaries into Pydantic schema representations."""
        return [
            DetectedObjectSchema(
                id=UUID(str(d["id"])),
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=BoundingBox(
                    x_min=d["bbox"]["x_min"],
                    y_min=d["bbox"]["y_min"],
                    x_max=d["bbox"]["x_max"],
                    y_max=d["bbox"]["y_max"],
                ),
                normalized_bbox=BoundingBox(
                    x_min=d["normalized_bbox"]["x_min"],
                    y_min=d["normalized_bbox"]["y_min"],
                    x_max=d["normalized_bbox"]["x_max"],
                    y_max=d["normalized_bbox"]["y_max"],
                ) if "normalized_bbox" in d else None,
            )
            for d in detections
        ]

    async def upload_and_process_image(
        self,
        db: AsyncSession,
        file: UploadFile,
        session_id: Optional[UUID] = None,
    ) -> ImageResponse:
        """
        Validates, persists, and detects objects on an uploaded image.
        1. Ensures session existence or auto-creates it.
        2. Memory-safe streaming chunked read and validation.
        3. Saves image to disk with secure UUID.
        4. Executes YOLOv8 object detection.
        5. Persists records to database.
        """
        target_session_id = session_id or uuid4()
        session = await SessionRepository.get_by_id(db, target_session_id, load_relations=False)
        if not session:
            session = await SessionRepository.create(
                db,
                title=f"Recon ({file.filename or 'Image'})",
                session_id=target_session_id,
            )

        # 1. Memory-safe streaming validation
        file_bytes, detected_mime, ext = await validate_and_read_upload(file)

        # 2. Disk persistence & downscaling
        image_meta = await validate_and_save_image(
            file_bytes=file_bytes,
            original_filename=file.filename or f"image{ext}",
        )
        del file_bytes
        trim_process_memory()

        # 3. Save Image record
        db_image = await ImageRepository.create(
            db,
            image_id=image_meta["id"],
            session_id=target_session_id,
            filename=image_meta["filename"],
            storage_path=image_meta["storage_path"],
            mime_type=image_meta["mime_type"],
            file_size_bytes=image_meta["file_size_bytes"],
            width=image_meta["width"],
            height=image_meta["height"],
        )

        # 4. YOLO Object Detection pass
        detection_results = await self.yolo_client.detect_objects_async(
            image_source=image_meta["storage_path"],
            conf_threshold=settings.CONFIDENCE_THRESHOLD,
            orig_dimensions=(image_meta["width"], image_meta["height"]),
        )

        # 5. Persist detected objects
        await ImageRepository.save_detected_objects(
            db,
            image_id=db_image.id,
            detections=detection_results["detections"],
        )

        detected_schemas = self._map_detection_schemas(detection_results["detections"])

        return ImageResponse(
            id=db_image.id,
            session_id=target_session_id,
            filename=db_image.filename,
            file_size_bytes=db_image.file_size_bytes,
            width=db_image.width,
            height=db_image.height,
            mime_type=db_image.mime_type,
            url=image_meta["url"],
            detected_objects=detected_schemas,
            created_at=db_image.created_at,
        )

    async def rerun_detection(
        self,
        db: AsyncSession,
        image_id: UUID,
        confidence_threshold: float = 0.35,
    ) -> ImageResponse:
        """Re-runs YOLO detection on an existing image with an adjusted confidence slider."""
        image = await ImageRepository.get_by_id(db, image_id, load_objects=False)
        if not image:
            raise EntityNotFoundError(f"Image {image_id} not found.")

        detection_results = await self.yolo_client.detect_objects_async(
            image_source=image.storage_path,
            conf_threshold=confidence_threshold,
            orig_dimensions=(image.width, image.height),
        )

        await ImageRepository.replace_detected_objects(
            db,
            image_id=image.id,
            detections=detection_results["detections"],
        )

        detected_schemas = self._map_detection_schemas(detection_results["detections"])

        return ImageResponse(
            id=image.id,
            session_id=image.session_id,
            filename=image.filename,
            file_size_bytes=image.file_size_bytes,
            width=image.width,
            height=image.height,
            mime_type=image.mime_type,
            url=f"/uploads/{Path(image.storage_path).name}",
            detected_objects=detected_schemas,
            created_at=image.created_at,
        )

    @staticmethod
    def get_image_file_path(filename: str) -> Path:
        """Resolves and validates the local path for an image file."""
        safe_name = Path(filename).name
        file_path = settings.UPLOAD_DIR / safe_name
        if not file_path.exists():
            raise EntityNotFoundError(f"Image {filename} not found.")
        return file_path


_image_service_instance: Optional[ImageService] = None


def get_image_service() -> ImageService:
    global _image_service_instance
    if _image_service_instance is None:
        _image_service_instance = ImageService()
    return _image_service_instance
