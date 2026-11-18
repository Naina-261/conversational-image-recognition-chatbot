"""
Pydantic Schemas Package for Request and Response validation.
"""
from app.schemas.common import ErrorResponse
from app.schemas.session import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionDetailResponse,
    SessionClearResponse,
)
from app.schemas.image import (
    BoundingBox,
    DetectedObjectSchema,
    ImageResponse,
    DetectionRequest,
)
from app.schemas.chat import (
    ChatMessageRequest,
    MessageResponse,
    ChatCompletionResponse,
)

__all__ = [
    "ErrorResponse",
    "SessionCreate",
    "SessionUpdate",
    "SessionResponse",
    "SessionDetailResponse",
    "SessionClearResponse",
    "BoundingBox",
    "DetectedObjectSchema",
    "ImageResponse",
    "DetectionRequest",
    "ChatMessageRequest",
    "MessageResponse",
    "ChatCompletionResponse",
]
