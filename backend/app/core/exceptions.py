from typing import Optional, Dict, Any
from datetime import datetime, timezone
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger("bel_chatbot.exceptions")


class AppException(Exception):
    """Base application exception with status codes and structured details."""
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type or self.__class__.__name__
        self.details = details or {}


class ImageProcessingError(AppException):
    """Raised when uploaded image is invalid, corrupt, oversized, or unreadable."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_type="ImageProcessingError",
            details=details,
        )


class ModelInferenceError(AppException):
    """Raised when deep-learning model prediction fails unexpectedly."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type="ModelInferenceError",
            details=details,
        )


class EntityNotFoundError(AppException):
    """Raised when a requested resource (session, image, message) is not found."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_type="EntityNotFoundError",
            details=details,
        )


class ValidationError(AppException):
    """Raised when input parameters fail domain validation rules."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_type="ValidationError",
            details=details,
        )


class ExternalAPIError(AppException):
    """Raised when upstream AI inference services (Gemini, HuggingFace) fail."""
    def __init__(self, message: str, status_code: int = status.HTTP_502_BAD_GATEWAY, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="ExternalAPIError",
            details=details,
        )


class RateLimitExceededError(AppException):
    """Raised when external AI providers return HTTP 429 quota exhaustion."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_type="RateLimitExceededError",
            details=details,
        )


def _format_error_payload(
    status_code: int,
    message: str,
    error_type: str,
    details: Dict[str, Any],
    path: str,
) -> Dict[str, Any]:
    """Formats standardized error dictionary satisfying both new schema and client contract."""
    return {
        "success": False,
        "status_code": status_code,
        "error": message,
        "message": message,
        "error_type": error_type,
        "details": details,
        "path": path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handles all custom application domain exceptions."""
    logger.error(f"[{exc.error_type}] {exc.message} on {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content=_format_error_payload(
            status_code=exc.status_code,
            message=exc.message,
            error_type=exc.error_type,
            details=exc.details,
            path=str(request.url.path),
        ),
    )


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handles FastAPI Pydantic input validation failures."""
    errors = exc.errors()
    logger.warning(f"[ValidationError] {len(errors)} validation failure(s) on {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_format_error_payload(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Request validation failed. Check input parameters.",
            error_type="ValidationError",
            details={"errors": errors},
            path=str(request.url.path),
        ),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handles Starlette/FastAPI standard HTTPExceptions."""
    logger.warning(f"[HTTPException] {exc.status_code}: {exc.detail} on {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content=_format_error_payload(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_type="HTTPException",
            details={},
            path=str(request.url.path),
        ),
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all unhandled server error handler."""
    logger.exception(f"[UnhandledException] Internal server error on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_format_error_payload(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="An unexpected internal server error occurred.",
            error_type="InternalServerError",
            details={},
            path=str(request.url.path),
        ),
    )
