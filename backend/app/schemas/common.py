from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class ErrorResponse(BaseModel):
    """
    Standardized Error Response schema across all API endpoints.
    Provides consistent machine-readable and human-readable error telemetry.
    """
    success: bool = Field(default=False, description="Indicates request failure (for client backward-compatibility)")
    status_code: int = Field(..., description="HTTP response status code")
    message: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Categorical error code/type")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual details or parameters")
    path: Optional[str] = Field(default=None, description="Request path where error occurred")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the error event",
    )

    model_config = ConfigDict(from_attributes=True)
