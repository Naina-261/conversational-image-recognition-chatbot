"""
Backend utilities package.
"""
from app.utils.logger import setup_logger, get_logger
from app.utils.memory import trim_process_memory
from app.utils.file_validation import (
    validate_and_read_upload,
    detect_mime_from_magic_bytes,
    ALLOWED_EXTENSIONS,
    MAGIC_SIGNATURES,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "trim_process_memory",
    "validate_and_read_upload",
    "detect_mime_from_magic_bytes",
    "ALLOWED_EXTENSIONS",
    "MAGIC_SIGNATURES",
]
