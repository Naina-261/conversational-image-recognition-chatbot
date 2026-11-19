from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import io
import logging
from PIL import Image
from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import ImageProcessingError

logger = logging.getLogger("bel_chatbot.file_validation")
settings = get_settings()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAGIC_SIGNATURES = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/webp": [b"RIFF"],
}


def detect_mime_from_magic_bytes(header: bytes) -> str:
    """
    Inspect raw binary header bytes to reliably verify image MIME type.
    Prevents extension-spoofing attacks (e.g. .jpg file containing executable code).
    """
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


async def validate_and_read_upload(
    upload_file: UploadFile,
    max_bytes: Optional[int] = None,
    chunk_size: int = 64 * 1024,
) -> Tuple[bytes, str, str]:
    """
    Validates and reads an uploaded image with memory-safe streaming guards:
    1. Validates Content-Length header if available prior to buffer allocation.
    2. Streams file chunk-by-chunk, aborting immediately if max_bytes is exceeded.
    3. Validates binary magic byte headers.
    4. Verifies image integrity and structure via Pillow to reject corrupt files.

    Returns:
        Tuple of (file_bytes, detected_mime, normalized_extension)
    """
    max_allowed = max_bytes or (settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    max_mb = settings.MAX_UPLOAD_SIZE_MB
    filename = upload_file.filename or "uploaded_image.jpg"

    # 1. Fast Content-Length pre-check
    content_length = upload_file.headers.get("content-length")
    if content_length:
        try:
            cl_int = int(content_length)
            if cl_int > max_allowed:
                raise ImageProcessingError(
                    f"File exceeds maximum allowed size of {max_mb}MB.",
                    details={"file_size_bytes": cl_int, "max_allowed_bytes": max_allowed},
                )
        except ValueError:
            pass

    # 2. Chunked streaming read with memory boundary enforcement
    buffer = io.BytesIO()
    total_bytes = 0
    first_chunk = True
    detected_mime = "application/octet-stream"

    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > max_allowed:
            del buffer
            raise ImageProcessingError(
                f"File exceeds maximum allowed size of {max_mb}MB.",
                details={"file_size_bytes": total_bytes, "max_allowed_bytes": max_allowed},
            )

        if first_chunk:
            detected_mime = detect_mime_from_magic_bytes(chunk[:16])
            if detected_mime not in MAGIC_SIGNATURES:
                del buffer
                raise ImageProcessingError(
                    f"Invalid or unsafe image format. Detected: {detected_mime}. Allowed: JPEG, PNG, WebP.",
                    details={"detected_mime": detected_mime},
                )
            first_chunk = False

        buffer.write(chunk)

    if total_bytes == 0:
        del buffer
        raise ImageProcessingError("Empty file uploaded.")

    file_bytes = buffer.getvalue()
    buffer.close()
    del buffer

    # 3. Normalize extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".jpg" if detected_mime == "image/jpeg" else ".png" if detected_mime == "image/png" else ".webp"

    # 4. Pillow integrity & corruption check
    try:
        bio = io.BytesIO(file_bytes)
        with Image.open(bio) as pil_img:
            # verify() verifies stream integrity without decompressing all pixel data
            pil_img.verify()
        bio.close()
        del bio
    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(
            f"Corrupted or invalid image data: {str(e)}",
            details={"original_filename": filename},
        )

    return file_bytes, detected_mime, ext
