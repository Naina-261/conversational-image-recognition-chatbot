import pytest
import io
from PIL import Image
from starlette.datastructures import UploadFile, Headers

from app.utils.file_validation import (
    validate_and_read_upload,
    detect_mime_from_magic_bytes,
    ALLOWED_EXTENSIONS,
)
from app.core.exceptions import ImageProcessingError


def create_image_upload(format: str = "JPEG", size=(100, 100), color="red", filename: str = "test.jpg") -> UploadFile:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return UploadFile(
        file=buf,
        filename=filename,
        headers=Headers({"content-type": f"image/{format.lower()}"}),
    )


@pytest.mark.asyncio
async def test_valid_jpeg_upload():
    upload = create_image_upload("JPEG", filename="valid.jpg")
    file_bytes, mime, ext = await validate_and_read_upload(upload)
    assert mime == "image/jpeg"
    assert ext == ".jpg"
    assert len(file_bytes) > 0


@pytest.mark.asyncio
async def test_valid_png_upload():
    upload = create_image_upload("PNG", filename="valid.png")
    file_bytes, mime, ext = await validate_and_read_upload(upload)
    assert mime == "image/png"
    assert ext == ".png"
    assert len(file_bytes) > 0


@pytest.mark.asyncio
async def test_valid_webp_upload():
    upload = create_image_upload("WEBP", filename="valid.webp")
    file_bytes, mime, ext = await validate_and_read_upload(upload)
    assert mime == "image/webp"
    assert ext == ".webp"
    assert len(file_bytes) > 0


@pytest.mark.asyncio
async def test_reject_empty_file():
    upload = UploadFile(file=io.BytesIO(b""), filename="empty.jpg")
    with pytest.raises(ImageProcessingError) as exc_info:
        await validate_and_read_upload(upload)
    assert "empty file" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_reject_unsupported_magic_bytes():
    # File named .jpg but containing plain text
    fake_data = b"This is a bash script or binary, not an image header"
    upload = UploadFile(file=io.BytesIO(fake_data), filename="fake.jpg")
    with pytest.raises(ImageProcessingError) as exc_info:
        await validate_and_read_upload(upload)
    assert "invalid or unsafe image format" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_reject_file_exceeding_max_size():
    # Set limit to 100 bytes and pass 500 bytes of data with JPEG header
    fake_large_jpeg = b"\xff\xd8\xff\xe0" + (b"A" * 500)
    upload = UploadFile(file=io.BytesIO(fake_large_jpeg), filename="large.jpg")
    with pytest.raises(ImageProcessingError) as exc_info:
        await validate_and_read_upload(upload, max_bytes=100)
    assert "exceeds maximum allowed size" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_reject_corrupt_image_stream():
    # Valid JPEG header but truncated/corrupt payload that Pillow cannot parse
    corrupt_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"CORRUPTED"
    upload = UploadFile(file=io.BytesIO(corrupt_jpeg), filename="corrupt.jpg")
    with pytest.raises(ImageProcessingError) as exc_info:
        await validate_and_read_upload(upload)
    assert "corrupted or invalid image data" in str(exc_info.value).lower()
