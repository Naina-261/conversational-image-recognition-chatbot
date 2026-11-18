import pytest
import io
from PIL import Image
from httpx import AsyncClient, ASGITransport
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app
from app.services.image_service import get_image_service


@pytest.fixture(autouse=True)
def mock_yolo_for_upload(monkeypatch):
    image_svc = get_image_service()
    async def mock_detect(*args, **kwargs):
        return {
            "detections": [],
            "count": 0,
            "image_dimensions": {"width": 200, "height": 200},
            "latency_ms": 1.0,
            "device": "cpu",
            "model": "mock",
        }
    monkeypatch.setattr(image_svc.yolo_client, "detect_objects_async", mock_detect)


def create_test_image(format="JPEG", size=(200, 200), color="blue") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_upload_valid_jpeg():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        img_bytes = create_test_image("JPEG")
        files = {"file": ("test_recon.jpg", img_bytes, "image/jpeg")}
        response = await client.post("/api/v1/images/upload", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["mime_type"] == "image/jpeg"
        assert data["width"] == 200
        assert data["height"] == 200
        assert "url" in data


@pytest.mark.asyncio
async def test_reject_invalid_file_type():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Text file pretending to be jpg
        fake_bytes = b"This is a plaintext script, not an image"
        files = {"file": ("malicious.jpg", fake_bytes, "image/jpeg")}
        response = await client.post("/api/v1/images/upload", files=files)

        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "Invalid or unsafe image format" in data["error"]
